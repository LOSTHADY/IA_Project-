"""Orchestration du système complet.

Deux architectures, un seul point d'entrée, pour que la comparaison porte bien
sur l'architecture et pas sur des différences d'implémentation :

  cascade : audio bm --ASR--> texte bm --MT--> fr --LLM--> fr --MT--> bm --TTS--> audio
  e2e     : audio bm --------ASR(translate)--> fr --LLM--> fr --MT--> bm --TTS--> audio

La variante e2e supprime un maillon à l'entrée. Elle ne produit pas de
transcription bambara : c'est le compromis à documenter.

Chaque étape est chronométrée et conservée dans `TurnTrace`, ce qui permet de
mesurer la propagation d'erreurs et le budget de latence sans réinstrumenter
le code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .asr import SpeechRecognizer, load_audio
from .config import PipelineConfig
from .llm import ChatModel
from .mt import Translator
from .templates import TemplateBank
from .tts import SpeechSynthesizer, SpeechResult

logger = logging.getLogger(__name__)


@dataclass
class TurnTrace:
    """Trace complète d'un tour de parole. C'est l'unité d'analyse du mémoire."""

    architecture: str
    asr_text: str = ""            # bambara (cascade) ou français (e2e)
    source_bm: str = ""           # texte bambara si disponible
    source_fr: str = ""           # entrée du LLM
    reply_fr: str = ""            # sortie du LLM, après simplification
    reply_bm: str = ""            # réponse finale en bambara
    reply_source: str = "mt"      # "template" ou "mt"
    template_id: str | None = None
    template_score: float | None = None
    timings: dict[str, float] = field(default_factory=dict)
    audio_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return sum(self.timings.values())

    @property
    def rtf(self) -> float:
        """Real-time factor : < 1 signifie plus rapide que le temps réel."""
        return self.total_seconds / self.audio_seconds if self.audio_seconds else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_seconds"] = round(self.total_seconds, 3)
        d["rtf"] = round(self.rtf, 3)
        return d


class _Timer:
    """Chronomètre chaque étape dans le dict de trace."""

    def __init__(self, timings: dict[str, float], key: str):
        self.timings, self.key = timings, key

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.timings[self.key] = time.perf_counter() - self.t0
        return False


class VoicePipeline:
    """Chaîne bambara -> bambara. Composants chargés à la première utilisation."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        dev = config.device
        self.asr = SpeechRecognizer(config.asr, dev)
        self.mt_in = Translator(config.mt_in, dev) if config.architecture == "cascade" else None
        self.mt_out = Translator(config.mt_out, dev)
        self.llm = ChatModel(config.llm, dev)
        self.tts = SpeechSynthesizer(config.tts, dev)
        self.templates = TemplateBank.from_config(config.templates)

    # --- étapes ------------------------------------------------------------

    def understand(self, audio: np.ndarray | str | Path, trace: TurnTrace) -> str:
        """Audio bambara -> texte français, quelle que soit l'architecture."""
        if isinstance(audio, (str, Path)):
            audio = load_audio(audio, self.config.asr.sample_rate)
        trace.audio_seconds = len(audio) / self.config.asr.sample_rate

        with _Timer(trace.timings, "asr"):
            res = self.asr.transcribe(audio)
        trace.asr_text = res.text

        if res.language == "fr":
            trace.source_fr = res.text
            return res.text

        trace.source_bm = res.text
        with _Timer(trace.timings, "mt_in"):
            trace.source_fr = self.mt_in.translate(res.text)
        return trace.source_fr

    def respond(self, trace: TurnTrace, history: list[dict] | None = None) -> str:
        """Texte français -> réponse bambara, gabarit d'abord puis repli MT."""
        # Un gabarit ne peut être testé que si l'on dispose du texte bambara,
        # donc jamais en e2e — limite à documenter dans le mémoire.
        if trace.source_bm:
            match = self.templates.match(trace.source_bm)
            if match is not None:
                trace.reply_source = "template"
                trace.template_id = match.template.id
                trace.template_score = round(match.score, 3)
                trace.reply_fr = match.template.response_fr
                trace.reply_bm = match.template.response_bm
                return trace.reply_bm

        with _Timer(trace.timings, "llm"):
            trace.reply_fr = self.llm.reply(trace.source_fr, history)
        with _Timer(trace.timings, "mt_out"):
            trace.reply_bm = self.mt_out.translate(trace.reply_fr)
        trace.reply_source = "mt"
        return trace.reply_bm

    def speak(self, trace: TurnTrace) -> SpeechResult:
        with _Timer(trace.timings, "tts"):
            return self.tts.synthesize(trace.reply_bm)

    # --- tour complet ------------------------------------------------------

    def run(
        self,
        audio: np.ndarray | str | Path,
        history: list[dict] | None = None,
        synthesize: bool = True,
    ) -> tuple[TurnTrace, SpeechResult | None]:
        trace = TurnTrace(architecture=self.config.architecture)
        self.understand(audio, trace)
        self.respond(trace, history)
        speech = self.speak(trace) if synthesize else None
        logger.info(
            "[%s] %.1fs audio -> %.1fs calcul (RTF %.2f), réponse via %s",
            trace.architecture, trace.audio_seconds, trace.total_seconds,
            trace.rtf, trace.reply_source,
        )
        return trace, speech

    def run_text(self, text_bm: str, history: list[dict] | None = None) -> TurnTrace:
        """Tour de parole en partant du texte bambara — utile pour évaluer la
        chaîne MT/LLM sans l'ASR, et donc isoler sa contribution à l'erreur."""
        trace = TurnTrace(architecture=self.config.architecture)
        trace.source_bm = text_bm
        translator = self.mt_in or Translator(self.config.mt_in, self.config.device)
        with _Timer(trace.timings, "mt_in"):
            trace.source_fr = translator.translate(text_bm)
        self.respond(trace, history)
        return trace
