"""Reconnaissance et traduction de la parole bambara.

Un seul modèle Whisper peut être entraîné à produire soit la transcription
bambara, soit directement la traduction française (Jeli-ASR fournit les deux
cibles pour chaque audio). C'est ce que `ASRConfig.task` sélectionne, et c'est
la base de la comparaison cascade / bout-en-bout.

Les modèles CTC (wav2vec2, MMS) sont aussi pris en charge, en transcription
seule : `facebook/mms-1b-all` couvre le bambara sans fine-tuning et fournit la
référence zero-shot la plus sérieuse côté ASR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ASRConfig
from .normalize import normalize

logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    text: str
    language: str  # "bm" en transcription, "fr" en traduction directe
    audio_seconds: float


def load_audio(path: str | Path, sample_rate: int = 16_000) -> np.ndarray:
    """Charge un fichier audio en mono float32 au taux d'échantillonnage voulu."""
    import librosa

    wav, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return wav.astype(np.float32)


class SpeechRecognizer:
    """Enveloppe Whisper ou CTC. Chargement paresseux : rien n'est lu tant
    qu'on n'appelle pas `transcribe`, ce qui garde les imports du pipeline
    légers."""

    def __init__(self, config: ASRConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self.config.kind == "ctc":
            self._load_ctc()
        else:
            self._load_whisper()

    def _load_whisper(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        logger.info("Chargement ASR %s sur %s", self.config.model_id, self.device)
        self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float32,  # CPU : pas de fp16
        ).to(self.device)
        self._model.eval()

    def _load_ctc(self) -> None:
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        cfg = self.config
        logger.info("Chargement ASR CTC %s (%s) sur %s",
                    cfg.model_id, cfg.target_lang or "-", self.device)
        proc_kwargs: dict = {}
        model_kwargs: dict = {}
        if cfg.target_lang:
            # MMS : un adaptateur par langue remplace la tête de sortie, d'où
            # ignore_mismatched_sizes (la tête par défaut n'a pas la même taille).
            proc_kwargs = {"target_lang": cfg.target_lang}
            model_kwargs = {"target_lang": cfg.target_lang, "ignore_mismatched_sizes": True}
        self._processor = AutoProcessor.from_pretrained(cfg.model_id, **proc_kwargs)
        self._model = Wav2Vec2ForCTC.from_pretrained(
            cfg.model_id, torch_dtype=torch.float32, **model_kwargs
        ).to(self.device)
        self._model.eval()

    def transcribe(self, audio: np.ndarray | str | Path) -> ASRResult:
        self._ensure_loaded()
        cfg = self.config

        if isinstance(audio, (str, Path)):
            audio = load_audio(audio, cfg.sample_rate)

        inputs = self._processor(
            audio, sampling_rate=cfg.sample_rate, return_tensors="pt"
        ).to(self.device)

        if cfg.kind == "ctc":
            text = self._decode_ctc(inputs)
        else:
            text = self._decode_whisper(inputs)

        out_lang = "fr" if cfg.task == "translate" else "bm"
        # On ne normalise en bambara que si la sortie est bien du bambara.
        text = normalize(text, lower=False) if out_lang == "bm" else text.strip()
        return ASRResult(
            text=text,
            language=out_lang,
            audio_seconds=len(audio) / cfg.sample_rate,
        )

    def _decode_ctc(self, inputs) -> str:
        import torch

        if self.config.task == "translate":
            raise ValueError("un modèle CTC ne sait que transcrire")
        with torch.no_grad():
            logits = self._model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)
        return self._processor.batch_decode(ids)[0]

    def _decode_whisper(self, inputs) -> str:
        import torch

        cfg = self.config
        gen_kwargs: dict = {
            "max_new_tokens": cfg.max_new_tokens,
            "num_beams": cfg.beam_size,
            "task": cfg.task,
        }
        if cfg.language:
            gen_kwargs["language"] = cfg.language
        try:
            with torch.no_grad():
                ids = self._model.generate(**inputs, **gen_kwargs)
        except (ValueError, TypeError) as exc:
            # Modèle sans tokens de langue/tâche (ex. checkpoint mono-langue
            # aux tokens figés). Jamais de repli en traduction : sans tâche
            # forcée, Whisper transcrirait, et la variante e2e mesurerait
            # silencieusement autre chose.
            if cfg.task == "translate":
                raise ValueError(
                    f"{cfg.model_id} refuse task='translate' "
                    f"(language={cfg.language!r}) : {exc}"
                ) from exc
            logger.warning("Langue/tâche non forçables (%s) : génération sans contrainte", exc)
            gen_kwargs.pop("task")
            gen_kwargs.pop("language", None)
            with torch.no_grad():
                ids = self._model.generate(**inputs, **gen_kwargs)

        return self._processor.batch_decode(ids, skip_special_tokens=True)[0]

    __call__ = transcribe
