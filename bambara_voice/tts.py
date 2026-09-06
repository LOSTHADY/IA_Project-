"""Synthèse vocale bambara.

Maillon le plus fragile de la chaîne : le bambara est une langue à tons et
l'orthographe ne les note pas, donc le modèle doit deviner la prosodie. Aucune
métrique automatique ne mesure ça correctement — prévoir une évaluation MOS
sur un échantillon (cf. eval/README.md).

Modèles connus au moment de l'écriture : `facebook/mms-tts-bam` (VITS,
mono-locuteur) et les modèles MalianTTS de MALIBA-AI. Vérifier la disponibilité
avant de s'engager : l'écosystème bouge vite et des modèles ont déjà été
retirés.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TTSConfig
from .normalize import normalize

logger = logging.getLogger(__name__)


@dataclass
class SpeechResult:
    audio: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    def save(self, path: str | Path) -> Path:
        import soundfile as sf

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), self.audio, self.sample_rate)
        return path


class SpeechSynthesizer:
    """Enveloppe VITS/MMS à chargement paresseux."""

    def __init__(self, config: TTSConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, VitsModel

        logger.info("Chargement TTS %s sur %s", self.config.model_id, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self._model = VitsModel.from_pretrained(
            self.config.model_id, torch_dtype=torch.float32
        ).to(self.device)
        self._model.eval()

    def synthesize(self, text_bm: str) -> SpeechResult:
        import torch

        text_bm = normalize(text_bm, lower=False)
        sr = self.config.sample_rate
        if not text_bm:
            return SpeechResult(np.zeros(0, dtype=np.float32), sr)

        self._ensure_loaded()
        inputs = self._tokenizer(text_bm, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(**inputs).waveform

        audio = out.squeeze().cpu().numpy().astype(np.float32)
        # Le taux réel du modèle prime sur celui de la config.
        model_sr = getattr(self._model.config, "sampling_rate", sr)
        return SpeechResult(audio, model_sr)

    __call__ = synthesize
