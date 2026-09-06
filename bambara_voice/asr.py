"""Reconnaissance et traduction de la parole bambara.

Un seul modèle Whisper peut être entraîné à produire soit la transcription
bambara, soit directement la traduction française (Jeli-ASR fournit les deux
cibles pour chaque audio). C'est ce que `ASRConfig.task` sélectionne, et c'est
la base de la comparaison cascade / bout-en-bout.
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
    """Enveloppe Whisper. Chargement paresseux : rien n'est lu tant qu'on
    n'appelle pas `transcribe`, ce qui garde les imports du pipeline légers."""

    def __init__(self, config: ASRConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        logger.info("Chargement ASR %s sur %s", self.config.model_id, self.device)
        self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float32,  # CPU : pas de fp16
        ).to(self.device)
        self._model.eval()

    def transcribe(self, audio: np.ndarray | str | Path) -> ASRResult:
        import torch

        self._ensure_loaded()
        cfg = self.config

        if isinstance(audio, (str, Path)):
            audio = load_audio(audio, cfg.sample_rate)

        inputs = self._processor(
            audio, sampling_rate=cfg.sample_rate, return_tensors="pt"
        ).to(self.device)

        gen_kwargs: dict = {
            "max_new_tokens": cfg.max_new_tokens,
            "num_beams": cfg.beam_size,
        }
        # Un modèle fine-tuné sur une seule langue a souvent ses tokens de tâche
        # figés ; on ne force la tâche que si le modèle l'accepte.
        try:
            gen_kwargs["task"] = cfg.task
            gen_kwargs["language"] = cfg.language
            with torch.no_grad():
                ids = self._model.generate(**inputs, **gen_kwargs)
        except (ValueError, TypeError) as exc:
            logger.debug("Tâche/langue non forçables (%s), repli sans contrainte", exc)
            gen_kwargs.pop("task", None)
            gen_kwargs.pop("language", None)
            with torch.no_grad():
                ids = self._model.generate(**inputs, **gen_kwargs)

        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        out_lang = "fr" if cfg.task == "translate" else "bm"
        # On ne normalise en bambara que si la sortie est bien du bambara.
        text = normalize(text, lower=False) if out_lang == "bm" else text.strip()
        return ASRResult(
            text=text,
            language=out_lang,
            audio_seconds=len(audio) / cfg.sample_rate,
        )

    __call__ = transcribe
