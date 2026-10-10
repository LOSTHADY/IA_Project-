"""Reconnaissance et traduction de la parole bambara.

Un seul modèle Whisper peut être entraîné à produire soit la transcription
bambara, soit directement la traduction française (Jeli-ASR fournit les deux
cibles pour chaque audio). C'est ce que `ASRConfig.task` sélectionne, et c'est
la base de la comparaison cascade / bout-en-bout.

Les modèles CTC (wav2vec2, MMS) sont aussi pris en charge, en transcription
seule : `facebook/mms-1b-all` couvre le bambara sans fine-tuning et fournit la
référence zero-shot la plus sérieuse côté ASR.

Les modèles NeMo aussi (`kind="nemo"`), en transcription seule : Soloni, de
RobotsMali, est un Parakeet de NVIDIA (114 millions de paramètres) affiné sur
le bambara. NeMo est une dépendance lourde, importée seulement pour eux.

Pour le déploiement CPU, un Whisper converti par `scripts/export_cpu.py`
tourne sous CTranslate2 (`backend="ctranslate2"`, int8) : même modèle, mêmes
tokens de langue et de tâche, 4 fois plus léger. Le décodage va environ deux
fois plus vite, pas l'encodeur (cf. docs/DEPLOIEMENT.md).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ASRConfig, split_device
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
        if self.config.backend == "ctranslate2":
            if self.config.kind != "whisper":
                raise ValueError("backend ctranslate2 : Whisper uniquement")
            self._load_ct2()
        elif self.config.kind == "ctc":
            self._load_ctc()
        elif self.config.kind == "nemo":
            self._load_nemo()
        else:
            self._load_whisper()

    def _load_ct2(self) -> None:
        import ctranslate2
        from transformers import AutoProcessor

        cfg = self.config
        logger.info("Chargement ASR CTranslate2 %s (%s) sur %s",
                    cfg.model_id, cfg.compute_type, self.device)
        # Le dossier converti contient aussi extracteur et tokenizer.
        self._processor = AutoProcessor.from_pretrained(cfg.model_id)
        device, index = split_device(self.device)
        self._model = ctranslate2.models.Whisper(
            cfg.model_id, device=device, device_index=index, compute_type=cfg.compute_type
        )

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

    def _load_nemo(self) -> None:
        import torch
        from nemo.collections.asr.models import ASRModel

        cfg = self.config
        device = torch.device(self.device)
        logger.info("Chargement ASR NeMo %s (décodeur %s) sur %s",
                    cfg.model_id, cfg.nemo_decoder or "par défaut", self.device)
        if cfg.model_id.endswith(".nemo"):
            model = ASRModel.restore_from(cfg.model_id, map_location=device)
        else:
            model = self._load_nemo_hub(ASRModel, device)
        if cfg.nemo_decoder == "ctc":
            model.change_decoding_strategy(decoder_type="ctc",
                                           decoding_cfg=model.cfg.aux_ctc.decoding)
        elif device.type == "cuda":
            # Fiche de Soloni : les graphes CUDA du décodeur TDT échouent sur
            # certains GPU (« CUDA error: invalid argument »).
            from omegaconf import open_dict

            decoding = model.cfg.decoding
            with open_dict(decoding):
                decoding.greedy.use_cuda_graph_decoder = False
            model.change_decoding_strategy(decoding_cfg=decoding)
        model.eval()
        self._model = model

    def _load_nemo_hub(self, ASRModel, device):
        """Depuis Hugging Face. Un modèle enregistré avec NeMo 2.5 (Soloni)
        ne se charge plus tel quel avec une version récente : le schéma de
        décodage exige `key_phrase_items_list`, absent de sa configuration.
        On l'ajoute avant le chargement, comme le fait la fiche de Soloni
        (NVIDIA-NeMo/Speech#15658). Constaté avec NeMo 3.0."""
        import tempfile

        from omegaconf import OmegaConf

        model_id = self.config.model_id
        conf = ASRModel.from_pretrained(model_id, return_config=True)
        OmegaConf.set_struct(conf, False)
        trees = [OmegaConf.select(conf, f"decoding.{d}.boosting_tree") for d in ("greedy", "beam")]
        old = [t for t in trees if t is not None and "key_phrase_items_list" not in t]
        if not old:
            return ASRModel.from_pretrained(model_name=model_id, map_location=device)
        logger.warning("%s : configuration d'un ancien NeMo, corrigée avant chargement", model_id)
        for tree in old:
            tree.key_phrase_items_list = None
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            OmegaConf.save(conf, path)
            return ASRModel.from_pretrained(model_name=model_id, override_config_path=str(path),
                                            map_location=device, strict=False)

    def transcribe(self, audio: np.ndarray | str | Path) -> ASRResult:
        self._ensure_loaded()
        cfg = self.config

        if isinstance(audio, (str, Path)):
            audio = load_audio(audio, cfg.sample_rate)

        if cfg.backend == "ctranslate2":
            text = self._decode_ct2(audio)
        elif cfg.kind == "nemo":
            text = self._decode_nemo(audio)
        else:
            inputs = self._processor(
                audio, sampling_rate=cfg.sample_rate, return_tensors="pt"
            ).to(self.device)
            text = self._decode_ctc(inputs) if cfg.kind == "ctc" else self._decode_whisper(inputs)

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

    def _decode_nemo(self, audio: np.ndarray) -> str:
        import tempfile

        import soundfile as sf

        if self.config.task == "translate":
            raise ValueError("un modèle NeMo (Soloni) ne sait que transcrire")
        # Un fichier temporaire : la seule entrée que toutes les versions de
        # NeMo acceptent.
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "audio.wav")
            sf.write(path, audio, self.config.sample_rate)
            out = self._model.transcribe([path], batch_size=1, verbose=False)
        if isinstance(out, tuple):  # modèles hybrides, anciennes versions : (meilleures, toutes)
            out = out[0]
        hyp = out[0]
        return getattr(hyp, "text", hyp)  # Hypothesis récent, str autrefois

    def _decode_whisper(self, inputs) -> str:
        import torch

        cfg = self.config
        gen_kwargs: dict = {"max_new_tokens": cfg.max_new_tokens, "num_beams": cfg.beam_size}
        if self._prompt_is_forceable():
            gen_kwargs["task"] = cfg.task
            if cfg.language:
                gen_kwargs["language"] = cfg.language
        with torch.no_grad():
            ids = self._model.generate(**inputs, **gen_kwargs)
        return self._processor.batch_decode(ids, skip_special_tokens=True)[0]

    def _decode_ct2(self, audio: np.ndarray) -> str:
        import ctranslate2

        cfg = self.config
        feats = self._processor(audio, sampling_rate=cfg.sample_rate,
                                return_tensors="np").input_features
        feats = ctranslate2.StorageView.from_array(feats.astype(np.float32))
        tok = self._processor.tokenizer

        # Même préfixe que transformers : <|startoftranscript|><|langue|><|tâche|>.
        prompt = ["<|startoftranscript|>"]
        if self._prompt_is_forceable():
            if cfg.language:
                prompt.append(self._lang_token())
            else:
                # Détection automatique, comme transformers sans langue imposée.
                prompt.append(self._model.detect_language(feats)[0][0][0])
            prompt.append(f"<|{cfg.task}|>")
        prompt.append("<|notimestamps|>")
        ids = tok.convert_tokens_to_ids(prompt)

        # CTranslate2 génère au plus max_length // 2 tokens, quel que soit le
        # préfixe (mesuré : 40 -> 20, 448 -> 224 ; convention sample_len de
        # Whisper). Le double donne le même plafond que max_new_tokens côté
        # transformers, dans la limite du contexte du décodeur.
        res = self._model.generate(feats, [ids], beam_size=cfg.beam_size,
                                   max_length=min(448, 2 * cfg.max_new_tokens))
        return tok.decode(res[0].sequences_ids[0], skip_special_tokens=True)

    def _lang_token(self) -> str:
        lang = self.config.language
        return lang if lang.startswith("<|") else f"<|{lang}|>"

    def _prompt_is_forceable(self) -> bool:
        """Vrai si la langue et la tâche demandées existent dans le modèle.

        Vérifié à l'avance plutôt qu'en rattrapant l'erreur de `generate` :
        une erreur sans rapport (longueur, mémoire) ne doit pas passer pour un
        problème de langue. Jamais de repli en traduction : sans tâche forcée,
        Whisper transcrirait, et la variante e2e mesurerait silencieusement
        autre chose.
        """
        cfg = self.config
        if self.config.backend == "ctranslate2":
            # Pas de generation_config dans un modèle converti : on consulte
            # le vocabulaire, qui contient les tokens spéciaux.
            vocab = self._processor.tokenizer.get_vocab()
            langs = vocab
            tasks = {t for t in ("transcribe", "translate") if f"<|{t}|>" in vocab}
        else:
            gc = self._model.generation_config
            langs = getattr(gc, "lang_to_id", None) or {}
            tasks = getattr(gc, "task_to_id", None) or {}

        missing = []
        if cfg.language and self._lang_token() not in langs:
            missing.append(self._lang_token())
        if cfg.task not in tasks:
            missing.append(cfg.task)
        if not missing:
            return True
        if cfg.task == "translate":
            raise ValueError(f"{cfg.model_id} ne connaît pas {missing} : "
                             f"traduction directe impossible")
        logger.warning("%s ne connaît pas %s : génération sans langue ni tâche imposées",
                       cfg.model_id, missing)
        return False

    __call__ = transcribe
