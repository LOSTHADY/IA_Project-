"""Traduction bambara <-> français (NLLB-200).

NLLB couvre nativement `bam_Latn`. Le zero-shot sert de référence basse ; le
fine-tuning sur Bayelemabaga (cf. scripts/finetune_nllb.py) est ce qui fait
gagner l'essentiel des points de chrF++.

Pour le déploiement CPU, un modèle converti par `scripts/export_cpu.py`
tourne sous CTranslate2 (`backend="ctranslate2"`, int8).
"""

from __future__ import annotations

import logging
import math

from .config import MTConfig, split_device
from .normalize import normalize

logger = logging.getLogger(__name__)


class Translator:
    """Traducteur NLLB à chargement paresseux, réutilisable pour les deux sens.

    Deux instances (bm->fr et fr->bm) partagent le même modèle en mémoire si
    elles pointent le même `model_id` : voir `Translator.shared`.
    """

    _cache: dict[tuple[str, str], tuple] = {}

    def __init__(self, config: MTConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self._model = None
        self._tokenizer = None
        # La dernière traduction s'est-elle arrêtée sur la longueur maximale
        # plutôt que sur une fin de phrase ? Presque toujours une boucle.
        self.last_capped = False

    def max_tokens(self, n_source: int) -> int:
        """Longueur maximale de la traduction, en jetons, token de langue
        compris (c'est ainsi que comptent generate et CTranslate2)."""
        cfg = self.config
        if cfg.max_length_ratio is None:
            return cfg.max_new_tokens
        return min(cfg.max_new_tokens,
                   math.ceil(cfg.max_length_ratio * n_source) + cfg.max_length_margin)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        cfg = self.config
        key = (cfg.model_id, self.device, cfg.backend, cfg.compute_type)
        if key in Translator._cache:
            self._tokenizer, self._model = Translator._cache[key]
            return

        logger.info("Chargement MT %s (%s) sur %s", cfg.model_id, cfg.backend, self.device)
        # Un dossier converti contient aussi le tokenizer d'origine.
        tok = AutoTokenizer.from_pretrained(cfg.model_id)
        if cfg.backend == "ctranslate2":
            import ctranslate2

            device, index = split_device(self.device)
            model = ctranslate2.Translator(cfg.model_id, device=device, device_index=index,
                                           compute_type=cfg.compute_type)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                cfg.model_id, torch_dtype=torch.float32
            ).to(self.device)
            model.eval()
        Translator._cache[key] = (tok, model)
        self._tokenizer, self._model = tok, model

    def _target_token_id(self) -> int | None:
        """Récupère l'id du token de langue cible.

        L'API a changé entre versions de transformers (`lang_code_to_id`, puis
        `convert_tokens_to_ids`) ; on tente les deux plutôt que d'épingler une
        version.
        """
        tok = self._tokenizer
        lang = self.config.tgt_lang
        mapping = getattr(tok, "lang_code_to_id", None)
        if mapping and lang in mapping:
            return mapping[lang]
        tid = tok.convert_tokens_to_ids(lang)
        unk = getattr(tok, "unk_token_id", None)
        if tid is None or tid == unk:
            logger.warning("Token de langue %s introuvable dans le tokenizer", lang)
            return None
        return tid

    def translate(self, text: str) -> str:
        import torch

        self.last_capped = False
        text = text.strip()
        if not text:
            return ""

        self._ensure_loaded()
        cfg = self.config
        # Le tokenizer NLLB doit connaître la langue source avant l'encodage.
        self._tokenizer.src_lang = cfg.src_lang
        if cfg.backend == "ctranslate2":
            return self._finish(self._translate_ct2(text))

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        budget = self.max_tokens(inputs["input_ids"].shape[1])
        gen_kwargs: dict = {
            "max_new_tokens": budget,
            "num_beams": cfg.beam_size,
        }
        bos = self._target_token_id()
        if bos is not None:
            gen_kwargs["forced_bos_token_id"] = bos

        with torch.no_grad():
            ids = self._model.generate(**inputs, **gen_kwargs)
        # Une séquence commence par le token de début du décodeur.
        self.last_capped = (ids.shape[1] - 1 >= budget
                            and ids[0, -1].item() != self._tokenizer.eos_token_id)
        return self._finish(self._tokenizer.batch_decode(ids, skip_special_tokens=True)[0])

    def _translate_ct2(self, text: str) -> str:
        tok, cfg = self._tokenizer, self.config
        source = tok.convert_ids_to_tokens(tok.encode(text, truncation=True, max_length=512))
        budget = self.max_tokens(len(source))
        # Le préfixe cible joue le rôle de forced_bos_token_id.
        res = self._model.translate_batch(
            [source], target_prefix=[[cfg.tgt_lang]], beam_size=cfg.beam_size,
            max_decoding_length=budget,
        )
        # Les hypothèses comptent le préfixe mais pas la fin de phrase : une
        # hypothèse qui remplit le budget a été coupée.
        self.last_capped = len(res[0].hypotheses[0]) >= budget
        target = res[0].hypotheses[0][1:]  # sans le token de langue
        return tok.decode(tok.convert_tokens_to_ids(target), skip_special_tokens=True)

    def _finish(self, out: str) -> str:
        if self.config.tgt_lang.startswith("bam"):
            out = normalize(out, lower=False)
        return out.strip()

    __call__ = translate
