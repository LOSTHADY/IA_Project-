"""Modèle de dialogue — composant délibérément interchangeable.

Le LLM n'est pas l'objet de l'étude : il raisonne en français et sa sortie est
traduite. On lui demande un français simple et des phrases courtes (prompt
système), et l'on retire de sa sortie ce que la TTS ne saurait pas prononcer.
L'idée que des phrases plus courtes se traduisent mieux vers le bambara n'a
pas été confirmée à contenu égal (docs/RESULTATS.md, découpage avant
traduction) : la sortie n'est donc plus coupée au milieu des phrases.

Trois backends :
  - "transformers" : n'importe quel modèle instruct HF (défaut)
  - "llamacpp"     : GGUF quantisé, pour le déploiement CPU
  - "echo"         : bouchon déterministe, pour tester le pipeline sans modèle
"""

from __future__ import annotations

import logging
import re

from .config import LLMConfig

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_RE = re.compile(r"(?<=[,;:])\s+")
# Marqueurs de mise en forme que la TTS ne saurait pas prononcer.
_MARKUP_RE = re.compile(r"[*_#`|]+")
_BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s*", flags=re.MULTILINE)


def simplify_for_translation(text: str, max_words: int | None = None,
                             max_sentences: int = 3) -> str:
    """Prépare la sortie du LLM pour la traduction et la voix.

    Supprime toute mise en forme et garde au plus `max_sentences` phrases :
    une réponse vocale doit rester brève. Avec `max_words`, coupe aussi les
    phrases trop longues à la dernière virgule dans la limite. Ce n'est plus
    le défaut : la coupure perd la fin de la phrase et fabrique un fragment,
    que NLLB traduit moins bien que la phrase entière (docs/RESULTATS.md).
    """
    if not text:
        return ""
    text = _MARKUP_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()

    kept: list[str] = []
    for sentence in _SENTENCE_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        words = sentence.split()
        if max_words and len(words) > max_words:
            # Couper à la dernière virgule dans la limite, sinon couper net.
            head = words[:max_words]
            cut = max(
                (i for i, w in enumerate(head) if w.endswith(",")), default=None
            )
            head = head[: cut + 1] if cut is not None else head
            sentence = " ".join(head).rstrip(",") + "."
        kept.append(sentence)
        if len(kept) >= max_sentences:
            break
    return " ".join(kept)


def split_for_translation(text: str, max_words: int = 10, min_words: int = 4) -> list[str]:
    """Découpe un texte français en segments courts, sans rien en retirer.

    Une phrase de plus de `max_words` mots est coupée après une virgule, un
    point-virgule ou un deux-points. Les morceaux voisins sont regroupés tant
    qu'ils tiennent dans `max_words`, et un morceau de moins de `min_words`
    mots rejoint son voisin. Une phrase sans ponctuation interne reste
    entière : on ne coupe jamais au milieu d'une proposition.

    Sert à mesurer l'effet de la seule longueur des phrases sur la
    traduction (`eval.baselines mt --decoupe`). Mesuré sur NLLB zero-shot :
    les morceaux se traduisent moins bien que la phrase entière
    (docs/RESULTATS.md). La chaîne ne s'en sert donc pas.
    """
    segments: list[str] = []
    for sentence in _SENTENCE_RE.split(re.sub(r"\s+", " ", text).strip()):
        if not sentence:
            continue
        if len(sentence.split()) <= max_words:
            segments.append(sentence)
            continue
        merged: list[str] = []
        for piece in _CLAUSE_RE.split(sentence):
            n = len(piece.split())
            if merged and (len(merged[-1].split()) + n <= max_words or n < min_words
                           or len(merged[-1].split()) < min_words):
                merged[-1] += " " + piece
            else:
                merged.append(piece)
        segments.extend(merged)
    return segments


class ChatModel:
    """Interface unique par-dessus les trois backends."""

    def __init__(self, config: LLMConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self._model = None
        self._tokenizer = None

    # --- chargement ---------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None or self.config.backend == "echo":
            return
        if self.config.backend == "llamacpp":
            self._load_llamacpp()
        else:
            self._load_transformers()

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info("Chargement LLM %s sur %s", self.config.model_id, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id, torch_dtype=torch.float32
        ).to(self.device)
        self._model.eval()

    def _load_llamacpp(self) -> None:
        from llama_cpp import Llama

        if not self.config.gguf_path:
            raise ValueError("backend 'llamacpp' : gguf_path est requis")
        logger.info("Chargement GGUF %s", self.config.gguf_path)
        self._model = Llama(
            model_path=self.config.gguf_path,
            n_ctx=2048,
            verbose=False,
        )

    # --- génération ---------------------------------------------------------

    def reply(self, user_text_fr: str, history: list[dict] | None = None) -> str:
        """Produit une réponse en français simple, prête à être traduite."""
        self._ensure_loaded()
        cfg = self.config

        if cfg.backend == "echo":
            raw = f"J'ai compris : {user_text_fr.strip()}"
        elif cfg.backend == "llamacpp":
            raw = self._reply_llamacpp(user_text_fr, history)
        else:
            raw = self._reply_transformers(user_text_fr, history)

        return simplify_for_translation(raw, max_words=cfg.max_words)

    def _messages(self, user_text_fr: str, history: list[dict] | None) -> list[dict]:
        msgs = [{"role": "system", "content": self.config.system_prompt}]
        msgs.extend(history or [])
        msgs.append({"role": "user", "content": user_text_fr})
        return msgs

    def _reply_transformers(self, user_text_fr: str, history: list[dict] | None) -> str:
        import torch

        msgs = self._messages(user_text_fr, history)
        try:
            # Toujours un dictionnaire (input_ids, attention_mask) : selon la
            # version de transformers, sans return_dict, on reçoit un tenseur
            # ou déjà un dictionnaire.
            inputs = self._tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
        except (ValueError, AttributeError):
            # Modèle sans gabarit de chat : repli sur une mise en forme simple.
            flat = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
            inputs = self._tokenizer(flat + "\nassistant:", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        sampling = self.config.temperature > 0
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=sampling,
                **({"temperature": self.config.temperature} if sampling else {}),
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        # Ne décoder que ce qui a été ajouté au prompt.
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _reply_llamacpp(self, user_text_fr: str, history: list[dict] | None) -> str:
        res = self._model.create_chat_completion(
            messages=self._messages(user_text_fr, history),
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        return res["choices"][0]["message"]["content"].strip()

    __call__ = reply
