"""Modèle de dialogue — composant délibérément interchangeable.

Le LLM n'est pas l'objet de l'étude : il raisonne en français et sa sortie est
traduite. Ce qui compte ici, c'est la *contrainte de style* appliquée à sa
sortie (phrases courtes, français simple), qui améliore nettement la fidélité
de la traduction sortante sans rien coûter.

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
# Marqueurs de mise en forme que la TTS ne saurait pas prononcer.
_MARKUP_RE = re.compile(r"[*_#`|]+")
_BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s*", flags=re.MULTILINE)


def simplify_for_translation(text: str, max_words: int = 15, max_sentences: int = 3) -> str:
    """Force la sortie du LLM dans la forme la plus traduisible possible.

    Le prompt système demande déjà des phrases courtes, mais aucun modèle ne
    respecte une consigne à 100 %. Ce post-traitement est la garantie dure :
    il coupe les phrases trop longues à la frontière de proposition la plus
    proche et supprime toute mise en forme.
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
        if len(words) > max_words:
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
            prompt_ids = self._tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)
        except (ValueError, AttributeError):
            # Modèle sans gabarit de chat : repli sur une mise en forme simple.
            flat = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
            prompt_ids = self._tokenizer(
                flat + "\nassistant:", return_tensors="pt"
            ).input_ids.to(self.device)

        with torch.no_grad():
            out = self._model.generate(
                prompt_ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=max(self.config.temperature, 1e-4),
                pad_token_id=self._tokenizer.eos_token_id,
            )
        # Ne décoder que ce qui a été ajouté au prompt.
        new_tokens = out[0][prompt_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _reply_llamacpp(self, user_text_fr: str, history: list[dict] | None) -> str:
        res = self._model.create_chat_completion(
            messages=self._messages(user_text_fr, history),
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        return res["choices"][0]["message"]["content"].strip()

    __call__ = reply
