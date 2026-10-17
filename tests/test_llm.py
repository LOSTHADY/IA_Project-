"""Tests de bambara_voice/llm.py avec un vrai petit LM causal.

Les autres tests remplacent le LLM par le bouchon « echo ». Celui-ci fait
tourner ChatModel de bout en bout (gabarit de chat, génération, décodage) :
c'est ce chemin qui a cassé avec transformers 5, où apply_chat_template
renvoie un dictionnaire et non plus un tenseur.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiny_models  # noqa: E402
from bambara_voice.config import LLMConfig  # noqa: E402
from bambara_voice.llm import ChatModel  # noqa: E402


class GenerateSpy:
    """Enregistre ce que ChatModel passe à generate."""

    def __init__(self, inner):
        self.inner = inner

    def generate(self, **kw):
        self.kwargs = kw
        self.out = self.inner.generate(**kw)
        return self.out


@pytest.fixture(scope="module")
def lms(tmp_path_factory):
    root = tmp_path_factory.mktemp("lm")
    return {flag: tiny_models.causal_lm(root / str(flag), chat_template=flag)
            for flag in (True, False)}


@pytest.mark.parametrize("chat_template", [True, False])
@pytest.mark.parametrize("temperature", [0.0, 0.3])
def test_reponse_avec_un_vrai_petit_lm(lms, chat_template, temperature):
    llm = ChatModel(LLMConfig(backend="transformers", model_id=str(lms[chat_template]),
                              max_new_tokens=6, temperature=temperature))
    history = [{"role": "user", "content": "i ni ce"}, {"role": "assistant", "content": "i ni ce"}]
    # D'abord tel quel : c'est cet appel qui échouait avec transformers 5.
    assert isinstance(llm.reply("bonjour", history), str)

    llm._model = GenerateSpy(llm._model)
    llm.reply("bonjour", history)
    kw = llm._model.kwargs
    # Le prompt arrive en identifiants et masque, pas en objet du tokenizer.
    assert set(kw) >= {"input_ids", "attention_mask", "max_new_tokens"}
    prompt = llm._tokenizer.decode(kw["input_ids"][0])
    assert "bonjour" in prompt and prompt.rstrip().endswith("assistant:")
    # Seuls les jetons générés sont décodés, jamais le prompt.
    generated = llm._model.out.shape[-1] - kw["input_ids"].shape[-1]
    assert 0 < generated <= 6
    assert kw["do_sample"] is (temperature > 0)


def test_backend_llamacpp(monkeypatch, tmp_path):
    """Le LLM allégé du CPU (GGUF via llama.cpp) : chargé depuis gguf_path,
    même consigne et même post-traitement que le backend transformers."""
    import sys
    import types

    vus = {}

    class Llama:
        def __init__(self, model_path, n_ctx, verbose):
            vus["chemin"] = model_path

        def create_chat_completion(self, messages, max_tokens, temperature):
            vus["messages"], vus["max_tokens"] = messages, max_tokens
            return {"choices": [{"message": {"content": " Bonjour. Je vais bien. "}}]}

    module = types.ModuleType("llama_cpp")
    module.Llama = Llama
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    from bambara_voice.config import LLMConfig
    from bambara_voice.llm import ChatModel

    gguf = tmp_path / "qwen.gguf"
    llm = ChatModel(LLMConfig(backend="llamacpp", gguf_path=str(gguf), max_new_tokens=64))
    assert llm.reply("Comment vas-tu ?") == "Bonjour. Je vais bien."
    assert vus["chemin"] == str(gguf) and vus["max_tokens"] == 64
    assert vus["messages"][-1] == {"role": "user", "content": "Comment vas-tu ?"}
    with pytest.raises(ValueError, match="gguf_path"):
        ChatModel(LLMConfig(backend="llamacpp")).reply("x")
