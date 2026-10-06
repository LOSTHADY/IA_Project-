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
