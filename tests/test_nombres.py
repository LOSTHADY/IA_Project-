"""Nombres écrits par règle autour de la traduction sortante
(bambara_voice.nombres), et leur place dans la chaîne."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bambara_voice.config import PipelineConfig  # noqa: E402
from bambara_voice.nombres import bambara, en_bambara, en_chiffres  # noqa: E402


@pytest.mark.parametrize("n, attendu", [
    (1, "kelen"), (9, "kɔnɔntɔn"), (10, "tan"), (11, "tan ni kelen"),
    (19, "tan ni kɔnɔntɔn"), (20, "mugan"), (21, "mugan ni kelen"), (30, "bisaba"),
    (45, "binaani ni duuru"), (99, "bikɔnɔntɔn ni kɔnɔntɔn"), (100, "kɛmɛ"),
    (125, "kɛmɛ ni mugan ni duuru"), (200, "kɛmɛ fila"), (1000, "waa kelen"),
    (2026, "waa fila ni mugan ni wɔɔrɔ"), (1_000_000, "miliyɔn kelen"),
    (0, "fu"),  # corrigé par un locuteur natif (au lieu de « sifiri »)
])
def test_numeration_bambara(n, attendu):
    assert bambara(n) == attendu


def test_compter_jusqu_a_vingt():
    # Test au micro du 15 octobre : NLLB répétait « tan » et sautait 11 et 19.
    fr = ("Je commence : un, deux, trois, quatre, cinq, six, sept, huit, neuf, dix, onze, "
          "douze, treize, quatorze, quinze, seize, dix-sept, dix-huit, dix-neuf, vingt.")
    chiffres = en_chiffres(fr)
    assert chiffres == "Je commence : " + ", ".join(str(i) for i in range(1, 21)) + "."
    bm = en_bambara(chiffres.replace("Je commence", "N bɛ daminɛ"))
    assert bm.count("tan ni") == 9 and "tan ni kelen," in bm and "tan ni kɔnɔntɔn," in bm
    assert bm.endswith("mugan.")


@pytest.mark.parametrize("fr, attendu", [
    ("Vingt et un, quatre-vingt-dix-neuf, soixante et onze.", "21, 99, 71."),
    ("deux cent quatre-vingts, deux mille vingt-six", "280, 2026"),
    ("Il a neuf ans.", "Il a 9 ans."),
    ("J'en veux une.", "J'en veux 1."),
    ("Cinq six sept.", "5 6 7."),  # une énumération, pas 18
    # Articles, adjectif et mots qui contiennent un nombre : intacts.
    ("Il y a un homme et une femme.", "Il y a un homme et une femme."),
    ("Un livre neuf.", "Un livre neuf."),
    ("Le dixième jour de septembre.", "Le dixième jour de septembre."),
])
def test_en_chiffres(fr, attendu):
    assert en_chiffres(fr) == attendu


def test_en_bambara_garde_les_decimaux():
    assert en_bambara("1,5 ni 1 000 ni 7") == "1,5 ni waa kelen ni wolonwula"


class _MT:
    """Traduction factice : recopie, comme NLLB pour les chiffres."""

    def __init__(self):
        self.calls = []

    def translate(self, text):
        self.calls.append(text)
        return text


def _pipeline(**kw):
    from bambara_voice.pipeline import VoicePipeline

    pipe = VoicePipeline(PipelineConfig(**kw))
    pipe.mt_out = _MT()
    return pipe


def test_la_chaine_ecrit_les_nombres_et_traduit_phrase_par_phrase():
    pipe = _pipeline()
    out = pipe._translate_reply("Je compte : un, deux, onze. C'est fini !")
    assert pipe.mt_out.calls == ["Je compte : 1, 2, 11.", "C'est fini !"]
    assert out == "Je compte : kelen, fila, tan ni kelen. C'est fini !"


def test_reglages_desactivables():
    pipe = _pipeline(spell_numbers=False, translate_by_sentence=False)
    assert pipe._translate_reply("Un, deux. Trois.") == "Un, deux. Trois."
    assert pipe.mt_out.calls == ["Un, deux. Trois."]


class _ASR:
    def __init__(self, text, language="bm"):
        self.text, self.language = text, language

    def transcribe(self, audio):
        from bambara_voice.asr import ASRResult
        return ASRResult(text=self.text, language=self.language, audio_seconds=1.0)


class _LLM:
    def __init__(self):
        self.calls = []

    def reply(self, text, history=None):
        self.calls.append(text)
        return "Réponse."


def test_annotation_seule_de_la_reconnaissance_demande_de_repeter():
    """Soloni rend parfois « <tag> » seul : ce n'est pas de la parole, et le
    traduire donnait « Tag », puis « je ne peux pas vous aider »."""
    import numpy as np

    from bambara_voice.normalize import strip_tags
    from bambara_voice.pipeline import PAS_COMPRIS_FR

    assert strip_tags("<tag>") == "" and strip_tags("i ni ce <tag> [musique] wa") == "i ni ce wa"
    assert strip_tags("n b'a fɔ") == "n b'a fɔ"

    pipe = _pipeline()
    pipe.asr, pipe.mt_in, pipe.llm = _ASR("<tag>"), _MT(), _LLM()
    trace, _ = pipe.run(np.zeros(16_000, np.float32), synthesize=False)
    assert trace.asr_text == "<tag>" and trace.source_bm == "" and trace.source_fr == ""
    assert pipe.mt_in.calls == [] and pipe.llm.calls == []
    assert trace.reply_source == "pas_compris" and trace.reply_fr == PAS_COMPRIS_FR

    # Une annotation au milieu de la parole est seulement retirée.
    pipe = _pipeline()
    pipe.asr, pipe.mt_in, pipe.llm = _ASR("i ni sɔgɔma <tag>"), _MT(), _LLM()
    trace, _ = pipe.run(np.zeros(16_000, np.float32), synthesize=False)
    assert trace.source_bm == "i ni sɔgɔma" and pipe.mt_in.calls == ["i ni sɔgɔma"]
    assert trace.reply_source != "pas_compris"
