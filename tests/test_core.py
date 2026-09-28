"""Tests des briques qui ne demandent aucun modèle.

    python -m pytest tests/ -q

Volontairement limités à la logique déterministe : normalisation, appariement
de gabarits, simplification de sortie, métriques. Les composants à modèles
sont vérifiés par `python -m bambara_voice.cli check`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bambara_voice.normalize import normalize, fold, orthography_ratio
from bambara_voice.llm import simplify_for_translation
from bambara_voice.templates import TemplateBank, Template, dice
from eval.metrics import score_asr, score_mt
from eval.dataset import load_testset, describe


# --- normalisation ---------------------------------------------------------

def test_normalize_preserve_les_caracteres_bambara():
    assert normalize("I ni cɛ") == "i ni cɛ"
    assert "ɛ" in normalize("Ɛntɛrinɛti", lower=False)


def test_normalize_unifie_apostrophes_et_espaces():
    assert normalize("n’bɛ   taa") == "n'bɛ taa"


def test_fold_rend_equivalentes_les_graphies_concurrentes():
    assert fold("Ɛntɛrinɛti bɛ yen wa?") == fold("Enterineti be yen wa")
    assert fold("ɲɔgɔn") == fold("nyogon")
    assert fold("kɔŋɔ") == fold("kongo")


def test_orthography_ratio():
    assert orthography_ratio(["a bɛ", "a be"]) == 0.5
    assert orthography_ratio([]) == 0.0


# --- simplification de la sortie LLM ---------------------------------------

def test_simplify_coupe_les_phrases_trop_longues():
    long = " ".join(["mot"] * 40) + "."
    out = simplify_for_translation(long, max_words=15)
    assert len(out.split()) <= 16  # 15 mots + la ponctuation finale


def test_simplify_retire_la_mise_en_forme():
    out = simplify_for_translation("**Gras** et `code` et *italique*.")
    assert "*" not in out and "`" not in out


def test_simplify_limite_le_nombre_de_phrases():
    out = simplify_for_translation("Un. Deux. Trois. Quatre. Cinq.")
    assert out.count(".") == 3


# --- gabarits ---------------------------------------------------------------

def _bank():
    return TemplateBank(
        [Template(id="greeting", triggers_bm=["i ni sɔgɔma"],
                  response_bm="Nba, i ni ce.", response_fr="Bonjour.")],
        threshold=0.62,
    )


def test_gabarit_tolere_les_erreurs_d_asr():
    # "soguma" est l'erreur d'ASR typique sur "sɔgɔma" : elle doit matcher.
    match = _bank().match("i ni soguma")
    assert match is not None and match.template.id == "greeting"


def test_gabarit_ne_matche_pas_hors_sujet():
    assert _bank().match("dugu kɔnɔ mobili bɛ yen wa") is None


def test_banque_vide_renvoie_toujours_none():
    assert TemplateBank([]).match("i ni ce") is None


def test_dice_borne():
    assert dice("abc", "abc") == 1.0
    assert dice("", "abc") == 0.0


# --- métriques --------------------------------------------------------------

def test_ecart_orthographique_est_isole_par_le_repli():
    # Hypothèse correcte au son, écrite en ASCII : le WER strict la pénalise,
    # le WER relâché non. C'est exactement ce que l'écart doit mesurer.
    scores = score_asr(["i ni ce n be taa"], ["i ni cɛ n bɛ taa"])
    assert scores.wer > 0
    assert scores.wer_folded == 0
    assert scores.orthographic_gap == scores.wer


def test_la_ponctuation_ne_compte_ni_en_strict_ni_dans_l_ecart():
    # Une virgule ou un point de plus n'est pas une erreur de reconnaissance,
    # et ne doit pas gonfler l'écart « orthographique ».
    scores = score_asr(["I ni cɛ, n bɛ taa."], ["i ni cɛ n bɛ taa"])
    assert scores.wer == 0 and scores.orthographic_gap == 0
    # L'apostrophe, elle, est orthographique : comptée en strict seulement.
    scores = score_asr(["ka fɔ"], ["k'a fɔ"])
    assert scores.wer > 0 and scores.wer_folded == 0


def test_score_mt_identique_donne_100():
    scores = score_mt(["bonjour"], ["bonjour"])
    assert round(scores.chrf) == 100


def test_score_asr_refuse_les_longueurs_incoherentes():
    try:
        score_asr(["a"], ["a", "b"])
    except ValueError:
        return
    raise AssertionError("aurait dû lever ValueError")


# --- jeu de test ------------------------------------------------------------

def test_chargement_du_jeu_de_test_exemple():
    path = Path(__file__).resolve().parent.parent / "data/testset/testset.example.jsonl"
    items = load_testset(path)
    assert len(items) == 3
    stats = describe(items)
    assert stats["locuteurs"] == 3
    assert 0 < stats["part_code_switching"] < 1
