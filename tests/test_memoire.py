"""La mémoire (bambara_voice/memoire.py) : ce que l'assistant apprend du
locuteur est retenu tout de suite, sans réentraînement, et mesuré
(eval/redire.py)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bambara_voice.config import MemoireConfig, PipelineConfig, load_config  # noqa: E402
from bambara_voice.memoire import Memoire  # noqa: E402
from test_nombres import _ASR, _LLM, _MT  # noqa: E402


def test_un_souvenir_sert_tout_de_suite_et_la_fois_suivante(tmp_path):
    corrections = tmp_path / "memoire.jsonl"
    m = Memoire(corrections, seuil=0.8)
    assert m.chercher("I ni ce, i ka kɛnɛ wa") is None
    m.apprendre("I ni ce i ka kɛnɛ wa", "Bonjour, tu vas bien ?")
    # Même phrase, entendue un peu autrement (graphie, ponctuation, un mot).
    for entendu in ("i ni ce, i ka kene wa ?", "I ni ce i ka kɛnɛ wa"):
        souvenir, score = m.chercher(entendu)
        assert souvenir.fr == "Bonjour, tu vas bien ?" and score >= 0.8
    assert m.chercher("N bɛ taa sugu la") is None          # rien à voir : la traduction
    # Relue au démarrage suivant ; une correction plus récente l'emporte.
    m.apprendre("i ni ce i ka kɛnɛ wa", "Salut, ça va ?")
    relue = Memoire(corrections, seuil=0.8)
    assert len(relue) == 1 and relue.chercher("I ni ce i ka kɛnɛ wa")[0].fr == "Salut, ça va ?"
    with pytest.raises(ValueError):
        m.apprendre("", "Bonjour")


def test_les_mots_du_lexique_arrivent_au_traducteur_sous_une_forme_connue(tmp_path):
    lexique = tmp_path / "lexique.jsonl"
    lexique.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in [
        {"forme": "wai", "sens_fr": "salut", "equivalent_bm": "i ni ce"},
        {"forme": "a ka ɲi kojugu", "sens_fr": "c'est super", "equivalent_bm": "a ka ɲi kosɛbɛ"},
        {"forme": "yoh", "sens_fr": "hé", "equivalent_bm": ""}]), encoding="utf-8")
    m = Memoire(lexique=lexique)
    assert m.remplacer_mots("Wai, a ka ɲi kojugu yoh") == "i ni ce a ka ɲi kosɛbɛ yoh"
    assert Memoire().remplacer_mots("wai") == "wai"


def _pipeline(tmp_path, **memoire):
    from bambara_voice.pipeline import VoicePipeline

    pipe = VoicePipeline(PipelineConfig(memoire=MemoireConfig(**memoire)))
    pipe.mt_out, pipe.mt_in, pipe.llm = _MT(), _MT(), _LLM()
    return pipe


def test_la_chaine_comprend_par_la_memoire_puis_apprend_d_une_correction(tmp_path):
    atelier = tmp_path / "memoire-atelier.jsonl"
    atelier.write_text(json.dumps({"bm": "N bɛ taa sugu la", "fr": "Je vais au marché.",
                                   "source": "atelier"}, ensure_ascii=False) + "\n")
    corrections = tmp_path / "corrections.jsonl"
    pipe = _pipeline(tmp_path, path=str(corrections), atelier=str(atelier), seuil=0.8)
    son = np.zeros(16_000, np.float32)

    pipe.asr = _ASR("n bɛ taa sugu la")
    trace, _ = pipe.run(son, synthesize=False)
    assert trace.source_fr == "Je vais au marché." and trace.compris_par == "memoire"
    assert pipe.mt_in.calls == []                            # pas de traduction

    pipe.asr = _ASR("a bɛ dumuni kɛ")
    trace, _ = pipe.run(son, synthesize=False)
    assert trace.compris_par == "mt" and pipe.mt_in.calls == ["a bɛ dumuni kɛ"]

    # « Retiens-le » dans la démo : la même phrase est comprise la fois suivante.
    from app.gradio_app import retenir
    assert retenir(pipe, trace, "") == "Écris en français ce que tu voulais dire."
    assert "Retenu" in retenir(pipe, trace, "Il mange.")
    trace2, _ = pipe.run(son, synthesize=False)
    assert trace2.source_fr == "Il mange." and trace2.compris_par == "memoire"
    assert json.loads(corrections.read_text())["source"] == "correction"
    assert len(_pipeline(tmp_path, path=str(corrections), atelier=str(atelier)).memoire) == 2
    assert retenir(pipe, None, "x").startswith("Rien à retenir")


def test_memoire_dans_un_fichier_de_deploiement(tmp_path):
    (tmp_path / "lexique.jsonl").write_text("")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"memoire": {"path": "/content/drive/memoire.jsonl",
                                           "lexique": "lexique.jsonl", "seuil": 0.9}}))
    c = load_config(cfg)
    assert c.memoire.path == "/content/drive/memoire.jsonl" and c.memoire.seuil == 0.9
    assert c.memoire.lexique == str(tmp_path / "lexique.jsonl")


# --- eval/redire.py -------------------------------------------------------------------

def _jeu(tmp_path, comprend_mal: bool = True):
    """Trente phrases dites deux fois (dev et test) ; Soloni entend la même
    chose les deux fois ; le traducteur comprend mal (ou bien)."""
    prises, rows = [], []
    for k in range(60):
        part = "dev" if k < 30 else "test"
        for fois in range(2):
            item = f"di-{fois * 100 + k:05d}"
            ref = f"phrase numéro {k} pour aller au marché"
            prises.append({"id": item, "prompt_id": f"p-{k}"})
            rows.append({"item": item, "source_bm": f"n bɛ {'taa ' * (k % 7)}sugu {k} la",
                         "source_fr": "il est parti hier" if comprend_mal else ref,
                         "ref_translation_fr": ref, "partition": part, "speaker": "spk-01"})
    (tmp_path / "dire.jsonl").write_text("\n".join(json.dumps(p) for p in prises))
    rapport = tmp_path / "r" / "cascade-1.json"
    rapport.parent.mkdir()
    rapport.write_text(json.dumps({"nom": "Soloni + NLLB v1"}))
    rapport.with_suffix(".details.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return rapport, tmp_path / "dire.jsonl"


def test_redire_une_phrase_apprise_est_comprise_la_fois_suivante(tmp_path, capsys):
    pytest.importorskip("sacrebleu")
    from eval import redire

    rapport, prises = _jeu(tmp_path)
    d = redire.mesurer(rapport, prises, tmp_path / "memoire", n_boot=200)
    assert d["promu"] and d["phrases_apprises"] == 60 and d["redites"] == 60
    assert d["redire"]["candidat"] == pytest.approx(100.0) and d["redire"]["ic"][0] > 0
    assert d["souvenirs_justes"] == d["souvenirs_utilises"] == 60
    assert d["jamais_dit"]["ecart"] >= 0

    redire.main(["--rapport", str(rapport), "--prises", str(prises), "--out",
                 str(tmp_path / "m2"), "--memoire", str(tmp_path / "memoire-atelier.jsonl"),
                 "--decision", str(tmp_path / "decision.json")])
    sortie = capsys.readouterr().out
    assert "**Mémoire en service.**" in sortie and "Mémoire à déployer : 120 phrases" in sortie
    assert json.loads((tmp_path / "decision.json").read_text())["promu"]
    deployee = Memoire(atelier=tmp_path / "memoire-atelier.jsonl", seuil=d["seuil"])
    assert deployee.chercher("n bɛ taa sugu 1 la")[0].fr == "phrase numéro 1 pour aller au marché"


def test_redire_refusee_si_elle_n_apporte_rien(tmp_path):
    pytest.importorskip("sacrebleu")
    from eval import redire

    rapport, prises = _jeu(tmp_path, comprend_mal=False)    # déjà tout compris
    d = redire.mesurer(rapport, prises, tmp_path / "memoire", n_boot=200)
    assert not d["promu"] and "pas mieux comprise" in d["raisons"][0]
    assert "**Mémoire refusée**" in redire.to_markdown(d)
