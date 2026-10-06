"""Tests de eval/analysis.py (analyse d'erreurs)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import analysis  # noqa: E402
from eval.analysis import asr_analysis, has_loop, mt_analysis  # noqa: E402
from eval.metrics import score_asr, score_mt  # noqa: E402

ASR_ROWS = [
    {"id": 0, "ref_bm": "n bɛ taa sugu la", "hyp_bm": "n be taa sugu la"},     # orthographe
    {"id": 1, "ref_bm": "i ni sɔgɔma", "hyp_bm": "i ni sɔgɔma"},                # parfait
    {"id": 2, "ref_bm": "a bɛ yen", "hyp_bm": "a bɛ yen a bɛ yen a bɛ yen"},    # boucle
    {"id": 3, "ref_bm": "sugu ka jan kosɛbɛ", "hyp_bm": ""},                    # vide
    {"id": 4, "ref_bm": "réseau tɛ yen bi", "hyp_bm": "réseau tɛ bi"},          # suppression
]


def test_boucles_de_repetition():
    assert has_loop("a bɛ yen a bɛ yen a bɛ yen")
    assert not has_loop("n bɛ taa sugu la")
    assert not has_loop("")
    # Boucle dans un seul « mot » (pire sortie de NLLB en phase 1), et
    # redoublement légitime, qui n'en est pas une.
    assert has_loop("sɔgɔsɔgɔsɔgɔsɔgɔsɔgɔsɔgɔs")
    assert not has_loop("bama kulela a y'i munumunu")
    assert not has_loop("a bɛ sɔgɔsɔgɔ")


def test_decomposition_redonne_le_wer_de_eval_metrics():
    a = asr_analysis(ASR_ROWS)
    expected = score_asr([r["hyp_bm"] for r in ASR_ROWS], [r["ref_bm"] for r in ASR_ROWS]).wer
    assert a["wer"] == pytest.approx(expected)
    assert a["substitutions"] + a["suppressions"] + a["insertions"] == pytest.approx(expected)


def test_types_d_erreurs_asr():
    a = asr_analysis(ASR_ROWS)
    assert a["n"] == 5
    assert a["substitutions_orthographiques"] == 1.0  # la seule substitution : bɛ → be
    assert a["confusions"] == [("bɛ → be", 1)]
    assert a["sorties_vides"] == pytest.approx(1 / 5)
    assert a["hallucinations"] == pytest.approx(1 / 5) and a["boucles"] == pytest.approx(1 / 5)
    assert dict(a["mots_inseres"])["yen"] == 2
    assert "yen" in dict(a["mots_supprimes"])
    assert a["exemples"]["meilleur"]["reference"] == "i ni sɔgɔma"
    assert a["exemples"]["pire"]["sortie"] == "a bɛ yen a bɛ yen a bɛ yen"
    assert set(a["wer_par_longueur"]) == {"1–3", "4–7"}


MT_ROWS = [
    {"ref_bm": "i ni sɔgɔma", "ref_fr": "bonjour", "hyp_fr": "bonjour"},
    {"ref_bm": "a bɛ yen", "ref_fr": "il est là", "hyp_fr": "a bɛ yen"},          # recopie
    {"ref_bm": "n bɛ taa sugu la sisan", "ref_fr": "je vais au marché maintenant",
     "hyp_fr": "je vais"},                                                       # trop courte
    {"ref_bm": "n tɛ se", "ref_fr": "je ne peux pas", "hyp_fr": ""},             # vide
]


def test_analyse_traduction():
    m = mt_analysis(MT_ROWS, "ref_bm", "hyp_fr", "ref_fr")
    expected = score_mt([r["hyp_fr"] for r in MT_ROWS], [r["ref_fr"] for r in MT_ROWS]).chrf
    assert m["chrf"] == pytest.approx(expected)
    assert m["recopies_de_la_source"] == pytest.approx(1 / 4)
    assert m["sorties_vides"] == pytest.approx(1 / 4)
    assert m["sorties_trop_courtes"] == pytest.approx(2 / 4)  # la vide et « je vais »
    assert m["exemples"]["meilleur"]["sortie"] == "bonjour"
    assert m["exemples"]["meilleur"]["source"] == "i ni sɔgɔma"
    assert sum(v["n"] for v in m["chrf_par_longueur_source"].values()) == 4


def test_sorties_coupees_et_boucles():
    rows = [{**r, "coupe_fr": i == 1} for i, r in enumerate(MT_ROWS)]
    rows[1]["hyp_fr"] = "il il il il il il il il il"
    m = mt_analysis(rows, "ref_bm", "hyp_fr", "ref_fr", "coupe_fr")
    assert m["coupees"] == pytest.approx(1 / 4) and m["coupees_en_boucle"] == 1.0
    # Sans l'information (rapports de la phase 1), rien n'est inventé.
    assert "coupees" not in mt_analysis(MT_ROWS, "ref_bm", "hyp_fr", "ref_fr", "coupe_fr")


def _report(path: Path, report: dict, rows: list[dict]) -> Path:
    path.write_text(json.dumps(report), encoding="utf-8")
    path.with_suffix(".details.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def test_cli_rapports_asr_et_mt(tmp_path, capsys):
    rows = [{**r, "ref_fr": "bonjour", "mt_depuis_asr": "bonjour"} for r in ASR_ROWS]
    _report(tmp_path / "asr.json", {"nom": "MMS", "architecture": "cascade", "maillon": "asr"}, rows)
    _report(tmp_path / "mt.json", {"nom": "NLLB", "architecture": "traduction", "maillon": "mt"},
            [{**r, "hyp_bm": r["ref_bm"]} for r in MT_ROWS])
    out_json = tmp_path / "analyse.json"
    analysis.main([str(p) for p in sorted(tmp_path.glob("*.json"))] + ["--json", str(out_json)])
    out = capsys.readouterr().out
    assert "### MMS" in out and "Substitutions purement orthographiques" in out
    assert "Traduction de la sortie de l'ASR" in out
    assert "### NLLB" in out and "français → bambara" in out
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert set(saved) == {"MMS", "NLLB"}
    assert saved["NLLB"]["fr_bm"]["chrf"] == pytest.approx(100)


def _run_eval_rows(e2e: bool = False) -> list[dict]:
    """Sorties de eval.run_eval sur le jeu maison, avec ses métadonnées."""
    rows = []
    for i, r in enumerate(ASR_ROWS):
        rows.append({
            "item": f"bv-{i:04d}", "source_bm": "" if e2e else r["hyp_bm"],
            "source_fr": "bonjour" if i % 2 else "il est là",
            "ref_transcript_bm": r["ref_bm"], "ref_translation_fr": "bonjour",
            "speaker": f"spk-0{i % 2}", "gender": "f", "register": "lu" if i < 2 else "spontane",
            "code_switching": i == 4, "conditions": "salle calme",
        })
    return rows


def test_rapport_run_eval_et_sous_groupes(tmp_path, capsys):
    rows = _run_eval_rows()
    path = _report(tmp_path / "cascade-20261006.json", {"architecture": "cascade"}, rows)
    name, a = analysis.analyse_report(path)
    assert name == "cascade-20261006"
    assert a["asr"]["wer"] == pytest.approx(asr_analysis(ASR_ROWS)["wer"])
    assert a["mt_depuis_asr"]["n"] == 5

    groups = a["par_groupe"]
    # Un champ qui ne prend qu'une valeur (genre, conditions) n'apprend rien.
    assert set(groups) == {"registre", "code-switching", "locuteur"}
    lu = ASR_ROWS[:2]
    assert groups["registre"]["lu"] == {
        "n": 2, "locuteurs": 2, "chrf": pytest.approx(score_mt(["il est là", "bonjour"],
                                                               ["bonjour"] * 2).chrf),
        "wer": pytest.approx(score_asr([r["hyp_bm"] for r in lu], [r["ref_bm"] for r in lu]).wer)}
    assert groups["code-switching"]["avec"]["n"] == 1
    assert groups["locuteur"]["spk-00"]["n"] == 3

    analysis.main([str(path)])
    out = capsys.readouterr().out
    assert "**Par sous-groupe**" in out and "| registre : spontane | 3 | 2 |" in out
    assert "| Sous-groupe | Énoncés | Locuteurs | WER strict | chrF++ |" in out


def test_rapport_run_eval_bout_en_bout(tmp_path, capsys):
    path = _report(tmp_path / "e2e-20261006.json", {"architecture": "e2e"},
                   _run_eval_rows(e2e=True))
    _, a = analysis.analyse_report(path)
    assert set(a) == {"e2e", "par_groupe"}
    assert all("wer" not in e for t in a["par_groupe"].values() for e in t.values())
    analysis.main([str(path)])
    out = capsys.readouterr().out
    assert "Traduction directe (bout-en-bout)" in out
    assert "| Sous-groupe | Énoncés | Locuteurs | chrF++ |" in out
