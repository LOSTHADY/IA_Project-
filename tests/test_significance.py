"""Tests de eval/significance.py (bootstrap) et eval/mos.py (grilles MOS)."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import mos, significance  # noqa: E402
from eval.metrics import score_asr, score_mt  # noqa: E402
from eval.significance import (  # noqa: E402
    Series, bootstrap, chrf_stats, corpus_score, load_series, wer_stats,
)

HYPS = ["i ni ce, n be taa.", "a bɛ yen", "", "sugu ka jan"]
REFS = ["i ni cɛ n bɛ taa", "a bɛ yen wa", "n ma faamu", "sugu ka jan kosɛbɛ"]
FR_H = ["bonjour je pars", "il est là", "", "le marché est loin"]
FR_R = ["bonjour, je m'en vais", "il est là ?", "je n'ai pas compris", "le marché est très loin"]


# --- les statistiques par énoncé redonnent les scores du corpus -----------------

def test_wer_identique_a_eval_metrics():
    ref = score_asr(HYPS, REFS)
    assert corpus_score("wer", wer_stats(HYPS, REFS).sum(axis=0)) == pytest.approx(ref.wer)
    assert corpus_score("wer", wer_stats(HYPS, REFS, folded=True).sum(axis=0)) == \
        pytest.approx(ref.wer_folded)


def test_chrf_identique_a_eval_metrics():
    # Garde aussi l'API interne de sacrebleu utilisée pour le bootstrap.
    stats = chrf_stats(FR_H, FR_R)
    assert stats.shape[0] == len(FR_H)
    assert corpus_score("chrf", stats.sum(axis=0)) == pytest.approx(score_mt(FR_H, FR_R).chrf)


# --- bootstrap ----------------------------------------------------------------

def _chrf_series(name, hyps, refs):
    return Series(name, "chrf", [str(i) for i in range(len(hyps))], chrf_stats(hyps, refs))


def test_systemes_identiques_ecart_non_significatif():
    a = _chrf_series("a", FR_H * 10, FR_R * 10)
    b = _chrf_series("b", FR_H * 10, FR_R * 10)
    res = bootstrap([a, b], n_boot=200)
    entry = res["systemes"][1]
    assert entry["ecart"] == 0 and entry["p"] == 1.0
    assert entry["ic_ecart"][0] <= 0 <= entry["ic_ecart"][1]


def test_systeme_nettement_meilleur_ecart_significatif():
    rng = np.random.default_rng(0)
    refs = [f"le marché numéro {i} est loin" for i in range(60)]
    good = list(refs)
    bad = [" ".join(rng.permutation(r.split())[:3]) for r in refs]
    res = bootstrap([_chrf_series("mauvais", bad, refs), _chrf_series("bon", good, refs)],
                    n_boot=300)
    entry = res["systemes"][1]
    assert entry["ecart"] > 0 and entry["ic_ecart"][0] > 0 and entry["p"] < 0.05
    lo, hi = res["systemes"][0]["ic"]
    assert lo <= res["systemes"][0]["score"] <= hi


def test_bootstrap_aligne_sur_les_enonces_communs():
    a = Series("a", "wer", ["1", "2", "3"], wer_stats(HYPS[:3], REFS[:3]))
    b = Series("b", "wer", ["3", "2"], wer_stats([HYPS[2], HYPS[1]], [REFS[2], REFS[1]]))
    assert bootstrap([a, b], n_boot=50)["n"] == 2
    with pytest.raises(ValueError, match="aucun énoncé commun"):
        bootstrap([a, Series("c", "wer", ["9"], wer_stats(["x"], ["y"]))], n_boot=10)


# --- lecture des rapports -----------------------------------------------------

def _report(path: Path, report: dict, rows: list[dict]) -> Path:
    path.write_text(json.dumps(report), encoding="utf-8")
    with path.with_suffix(".details.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _asr_rows(e2e: bool):
    rows = []
    for i, (h, r, fh_, fr) in enumerate(zip(HYPS, REFS, FR_H, FR_R)):
        row = {"id": i, "ref_bm": r, "ref_fr": fr}
        row.update({"hyp_fr": fh_} if e2e else {"hyp_bm": h, "mt_depuis_asr": fh_})
        rows.append(row)
    return rows


def test_series_des_rapports_baselines_et_run_eval(tmp_path):
    cascade = _report(tmp_path / "c.json", {"nom": "cascade", "architecture": "cascade",
                                            "maillon": "asr"}, _asr_rows(e2e=False))
    name, series = load_series(cascade)
    assert name == "cascade"
    assert set(series) == {"WER strict", "WER relâché", "chrF++ (depuis ASR)"}
    assert series["WER strict"].value == pytest.approx(score_asr(HYPS, REFS).wer)

    e2e = _report(tmp_path / "e.json", {"nom": "e2e", "architecture": "e2e", "maillon": "asr"},
                  _asr_rows(e2e=True))
    assert set(load_series(e2e)[1]) == {"chrF++ (depuis ASR)"}

    run_eval = _report(tmp_path / "r.json", {"architecture": "cascade"}, [
        {"item": f"bv-{i}", "source_bm": h, "source_fr": f, "ref_transcript_bm": r,
         "ref_translation_fr": fr} for i, (h, r, f, fr) in enumerate(zip(HYPS, REFS, FR_H, FR_R))])
    name, series = load_series(run_eval)
    assert name == "cascade" and series["chrF++ (depuis ASR)"].ids[0] == "bv-0"

    mt = _report(tmp_path / "m.json", {"nom": "nllb", "architecture": "traduction",
                                       "maillon": "mt"},
                 [{"id": 0, "hyp_fr": "il est là", "ref_fr": "il est là",
                   "hyp_bm": "a be yen", "ref_bm": "a bɛ yen"}])
    series = load_series(mt)[1]
    assert series["chrF++ fr→bm (replié)"].value == pytest.approx(100)
    assert series["chrF++ fr→bm"].value < 100


def test_cli_compare_cascade_et_bout_en_bout(tmp_path, capsys):
    _report(tmp_path / "c.json", {"nom": "cascade", "architecture": "cascade", "maillon": "asr"},
            _asr_rows(e2e=False))
    _report(tmp_path / "e.json", {"nom": "bout-en-bout", "architecture": "e2e", "maillon": "asr"},
            _asr_rows(e2e=True))
    significance.main([str(p) for p in sorted(tmp_path.glob("*"))]
                      + ["--noms", "bout-en-bout", "cascade", "--n-boot", "100"])
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == "| Métrique | n | bout-en-bout | cascade |"
    # Seule métrique commune aux deux architectures.
    assert len([line for line in lines if line.startswith("| chrF++")]) == 1
    assert "écart" in out and "IC à 95 %" in out

    with pytest.raises(SystemExit, match="introuvables"):
        significance.main([str(tmp_path / "c.json"), "--noms", "inconnu"])


# --- MOS ----------------------------------------------------------------------

def _grid(path: Path, listener: str, notes: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "texte_bm", "fichier", "auditeur", "note_1_a_5", "commentaire"])
        for i, n in enumerate(notes):
            w.writerow([f"tts-{i:03d}", "i ni ce", f"tts-{i:03d}.wav", listener, n, ""])
    return path


def test_mos_moyenne_ic_et_auditeurs(tmp_path, capsys):
    d = tmp_path / "tts-mms"
    _grid(d / "mos-a.csv", "A", ["4", "5", "3", ""])  # une phrase non notée
    _grid(d / "mos-b.csv", "B", ["2", "3", "2", "3"])
    mos.main([str(p) for p in sorted(d.glob("*.csv"))])
    out = capsys.readouterr().out
    assert "| tts-mms | 3.14 [" in out and "| 4 | 7 |" in out
    assert "A 4.00, B 2.50" in out
    assert "2 auditeur(s)" in out  # moins de trois auditeurs : avertissement


def test_mos_note_invalide_signalee(tmp_path):
    with pytest.raises(ValueError, match="hors de 1 à 5"):
        mos.read_grid(_grid(tmp_path / "t" / "mos.csv", "A", ["6"]))
    with pytest.raises(ValueError, match="illisible"):
        mos.read_grid(_grid(tmp_path / "u" / "mos.csv", "A", ["bien"]))
    assert mos.read_grid(_grid(tmp_path / "v" / "mos.csv", "A", ["3,5"]))[0]["note"] == 3.5
