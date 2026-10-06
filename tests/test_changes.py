"""Tests de eval/changes.py (ce qui change d'une exécution à l'autre)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import changes  # noqa: E402

CORPUS = {"id": "RobotsMaliAI/bayelemabaga", "partition": "test"}


def _report(path: Path, name: str, rows: list[dict], torch: str = "2.14") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"nom": name, "maillon": "mt", "corpus": CORPUS,
                                "environnement": {"appareil": "cpu", "torch": torch}}),
                    encoding="utf-8")
    path.with_suffix(".details.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def _row(i, hyp_bm, **kw):
    return {"id": i, "ref_bm": "a bɛ yen", "ref_fr": "il est là", "hyp_fr": "il est là",
            "hyp_bm": hyp_bm, **kw}


BEFORE = [_row(0, "a bɛ yen"), _row(1, "sɔgɔ" * 60), _row(2, "a bɛ yan")]


def test_seules_les_phrases_coupees_changent(tmp_path, capsys):
    a = _report(tmp_path / "a" / "mt.json", "sans plafond", BEFORE)
    b = _report(tmp_path / "b" / "mt.json", "avec plafond", [
        _row(0, "a bɛ yen", coupe_bm=False), _row(1, "sɔgɔ" * 6, coupe_bm=True),
        _row(2, "a bɛ yan", coupe_bm=False)])
    res = changes.compare(changes._load(a), changes._load(b))
    assert res["n"] == 3 and res["references_differentes"] == 0
    assert res["versions_differentes"] == {}
    assert res["sorties"]["hyp_bm"] == {
        "n": 3, "changees": 1, "changees_coupees": 1, "changees_sans_coupure": 0,
        "coupees_apres": 1, "exemples_sans_coupure": [],
        "exemples_coupees": [{"id": "1", "avant": changes._short("sɔgɔ" * 60),
                              "apres": "sɔgɔ" * 6}]}
    assert res["sorties"]["hyp_fr"]["changees"] == 0

    changes.main([str(a), "--apres", str(b)])
    out = capsys.readouterr().out
    assert "### sans plafond → avec plafond" in out and "Mêmes versions" in out
    assert "| hyp_bm | 3 | 1 | 1 | 0 |" in out


def test_changement_sans_coupure_et_versions_signales(tmp_path, capsys):
    a = _report(tmp_path / "a" / "mt.json", "avant", BEFORE)
    b = _report(tmp_path / "b" / "mt.json", "après", [
        _row(0, "a bɛ yen"), _row(1, "sɔgɔ" * 60), _row(2, "a be yan")], torch="2.15")
    res = changes.compare(changes._load(a), changes._load(b))
    assert res["sorties"]["hyp_bm"]["changees_sans_coupure"] == 1
    assert res["versions_differentes"] == {"torch": ("2.14", "2.15")}
    changes.main([str(a), "--apres", str(b)])
    assert "torch 2.14 → 2.15" in capsys.readouterr().out


def test_autres_phrases_sous_le_meme_identifiant(tmp_path):
    a = _report(tmp_path / "a" / "mt.json", "avant", BEFORE)
    moved = [dict(r, ref_bm="autre phrase") for r in BEFORE]
    b = _report(tmp_path / "b" / "mt.json", "après", moved)
    assert changes.compare(changes._load(a), changes._load(b))["references_differentes"] == 3


def test_rien_a_apparier(tmp_path):
    a = _report(tmp_path / "a" / "mt.json", "avant", BEFORE)
    b = tmp_path / "b" / "mt.json"
    _report(b, "après", BEFORE)
    report = json.loads(b.read_text(encoding="utf-8"))
    report["corpus"] = {"id": "RobotsMali/jeli-asr", "partition": "test"}
    b.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit, match="aucun rapport apparié"):
        changes.main([str(a), "--apres", str(b)])
