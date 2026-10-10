"""Tests de eval/recouvrement.py : nos phrases de test dans l'entraînement
d'un modèle publié."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("datasets")
pytest.importorskip("soundfile")

import tiny_models  # noqa: E402
from eval import recouvrement  # noqa: E402


def _report(path: Path, refs: list[str]) -> Path:
    path.write_text(json.dumps({"nom": "x"}), encoding="utf-8")
    path.with_suffix(".details.jsonl").write_text(
        "\n".join(json.dumps({"id": i, "ref_bm": r}, ensure_ascii=False)
                  for i, r in enumerate(refs)), encoding="utf-8")
    return path


def test_phrases_de_test_retrouvees_dans_l_entrainement(tmp_path, capsys):
    jeu = tiny_models.speech_dataset(tmp_path / "jeu")  # train : « n bɛ taa sugu la »...
    rapport = _report(tmp_path / "asr-1.json", [
        "N bɛ taa sugu la.",           # dans train, autre graphie de surface
        "n be taa sugu la",            # repli orthographique : la même phrase
        "a tɛ taa sugu la bi",         # absente
        "a bɛ yen",                    # présente mais trop courte : ignorée
    ])
    vus = tmp_path / "vus.txt"
    for _ in range(2):  # ajouté au fichier : plusieurs jeux d'entraînement à la suite
        recouvrement.main([str(rapport), "--dataset", str(jeu), "--split", "train",
                           "--min-mots", "4", "--ids-vus", str(vus)])
    out = capsys.readouterr().out
    assert "colonnes : bambara=bam" in out
    assert "**2 sur 3** références de test d'au moins 4 mots (66.7 %)" in out
    assert "(4 références en tout)" in out and "« N bɛ taa sugu la. »" in out
    # Toutes longueurs : la phrase courte aussi, au plus prudent.
    assert "Toutes longueurs confondues : 3 références." in out
    assert vus.read_text().split() == ["0", "1", "3"] * 2


def test_rapport_sans_reference(tmp_path):
    rapport = _report(tmp_path / "mt-1.json", [])
    with pytest.raises(SystemExit, match="aucune référence"):
        recouvrement.main([str(rapport), "--dataset", "x"])
