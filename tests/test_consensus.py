"""Le consensus des traducteurs déjà entraînés (eval/consensus.py)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("sacrebleu")

from eval import consensus  # noqa: E402

PHRASES = [f"il va au marché acheter {k} poissons ce matin" for k in range(60)]


def _rapport(path: Path, faux: set[int]) -> Path:
    """Un traducteur qui se trompe sur les phrases `faux` (60 : 30 dev, 30 test)."""
    rows = [{"item": f"di-{i:03d}", "source_fr": "il est parti hier soir" if i in faux else ref,
             "ref_translation_fr": ref, "partition": "dev" if i < 30 else "test",
             "speaker": "spk-01"} for i, ref in enumerate(PHRASES)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"nom": path.stem}))
    path.with_suffix(".details.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False)
                                                            for r in rows))
    return path


def test_la_traduction_la_plus_partagee_l_emporte():
    assert consensus.choisir(["il va au marché", "il va au marché ce matin", "le chien dort"]) \
        in ("il va au marché", "il va au marché ce matin")
    assert consensus.choisir(["", "bonjour"]) == "bonjour"
    assert consensus.choisir(["", ""]) == ""


def test_trois_traducteurs_qui_se_trompent_ailleurs_font_mieux_ensemble(tmp_path, capsys):
    # Chacun se trompe sur un tiers des phrases, jamais sur les mêmes.
    r = {"NLLB v1": _rapport(tmp_path / "v1.json", set(range(0, 60, 3))),
         "NLLB v2": _rapport(tmp_path / "v2.json", set(range(1, 60, 3))),
         "NLLB 3.3B": _rapport(tmp_path / "p.json", set(range(2, 60, 3)))}
    d = consensus.mesurer(r, "NLLB v1", tmp_path / "out", n_boot=200)
    assert d["promu"] and d["comparaison"]["candidat"] == pytest.approx(100.0)
    assert d["ensemble"] == "tous" and set(d["traducteurs"]) == set(r)
    assert "**Le consensus fait mieux.**" in consensus.to_markdown(d, "NLLB v1")

    consensus.main(["--en-service", f"NLLB v1={r['NLLB v1']}",
                    "--rapport", f"NLLB v2={r['NLLB v2']}",
                    "--rapport", f"NLLB 3.3B={r['NLLB 3.3B']}", "--out", str(tmp_path / "o2")])
    assert "consensus" in capsys.readouterr().out


def test_des_traducteurs_qui_se_trompent_ensemble_n_aident_pas(tmp_path):
    faux = set(range(0, 60, 2))
    r = {"NLLB v1": _rapport(tmp_path / "v1.json", faux),
         "NLLB v2": _rapport(tmp_path / "v2.json", faux),
         "NLLB v3": _rapport(tmp_path / "v3.json", faux)}
    d = consensus.mesurer(r, "NLLB v1", tmp_path / "out", n_boot=200)
    assert not d["promu"] and "pas meilleur" in d["raisons"][0]
