"""Le taux de compréhension jugé par le locuteur (eval/comprehension.py) et
les paires tirées de sa voix (scripts/paires_maison.py)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval import comprehension  # noqa: E402
from eval.atelier import charger_phrases_a_dire, usage  # noqa: E402


def _rapport(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"nom": "Soloni + NLLB v1", "architecture": "cascade"}))
    path.with_suffix(".details.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def test_a_juger_toujours_les_memes_phrases_et_reprise(tmp_path):
    rows = [{"item": f"di-{i:05d}", "ref_translation_fr": f"phrase {i}", "source_fr": f"compris {i}",
             "partition": "dev"} for i in range(150)] + [{"item": "di-99999", "source_fr": "x"}]
    rapport = _rapport(tmp_path / "r.json", rows)
    notes = tmp_path / "atelier" / "comprehension.jsonl"
    lot = comprehension.a_juger(rapport, notes, "S")
    assert len(lot) == 100 and lot == comprehension.a_juger(rapport, notes, "S")
    assert all(x["item"] != "di-99999" for x in lot)       # sans référence : pas jugeable
    assert lot[0]["dit"] == lot[0]["compris"].replace("compris", "phrase")
    comprehension.noter(notes, lot[0]["item"], 2, "S")
    comprehension.noter(notes, lot[1]["item"], 0, "S")
    reste = comprehension.a_juger(rapport, notes, "S")
    assert len(reste) == 98 and reste == lot[2:]
    assert len(comprehension.a_juger(rapport, notes, "autre système")) == 100
    with pytest.raises(ValueError):
        comprehension.noter(notes, "di-00000", 3, "S")


def test_bilan_avec_intervalle(tmp_path):
    notes = tmp_path / "c.jsonl"
    for i, note in enumerate([2] * 30 + [1] * 10 + [0] * 60):
        comprehension.noter(notes, f"di-{i}", note, "S")
    comprehension.noter(notes, "di-0", 0, "S")              # corrigée : la dernière compte
    b = comprehension.bilan(notes, "S")
    assert b["n"] == 100
    assert b["même sens"]["part"] == pytest.approx(0.29) and b["faux"]["part"] == pytest.approx(0.61)
    lo, hi = b["même sens"]["ic"]
    assert 0.20 < lo < 0.29 < hi < 0.40
    md = comprehension.to_markdown(b)
    assert "même sens : 29 %" in md and "100 phrases" in md
    assert comprehension.to_markdown(comprehension.bilan(notes, "autre")) == "Aucune note pour autre."
    assert comprehension.wilson(0, 10)[0] == 0.0 and comprehension.wilson(10, 10)[1] == 1.0


def test_d_ou_viennent_les_erreurs(tmp_path):
    """Seconde question : la reconnaissance a-t-elle bien entendu ? Elle
    partage les phrases mal comprises entre reconnaissance et traduction."""
    notes = tmp_path / "c.jsonl"
    for i, (sens, entendu) in enumerate([(2, 2)] * 5 + [(0, 0)] * 3 + [(1, 1)] * 2 + [(0, 2)] * 4):
        comprehension.noter(notes, f"di-{i}", sens, "S", entendu)
    comprehension.noter(notes, "di-99", 0, "S")              # sans la seconde réponse
    b = comprehension.bilan(notes, "S")
    assert b["erreurs"] == {"n": 9, "reconnaissance": 3, "les deux": 2, "traduction": 4}
    md = comprehension.to_markdown(b)
    assert "bien entendues, mal traduites (traduction) : 4" in md
    with pytest.raises(ValueError):
        comprehension.noter(notes, "di-0", 2, "S", 5)
    rapport = _rapport(tmp_path / "r.json", [{"item": "di-1", "ref_translation_fr": "bonjour",
                                              "source_fr": "salut", "source_bm": "i ni ce"}])
    assert comprehension.a_juger(rapport, tmp_path / "vide.jsonl", "S")[0]["entendu"] == "i ni ce"


def test_paires_de_ta_voix_jamais_tirees_du_test(tmp_path):
    import paires_maison

    phrases = [p["id"] for p in charger_phrases_a_dire(ROOT / "data" / "atelier" / "a_dire.json")]
    prises, rows = [], []
    for k, pid in enumerate(phrases[:80]):
        for v in range(2):                                   # deux prises par phrase
            item = f"di-{2 * k + v:05d}"
            prises.append({"id": item, "prompt_id": pid})
            rows.append({"item": item, "source_bm": "" if k == 0 else f"bm {item}",
                         "ref_translation_fr": f"fr {pid}"})
    (tmp_path / "dire.jsonl").write_text("\n".join(json.dumps(p) for p in prises))
    rapport = _rapport(tmp_path / "r" / "cascade-1.json", rows)

    paires_maison.main(["--rapport", str(rapport), "--prises", str(tmp_path / "dire.jsonl"),
                        "--out", str(tmp_path / "paires")])
    train = [json.loads(x) for x in (tmp_path / "paires" / "apprentissage" / "train.jsonl").open()]
    choix = [json.loads(x) for x in (tmp_path / "paires" / "choix.jsonl").open()]
    assert train and choix
    assert {usage(x["prompt_id"]) for x in train} == {"apprentissage"}
    assert {usage(x["prompt_id"]) for x in choix} == {"choix"}
    assert not {x["prompt_id"] for x in train} & {x["prompt_id"] for x in choix}
    # La première phrase, où rien n'a été entendu, est écartée.
    assert len(train) == sum(2 for pid in phrases[1:80] if usage(pid) == "apprentissage")
    assert train[0]["bm"].startswith("bm di-") and train[0]["fr"].startswith("fr ")
