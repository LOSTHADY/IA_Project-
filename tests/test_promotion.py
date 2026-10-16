"""La règle de promotion (eval/promotion.py) : un modèle ne remplace le
précédent que s'il est meilleur, mesure à l'appui."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("sacrebleu")

from eval import promotion  # noqa: E402

PHRASES = [f"il va au marché acheter {k} poissons ce matin" for k in range(40)]


def _maison(path: Path, juste: dict[str, float], nom: str = "") -> Path:
    """Rapport eval.run_eval sur un jeu maison de 80 énoncés (40 dev, 40
    test) : `juste[partition]` est la part d'énoncés bien compris."""
    rows = []
    for i in range(80):
        part = "dev" if i < 40 else "test"
        ref = PHRASES[i % 40]
        ok = (i % 40) < juste[part] * 40
        rows.append({"item": f"di-{i:05d}", "source_fr": ref if ok else "il est parti hier",
                     "ref_translation_fr": ref, "partition": part,
                     "speaker": f"spk-0{i % 2}"})
    path.write_text(json.dumps({"nom": nom, "architecture": "cascade"}), encoding="utf-8")
    path.with_suffix(".details.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def _mt(path: Path, juste: float) -> Path:
    """Rapport eval.baselines mt (contrôle écrit), 60 phrases."""
    rows = [{"id": i, "ref_bm": "a bɛ taa sugu la", "hyp_bm": "a bɛ taa sugu la",
             "ref_fr": PHRASES[i % 40],
             "hyp_fr": PHRASES[i % 40] if i < juste * 60 else "rien"} for i in range(60)]
    path.write_text(json.dumps({"nom": path.stem, "maillon": "mt"}), encoding="utf-8")
    path.with_suffix(".details.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def test_promu_seulement_si_meilleur_sur_le_test(tmp_path):
    titulaire = _maison(tmp_path / "t.json", {"dev": 0.5, "test": 0.5})
    meilleur = _maison(tmp_path / "m.json", {"dev": 0.8, "test": 0.8})
    egal = _maison(tmp_path / "e.json", {"dev": 0.5, "test": 0.5})
    chanceux = _maison(tmp_path / "c.json", {"dev": 0.9, "test": 0.45})  # dev seulement

    assert promotion.choisir({"t": titulaire, "m": meilleur, "c": chanceux}) == "c"
    assert promotion.score(meilleur, "test") > promotion.score(titulaire, "test")

    ok = promotion.verdict(promotion.comparer(meilleur, titulaire, n_boot=300))
    assert ok["promu"] and ok["comparaison"]["n"] == 40 and ok["comparaison"]["ic"][0] > 0
    assert not promotion.verdict(promotion.comparer(egal, titulaire, n_boot=300))["promu"]
    refus = promotion.verdict(promotion.comparer(chanceux, titulaire, n_boot=300))
    assert not refus["promu"] and "pas meilleur" in refus["raisons"][0]
    assert "**Refusé : titulaire reste en service.**" in promotion.to_markdown(refus)


def test_une_regression_sur_un_test_ecrit_bloque(tmp_path):
    titulaire = _maison(tmp_path / "t.json", {"dev": 0.5, "test": 0.5})
    meilleur = _maison(tmp_path / "m.json", {"dev": 0.8, "test": 0.8})
    bayel_t, bayel_pire = _mt(tmp_path / "bt.json", 0.8), _mt(tmp_path / "bc.json", 0.4)
    bayel_egal = _mt(tmp_path / "be.json", 0.8)

    casse = promotion.controle(bayel_pire, bayel_t, "chrF++ bm→fr", n_boot=300)
    assert casse["regression"] and casse["ecart"] < -promotion.TOLERANCE
    assert not promotion.controle(bayel_egal, bayel_t, "chrF++ bm→fr", n_boot=300)["regression"]

    d = promotion.verdict(promotion.comparer(meilleur, titulaire, n_boot=300), [casse])
    assert not d["promu"] and "régression sur chrF++ bm→fr" in d["raisons"][0]
    with pytest.raises(ValueError, match="absente"):
        promotion.controle(bayel_pire, bayel_t, "WER strict")


def test_jugements_humains_pour_le_francais_vers_bambara():
    res = {"preferences": {"v1": 3, "v2": 15}, "p_signe": 0.008, "jugements": 25}
    assert promotion.verdict_jugements(res, "v2", "v1")["promu"]
    assert not promotion.verdict_jugements(res, "v1", "v2")["promu"]
    peu = promotion.verdict_jugements({"preferences": {"v2": 3, "v1": 1}, "p_signe": 0.62},
                                      "v2", "v1")
    assert not peu["promu"] and "pas assez de jugements" in peu["raisons"][0]


def test_retenu_garde_l_assemblage_et_l_historique(tmp_path):
    path = tmp_path / "retenu.json"
    defaut = {"asr": {"model_id": "whisper"}, "bm2fr": "v1", "fr2bm": "v1"}
    r = promotion.lire_retenu(path, defaut)
    assert r == {**defaut, "historique": []}
    refus = {"promu": False, "raisons": ["pas meilleur"], "comparaison": {"ecart": 0.2}}
    r = promotion.ecrire_retenu(path, r, "bm2fr", {"bm2fr": "v2"}, refus)
    assert r["bm2fr"] == "v1" and r["historique"][0]["promu"] is False
    r = promotion.ecrire_retenu(path, r, "bm2fr", {"bm2fr": "v2"}, refus)  # relance
    assert len(r["historique"]) == 1
    ok = {"promu": True, "raisons": [], "comparaison": {"ecart": 3.0}}
    r = promotion.ecrire_retenu(path, r, "bm2fr", {"bm2fr": "v3"}, ok)
    lu = promotion.lire_retenu(path, defaut)
    assert lu["bm2fr"] == "v3" and lu["fr2bm"] == "v1" and len(lu["historique"]) == 2
    assert lu["historique"][1]["preuves"]["comparaison"]["ecart"] == 3.0


def test_cli(tmp_path, capsys):
    titulaire = _maison(tmp_path / "t.json", {"dev": 0.5, "test": 0.5})
    meilleur = _maison(tmp_path / "m.json", {"dev": 0.8, "test": 0.8})
    sortie = tmp_path / "decision.json"
    promotion.main(["--candidat", str(meilleur), "--titulaire", str(titulaire),
                    "--controle", str(_mt(tmp_path / "bc.json", 0.8)),
                    str(_mt(tmp_path / "bt.json", 0.8)), "chrF++ bm→fr",
                    "--n-boot", "200", "--sortie", str(sortie)])
    out = capsys.readouterr().out
    assert "Test maison (40 énoncés, tirage par enonce)" in out and "**Promu" in out
    assert json.loads(sortie.read_text())["promu"]
