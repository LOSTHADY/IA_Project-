"""L'étage de la voix (scripts/affiner_soloni.py) : ce qui sert à affiner
Soloni, et comment c'est réparti. L'entraînement lui-même a été essayé avec
NeMo 3.0 sur un Soloni miniature (hybride, deux décodeurs) ; il demande NeMo
et n'est pas rejoué ici."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import affiner_soloni as af  # noqa: E402


def _atelier(tmp_path: Path, n: int = 200) -> Path:
    racine = tmp_path / "atelier"
    racine.mkdir()
    lire = [{"id": f"li-{k:05d}", "audio": f"audio/li-{k:05d}.wav", "speaker": "spk-01",
             "prompt_id": f"lu-{k:04d}", "texte": f"N bɛ taa sugu la, {k} !", "duration_s": 2.0}
            for k in range(n)]
    conv = [{"id": "co-00001", "audio": "audio/co-00001.wav", "speaker": "spk-02",
             "entendu_statut": "corrige", "entendu_std": "i ni sɔgɔma", "duration_s": 1.5},
            {"id": "co-00002", "audio": "audio/co-00002.wav", "speaker": "spk-02",
             "entendu_statut": "inaudible", "entendu_std": "", "duration_s": 1.0}]
    (racine / "lire.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lire))
    (racine / "conversations.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False)
                                                         for x in conv))
    return racine


def test_le_texte_s_ecrit_comme_soloni_l_ecrit():
    assert af.texte_asr("N bɛ taa sugu la, sisan !") == "N bɛ taa sugu la sisan"
    assert af.texte_asr("k’a fɔ") == "k'a fɔ"


def test_lectures_et_tours_corriges_repartis_une_fois_pour_toutes(tmp_path):
    ex = af.exemples(_atelier(tmp_path))
    assert len(ex) == 201                                   # le tour inaudible n'a pas de texte
    assert {e["source"] for e in ex} == {"lecture", "conversation"}
    parts = [e["partie"] for e in ex]
    assert 0.6 < parts.count("apprentissage") / len(ex) < 0.8
    assert parts.count("choix") > 10 and parts.count("test") > 10
    assert af.partie("lu:lu-0007") == af.partie("lu:lu-0007")   # stable
    assert ex[0]["audio_filepath"].endswith("atelier/audio/li-00000.wav")


def test_manifestes_et_caracteres_inconnus(tmp_path):
    class Tok:
        unk_id = 0

        def text_to_ids(self, text):
            return [0] if "ɔ" in text else [5, 6]

    racine = _atelier(tmp_path)
    comptes = af.manifestes(racine, tmp_path / "out", Tok())
    assert comptes["ecartes"] == 1                          # « i ni sɔgɔma » : ɔ inconnu
    assert sum(comptes[k] for k in ("apprentissage", "choix", "test")) == 200
    ligne = json.loads((tmp_path / "out" / "manifest-test.json").read_text().splitlines()[0])
    assert set(ligne) == {"audio_filepath", "duration", "text"} and "," not in ligne["text"]


def test_voix_nouvelles_jamais_apprises_et_part_de_la_courbe(tmp_path):
    """Un contributeur sur huit (spk-08…) n'est jamais appris : ses lectures
    forment le test des voix nouvelles. `part` ne garde qu'une fraction de
    l'apprentissage, toujours la même."""
    racine = _atelier(tmp_path)
    lignes = (racine / "lire.jsonl").read_text().splitlines()
    autres = [json.dumps({**json.loads(x), "speaker": "spk-08", "id": f"r{k}",
                          "audio": f"audio/spk-08/r{k}.wav"}) for k, x in enumerate(lignes[:30])]
    (racine / "lire.jsonl").write_text("\n".join(lignes + autres))
    ex = af.exemples(racine)
    assert {e["partie"] for e in ex if e["speaker"] == "spk-08"} == {"voix_nouvelles"}
    assert "voix_nouvelles" not in {e["partie"] for e in ex if e["speaker"] != "spk-08"}

    tout = af.manifestes(racine, tmp_path / "tout")
    assert tout["voix_nouvelles"] == 30
    assert tout["heures"] == round(sum(e["duration"] for e in ex
                                       if e["partie"] == "apprentissage") / 3600, 2) > 0
    moitie = af.manifestes(racine, tmp_path / "moitie", part=0.5)
    assert 0.3 < moitie["apprentissage"] / tout["apprentissage"] < 0.7
    # Les parties jugées ne bougent pas, et la moitié est toujours la même.
    assert {k: moitie[k] for k in ("choix", "test", "voix_nouvelles")} == \
        {k: tout[k] for k in ("choix", "test", "voix_nouvelles")}
    assert af.manifestes(racine, tmp_path / "bis", part=0.5) == moitie
    assert (tmp_path / "bis" / "manifest-apprentissage.json").read_text() == \
        (tmp_path / "moitie" / "manifest-apprentissage.json").read_text()


def test_lectures_suspectes_ecartees(tmp_path, monkeypatch):
    """Soloni d'origine entend tout autre chose que le texte : la lecture est
    écartée de l'apprentissage."""
    import pytest
    pytest.importorskip("jiwer")
    manifeste = tmp_path / "m.json"
    manifeste.write_text("".join(json.dumps({"audio_filepath": a, "duration": 1.0, "text": t}) + "\n"
                                 for a, t in (("a.wav", "n bɛ taa sugu la"),
                                              ("b.wav", "a ka kɛnɛ"),
                                              ("c.wav", "i ni ce"))))
    entendu = {"a.wav": "n bɛ ta sugu la", "b.wav": "", "c.wav": "furaw ka di kosɛbɛ"}
    monkeypatch.setattr(af, "transcrire",
                        lambda model, chemins, batch_size=8: [entendu[c] for c in chemins])
    assert af.filtrer_suspectes(None, manifeste, 0.7) == 2
    assert [json.loads(x)["audio_filepath"] for x in manifeste.read_text().splitlines()] == ["a.wav"]
    assert af.filtrer_suspectes(None, manifeste, 0) == 0


def test_courbe_d_apprentissage():
    import pytest
    from eval.courbe import ajuster, predire, to_markdown

    # 4 points de WER en moins à chaque doublement des heures.
    points = [(5, 40.0), (10, 36.0), (20, 32.0)]
    a, b = ajuster(points)
    assert b == pytest.approx(-4 / 0.693147, rel=1e-4)
    assert predire(a, b, 40) == pytest.approx(28.0, abs=1e-6)
    texte = to_markdown(points, cibles=(80,))
    assert "| 20.0 | 32.0 % |" in texte and "+4.0 points" in texte and "à 80 h : WER ≈ 24 %" in texte
    assert "courbe est plate" in to_markdown([(5, 30.0), (10, 30.0), (20, 30.5)])
    with pytest.raises(ValueError):
        ajuster([(10, 30.0)])
    with pytest.raises(ValueError):
        ajuster([(10, 30.0), (10, 28.0)])
