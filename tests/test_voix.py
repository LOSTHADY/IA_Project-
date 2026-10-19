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
