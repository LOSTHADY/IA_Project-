"""Le mode entraînement de l'atelier (onglet « Discuter ») : on parle
librement, on choisit ou corrige ce que l'assistant a entendu et compris ;
il le retient, et chaque tour devient une donnée d'entraînement."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("soundfile")

from bambara_voice.orthographe import Correcteur, cle  # noqa: E402
from eval.atelier import Atelier, charger_phrases_a_dire  # noqa: E402


# --- orthographe ----------------------------------------------------------------------

def test_le_bambara_ecrit_a_la_francaise_retrouve_l_orthographe_standard():
    c = Correcteur()
    assert cle("tchè") == cle("cɛ") and cle("gnogon") == cle("ɲɔgɔn") and cle("ouari") == "wari"
    r = c.corriger("tchè nin bé baara kè")
    assert r.texte == "cɛ nin bɛ baara kɛ"
    assert ("tchè", "cɛ") in r.changements and ("baara", "baara") not in r.changements
    assert c.corriger("gnogon").texte == "ɲɔgɔn"
    assert c.corriger("ouari tè n bolo").texte == "wari tɛ n bolo"
    assert c.corriger("i ka kéné wa").texte == "i ka kɛnɛ wa"


def test_ce_que_la_reconnaissance_a_entendu_passe_en_premier():
    # « ta » existe dans le vocabulaire, mais la reconnaissance a entendu « taa ».
    c = Correcteur({"ta": 100, "taa": 50, "sugu": 10, "la": 10, "bɛ": 10})
    assert c.corriger("a bè ta sougou la").texte == "a bɛ ta sugu la"
    assert c.corriger("a bè taa sougou la", "A bɛ taa sugu la").texte == "a bɛ taa sugu la"
    # Un mot inconnu (nom propre) reste tel qu'écrit, sans accents du français.
    assert c.corriger("Séga bè taa").texte == "sega bɛ taa"


# --- l'atelier ------------------------------------------------------------------------

def _voix(secondes=1.0):
    t = np.arange(int(16_000 * secondes)) / 16_000
    return 16_000, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class _ASR:
    def __init__(self, propositions):
        self.propositions = propositions

    def alternatives(self, audio):
        return list(self.propositions)


class _MT:
    def __init__(self, sortie="Bonjour, comment vas-tu ?"):
        self.sortie, self.calls = sortie, []

    def translate(self, text):
        self.calls.append(text)
        return self.sortie


class _LLM:
    def reply(self, text, history=None):
        return "Je vais bien, merci."


class _TTS:
    def synthesize(self, text):
        from bambara_voice.tts import SpeechResult
        return SpeechResult(np.zeros(1600, np.float32), 16_000)


def _pipeline(tmp_path, propositions):
    from bambara_voice.config import MemoireConfig, PipelineConfig
    from bambara_voice.pipeline import VoicePipeline

    pipe = VoicePipeline(PipelineConfig(memoire=MemoireConfig(path=str(tmp_path / "memoire.jsonl"))))
    pipe.asr, pipe.mt_in, pipe.mt_out = _ASR(propositions), _MT(), _MT("N bɛ kɛnɛ, i ni ce.")
    pipe.llm, pipe.tts = _LLM(), _TTS()
    return pipe


def _fonctions(atelier, pipeline):
    pytest.importorskip("gradio")
    from app.atelier_app import build_interface

    demo = build_interface(atelier, charger_phrases_a_dire(), pipeline)
    return {f.fn.__name__: f.fn for f in demo.fns.values()}


def test_un_tour_corrige_est_enregistre_retenu_et_repondu(tmp_path):
    atelier = Atelier(tmp_path / "atelier")
    spk = atelier.ajouter_locuteur("f", consent=True).id
    pipe = _pipeline(tmp_path, ["I ni sɔgɔma i ka kɛnɛ", "I ni sɔgɔma i ka kɛnɛ wa"])
    f = _fonctions(atelier, pipe)

    assert "qui tu es" in f["ecouter"](None, _voix())[2]
    radio, entendu, etape, etat = f["ecouter"](spk, _voix())
    assert entendu == "I ni sɔgɔma i ka kɛnɛ" and "Ce que j'ai entendu" in etape
    assert f["choisir_proposition"]("I ni sɔgɔma i ka kɛnɛ wa") == "I ni sɔgɔma i ka kɛnɛ wa"

    # Deux mots corrigés, écrits à la française.
    etape, compris, etat = f["valider_entendu"](etat, "i ni sogoma, i ka kéné wa")
    assert "**i ni sɔgɔma i ka kɛnɛ wa**" in etape and etat["entendu_statut"] == "juste"
    assert compris == "Bonjour, comment vas-tu ?"
    assert pipe.mt_in.calls == ["i ni sɔgɔma i ka kɛnɛ wa"]

    msg, audio, micro, etat, vide = f["valider_compris"](spk, "salle calme", etat,
                                                         "Bonjour, tu vas bien ?")
    # Une salutation : la réponse vient d'un gabarit validé.
    assert msg.startswith("✅ Retenu (co-00001)") and "**3. Ma réponse** : Nba, i ni ce." in msg
    assert audio[0] == 16_000 and micro is None and vide == ""
    [tour] = atelier.tours()
    assert tour.entendu_auto == ["I ni sɔgɔma i ka kɛnɛ", "I ni sɔgɔma i ka kɛnɛ wa"]
    assert tour.entendu == "i ni sogoma, i ka kéné wa"
    assert tour.entendu_std == "i ni sɔgɔma i ka kɛnɛ wa" and tour.compris_statut == "corrige"
    assert (atelier.root / tour.audio).exists()
    # Retenu : la même phrase, entendue comme la première fois, est comprise.
    assert pipe.memoire.chercher("I ni sɔgɔma i ka kɛnɛ")[0].fr == "Bonjour, tu vas bien ?"

    assert f["reponse_juste"](spk, etat)[0].startswith("Merci")
    reponses = [json.loads(x) for x in (atelier.root / "reponses.jsonl").read_text().splitlines()]
    assert reponses[0]["tour"] == "co-00001" and reponses[0]["juste"] is True

    # Export : audio et bambara standard pour la reconnaissance, avec le sens.
    [ligne] = [json.loads(x) for x in atelier.exporter()["conversations"].read_text().splitlines()]
    assert ligne["text"] == "i ni sɔgɔma i ka kɛnɛ wa" and ligne["fr"] == "Bonjour, tu vas bien ?"
    assert atelier.bilan([])["tours"] == {"total": 1, "entendu_corrige": 0, "compris_corrige": 1}

    # Entraînement (notebook progres) : ce que Soloni a entendu, et le bambara
    # corrigé, chacun vers le sens voulu.
    sys.path.insert(0, str(ROOT / "scripts"))
    import paires_maison
    paires = paires_maison.paires_conversations(atelier.root / "conversations.jsonl")
    assert [(x["bm"], x["fr"]) for x in paires] == [
        ("I ni sɔgɔma i ka kɛnɛ", "Bonjour, tu vas bien ?"),
        ("i ni sɔgɔma i ka kɛnɛ wa", "Bonjour, tu vas bien ?")]


def test_un_tour_inaudible_garde_le_sens_mais_pas_le_bambara(tmp_path):
    atelier = Atelier(tmp_path / "atelier")
    spk = atelier.ajouter_locuteur("m", consent=True).id
    pipe = _pipeline(tmp_path, ["a bɛ yen"])
    f = _fonctions(atelier, pipe)
    *_, etat = f["ecouter"](spk, _voix())
    etape, compris, etat = f["valider_entendu"](etat, "", inaudible=True)
    assert "Pas de bambara retenu" in etape and etat["entendu_statut"] == "inaudible"
    f["valider_compris"](spk, "", etat, "Il est là.")
    [tour] = atelier.tours()
    assert tour.entendu_statut == "inaudible" and tour.entendu_std == ""
    assert atelier.exporter()["conversations"].read_text() == ""   # rien pour la reconnaissance
    with pytest.raises(ValueError):
        atelier.enregistrer_tour(spk, _voix()[1], 16_000, ["x"], "", "", "faux", "", "y", "juste")


def test_sans_modeles_pas_d_onglet_discuter(tmp_path):
    f = _fonctions(Atelier(tmp_path / "atelier"), None)
    assert "ecouter" not in f
