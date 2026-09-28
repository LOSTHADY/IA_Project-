"""Tests de la collecte du jeu de test (eval/collection.py), sans interface.

Les écritures audio demandent `soundfile` ; le rééchantillonnage, `librosa`.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.collection import (
    Collection, Recording, Speaker, check_audio, export_testset, load_prompts,
    next_prompt, progress, read_candidates, to_mono_16k, _next_id,
)
from eval.dataset import load_testset

sf = pytest.importorskip("soundfile")

ROOT = Path(__file__).resolve().parent.parent
PROMPT = {"id": "salut-matin", "theme": "salutation", "consigne": "Saluez."}


def _tone(seconds=3.0, sr=16_000, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _speaker(col, gender="f", **kw):
    return col.add_speaker(gender=gender, consent=True, **kw)


# --- identifiants et audio ----------------------------------------------------

def test_next_id_suit_le_maximum_pas_le_nombre_de_lignes():
    assert _next_id([], "bv-", 4) == "bv-0001"
    # bv-0002 supprimé : le suivant ne doit pas réutiliser bv-0003.
    assert _next_id(["bv-0001", "bv-0003"], "bv-", 4) == "bv-0004"


def test_to_mono_16k_convertit_entiers_stereo_et_taux():
    stereo = (np.stack([_tone(1.0, 48_000)] * 2, axis=1) * 32767).astype(np.int16)
    wav = to_mono_16k(stereo, 48_000)
    assert wav.dtype == np.float32 and wav.ndim == 1
    assert abs(len(wav) - 16_000) <= 1
    assert 0.25 < np.abs(wav).max() < 0.35


def test_check_audio():
    assert check_audio(_tone()) == []
    assert any("court" in p for p in check_audio(_tone(0.5)))
    assert any("plage" in p for p in check_audio(_tone(20.0)))
    assert any("saturation" in p for p in check_audio(np.clip(_tone(amp=3.0), -1, 1)))
    assert any("faible" in p for p in check_audio(_tone(amp=0.001)))
    assert check_audio(np.zeros(0, dtype=np.float32)) == ["enregistrement vide"]


# --- consentement et enregistrement ------------------------------------------

def test_pas_de_locuteur_sans_consentement(tmp_path):
    with pytest.raises(ValueError, match="consentement"):
        Collection(tmp_path).add_speaker(gender="f", consent=False)


def test_enregistrement_ecrit_un_wav_16k_mono(tmp_path):
    col = Collection(tmp_path)
    spk = _speaker(col)
    rec, problems = col.add_recording(spk.id, _tone(3.0, 44_100), 44_100,
                                      "spontane", PROMPT, "salle calme")
    assert problems == []
    assert rec.id == "bv-0001" and rec.status == "a_transcrire"
    info = sf.info(str(tmp_path / rec.audio))
    assert (info.samplerate, info.channels, info.subtype) == (16_000, 1, "PCM_16")
    assert rec.duration_s == pytest.approx(3.0, abs=0.01)


def test_enonce_lu_est_pre_transcrit(tmp_path):
    col = Collection(tmp_path)
    spk = _speaker(col)
    prompt = {"id": "lu:n bɛ taa", "theme": "lecture", "consigne": "n bɛ taa"}
    rec, _ = col.add_recording(spk.id, _tone(), 16_000, "lu", prompt)
    assert rec.status == "transcrit" and rec.transcript_bm == "n bɛ taa"


def test_suppression_retire_ligne_et_fichier(tmp_path):
    col = Collection(tmp_path)
    spk = _speaker(col)
    rec, _ = col.add_recording(spk.id, _tone(), 16_000, "spontane", PROMPT)
    col.delete_recording(rec.id)
    assert col.recordings() == []
    assert not (tmp_path / rec.audio).exists()


# --- transcription et validation ---------------------------------------------

def test_validation_exige_une_seconde_personne(tmp_path):
    col = Collection(tmp_path)
    spk = _speaker(col)
    rec, _ = col.add_recording(spk.id, _tone(), 16_000, "spontane", PROMPT)

    with pytest.raises(ValueError, match="transcrire d'abord"):
        col.update_recording(rec.id, status="valide", validator="AK")

    col.update_recording(rec.id, transcript_bm="i ni sɔgɔma", status="transcrit",
                         transcriber="AK")
    with pytest.raises(ValueError, match="autre personne"):
        col.update_recording(rec.id, status="valide", validator="AK")

    done = col.update_recording(rec.id, status="valide", validator="MD")
    assert done.status == "valide" and done.transcriber == "AK"


def test_transcription_normalisee(tmp_path):
    col = Collection(tmp_path)
    spk = _speaker(col)
    rec, _ = col.add_recording(spk.id, _tone(), 16_000, "spontane", PROMPT)
    out = col.update_recording(rec.id, transcript_bm="  n’bɛ   taa ",
                               status="transcrit", transcriber="AK")
    assert out.transcript_bm == "n'bɛ taa"


# --- choix des consignes ------------------------------------------------------

def test_consignes_du_depot_sont_valides():
    prompts = load_prompts(ROOT / "data/testset/consignes.json")
    ids = [p["id"] for p in prompts]
    assert len(ids) == len(set(ids)) >= 30
    assert all(p["consigne"] and p["theme"] for p in prompts)


def test_next_prompt_evite_les_repetitions_et_equilibre():
    prompts = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    recs = [Recording(id="bv-0001", audio="", speaker="spk-01", register="spontane",
                      prompt_id="a"),
            Recording(id="bv-0002", audio="", speaker="spk-02", register="spontane",
                      prompt_id="b")]
    # spk-01 a fait "a" ; entre "b" (déjà faite une fois) et "c", on prend "c".
    assert next_prompt(prompts, recs, "spk-01")["id"] == "c"
    assert next_prompt([{"id": "a"}], recs, "spk-01") is None


def test_phrases_lues_viennent_des_spontanes_valides_des_autres():
    def rec(i, spk, status, text):
        return Recording(id=i, audio="", speaker=spk, register="spontane",
                         status=status, transcript_bm=text)

    recs = [rec("bv-0001", "spk-01", "valide", "i ni sɔgɔma"),
            rec("bv-0002", "spk-01", "transcrit", "pas encore relu"),
            rec("bv-0003", "spk-02", "valide", "n bɛ taa sugu la")]
    phrases = [c["consigne"] for c in read_candidates(recs, "spk-02")]
    assert phrases == ["i ni sɔgɔma"]


# --- suivi et export ----------------------------------------------------------

def test_progress_signale_les_desequilibres():
    speakers = [Speaker(id="spk-01", gender="f", consent=True)]
    recs = [Recording(id=f"bv-{i:04d}", audio="", speaker="spk-01", register="lu",
                      duration_s=3.0, conditions="salle calme") for i in range(5)]
    rows = {r["critere"]: r for r in progress(speakers, recs)}
    assert not rows["énoncés enregistrés"]["ok"]
    assert not rows["part du locuteur le plus présent"]["ok"]
    assert not rows["part de spontané"]["ok"]
    assert rows["durées entre 2 et 15 s"]["ok"]


def test_progress_collecte_vide():
    rows = progress([], [])
    assert [r["critere"] for r in rows][:2] == ["énoncés enregistrés", "énoncés validés"]


def test_export_relisible_par_load_testset(tmp_path):
    col = Collection(tmp_path)
    public = _speaker(col, "f", region="Bamako", public_release=True)
    private = _speaker(col, "m", region="Ségou")
    for spk in (public, private):
        rec, _ = col.add_recording(spk.id, _tone(), 16_000, "spontane", PROMPT)
        col.update_recording(rec.id, transcript_bm="réseau tɛ yen",
                             translation_fr="Il n'y a pas de réseau.",
                             code_switching=True, status="transcrit", transcriber="AK")
        col.update_recording(rec.id, status="valide", validator="MD")
    col.add_recording(public.id, _tone(), 16_000, "spontane", PROMPT)  # non transcrit

    out = tmp_path / "testset.jsonl"
    assert export_testset(col, out) == 2
    items = load_testset(out)
    assert {i.region for i in items} == {"Bamako", "Ségou"}
    assert all(i.audio_path(tmp_path).exists() for i in items)
    assert items[0].extra["conditions"] == ""

    assert export_testset(col, out, public_only=True) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["speaker"] == public.id
