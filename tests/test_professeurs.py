"""Distillation : le bambara parlé de RobotsMali (scripts/donnees_robotsmali.py)
traduit par un professeur (scripts/professeurs.py), qui n'est utilisé que
s'il traduit mieux la voix de l'utilisateur que le modèle en service."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import donnees_robotsmali as rm  # noqa: E402


# --- transcriptions ------------------------------------------------------------------

def test_afvoices_ecarte_les_passages_manquants():
    assert rm.nettoyer_afvoices("Tiɲɛ ne don [um]") == "Tiɲɛ ne don"
    assert rm.nettoyer_afvoices("U nana fan da o kɔnɔ [noise] [pause]") == "U nana fan da o kɔnɔ"
    assert rm.nettoyer_afvoices("U ye [cs] [?]  u ye bolo [?]") is None   # français non transcrit
    assert rm.nettoyer_afvoices("A jiginna bɔ[?]") is None                # inaudible
    assert rm.nettoyer_afvoices("[cs ee [?]") is None                     # annotation mal fermée
    assert rm.nettoyer_afvoices("[um]") is None


def test_kunkado_garde_le_francais_sans_souligne():
    assert rm.nettoyer_kunkado("<BRUITS> a ye __problème__ in ye") == "a ye problème in ye"
    assert rm.nettoyer_kunkado("n b'a fɔ ko <INCOMPRÉHENSIBLE>") is None
    assert rm.nettoyer_kunkado("a bɛ taa sug…") is None                   # mot coupé


def test_la_source_de_l_eleve_s_ecrit_comme_soloni():
    # Sans ponctuation (le vocabulaire de Soloni n'en a pas), majuscules et
    # apostrophes gardées.
    assert rm.forme_asr("N ye ji k'o la dɔrɔn, ji kɛ ke di k’o labɔ.") == \
        "N ye ji k'o la dɔrɔn ji kɛ ke di k'o labɔ"
    assert rm.forme_asr("I m'a ye wa!") == "I m'a ye wa"


def test_segments_dedoublonnes_et_trop_courts_ecartes():
    seg = rm.segments(["A bɛ yen.", "a bɛ yen", "Ayiwa.", "N bɛ taa sugu la [?]"],
                      rm.nettoyer_afvoices, "afvoices")
    assert [s["bm"] for s in seg] == ["A bɛ yen."]
    assert seg[0]["bm_asr"] == "A bɛ yen"


def _kunkado_local(dossier: Path, textes: list[str]) -> str:
    """Un Parquet façon Kunkado (audio compris), lu par le même code que sur
    Hugging Face."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dossier.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"audio": [b"\x00" * 1000] * len(textes),
                             "semi-label": ["x"] * len(textes),
                             rm.KUNKADO_COLONNE: textes}), dossier / "train-00000-of-00001.parquet")
    return str(dossier / "train-*.parquet")


def test_corpus_afvoices_et_kunkado(tmp_path, monkeypatch):
    import fsspec

    manifeste = tmp_path / "af.jsonl"
    manifeste.write_text("\n".join(json.dumps({"text": t}) for t in [
        "Ayiwa, an ni ɲan kuma.", "Ne tɔgɔ Awa So.", "U ye [cs] ye.", "Ne tɔgɔ Awa So"]),
        encoding="utf-8")
    motif = _kunkado_local(tmp_path / "kk", ["<MUSIQUE> i ka kɛnɛ wa", "ne tɔgɔ awa so",
                                             "a ye __réseau__ ɲini"])
    lire = rm.lire_kunkado
    monkeypatch.setattr(rm, "lire_kunkado", lambda: lire(fsspec.filesystem("file"), motif))

    comptes = rm.corpus(tmp_path / "out", manifeste)
    lignes = [json.loads(x) for x in (tmp_path / "out" / "bm.jsonl").read_text().splitlines()]
    assert comptes == {"afvoices": 2, "kunkado": 2}       # « ne tɔgɔ awa so » : doublon
    assert {x["bm_asr"] for x in lignes} == {"Ayiwa an ni ɲan kuma", "Ne tɔgɔ Awa So",
                                             "i ka kɛnɛ wa", "a ye réseau ɲini"}
    assert len({x["id"] for x in lignes}) == 4


def test_corpus_sans_kunkado_si_hugging_face_refuse(tmp_path, monkeypatch, capsys):
    manifeste = tmp_path / "af.jsonl"
    manifeste.write_text(json.dumps({"text": "Ne tɔgɔ Awa So."}), encoding="utf-8")

    def refus():
        raise PermissionError("403")
    monkeypatch.setattr(rm, "lire_kunkado", refus)
    assert rm.corpus(tmp_path / "out", manifeste) == {"afvoices": 1}
    assert "AfVoices seulement" in capsys.readouterr().out


def test_filtrer_les_traductions_douteuses(tmp_path):
    paires = [
        {"bm": "I ni sɔgɔma.", "bm_asr": "I ni sɔgɔma", "fr": "Bonjour.", "source": "afvoices"},
        {"bm": "A bɛ taa sugu la", "fr": "", "source": "afvoices"},
        {"bm": "A bɛ taa sugu la", "fr": "Il va il va il va il va au marché", "source": "a"},
        {"bm": "A bɛ taa sugu la", "fr": "A bɛ taa sugu la", "source": "a"},
        {"bm": "A bɛ taa sugu la", "fr": "Il va ɲɛ au marché", "source": "a"},
        {"bm": "A bɛ taa sugu la ka jɛgɛ san", "fr": "Il", "source": "a"},
    ]
    entree = tmp_path / "brutes.jsonl"
    entree.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in paires), encoding="utf-8")
    stats = rm.filtrer(entree, tmp_path / "v4")
    assert stats == {"gardees": 1, "vide": 1, "boucle": 1, "non traduite": 2, "longueur": 1}
    train = [json.loads(x) for x in (tmp_path / "v4" / "train.jsonl").read_text().splitlines()]
    assert train == [{"bm": "I ni sɔgɔma", "fr": "Bonjour.", "source": "afvoices"}]


def test_les_paires_se_lisent_comme_un_jeu_d_entrainement(tmp_path):
    pytest.importorskip("datasets")
    from eval.corpora import load_split

    (tmp_path / "v4").mkdir()
    (tmp_path / "v4" / "train.jsonl").write_text(
        json.dumps({"bm": "I ni sɔgɔma", "fr": "Bonjour.", "source": "afvoices"},
                   ensure_ascii=False) + "\n", encoding="utf-8")
    ds = load_split(str(tmp_path / "v4"), "train")
    assert ds[0]["bm"] == "I ni sɔgɔma" and ds[0]["fr"] == "Bonjour."


# --- professeur ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("transformers")
    pytest.importorskip("sentencepiece")
    import tiny_models
    return tiny_models.nllb(tmp_path_factory.mktemp("prof") / "tiny-nllb")


@pytest.fixture(scope="module")
def prof(tiny):
    import professeurs
    return professeurs.Professeur(str(tiny), "cpu", lot=2)


def test_professeur_traduit_par_lots_dans_l_ordre(prof):
    textes = ["n bɛ taa sugu la", "", "i ni sɔgɔma", "a bɛ yen", "réseau tɛ yen bi"]
    sorties = prof.traduire(textes)
    assert len(sorties) == 5 and sorties[1] == ""
    # Même résultat phrase à phrase : le tri par longueur ne mélange rien.
    assert sorties[3] == prof.traduire(["a bɛ yen"])[0]
    assert not prof.madlad and prof._entrees(["a"]) == ["a"] and prof.tok.src_lang == "bam_Latn"


def test_madlad_choisit_le_francais_par_sa_balise(tiny, tmp_path):
    from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration

    import professeurs
    tok = AutoTokenizer.from_pretrained(tiny)
    T5ForConditionalGeneration(T5Config(
        vocab_size=len(tok), d_model=16, d_kv=8, d_ff=16, num_layers=1, num_heads=2,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        decoder_start_token_id=tok.pad_token_id)).save_pretrained(tmp_path / "t5")
    tok.save_pretrained(tmp_path / "t5")
    madlad = professeurs.Professeur(str(tmp_path / "t5"), "cpu")
    assert madlad.madlad and madlad._entrees(["i ni sɔgɔma"]) == ["<2fr> i ni sɔgɔma"]
    assert len(madlad.traduire(["i ni sɔgɔma", "a bɛ yen"])) == 2


def test_mesurer_refait_seulement_la_traduction(prof, tmp_path):
    pytest.importorskip("sacrebleu")
    import professeurs
    from eval import promotion

    lignes = [{"item": f"di-{i}", "asr_text": "<tag>" if i == 3 else "a bɛ yen",
               "source_bm": "" if i == 3 else ["a bɛ yen", "i ni sɔgɔma"][i % 2],
               "source_fr": "il est là", "ref_translation_fr": "il est là",
               "partition": "dev" if i < 4 else "test", "speaker": "spk-00"} for i in range(8)]
    rapport = tmp_path / "service" / "cascade-20261017-100000.json"
    rapport.parent.mkdir()
    rapport.write_text(json.dumps({"nom": "Soloni + NLLB v1", "architecture": "cascade",
                                   "testset": "maison.jsonl",
                                   "modeles": {"asr": "nemo:soloni", "mt_in": "transformers:v1"}}),
                       encoding="utf-8")
    rapport.with_suffix(".details.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in lignes), encoding="utf-8")

    nouveau = professeurs.mesurer(rapport, prof, "NLLB 3.3B", tmp_path / "prof")
    report = json.loads(nouveau.read_text())
    assert report["nom"] == "NLLB 3.3B" and report["modeles"]["asr"] == "nemo:soloni"
    assert report["modeles"]["mt_in"] == f"professeur:{prof.modele}"
    details = [json.loads(x) for x in nouveau.with_suffix(".details.jsonl").read_text().splitlines()]
    attendu = dict(zip(["a bɛ yen", "i ni sɔgɔma"], prof.traduire(["a bɛ yen", "i ni sɔgɔma"])))
    for avant, apres in zip(lignes, details):
        assert {k: v for k, v in apres.items() if k != "source_fr"} == \
            {k: v for k, v in avant.items() if k != "source_fr"}
        assert apres["source_fr"] == attendu.get(avant["source_bm"], "")
    # Comparable au modèle en service par la règle de promotion.
    c = promotion.comparer(nouveau, rapport, "dev", n_boot=50)
    assert c["n"] == 4 and c["titulaire"] == pytest.approx(100.0)


def test_traduire_reprend_apres_une_coupure(prof, tmp_path):
    import professeurs

    entree = tmp_path / "bm.jsonl"
    entree.write_text("\n".join(json.dumps({"id": f"afvoices-{i}", "bm": b, "bm_asr": b,
                                            "source": "afvoices"}, ensure_ascii=False)
                                for i, b in enumerate(["a bɛ yen", "i ni sɔgɔma",
                                                       "n bɛ taa sugu la", "réseau tɛ yen bi"])),
                      encoding="utf-8")
    out = tmp_path / "drive" / "brutes.jsonl"
    assert professeurs.traduire(entree, prof, out, maximum=2) == 2
    assert professeurs.traduire(entree, prof, out, morceau=1) == 4    # reprise
    assert professeurs.traduire(entree, prof, out) == 4               # rien à refaire
    lignes = [json.loads(x) for x in out.read_text().splitlines()]
    assert [x["id"] for x in lignes] == [f"afvoices-{i}" for i in range(4)]
    assert all(x["professeur"] == prof.modele and "fr" in x for x in lignes)

    # Colab coupé au milieu d'une écriture : la ligne tronquée est refaite.
    out.write_text(out.read_text()[:-25])
    assert professeurs.traduire(entree, prof, out) == 4
    assert [json.loads(x)["id"] for x in out.read_text().splitlines()] == \
        [f"afvoices-{i}" for i in range(4)]
