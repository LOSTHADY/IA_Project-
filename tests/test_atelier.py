"""Atelier de données (eval/atelier.py) et son application (app/atelier_app.py)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("soundfile")

from eval.atelier import (  # noqa: E402
    Atelier, _ecrire_lecture, charger_phrases_a_dire, p_signe, preparer_lot, rubriques,
    texte_fr, trouver_rapport,
)
from eval.dataset import describe, load_testset  # noqa: E402

PHRASES = charger_phrases_a_dire()


def _voix(seconds=2.0, sr=44_100):
    """Micro de navigateur : entiers 16 bits, stéréo, 44,1 kHz."""
    t = np.arange(int(seconds * sr)) / sr
    mono = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    return sr, np.stack([mono, mono], axis=1)


@pytest.fixture
def atelier(tmp_path):
    a = Atelier(tmp_path / "atelier")
    a.ajouter_locuteur("m", consent=True, region="Bamako")
    return a


def test_phrases_a_dire():
    assert len(PHRASES) == 300 and len({p["id"] for p in PHRASES}) == 300
    assert len(rubriques(PHRASES)) == 12
    assert {sum(p["rubrique"] == r for p in PHRASES) for r, _ in rubriques(PHRASES)} == {25}
    assert PHRASES[0] == {"id": "sal-01", "rubrique": "sal", "titre": "Salutations et politesse",
                          "fr": "Bonjour, comment vas-tu ce matin ?",
                          "vous": "Bonjour, comment allez-vous ce matin ?"}
    # Une phrase qui s'adresse à quelqu'un a sa forme vouvoyée, et elle seule.
    assert sum("vous" in p for p in PHRASES) == 70 and "vous" not in PHRASES[3]
    assert texte_fr(PHRASES[0], "ami") == PHRASES[0]["fr"]
    assert texte_fr(PHRASES[0], "aine") == texte_fr(PHRASES[0], "inconnu") == PHRASES[0]["vous"]
    assert texte_fr(PHRASES[3], "aine") == PHRASES[3]["fr"]


def test_pas_de_contributeur_sans_consentement(tmp_path):
    with pytest.raises(ValueError, match="consentement"):
        Atelier(tmp_path).ajouter_locuteur("f", consent=False)


def test_dire_enregistrer_passer_annuler_et_reprendre(atelier):
    sr, data = _voix()
    p1 = atelier.prochaine("dire", "spk-01", PHRASES)
    prise, problemes = atelier.enregistrer("dire", "spk-01", p1, data, sr,
                                           code_switching=True, argot=" wai ")
    assert problemes == [] and prise.id == "di-00001" and prise.texte == p1["fr"]
    assert prise.code_switching and prise.argot == "wai" and prise.duration_s == 2.0
    import soundfile as sf
    wav, wsr = sf.read(atelier.root / prise.audio)
    assert wsr == 16_000 and wav.ndim == 1

    # La suivante, puis on passe la troisième : la reprise saute les deux.
    p2 = atelier.prochaine("dire", "spk-01", PHRASES)
    assert p2["id"] == "sal-02"
    atelier.enregistrer("dire", "spk-01", p2, data, sr)
    atelier.passer("dire", "spk-01", "sal-03")
    assert atelier.prochaine("dire", "spk-01", PHRASES)["id"] == "sal-04"
    assert atelier.prochaine("dire", "spk-01", PHRASES, rubrique="san")["id"] == "san-01"
    prog = {r["rubrique"]: r for r in atelier.progression("dire", "spk-01", PHRASES)}
    assert prog["sal"]["faites"] == 3 and prog["sal"]["total"] == 25 and prog["san"]["faites"] == 0

    # Annuler : la dernière prise disparaît (ligne et fichier), sa phrase revient.
    annulee = atelier.annuler_derniere("dire", "spk-01")
    assert annulee.prompt_id == "sal-02" and not (atelier.root / annulee.audio).exists()
    assert atelier.prochaine("dire", "spk-01", PHRASES)["id"] == "sal-02"
    # Un autre contributeur commence au début.
    atelier.ajouter_locuteur("f", consent=True)
    assert atelier.prochaine("dire", "spk-02", PHRASES)["id"] == "sal-01"


def test_dire_a_plusieurs_personnes(atelier):
    """Une phrase se dit à un ami, à un inconnu, à une personne âgée, et
    plusieurs fois à la même personne : chaque version est une prise, et la
    progression se compte par interlocuteur."""
    sr, data = _voix()
    sal01 = PHRASES[0]
    ami, _ = atelier.enregistrer("dire", "spk-01", sal01, data, sr)  # « ami » par défaut
    assert ami.interlocuteur == "ami" and ami.texte == sal01["fr"]
    aine, _ = atelier.enregistrer("dire", "spk-01", sal01, data, sr, interlocuteur="aine")
    assert aine.texte == "Bonjour, comment allez-vous ce matin ?"
    atelier.enregistrer("dire", "spk-01", sal01, data, sr, interlocuteur="aine")  # autre façon
    assert atelier.versions("spk-01", "sal-01") == {"ami": 1, "aine": 2}
    with pytest.raises(ValueError, match="interlocuteur"):
        atelier.enregistrer("dire", "spk-01", sal01, data, sr, interlocuteur="patron")

    # Chacun sa file : sal-01 reste à dire à un inconnu.
    assert atelier.prochaine("dire", "spk-01", PHRASES)["id"] == "sal-02"
    assert atelier.prochaine("dire", "spk-01", PHRASES, interlocuteur="inconnu")["id"] == "sal-01"
    assert atelier.autre_interlocuteur("spk-01", "sal-01", "ami") == "inconnu"
    atelier.passer("dire", "spk-01", "sal-01", "inconnu")
    assert atelier.autre_interlocuteur("spk-01", "sal-01", "ami") is None
    assert atelier.deja_faite("dire", "spk-01", "sal-01", "inconnu")
    prog = atelier.progression("dire", "spk-01", PHRASES, "aine")
    assert sum(r["faites"] for r in prog) == 1
    assert atelier.par_interlocuteur() == {"ami": 1, "inconnu": 0, "aine": 2}

    # Une prise d'avant le choix de l'interlocuteur comptait pour un ami.
    import json as _json
    lignes = (atelier.root / "dire.jsonl").read_text(encoding="utf-8").splitlines()
    vieille = {k: v for k, v in _json.loads(lignes[0]).items() if k != "interlocuteur"}
    (atelier.root / "dire.jsonl").write_text(
        "\n".join([_json.dumps(vieille)] + lignes[1:]) + "\n", encoding="utf-8")
    assert atelier.versions("spk-01", "sal-01") == {"ami": 1, "aine": 2}


def test_enregistrement_refuse_le_vide_et_les_inconnus(atelier):
    with pytest.raises(ValueError, match="vide"):
        atelier.enregistrer("dire", "spk-01", PHRASES[0], np.zeros(0, np.int16), 16_000)
    with pytest.raises(KeyError):
        atelier.enregistrer("dire", "spk-99", PHRASES[0], *_voix()[::-1])
    # Trop faible : enregistré quand même, mais signalé pour recommencer.
    _, problemes = atelier.enregistrer("dire", "spk-01", PHRASES[0],
                                       np.full(32_000, 10, np.int16), 16_000)
    assert any("faible" in p for p in problemes)


def test_phrases_a_lire_filtrees(atelier):
    textes = ["A bɛ taa sugu la sisan.", "a bɛ taa sugu la sisan.",   # doublon
              "Yesu ko u ma ko u ka taa.",                              # religieux
              "N bɛ 5 di i ma.",                                        # chiffre
              "Ɔwɔ.",                                                   # trop court
              "Den ninnu bɛ tulon kɛ lu kɔnɔ."]
    n = _ecrire_lecture(textes, atelier.root / "lecture.jsonl", "test", 10, 0, 4, 14)
    lues = atelier.phrases_a_lire()
    assert n == 2 and {p["bm"] for p in lues} == {"A bɛ taa sugu la sisan.",
                                                   "Den ninnu bɛ tulon kɛ lu kɔnɔ."}
    sr, data = _voix()
    p = atelier.prochaine("lire", "spk-01", lues)
    prise, _ = atelier.enregistrer("lire", "spk-01", p, data, sr)
    assert prise.id == "li-00001" and prise.texte == p["bm"]
    assert atelier.minutes("lire", "spk-01") == 0.0  # 2 s : arrondi au dixième de minute


def _rapport(path: Path, nom: str, hyps: list[str]) -> Path:
    path.write_text(json.dumps({"nom": nom, "maillon": "mt"}), encoding="utf-8")
    rows = [{"id": i, "ref_fr": f"phrase {i}", "ref_bm": f"kuma {i}", "hyp_bm": h,
             "hyp_fr": f"traduction {i}"} for i, h in enumerate(hyps)]
    path.with_suffix(".details.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def test_lot_a_l_aveugle_et_resultats(atelier, tmp_path):
    res = tmp_path / "resultats"
    res.mkdir()
    a = _rapport(res / "mt-1.json", "NLLB v1", ["a bɛ yen", "same", "n bɛ taa", "x", "y"])
    b = _rapport(res / "mt-2.json", "NLLB v2", ["a bɛ yan", "same", "n bɛ na", "x2", "y2"])
    assert trouver_rapport(res, "NLLB v2") == b and trouver_rapport(res, "inconnu") is None
    path, stats = preparer_lot(a, b, "fr2bm", atelier.root, n=10, seed=1, nom="v1-v2")
    assert stats == {"communes": 5, "identiques": 1, "retenues": 4}
    items = atelier.items("v1-v2")
    assert path.exists() and [i["ref_id"] for i in items] == ["0", "2", "3", "4"]
    assert all(i["source"].startswith("phrase") for i in items)
    assert {i["systeme_a"] for i in items} == {"NLLB v1", "NLLB v2"}  # côtés tirés au hasard

    # Le contributeur préfère toujours la v2, quel que soit son côté.
    while (it := atelier.prochain_item("v1-v2", "spk-01")) is not None:
        choix = "A" if it["systeme_a"] == "NLLB v2" else "B"
        if it["ref_id"] == "4":
            choix = "egal_mauvais"
        atelier.juger("v1-v2", "spk-01", it["id"], choix)
    r = atelier.resultats("v1-v2")
    assert r["preferences"] == {"NLLB v1": 0, "NLLB v2": 3} and r["egal_mauvais"] == 1
    assert r["p_signe"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        atelier.juger("v1-v2", "spk-01", items[0]["id"], "C")


def test_lot_cascade_contre_bout_en_bout(atelier, tmp_path):
    """Depuis l'audio : la traduction de ce que la cascade a entendu contre le
    français direct du bout-en-bout, sur la transcription de référence."""
    res = tmp_path / "resultats"
    res.mkdir()
    lignes = {"cascade": [{"id": i, "ref_bm": f"kuma {i}", "ref_fr": f"phrase {i}",
                           "hyp_bm": f"kuma {i}?", "mt_depuis_asr": f"cascade {i}"}
                          for i in range(3)],
              "e2e": [{"id": i, "ref_bm": f"kuma {i}", "ref_fr": f"phrase {i}",
                       "hyp_fr": f"direct {i}"} for i in range(3)]}
    for nom, rows in lignes.items():
        (res / f"asr-{nom}.json").write_text(json.dumps({"nom": nom, "maillon": "asr"}))
        (res / f"asr-{nom}.details.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    _, stats = preparer_lot(res / "asr-cascade.json", res / "asr-e2e.json", "bm2fr",
                            atelier.root, nom="c-e")
    assert stats["retenues"] == 3
    it = atelier.items("c-e")[0]
    assert it["source"] == "kuma 0" and {it["a"], it["b"]} == {"cascade 0", "direct 0"}
    with pytest.raises(ValueError, match="transcription"):
        preparer_lot(res / "asr-cascade.json", res / "asr-e2e.json", "fr2bm", atelier.root)


def test_p_signe():
    assert p_signe(0, 0) == 1.0
    assert p_signe(5, 5) == 1.0
    assert p_signe(10, 0) == pytest.approx(2 / 1024)


def test_lexique(atelier):
    atelier.ajouter_mot("wai", "salut", "i ni ce", auteur="spk-01")
    atelier.ajouter_mot("Wai", "salut (entre jeunes)")   # même forme : remplacé
    atelier.ajouter_mot("yoh", "salut, hé", type="expression")
    assert [(m.forme, m.sens_fr) for m in atelier.lexique()] == [
        ("Wai", "salut (entre jeunes)"), ("yoh", "salut, hé")]
    with pytest.raises(ValueError):
        atelier.ajouter_mot("x", "")
    assert atelier.retirer_mot("YOH") and not atelier.retirer_mot("absent")


def test_exports_et_jeu_de_test_sans_transcription(atelier):
    sr, data = _voix()
    for p in PHRASES[:3]:
        atelier.enregistrer("dire", "spk-01", p, data, sr, code_switching=p["id"] == "sal-02")
    atelier.enregistrer("dire", "spk-01", PHRASES[0], data, sr, interlocuteur="aine")
    atelier.ajouter_mot("wai", "salut")
    out = atelier.exporter()
    items = load_testset(out["testset"])
    assert [i.translation_fr for i in items] == [p["fr"] for p in PHRASES[:3]] + [
        "Bonjour, comment allez-vous ce matin ?"]
    assert [i.extra["interlocuteur"] for i in items] == ["ami", "ami", "ami", "aine"]
    # Partition fixée par la phrase : ses deux prises (ami, aîné) du même côté.
    from eval.atelier import partition
    assert items[0].extra["prompt_id"] == "sal-01" == items[3].extra["prompt_id"]
    assert items[0].extra["partition"] == items[3].extra["partition"] == partition("sal-01")
    assert {partition(p["id"]) for p in PHRASES} == {"dev", "test"}
    assert items[0].transcript_bm == "" and items[0].register == "dit"
    assert items[0].audio_path(out["testset"].parent).exists()
    d = describe(items)
    assert d["n"] == 4 and d["part_code_switching"] == 0.25
    assert out["lexique"].read_text(encoding="utf-8").splitlines()[1].startswith("wai\tsalut")


def test_jeu_de_test_exige_une_reference(tmp_path):
    f = tmp_path / "t.jsonl"
    f.write_text('{"id": "x", "audio": "a.wav"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="ni transcript_bm ni translation_fr"):
        load_testset(f)


def test_application(tmp_path):
    """Les vrais rappels de l'interface, sans navigateur."""
    pytest.importorskip("gradio")
    from app.atelier_app import build_interface

    atelier = Atelier(tmp_path / "atelier")
    demo = build_interface(atelier, PHRASES)
    fns = {}
    for f in demo.fns.values():
        fns.setdefault(f.fn.__name__, []).append(f.fn)
    one = {k: v[0] for k, v in fns.items()}

    msg, _ = one["ajouter_locuteur"]("f", "18-25", "Ségou", True, False)
    assert "spk-01" in msg
    phrase, texte, prog = one["montrer_dire"]("spk-01", "")
    assert phrase["id"] == "sal-01" and "Bonjour" in texte and "0/300" in prog

    sortie = one["enregistrer_dire"]("spk-01", "", "ami", phrase, _voix(), True, "wai",
                                     "salle calme")
    assert sortie[0].startswith("✅") and sortie[2]["id"] == "sal-02" and "1/300" in sortie[4]
    sortie = one["enregistrer_dire"]("spk-01", "", "ami", sortie[2], None, False, "", "")
    assert "Rien d'enregistré" in sortie[0]
    annule = one["annuler_dire"]("spk-01", "", "ami")
    assert annule[2]["id"] == "sal-01" and annule[8] == "ami"

    # La même phrase à un ami, puis à un inconnu (vouvoyée), deux fois.
    sortie = one["enregistrer_autre_personne"]("spk-01", "", "ami", phrase, _voix(), False,
                                               "", "")
    assert sortie[2]["id"] == "sal-01" and sortie[8] == "inconnu"
    assert "allez-vous" in sortie[3] and "déjà dite : à un ami" in sortie[3]
    sortie = one["enregistrer_autre_facon"]("spk-01", "", "inconnu", phrase, _voix(), False,
                                            "", "")
    assert sortie[2]["id"] == "sal-01" and sortie[8] == "inconnu"
    sortie = one["enregistrer_autre_personne"]("spk-01", "", "inconnu", phrase, _voix(),
                                               False, "", "")
    assert sortie[8] == "aine" and "à un inconnu (2 fois)" in sortie[3]
    # Revenir à « un ami » : sal-01 lui est déjà dite, on passe à la suivante.
    assert one["changer_qui"]("spk-01", "", "ami", phrase)[0]["id"] == "sal-02"
    assert one["changer_qui"]("spk-01", "", "aine", phrase)[0]["id"] == "sal-01"

    # Jugement : un lot, puis le bouton « B ».
    res = tmp_path / "res"
    res.mkdir()
    preparer_lot(_rapport(res / "a.json", "v1", ["a", "b"]), _rapport(res / "b.json", "v2",
                 ["a2", "b2"]), "fr2bm", atelier.root, nom="lot")
    item, source, a, b, prog = one["montrer_item"]("spk-01", "lot")
    assert item["id"] == "lot-001" and "phrase 0" in source and "0/2" in prog
    juger_b = [f for f in fns["<lambda>"] if f.__defaults__ == ("B",)][0]
    assert juger_b("spk-01", "lot", item)[0]["id"] == "lot-002"

    assert "✅" in one["ajouter_mot"]("spk-01", "wai", "salut", "", "argot", "")[0]
    bilan = one["bilan_markdown"]()
    assert "| spk-01 (f) | 1 · 2 · 0 |" in bilan and "lot" in bilan and "1 mot" in bilan
    assert "testset-dire.jsonl" in one["exporter"]()
    assert len(one["tout_rafraichir"]("spk-01", "", "ami", "lot")) == 13


# --- notebook Colab -----------------------------------------------------------

NB = ROOT / "notebooks" / "atelier_collecte.ipynb"


def _code(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.mark.parametrize("nb_path", [NB, ROOT / "notebooks" / "phase3_finetuning.ipynb"])
def test_notebook_se_compile_sans_apostrophe_dans_les_commandes(nb_path):
    """« Tout exécuter » s'arrête à la première erreur ; et IPython ne
    remplace pas $VAR quand une apostrophe suit sur la ligne."""
    for k, cellule in enumerate(_code(nb_path)):
        lignes, suite = [], False
        for line in cellule.splitlines():
            stripped = line.lstrip()
            if suite or stripped.startswith(("!", "%")):
                assert suite or "'" not in line, line
                if not suite:
                    lignes.append(line[:len(line) - len(stripped)] + "pass")
                suite = line.rstrip().endswith("\\")
            else:
                lignes.append(line)
        compile("\n".join(lignes), f"{nb_path.name}, cellule {k}", "exec")


def test_notebook_prepare_chaque_lot_une_fois(tmp_path, capsys):
    """Un lot naît dès que ses deux rapports existent, sous le nom que leur
    donne le notebook de la phase 3, et n'est jamais refait."""
    [cellule] = [c for c in _code(NB) if c.startswith("# (nom du lot")]
    res = tmp_path / "resultats-phase3"
    res.mkdir()
    _rapport(res / "mt-0.json", "NLLB zero-shot / Jeli-ASR", ["a", "b", "c"])
    _rapport(res / "mt-1.json", "NLLB affiné / Jeli-ASR", ["a2", "b", "c2"])
    ns = {"ATELIER": str(tmp_path / "atelier"), "OUT": str(res)}
    exec(cellule, ns)
    out = capsys.readouterr().out
    assert "2-affinage-fr2bm : 2 paires à juger (1 traductions identiques" in out
    assert "1-cascade-contre-bout-en-bout : en attente" in out
    assert "4-v1-contre-v2-fr2bm : en attente du rapport ['NLLB v2 / Jeli-ASR']" in out

    exec(cellule, ns)
    assert "2-affinage-fr2bm : déjà prêt" in capsys.readouterr().out
    lots = {p.stem for p in (tmp_path / "atelier" / "lots").glob("*.jsonl")}
    assert lots == {"2-affinage-fr2bm", "3-affinage-bm2fr"}

    # Les noms attendus sont bien ceux que donne la phase 3.
    phase3 = "\n".join(_code(ROOT / "notebooks" / "phase3_finetuning.ipynb"))
    for nom in ("cascade Soloni, NLLB affiné", "bout-en-bout affiné", "NLLB zero-shot / Jeli-ASR",
                "NLLB v2 / Jeli-ASR"):
        assert nom in phase3, nom
    assert 'MT_LABEL = "NLLB affiné"' in phase3 and "$MT_LABEL / Jeli-ASR" in phase3


def _sans_shell(cellule: str) -> str:
    """La cellule en Python pur : chaque commande « ! » (et sa suite après
    « \\ ») devient « pass », à la même indentation."""
    lignes, suite = [], False
    for line in cellule.splitlines():
        stripped = line.lstrip()
        if suite:
            suite = line.rstrip().endswith("\\")
            continue
        if stripped.startswith(("!", "%")):
            lignes.append(line[:len(line) - len(stripped)] + "pass")
            suite = line.rstrip().endswith("\\")
        else:
            lignes.append(line)
    return "\n".join(lignes)


PHRASES_MAISON = [f"il va au marché acheter {k} poissons ce matin" for k in range(40)]


def _rapport_maison(dossier: Path, juste: dict[str, float]) -> None:
    """Rapport eval.run_eval sur 80 énoncés (40 dev, 40 test) ; `juste` : part
    d'énoncés bien compris dans chaque partition."""
    dossier.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(80):
        part = "dev" if i < 40 else "test"
        ref = PHRASES_MAISON[i % 40]
        ok = (i % 40) < juste[part] * 40
        rows.append({"item": f"di-{i:05d}", "source_bm": "a bɛ taa", "partition": part,
                     "source_fr": ref if ok else "il est parti hier",
                     "ref_translation_fr": ref, "speaker": f"spk-0{i % 2}"})
    path = dossier / "cascade-20261017-090000.json"
    path.write_text(json.dumps({"nom": dossier.name}), encoding="utf-8")
    path.with_suffix(".details.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_notebook_phase3_ne_change_d_assemblage_que_s_il_fait_mieux(tmp_path, capsys):
    """§9a : le meilleur sur le dev est le candidat ; il remplace l'assemblage
    en service (Whisper + NLLB v1 au départ) seulement s'il fait mieux sur le
    test. La décision va dans retenu.json, que lit la suite du notebook."""
    pytest.importorskip("transformers")
    pytest.importorskip("sacrebleu")
    cellules = _code(ROOT / "notebooks" / "phase3_finetuning.ipynb")
    [aides] = [c for c in cellules if c.startswith("# Modèle final")]
    [cellule] = [c for c in cellules
                 if c.startswith("import glob, json, os\nfrom pathlib import Path\nfrom eval.promotion")]
    out, ckpt, atelier = tmp_path / "out", tmp_path / "ckpt", tmp_path / "atelier"
    for d in (ckpt / "nllb-bm2fr-v2", tmp_path / "moyenne", atelier):
        d.mkdir(parents=True)
    (ckpt / "nllb-bm2fr-v2" / "model.safetensors").write_bytes(b"")
    (tmp_path / "moyenne" / "model.safetensors").write_bytes(b"")
    (atelier / "dire.jsonl").write_text("{}\n")

    def run(scores):
        for cle, juste in scores.items():
            _rapport_maison(out / "maison-assemblages" / cle, juste)
        ns = {"CKPT": str(ckpt), "OUT": str(out)}
        exec(_sans_shell(aides), ns)
        ns.update(chemins=lambda: ("whisper", "nllb-v1", "fr2bm-v1"),
                  a_jour=lambda *a, **k: True, MOYENNE=str(tmp_path / "moyenne"),
                  CFG=str(tmp_path / "assistant.json"), WHISPER="whisper", SOLONI="soloni",
                  NLLB_FR2BM="fr2bm-v1")
        exec(_sans_shell(cellule.replace("/content/drive/MyDrive/bambara/atelier", str(atelier))
                         .replace("/content/assemblage-", str(tmp_path / "assemblage-"))), ns)
        return ns, json.loads((tmp_path / "assistant.json").read_text())

    egal = {"dev": 0.5, "test": 0.5}
    # 1. Soloni + moyenne : meilleur sur le dev et sur le test → mis en service.
    ns, cfg = run({"Whisper-NLLB-v1": egal, "Whisper-NLLB-v2": egal,
                   "Whisper-NLLB-moyenne": egal, "Soloni-NLLB-v1": egal,
                   "Soloni-NLLB-v2": {"dev": 0.6, "test": 0.5},
                   "Soloni-NLLB-moyenne": {"dev": 0.8, "test": 0.8}})
    sortie = capsys.readouterr().out
    assert "Candidat (meilleur sur le dev) : Soloni + NLLB moyenne" in sortie
    assert "**Promu" in sortie and "| Whisper + NLLB v1 (en service) | " in sortie
    assert cfg == {"device": "cuda", "asr": {"model_id": "soloni", "kind": "nemo"},
                   "mt_in": {"model_id": str(tmp_path / "moyenne")},
                   "mt_out": {"model_id": "fr2bm-v1"}}
    retenu = json.loads((ckpt / "retenu.json").read_text())
    assert retenu["nom_bm2fr"] == "NLLB moyenne" and retenu["historique"][0]["promu"]

    # 2. Whisper + v2 : bien meilleur sur le dev, pas sur le test → refusé.
    ns, cfg = run({"Soloni-NLLB-moyenne": {"dev": 0.5, "test": 0.5},
                   "Whisper-NLLB-v2": {"dev": 0.9, "test": 0.5}})
    sortie = capsys.readouterr().out
    assert "Candidat (meilleur sur le dev) : Whisper + NLLB v2" in sortie
    assert "**Refusé : Soloni + NLLB moyenne reste en service.**" in sortie
    assert cfg["asr"]["model_id"] == "soloni" and ns["EN_SERVICE"]["nom_bm2fr"] == "NLLB moyenne"
    assert ns["RAPPORT_EN_SERVICE"].endswith("Soloni-NLLB-moyenne/cascade-20261017-090000.json")


def _rapport_mt(dossier: Path, nom: str, juste: float) -> None:
    dossier.mkdir(parents=True, exist_ok=True)
    rows = [{"id": i, "ref_bm": "a bɛ taa", "hyp_bm": "a bɛ taa",
             "ref_fr": PHRASES_MAISON[i % 40],
             "hyp_fr": PHRASES_MAISON[i % 40] if i < juste * 60 else "rien"} for i in range(60)]
    path = dossier / f"mt-{nom.replace(' ', '-')}.json"
    path.write_text(json.dumps({"nom": nom, "maillon": "mt"}), encoding="utf-8")
    path.with_suffix(".details.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


@pytest.mark.parametrize("bayelemabaga_v3, promu", [(0.8, True), (0.3, False)])
def test_notebook_phase3_v3_en_service_seulement_si_meilleure(tmp_path, capsys,
                                                              bayelemabaga_v3, promu):
    """§9c : dev maison transcrit pour choisir le checkpoint ; la v3 n'entre en
    service que si elle fait mieux sur le test maison, sans régression sur
    les contrôles écrits."""
    pytest.importorskip("transformers")
    pytest.importorskip("sacrebleu")
    cellules = _code(ROOT / "notebooks" / "phase3_finetuning.ipynb")
    [aides] = [c for c in cellules if c.startswith("# Modèle final")]
    [cellule] = [c for c in cellules if "JELI_V3 = " in c]
    out, ckpt = tmp_path / "out", tmp_path / "ckpt"
    (ckpt / "nllb-bm2fr-v3").mkdir(parents=True)
    (ckpt / "nllb-bm2fr-v3" / "model.safetensors").write_bytes(b"")
    _rapport_maison(out / "maison-assemblages" / "Whisper-NLLB-v1", {"dev": 0.5, "test": 0.5})
    _rapport_maison(out / "maison-assemblages" / "Whisper-NLLB-v3", {"dev": 0.8, "test": 0.8})
    for corpus in ("Bayelemabaga", "MAFAND"):
        _rapport_mt(out / "controles", f"controle {corpus} nllb-v1", 0.8)
        _rapport_mt(out / "controles", f"controle {corpus} nllb-bm2fr-v3",
                    bayelemabaga_v3 if corpus == "Bayelemabaga" else 0.8)
    prises = tmp_path / "dire.jsonl"
    prises.write_text("{}\n")

    ns = {"CKPT": str(ckpt), "OUT": str(out)}
    exec(_sans_shell(aides), ns)
    ns.update(chemins=lambda: ("whisper", "nllb-v1", "fr2bm-v1"), a_jour=lambda *a, **k: True,
              place_drive=lambda go: None, CFG=str(tmp_path / "assistant.json"),
              NLLB_V3=True, JELI_CONFIG="", PRISES=str(prises), MAISON="maison.jsonl",
              NLLB_FR2BM="fr2bm-v1", B2F_V3=str(ckpt / "nllb-bm2fr-v3"),
              TITULAIRE="Whisper + NLLB v1",
              RAPPORT_EN_SERVICE=str(out / "maison-assemblages" / "Whisper-NLLB-v1"
                                     / "cascade-20261017-090000.json"))
    ns["EN_SERVICE"] = ns["en_service"]()
    code = cellule
    for chemin in ("/content/donnees-v3", "/content/nllb-bm2fr-v3", "/content/maison-dev.jsonl",
                   "/content/assemblage-v3.json"):
        code = code.replace(chemin, str(tmp_path / Path(chemin).name))
    exec(_sans_shell(code), ns)

    dev = (tmp_path / "maison-dev.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(dev) == 40 and json.loads(dev[0]) == {"bm": "a bɛ taa", "fr": PHRASES_MAISON[0]}
    sortie = capsys.readouterr().out
    assert "Dev maison : 40 énoncés transcrits par Whisper" in sortie
    retenu = json.loads((ckpt / "retenu.json").read_text())
    cfg = json.loads((tmp_path / "assistant.json").read_text())
    if promu:
        assert "**Promu : Whisper + NLLB v3 remplace Whisper + NLLB v1.**" in sortie
        assert retenu["bm2fr"] == cfg["mt_in"]["model_id"] == str(ckpt / "nllb-bm2fr-v3")
    else:
        assert "régression sur chrF++ bm→fr" in sortie and retenu["bm2fr"] == "nllb-v1"
        assert cfg["mt_in"]["model_id"] == "nllb-v1"
    assert retenu["historique"][-1]["maillon"] == "bambara → français (v3)"
