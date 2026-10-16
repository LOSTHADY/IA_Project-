"""Tests de eval/corpora.py sur de petits jeux parquet locaux, construits
comme ceux du Hub (colonne audio {bytes, path}, ou translation {bam, fr})."""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

datasets = pytest.importorskip("datasets")
sf = pytest.importorskip("soundfile")

from eval.corpora import (  # noqa: E402
    decode_audio, detect_columns, iter_examples, load_split, sample_indices,
)


def _flac(seconds=1.0, sr=44_100, channels=2):
    t = np.arange(int(seconds * sr)) / sr
    wav = np.stack([0.2 * np.sin(2 * np.pi * 220 * t)] * channels, axis=1)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="FLAC")
    return buf.getvalue()


def _jeli_like(root: Path) -> Path:
    """Un jeu parole + transcription + traduction, partitions train/test."""
    for split, n in (("train", 2), ("test", 3)):
        datasets.Dataset.from_dict({
            "audio": [{"bytes": _flac(), "path": f"{split}-{i}.flac"} for i in range(n)],
            "duration": [1.0] * n,
            "bam": [f"n bɛ taa {i}" for i in range(n)],
            "french": [f"je pars {i}" for i in range(n)],
        }).to_parquet(str(root / f"{split}.parquet"))
    return root


def _bayelemabaga_like(root: Path) -> Path:
    for split in ("train", "validation"):
        datasets.Dataset.from_dict({
            "translation": [{"bam": "i ni ce", "fr": "bonjour"},
                            {"bam": "a bɛ yen", "fr": "il est là"}],
        }).to_parquet(str(root / f"{split}.parquet"))
    return root


# --- détection des colonnes ---------------------------------------------------

def test_colonnes_plates_et_audio_du_hub():
    from datasets import Audio, Features, Value

    feats = Features({"audio": Audio(), "duration": Value("float32"),
                      "bam": Value("string"), "french": Value("string")})
    cols = detect_columns(feats, need_audio=True, need_fr=True)
    assert (cols.bm, cols.fr, cols.audio, cols.nested) == ("bam", "french", "audio", None)


def test_colonne_translation_facon_wmt():
    from datasets import Features, Translation

    cols = detect_columns(Features({"translation": Translation(languages=["bam", "fr"])}))
    assert (cols.nested, cols.bm, cols.fr) == ("translation", "bam", "fr")


def test_forcer_une_colonne_et_erreurs_explicites():
    from datasets import Features, Value

    feats = Features({"text": Value("string"), "sentence_fr": Value("string")})
    assert detect_columns(feats, fr="sentence_fr").fr == "sentence_fr"
    with pytest.raises(ValueError, match="absente"):
        detect_columns(feats, bm="bambara")
    with pytest.raises(ValueError, match="--fr-column"):
        detect_columns(feats, need_fr=True)
    with pytest.raises(ValueError, match="audio"):
        detect_columns(feats, need_audio=True)


# --- audio --------------------------------------------------------------------

def test_decode_audio_tous_formats(tmp_path):
    out = decode_audio({"bytes": _flac(), "path": None})
    assert out.dtype == np.float32 and out.ndim == 1 and abs(len(out) - 16_000) <= 1

    path = tmp_path / "a.flac"
    path.write_bytes(_flac(sr=16_000, channels=1))
    assert len(decode_audio({"bytes": None, "path": str(path)})) == 16_000

    arr = decode_audio({"array": np.zeros(8_000), "sampling_rate": 8_000})
    assert len(arr) == 16_000

    with pytest.raises(ValueError):
        decode_audio({"bytes": None, "path": None})


# --- chargement et parcours ---------------------------------------------------

def test_load_split_prend_test_par_defaut(tmp_path):
    ds = load_split(str(_jeli_like(tmp_path)))
    assert str(ds.split) == "test" and len(ds) == 3


def test_load_split_repli_sur_validation(tmp_path):
    ds = load_split(str(_bayelemabaga_like(tmp_path)))
    assert str(ds.split) == "validation"


def test_load_split_repli_parquet_pour_les_jeux_a_script(monkeypatch):
    calls = []

    def fake_load(path, config=None, split=None, revision=None):
        calls.append(revision)
        if revision is None:
            raise RuntimeError("Dataset scripts are no longer supported, but found bayelemabaga.py")
        return "ok"

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    assert load_split("RobotsMaliAI/bayelemabaga", split="test") == "ok"
    assert calls == [None, "refs/convert/parquet"]


class _Http429(Exception):
    """Imite huggingface_hub.errors.HfHubHTTPError sur une réponse 429."""

    class response:
        status_code = 429


def test_limite_de_debit_du_hub_reessaye_puis_reussit(monkeypatch):
    from eval import corpora

    calls, waits = [], []

    def fake_load(path, config=None, split=None, revision=None):
        calls.append(path)
        if len(calls) < 3:
            raise _Http429("429 Too Many Requests: maximum queue size reached")
        return "ok"

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    monkeypatch.setattr(corpora.time, "sleep", waits.append)
    assert load_split("RobotsMali/jeli-asr", split="test") == "ok"
    assert len(calls) == 3 and len(waits) == 2 and waits[1] > waits[0] >= 30


def test_limite_de_debit_abandon_apres_plusieurs_essais():
    from eval.corpora import with_retries

    def always_429():
        raise _Http429("429 Too Many Requests")

    waits = []
    with pytest.raises(_Http429):
        with_retries(always_429, "essai", attempts=3, sleep=waits.append)
    assert len(waits) == 2


def test_une_autre_erreur_n_est_pas_reessayee():
    from eval.corpora import with_retries

    waits = []
    with pytest.raises(ValueError):
        with_retries(lambda: (_ for _ in ()).throw(ValueError("colonne")), "essai",
                     sleep=waits.append)
    assert waits == []


def test_iter_examples_audio_et_texte(tmp_path):
    ds = load_split(str(_jeli_like(tmp_path)))
    cols = detect_columns(ds.features, need_audio=True)
    exs = list(iter_examples(ds, cols, [0, 2]))
    assert [e.bm for e in exs] == ["n bɛ taa 0", "n bɛ taa 2"]
    assert exs[0].fr == "je pars 0" and len(exs[0].audio) == 16_000

    sans_audio = list(iter_examples(ds, cols, with_audio=False))
    assert len(sans_audio) == 3 and sans_audio[0].audio is None


def test_iter_examples_translation(tmp_path):
    ds = load_split(str(_bayelemabaga_like(tmp_path)))
    exs = list(iter_examples(ds, detect_columns(ds.features)))
    assert (exs[1].bm, exs[1].fr) == ("a bɛ yen", "il est là")


def test_echantillon_reproductible():
    a = sample_indices(1000, 10, seed=3)
    assert a == sample_indices(1000, 10, seed=3) and a == sorted(a) and len(set(a)) == 10
    assert a != list(range(10))  # pas les 10 premières lignes
    assert sample_indices(5, 10) == [0, 1, 2, 3, 4]


# --- données de la V3 (scripts/donnees_v3.py) et jeux locaux ------------------------

def _sources_v3(d: Path) -> Path:
    """Copies minuscules des vrais formats : Toolbox du Bamadaba, TSV de
    GATITOS, JSON Lines de MAFAND."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "bamadaba.txt").write_text(
        "\\lx dàba\n\\xv dàba d'á mà !\n\\xf donne-lui la houe !\n"
        "\\xv à tá tɛ́\n\\xf ce n'est pas le sien ..., la sienne ... (ce n'est pas sa propriété)\n"
        "\\xv kà táa kà mɔ̀gɔ bólo dàn\n\\xf dépasser la compétence de qqn., sortir ; déborder\n"
        "\\xv à y'à fɔ́ ...\n\\xf il l'a dit ...\n"
        "\\xv í ní sɔ̀gɔma\n\\xf bonjour, comment vas-tu ce matin ?\n", encoding="utf-8")
    (d / "en_bm.tsv").write_text("good morning\ta' ni sɔgɔma\nmarket\tsugu\nonly\tdɔrɔn\n",
                                 encoding="utf-8")
    (d / "en_fr.tsv").write_text("good morning\tbonjour\nmarket\tmarché\n", encoding="utf-8")
    rows = {"train": [("O ye kɛ.", "Cela a été fait."), ("A bɛ yen.", "Il est là."),
                      ("A bɛ yen.", "Il est là.")],
            "dev": [("Dev bɛ.", "Développement.")], "test": [("A bɛ yen.", "Test.")]}
    for split, pairs in rows.items():
        (d / f"mafand-{split}.json").write_text("\n".join(
            json.dumps({"translation": {"fr": fr, "bam": bm}}, ensure_ascii=False)
            for bm, fr in pairs), encoding="utf-8")
    return d


def test_donnees_v3_nettoie_et_tient_les_tests_a_l_ecart(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import donnees_v3

    assert donnees_v3.sans_tons("à tá tɛ́ mɔ̀gɔ") == "a ta tɛ mɔgɔ"
    stats = donnees_v3.preparer(tmp_path / "out", _sources_v3(tmp_path / "src"),
                                ROOT / "data" / "atelier" / "a_dire.json")
    lire = lambda p: [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    bamadaba = lire(tmp_path / "out" / "bamadaba" / "train.jsonl")
    assert [(p["bm"], p["fr"]) for p in bamadaba] == [
        ("daba d'a ma !", "donne-lui la houe !"),
        ("a ta tɛ", "ce n'est pas le sien, la sienne"),
        ("ka taa ka mɔgɔ bolo dan", "dépasser la compétence de quelqu'un, sortir"),
    ]  # l'exemple inachevé est écarté ; la phrase du jeu maison aussi
    assert stats["bamadaba"] == {"gardees": 3, "ecartees": 1}
    gatitos = lire(tmp_path / "out" / "gatitos" / "train.jsonl")
    assert [(p["bm"], p["fr"]) for p in gatitos] == [("a' ni sɔgɔma", "bonjour"),
                                                      ("sugu", "marché")]
    # MAFAND : test et validation intacts, et rien d'eux dans l'entraînement.
    mafand = tmp_path / "out" / "mafand"
    assert [p["bm"] for p in lire(mafand / "train.jsonl")] == ["O ye kɛ."]
    assert len(lire(mafand / "test.jsonl")) == 1 and len(lire(mafand / "validation.jsonl")) == 1


def test_jeux_locaux(tmp_path):
    pytest.importorskip("datasets")
    from eval.corpora import load_any, load_split, local_files

    d = tmp_path / "jeu"
    d.mkdir()
    for split in ("train", "validation", "test"):
        (d / f"{split}.jsonl").write_text(json.dumps({"bm": f"bm {split}", "fr": f"fr {split}"}),
                                          encoding="utf-8")
    (d / "notes.txt").write_text("ignoré")
    assert set(local_files(str(d))) == {"train", "validation", "test"}
    assert local_files(str(d / "train.jsonl")) == {"train": str(d / "train.jsonl")}
    assert local_files("RobotsMali/jeli-asr") is None
    assert load_split(str(d))[0]["bm"] == "bm test"  # test d'abord, comme sur le Hub
    assert load_any(str(d / "train.jsonl"))["train"][0]["fr"] == "fr train"
