"""Tests de eval/corpora.py sur de petits jeux parquet locaux, construits
comme ceux du Hub (colonne audio {bytes, path}, ou translation {bam, fr})."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
