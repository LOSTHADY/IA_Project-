"""Corpus publics : Jeli-ASR (parole) et Bayelemabaga (texte parallèle).

Ces jeux évoluent (le dépôt Jeli-ASR se dit lui-même « en révision ») et les
noms de colonnes ne sont pas documentés de façon stable. Plutôt que de les
coder en dur, on les détecte au chargement, on affiche le schéma, et
`--bm-column` / `--fr-column` permettent toujours de forcer.

L'audio est lu sans décodage par `datasets` (`Audio(decode=False)`) puis
décodé par soundfile : depuis datasets 4, le décodage intégré exige
torchcodec, une dépendance lourde qui n'apporte rien ici.
"""

from __future__ import annotations

import io
import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Iterator, TypeVar

import numpy as np

from .collection import to_mono_16k

logger = logging.getLogger(__name__)

JELI_ASR = "RobotsMali/jeli-asr"
BAYELEMABAGA = "RobotsMaliAI/bayelemabaga"

# Par ordre de préférence. Les codes courts d'abord : "text" ou "sentence"
# sont ambigus et ne servent qu'en dernier recours.
BM_CANDIDATES = ("bam", "bm", "bambara", "bam_latn", "transcription", "text", "sentence")
FR_CANDIDATES = ("french", "fr", "fra", "fra_latn", "francais", "français", "translation_fr")
EVAL_SPLITS = ("test", "validation", "valid", "dev")


@dataclass
class Columns:
    """Où trouver chaque champ dans un jeu donné."""

    bm: str
    fr: str | None = None
    audio: str | None = None
    nested: str | None = None  # colonne {bam: ..., fr: ...} façon WMT

    def describe(self) -> str:
        prefix = f"{self.nested}." if self.nested else ""
        return (f"bambara={prefix}{self.bm}, français="
                f"{prefix + self.fr if self.fr else '—'}, audio={self.audio or '—'}")


def _is_audio(name: str, feature) -> bool:
    if type(feature).__name__ == "Audio":
        return True
    # Colonne audio déjà relue sans décodage : {bytes, path}.
    return isinstance(feature, dict) and {"bytes", "path"} <= set(feature)


def _pick(names, candidates, wanted: str | None, what: str) -> str | None:
    lower = {n.lower(): n for n in names}
    if wanted:
        if wanted not in names:
            raise ValueError(f"colonne {what} '{wanted}' absente ; colonnes : {sorted(names)}")
        return wanted
    return next((lower[c] for c in candidates if c in lower), None)


def detect_columns(features, bm: str | None = None, fr: str | None = None,
                   need_audio: bool = False, need_fr: bool = False) -> Columns:
    """Devine les colonnes bambara, français et audio d'un `datasets.Features`."""
    names = list(features)
    audio = next((n for n in names if _is_audio(n, features[n])), None)
    if need_audio and audio is None:
        raise ValueError(f"aucune colonne audio ; colonnes : {names}")

    # Format parallèle façon WMT : translation = {"bam": ..., "fr": ...}.
    nested = None
    for n in names:
        langs = getattr(features[n], "languages", None)
        if langs is None and isinstance(features[n], dict):
            langs = list(features[n])
        if langs and _pick(langs, BM_CANDIDATES, None, "") and _pick(langs, FR_CANDIDATES, None, ""):
            nested, names = n, list(langs)
            break

    bm_col = _pick(names, BM_CANDIDATES, bm, "bambara")
    if bm_col is None:
        raise ValueError(f"colonne bambara introuvable (--bm-column) ; colonnes : {names}")
    fr_col = _pick(names, FR_CANDIDATES, fr, "française")
    if need_fr and fr_col is None:
        raise ValueError(f"colonne française introuvable (--fr-column) ; colonnes : {names}")
    return Columns(bm=bm_col, fr=fr_col, audio=audio, nested=nested)


T = TypeVar("T")


def _rate_limited(exc: BaseException) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 429 or "429" in str(exc) or "Too Many Requests" in str(exc)


def with_retries(fn: Callable[[], T], what: str, attempts: int = 6,
                 base_delay: float = 30.0, sleep: Callable[[float], None] | None = None) -> T:
    """Réessaie quand le Hub limite le débit (HTTP 429).

    Sans jeton, plusieurs tâches qui listent le même gros dépôt au même
    moment (les mesures parallèles de la CI) se font refuser. Attente
    exponentielle, avec une part aléatoire pour que les tâches ne
    réessaient pas toutes ensemble ; toute autre erreur remonte aussitôt.
    """
    sleep = sleep or time.sleep  # résolu à l'appel : remplaçable dans les tests
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if not _rate_limited(exc) or attempt == attempts - 1:
                raise
            delay = base_delay * 2 ** attempt * (1 + random.random() / 2)
            logger.warning("%s : Hugging Face limite le débit (429), nouvel essai dans %.0f s "
                           "(%d/%d). Un jeton HF_TOKEN relève cette limite.",
                           what, delay, attempt + 1, attempts - 1)
            sleep(delay)
    raise AssertionError("inatteignable")


def load_split(dataset_id: str, split: str | None = None, config: str | None = None,
               revision: str | None = None):
    """Charge une partition d'évaluation, avec deux replis :

    - partition non précisée : la première trouvée parmi test/validation/dev ;
    - jeu à script de chargement (refusé depuis datasets 4) : la conversion
      parquet que le Hub en conserve.
    """
    from datasets import get_dataset_split_names

    if split is None:
        try:
            names = with_retries(
                lambda: get_dataset_split_names(dataset_id, config, revision=revision),
                f"{dataset_id} (partitions)")
        except Exception as exc:  # noqa: BLE001 - on retombe sur "test"
            logger.debug("partitions non listables (%s)", exc)
            names = []
        split = next((s for s in EVAL_SPLITS if s in names), "test")

    return load_any(dataset_id, config, split=split, revision=revision)


def load_any(dataset_id: str, config: str | None = None, **kwargs):
    """`load_dataset` avec des erreurs lisibles, et le repli sur la conversion
    parquet du Hub pour les jeux à script (refusés depuis datasets 4)."""
    from datasets import load_dataset

    try:
        return with_retries(lambda: load_dataset(dataset_id, config, **kwargs), dataset_id)
    except Exception as exc:
        msg = str(exc).lower()
        if kwargs.get("revision") is None and "script" in msg:
            logger.warning("%s : script de chargement non supporté, "
                           "repli sur la conversion parquet du Hub", dataset_id)
            return with_retries(
                lambda: load_dataset(dataset_id, config,
                                     **{**kwargs, "revision": "refs/convert/parquet"}),
                dataset_id)
        if "config" in msg and config is None:
            raise ValueError(f"{dataset_id} a plusieurs sous-ensembles : préciser --config. "
                             f"Détail : {exc}") from exc
        raise


def sample_indices(n_total: int, limit: int | None, seed: int = 0) -> list[int]:
    """Échantillon reproductible. Prendre les N premières lignes biaiserait
    l'évaluation vers un seul enregistrement ou un seul locuteur."""
    if limit is None or limit >= n_total:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_total, size=limit, replace=False).tolist())


def decode_audio(value) -> np.ndarray:
    """Audio d'une ligne `datasets` -> mono float32 16 kHz, quel que soit le
    format : {bytes, path}, {array, sampling_rate}, ou décodeur torchcodec."""
    import soundfile as sf

    if isinstance(value, dict):
        if value.get("array") is not None:
            return to_mono_16k(np.asarray(value["array"]), value["sampling_rate"])
        if value.get("bytes"):
            data, sr = sf.read(io.BytesIO(value["bytes"]), dtype="float32")
            return to_mono_16k(data, sr)
        if value.get("path"):
            data, sr = sf.read(value["path"], dtype="float32")
            return to_mono_16k(data, sr)
        raise ValueError("audio vide (ni bytes, ni path, ni array)")
    if hasattr(value, "get_all_samples"):  # torchcodec AudioDecoder (datasets >= 4)
        samples = value.get_all_samples()
        return to_mono_16k(samples.data.numpy().T, samples.sample_rate)
    raise TypeError(f"format audio inattendu : {type(value).__name__}")


@dataclass
class Example:
    id: str
    bm: str
    fr: str = ""
    audio: np.ndarray | None = None


def iter_examples(ds, cols: Columns, indices: list[int] | None = None,
                  with_audio: bool = True) -> Iterator[Example]:
    """Parcourt le jeu en exemples normalisés. L'audio n'est décodé que si
    demandé : l'évaluation de la traduction n'en a pas besoin."""
    from datasets import Audio

    if cols.audio:
        if with_audio:
            ds = ds.cast_column(cols.audio, Audio(decode=False))
        else:
            ds = ds.remove_columns(cols.audio)
    indices = range(len(ds)) if indices is None else indices
    for i in indices:
        row = ds[int(i)]
        text = row[cols.nested] if cols.nested else row
        yield Example(
            id=str(row.get("id", i)),
            bm=(text.get(cols.bm) or "").strip(),
            fr=(text.get(cols.fr) or "").strip() if cols.fr else "",
            audio=decode_audio(row[cols.audio]) if cols.audio and with_audio else None,
        )
