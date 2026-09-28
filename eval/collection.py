"""Collecte du jeu de test : stockage, contrôle audio, suivi de composition.

Toute la logique vit ici pour être testée sans interface ; l'application
d'enregistrement (`app/collect_app.py`) n'est qu'une façade.

Un dossier de collecte (par défaut `data/testset/`) contient :

- `locuteurs.jsonl` : un locuteur par ligne, **pseudonymisé** (spk-01...),
  avec la trace de son consentement. Aucun nom : les formulaires signés
  restent hors du dépôt.
- `collecte.jsonl` : un enregistrement par ligne, avec son statut
  `a_transcrire` -> `transcrit` -> `valide`.
- `audio/*.wav` : WAV mono 16 kHz 16 bits, jamais versionnés (.gitignore).

Seuls les enregistrements validés par un second locuteur passent dans
`testset.jsonl` (`export_testset`) : une transcription non relue ne vaut pas
référence. Cf. docs/COLLECTE.md pour le protocole.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from bambara_voice.normalize import normalize

from .dataset import MIN_ITEMS, MAX_ITEMS, MIN_SPEAKERS

SAMPLE_RATE = 16_000

REGISTERS = ("spontane", "lu")
STATUSES = ("a_transcrire", "transcrit", "valide")
CONDITIONS = ("salle calme", "intérieur bruyant", "extérieur", "téléphone")

# Cibles de composition. Les seuils au-delà du nombre d'items et de locuteurs
# sont indicatifs : ils signalent un déséquilibre, ils ne le définissent pas.
MIN_DURATION_S, MAX_DURATION_S = 2.0, 15.0
MAX_SPEAKER_SHARE = 0.20      # au-delà, le WER mesure surtout une voix
MIN_GENDER_SHARE = 0.40
MIN_REGISTER_SHARE = 0.25     # ni tout lu, ni tout spontané
MIN_CODE_SWITCHING = 0.10
MIN_CONDITIONS = 2

# Contrôle audio à la prise de son : mieux vaut réenregistrer tout de suite
# que découvrir un fichier inutilisable à la transcription.
_CLIP_LEVEL = 0.99
_MAX_CLIPPED_SHARE = 0.001
_MIN_RMS = 0.005              # environ -46 dBFS : quasi-silence


@dataclass
class Speaker:
    id: str
    gender: str                    # "f" ou "m"
    age_range: str = ""
    region: str = ""
    first_language: str = "bambara"
    consent: bool = False          # formulaire signé (ou consentement oral attesté)
    consent_date: str = ""
    public_release: bool = False   # accepte la diffusion publique de sa voix
    notes: str = ""


@dataclass
class Recording:
    id: str
    audio: str                     # chemin relatif au dossier de collecte
    speaker: str
    register: str                  # "spontane" ou "lu"
    prompt: str = ""               # consigne ou phrase montrée au locuteur
    prompt_id: str = ""
    conditions: str = ""
    duration_s: float = 0.0
    recorded_at: str = ""
    status: str = "a_transcrire"
    transcript_bm: str = ""
    translation_fr: str = ""
    code_switching: bool = False
    transcriber: str = ""
    validator: str = ""
    notes: str = ""


def _from_dict(cls, raw: dict):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in known})


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Réécriture atomique : un plantage en cours d'écriture ne perd rien."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _next_id(existing: list[str], prefix: str, width: int) -> str:
    """Identifiant suivant, calculé sur le maximum et non sur le nombre de
    lignes : une suppression ne doit jamais provoquer de doublon."""
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    nums = [int(m.group(1)) for i in existing if (m := pattern.match(i))]
    return f"{prefix}{max(nums, default=0) + 1:0{width}d}"


# --- audio --------------------------------------------------------------------

def to_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convertit ce que renvoie un micro ou un fichier en mono float32 16 kHz."""
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        # Gradio renvoie (échantillons, canaux) ; certains lecteurs l'inverse.
        axis = 1 if audio.shape[1] <= audio.shape[0] else 0
        audio = audio.mean(axis=axis)
    if sample_rate != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def check_audio(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> list[str]:
    """Problèmes détectables à la prise de son. Liste vide = rien à signaler."""
    if len(wav) == 0:
        return ["enregistrement vide"]
    problems = []
    duration = len(wav) / sample_rate
    if duration < 1.0:
        problems.append(f"trop court ({duration:.1f} s) : probablement coupé")
    elif not MIN_DURATION_S <= duration <= MAX_DURATION_S:
        problems.append(
            f"durée {duration:.1f} s, hors de la plage visée "
            f"{MIN_DURATION_S:.0f}–{MAX_DURATION_S:.0f} s"
        )
    clipped = float(np.mean(np.abs(wav) >= _CLIP_LEVEL))
    if clipped > _MAX_CLIPPED_SHARE:
        problems.append(f"saturation ({clipped:.1%} des échantillons) : s'éloigner du micro")
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    if rms < _MIN_RMS:
        problems.append("niveau très faible : se rapprocher du micro ou vérifier l'entrée")
    return problems


# --- choix des consignes ------------------------------------------------------

def load_prompts(path: str | Path) -> list[dict]:
    """Consignes de parole spontanée : liste de {id, theme, consigne}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["consignes"] if isinstance(data, dict) else data


def next_prompt(candidates: list[dict], recordings: list[Recording],
                speaker_id: str) -> dict | None:
    """Prochaine consigne pour ce locuteur : une qu'il n'a pas encore faite,
    la moins enregistrée jusqu'ici, pour répartir la couverture thématique."""
    done = {r.prompt_id for r in recordings if r.speaker == speaker_id}
    counts = Counter(r.prompt_id for r in recordings)
    todo = [c for c in candidates if c["id"] not in done]
    if not todo:
        return None
    return min(todo, key=lambda c: counts[c["id"]])


def read_candidates(recordings: list[Recording], speaker_id: str,
                    extra_sentences: list[str] = ()) -> list[dict]:
    """Phrases à faire lire à ce locuteur.

    Source principale : les transcriptions *validées* d'énoncés spontanés
    d'autres locuteurs. On n'invente pas de bambara, et un même contenu
    existe alors en version spontanée et en version lue — ce qui permet de
    mesurer l'effet du registre à contenu égal.
    """
    own = {r.transcript_bm for r in recordings if r.speaker == speaker_id}
    sentences = [
        r.transcript_bm for r in recordings
        if r.register == "spontane" and r.status == "valide"
        and r.speaker != speaker_id and r.transcript_bm
    ]
    sentences += [normalize(s, lower=False) for s in extra_sentences if s.strip()]
    out, seen = [], set()
    for s in sentences:
        if s not in seen and s not in own:
            seen.add(s)
            out.append({"id": "lu:" + s, "theme": "lecture", "consigne": s})
    return out


# --- stockage -----------------------------------------------------------------

class Collection:
    """Dossier de collecte. Sûr en présence de plusieurs requêtes simultanées
    (Gradio traite les événements dans des fils séparés)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.speakers_path = self.root / "locuteurs.jsonl"
        self.recordings_path = self.root / "collecte.jsonl"
        self.audio_dir = self.root / "audio"
        self._lock = threading.Lock()

    # lecture

    def speakers(self) -> list[Speaker]:
        return [_from_dict(Speaker, r) for r in _read_jsonl(self.speakers_path)]

    def recordings(self) -> list[Recording]:
        return [_from_dict(Recording, r) for r in _read_jsonl(self.recordings_path)]

    def speaker(self, speaker_id: str) -> Speaker:
        for s in self.speakers():
            if s.id == speaker_id:
                return s
        raise KeyError(f"locuteur inconnu : {speaker_id}")

    def recording(self, rec_id: str) -> Recording:
        for r in self.recordings():
            if r.id == rec_id:
                return r
        raise KeyError(f"enregistrement inconnu : {rec_id}")

    # écriture

    def add_speaker(self, gender: str, **info) -> Speaker:
        if not info.get("consent"):
            raise ValueError("pas d'enregistrement sans consentement du locuteur")
        if gender not in ("f", "m"):
            raise ValueError("genre attendu : 'f' ou 'm'")
        with self._lock:
            rows = _read_jsonl(self.speakers_path)
            spk = Speaker(id=_next_id([r["id"] for r in rows], "spk-", 2),
                          gender=gender, **info)
            self.root.mkdir(parents=True, exist_ok=True)
            _write_jsonl(self.speakers_path, rows + [asdict(spk)])
        return spk

    def add_recording(self, speaker_id: str, audio: np.ndarray, sample_rate: int,
                      register: str, prompt: dict, conditions: str = "",
                      ) -> tuple[Recording, list[str]]:
        """Écrit le WAV et sa ligne de collecte. Renvoie aussi les problèmes
        audio détectés, pour proposer de réenregistrer tout de suite."""
        spk = self.speaker(speaker_id)
        if not spk.consent:
            raise ValueError(f"{speaker_id} n'a pas donné son consentement")
        if register not in REGISTERS:
            raise ValueError(f"registre inconnu : {register}")

        wav = to_mono_16k(audio, sample_rate)
        problems = check_audio(wav)

        import soundfile as sf

        with self._lock:
            rows = _read_jsonl(self.recordings_path)
            rec_id = _next_id([r["id"] for r in rows], "bv-", 4)
            rel = f"audio/{rec_id}.wav"
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            sf.write(str(self.root / rel), wav, SAMPLE_RATE, subtype="PCM_16")

            rec = Recording(
                id=rec_id, audio=rel, speaker=speaker_id, register=register,
                prompt=prompt["consigne"], prompt_id=prompt["id"],
                conditions=conditions,
                duration_s=round(len(wav) / SAMPLE_RATE, 2),
                recorded_at=datetime.now().isoformat(timespec="seconds"),
            )
            if register == "lu":
                # Le texte lu sert de première transcription ; la validation
                # vérifie que le locuteur a bien dit ce qui était écrit.
                rec.transcript_bm = prompt["consigne"]
                rec.status = "transcrit"
            _write_jsonl(self.recordings_path, rows + [asdict(rec)])
        return rec, problems

    def update_recording(self, rec_id: str, **changes) -> Recording:
        unknown = set(changes) - {f.name for f in fields(Recording)}
        if unknown:
            raise ValueError(f"champs inconnus : {sorted(unknown)}")
        if "transcript_bm" in changes:
            changes["transcript_bm"] = normalize(changes["transcript_bm"], lower=False)
        if "translation_fr" in changes:
            changes["translation_fr"] = changes["translation_fr"].strip()

        with self._lock:
            rows = _read_jsonl(self.recordings_path)
            for i, row in enumerate(rows):
                if row["id"] == rec_id:
                    if changes.get("status") == "valide" and row["status"] == "a_transcrire":
                        raise ValueError(
                            f"{rec_id} : à transcrire d'abord, la validation "
                            f"est une relecture par une seconde personne"
                        )
                    rec = _from_dict(Recording, {**row, **changes})
                    _check_status(rec)
                    rows[i] = {**row, **asdict(rec)}
                    _write_jsonl(self.recordings_path, rows)
                    return rec
        raise KeyError(f"enregistrement inconnu : {rec_id}")

    def delete_recording(self, rec_id: str) -> None:
        """Pour réenregistrer : retire la ligne et le fichier audio."""
        with self._lock:
            rows = _read_jsonl(self.recordings_path)
            kept = [r for r in rows if r["id"] != rec_id]
            if len(kept) == len(rows):
                raise KeyError(f"enregistrement inconnu : {rec_id}")
            removed = next(r for r in rows if r["id"] == rec_id)
            (self.root / removed["audio"]).unlink(missing_ok=True)
            _write_jsonl(self.recordings_path, kept)


def _check_status(rec: Recording) -> None:
    if rec.status not in STATUSES:
        raise ValueError(f"statut inconnu : {rec.status}")
    if rec.status in ("transcrit", "valide") and not rec.transcript_bm:
        raise ValueError(f"{rec.id} : pas de transcription, statut {rec.status} impossible")
    if rec.status == "valide":
        if not rec.validator:
            raise ValueError(f"{rec.id} : indiquer qui valide la transcription")
        if rec.validator == rec.transcriber:
            raise ValueError(
                f"{rec.id} : la validation doit être faite par une autre "
                f"personne que la transcription"
            )


# --- suivi et export ----------------------------------------------------------

def progress(speakers: list[Speaker], recordings: list[Recording]) -> list[dict]:
    """Avancement de la collecte, critère par critère, face aux cibles du
    protocole. Chaque ligne : {critere, valeur, cible, ok}."""
    n = len(recordings)
    by_id = {s.id: s for s in speakers}
    rows: list[dict] = []

    def row(critere, valeur, cible, ok):
        rows.append({"critere": critere, "valeur": valeur, "cible": cible, "ok": bool(ok)})

    validated = sum(r.status == "valide" for r in recordings)
    row("énoncés enregistrés", str(n), f"{MIN_ITEMS}–{MAX_ITEMS}", n >= MIN_ITEMS)
    row("énoncés validés", str(validated), f"{MIN_ITEMS}–{MAX_ITEMS}",
        validated >= MIN_ITEMS)

    active = Counter(r.speaker for r in recordings)
    row("locuteurs enregistrés", str(len(active)), f"≥ {MIN_SPEAKERS}",
        len(active) >= MIN_SPEAKERS)
    if n:
        top, top_n = active.most_common(1)[0]
        row("part du locuteur le plus présent", f"{top_n / n:.0%} ({top})",
            f"≤ {MAX_SPEAKER_SHARE:.0%}", top_n / n <= MAX_SPEAKER_SHARE)

        genders = Counter(by_id[r.speaker].gender for r in recordings if r.speaker in by_id)
        f_share = genders["f"] / n
        row("part de voix féminines", f"{f_share:.0%}",
            f"{MIN_GENDER_SHARE:.0%}–{1 - MIN_GENDER_SHARE:.0%}",
            MIN_GENDER_SHARE <= f_share <= 1 - MIN_GENDER_SHARE)

        spont = sum(r.register == "spontane" for r in recordings) / n
        row("part de spontané", f"{spont:.0%}",
            f"{MIN_REGISTER_SHARE:.0%}–{1 - MIN_REGISTER_SHARE:.0%}",
            MIN_REGISTER_SHARE <= spont <= 1 - MIN_REGISTER_SHARE)

        in_range = sum(MIN_DURATION_S <= r.duration_s <= MAX_DURATION_S
                       for r in recordings) / n
        row(f"durées entre {MIN_DURATION_S:.0f} et {MAX_DURATION_S:.0f} s",
            f"{in_range:.0%}", "≥ 90%", in_range >= 0.9)

        conds = {r.conditions for r in recordings if r.conditions}
        row("conditions d'enregistrement", str(len(conds)) + (
            f" ({', '.join(sorted(conds))})" if conds else ""),
            f"≥ {MIN_CONDITIONS}", len(conds) >= MIN_CONDITIONS)

    transcribed = [r for r in recordings if r.status != "a_transcrire"]
    if transcribed:
        cs = sum(r.code_switching for r in transcribed) / len(transcribed)
        row("code-switching (énoncés transcrits)", f"{cs:.0%}",
            f"≥ {MIN_CODE_SWITCHING:.0%}", cs >= MIN_CODE_SWITCHING)
        with_fr = sum(bool(r.translation_fr) for r in transcribed)
        row("traductions françaises", f"{with_fr}/{len(transcribed)}",
            "toutes", with_fr == len(transcribed))
    return rows


def export_testset(collection: Collection, out_path: str | Path, *,
                   include_unvalidated: bool = False,
                   public_only: bool = False) -> int:
    """Écrit le jeu de test au format lu par `eval.dataset.load_testset`.

    `public_only` ne garde que les locuteurs ayant accepté la diffusion
    publique : c'est la version à publier, l'autre reste interne.
    """
    out_path = Path(out_path)
    speakers = {s.id: s for s in collection.speakers()}
    keep = {"valide", "transcrit"} if include_unvalidated else {"valide"}

    lines = []
    for rec in collection.recordings():
        if rec.status not in keep or not rec.transcript_bm:
            continue
        spk = speakers.get(rec.speaker)
        if spk is None or (public_only and not spk.public_release):
            continue
        audio = Path(os.path.relpath(collection.root / rec.audio, out_path.parent))
        lines.append({
            "id": rec.id,
            "audio": audio.as_posix(),
            "transcript_bm": rec.transcript_bm,
            "translation_fr": rec.translation_fr,
            "speaker": rec.speaker,
            "gender": spk.gender,
            "region": spk.region,
            "register": rec.register,
            "code_switching": rec.code_switching,
            "duration_s": rec.duration_s,
            "conditions": rec.conditions,
            "status": rec.status,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_path, lines)
    return len(lines)
