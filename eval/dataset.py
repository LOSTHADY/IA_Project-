"""Chargement et validation du jeu de test."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Taille et diversité minimales pour qu'un chiffre soit défendable : en dessous
# de 200 items, l'intervalle de confiance sur le WER est trop large pour
# conclure ; en dessous de 8 locuteurs, le WER mesure surtout des voix
# particulières.
MIN_ITEMS, MAX_ITEMS = 200, 500
MIN_SPEAKERS = 8


@dataclass
class TestItem:
    id: str
    audio: str
    transcript_bm: str
    translation_fr: str = ""
    expected_reply_bm: str = ""
    speaker: str = ""
    gender: str = ""
    region: str = ""
    register: str = ""
    code_switching: bool = False
    duration_s: float = 0.0
    extra: dict = field(default_factory=dict)

    def audio_path(self, root: Path) -> Path:
        p = Path(self.audio)
        return p if p.is_absolute() else root / p


_KNOWN = set(TestItem.__annotations__) - {"extra"}


def load_testset(path: str | Path) -> list[TestItem]:
    """Lit un JSONL de test. Les champs inconnus sont conservés dans `extra`."""
    path = Path(path)
    items: list[TestItem] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} — JSON invalide : {exc}") from exc
        for required in ("id", "audio", "transcript_bm"):
            if not raw.get(required):
                raise ValueError(f"{path}:{lineno} — champ obligatoire manquant : {required}")
        extra = {k: v for k, v in raw.items() if k not in _KNOWN}
        known = {k: v for k, v in raw.items() if k in _KNOWN}
        items.append(TestItem(**known, extra=extra))

    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"{path} — identifiants dupliqués : {sorted(dupes)}")
    return items


def describe(items: list[TestItem]) -> dict:
    """Statistiques de composition, à reporter dans le mémoire."""
    if not items:
        return {"n": 0}
    speakers = {i.speaker for i in items if i.speaker}
    total = sum(i.duration_s for i in items)
    return {
        "n": len(items),
        "duree_totale_min": round(total / 60, 1),
        "duree_moyenne_s": round(total / len(items), 2) if total else 0.0,
        "locuteurs": len(speakers),
        "avec_traduction_fr": sum(bool(i.translation_fr) for i in items),
        "part_spontane": round(
            sum(i.register == "spontane" for i in items) / len(items), 3
        ),
        "part_code_switching": round(
            sum(i.code_switching for i in items) / len(items), 3
        ),
        "regions": sorted({i.region for i in items if i.region}),
    }
