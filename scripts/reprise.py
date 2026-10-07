"""Reprise d'un entraînement : le dernier checkpoint complet.

Un checkpoint peut être incomplet sur Google Drive : copie inachevée quand la
session Colab s'arrête, ou Drive plein. Le Trainer reprend alors quand même,
sans prévenir, avec un optimiseur et un plan de taux d'apprentissage neufs
s'il manque optimizer.pt ou scheduler.pt : le préchauffage recommence et la
fin de l'entraînement se fait au taux maximal. Constaté le 7 octobre 2026 :
reprise de Whisper à l'étape 3000, WER de validation 49 % au lieu de 44 %.
"""

from __future__ import annotations

from pathlib import Path

REQUIRED = ("trainer_state.json", "optimizer.pt", "scheduler.pt")


def last_complete_checkpoint(output: str | Path) -> str | None:
    """Le checkpoint le plus avancé qui contient poids, optimiseur, plan de
    taux d'apprentissage et état du Trainer ; None s'il n'y en a aucun."""
    out = Path(output)
    if not out.is_dir():
        return None
    numbered = [p for p in out.glob("checkpoint-*")
                if p.is_dir() and p.name.split("-")[-1].isdigit()]
    for ck in sorted(numbered, key=lambda p: int(p.name.split("-")[-1]), reverse=True):
        missing = [f for f in REQUIRED if not (ck / f).is_file()]
        if not any(ck.glob("*.safetensors")):
            missing.append("poids")
        if not missing:
            return str(ck)
        print(f"Checkpoint incomplet, ignoré : {ck.name} (manque : {', '.join(missing)})")
    return None
