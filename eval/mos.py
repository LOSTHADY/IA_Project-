"""Dépouillement des grilles MOS de la synthèse vocale.

    python -m eval.mos eval/results/tts-mms-tts-bam-*/mos*.csv

`eval.baselines tts` écrit une grille `mos.csv` par système évalué. Chaque
auditeur en remplit une copie (`mos-aminata.csv`, `mos-issa.csv`...) ou
l'on ajoute ses lignes dans le même fichier, colonne `auditeur` renseignée.
Les grilles d'un même dossier sont regroupées : un dossier = un système.

Résultat : MOS moyen avec IC à 95 % par bootstrap sur les *phrases*. Les
notes d'une même phrase ne sont pas indépendantes : rééchantillonner les
notes une à une donnerait un intervalle trop étroit. S'y ajoutent la
moyenne par auditeur, pour repérer un auditeur nettement plus sévère ou
plus indulgent que les autres, et un avertissement en dessous de trois
auditeurs.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_LISTENERS = 3


def read_grid(path: Path) -> list[dict]:
    """Notes valides d'une grille. Une note vide est ignorée (phrase non
    écoutée) ; une note hors de 1 à 5 est une erreur de saisie à corriger."""
    rows = []
    with path.open(encoding="utf-8", newline="") as fh:
        for lineno, row in enumerate(csv.DictReader(fh), 2):
            raw = (row.get("note_1_a_5") or "").strip().replace(",", ".")
            if not raw:
                continue
            try:
                note = float(raw)
            except ValueError:
                raise ValueError(f"{path}:{lineno} — note illisible : {raw!r}") from None
            if not 1 <= note <= 5:
                raise ValueError(f"{path}:{lineno} — note hors de 1 à 5 : {note}")
            listener = (row.get("auditeur") or "").strip() or path.stem
            rows.append({"phrase": row["id"], "auditeur": listener, "note": note})
    return rows


def summarize(rows: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    by_sentence: dict[str, list[float]] = defaultdict(list)
    by_listener: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_sentence[r["phrase"]].append(r["note"])
        by_listener[r["auditeur"]].append(r["note"])
    if not rows:
        raise ValueError("aucune note")

    sentences = list(by_sentence.values())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(sentences), size=len(sentences))
        boots.append(np.mean([n for i in picked for n in sentences[i]]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mos": float(np.mean([r["note"] for r in rows])),
        "ic": (float(lo), float(hi)),
        "notes": len(rows),
        "phrases": len(by_sentence),
        "auditeurs": {k: round(float(np.mean(v)), 2) for k, v in sorted(by_listener.items())},
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("grids", nargs="+", type=Path)
    args = p.parse_args(argv)

    systems: dict[str, list[dict]] = defaultdict(list)
    for grid in args.grids:
        systems[grid.parent.name].extend(read_grid(grid))

    print("| Système | MOS [IC 95 %] | Phrases | Notes | Auditeurs (moyenne) |")
    print("|---|---|---|---|---|")
    warnings = []
    for name, rows in systems.items():
        s = summarize(rows)
        listeners = ", ".join(f"{k} {v:.2f}" for k, v in s["auditeurs"].items())
        print(f"| {name} | {s['mos']:.2f} [{s['ic'][0]:.2f} ; {s['ic'][1]:.2f}] | "
              f"{s['phrases']} | {s['notes']} | {listeners} |")
        if len(s["auditeurs"]) < MIN_LISTENERS:
            warnings.append(f"{name} : {len(s['auditeurs'])} auditeur(s), "
                            f"en viser au moins {MIN_LISTENERS}")
    for w in warnings:
        print(f"\nAttention — {w}.")


if __name__ == "__main__":
    main()
