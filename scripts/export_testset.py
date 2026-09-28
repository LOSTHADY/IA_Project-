"""Exporte les énoncés validés de la collecte vers le jeu de test.

    python scripts/export_testset.py                       # -> data/testset/testset.jsonl
    python scripts/export_testset.py --public-only --out data/testset/testset.public.jsonl
    python scripts/prepare_testset.py data/testset/testset.jsonl   # puis contrôler

Par défaut, seuls les énoncés **validés** (relus par une seconde personne)
sont exportés : une transcription non relue ne vaut pas référence.
`--public-only` ne garde que les locuteurs ayant accepté la diffusion
publique — c'est la seule version publiable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.collection import Collection, export_testset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT / "data" / "testset",
                   help="dossier de collecte")
    p.add_argument("--out", type=Path, default=None,
                   help="fichier de sortie (défaut : <root>/testset.jsonl)")
    p.add_argument("--include-unvalidated", action="store_true",
                   help="inclure les énoncés transcrits mais pas encore validés "
                        "(pour tester la chaîne, jamais pour les chiffres du mémoire)")
    p.add_argument("--public-only", action="store_true",
                   help="seulement les locuteurs ayant accepté la diffusion publique")
    args = p.parse_args()

    col = Collection(args.root)
    out = args.out or args.root / "testset.jsonl"
    n = export_testset(col, out, include_unvalidated=args.include_unvalidated,
                       public_only=args.public_only)
    total = len(col.recordings())
    print(f"{n} énoncés exportés sur {total} enregistrés -> {out}")
    if n < total:
        why = "à transcrire ou à valider dans app/collect_app.py"
        if args.public_only:
            why += ", ou de locuteurs sans accord de diffusion publique"
        print(f"Les autres sont {why}.")


if __name__ == "__main__":
    main()
