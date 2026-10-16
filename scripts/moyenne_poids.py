"""Moyenne des poids de deux versions d'un même modèle.

    python scripts/moyenne_poids.py --a nllb-bm2fr --b nllb-bm2fr-v2 --out nllb-bm2fr-moyenne

NLLB v2 a été entraîné à partir de la v1 : les deux sont deux points d'une
même trajectoire, et leur moyenne est un modèle valide, souvent meilleur
hors du domaine du second entraînement (ici Jeli-ASR), parce qu'elle garde
une part de ce que la v1 savait. Une minute de CPU, aucun entraînement :
le §9a du notebook la mesure sur le jeu maison à côté de la v1 et de la v2.

`--poids` est la part de B (0,5 par défaut : la moyenne simple).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

POIDS = "model.safetensors"
# État d'entraînement : propre à chaque entraînement, rien à moyenner.
_IGNORES = ("optimizer", "scheduler", "rng_state", "trainer_state", "training_args")


def moyenne(a: str | Path, b: str | Path, out: str | Path, poids: float = 0.5) -> Path:
    """Écrit dans `out` le modèle (1 - poids) · A + poids · B, avec la
    configuration et le tokenizer de B."""
    from safetensors.torch import load_file, save_file

    if not 0.0 <= poids <= 1.0:
        raise ValueError("poids attendu entre 0 et 1")
    a, b, out = Path(a), Path(b), Path(out)
    sa, sb = load_file(str(a / POIDS)), load_file(str(b / POIDS))
    if sa.keys() != sb.keys():
        raise ValueError(f"architectures différentes : {sorted(sa.keys() ^ sb.keys())[:5]}")
    for k in sa:
        if sa[k].shape != sb[k].shape:
            raise ValueError(f"{k} : formes différentes {tuple(sa[k].shape)} et "
                             f"{tuple(sb[k].shape)}")
    melange = {k: ((1 - poids) * sa[k].float() + poids * sb[k].float()).to(sa[k].dtype)
               for k in sa}

    out.mkdir(parents=True, exist_ok=True)
    for f in b.iterdir():
        if f.is_file() and f.name != POIDS and not f.name.startswith(_IGNORES):
            shutil.copy2(f, out / f.name)
    save_file(melange, str(out / POIDS), metadata={"format": "pt"})
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", type=Path, required=True, help="première version (ex. la v1)")
    p.add_argument("--b", type=Path, required=True, help="seconde version (ex. la v2)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--poids", type=float, default=0.5, help="part de B")
    args = p.parse_args(argv)
    print(f"Moyenne écrite dans {moyenne(args.a, args.b, args.out, args.poids)}")


if __name__ == "__main__":
    main()
