"""Le vocabulaire du bambara parlé, pour corriger l'orthographe des locuteurs.

    python scripts/vocabulaire.py --out data/vocabulaire_bambara.tsv

Les mots des transcriptions relues d'AfVoices (conversations du sud du Mali,
RobotsMali, CC-BY-4.0 ; scripts/donnees_robotsmali.py), en orthographe
standard, avec leur fréquence. `bambara_voice.orthographe` s'en sert pour
ramener un mot écrit « comme on l'entend en français » (« tchè », « gnogon »)
à sa forme standard (« cɛ », « ɲɔgɔn »). Seuls les mots vus au moins
`--min` fois sont gardés : un mot rare est plus souvent une coquille.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bambara_voice.normalize import normalize, strip_punctuation  # noqa: E402
import donnees_robotsmali as rm  # noqa: E402

_MOT = re.compile(r"^[^\W\d_]+(?:'[^\W\d_]*)?$")


def compter(textes: list[str]) -> Counter:
    mots: Counter = Counter()
    for t in textes:
        t = rm.nettoyer_afvoices(t)
        if t:
            mots.update(m for m in strip_punctuation(normalize(t)).split() if _MOT.match(m))
    return mots


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "vocabulaire_bambara.tsv")
    p.add_argument("--afvoices", type=Path, default=None,
                   help="manifeste déjà téléchargé (défaut : depuis GitHub)")
    p.add_argument("--min", type=int, default=3)
    args = p.parse_args(argv)

    source = args.afvoices
    if source is None:
        import urllib.request
        source = args.out.parent / "afvoices-train.jsonl"
        with urllib.request.urlopen(rm.AFVOICES, timeout=300) as r:
            source.write_bytes(r.read())
    mots = compter(rm.lire_afvoices(source))
    gardes = [(m, n) for m, n in mots.most_common() if n >= args.min]
    lignes = ["# Vocabulaire des transcriptions relues d'AfVoices (RobotsMali, CC-BY-4.0),",
              f"# {rm.AFVOICES}",
              "# mot\tfréquence"] + [f"{m}\t{n}" for m, n in gardes]
    args.out.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"{len(gardes)} mots (vus au moins {args.min} fois) → {args.out}")


if __name__ == "__main__":
    main()
