"""Nos énoncés de test ont-ils servi à entraîner un modèle publié ?

    python -m eval.recouvrement results/asr-*.json \
        --dataset RobotsMali/bam-asr-early --split train

Un modèle publié peut avoir été entraîné sur des phrases de notre partition
de test : son score serait alors optimiste. Soloni v0, l'ancêtre de toutes
les versions de Soloni, a été affiné sur bam-asr-early, tiré à 87 % de
Jeli-ASR.

On compte les références de test (celles des rapports de eval.baselines)
qui figurent telles quelles dans le jeu d'entraînement, après repli
orthographique (`fold`). Seulement les phrases d'au moins --min-mots mots :
les très courtes (« ɔwɔ », « naamu ») se retrouvent partout par hasard. Une
correspondance exacte d'une phrase longue trahit en revanche le même
enregistrement, ou sa transcription.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bambara_voice.normalize import fold
from eval.corpora import detect_columns, load_any

EXAMPLES = 5


def references(paths: list[Path]) -> dict[str, str]:
    """Références bambara des rapports, par identifiant d'énoncé."""
    refs: dict[str, str] = {}
    for path in paths:
        details = path if path.name.endswith(".details.jsonl") else path.with_suffix(".details.jsonl")
        for line in details.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                if row.get("ref_bm"):
                    refs[str(row["id"])] = row["ref_bm"]
    return refs


def train_texts(dataset: str, config: str | None, split: str, bm_column: str | None) -> set[str]:
    ds = load_any(dataset, config, split=split)
    cols = detect_columns(ds.features, bm=bm_column)
    print(f"{dataset} [{split}] : {len(ds)} lignes ; colonnes : {cols.describe()}")
    if cols.audio:
        ds = ds.remove_columns([cols.audio])  # le texte seul : rien à décoder
    if cols.nested:
        texts = (row[cols.bm] for row in ds[cols.nested])
    else:
        texts = ds[cols.bm]
    return {fold(t) for t in texts if t}


def overlap(refs: dict[str, str], texts: set[str], min_words: int) -> dict:
    long = {i: r for i, r in refs.items() if len(fold(r).split()) >= min_words}
    found = sorted(i for i, r in long.items() if fold(r) in texts)
    return {"references": len(refs), "longues": len(long), "retrouvees": len(found),
            "exemples": [long[i] for i in found[:EXAMPLES]],
            # Toutes longueurs : pour exclure, au plus prudent, tout ce qui a pu être vu.
            "ids_vus": sorted(i for i, r in refs.items() if fold(r) in texts)}


def to_markdown(res: dict, dataset: str, split: str, min_words: int) -> str:
    share = 100 * res["retrouvees"] / res["longues"] if res["longues"] else 0.0
    lines = [f"**{res['retrouvees']} sur {res['longues']}** références de test d'au moins "
             f"{min_words} mots ({share:.1f} %) figurent telles quelles dans {dataset} [{split}] "
             f"({res['references']} références en tout)."]
    lines += [f"- « {e} »" for e in res["exemples"]]
    lines += ["", f"Toutes longueurs confondues : {len(res['ids_vus'])} références."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rapports", nargs="+", type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--bm-column", default=None)
    p.add_argument("--min-mots", type=int, default=5)
    p.add_argument("--ids-vus", type=Path, default=None, metavar="FICHIER",
                   help="ajoute à ce fichier les identifiants des énoncés de test retrouvés, "
                        "toutes longueurs (pour eval.significance --exclure)")
    args = p.parse_args(argv)

    refs = references([r for r in args.rapports if r.suffix == ".json"])
    if not refs:
        raise SystemExit("aucune référence bambara dans ces rapports")
    texts = train_texts(args.dataset, args.config, args.split, args.bm_column)
    res = overlap(refs, texts, args.min_mots)
    print(to_markdown(res, args.dataset, args.split, args.min_mots))
    if args.ids_vus:
        with args.ids_vus.open("a", encoding="utf-8") as f:
            f.writelines(f"{i}\n" for i in res["ids_vus"])


if __name__ == "__main__":
    main()
