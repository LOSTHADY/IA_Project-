"""Ce qui change d'une exécution à l'autre, phrase par phrase.

    python -m eval.changes avant/*.json --apres apres/*.json

Apparie les rapports de eval.baselines qui portent sur le même corpus, la
même partition et le même maillon, puis compte, pour chaque sortie, les
phrases dont la sortie a changé, et parmi elles celles que la nouvelle
exécution a arrêtées sur la longueur maximale (--plafond).

Le décodage en faisceau est déterministe sur CPU : si l'on n'a changé
qu'une option, seules les phrases qu'elle touche doivent changer. Toute
autre différence vient d'ailleurs (version d'une bibliothèque, corpus
modifié) et fausserait l'attribution de l'écart à l'option. Les versions des
bibliothèques des deux exécutions sont donc affichées aussi.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Sorties comparées, et l'indicateur de coupure qui les accompagne.
OUTPUTS = {
    "mt": (("hyp_fr", "coupe_fr"), ("hyp_bm", "coupe_bm")),
    "asr": (("hyp_bm", None), ("hyp_fr", None), ("mt_depuis_asr", "coupe_fr")),
}
REFS = ("ref_bm", "ref_fr")
EXAMPLES, WIDTH = 3, 160


def _load(path: Path) -> tuple[dict, dict[str, dict]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    lines = path.with_suffix(".details.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line]
    return report, {str(r["id"]): r for r in rows}


def _key(report: dict) -> tuple:
    corpus = report.get("corpus", {})
    return corpus.get("id"), corpus.get("partition"), report.get("maillon")


def _short(text: str) -> str:
    text = text or ""
    return text if len(text) <= WIDTH else text[:WIDTH] + "…"


def compare(before: tuple[dict, dict], after: tuple[dict, dict]) -> dict:
    (rep_a, rows_a), (rep_b, rows_b) = before, after
    common = sorted(set(rows_a) & set(rows_b))
    other_refs = [i for i in common
                  if any(rows_a[i].get(k) != rows_b[i].get(k) for k in REFS)]
    out = {"avant": rep_a.get("nom"), "apres": rep_b.get("nom"), "n": len(common),
           "references_differentes": len(other_refs), "sorties": {}}
    env_a, env_b = rep_a.get("environnement", {}), rep_b.get("environnement", {})
    out["versions_differentes"] = {
        k: (env_a.get(k), env_b.get(k)) for k in sorted(set(env_a) | set(env_b))
        if k != "appareil" and env_a.get(k) != env_b.get(k)}

    for field, cut_key in OUTPUTS.get(rep_b.get("maillon"), ()):
        ids = [i for i in common if field in rows_a[i] and field in rows_b[i]]
        if not ids:
            continue
        changed = [i for i in ids if rows_a[i][field] != rows_b[i][field]]
        cut = [i for i in changed if cut_key and rows_b[i].get(cut_key)]
        not_cut = [i for i in changed if i not in set(cut)]

        def examples(sel):
            return [{"id": i, "avant": _short(rows_a[i][field]),
                     "apres": _short(rows_b[i][field])} for i in sel[:EXAMPLES]]

        out["sorties"][field] = {
            "n": len(ids), "changees": len(changed), "changees_coupees": len(cut),
            "changees_sans_coupure": len(not_cut),
            "coupees_apres": sum(bool(cut_key and rows_b[i].get(cut_key)) for i in ids),
            "exemples_coupees": examples(cut), "exemples_sans_coupure": examples(not_cut)}
    return out


def to_markdown(res: dict) -> str:
    lines = [f"### {res['avant']} → {res['apres']}", "",
             f"{res['n']} phrases communes."]
    if res["references_differentes"]:
        lines.append(f"**{res['references_differentes']} phrases ont le même identifiant mais "
                     f"une autre référence : ce ne sont pas les mêmes échantillons.**")
    if res["versions_differentes"]:
        lines.append("Versions différentes : " + " ; ".join(
            f"{k} {a} → {b}" for k, (a, b) in res["versions_differentes"].items()))
    else:
        lines.append("Mêmes versions de bibliothèques.")
    lines += ["", "| Sortie | Phrases | Changées | dont coupées après | changées sans coupure |",
              "|---|---|---|---|---|"]
    for field, s in res["sorties"].items():
        lines.append(f"| {field} | {s['n']} | {s['changees']} | {s['changees_coupees']} | "
                     f"{s['changees_sans_coupure']} |")
    for field, s in res["sorties"].items():
        for label, key in (("coupées", "exemples_coupees"),
                           ("changées sans coupure", "exemples_sans_coupure")):
            if s[key]:
                lines += ["", f"*{field}, {label} :*", ""]
                lines += [f"- {e['id']} — avant : « {e['avant']} » ; après : « {e['apres']} »"
                          for e in s[key]]
    return "\n".join(lines + [""])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("avant", nargs="+", type=Path)
    p.add_argument("--apres", nargs="+", type=Path, required=True)
    args = p.parse_args(argv)

    def load_all(paths):
        loaded = [_load(path) for path in paths if not path.name.endswith(".details.jsonl")]
        return {_key(rep): (rep, rows) for rep, rows in loaded}

    before, after = load_all(args.avant), load_all(args.apres)
    pairs = [k for k in after if k in before]
    if not pairs:
        raise SystemExit("aucun rapport apparié : ni même corpus, ni même partition, "
                         "ni même maillon")
    for key in pairs:
        print(to_markdown(compare(before[key], after[key])))


if __name__ == "__main__":
    main()
