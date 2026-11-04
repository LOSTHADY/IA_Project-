"""La campagne de lecture : la liste des phrases à lire, et où on en est.

    python scripts/campagne.py liste --atelier ATELIER [--conversations 22000] [--traduites 8000]
    python scripts/campagne.py bilan --atelier ATELIER

`liste` complète `lecture.jsonl` (les phrases de l'onglet « Lire ») pour une
campagne de plusieurs dizaines de lecteurs (docs/CAMPAGNE.md) :

- **des phrases de conversation** : les transcriptions relues d'AfVoices
  (RobotsMali, CC-BY-4.0), du bambara parlé de tous les jours, en
  orthographe standard ;
- **des phrases avec leur français** : Bayelemabaga, pour que la même
  campagne serve un jour à la traduction et au bout-en-bout (audio, bambara
  et français).

Les phrases déjà dans la liste, et donc déjà lues, sont gardées telles
quelles. Celles qu'on ajoute ont de 4 à 14 mots, sans chiffres (qui se
disent de plusieurs façons), sans texte religieux, sans doublon. L'atelier
distribue ensuite les phrases : chaque lecteur dans son propre ordre, trois
lecteurs au plus par phrase tant qu'il en reste, et les lecteurs réservés
au test des voix nouvelles sur les seules phrases du test
(`Atelier.prochaine_lecture`).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bambara_voice.normalize import normalize  # noqa: E402
from eval.atelier import _RELIGIEUX, PALIERS_CAMPAGNE, Atelier  # noqa: E402
import donnees_robotsmali as rm  # noqa: E402


def lisible(texte: str, min_mots: int = 4, max_mots: int = 14) -> str | None:
    """La phrase telle qu'à lire, ou None si elle ne convient pas."""
    t = normalize(texte or "", lower=False)
    if (not t or re.search(r"\d", t) or _RELIGIEUX.search(t) or "[" in t or "<" in t
            or not min_mots <= len(t.split()) <= max_mots):
        return None
    return t


def conversations(manifeste: Path, n: int, vues: set[str], seed: int = 0) -> list[dict]:
    out = []
    textes = rm.lire_afvoices(manifeste)
    random.Random(seed).shuffle(textes)
    for brut in textes:
        t = rm.nettoyer_afvoices(brut)
        t = lisible(t) if t else None
        if t and t.lower() not in vues:
            vues.add(t.lower())
            out.append({"bm": t, "source": "afvoices"})
            if len(out) >= n:
                break
    return out


def traduites(dataset: str, n: int, vues: set[str], seed: int = 0) -> list[dict]:
    from eval.corpora import detect_columns, load_any

    ds = load_any(dataset, split="train")
    cols = detect_columns(ds.features, need_fr=True)
    if cols.audio:
        ds = ds.remove_columns([cols.audio])
    lignes = ([(r[cols.bm], r[cols.fr]) for r in ds[cols.nested]] if cols.nested
              else list(zip(ds[cols.bm], ds[cols.fr])))
    random.Random(seed).shuffle(lignes)
    out = []
    for bm, fr in lignes:
        t = lisible(bm)
        if t and (fr or "").strip() and t.lower() not in vues:
            vues.add(t.lower())
            out.append({"bm": t, "fr": fr.strip(), "source": dataset})
            if len(out) >= n:
                break
    return out


def liste(atelier: Path, afvoices: Path | None, n_conversations: int, n_traduites: int,
          dataset: str = "RobotsMaliAI/bayelemabaga") -> dict[str, int]:
    """Ajoute les phrases de la campagne à `atelier/lecture.jsonl` ; renvoie
    le nombre de phrases par source."""
    chemin = atelier / "lecture.jsonl"
    deja = Atelier(atelier).phrases_a_lire()
    vues = {p["bm"].lower() for p in deja}
    if afvoices is None:
        afvoices = atelier / "afvoices-train.jsonl"
        if not afvoices.exists():
            with urllib.request.urlopen(rm.AFVOICES, timeout=300) as r:
                afvoices.write_bytes(r.read())
    nouvelles = conversations(afvoices, n_conversations, vues)
    if n_traduites:
        nouvelles += traduites(dataset, n_traduites, vues)
    random.Random(1).shuffle(nouvelles)
    deja_ids = {p["id"] for p in deja}
    k = sum(i.startswith("lc-") for i in deja_ids)
    for p in nouvelles:
        k += 1
        p["id"] = f"lc-{k:05d}"
    if chemin.exists():
        chemin.with_name("lecture.avant-campagne.jsonl").write_bytes(chemin.read_bytes())
    atelier.mkdir(parents=True, exist_ok=True)
    chemin.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in deja + nouvelles),
                      encoding="utf-8")
    comptes = {"deja": len(deja)}
    for p in nouvelles:
        comptes[p["source"]] = comptes.get(p["source"], 0) + 1
    return comptes


def bilan_markdown(c: dict) -> str:
    lignes = [f"- {c['heures']} h lues par {c['lecteurs']} lecteurs ({c['femmes']} femmes), "
              f"dont {c['reserves']} réservés au test des voix nouvelles ({c['heures_reservees']} h)",
              f"- {c['telephones']} téléphones différents ; lieux : "
              + (", ".join(f"{k} {v}" for k, v in c["lieux"].items()) or "aucun encore"),
              f"- {c['phrases_lues']} phrases différentes lues, sur {c['phrases_disponibles']}"]
    for nom, h, n in PALIERS_CAMPAGNE:
        fait = "✅" if c["heures"] >= h and c["lecteurs"] >= n else "…"
        lignes.append(f"- {fait} palier {nom} : {h} h et {n} lecteurs "
                      f"({min(100, round(100 * c['heures'] / h))} % des heures, "
                      f"{min(100, round(100 * c['lecteurs'] / n))} % des lecteurs)")
    return "\n".join(lignes)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="commande", required=True)
    l = sub.add_parser("liste")
    l.add_argument("--atelier", type=Path, required=True)
    l.add_argument("--afvoices", type=Path, default=None,
                   help="manifeste AfVoices déjà téléchargé (défaut : depuis GitHub)")
    l.add_argument("--conversations", type=int, default=22_000)
    l.add_argument("--traduites", type=int, default=8_000)
    l.add_argument("--dataset", default="RobotsMaliAI/bayelemabaga")
    b = sub.add_parser("bilan")
    b.add_argument("--atelier", type=Path, required=True)
    args = p.parse_args(argv)
    if args.commande == "liste":
        comptes = liste(args.atelier, args.afvoices, args.conversations, args.traduites,
                        args.dataset)
        print(", ".join(f"{k} : {v}" for k, v in comptes.items())
              + f" → {args.atelier / 'lecture.jsonl'}")
    else:
        print(bilan_markdown(Atelier(args.atelier).bilan_campagne()))


if __name__ == "__main__":
    main()
