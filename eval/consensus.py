"""Mieux utiliser les traducteurs qu'on a déjà : le consensus.

    python -m eval.consensus --en-service RAPPORT.json --rapport "NLLB v2=R2.json" ... --out DOSSIER

Sept traducteurs ont déjà traduit ce que Soloni a entendu de tes phrases
(NLLB v1, v2, v3, leur moyenne, les professeurs) : chacun seul plafonne vers
27-28 de chrF++, mais ils ne se trompent pas tous sur les mêmes phrases.
Pour chaque phrase, le consensus garde la traduction qui ressemble le plus
à celles des autres (décodage à risque bayésien minimal, MBR : chrF++ moyen
contre les autres candidats) ; une traduction isolée, propre à un seul
modèle, a plus de chances d'être une erreur.

Rien n'est entraîné, et tout se calcule sur les rapports déjà écrits (sans
GPU, en quelques secondes). L'ensemble de traducteurs se choisit sur le dev ;
le consensus n'est retenu que s'il fait mieux que le traducteur en service
sur le test (eval.promotion).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .promotion import comparer, score


def _details(rapport: str | Path) -> list[dict]:
    path = Path(rapport).with_suffix(".details.jsonl")
    return [json.loads(ligne) for ligne in path.read_text(encoding="utf-8").splitlines()
            if ligne.strip()]


def choisir(candidats: list[str]) -> str:
    """La traduction la plus proche des autres (chrF++ moyen) ; à égalité,
    la première."""
    from sacrebleu.metrics import CHRF

    chrf = CHRF(word_order=2)
    uniques = list(dict.fromkeys(c for c in candidats if c.strip()))
    if len(uniques) <= 1:
        return uniques[0] if uniques else ""

    def accord(c):
        return sum(chrf.sentence_score(c, [autre]).score if autre.strip() else 0.0
                   for autre in candidats) - 100.0

    return max(uniques, key=accord)


def consensus(rapports: dict[str, str | Path], noms: list[str], base: str | Path,
              out: Path) -> Path:
    """Rapport dérivé de `base` (le traducteur en service : références,
    partitions, locuteurs) où chaque phrase prend le consensus des
    traducteurs `noms` ; seules les phrases traduites par tous sont gardées."""
    par_nom = {n: {r["item"]: r.get("source_fr") or "" for r in _details(rapports[n])}
               for n in noms}
    lignes = []
    for r in _details(base):
        traductions = [par_nom[n].get(r["item"]) for n in noms]
        if None not in traductions:
            lignes.append({**r, "source_fr": choisir(traductions)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"nom": "consensus : " + ", ".join(noms),
                               "architecture": "cascade", "traducteurs": noms},
                              ensure_ascii=False), encoding="utf-8")
    out.with_suffix(".details.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lignes), encoding="utf-8")
    return out


def ensembles(noms: list[str], en_service: str) -> dict[str, list[str]]:
    """Ensembles essayés : tous, et le traducteur en service avec chacun des
    autres pris par trois (les plus petits ensembles se déploient mieux)."""
    autres = [n for n in noms if n != en_service]
    out = {"tous": [en_service] + autres}
    for i, a in enumerate(autres):
        for b in autres[i + 1:]:
            out[f"{en_service} + {a} + {b}"] = [en_service, a, b]
    return out


def mesurer(rapports: dict[str, str | Path], en_service: str, out: Path,
            n_boot: int = 1000) -> dict:
    """Choisit l'ensemble sur le dev, puis le compare au traducteur en
    service sur le test."""
    essais = {}
    for nom, noms in ensembles(list(rapports), en_service).items():
        chemin = consensus(rapports, noms, rapports[en_service],
                           out / f"consensus-{len(essais):02d}.json")
        essais[nom] = (chemin, score(chemin, "dev"))
    meilleur = max(essais, key=lambda n: essais[n][1])
    c = comparer(essais[meilleur][0], rapports[en_service], "test", n_boot=n_boot)
    return {"promu": c["meilleur"], "ensemble": meilleur, "traducteurs": ensembles(
        list(rapports), en_service)[meilleur], "dev": {n: s for n, (_, s) in essais.items()},
        "dev_en_service": score(rapports[en_service], "dev"), "comparaison": c,
        "raisons": [] if c["meilleur"] else [
            f"pas meilleur de façon établie sur le test : écart {c['ecart']:+.1f} "
            f"[{c['ic'][0]:.1f} ; {c['ic'][1]:.1f}]"]}


def to_markdown(d: dict, en_service: str) -> str:
    c = d["comparaison"]
    lignes = [f"Meilleur ensemble sur le dev : {d['ensemble']} ({d['dev'][d['ensemble']]:.1f}, "
              f"contre {d['dev_en_service']:.1f} pour {en_service} seul)",
              f"Test ({c['n']} énoncés) : {en_service} {c['titulaire']:.1f}, consensus "
              f"{c['candidat']:.1f}, écart {c['ecart']:+.1f} [{c['ic'][0]:.1f} ; {c['ic'][1]:.1f}]",
              "**Le consensus fait mieux.**" if d["promu"] else
              "**Le consensus ne fait pas mieux** : " + " ; ".join(d["raisons"])]
    return "\n".join(f"- {x}" for x in lignes)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--en-service", required=True, help="NOM=RAPPORT du traducteur en service")
    p.add_argument("--rapport", action="append", default=[], help="NOM=RAPPORT (répétable)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    nom_service, _, chemin = args.en_service.partition("=")
    rapports = {nom_service: chemin, **dict(r.split("=", 1) for r in args.rapport)}
    print(to_markdown(mesurer(rapports, nom_service, args.out), nom_service))


if __name__ == "__main__":
    main()
