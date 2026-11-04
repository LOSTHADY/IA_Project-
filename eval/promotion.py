"""Un modèle ne remplace le précédent que s'il est meilleur, mesure à l'appui.

    python -m eval.promotion --candidat C.json --titulaire T.json \\
        [--controle C_bayel.json T_bayel.json "chrF++ bm→fr"] [--sortie decision.json]

Le 16 octobre, Soloni + NLLB v2 a remplacé Whisper + NLLB v1 parce qu'ils
gagnaient sur Jeli-ASR ; au micro, l'assistant était moins bon. D'où la
règle, appliquée à chaque nouveau modèle ou assemblage :

1. **Choisir sur le dev.** Entre plusieurs candidats (assemblages, checkpoints),
   le meilleur chrF++ sur la partition « dev » du jeu maison
   (eval.atelier.partition) : de la parole quotidienne, pas des contes.
2. **Décider sur le test.** Le candidat retenu remplace le titulaire seulement
   s'il est meilleur sur la partition « test », que le choix n'a jamais vue :
   écart de chrF++ (français compris depuis l'audio) dont l'IC à 95 % du
   bootstrap apparié est entièrement au-dessus de zéro. « Peut-être
   meilleur » ne suffit pas : à égalité, le titulaire reste.
3. **Ne rien casser ailleurs.** Sur les tests écrits (Bayelemabaga, MAFAND),
   contrôles de non-régression : refusé si le candidat y est moins bon de
   façon établie et de plus d'un point (TOLERANCE). Ces tests ne choisissent
   jamais : leur domaine n'est pas celui de l'assistant.

Le sens français → bambara, que le jeu maison ne mesure pas (pas de bambara
écrit), se décide par les jugements à l'aveugle de l'atelier :
`verdict_jugements`, test du signe.

`retenu.json` (sur Drive) garde l'assemblage en service et l'historique des
décisions, preuves comprises : le notebook le lit pour la démo et l'export.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .significance import LOWER_IS_BETTER, Series, _details, _series, bootstrap, load_series

TOLERANCE = 1.0   # points de chrF++ perdus sur un contrôle écrit, au-delà : refus
ALPHA = 0.05


# --- jeu maison -----------------------------------------------------------------

def series_maison(rapport: str | Path, partition: str | None = None) -> Series:
    """chrF++ du français compris depuis l'audio (rapport de eval.run_eval),
    restreint à une partition du jeu maison."""
    rows = _details(Path(rapport))
    if partition:
        rows = [r for r in rows if r.get("partition") == partition]
    s = _series("chrF++ (depuis ASR)", "chrf", rows, "item", "source_fr", "ref_translation_fr")
    if s is None:
        raise ValueError(f"{rapport} : aucun énoncé{' ' + partition if partition else ''} "
                         f"avec une traduction de référence")
    return s


def score(rapport: str | Path, partition: str | None = None) -> float:
    return series_maison(rapport, partition).value


def choisir(rapports: dict[str, str | Path], partition: str = "dev") -> str:
    """Le nom du meilleur candidat sur le dev."""
    if not rapports:
        raise ValueError("aucun candidat")
    return max(rapports, key=lambda nom: score(rapports[nom], partition))


def comparer(candidat: str | Path, titulaire: str | Path, partition: str = "test",
             n_boot: int = 1000, seed: int = 0) -> dict:
    """Candidat contre titulaire sur les mêmes énoncés de la partition."""
    s_tit, s_cand = series_maison(titulaire, partition), series_maison(candidat, partition)
    res = bootstrap([s_tit, s_cand], n_boot=n_boot, seed=seed)
    tit, cand = res["systemes"]
    lo, hi = cand["ic_ecart"]
    return {"partition": partition, "n": res["n"], "unite": res["unite"],
            "titulaire": tit["score"], "candidat": cand["score"],
            "ecart": cand["ecart"], "ic": [lo, hi], "p": cand["p"],
            "meilleur": lo > 0}


def controle(candidat: str | Path, titulaire: str | Path, metrique: str,
             n_boot: int = 1000, seed: int = 0, tolerance: float = TOLERANCE) -> dict:
    """Non-régression sur un test écrit (rapports de eval.baselines mt) :
    `regression` si le candidat y est moins bon de façon établie et de plus
    de `tolerance` points."""
    s_tit = load_series(titulaire)[1].get(metrique)
    s_cand = load_series(candidat)[1].get(metrique)
    if s_tit is None or s_cand is None:
        raise ValueError(f"« {metrique} » absente de {candidat} ou de {titulaire}")
    res = bootstrap([s_tit, s_cand], n_boot=n_boot, seed=seed)
    cand = res["systemes"][1]
    lo, hi = cand["ic_ecart"]
    signe = -1 if metrique in LOWER_IS_BETTER else 1
    pire_etabli = (hi < 0) if signe > 0 else (lo > 0)
    return {"metrique": metrique, "n": res["n"], "titulaire": res["systemes"][0]["score"],
            "candidat": cand["score"], "ecart": cand["ecart"], "ic": [lo, hi],
            "regression": pire_etabli and signe * cand["ecart"] < -tolerance}


def verdict(comparaison: dict, controles: list[dict] | None = None) -> dict:
    """Promu seulement si meilleur sur le test maison et sans régression."""
    raisons = []
    if not comparaison["meilleur"]:
        lo, hi = comparaison["ic"]
        raisons.append(f"pas meilleur de façon établie sur le test maison : écart "
                       f"{comparaison['ecart']:+.1f} chrF++ [{lo:.1f} ; {hi:.1f}]")
    for c in controles or []:
        if c["regression"]:
            raisons.append(f"régression sur {c['metrique']} : {c['ecart']:+.1f} "
                           f"[{c['ic'][0]:.1f} ; {c['ic'][1]:.1f}]")
    return {"promu": not raisons, "raisons": raisons, "comparaison": comparaison,
            "controles": controles or []}


# --- jugements humains (français → bambara) -------------------------------------

def verdict_jugements(resultats: dict, candidat: str, titulaire: str,
                      alpha: float = ALPHA) -> dict:
    """Résultats d'un lot de l'atelier (Atelier.resultats) : le candidat est
    promu s'il est préféré plus souvent, test du signe sous `alpha`."""
    prefs = resultats.get("preferences", {})
    gagnees, perdues = prefs.get(candidat, 0), prefs.get(titulaire, 0)
    p = resultats.get("p_signe", 1.0)
    promu = gagnees > perdues and p < alpha
    raisons = [] if promu else [
        f"jugements : {candidat} préféré {gagnees} fois, {titulaire} {perdues} fois "
        f"(p = {p:.3f}{', pas assez de jugements' if gagnees + perdues < 6 else ''})"]
    return {"promu": promu, "raisons": raisons, "gagnees": gagnees, "perdues": perdues,
            "p": p, "jugements": resultats.get("jugements", 0)}


# --- assemblage en service -------------------------------------------------------

def lire_retenu(path: str | Path, defaut: dict) -> dict:
    """L'assemblage en service : `defaut` tant qu'aucune décision n'a été prise."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {**defaut, "historique": []}


def ecrire_retenu(path: str | Path, retenu: dict, maillon: str, candidat: dict,
                  decision: dict) -> dict:
    """Consigne la décision ; si le candidat est promu, il entre en service
    (seules ses clés changent : un nouveau bm2fr ne touche ni la
    reconnaissance ni le fr2bm)."""
    nouveau = {**retenu, **(candidat if decision["promu"] else {})}
    entree = {"maillon": maillon, "candidat": candidat, "promu": decision["promu"],
              "raisons": decision["raisons"],
              "preuves": {k: v for k, v in decision.items() if k not in ("promu", "raisons")}}
    historique = list(retenu.get("historique", []))
    # Une relance du notebook refait la même décision sur les mêmes mesures :
    # on ne la consigne qu'une fois.
    if not historique or {k: v for k, v in historique[-1].items() if k != "date"} != entree:
        historique.append({"date": datetime.now().isoformat(timespec="seconds"), **entree})
    nouveau["historique"] = historique
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(nouveau, ensure_ascii=False, indent=1), encoding="utf-8")
    return nouveau


def to_markdown(decision: dict, candidat: str = "candidat", titulaire: str = "titulaire") -> str:
    c = decision.get("comparaison")
    lignes = []
    if c:
        lo, hi = c["ic"]
        lignes.append(f"Test maison ({c['n']} énoncés, tirage par {c['unite']}) : "
                      f"{titulaire} {c['titulaire']:.1f}, {candidat} {c['candidat']:.1f}, "
                      f"écart {c['ecart']:+.1f} [{lo:.1f} ; {hi:.1f}], p = {c['p']:.3f}")
    for k in decision.get("controles", []):
        lignes.append(f"Contrôle {k['metrique']} ({k['n']}) : {k['titulaire']:.1f} → "
                      f"{k['candidat']:.1f}, écart {k['ecart']:+.1f} "
                      f"[{k['ic'][0]:.1f} ; {k['ic'][1]:.1f}]"
                      + (" — régression" if k["regression"] else ""))
    if decision["promu"]:
        lignes.append(f"**Promu : {candidat} remplace {titulaire}.**")
    else:
        lignes.append(f"**Refusé : {titulaire} reste en service.** "
                      + " ; ".join(decision["raisons"]))
    return "\n".join(f"- {line}" for line in lignes)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidat", type=Path, required=True, help="rapport eval.run_eval")
    p.add_argument("--titulaire", type=Path, required=True, help="rapport eval.run_eval")
    p.add_argument("--partition", default="test")
    p.add_argument("--controle", nargs=3, action="append", default=[],
                   metavar=("CANDIDAT", "TITULAIRE", "METRIQUE"),
                   help="rapports eval.baselines mt, ex. \"chrF++ bm→fr\" (répétable)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--sortie", type=Path, default=None, help="décision en JSON")
    args = p.parse_args(argv)

    decision = verdict(comparer(args.candidat, args.titulaire, args.partition, args.n_boot),
                       [controle(c, t, m, args.n_boot) for c, t, m in args.controle])
    print(to_markdown(decision))
    if args.sortie:
        args.sortie.write_text(json.dumps(decision, ensure_ascii=False, indent=1),
                               encoding="utf-8")


if __name__ == "__main__":
    main()
