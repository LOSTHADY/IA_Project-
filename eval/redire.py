"""Une phrase apprise est-elle comprise la fois suivante ? La mémoire, mesurée.

    python -m eval.redire --rapport RAPPORT.json --prises dire.jsonl --out DOSSIER \\
        [--memoire memoire-atelier.jsonl]

La mémoire (bambara_voice.memoire) retient ce que la reconnaissance a entendu
et ce que la phrase voulait dire. Dans le jeu maison, la plupart des phrases
ont été dites plusieurs fois (deux locuteurs, plusieurs interlocuteurs) : la
première prise de chaque phrase entre dans la mémoire, les suivantes disent
si elle est comprise « la fois suivante ». Rien n'est réentraîné, et tout se
calcule sur les transcriptions déjà faites (rapport eval.run_eval) : sans
GPU, en quelques secondes.

Deux mesures, sur les mêmes énoncés, avec et sans mémoire (chrF++ contre la
phrase affichée) :

1. **redire** : les prises suivantes des phrases apprises, le gain attendu ;
2. **jamais dit** : les phrases du test quand la mémoire ne contient que
   celles du dev, le risque (une phrase inconnue prise pour une autre).

Le seuil de similarité se choisit sur le dev (redire), puis vaut pour le
test. La mémoire entre en service (eval.promotion, retenu.json) si elle fait
mieux sur « redire » au test (IC à 95 % au-dessus de zéro) sans perdre plus
d'un point, de façon établie, sur « jamais dit ».

`--memoire` écrit la mémoire à déployer : toutes les prises de l'atelier.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bambara_voice.memoire import Memoire

from .promotion import TOLERANCE, comparer, score

SEUILS = (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def _lignes(path: Path) -> list[dict]:
    return [json.loads(ligne) for ligne in path.read_text(encoding="utf-8").splitlines()
            if ligne.strip()]


def prises_du_rapport(rapport: Path, prises: Path) -> list[dict]:
    """Les énoncés du rapport, dans l'ordre d'enregistrement, avec leur phrase."""
    phrase_de = {p["id"]: p["prompt_id"] for p in _lignes(prises) if p.get("prompt_id")}
    rows = [{**r, "prompt_id": phrase_de[r["item"]]} for r in
            _lignes(rapport.with_suffix(".details.jsonl"))
            if r.get("item") in phrase_de and r.get("ref_translation_fr")]
    return sorted(rows, key=lambda r: str(r["item"]))


def _memoire(rows: list[dict], seuil: float) -> Memoire:
    m = Memoire(seuil=seuil)
    for r in rows:
        if (r.get("source_bm") or "").strip():
            m.apprendre(r["source_bm"], r["ref_translation_fr"], f"atelier:{r['prompt_id']}")
    return m


def _comprendre(rows: list[dict], memoire: Memoire) -> tuple[list[dict], int, int]:
    """Les énoncés compris avec la mémoire ; renvoie aussi le nombre de
    souvenirs utilisés, et de souvenirs de la bonne phrase."""
    out, utilises, justes = [], 0, 0
    for r in rows:
        trouve = memoire.chercher(r.get("source_bm") or "")
        if trouve:
            utilises += 1
            justes += trouve[0].source == f"atelier:{r['prompt_id']}"
        out.append({**r, "source_fr": trouve[0].fr if trouve else (r.get("source_fr") or "")})
    return out, utilises, justes


def _ecrire(dossier: Path, nom: str, rows: list[dict]) -> Path:
    dossier.mkdir(parents=True, exist_ok=True)
    path = dossier / f"{nom}.json"
    path.write_text(json.dumps({"nom": nom, "architecture": "cascade"}, ensure_ascii=False),
                    encoding="utf-8")
    path.with_suffix(".details.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def mesurer(rapport: Path, prises: Path, dossier: Path, n_boot: int = 1000) -> dict:
    rows = prises_du_rapport(rapport, prises)
    premieres, suivantes, vues = [], [], set()
    for r in rows:
        (suivantes if r["prompt_id"] in vues else premieres).append(r)
        vues.add(r["prompt_id"])
    sans = _ecrire(dossier, "redire-sans-memoire", suivantes)

    # 1. Redire : seuil choisi sur le dev, puis le test.
    essais = {}
    for seuil in SEUILS:
        compris, utilises, justes = _comprendre(suivantes, _memoire(premieres, seuil))
        chemin = _ecrire(dossier, f"redire-memoire-{seuil}", compris)
        essais[seuil] = (chemin, utilises, justes)
    seuil = max(SEUILS, key=lambda s: (score(essais[s][0], "dev"), s))  # à égalité, le plus prudent
    redire = comparer(essais[seuil][0], sans, "test", n_boot=n_boot)

    # 2. Jamais dit : le test, avec une mémoire du dev seulement.
    test = [r for r in rows if r.get("partition") == "test"]
    dev = [r for r in rows if r.get("partition") == "dev"]
    compris, faux_souvenirs, _ = _comprendre(test, _memoire(dev, seuil))
    jamais = comparer(_ecrire(dossier, "jamais-dit-memoire", compris),
                      _ecrire(dossier, "jamais-dit-sans-memoire", test), "test", n_boot=n_boot)

    raisons = []
    if not redire["meilleur"]:
        raisons.append(f"pas mieux comprise la fois suivante : écart {redire['ecart']:+.1f} "
                       f"[{redire['ic'][0]:.1f} ; {redire['ic'][1]:.1f}]")
    if jamais["ic"][1] < 0 and jamais["ecart"] < -TOLERANCE:
        raisons.append(f"phrases jamais dites dégradées : {jamais['ecart']:+.1f} "
                       f"[{jamais['ic'][0]:.1f} ; {jamais['ic'][1]:.1f}]")
    return {"promu": not raisons, "raisons": raisons, "seuil": seuil, "redire": redire,
            "jamais_dit": jamais, "phrases_apprises": len(premieres),
            "souvenirs_utilises": essais[seuil][1], "souvenirs_justes": essais[seuil][2],
            "faux_souvenirs": faux_souvenirs, "redites": len(suivantes)}


def ecrire_memoire(rapport: Path, prises: Path, out: Path) -> int:
    """La mémoire à déployer : chaque prise de l'atelier (ce que la
    reconnaissance a entendu → la phrase voulue)."""
    date = datetime.now().isoformat(timespec="seconds")
    lignes = [{"bm": r["source_bm"], "fr": r["ref_translation_fr"], "source": "atelier",
               "date": date} for r in prises_du_rapport(rapport, prises)
              if (r.get("source_bm") or "").strip()]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lignes),
                   encoding="utf-8")
    return len(lignes)


def to_markdown(d: dict) -> str:
    r, j = d["redire"], d["jamais_dit"]
    lignes = [
        f"Phrases apprises : {d['phrases_apprises']} ; redites ensuite : {d['redites']} "
        f"(seuil choisi sur le dev : {d['seuil']})",
        f"Redites, test ({r['n']} énoncés) : sans mémoire {r['titulaire']:.1f}, avec "
        f"{r['candidat']:.1f}, écart {r['ecart']:+.1f} [{r['ic'][0]:.1f} ; {r['ic'][1]:.1f}]",
        f"Souvenirs utilisés : {d['souvenirs_utilises']} dont {d['souvenirs_justes']} de la "
        f"bonne phrase",
        f"Jamais dites, test ({j['n']} énoncés) : sans mémoire {j['titulaire']:.1f}, avec "
        f"{j['candidat']:.1f}, écart {j['ecart']:+.1f} [{j['ic'][0]:.1f} ; {j['ic'][1]:.1f}] "
        f"({d['faux_souvenirs']} faux souvenirs)",
        "**Mémoire en service.**" if d["promu"] else
        "**Mémoire refusée** : " + " ; ".join(d["raisons"]),
    ]
    return "\n".join(f"- {x}" for x in lignes)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rapport", type=Path, required=True, help="rapport eval.run_eval du jeu maison")
    p.add_argument("--prises", type=Path, required=True, help="dire.jsonl de l'atelier")
    p.add_argument("--out", type=Path, required=True, help="dossier des rapports dérivés")
    p.add_argument("--memoire", type=Path, default=None, help="écrire la mémoire à déployer")
    p.add_argument("--decision", type=Path, default=None, help="décision en JSON")
    args = p.parse_args(argv)

    d = mesurer(args.rapport, args.prises, args.out)
    print(to_markdown(d))
    if args.decision:
        args.decision.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.memoire:
        n = ecrire_memoire(args.rapport, args.prises, args.memoire)
        print(f"Mémoire à déployer : {n} phrases → {args.memoire}")


if __name__ == "__main__":
    main()
