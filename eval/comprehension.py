"""Le français compris a-t-il le sens de la phrase dite ? Jugé par le locuteur.

Le chrF++ du jeu maison compare le français compris à la phrase affichée,
alors que la phrase est dite librement en bambara : une compréhension juste
mais formulée autrement y perd des points, et le score d'un système parfait
est inconnu. Ici, celui qui a dit la phrase note chaque compréhension :

- 2 : même sens ;
- 1 : à moitié (l'idée générale, ou une partie) ;
- 0 : faux, ou rien compris.

D'où un taux de compréhension, avec son IC à 95 %, qui dit ce que vaut un
chrF++ de 28.

Une seconde question dit d'où viennent les erreurs : ce que la
reconnaissance a écrit en bambara est-il bien ce qui a été dit (2 : oui,
1 : à peu près, 0 : non) ? Si elle a bien entendu et que le sens est faux,
c'est la traduction qui se trompe ; si elle a mal entendu, c'est la
reconnaissance. Le locuteur sait lire le bambara même sans l'écrire : c'est
le seul moyen de faire ce partage sans transcription.

Les notes vont dans `comprehension.jsonl` (dossier de l'atelier), une par
ligne, avec le système jugé : on reprend là où on s'était arrêté, et un
autre système se juge sur les mêmes phrases.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path

NOTES = {2: "même sens", 1: "à moitié", 0: "faux"}


def _lignes(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(ligne) for ligne in path.read_text(encoding="utf-8").splitlines()
            if ligne.strip()]


def a_juger(rapport: str | Path, notes: str | Path, systeme: str, n: int = 100,
            seed: int = 0) -> list[dict]:
    """Les énoncés qui restent à noter pour `systeme` : `n` tirés au hasard,
    toujours les mêmes (graine fixe), dans le rapport eval.run_eval du jeu
    maison ; chacun avec la phrase affichée et le français compris."""
    rows = [r for r in _lignes(Path(rapport).with_suffix(".details.jsonl"))
            if r.get("ref_translation_fr")]
    rows.sort(key=lambda r: str(r["item"]))
    tirage = random.Random(seed).sample(rows, min(n, len(rows)))
    faits = {x["item"] for x in _lignes(notes) if x.get("systeme") == systeme}
    return [{"item": r["item"], "dit": r["ref_translation_fr"], "compris": r.get("source_fr") or "",
             "entendu": r.get("source_bm") or "", "partition": r.get("partition", "")}
            for r in tirage if r["item"] not in faits]


def noter(notes: str | Path, item: str, note: int, systeme: str,
          entendu: int | None = None) -> None:
    """`note` : le sens (NOTES) ; `entendu` : la reconnaissance a-t-elle bien
    entendu (2 oui, 1 à peu près, 0 non) ?"""
    if note not in NOTES or entendu not in (None, 0, 1, 2):
        raise ValueError(f"note inconnue : {note}, {entendu} ; possibles : {sorted(NOTES)}")
    path = Path(notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"item": item, "note": note, "entendu": entendu, "systeme": systeme,
                             "date": datetime.now().isoformat(timespec="seconds")},
                            ensure_ascii=False) + "\n")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC à 95 % d'une proportion k/n (Wilson : juste même près de 0 ou 1)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    marge = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - marge), min(1.0, centre + marge))


def bilan(notes: str | Path, systeme: str) -> dict:
    """Part de chaque note pour `systeme` (la dernière note d'un énoncé
    compte), et l'origine des phrases mal comprises : mal entendues
    (reconnaissance) ou bien entendues mais mal traduites (traduction)."""
    derniere = {x["item"]: x for x in _lignes(notes) if x.get("systeme") == systeme}
    n = len(derniere)
    out = {"systeme": systeme, "n": n}
    for note, nom in NOTES.items():
        k = sum(x["note"] == note for x in derniere.values())
        lo, hi = wilson(k, n)
        out[nom] = {"part": k / n if n else 0.0, "ic": [lo, hi]}
    erreurs = [x for x in derniere.values() if x["note"] < 2 and x.get("entendu") is not None]
    out["erreurs"] = {"n": len(erreurs),
                      "reconnaissance": sum(x["entendu"] == 0 for x in erreurs),
                      "les deux": sum(x["entendu"] == 1 for x in erreurs),
                      "traduction": sum(x["entendu"] == 2 for x in erreurs)}
    return out


def to_markdown(b: dict) -> str:
    if not b["n"]:
        return f"Aucune note pour {b['systeme']}."
    lignes = [f"Compréhension de {b['systeme']}, jugée par toi sur {b['n']} phrases :", ""]
    for nom in NOTES.values():
        lo, hi = b[nom]["ic"]
        lignes.append(f"- {nom} : {100 * b[nom]['part']:.0f} % [{100 * lo:.0f} ; {100 * hi:.0f}]")
    e = b.get("erreurs", {})
    if e.get("n"):
        lignes += ["", f"D'où viennent les {e['n']} phrases mal ou à moitié comprises :",
                   f"- mal entendues (reconnaissance) : {e['reconnaissance']}",
                   f"- à peu près entendues (les deux) : {e['les deux']}",
                   f"- bien entendues, mal traduites (traduction) : {e['traduction']}"]
    return "\n".join(lignes)
