"""Apprendre de ta voix : des paires d'entraînement tirées de tes enregistrements.

    python scripts/paires_maison.py --rapport RAPPORT.json --prises dire.jsonl --out DOSSIER

Chaque phrase dite dans l'atelier donne une paire : ce que la reconnaissance
en service a entendu (`source_bm`, tel que le traducteur le recevra au micro,
erreurs comprises) et la phrase française affichée, le sens voulu. Rien de
plus proche de l'usage n'existe : ni Bayelemabaga, ni la presse, ni les
professeurs (docs/RESULTATS.md, 17 et 19 octobre).

Les phrases sont réparties par `eval.atelier.usage` :

- **test** : jamais apprises. C'est la moitié test du jeu maison, celle qui
  décide si le modèle entraîné remplace celui en service ;
- **choix** : un quart du dev, jamais appris non plus, qui choisit le
  checkpoint (`finetune_nllb.py --eval-file`) ;
- **apprentissage** : le reste du dev.

Toutes les prises d'une phrase (locuteurs, interlocuteurs, façons de la dire)
vont du même côté. Chaque séance d'enregistrement ajoute des paires : relancer
après chaque séance donne un modèle entraîné sur plus de ta parole, jugé sur
le même test.

`--conversations` ajoute les tours du mode « Discuter » de l'atelier (parole
libre, corrigée par le locuteur) : tous à l'apprentissage, car aucune phrase
du test n'y est imposée. Chaque tour donne la paire « ce que la
reconnaissance a entendu → le sens voulu », et, si le locuteur a confirmé ou
corrigé le bambara, « bambara corrigé → sens voulu ».

Écrit `DOSSIER/apprentissage/train.jsonl` (lisible par `--extra-dataset`) et
`DOSSIER/choix.jsonl` (pour `--eval-file`). Ces paires viennent de ta voix :
elles restent sur Drive, jamais dans le dépôt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.atelier import usage  # noqa: E402


def _lignes(path: Path) -> list[dict]:
    return [json.loads(ligne) for ligne in path.read_text(encoding="utf-8").splitlines()
            if ligne.strip()]


def paires(rapport: Path, prises: Path) -> dict[str, list[dict]]:
    """Les paires (bm entendu, fr voulu) d'apprentissage et de choix ; les
    énoncés où rien n'a été entendu sont écartés."""
    phrase_de = {p["id"]: p["prompt_id"] for p in _lignes(prises) if p.get("prompt_id")}
    out: dict[str, list[dict]] = {"apprentissage": [], "choix": []}
    for r in _lignes(rapport.with_suffix(".details.jsonl")):
        prompt = phrase_de.get(r.get("item"))
        bm, fr = (r.get("source_bm") or "").strip(), (r.get("ref_translation_fr") or "").strip()
        if not prompt or not bm or not fr:
            continue
        u = usage(prompt)
        if u in out:
            out[u].append({"bm": bm, "fr": fr, "item": r["item"], "prompt_id": prompt})
    return out


def paires_conversations(conversations: Path) -> list[dict]:
    """Les paires des tours du mode « Discuter » ; sans doublon."""
    out, vues = [], set()
    for t in _lignes(conversations):
        fr = (t.get("compris") or "").strip()
        entendus = [(t.get("entendu_auto") or [""])[0]]
        if t.get("entendu_statut") in ("juste", "corrige"):
            entendus.append(t.get("entendu_std") or "")
        for bm in entendus:
            bm = bm.strip()
            if bm and fr and (bm, fr) not in vues:
                vues.add((bm, fr))
                out.append({"bm": bm, "fr": fr, "item": t["id"],
                            "prompt_id": f"conversation:{t['id']}"})
    return out


def ecrire(dossier: Path, p: dict[str, list[dict]]) -> None:
    (dossier / "apprentissage").mkdir(parents=True, exist_ok=True)
    for chemin, lignes in ((dossier / "apprentissage" / "train.jsonl", p["apprentissage"]),
                           (dossier / "choix.jsonl", p["choix"])):
        chemin.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lignes),
                          encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--rapport", type=Path, required=True,
                   help="rapport eval.run_eval du jeu maison (reconnaissance en service)")
    a.add_argument("--prises", type=Path, required=True, help="dire.jsonl de l'atelier")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--conversations", type=Path, default=None,
                   help="conversations.jsonl de l'atelier (mode Discuter), s'il existe")
    args = a.parse_args(argv)
    p = paires(args.rapport, args.prises)
    if args.conversations and args.conversations.exists():
        p["apprentissage"] += paires_conversations(args.conversations)
    ecrire(args.out, p)
    phrases = {k: len({x["prompt_id"] for x in v}) for k, v in p.items()}
    print(f"Apprentissage : {len(p['apprentissage'])} prises ({phrases['apprentissage']} phrases) ; "
          f"choix du checkpoint : {len(p['choix'])} prises ({phrases['choix']} phrases)")


if __name__ == "__main__":
    main()
