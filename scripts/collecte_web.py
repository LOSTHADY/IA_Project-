"""Outils de la collecte web (docs/collecte, deployment/COLLECTE_WEB.md).

    python scripts/collecte_web.py phrases --n 500 > phrases.sql
    python scripts/collecte_web.py importer bambara-lectures.zip --atelier ATELIER

`phrases` choisit des phrases à lire dans Bayelemabaga (bambara et français,
4 à 14 mots, sans chiffres ni texte religieux, sans doublon : les règles de
la campagne de lecture, `scripts/campagne.py`). Il les écrit en SQL, prêt à
coller dans le SQL Editor de Supabase. `--deja FICHIER` (un export CSV de la
table `reading_prompts`) écarte les phrases déjà en base. Le jeu de données
se télécharge depuis le Hub : à lancer sur Colab ou sur un ordinateur, pas
depuis un réseau qui bloque huggingface.co.

`importer` verse un export de l'espace équipe (le ZIP « Exporter les lectures
validées ») dans le dossier de l'atelier. Chaque lecteur devient un
contributeur, et chaque lecture une prise du mode « lire ». Le notebook
`voix` et `scripts/affiner_soloni.py` s'en servent alors comme des lectures
faites dans l'atelier. On peut réimporter le même ZIP sans créer de doublons.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


# --- phrases à lire -----------------------------------------------------------

def _sql_texte(t: str) -> str:
    return "'" + t.replace("'", "''") + "'"


def phrases_sql(paires: list[dict]) -> str:
    """Les paires {bm, fr} en une instruction SQL pour `public.reading_prompts`."""
    if not paires:
        return "-- aucune phrase nouvelle\n"
    valeurs = ",\n".join(f"  ({_sql_texte(p['bm'])}, {_sql_texte(p['fr'])})" for p in paires)
    return ("insert into public.reading_prompts (sentence_bm, translation_fr) values\n"
            + valeurs + ";\n")


def deja_en_base(chemin: Path | None) -> set[str]:
    if chemin is None:
        return set()
    with open(chemin, encoding="utf-8", newline="") as f:
        return {(r.get("sentence_bm") or "").strip().lower() for r in csv.DictReader(f)}


def phrases(n: int, deja: set[str], dataset: str, seed: int) -> list[dict]:
    from campagne import traduites
    return traduites(dataset, n, set(deja), seed=seed)


# --- import d'un export -------------------------------------------------------

def _lire_jsonl(chemin: Path) -> list[dict]:
    if not chemin.exists():
        return []
    return [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]


def _ajouter_jsonl(chemin: Path, lignes: list[dict]) -> None:
    if not lignes:
        return
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "a", encoding="utf-8") as f:
        for ligne in lignes:
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")


def importer(zip_path: Path, atelier: Path) -> dict[str, int]:
    """Ajoute les lectures validées du ZIP à l'atelier ; renvoie les comptes."""
    from eval.atelier import Prise
    from eval.collection import Speaker
    from dataclasses import asdict

    atelier.mkdir(parents=True, exist_ok=True)
    locuteurs = {r["id"] for r in _lire_jsonl(atelier / "locuteurs.jsonl")}
    prises = {r["id"] for r in _lire_jsonl(atelier / "lire.jsonl")}
    maintenant = datetime.now(timezone.utc).isoformat(timespec="seconds")
    nouveaux_locuteurs, nouvelles_prises = [], []
    with zipfile.ZipFile(zip_path) as z:
        lignes = [json.loads(l) for l in z.read("testset.jsonl").decode("utf-8").splitlines() if l.strip()]
        for r in lignes:
            speaker = r["speaker"]
            if speaker not in locuteurs:
                locuteurs.add(speaker)
                nouveaux_locuteurs.append(asdict(Speaker(
                    id=speaker, gender=r.get("gender") or "", consent=True,
                    consent_date=maintenant[:10], notes="collecte web (lecture)")))
            ident = "li-web-" + r["id"]
            if ident in prises:
                continue
            prises.add(ident)
            rel = f"audio/{speaker}/{ident}.wav"
            (atelier / rel).parent.mkdir(parents=True, exist_ok=True)
            (atelier / rel).write_bytes(z.read(r["audio"]))
            # même phrase -> même identifiant : elle tombe toujours du même côté
            # (apprentissage ou test), pour tous les lecteurs
            prompt = r.get("prompt_id")
            prompt_id = f"web-{prompt}" if prompt is not None else "web-" + r["transcript_bm"].lower()
            nouvelles_prises.append(asdict(Prise(
                id=ident, mode="lire", audio=rel, speaker=speaker, prompt_id=prompt_id,
                texte=r["transcript_bm"], code_switching=bool(r.get("code_switching")),
                conditions="téléphone (collecte web)", duration_s=float(r.get("duration_s") or 0),
                recorded_at=maintenant)))
    _ajouter_jsonl(atelier / "locuteurs.jsonl", nouveaux_locuteurs)
    _ajouter_jsonl(atelier / "lire.jsonl", nouvelles_prises)
    return {"lectures": len(nouvelles_prises), "lecteurs": len(nouveaux_locuteurs),
            "deja_importees": len(lignes) - len(nouvelles_prises)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="commande", required=True)
    ph = sub.add_parser("phrases", help="phrases à lire, en SQL pour Supabase")
    ph.add_argument("--n", type=int, default=500)
    ph.add_argument("--deja", type=Path, help="export CSV de reading_prompts")
    ph.add_argument("--dataset", default="RobotsMaliAI/bayelemabaga")
    ph.add_argument("--seed", type=int, default=0)
    im = sub.add_parser("importer", help="export ZIP de l'espace équipe -> atelier")
    im.add_argument("zip", type=Path)
    im.add_argument("--atelier", type=Path, required=True)
    a = p.parse_args(argv)
    if a.commande == "phrases":
        sys.stdout.write(phrases_sql(phrases(a.n, deja_en_base(a.deja), a.dataset, a.seed)))
    else:
        c = importer(a.zip, a.atelier)
        print(f"{c['lectures']} lecture(s) et {c['lecteurs']} lecteur(s) ajoutés "
              f"({c['deja_importees']} déjà importée(s)).")


if __name__ == "__main__":
    main()
