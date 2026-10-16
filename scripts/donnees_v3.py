"""Données du troisième tour de NLLB, tirées de ressources publiques.

    python scripts/donnees_v3.py --out /content/donnees-v3

Trois sources, téléchargées depuis GitHub (docs/RECHERCHE.md) :

- **Bamadaba** (Vydrin, Maslinsky, Méric ; LLACAN) : les 4 644 exemples du
  dictionnaire, des phrases courtes et souvent orales (« dàba d'á mà ! » →
  « donne-lui la houe ! »). Tons retirés, renvois et commentaires en
  parenthèses retirés du français, exemples inachevés (« ... ») écartés.
  Licence CC BY-NC-SA 3.0 France (page officielle du Bamadaba ; le dépôt
  GitHub n'a pas de fichier de licence) : pas d'usage commercial ;
- **GATITOS** (Google, CC-BY-4.0) : environ 4 000 mots et formules, bambara
  et français joints par leur clé anglaise (« a' ni sɔgɔma » → « bonjour »).
  Quelques paires sont fausses (« i danse » → « accueillir ») : elles pèsent
  peu ;
- **MAFAND-MT fr-bam** (Adelani et al. 2022, CC-BY-4.0-NC) : 3 000 phrases de
  presse malienne traduites par des humains, en orthographe standard. Son
  test (1 500 phrases) sert de contrôle de non-régression, jamais à
  l'entraînement.

Écrit un dossier par source (`bamadaba/`, `gatitos/`, `mafand/`), chacun
lisible par `--extra-dataset` de finetune_nllb.py (colonnes bm et fr). Retire
de l'entraînement toute paire dont le bambara ou le français figure dans un
test : celui de MAFAND, et les phrases du jeu maison de l'atelier
(data/atelier/a_dire.json, tutoyées et vouvoyées), pour qu'il reste une
mesure honnête de la parole quotidienne.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bambara_voice.normalize import normalize, strip_punctuation  # noqa: E402

SOURCES = {
    "bamadaba.txt": "https://raw.githubusercontent.com/maslinych/bamadaba/master/bamadaba.txt",
    "en_bm.tsv": "https://raw.githubusercontent.com/google-research/url-nlp/"
                 "3622039cf51f7eeffa58b957332a8e8c337981d7/gatitos/en_bm.tsv",
    "en_fr.tsv": "https://raw.githubusercontent.com/google-research/url-nlp/"
                 "3622039cf51f7eeffa58b957332a8e8c337981d7/gatitos/en_fr.tsv",
    **{f"mafand-{s}.json": "https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/"
                           f"data/json_files/fr-bam/{s}.json" for s in ("train", "dev", "test")},
}

_ABREVIATIONS = {r"\bqqn\b\.?": "quelqu'un", r"\bqqch\b\.?": "quelque chose"}


def telecharger(dossier: Path) -> Path:
    """Les fichiers sources, dans `dossier` ; ceux déjà présents sont gardés."""
    dossier.mkdir(parents=True, exist_ok=True)
    for nom, url in SOURCES.items():
        if not (dossier / nom).exists():
            with urllib.request.urlopen(url, timeout=120) as r:
                (dossier / nom).write_bytes(r.read())
    return dossier


def sans_tons(text: str) -> str:
    """Retire les tons (accents combinants) ; ɛ, ɔ, ɲ, ŋ sont des lettres et
    restent."""
    decompose = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in decompose
                                                if not unicodedata.combining(c)))


def _espaces(text: str) -> str:
    """Espaces simples, aucune avant une virgule ou un point ; celle qui
    précède « ! ? ; : » reste, comme dans le français des phrases de test."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.])", r"\1", text)


def nettoyer_francais(text: str) -> str:
    """Le français d'un exemple de dictionnaire : sans renvois ni commentaires
    entre parenthèses, sans points de suspension, première traduction quand
    il y en a plusieurs, abréviations écrites en entier."""
    text = re.sub(r"\([^()]*\)", " ", text)
    text = text.replace("…", " ").replace("...", " ")
    text = text.split(" ; ")[0]
    for motif, mot in _ABREVIATIONS.items():
        text = re.sub(motif, mot, text)
    return _espaces(text).strip(" ,;:")


def bamadaba(path: Path) -> list[dict]:
    """Paires (exemple bambara \\xv, traduction française \\xf) du Bamadaba."""
    lignes = path.read_text(encoding="utf-8").splitlines()
    paires = []
    for ligne, suivante in zip(lignes, lignes[1:]):
        if not (ligne.startswith("\\xv ") and suivante.startswith("\\xf ")):
            continue
        bm, fr = ligne[4:], suivante[4:]
        if "..." in bm or "…" in bm:
            continue  # exemple inachevé
        bm, fr = _espaces(sans_tons(bm)), nettoyer_francais(fr)
        if bm and fr:
            paires.append({"bm": bm, "fr": fr, "source": "bamadaba"})
    return paires


def gatitos(en_bm: Path, en_fr: Path) -> list[dict]:
    """Bambara et français de GATITOS, joints par leur clé anglaise."""
    def lire(path):
        out = {}
        for ligne in path.read_text(encoding="utf-8").splitlines():
            parts = ligne.split("\t")
            if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                out.setdefault(parts[0].strip(), parts[1].strip())
        return out
    bm, fr = lire(en_bm), lire(en_fr)
    return [{"bm": bm[k], "fr": fr[k], "source": "gatitos"} for k in bm if k in fr]


def mafand(path: Path) -> list[dict]:
    paires = []
    for ligne in path.read_text(encoding="utf-8").splitlines():
        if ligne.strip():
            t = json.loads(ligne)["translation"]
            if t.get("bam", "").strip() and t.get("fr", "").strip():
                paires.append({"bm": t["bam"].strip(), "fr": t["fr"].strip(),
                               "source": "mafand"})
    return paires


def cle(text: str) -> str:
    """Forme de comparaison : sans casse, ponctuation ni espaces superflus."""
    return strip_punctuation(normalize(text))


def phrases_maison(path: Path) -> set[str]:
    """Les phrases françaises du jeu maison (formes tutoyée et vouvoyée)."""
    from eval.atelier import charger_phrases_a_dire

    return {cle(p[k]) for p in charger_phrases_a_dire(path) for k in ("fr", "vous") if k in p}


def ecrire(dossier: Path, partitions: dict[str, list[dict]]) -> None:
    dossier.mkdir(parents=True, exist_ok=True)
    for nom, paires in partitions.items():
        with (dossier / f"{nom}.jsonl").open("w", encoding="utf-8") as fh:
            for p in paires:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")


def preparer(out: Path, sources: Path, maison: Path | None) -> dict[str, dict[str, int]]:
    """Écrit les trois dossiers ; renvoie, par source, les paires gardées et
    écartées."""
    m_train, m_dev, m_test = (mafand(sources / f"mafand-{s}.json")
                              for s in ("train", "dev", "test"))
    interdits_bm = {cle(p["bm"]) for p in m_dev + m_test}
    interdits_fr = {cle(p["fr"]) for p in m_dev + m_test}
    if maison is not None:
        interdits_fr |= phrases_maison(maison)

    vues: set[tuple[str, str]] = set()

    def garder(paires):
        out = []
        for p in paires:
            k = (cle(p["bm"]), cle(p["fr"]))
            if k[0] in interdits_bm or k[1] in interdits_fr or k in vues:
                continue
            vues.add(k)
            out.append(p)
        return out

    stats = {}
    jeux = {"mafand": m_train, "bamadaba": bamadaba(sources / "bamadaba.txt"),
            "gatitos": gatitos(sources / "en_bm.tsv", sources / "en_fr.tsv")}
    for nom, paires in jeux.items():
        gardees = garder(paires)
        parts = {"train": gardees}
        if nom == "mafand":
            parts.update(validation=m_dev, test=m_test)
        ecrire(out / nom, parts)
        stats[nom] = {"gardees": len(gardees), "ecartees": len(paires) - len(gardees)}
    return stats


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sources", type=Path, default=None,
                   help="dossier des fichiers sources (défaut : OUT/sources, téléchargés)")
    p.add_argument("--maison", type=Path, default=ROOT / "data" / "atelier" / "a_dire.json",
                   help="phrases du jeu maison, à tenir hors de l'entraînement")
    args = p.parse_args(argv)

    sources = args.sources or telecharger(args.out / "sources")
    for nom, s in preparer(args.out, sources, args.maison).items():
        print(f"{nom} : {s['gardees']} paires d'entraînement ({s['ecartees']} écartées : "
              f"doublons, ou présentes dans un test)")


if __name__ == "__main__":
    main()
