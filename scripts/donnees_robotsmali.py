"""Du bambara parlé réel à faire traduire par un professeur (distillation).

    python scripts/donnees_robotsmali.py corpus --out /content/robotsmali
    python scripts/donnees_robotsmali.py filtrer --entree paires-brutes.jsonl --out /content/donnees-v4/robotsmali

Le traducteur en service a appris sur du bambara écrit (Bayelemabaga) ; la
V3, nourrie de presse et de dictionnaire, n'a rien gagné sur la voix de
l'utilisateur. Ce qui manque, c'est du bambara de tous les jours avec son
français. RobotsMali publie le premier sans le second :

- **AfVoices** (conversations du sud du Mali, CC-BY-4.0) : les transcriptions
  relues de la partition d'entraînement de l'article (environ 100 h,
  168 000 segments), dans un fichier texte du dépôt GitHub
  RobotsMali-AI/bambara-asr ; aucun audio à télécharger ;
- **Kunkado** (radio et télévision de Bamako, alternance avec le français,
  CC-BY-SA-4.0) : la colonne `corrected-label` du sous-ensemble relu, lue
  seule dans les fichiers Parquet de Hugging Face (l'audio, 11 Go, n'est pas
  téléchargé).

`corpus` nettoie ces transcriptions et écrit `bm.jsonl` : le texte tel que
transcrit (`bm`, donné au professeur) et sa forme « telle que Soloni
l'écrit » (`bm_asr` : sans ponctuation, majuscules gardées), qui sera la
source de l'élève, pour qu'il apprenne sur ce qu'il recevra au micro.
`filtrer` écarte ensuite les traductions douteuses du professeur
(scripts/professeurs.py) et écrit `train.jsonl`, lisible par
`--extra-dataset` de finetune_nllb.py.

Segments écartés : ceux où un passage manque (AfVoices `[?]` inaudible et
`[cs]` français non transcrit ; Kunkado `<INCOMPRÉHENSIBLE>` et mots coupés
« … »), et ceux de moins de deux mots. Bruits, hésitations et pauses sont
retirés du texte ; les mots français de Kunkado (`__ça va__`) restent, sans
leurs soulignés : l'utilisateur en dit aussi.

Les paires obtenues dérivent de Kunkado (partage à l'identique) et de
NLLB (non commercial) : elles restent sur le Drive, jamais dans ce dépôt.
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

from bambara_voice.normalize import has_loop, normalize, strip_punctuation  # noqa: E402

AFVOICES = ("https://raw.githubusercontent.com/RobotsMali-AI/bambara-asr/"
            "270f134bdc0dd7e1190333b427b5a1d736b28967/afvoices/pre-manifests/"
            "annotated-100-train.jsonl")
KUNKADO = "datasets/RobotsMali/kunkado/human-reviewed/train-*.parquet"
KUNKADO_COLONNE = "corrected-label"

_AF_RETIRER = re.compile(r"\[(?:um|noise|pause)\]")
_KK_TAG = re.compile(r"<[^<>]*>")
_LETTRES_BAMBARA = re.compile(r"[ɛɔɲŋƐƆƝŊ]")


def _espaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def nettoyer_afvoices(text: str) -> str | None:
    """Le texte parlé d'un segment d'AfVoices, ou None si un passage manque."""
    text = _AF_RETIRER.sub(" ", text)
    if "[" in text or "]" in text:  # [?], [cs], ou annotation mal fermée
        return None
    return _espaces(text) or None


def nettoyer_kunkado(text: str) -> str | None:
    """Le texte parlé d'un segment de Kunkado, ou None si un passage manque."""
    if "INCOMPR" in text.upper() or "…" in text or "..." in text:
        return None
    text = _KK_TAG.sub(" ", text).replace("__", " ")
    if "<" in text or ">" in text:
        return None
    return _espaces(text) or None


def forme_asr(text: str) -> str:
    """Le texte comme Soloni l'écrit : sans ponctuation (son vocabulaire n'en a
    pas), apostrophes et majuscules gardées."""
    return strip_punctuation(normalize(text, lower=False)).replace("_", " ").strip()


def lire_afvoices(path: Path) -> list[str]:
    return [json.loads(ligne)["text"] for ligne in path.read_text(encoding="utf-8").splitlines()
            if ligne.strip()]


def lire_kunkado(fs=None, motif: str = KUNKADO, colonne: str = KUNKADO_COLONNE) -> list[str]:
    """La seule colonne de texte des fichiers Parquet : quelques Mo lus au lieu
    de 11 Go d'audio."""
    import pyarrow.parquet as pq

    if fs is None:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
    textes = []
    for chemin in sorted(fs.glob(motif)):
        table = pq.read_table(chemin, columns=[colonne], filesystem=fs)
        textes.extend(t for t in table.column(colonne).to_pylist() if t)
    return textes


def segments(textes: list[str], nettoyer, source: str, min_mots: int = 2,
             max_mots: int = 40) -> list[dict]:
    """Segments nettoyés et dédoublonnés (sur la forme de l'élève)."""
    vus, out = set(), []
    for text in textes:
        bm = nettoyer(text)
        if bm is None:
            continue
        asr = forme_asr(bm)
        if not min_mots <= len(asr.split()) <= max_mots or asr.lower() in vus:
            continue
        vus.add(asr.lower())
        out.append({"bm": bm, "bm_asr": asr, "source": source})
    return out


def corpus(out: Path, afvoices: Path | None = None, kunkado: bool = True,
           seed: int = 0) -> dict[str, int]:
    """Écrit `out/bm.jsonl`, mélangé (une traduction interrompue garde un
    échantillon des deux sources) ; renvoie le nombre de segments par source."""
    out.mkdir(parents=True, exist_ok=True)
    if afvoices is None:
        afvoices = out / "afvoices-train.jsonl"
        if not afvoices.exists():
            with urllib.request.urlopen(AFVOICES, timeout=300) as r:
                afvoices.write_bytes(r.read())
    tous = segments(lire_afvoices(afvoices), nettoyer_afvoices, "afvoices")
    if kunkado:
        try:
            tous += segments(lire_kunkado(), nettoyer_kunkado, "kunkado")
        except Exception as e:  # réseau, accès refusé : AfVoices suffit pour commencer
            print(f"⚠️ Kunkado non lu ({type(e).__name__}: {e}) : AfVoices seulement.")
    # Un segment de Kunkado identique à un d'AfVoices n'apporte rien.
    vus, uniques = set(), []
    for s in tous:
        if s["bm_asr"].lower() not in vus:
            vus.add(s["bm_asr"].lower())
            uniques.append(s)
    random.Random(seed).shuffle(uniques)
    with (out / "bm.jsonl").open("w", encoding="utf-8") as fh:
        for i, s in enumerate(uniques):
            fh.write(json.dumps({"id": f"{s['source']}-{i}", **s}, ensure_ascii=False) + "\n")
    comptes: dict[str, int] = {}
    for s in uniques:
        comptes[s["source"]] = comptes.get(s["source"], 0) + 1
    return comptes


# --- après le professeur -----------------------------------------------------------

def raison_de_rejet(bm: str, fr: str) -> str | None:
    """Pourquoi une traduction du professeur ne doit pas servir d'exemple."""
    if not fr.strip():
        return "vide"
    if has_loop(fr):
        return "boucle"
    if _LETTRES_BAMBARA.search(fr) or fr.strip().lower() == bm.strip().lower():
        return "non traduite"
    ratio = len(fr) / max(len(bm), 1)
    if not 0.4 <= ratio <= 3.0:
        return "longueur"
    return None


def filtrer(entree: Path, out: Path) -> dict[str, int]:
    """Écrit `out/train.jsonl` (bm : forme de l'élève, fr : professeur) ;
    renvoie les paires gardées et écartées par motif."""
    stats: dict[str, int] = {"gardees": 0}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.jsonl").open("w", encoding="utf-8") as fh:
        for ligne in entree.read_text(encoding="utf-8").splitlines():
            if not ligne.strip():
                continue
            p = json.loads(ligne)
            raison = raison_de_rejet(p["bm"], p.get("fr", ""))
            if raison:
                stats[raison] = stats.get(raison, 0) + 1
                continue
            stats["gardees"] += 1
            fh.write(json.dumps({"bm": p.get("bm_asr") or forme_asr(p["bm"]), "fr": p["fr"],
                                 "source": p.get("source", "")}, ensure_ascii=False) + "\n")
    return stats


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="commande", required=True)
    c = sub.add_parser("corpus", help="transcriptions nettoyées → bm.jsonl")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--afvoices", type=Path, default=None,
                   help="manifeste déjà téléchargé (défaut : depuis GitHub)")
    c.add_argument("--sans-kunkado", action="store_true")
    f = sub.add_parser("filtrer", help="traductions du professeur → train.jsonl")
    f.add_argument("--entree", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    if args.commande == "corpus":
        comptes = corpus(args.out, args.afvoices, not args.sans_kunkado)
        print(", ".join(f"{n} : {k} segments" for n, k in comptes.items())
              + f" → {args.out / 'bm.jsonl'}")
    else:
        stats = filtrer(args.entree, args.out)
        ecartees = {k: v for k, v in stats.items() if k != "gardees"}
        print(f"{stats['gardees']} paires gardées → {args.out / 'train.jsonl'}"
              + (f" ; écartées : {ecartees}" if ecartees else ""))


if __name__ == "__main__":
    main()
