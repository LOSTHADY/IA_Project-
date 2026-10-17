"""Un « professeur » : un traducteur ouvert plus gros que le nôtre, qui
fabrique des exemples pour l'entraîner (distillation).

    # 1. Sur ta voix d'abord : traduit-il mieux que le modèle en service ?
    python scripts/professeurs.py mesurer --rapport RAPPORT.json \\
        --modele facebook/nllb-200-3.3B --nom "NLLB 3.3B" --out DOSSIER
    # 2. Seulement s'il fait mieux : traduire le bambara parlé de RobotsMali
    python scripts/professeurs.py traduire --entree bm.jsonl \\
        --modele facebook/nllb-200-3.3B --out paires-brutes.jsonl --max 30000

Candidats, tous téléchargeables sans compte :

- **NLLB-200 3.3B** et **NLLB-200 1.3B distillé** (Meta, CC-BY-NC-4.0) : la
  même famille que notre NLLB 600M, en plus gros ; leur licence de recherche
  couvre cet usage, et le 600M a lui-même été obtenu ainsi, par distillation
  du modèle de 54 milliards de paramètres de Meta ;
- **MADLAD-400 3B** (Google, poids publiés ; Apache-2.0 d'après sa fiche,
  à vérifier) : la balise `<2fr>` en tête de la phrase choisit le français ;
  aucun score bambara publié.

`mesurer` reprend un rapport du jeu maison (eval.run_eval, §9a du notebook),
fait retraduire par le professeur ce que la reconnaissance a transcrit
(`source_bm`), sans rien changer d'autre, et écrit un rapport au même format :
eval.promotion le compare au modèle en service, phrase par phrase. Si aucun
professeur ne fait mieux sur le dev, ses traductions n'apprendraient rien à
notre modèle : on s'arrête là, sans dépenser d'entraînement.

`traduire` écrit au fil de l'eau, morceau par morceau : relancé après une
coupure de Colab, il reprend où il en était.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bambara_voice.config import BAM, FRA  # noqa: E402
from bambara_voice.normalize import has_loop  # noqa: E402

MADLAD_FR = "<2fr> "


class Professeur:
    """Traduction bambara → français par lots (sur GPU, une phrase à la fois
    serait des dizaines de fois plus lent), mêmes réglages que le traducteur
    en service : faisceau de 4, longueur plafonnée à 2 × la source + 10
    jetons, phrase refaite sans répétition si elle boucle."""

    def __init__(self, modele: str, device: str = "cpu", dtype: str = "auto",
                 beams: int = 4, lot: int = 16):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.modele, self.device, self.beams, self.lot = modele, device, beams, lot
        if dtype == "auto":  # T5 (MADLAD) garde ses couches sensibles en float32
            dtype = "float16" if device.startswith("cuda") else "float32"
        kwargs = {"torch_dtype": getattr(torch, dtype)}
        if device.startswith("cuda"):
            # Chargé morceau par morceau sur le GPU : le 3.3B (13 Go en float32
            # sur le disque) ne passe pas par les 12 Go de mémoire de Colab.
            kwargs["device_map"] = {"": device}
        self.tok = AutoTokenizer.from_pretrained(modele)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(modele, **kwargs)
        if not device.startswith("cuda"):
            self.model.to(device)
        self.model.eval()
        self.madlad = self.model.config.model_type in ("t5", "mt5")

    def _entrees(self, textes: list[str]) -> list[str]:
        if self.madlad:
            return [MADLAD_FR + t for t in textes]
        self.tok.src_lang = BAM
        return list(textes)

    def _generer(self, textes: list[str], no_repeat_ngram_size: int = 0) -> list[str]:
        import torch

        enc = self.tok(self._entrees(textes), return_tensors="pt", padding=True,
                       truncation=True, max_length=256).to(self.device)
        kwargs = {"num_beams": self.beams,
                  "max_new_tokens": min(256, math.ceil(2.0 * enc["input_ids"].shape[1]) + 10)}
        if no_repeat_ngram_size:
            kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        if not self.madlad:
            kwargs["forced_bos_token_id"] = self.tok.convert_tokens_to_ids(FRA)
        with torch.no_grad():
            ids = self.model.generate(**enc, **kwargs)
        return [t.strip() for t in self.tok.batch_decode(ids, skip_special_tokens=True)]

    def traduire(self, textes: list[str]) -> list[str]:
        """Traductions dans l'ordre des textes ; "" pour un texte vide."""
        sorties = [""] * len(textes)
        # Par longueur : moins de remplissage dans chaque lot.
        ordre = sorted((i for i, t in enumerate(textes) if t.strip()),
                       key=lambda i: len(textes[i]))
        for debut in range(0, len(ordre), self.lot):
            idx = ordre[debut:debut + self.lot]
            for i, fr in zip(idx, self._generer([textes[i].strip() for i in idx])):
                sorties[i] = fr
        for i, fr in enumerate(sorties):
            if fr and has_loop(fr):
                sorties[i] = self._generer([textes[i].strip()], no_repeat_ngram_size=3)[0]
        return sorties


def _lignes(path: Path) -> list[dict]:
    return [json.loads(ligne) for ligne in path.read_text(encoding="utf-8").splitlines()
            if ligne.strip()]


def mesurer(rapport: Path, prof: Professeur, nom: str, out: Path) -> Path:
    """Le rapport `rapport`, traduction refaite par le professeur ; renvoie le
    chemin du nouveau rapport."""
    from eval.metrics import score_mt

    lignes = _lignes(rapport.with_suffix(".details.jsonl"))
    sources = sorted({(r.get("source_bm") or "").strip() for r in lignes} - {""})
    traduction = dict(zip(sources, prof.traduire(sources)))
    nouvelles = [{**r, "source_fr": traduction.get((r.get("source_bm") or "").strip(), "")}
                 for r in lignes]

    origine = json.loads(rapport.read_text(encoding="utf-8"))
    report = {"nom": nom, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "architecture": "cascade", "testset": origine.get("testset", ""),
              "comprehension_seule": True, "appareil": prof.device,
              "modeles": {**origine.get("modeles", {}), "mt_in": f"professeur:{prof.modele}"},
              "professeur": prof.modele, "transcriptions_de": str(rapport)}
    avec_ref = [r for r in nouvelles if r.get("ref_translation_fr")]
    if avec_ref:
        report["mt_in_depuis_asr"] = score_mt([r["source_fr"] for r in avec_ref],
                                              [r["ref_translation_fr"] for r in avec_ref]).to_dict()
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    chemin = out / f"cascade-{stamp}.json"
    chemin.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with chemin.with_suffix(".details.jsonl").open("w", encoding="utf-8") as fh:
        for r in nouvelles:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return chemin


def traduire(entree: Path, prof: Professeur, out: Path, maximum: int | None = None,
             morceau: int = 512) -> int:
    """Ajoute à `out` la traduction des segments de `entree` (bm.jsonl de
    scripts/donnees_robotsmali.py) qui n'y sont pas encore ; renvoie le
    nombre de paires dans `out`."""
    segments = _lignes(entree)[:maximum]
    faits = set()
    if out.exists():
        texte = out.read_text(encoding="utf-8")
        if texte and not texte.endswith("\n"):  # coupure au milieu d'une ligne : on l'oublie
            texte = texte[:texte.rfind("\n") + 1]
            out.write_text(texte, encoding="utf-8")
        faits = {json.loads(ligne)["id"] for ligne in texte.splitlines() if ligne.strip()}
    a_faire = [s for s in segments if s["id"] not in faits]
    total, debut = len(segments), time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        for k in range(0, len(a_faire), morceau):
            lot = a_faire[k:k + morceau]
            for s, fr in zip(lot, prof.traduire([s["bm"] for s in lot])):
                fh.write(json.dumps({**s, "fr": fr, "professeur": prof.modele},
                                    ensure_ascii=False) + "\n")
            fh.flush()  # sur Drive : une coupure ne perd que le morceau en cours
            fait = k + len(lot)
            vitesse = fait / (time.time() - debut)
            print(f"{len(faits) + fait}/{total} phrases ({vitesse:.1f}/s, encore "
                  f"{(len(a_faire) - fait) / vitesse / 60:.0f} min)", flush=True)
    return len(faits) + len(a_faire)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="commande", required=True)
    for nom in ("mesurer", "traduire"):
        s = sub.add_parser(nom)
        s.add_argument("--modele", required=True)
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--device", default=None, help="défaut : cuda s'il y en a un")
        s.add_argument("--dtype", default="auto")
        s.add_argument("--lot", type=int, default=16)
        if nom == "mesurer":
            s.add_argument("--rapport", type=Path, required=True,
                           help="rapport eval.run_eval du jeu maison (modèle en service)")
            s.add_argument("--nom", required=True)
        else:
            s.add_argument("--entree", type=Path, required=True)
            s.add_argument("--max", type=int, default=None)
    args = p.parse_args(argv)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prof = Professeur(args.modele, device, args.dtype, lot=args.lot)
    if args.commande == "mesurer":
        print(f"Rapport : {mesurer(args.rapport, prof, args.nom, args.out)}")
    else:
        print(f"{traduire(args.entree, prof, args.out, args.max)} paires dans {args.out}")


if __name__ == "__main__":
    main()
