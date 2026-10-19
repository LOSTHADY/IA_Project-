"""Affiner Soloni sur la voix des locuteurs de l'atelier : l'étage de la voix.

    python scripts/affiner_soloni.py --atelier ATELIER --out DOSSIER \\
        [--modele RobotsMali/soloni-114m-tdt-ctc-v3] [--epochs 30] [--essai]

Tout le reste de l'assistant lit ce que la reconnaissance a écrit : une
phrase mal entendue est mal traduite, quoi que vaille le traducteur. Cet
étage apprend à Soloni les voix, les accents et le micro des locuteurs.

**Les données** : tout ce qui a un texte bambara exact, sans que personne
n'ait à transcrire :

- les lectures (onglet « Lire ») ;
- les tours du mode « Discuter » dont le locuteur a confirmé ou corrigé le
  bambara (ramené à l'orthographe standard, `bambara_voice.orthographe`).

Chaque phrase lue, ou chaque tour, va d'un seul côté, fixé une fois pour
toutes : 70 % pour apprendre, 15 % pour choisir le checkpoint (WER de
validation), 15 % pour le test, jamais appris : le WER avant/après s'y
mesure. Réciter ce qu'il a appris ne prouverait rien ; reconnaître des
phrases qu'il n'a jamais entendues, si. Le texte est écrit comme Soloni
l'écrit (sans ponctuation, majuscules gardées) ; une phrase contenant un
caractère que son vocabulaire ne connaît pas est écartée.

**La recette** est celle de RobotsMali (github.com/RobotsMali-AI/bambara-asr,
kunkado/scripts/train.py), à l'échelle d'un T4 : AdamW, taux d'apprentissage
bas (Soloni sait déjà le bambara : il s'agit de l'adapter sans lui faire
oublier le reste), recuit cosinus, précision 16 bits, arrêt quand le WER de
validation ne baisse plus, et retour au meilleur checkpoint.

Écrit `DOSSIER/soloni-perso.nemo` (lisible par `bambara_voice`, model_id en
`.nemo`) et `DOSSIER/rapport.json` (WER avant et après, sur le test). Le
modèle n'entre en service qu'après le jugement du notebook `voix` : meilleur
sur le test des lectures, et sur la compréhension de tes phrases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bambara_voice.normalize import normalize, strip_punctuation  # noqa: E402

SOLONI = "RobotsMali/soloni-114m-tdt-ctc-v3"


def partie(cle: str) -> str:
    """« apprentissage » (70 %), « choix » (15 %) ou « test » (15 %), fixé par
    la clé (la phrase lue, ou le tour)."""
    b = hashlib.sha1(cle.encode("utf-8")).digest()[0] * 100 // 256
    return "test" if b < 15 else "choix" if b < 30 else "apprentissage"


def texte_asr(texte: str) -> str:
    """Comme Soloni écrit : sans ponctuation, majuscules gardées."""
    return strip_punctuation(normalize(texte, lower=False)).replace("_", " ").strip()


def _lignes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def exemples(atelier: Path) -> list[dict]:
    """Les enregistrements au texte bambara exact, avec leur partie."""
    out = []
    for p in _lignes(atelier / "lire.jsonl"):
        out.append({"audio_filepath": str((atelier / p["audio"]).resolve()),
                    "duration": p.get("duration_s", 0.0), "text": texte_asr(p["texte"]),
                    "partie": partie(f"lu:{p['prompt_id']}"), "source": "lecture",
                    "speaker": p.get("speaker", "")})
    for t in _lignes(atelier / "conversations.jsonl"):
        if t.get("entendu_statut") in ("juste", "corrige") and (t.get("entendu_std") or "").strip():
            out.append({"audio_filepath": str((atelier / t["audio"]).resolve()),
                        "duration": t.get("duration_s", 0.0), "text": texte_asr(t["entendu_std"]),
                        "partie": partie(f"co:{t['id']}"), "source": "conversation",
                        "speaker": t.get("speaker", "")})
    return [e for e in out if e["text"]]


def manifestes(atelier: Path, out: Path, tokenizer=None) -> dict[str, int]:
    """Écrit `out/manifest-{partie}.json` (format NeMo) ; renvoie le nombre
    d'enregistrements par partie (et d'écartés, pour un caractère inconnu)."""
    out.mkdir(parents=True, exist_ok=True)
    garde, ecartes = {"apprentissage": [], "choix": [], "test": []}, 0
    unk = getattr(tokenizer, "unk_id", None) if tokenizer is not None else None
    for e in exemples(atelier):
        if tokenizer is not None and unk is not None and unk >= 0 \
                and unk in tokenizer.text_to_ids(e["text"]):
            ecartes += 1
            continue
        garde[e["partie"]].append(e)
    for nom, lignes in garde.items():
        (out / f"manifest-{nom}.json").write_text(
            "".join(json.dumps({k: e[k] for k in ("audio_filepath", "duration", "text")},
                               ensure_ascii=False) + "\n" for e in lignes), encoding="utf-8")
    return {**{k: len(v) for k, v in garde.items()}, "ecartes": ecartes}


def transcrire(model, chemins: list[str], batch_size: int = 8) -> list[str]:
    if not chemins:
        return []
    sorties = model.transcribe(chemins, batch_size=batch_size, verbose=False)
    if isinstance(sorties, tuple):  # anciennes versions, modèles hybrides
        sorties = sorties[0]
    return [texte_asr(getattr(h, "text", h) or "") for h in sorties]


def wer(model, manifest: Path) -> dict | None:
    """WER strict et relâché (variantes d'orthographe pardonnées) sur un manifeste."""
    from eval.metrics import score_asr

    lignes = _lignes(manifest)
    if not lignes:
        return None
    hyps = transcrire(model, [x["audio_filepath"] for x in lignes])
    s = score_asr(hyps, [x["text"] for x in lignes])
    return {"n": len(lignes), "wer": round(100 * s.wer, 2), "wer_releche": round(100 * s.wer_folded, 2)}


def configurer(model, out: Path, batch: int, epochs: int, lr: float, n_apprentissage: int,
               workers: int = 2) -> int:
    """Données et optimiseur ; renvoie le nombre d'étapes prévu."""
    from omegaconf import OmegaConf, open_dict

    commun = {"sample_rate": 16000, "max_duration": 30.0, "min_duration": 0.1,
              "num_workers": workers, "pin_memory": True, "is_tarred": False,
              "use_lhotse": False, "trim_silence": False}
    train = {**commun, "manifest_filepath": str(out / "manifest-apprentissage.json"),
             "batch_size": batch, "shuffle": True}
    valid = {**commun, "manifest_filepath": str(out / "manifest-choix.json"),
             "batch_size": batch, "shuffle": False}
    with open_dict(model.cfg):
        model.cfg.train_ds = OmegaConf.create(train)
        model.cfg.validation_ds = OmegaConf.create(valid)
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    etapes = max(1, epochs * math.ceil(n_apprentissage / batch))
    model.setup_optimization(optim_config=OmegaConf.create({
        "name": "adamw", "lr": lr, "betas": [0.9, 0.98], "weight_decay": 1e-3,
        "sched": {"name": "CosineAnnealing", "warmup_ratio": 0.1, "min_lr": lr / 100,
                  "max_steps": etapes}}))
    return etapes


def affiner(model, out: Path, epochs: int, precision: str, patience: int = 5) -> Path | None:
    """Entraîne ; revient au meilleur checkpoint (WER de validation) ;
    renvoie son chemin."""
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    ckpt = ModelCheckpoint(dirpath=str(out / "checkpoints"), monitor="val_wer", mode="min",
                           save_top_k=1, save_weights_only=True, filename="meilleur")
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        precision=precision, max_epochs=epochs, logger=False, enable_progress_bar=True,
        log_every_n_steps=5, num_sanity_val_steps=0,
        callbacks=[ckpt, EarlyStopping(monitor="val_wer", mode="min", patience=patience)])
    model.set_trainer(trainer)
    trainer.fit(model)
    meilleur = ckpt.best_model_path
    if meilleur:
        etat = torch.load(meilleur, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(etat)
        return Path(meilleur)
    return None


def charger(modele: str, device: str):
    """Soloni tel que l'assistant le charge (bambara_voice.asr : correctif des
    anciennes configurations NeMo compris)."""
    from bambara_voice.asr import SpeechRecognizer
    from bambara_voice.config import ASRConfig

    rec = SpeechRecognizer(ASRConfig(model_id=modele, kind="nemo"), device)
    rec._ensure_loaded()
    return rec._model


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atelier", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--modele", default=SOLONI)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--essai", action="store_true",
                   help="une époque, pour vérifier que tout s'enchaîne (scores sans valeur)")
    args = p.parse_args(argv)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = charger(args.modele, device)
    comptes = manifestes(args.atelier, args.out, getattr(model, "tokenizer", None))
    print(f"Apprentissage {comptes['apprentissage']}, choix {comptes['choix']}, "
          f"test {comptes['test']} (écartés : {comptes['ecartes']})")
    if comptes["apprentissage"] < 20 or not comptes["choix"] or not comptes["test"]:
        raise SystemExit("Trop peu d'enregistrements au texte exact (lectures, tours corrigés) : "
                         "au moins une vingtaine pour apprendre, et quelques-uns à part.")
    avant = wer(model, args.out / "manifest-test.json")
    print(f"WER sur le test, avant : {avant}")
    epochs = 1 if args.essai else args.epochs
    configurer(model, args.out, args.batch, epochs, args.lr, comptes["apprentissage"])
    meilleur = affiner(model, args.out, epochs, "16-mixed" if device == "cuda" else "32",
                       args.patience)
    model.eval()
    apres = wer(model, args.out / "manifest-test.json")
    print(f"WER sur le test, après : {apres}")
    model.save_to(str(args.out / "soloni-perso.nemo"))
    (args.out / "rapport.json").write_text(json.dumps({
        "modele_de_depart": args.modele, "donnees": comptes, "epochs": epochs, "lr": args.lr,
        "essai": args.essai, "meilleur_checkpoint": str(meilleur) if meilleur else None,
        "wer_test_avant": avant, "wer_test_apres": apres}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"Modèle : {args.out / 'soloni-perso.nemo'}")


if __name__ == "__main__":
    main()
