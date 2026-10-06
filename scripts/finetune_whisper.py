"""Fine-tuning Whisper pour le bambara — mono-tâche ou multi-tâche.

À exécuter sur Colab (T4 suffit pour whisper-small). Le point clé est
`--task both` : Jeli-ASR fournit pour chaque audio *la transcription bambara
et sa traduction française*. Entraîner un seul modèle sur les deux cibles
donne gratuitement les deux architectures comparées dans ce projet — cascade
(transcribe) et bout-en-bout (translate) — avec exactement le même encodeur,
ce qui rend la comparaison honnête.

    !pip install -q transformers datasets accelerate jiwer sacrebleu soundfile librosa
    !python scripts/finetune_whisper.py \\
        --dataset RobotsMali/jeli-asr --model openai/whisper-small --task both \\
        --output /content/drive/MyDrive/bambara/whisper-small-bm --resume

Notes de terrain :
  - whisper-small est le meilleur compromis qualité/vitesse pour un T4 gratuit ;
    whisper-medium demande du gradient checkpointing et beaucoup de patience.
  - Whisper n'a pas de token de langue pour le bambara. On réutilise un token
    existant peu utilisé (par défaut le swahili "sw") comme emplacement : le
    modèle réapprend ce qu'il désigne pendant le fine-tuning. C'est un
    contournement standard, à mentionner dans le mémoire.
  - Les colonnes (audio, bambara, français) sont détectées automatiquement
    et affichées au lancement ; --bm-column / --fr-column pour forcer.
  - Les spectrogrammes sont calculés à la volée, lot par lot. Les précalculer,
    comme le font la plupart des tutoriels, coûte ~0,9 Mo par exemple et par
    tâche : des dizaines de Go pour Jeli-ASR, plus que le disque de Colab.
  - Le meilleur checkpoint est choisi sur une validation (prise dans train si
    le jeu n'en a pas) : la partition de test reste intacte pour
    `eval.baselines`, sinon le score final serait optimiste.
  - Sauvegarde toutes les --save-steps étapes ; avec --resume, une session
    Colab interrompue repart du dernier checkpoint au lieu de zéro.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Token de langue Whisper détourné pour désigner le bambara (cf. docstring).
# Partagé avec l'inférence : un écart désactiverait le choix de la tâche.
from bambara_voice.config import WHISPER_LANG_SLOT as LANG_SLOT  # noqa: E402
from bambara_voice.normalize import normalize  # noqa: E402
from eval.corpora import decode_audio  # noqa: E402

MAX_AUDIO_S = 30.0  # fenêtre de Whisper : au-delà, l'audio serait tronqué


def target_text(row: dict, task: str, cols) -> str:
    """Cible d'entraînement : bambara (normalisation conservatrice) ou français."""
    if task == "transcribe":
        return normalize(row[cols.bm] or "", lower=False)
    return (row[cols.fr] or "").strip()


def build_plan(base, cols, tasks: list[str], source: int) -> dict[str, list]:
    """Liste des exemples (source, index, tâche), sans aucun audio.

    Chaque audio apparaît une fois par tâche : en `--task both`, le même
    enregistrement sert aux deux cibles, avec le même encodeur.
    """
    columns = {c: base[c] for c in (cols.bm, cols.fr) if c}
    durations = base["duration"] if "duration" in base.column_names else None
    plan: dict[str, list] = {"source": [], "idx": [], "task": []}
    for task in tasks:
        texts = columns[cols.bm if task == "transcribe" else cols.fr]
        for i, text in enumerate(texts):
            if not (text or "").strip():
                continue
            if durations is not None and durations[i] and durations[i] > MAX_AUDIO_S:
                continue
            plan["source"].append(source)
            plan["idx"].append(i)
            plan["task"].append(task)
    return plan


@dataclass
class OnTheFlyCollator:
    """Construit chaque lot à partir du plan : décode l'audio, calcule le
    spectrogramme et tokenise la cible de la tâche demandée, précédée des
    tokens <|sw|><|tâche|> qui la désignent."""

    bases: list                 # [train, validation], audio non décodé
    cols: Any
    feature_extractor: Any
    tokenizers: dict            # tâche -> tokenizer préfixé pour cette tâche
    decoder_start_token_id: int
    max_label_length: int

    def __call__(self, features: list[dict]) -> dict:
        rows = [self.bases[f["source"]][int(f["idx"])] for f in features]
        inputs = [
            {"input_features": self.feature_extractor(
                decode_audio(row[self.cols.audio]), sampling_rate=16_000
            ).input_features[0]}
            for row in rows
        ]
        batch = self.feature_extractor.pad(inputs, return_tensors="pt")

        label_features = []
        for f, row in zip(features, rows):
            ids = self.tokenizers[f["task"]](target_text(row, f["task"], self.cols)).input_ids
            # Garde-fou : une cible plus longue que le décodeur ferait planter
            # l'entraînement. Rarissime sur des énoncés de quelques secondes.
            label_features.append({"input_ids": ids[: self.max_label_length]})
        tok = next(iter(self.tokenizers.values()))
        labels_batch = tok.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Le token de début est réinséré par le modèle : on l'enlève des cibles.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="RobotsMali/jeli-asr")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--task", choices=["transcribe", "translate", "both"], default="both",
                   help="cible d'entraînement : bambara, français, ou les deux")
    p.add_argument("--bm-column", default=None,
                   help="colonne de transcription bambara (défaut : détectée)")
    p.add_argument("--fr-column", default=None,
                   help="colonne de traduction française (défaut : détectée)")
    p.add_argument("--output", default="checkpoints/whisper-bambara")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--max-samples", type=int, default=None, help="pour un essai rapide")
    p.add_argument("--eval-samples", type=int, default=300,
                   help="taille de la validation (la génération y est lente)")
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--resume", action="store_true",
                   help="reprendre au dernier checkpoint de --output s'il existe")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--push-to-hub", default=None, metavar="REPO_ID")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    import torch
    from datasets import Audio, Dataset
    from transformers import (
        WhisperProcessor, WhisperForConditionalGeneration,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    from eval.corpora import detect_columns, load_any, sample_indices
    from eval.metrics import score_asr, score_mt

    tasks = ["transcribe", "translate"] if args.task == "both" else [args.task]
    # La sélection du meilleur modèle se fait sur la tâche cascade si elle est
    # entraînée (WER), sinon sur la traduction directe (chrF++).
    eval_task = tasks[0]

    processor = WhisperProcessor.from_pretrained(args.model, language=LANG_SLOT, task=eval_task)
    tokenizers = {
        task: WhisperProcessor.from_pretrained(args.model, language=LANG_SLOT, task=task).tokenizer
        for task in tasks
    }
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    # Les contraintes héritées gênent le fine-tuning ; la validation génère
    # dans la tâche choisie ci-dessus, avec le token de langue du projet.
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.generation_config.language = LANG_SLOT
    model.generation_config.task = eval_task
    model.config.use_cache = False  # incompatible avec le gradient checkpointing

    ds = load_any(args.dataset, args.dataset_config)
    cols = detect_columns(ds["train"].features, args.bm_column, args.fr_column,
                          need_audio=True, need_fr="translate" in tasks)
    print(f"Schéma : {dict(ds['train'].features)}\nColonnes : {cols.describe()}")
    # Décodage par soundfile plutôt que par datasets (qui exige torchcodec).
    ds = ds.cast_column(cols.audio, Audio(decode=False))

    val_key = next((k for k in ("validation", "valid", "dev") if k in ds), None)
    if val_key:
        train_base, val_base = ds["train"], ds[val_key]
    else:
        held = ds["train"].train_test_split(test_size=min(0.05, 2000 / len(ds["train"])),
                                            seed=42)
        train_base, val_base = held["train"], held["test"]
    if args.max_samples:
        train_base = train_base.select(range(min(args.max_samples, len(train_base))))
    val_base = val_base.select(sample_indices(len(val_base), args.eval_samples))

    train = Dataset.from_dict(build_plan(train_base, cols, tasks, source=0)).shuffle(seed=42)
    evalset = Dataset.from_dict(build_plan(val_base, cols, [eval_task], source=1))
    print(f"Entraînement : {len(train)} exemples ({' + '.join(tasks)}) ; "
          f"validation : {len(evalset)} ({eval_task}, partition "
          f"{val_key or 'prise dans train'})")

    tok = processor.tokenizer

    def compute_metrics(pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids
        # Les lots sont complétés par -100 à la concaténation : à neutraliser
        # des deux côtés avant de décoder.
        pred_ids[pred_ids == -100] = tok.pad_token_id
        label_ids[label_ids == -100] = tok.pad_token_id
        hyp = tok.batch_decode(pred_ids, skip_special_tokens=True)
        ref = tok.batch_decode(label_ids, skip_special_tokens=True)
        pairs = [(h, r) for h, r in zip(hyp, ref) if r.strip()]
        if not pairs:
            return {"wer": 1.0} if eval_task == "transcribe" else {"chrf": 0.0}
        hyp, ref = [p[0] for p in pairs], [p[1] for p in pairs]
        if eval_task == "transcribe":
            return {"wer": score_asr(hyp, ref).wer}
        return {"chrf": score_mt(hyp, ref).chrf}

    max_positions = model.config.max_target_positions
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=min(225, max_positions),
        logging_steps=25,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="wer" if eval_task == "transcribe" else "chrf",
        greater_is_better=eval_task != "transcribe",
        remove_unused_columns=False,  # le plan (source, idx, task) va au collator
        dataloader_num_workers=args.num_workers,
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=evalset,
        data_collator=OnTheFlyCollator(
            bases=[train_base, val_base], cols=cols,
            feature_extractor=processor.feature_extractor, tokenizers=tokenizers,
            decoder_start_token_id=model.config.decoder_start_token_id,
            max_label_length=max_positions,
        ),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    last = get_last_checkpoint(args.output) if args.resume and Path(args.output).is_dir() else None
    if args.resume:
        print(f"Reprise depuis {last}" if last else "Aucun checkpoint : départ de zéro")
    trainer.train(resume_from_checkpoint=last)
    # Historique et durée d'entraînement (trainer_state.json), pour estimer
    # la durée d'un entraînement complet à partir d'un essai.
    trainer.save_state()

    model.config.use_cache = True
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"Modèle enregistré dans {args.output}")
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
