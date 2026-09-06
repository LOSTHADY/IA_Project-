"""Fine-tuning Whisper pour le bambara — mono-tâche ou multi-tâche.

À exécuter sur Colab (T4 suffit pour whisper-small). Le point clé est
`--task both` : Jeli-ASR fournit pour chaque audio *la transcription bambara
et sa traduction française*. Entraîner un seul modèle sur les deux cibles
donne gratuitement les deux architectures comparées dans ce projet — cascade
(transcribe) et bout-en-bout (translate) — avec exactement le même encodeur,
ce qui rend la comparaison honnête.

    !pip install -q transformers datasets accelerate evaluate jiwer soundfile librosa
    !python scripts/finetune_whisper.py \
        --dataset RobotsMali/jeli-asr --model openai/whisper-small --task both

Notes de terrain :
  - whisper-small est le meilleur compromis qualité/vitesse pour un T4 gratuit ;
    whisper-medium demande du gradient checkpointing et beaucoup de patience.
  - Whisper n'a pas de token de langue pour le bambara. On réutilise un token
    existant peu utilisé (par défaut le swahili "sw") comme emplacement : le
    modèle réapprend ce qu'il désigne pendant le fine-tuning. C'est un
    contournement standard, à mentionner dans le mémoire.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

# Token de langue Whisper détourné pour désigner le bambara (cf. docstring).
LANG_SLOT = "sw"


@dataclass
class DataCollator:
    """Assemble les lots : les spectrogrammes sont déjà à taille fixe, seules
    les étiquettes doivent être complétées puis masquées."""

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        import torch

        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
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
    p.add_argument("--audio-column", default="audio")
    p.add_argument("--bm-column", default="bambara", help="colonne de transcription bambara")
    p.add_argument("--fr-column", default="french", help="colonne de traduction française")
    p.add_argument("--output", default="checkpoints/whisper-bambara")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--max-samples", type=int, default=None, help="pour un essai rapide")
    p.add_argument("--push-to-hub", default=None, metavar="REPO_ID")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    import torch
    from datasets import load_dataset, Audio, concatenate_datasets
    from transformers import (
        WhisperProcessor, WhisperForConditionalGeneration,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
    )
    import evaluate

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from bambara_voice.normalize import normalize

    processor = WhisperProcessor.from_pretrained(
        args.model, language=LANG_SLOT, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    # Les contraintes de génération héritées gênent le fine-tuning mono-langue.
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    ds = load_dataset(args.dataset, args.dataset_config)
    ds = ds.cast_column(args.audio_column, Audio(sampling_rate=16_000))

    def make_prepare(target_column: str, task: str):
        """Une passe de préparation par tâche ; le token de tâche distingue les
        deux cibles pour un même audio."""
        tokenizer = WhisperProcessor.from_pretrained(
            args.model, language=LANG_SLOT, task=task
        ).tokenizer

        def prepare(batch):
            audio = batch[args.audio_column]
            batch["input_features"] = processor.feature_extractor(
                audio["array"], sampling_rate=audio["sampling_rate"]
            ).input_features[0]
            text = batch[target_column] or ""
            # Normalisation conservatrice côté bambara uniquement.
            if task == "transcribe":
                text = normalize(text, lower=False)
            batch["labels"] = tokenizer(text.strip()).input_ids
            return batch

        return prepare

    def build_split(split):
        parts = []
        if args.task in ("transcribe", "both"):
            parts.append(split.map(
                make_prepare(args.bm_column, "transcribe"),
                remove_columns=split.column_names, num_proc=1,
            ))
        if args.task in ("translate", "both"):
            if args.fr_column not in split.column_names:
                raise SystemExit(
                    f"colonne '{args.fr_column}' absente : la tâche 'translate' "
                    f"exige une traduction française alignée"
                )
            parts.append(split.map(
                make_prepare(args.fr_column, "translate"),
                remove_columns=split.column_names, num_proc=1,
            ))
        return concatenate_datasets(parts).shuffle(seed=42) if len(parts) > 1 else parts[0]

    train_split = ds["train"]
    if args.max_samples:
        train_split = train_split.select(range(min(args.max_samples, len(train_split))))
    train = build_split(train_split)

    eval_key = next((k for k in ("validation", "test", "dev") if k in ds), None)
    evalset = build_split(ds[eval_key]) if eval_key else None

    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyp = processor.batch_decode(pred_ids, skip_special_tokens=True)
        ref = processor.batch_decode(label_ids, skip_special_tokens=True)
        pairs = [(h, r) for h, r in zip(hyp, ref) if r.strip()]
        if not pairs:
            return {"wer": 1.0}
        return {"wer": metric.compute(
            predictions=[p[0] for p in pairs], references=[p[1] for p in pairs]
        )}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch" if evalset else "no",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to=[],
        load_best_model_at_end=bool(evalset),
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=evalset,
        data_collator=DataCollator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics if evalset else None,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"Modèle enregistré dans {args.output}")
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
