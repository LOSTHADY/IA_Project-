"""Fine-tuning NLLB-200 sur Bayelemabaga (bambara <-> français).

    !pip install -q transformers datasets accelerate sacrebleu sentencepiece
    !python scripts/finetune_nllb.py --direction bm2fr
    !python scripts/finetune_nllb.py --direction fr2bm

Deux modèles distincts plutôt qu'un bidirectionnel : c'est un peu plus lourd
en disque mais nettement plus simple à évaluer, et le sens fr->bm (le plus
critique pour la qualité perçue, puisque c'est lui qui produit ce que
l'utilisateur entend) mérite d'être optimisé séparément.

Référence : Bayelemabaga contient ~47k paires alignées bambara-français
issues du Corpus Bambara de Référence (NAACL 2025). C'est peu — attention au
surapprentissage, garder le nombre d'époques bas et surveiller le chrF++ de
validation plutôt que la perte. La sélection se fait sur la validation, pour
que la partition de test reste intacte (cf. eval.baselines mt).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bambara_voice.config import BAM, FRA  # noqa: E402
from eval.corpora import BAYELEMABAGA  # noqa: E402
from reprise import last_complete_checkpoint  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=BAYELEMABAGA)
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--direction", choices=["bm2fr", "fr2bm"], required=True)
    p.add_argument("--bm-column", default=None, help="défaut : détectée")
    p.add_argument("--fr-column", default=None, help="défaut : détectée")
    p.add_argument("--output", default=None)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--optim", default="adafactor",
                   help="adafactor : état d'optimiseur quasi nul. Avec AdamW (adamw_torch), "
                        "NLLB-600M dépasse les 15 Go d'un T4 dès un lot de phrases longues "
                        "(vocabulaire de 256k : logits de 1 Go par lot de 8 × 128)")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--save-steps", type=int, default=500,
                   help="checkpoint (et évaluation) toutes les N étapes : une session Colab "
                        "coupée ne perd que le travail depuis le dernier")
    p.add_argument("--eval-samples", type=int, default=300,
                   help="phrases de validation évaluées à chaque checkpoint (génération "
                        "en faisceau : coûteuse)")
    p.add_argument("--resume", action="store_true",
                   help="reprendre au dernier checkpoint de --output s'il existe")
    p.add_argument("--push-to-hub", default=None, metavar="REPO_ID")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    output = args.output or f"checkpoints/nllb-{args.direction}"

    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
    )
    import sacrebleu

    from eval.corpora import detect_columns, load_any

    ds = load_any(args.dataset)
    cols = detect_columns(ds["train"].features, args.bm_column, args.fr_column,
                          need_fr=True)
    print(f"Schéma : {dict(ds['train'].features)}\nColonnes : {cols.describe()}")

    if args.direction == "bm2fr":
        src_lang, tgt_lang = BAM, FRA
        src_col, tgt_col = cols.bm, cols.fr
    else:
        src_lang, tgt_lang = FRA, BAM
        src_col, tgt_col = cols.fr, cols.bm

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, src_lang=src_lang, tgt_lang=tgt_lang
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    # Sans langue cible imposée, la génération de validation peut partir dans
    # une autre langue : le chrF++ qui choisit le meilleur checkpoint serait
    # faux. Même contrainte qu'à l'inférence (bambara_voice.mt).
    model.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    eval_key = next((k for k in ("validation", "valid", "dev") if k in ds), None)
    if eval_key is None:
        split = ds["train"].train_test_split(test_size=0.05, seed=42)
        train_raw, eval_raw = split["train"], split["test"]
    else:
        train_raw, eval_raw = ds["train"], ds[eval_key]
    if args.max_samples:
        train_raw = train_raw.select(range(min(args.max_samples, len(train_raw))))
    if args.eval_samples and len(eval_raw) > args.eval_samples:
        eval_raw = eval_raw.shuffle(seed=42).select(range(args.eval_samples))

    def texts(batch, column):
        # Format WMT (translation = {bam, fr}) ou colonnes plates.
        if cols.nested:
            return [row[column] for row in batch[cols.nested]]
        return batch[column]

    def preprocess(batch):
        model_inputs = tokenizer(
            texts(batch, src_col), text_target=texts(batch, tgt_col),
            max_length=args.max_length, truncation=True,
        )
        return model_inputs

    train = train_raw.map(preprocess, batched=True, remove_columns=train_raw.column_names)
    evalset = eval_raw.map(preprocess, batched=True, remove_columns=eval_raw.column_names)

    def compute_metrics(pred):
        preds = pred.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = pred.label_ids
        labels[labels == -100] = tokenizer.pad_token_id
        hyp = tokenizer.batch_decode(preds, skip_special_tokens=True)
        ref = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # chrF++ comme métrique de sélection : plus fiable que BLEU en faible
        # ressource et sur morphologie riche.
        chrf = sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score
        bleu = sacrebleu.corpus_bleu(hyp, [ref]).score
        return {"chrf": chrf, "bleu": bleu}

    training_args = Seq2SeqTrainingArguments(
        output_dir=output,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        optim=args.optim,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=4,
        logging_steps=50,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=evalset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    last = last_complete_checkpoint(output) if args.resume else None
    if args.resume:
        print(f"Reprise depuis {last}" if last else "Aucun checkpoint : départ de zéro")
    trainer.train(resume_from_checkpoint=last)
    # Historique et durée d'entraînement (trainer_state.json), pour estimer
    # la durée d'un entraînement complet à partir d'un essai.
    trainer.save_state()
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    print(f"Modèle {args.direction} enregistré dans {output}")
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
