"""Modèles minuscules mais complets (tokenizer, extracteur, poids aléatoires),
construits localement pour tester l'entraînement et l'évaluation sans rien
télécharger. Leurs sorties n'ont aucun sens : seule la plomberie compte.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

PAIRS = [("n bɛ taa sugu la", "je vais au marché"), ("i ni sɔgɔma", "bonjour"),
         ("a bɛ yen", "il est là"), ("réseau tɛ yen bi", "il n'y a pas de réseau aujourd'hui")]


def speech_dataset(root: Path, n_train: int = 16, n_test: int = 6) -> Path:
    """Jeu façon Jeli-ASR : audio {bytes, path}, bam, french, train/test."""
    import datasets
    import soundfile as sf

    def flac(i: int) -> bytes:
        t = np.arange(int((1 + i % 3) * 16_000)) / 16_000
        buf = io.BytesIO()
        sf.write(buf, 0.2 * np.sin(2 * np.pi * (200 + 20 * i) * t), 16_000, format="FLAC")
        return buf.getvalue()

    root.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", n_train), ("test", n_test)):
        datasets.Dataset.from_dict({
            "audio": [{"bytes": flac(i), "path": f"{split}-{i}.flac"} for i in range(n)],
            "duration": [1.0 + i % 3 for i in range(n)],
            "bam": [PAIRS[i % 4][0] for i in range(n)],
            "french": [PAIRS[i % 4][1] for i in range(n)],
        }).to_parquet(str(root / f"{split}.parquet"))
    return root


def parallel_dataset(root: Path) -> Path:
    """Jeu façon Bayelemabaga : translation {bam, fr}, train/validation/test."""
    import datasets

    root.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", 16), ("validation", 4), ("test", 4)):
        datasets.Dataset.from_dict({
            "translation": [{"bam": PAIRS[i % 4][0], "fr": PAIRS[i % 4][1]} for i in range(n)],
        }).to_parquet(str(root / f"{split}.parquet"))
    return root


def whisper(out: Path) -> Path:
    """Whisper à 1 couche, avec les tokens spéciaux dans l'ordre exact de
    Whisper (les tokens de langue doivent suivre <|startoftranscript|>)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import (
        GenerationConfig, WhisperConfig, WhisperFeatureExtractor,
        WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizerFast,
    )
    from transformers.models.whisper.tokenization_whisper import LANGUAGES

    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        [t for pair in PAIRS for t in pair] * 20,
        trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    specials = (["<|endoftext|>", "<|startoftranscript|>"]
                + [f"<|{k}|>" for k in LANGUAGES]
                + ["<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>",
                   "<|nocaptions|>", "<|notimestamps|>"])
    tok.add_special_tokens(specials)
    tok.save(str(out / "tokenizer.json"))
    wt = WhisperTokenizerFast(tokenizer_file=str(out / "tokenizer.json"),
                              unk_token="<|endoftext|>", bos_token="<|endoftext|>",
                              eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    ids = {t: wt.convert_tokens_to_ids(t) for t in specials}
    eot, sot = ids["<|endoftext|>"], ids["<|startoftranscript|>"]

    cfg = WhisperConfig(
        vocab_size=len(wt), d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=16,
        decoder_ffn_dim=16, num_mel_bins=80, max_source_positions=1500,
        max_target_positions=448, pad_token_id=eot, bos_token_id=eot, eos_token_id=eot,
        decoder_start_token_id=sot,
    )
    model = WhisperForConditionalGeneration(cfg)
    gc = GenerationConfig.from_model_config(cfg)
    gc.lang_to_id = {f"<|{k}|>": ids[f"<|{k}|>"] for k in LANGUAGES}
    gc.task_to_id = {"transcribe": ids["<|transcribe|>"], "translate": ids["<|translate|>"]}
    gc.is_multilingual = True
    gc.no_timestamps_token_id = ids["<|notimestamps|>"]
    gc.decoder_start_token_id = sot
    gc.max_length = 448
    # Sans ceci, transformers régénère la config de génération depuis celle
    # du modèle au chargement et perd lang_to_id (les vrais checkpoints
    # Whisper n'ont pas ce drapeau).
    gc._from_model_config = False
    model.generation_config = gc
    model.save_pretrained(out)
    WhisperProcessor(feature_extractor=WhisperFeatureExtractor(), tokenizer=wt).save_pretrained(out)
    return out


def nllb(out: Path) -> Path:
    """NLLB (M2M100) à 1 couche, tokenizer SentencePiece avec les codes de
    langue NLLB (bam_Latn, fra_Latn...)."""
    import sentencepiece as spm
    from transformers import (
        AutoTokenizer, M2M100Config, M2M100ForConditionalGeneration, NllbTokenizer,
    )

    out.mkdir(parents=True, exist_ok=True)
    corpus = out / "corpus.txt"
    corpus.write_text("\n".join([t for pair in PAIRS for t in pair] * 30), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(out / "sp"),
                                   vocab_size=45, character_coverage=1.0,
                                   minloglevel=2)
    NllbTokenizer(vocab_file=str(out / "sp.model")).save_pretrained(out)
    tok = AutoTokenizer.from_pretrained(out)
    cfg = M2M100Config(
        vocab_size=len(tok), d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=16,
        decoder_ffn_dim=16, max_position_embeddings=1024, pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        decoder_start_token_id=tok.eos_token_id,
    )
    M2M100ForConditionalGeneration(cfg).save_pretrained(out)
    return out


CHAT_TEMPLATE = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                 "{% if add_generation_prompt %}assistant:{% endif %}")


def causal_lm(out: Path, chat_template: bool = True) -> Path:
    """LM causal (Llama) à 1 couche, tokenizer BPE, avec ou sans gabarit de
    chat : la plomberie de bambara_voice.llm.ChatModel, de la conversation
    jusqu'à la génération."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator([t for pair in PAIRS for t in pair] * 20, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>",
                                  eos_token="</s>", pad_token="<pad>")
    if chat_template:
        tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(out)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                      max_position_embeddings=512, bos_token_id=tok.bos_token_id,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    LlamaForCausalLM(cfg).save_pretrained(out)
    return out
