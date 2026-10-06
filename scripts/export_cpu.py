"""Prépare les modèles affinés pour le déploiement CPU (phase 5).

    python scripts/export_cpu.py --out modeles-cpu \\
        --whisper ckpt/whisper-small-bm \\
        --nllb-bm2fr ckpt/nllb-bm2fr --nllb-fr2bm ckpt/nllb-fr2bm \\
        --llm-gguf modeles-cpu/qwen2.5-1.5b-instruct-q4_k_m.gguf

Convertit Whisper et NLLB au format CTranslate2, quantifiés en int8 (poids
4 fois plus légers ; NLLB 2 à 2,5 fois plus rapide, Whisper 1,3 à 1,5 fois
sur des énoncés courts, cf. docs/DEPLOIEMENT.md), copie à côté le tokenizer
et l'extracteur d'origine, puis écrit `<out>/config.json`, directement
utilisable :

    python -m bambara_voice.cli --config modeles-cpu/config.json audio in.wav
    python app/gradio_app.py --config modeles-cpu/config.json
    python -m eval.run_eval --config modeles-cpu/config.json --testset ...

La quantification coûte un peu de qualité : la mesurer avec eval.baselines
(--backend ctranslate2 contre transformers, même échantillon) avant de
retenir les chiffres de latence. Cf. docs/DEPLOIEMENT.md.

La synthèse vocale (VITS) n'a pas d'équivalent CTranslate2 et reste sous
PyTorch. Le LLM se déploie en GGUF via llama.cpp (--llm-gguf).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Fichiers du tokenizer et de l'extracteur à garder à côté du modèle converti
# (CTranslate2 n'emporte que les poids et son propre vocabulaire).
_KEEP = ("tokenizer", "processor", "preprocessor", "special_tokens", "added_tokens",
         "vocab.json", "merges.txt", "normalizer", "sentencepiece", ".model")
_SKIP = {"config.json", "generation_config.json", "training_args.bin"}


def companion_files(model_dir: Path) -> list[str]:
    """Fichiers de tokenizer / extracteur présents dans un checkpoint."""
    return sorted(
        p.name for p in model_dir.iterdir()
        if p.is_file() and p.name not in _SKIP and any(k in p.name for k in _KEEP)
    )


def convert(model_dir: Path, out_dir: Path, quantization: str) -> Path:
    from ctranslate2.converters import TransformersConverter

    files = companion_files(model_dir)
    print(f"{model_dir} -> {out_dir} ({quantization}) ; copie : {', '.join(files)}")
    TransformersConverter(str(model_dir), copy_files=files).convert(
        str(out_dir), quantization=quantization, force=True
    )
    return out_dir


def build_config(out: Path, converted: dict[str, Path], compute_type: str,
                 llm_gguf: str | None) -> dict:
    """Configuration de déploiement, chemins relatifs au dossier `out`."""
    def section(key: str) -> dict:
        return {"model_id": converted[key].relative_to(out).as_posix(),
                "backend": "ctranslate2", "compute_type": compute_type}

    config: dict = {"_comment": "Écrit par scripts/export_cpu.py ; cf. docs/DEPLOIEMENT.md",
                    "device": "cpu"}
    if "whisper" in converted:
        config["asr"] = section("whisper")
    if "nllb_bm2fr" in converted:
        config["mt_in"] = section("nllb_bm2fr")
    if "nllb_fr2bm" in converted:
        config["mt_out"] = section("nllb_fr2bm")
    if llm_gguf:
        config["llm"] = {"backend": "llamacpp", "gguf_path": llm_gguf}
    return config


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--whisper", type=Path, help="checkpoint Whisper affiné")
    p.add_argument("--nllb-bm2fr", type=Path, help="checkpoint NLLB bambara -> français")
    p.add_argument("--nllb-fr2bm", type=Path, help="checkpoint NLLB français -> bambara")
    p.add_argument("--quantization", default="int8",
                   help="int8 (défaut), int8_float32, int16, float32...")
    p.add_argument("--llm-gguf", default=None,
                   help="chemin du LLM en GGUF, relatif à --out ou absolu")
    args = p.parse_args(argv)

    sources = {"whisper": args.whisper, "nllb_bm2fr": args.nllb_bm2fr,
               "nllb_fr2bm": args.nllb_fr2bm}
    sources = {k: v for k, v in sources.items() if v is not None}
    if not sources:
        p.error("rien à convertir : --whisper, --nllb-bm2fr ou --nllb-fr2bm")

    args.out.mkdir(parents=True, exist_ok=True)
    converted = {key: convert(src, args.out / src.name, args.quantization)
                 for key, src in sources.items()}

    config = build_config(args.out, converted, args.quantization, args.llm_gguf)
    path = args.out / "config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nConfiguration écrite : {path}")
    if not args.llm_gguf:
        print("LLM non précisé : le modèle par défaut (transformers) sera utilisé ; "
              "voir --llm-gguf pour llama.cpp.")


if __name__ == "__main__":
    main()
