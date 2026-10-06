"""Interface en ligne de commande.

    python -m bambara_voice.cli text "i ni sɔgɔma"          # sans audio
    python -m bambara_voice.cli audio enregistrement.wav --out reponse.wav
    python -m bambara_voice.cli audio in.wav --arch e2e
    python -m bambara_voice.cli check                        # état des composants
    python -m bambara_voice.cli --config modeles-cpu/config.json audio in.wav
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import build_config
from .pipeline import VoicePipeline


def _pipeline(args) -> VoicePipeline:
    config = build_config(args.arch, args.config)
    if args.llm_backend:
        config.llm.backend = args.llm_backend
    if args.no_templates:
        config.templates.enabled = False
    return VoicePipeline(config)


def cmd_text(args) -> int:
    pipeline = _pipeline(args)
    trace = pipeline.run_text(args.text)
    print(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_audio(args) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"Fichier introuvable : {path}", file=sys.stderr)
        return 1
    pipeline = _pipeline(args)
    trace, speech = pipeline.run(path, synthesize=not args.no_tts)
    print(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2))
    if speech is not None and args.out:
        print(f"Audio écrit dans {speech.save(args.out)}", file=sys.stderr)
    return 0


def cmd_check(args) -> int:
    """Vérifie que chaque composant se charge, sans rien inférer."""
    pipeline = _pipeline(args)
    components = [
        ("ASR", pipeline.asr, pipeline.config.asr.model_id),
        ("MT bm->fr", pipeline.mt_in, pipeline.config.mt_in.model_id),
        ("MT fr->bm", pipeline.mt_out, pipeline.config.mt_out.model_id),
        ("LLM", pipeline.llm, pipeline.config.llm.model_id),
        ("TTS", pipeline.tts, pipeline.config.tts.model_id),
    ]
    failed = 0
    for label, component, model_id in components:
        if component is None:
            print(f"  --  {label:12} (non utilisé en {pipeline.config.architecture})")
            continue
        try:
            component._ensure_loaded()
            print(f"  ok  {label:12} {model_id}")
        except Exception as exc:  # noqa: BLE001 - diagnostic, on veut tout voir
            print(f"  KO  {label:12} {model_id}\n      {type(exc).__name__}: {exc}")
            failed += 1
    print(f"  --  gabarits     {len(pipeline.templates)} chargés")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bambara_voice", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=["cascade", "e2e"], default="cascade")
    parser.add_argument("--config", default=None,
                        help="fichier de déploiement JSON (cf. scripts/export_cpu.py)")
    parser.add_argument("--llm-backend", choices=["transformers", "llamacpp", "echo"],
                        default=None)
    parser.add_argument("--no-templates", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_text = sub.add_parser("text", help="partir d'un texte bambara (sans ASR)")
    p_text.add_argument("text")
    p_text.set_defaults(func=cmd_text)

    p_audio = sub.add_parser("audio", help="tour de parole complet")
    p_audio.add_argument("path")
    p_audio.add_argument("--out", default=None, help="fichier WAV de sortie")
    p_audio.add_argument("--no-tts", action="store_true")
    p_audio.set_defaults(func=cmd_audio)

    p_check = sub.add_parser("check", help="vérifier le chargement des composants")
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
