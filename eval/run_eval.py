"""Évaluation bout-en-bout d'une variante du pipeline.

Produit, pour un jeu de test donné :
  - WER/CER strict et relâché de l'ASR (variante cascade uniquement) ;
  - chrF++/BLEU de la traduction entrante, depuis la transcription de
    référence *et* depuis la sortie de l'ASR ;
  - la propagation d'erreurs entre les deux ;
  - le budget de latence par étape et le RTF ;
  - le taux de couverture des gabarits.

Utilisation :
    python -m eval.run_eval --testset data/testset/testset.jsonl --arch cascade
    python -m eval.run_eval --testset data/testset/testset.jsonl --arch e2e
    python -m eval.run_eval --testset ... --arch cascade --config modeles-cpu/config.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from bambara_voice.config import build_config
from bambara_voice.mt import Translator
from bambara_voice.pipeline import VoicePipeline

from .dataset import load_testset, describe
from .metrics import score_asr, score_mt, cascade_degradation

logger = logging.getLogger(__name__)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(
    testset_path: Path,
    architecture: str,
    out_dir: Path,
    limit: int | None = None,
    synthesize: bool = False,
    config_path: Path | None = None,
) -> dict:
    items = load_testset(testset_path)
    if limit:
        items = items[:limit]
    audio_root = testset_path.parent

    config = build_config(architecture, config_path)
    pipeline = VoicePipeline(config)

    traces, rows = [], []
    for idx, item in enumerate(items, 1):
        audio = item.audio_path(audio_root)
        if not audio.exists():
            logger.warning("[%d/%d] audio absent, ignoré : %s", idx, len(items), audio)
            continue
        trace, _ = pipeline.run(audio, synthesize=synthesize)
        traces.append(trace)
        rows.append({"item": item.id, **trace.to_dict(),
                     "ref_transcript_bm": item.transcript_bm,
                     "ref_translation_fr": item.translation_fr})
        logger.info("[%d/%d] %s — RTF %.2f", idx, len(items), item.id, trace.rtf)

    if not traces:
        raise RuntimeError("aucun item évalué : vérifier les chemins audio")

    evaluated = [i for i in items if any(r["item"] == i.id for r in rows)]
    report: dict = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "architecture": architecture,
        "testset": str(testset_path),
        "composition": describe(evaluated),
        "configuration": str(config_path) if config_path else "défaut",
        "appareil": config.device,
        "modeles": {
            "asr": f"{config.asr.backend}:{config.asr.model_id}",
            "mt_in": f"{config.mt_in.backend}:{config.mt_in.model_id}",
            "mt_out": f"{config.mt_out.backend}:{config.mt_out.model_id}",
            "llm": f"{config.llm.backend}:{config.llm.gguf_path or config.llm.model_id}",
            "tts": config.tts.model_id,
        },
    }

    # --- ASR : seulement si la variante produit du texte bambara ------------
    if architecture == "cascade":
        hyps = [t.source_bm for t in traces]
        refs = [i.transcript_bm for i in evaluated]
        report["asr"] = score_asr(hyps, refs).to_dict()

    # --- Traduction entrante ------------------------------------------------
    refs_fr = [i.translation_fr for i in evaluated]
    if all(refs_fr):
        hyps_fr = [t.source_fr for t in traces]
        from_asr = score_mt(hyps_fr, refs_fr)
        report["mt_in_depuis_asr"] = from_asr.to_dict()

        if architecture == "cascade":
            # Même traduction, mais depuis la transcription de référence :
            # isole le coût de l'ASR dans la chaîne.
            gold_translator = Translator(config.mt_in, config.device)
            gold_hyps = [gold_translator.translate(i.transcript_bm) for i in evaluated]
            from_gold = score_mt(gold_hyps, refs_fr)
            report["mt_in_depuis_texte_de_reference"] = from_gold.to_dict()
            report["propagation_erreurs"] = cascade_degradation(from_gold, from_asr)
    else:
        logger.warning(
            "traduction française de référence manquante sur certains items : "
            "métriques de traduction non calculées"
        )

    # --- Latence et gabarits ------------------------------------------------
    steps = sorted({k for t in traces for k in t.timings})
    report["latence"] = {
        "par_etape_s": {s: round(_mean([t.timings.get(s, 0.0) for t in traces]), 3)
                        for s in steps},
        "total_moyen_s": round(_mean([t.total_seconds for t in traces]), 3),
        "rtf_moyen": round(_mean([t.rtf for t in traces]), 3),
    }
    n_tpl = sum(t.reply_source == "template" for t in traces)
    report["gabarits"] = {
        "couverture": round(n_tpl / len(traces), 3),
        "repli_mt": round(1 - n_tpl / len(traces), 3),
        "banque": len(pipeline.templates),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (out_dir / f"{architecture}-{stamp}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / f"{architecture}-{stamp}.details.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Rapport écrit dans %s", out_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", type=Path, required=True)
    parser.add_argument("--arch", choices=["cascade", "e2e"], default="cascade")
    parser.add_argument("--out", type=Path, default=Path("eval/results"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None,
                        help="fichier de déploiement JSON (cf. scripts/export_cpu.py)")
    parser.add_argument("--synthesize", action="store_true",
                        help="inclure la TTS dans le chronométrage (plus lent)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = evaluate(args.testset, args.arch, args.out, args.limit, args.synthesize,
                      args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
