"""Évaluation maillon par maillon : références zero-shot, puis modèles affinés.

    # Phase 1 — modèles publics tels quels
    python -m eval.baselines asr --model openai/whisper-small --limit 300
    python -m eval.baselines asr --model facebook/mms-1b-all --kind ctc \\
        --target-lang bam --with-mt --limit 300
    python -m eval.baselines mt --dataset RobotsMaliAI/bayelemabaga --limit 500
    python -m eval.baselines tts --limit 20

    # Phase 3-4 — mêmes mesures, modèles affinés, les deux architectures
    python -m eval.baselines --label "cascade" asr --model ckpt/whisper-bm \\
        --with-mt --mt-model ckpt/nllb-bm2fr
    python -m eval.baselines --label "bout-en-bout" asr --model ckpt/whisper-bm \\
        --task translate

    # Phase 5 — mêmes modèles convertis pour le CPU (scripts/export_cpu.py) :
    # ce que la quantification int8 coûte en qualité, et la latence réelle
    python -m eval.baselines --device cpu --label "cascade int8" asr \\
        --model modeles-cpu/whisper-bm --with-mt --mt-model modeles-cpu/nllb-bm2fr \\
        --backend ctranslate2

    python -m eval.compare eval/results/*.json

Même corpus, même partition, même graine : les chiffres de chaque phase se
comparent directement. En bout-en-bout, Whisper produit le français lui-même
et son chrF++ se lit sur la même ligne du tableau que celui de la cascade
(« chrF++ (depuis ASR) ») : c'est la comparaison centrale du mémoire.

Pas de bout-en-bout zero-shot : la tâche `translate` de Whisper d'origine ne
produit que de l'anglais. La variante B n'existe qu'après fine-tuning.

Chaque rapport consigne ce qui a été mesuré (modèles, corpus, partition,
graine, appareil, versions) et garde les sorties ligne à ligne pour
l'analyse d'erreurs. Les composants sont ceux du pipeline (`bambara_voice`),
pas des copies : l'évaluation éprouve aussi le code de la démo.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from bambara_voice.config import BAM, FRA, ASRConfig, MTConfig, TTSConfig
from bambara_voice.normalize import fold

from .corpora import (
    BAYELEMABAGA, JELI_ASR, Example, detect_columns, iter_examples, load_split,
    sample_indices,
)
from .metrics import cascade_degradation, score_asr, score_mt

logger = logging.getLogger(__name__)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _latency(rows: list[dict], steps: dict[str, str]) -> dict:
    """Même forme que dans eval.run_eval : temps moyen par étape, total, RTF.
    `steps` associe un nom d'étape à la clé de durée dans chaque ligne."""
    per_step = {name: _mean([r[key] for r in rows]) for name, key in steps.items()}
    totals = [sum(r[key] for key in steps.values()) for r in rows]
    out = {"par_etape_s": {k: round(v, 3) for k, v in per_step.items()},
           "total_moyen_s": round(_mean(totals), 3)}
    rtf = [t / r["duree_audio_s"] for t, r in zip(totals, rows) if r.get("duree_audio_s")]
    if rtf:
        out["rtf_moyen"] = round(_mean(rtf), 3)
    return out


# --- maillons -----------------------------------------------------------------

def run_asr(examples: Iterable[Example], recognizer, translator=None,
            e2e: bool = False) -> tuple[dict, list[dict]]:
    """Entrée de la chaîne, de l'audio bambara jusqu'au français.

    - cascade : WER/CER strict et relâché ; avec un traducteur, chrF++ du
      français obtenu et propagation des erreurs de l'ASR (levier 1).
    - e2e : l'ASR produit directement le français, noté contre la même
      référence que la cascade.
    """
    rows = []
    for ex in examples:
        if ex.audio is None or not (ex.fr if e2e else ex.bm):
            continue
        t0 = time.perf_counter()
        res = recognizer.transcribe(ex.audio)
        t1 = time.perf_counter()
        row = {"id": ex.id, "ref_bm": ex.bm, "ref_fr": ex.fr,
               ("hyp_fr" if e2e else "hyp_bm"): res.text,
               "secondes_asr": t1 - t0, "duree_audio_s": len(ex.audio) / 16_000}
        if translator is not None and not e2e:
            # Traduction de la sortie de l'ASR : c'est le chemin réel de la
            # cascade, et son temps compte dans la latence.
            row["mt_depuis_asr"] = translator.translate(res.text)
            row["secondes_mt"] = time.perf_counter() - t1
            row["coupe_fr"] = getattr(translator, "last_capped", False)
        rows.append(row)
    if not rows:
        raise RuntimeError("aucun exemple avec audio et référence")

    steps = {"asr": "secondes_asr"}
    if translator is not None and not e2e:
        steps["mt_in"] = "secondes_mt"
    report: dict = {"latence": _latency(rows, steps)}

    if e2e:
        report["mt_in_depuis_asr"] = score_mt(
            [r["hyp_fr"] for r in rows], [r["ref_fr"] for r in rows]).to_dict()
        return report, rows

    report["asr"] = score_asr([r["hyp_bm"] for r in rows], [r["ref_bm"] for r in rows]).to_dict()
    with_fr = [r for r in rows if r["ref_fr"]]
    if translator is not None and with_fr:
        for r in with_fr:
            r["mt_depuis_ref"] = translator.translate(r["ref_bm"])
        refs = [r["ref_fr"] for r in with_fr]
        from_asr = score_mt([r["mt_depuis_asr"] for r in with_fr], refs)
        from_ref = score_mt([r["mt_depuis_ref"] for r in with_fr], refs)
        report["mt_in_depuis_asr"] = from_asr.to_dict()
        report["mt_in_depuis_texte_de_reference"] = from_ref.to_dict()
        report["propagation_erreurs"] = cascade_degradation(from_ref, from_asr)
    elif translator is not None:
        logger.warning("aucune traduction française de référence : pas de propagation")
    return report, rows


def run_mt(examples: Iterable[Example], bm2fr, fr2bm,
           split_fr: int | None = None) -> tuple[dict, list[dict]]:
    """chrF++/BLEU dans les deux sens. Vers le bambara, chrF++ aussi sur la
    forme repliée : même logique que le WER relâché, l'écart mesure ce que
    coûte la seule graphie (ɛ/e, ɔ/o...).

    `split_fr` : le français est découpé en segments d'au plus ce nombre de
    mots (bambara_voice.llm.split_for_translation), traduits un à un puis
    mis bout à bout. Même contenu, phrases plus courtes : c'est l'effet de
    la longueur seule sur la traduction vers le bambara (levier 2)."""
    from bambara_voice.llm import split_for_translation

    rows = []
    for ex in examples:
        if not (ex.bm and ex.fr):
            continue
        t0 = time.perf_counter()
        hyp_fr = bm2fr.translate(ex.bm)
        t1 = time.perf_counter()
        segments = split_for_translation(ex.fr, max_words=split_fr) if split_fr else [ex.fr]
        parts, capped_bm = [], False
        for segment in segments:
            parts.append(fr2bm.translate(segment))
            # Arrêtée sur la longueur maximale : presque toujours une boucle.
            capped_bm |= getattr(fr2bm, "last_capped", False)
        row = {"id": ex.id, "ref_bm": ex.bm, "ref_fr": ex.fr,
               "hyp_fr": hyp_fr, "hyp_bm": " ".join(p for p in parts if p),
               "secondes_bm_fr": t1 - t0,
               "secondes_fr_bm": time.perf_counter() - t1,
               "coupe_fr": getattr(bm2fr, "last_capped", False),
               "coupe_bm": capped_bm}
        if split_fr:
            row["segments_fr"] = len(segments)
        rows.append(row)
    if not rows:
        raise RuntimeError("aucune paire bambara-français")

    ref_bm = [r["ref_bm"] for r in rows]
    hyp_bm = [r["hyp_bm"] for r in rows]
    report = {
        "mt_bm_fr": score_mt([r["hyp_fr"] for r in rows], [r["ref_fr"] for r in rows]).to_dict(),
        "mt_fr_bm": score_mt(hyp_bm, ref_bm).to_dict(),
        "mt_fr_bm_replie": score_mt([fold(h) for h in hyp_bm], [fold(r) for r in ref_bm]).to_dict(),
        "latence": _latency(rows, {"mt_bm_fr": "secondes_bm_fr",
                                   "mt_fr_bm": "secondes_fr_bm"}),
        "sorties_coupees": {"bm_fr": _mean([float(r["coupe_fr"]) for r in rows]),
                            "fr_bm": _mean([float(r["coupe_bm"]) for r in rows])},
    }
    if split_fr:
        report["decoupe_fr"] = {
            "max_mots": split_fr,
            "phrases_decoupees": _mean([float(r["segments_fr"] > 1) for r in rows]),
            "segments_moyens": round(_mean([r["segments_fr"] for r in rows]), 2),
        }
    return report, rows


def run_tts(texts: Iterable[tuple[str, str]], synthesizer, out_dir: Path) -> tuple[dict, list[dict]]:
    """Synthèse d'un échantillon + grille MOS à remplir par des auditeurs
    natifs : aucune métrique automatique ne juge la prosodie d'une langue à
    tons non notés (cf. METHODE.md, §5)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item_id, text in texts:
        t0 = time.perf_counter()
        speech = synthesizer.synthesize(text)
        secs = time.perf_counter() - t0
        path = speech.save(out_dir / f"{item_id}.wav")
        rows.append({"id": item_id, "texte_bm": text, "fichier": path.name,
                     "secondes": secs, "duree_audio_s": speech.duration})
    if not rows:
        raise RuntimeError("aucun texte à synthétiser")

    with (out_dir / "mos.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "texte_bm", "fichier", "auditeur", "note_1_a_5", "commentaire"])
        for r in rows:
            writer.writerow([r["id"], r["texte_bm"], r["fichier"], "", "", ""])

    report = {
        "tts": {"n": len(rows), "dossier": str(out_dir),
                "evaluation": "MOS : faire remplir mos.csv par au moins 3 auditeurs natifs"},
        "latence": _latency(rows, {"tts": "secondes"}),
    }
    return report, rows


# --- rapports -----------------------------------------------------------------

def _environment(device: str) -> dict:
    env = {"appareil": device}
    try:
        import torch

        env["torch"] = torch.__version__
        if device.startswith("cuda") and torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    for lib in ("transformers", "datasets"):
        try:
            env[lib] = __import__(lib).__version__
        except ImportError:
            pass
    return env


def write_report(report: dict, rows: list[dict], out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{name}-{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with path.with_suffix(".details.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _short(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def _backend_suffix(args) -> str:
    return f" [ct2 {args.compute_type}]" if args.backend == "ctranslate2" else ""


def _engine_args(sp) -> None:
    sp.add_argument("--backend", choices=["transformers", "ctranslate2"], default="transformers",
                    help="ctranslate2 : modèles convertis par scripts/export_cpu.py")
    sp.add_argument("--compute-type", default="int8", help="ctranslate2 : int8, float32...")


def _mt_args(sp) -> None:
    sp.add_argument("--mt-model", default=MTConfig().model_id)
    sp.add_argument("--plafond", type=float, default=None, metavar="RATIO",
                    help=f"traduction limitée à RATIO × la source + marge, en jetons "
                         f"(défaut : {MTConfig().max_length_ratio:g} ; 0 : pas de plafond, "
                         f"comme en phase 1)")


def _ratio(args) -> float | None:
    """Plafond effectif : celui de MTConfig si l'option est absente."""
    if args.plafond is None:
        return MTConfig().max_length_ratio
    return args.plafond or None


def _mt_engine(args) -> dict:
    return {"backend": args.backend, "compute_type": args.compute_type,
            "max_length_ratio": _ratio(args)}


def _mt_suffix(args) -> str:
    """Le nom ne signale que l'écart au réglage par défaut."""
    if args.plafond is None:
        return ""
    return f" [plafond ×{args.plafond:g}]" if args.plafond else " [sans plafond]"


def _mt_models(args) -> dict:
    return {"plafond_longueur": _ratio(args)}


# --- ligne de commande --------------------------------------------------------

def _load(args, need_audio: bool, need_fr: bool):
    ds = load_split(args.dataset, args.split, args.config)
    cols = detect_columns(ds.features, args.bm_column, args.fr_column,
                          need_audio=need_audio, need_fr=need_fr)
    idx = sample_indices(len(ds), args.limit, args.seed)
    print(f"{args.dataset} [{ds.split}] : {len(ds)} lignes, {len(idx)} évaluées "
          f"(graine {args.seed})\n  schéma : {dict(ds.features)}\n  colonnes : {cols.describe()}")
    corpus = {"id": args.dataset, "config": args.config, "partition": str(ds.split),
              "lignes": len(ds), "evaluees": len(idx), "graine": args.seed,
              "colonnes": cols.describe()}
    return ds, cols, idx, corpus


def cmd_asr(args) -> tuple[dict, list[dict]]:
    from bambara_voice.asr import SpeechRecognizer
    from bambara_voice.mt import Translator

    e2e = args.task == "translate"
    if e2e and (args.with_mt or args.kind != "whisper"):
        raise SystemExit("--task translate : Whisper seul, sans --with-mt ni modèle CTC ou NeMo")
    ds, cols, idx, corpus = _load(args, need_audio=True, need_fr=e2e)
    language = None if args.language == "none" else args.language
    asr_cfg = ASRConfig(model_id=args.model, task=args.task, kind=args.kind,
                        language=language, target_lang=args.target_lang,
                        nemo_decoder=args.nemo_decoder,
                        backend=args.backend, compute_type=args.compute_type)
    recognizer = SpeechRecognizer(asr_cfg, args.device)
    translator = None
    if args.with_mt:
        translator = Translator(MTConfig(model_id=args.mt_model, src_lang=BAM, tgt_lang=FRA,
                                         **_mt_engine(args)), args.device)
    report, rows = run_asr(iter_examples(ds, cols, idx), recognizer, translator, e2e=e2e)

    name = _short(args.model)
    if args.kind == "whisper" and language != "sw":
        name += f" ({language or 'auto'})"
    name += " → fr" if e2e else (f" + {_short(args.mt_model)}" if args.with_mt else "")
    name += _backend_suffix(args) + (_mt_suffix(args) if args.with_mt else "")
    n = report["mt_in_depuis_asr" if e2e else "asr"]["n"]
    return {"nom": args.label or name, "architecture": "e2e" if e2e else "cascade",
            "maillon": "asr", "corpus": corpus, "composition": {"n": n},
            "modeles": {"asr": args.model, "kind": args.kind, "task": args.task,
                        "backend": args.backend,
                        **({"compute_type": args.compute_type}
                           if args.backend == "ctranslate2" else {}),
                        "language": language, "target_lang": args.target_lang,
                        **({"nemo_decoder": args.nemo_decoder or "défaut"}
                           if args.kind == "nemo" else {}),
                        **({"mt_in": args.mt_model, **_mt_models(args)}
                           if args.with_mt else {})},
            **report}, rows


def cmd_mt(args) -> tuple[dict, list[dict]]:
    from bambara_voice.mt import Translator

    ds, cols, idx, corpus = _load(args, need_audio=False, need_fr=True)
    # Un modèle par sens après fine-tuning (scripts/finetune_nllb.py), le
    # même NLLB pour les deux en zero-shot.
    bm2fr_id = args.bm2fr_model or args.mt_model
    fr2bm_id = args.fr2bm_model or args.mt_model
    engine = _mt_engine(args)
    bm2fr = Translator(MTConfig(model_id=bm2fr_id, src_lang=BAM, tgt_lang=FRA, **engine),
                       args.device)
    fr2bm = Translator(MTConfig(model_id=fr2bm_id, src_lang=FRA, tgt_lang=BAM, **engine),
                       args.device)
    examples = iter_examples(ds, cols, idx, with_audio=False)
    if args.min_mots:
        # Après le tirage : mêmes phrases d'une variante à l'autre.
        corpus["filtre"] = f"français d'au moins {args.min_mots} mots"
        examples = (ex for ex in examples if len((ex.fr or "").split()) >= args.min_mots)
    report, rows = run_mt(examples, bm2fr, fr2bm, split_fr=args.decoupe)
    models = _short(bm2fr_id) if bm2fr_id == fr2bm_id else \
        f"{_short(bm2fr_id)} + {_short(fr2bm_id)}"
    suffix = _backend_suffix(args) + _mt_suffix(args)
    suffix += f" [découpe ≤{args.decoupe} mots]" if args.decoupe else ""
    return {"nom": args.label or f"{models} / {_short(args.dataset)}{suffix}",
            "architecture": "traduction", "maillon": "mt", "corpus": corpus,
            "composition": {"n": report["mt_bm_fr"]["n"]},
            "modeles": {"mt_bm_fr": bm2fr_id, "mt_fr_bm": fr2bm_id, "backend": args.backend,
                        **({"compute_type": args.compute_type}
                           if args.backend == "ctranslate2" else {}),
                        **_mt_models(args)}, **report}, rows


def cmd_tts(args) -> tuple[dict, list[dict]]:
    from bambara_voice.tts import SpeechSynthesizer

    ds, cols, _, corpus = _load(args, need_audio=False, need_fr=False)
    # Phrases de longueur raisonnable pour une écoute : ni un mot isolé, ni
    # un paragraphe. On échantillonne large puis on filtre.
    candidates = iter_examples(ds, cols, sample_indices(len(ds), args.limit * 5, args.seed),
                               with_audio=False)
    texts = [(f"tts-{i:03d}", ex.bm) for i, ex in enumerate(
        e for e in candidates if 3 <= len(e.bm.split()) <= 20)][:args.limit]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    synth = SpeechSynthesizer(TTSConfig(model_id=args.tts_model), args.device)
    report, rows = run_tts(texts, synth, args.out / f"tts-{_short(args.tts_model)}-{stamp}")
    return {"nom": args.label or _short(args.tts_model), "architecture": "synthèse",
            "maillon": "tts", "corpus": corpus, "composition": {"n": len(rows)},
            "modeles": {"tts": args.tts_model}, **report}, rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("eval/results"))
    p.add_argument("--device", default=None, help="cpu, cuda... (défaut : cuda si disponible)")
    p.add_argument("--label", default=None,
                   help="nom de la colonne dans le tableau comparatif")
    sub = p.add_subparsers(dest="maillon", required=True)

    def corpus_args(sp, default_dataset, default_limit):
        sp.add_argument("--dataset", default=default_dataset)
        sp.add_argument("--config", default=None, help="sous-ensemble du jeu")
        sp.add_argument("--split", default=None, help="défaut : test, sinon validation")
        sp.add_argument("--limit", type=int, default=default_limit)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--bm-column", default=None)
        sp.add_argument("--fr-column", default=None)

    sp = sub.add_parser("asr", help="entrée de la chaîne : cascade ou bout-en-bout")
    corpus_args(sp, JELI_ASR, 300)
    sp.add_argument("--model", default="openai/whisper-small")
    sp.add_argument("--kind", choices=["whisper", "ctc", "nemo"], default="whisper")
    sp.add_argument("--task", choices=["transcribe", "translate"], default="transcribe",
                    help="translate : bout-en-bout, Whisper produit le français")
    sp.add_argument("--language", default="sw",
                    help="token de langue Whisper ('none' : détection automatique)")
    sp.add_argument("--target-lang", default=None, help="adaptateur MMS, ex. bam")
    sp.add_argument("--nemo-decoder", choices=["ctc"], default=None,
                    help="NeMo hybride (Soloni) : décodeur CTC au lieu du TDT par défaut")
    sp.add_argument("--with-mt", action="store_true",
                    help="cascade complète jusqu'au français, et propagation d'erreurs")
    _mt_args(sp)
    _engine_args(sp)
    sp.set_defaults(func=cmd_asr)

    sp = sub.add_parser("mt", help="chrF++/BLEU bm->fr et fr->bm")
    corpus_args(sp, BAYELEMABAGA, 500)
    _mt_args(sp)
    sp.add_argument("--bm2fr-model", default=None, help="défaut : --mt-model")
    sp.add_argument("--fr2bm-model", default=None, help="défaut : --mt-model")
    sp.add_argument("--decoupe", type=int, default=None, metavar="MOTS",
                    help="vers le bambara, découper le français en segments d'au plus MOTS "
                         "mots aux virgules (levier 2)")
    sp.add_argument("--min-mots", type=int, default=None, metavar="MOTS",
                    help="ne garder que les paires dont le français a au moins MOTS mots")
    _engine_args(sp)
    sp.set_defaults(func=cmd_mt)

    sp = sub.add_parser("tts", help="échantillon audio + grille MOS")
    corpus_args(sp, JELI_ASR, 20)
    sp.add_argument("--tts-model", default=TTSConfig().model_id)
    sp.set_defaults(func=cmd_tts)

    args = p.parse_args(argv)
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    report, rows = args.func(args)
    report = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              **report, "environnement": _environment(args.device)}
    prefix = f"asr-{report['architecture']}" if args.maillon == "asr" else args.maillon
    path = write_report(report, rows, args.out, prefix)
    print(json.dumps({k: v for k, v in report.items() if k != "corpus"},
                     ensure_ascii=False, indent=2))
    print(f"\nRapport : {path}")


if __name__ == "__main__":
    main()
