"""Extrait un échantillon d'un corpus public au format du jeu de test.

    python scripts/export_corpus_sample.py --limit 50 --out data/echantillons/jeli-test
    python -m eval.run_eval --testset data/echantillons/jeli-test/testset.jsonl \\
        --config modeles-cpu/config.json --arch cascade --synthesize

Sert à faire tourner `eval.run_eval` — la chaîne complète, LLM et synthèse
compris — avant que le jeu de test maison (phase 2) soit prêt, en
particulier pour mesurer la latence de bout en bout sur CPU (phase 5).
Même graine par défaut que `eval.baselines` : les premiers énoncés tirés
sont les mêmes.

Ce n'est pas un substitut au jeu maison : Jeli-ASR a servi à l'entraînement,
et ses locuteurs ne sont pas ceux que l'assistant entendra.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.corpora import (  # noqa: E402
    JELI_ASR, detect_columns, iter_examples, load_split, sample_indices,
)


def export(ds, cols, indices: list[int], out: Path) -> int:
    import soundfile as sf

    (out / "audio").mkdir(parents=True, exist_ok=True)
    lines = []
    for n, ex in enumerate(iter_examples(ds, cols, indices)):
        rel = f"audio/{n:04d}.wav"
        sf.write(str(out / rel), ex.audio, 16_000, subtype="PCM_16")
        lines.append({
            "id": f"{ds.split}-{ex.id}",
            "audio": rel,
            "transcript_bm": ex.bm,
            "translation_fr": ex.fr,
            "duration_s": round(len(ex.audio) / 16_000, 2),
        })
    with (out / "testset.jsonl").open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return len(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=JELI_ASR)
    p.add_argument("--config", default=None, help="sous-ensemble du jeu")
    p.add_argument("--split", default=None, help="défaut : test, sinon validation")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bm-column", default=None)
    p.add_argument("--fr-column", default=None)
    p.add_argument("--out", type=Path, default=Path("data/echantillons/jeli-test"))
    args = p.parse_args(argv)

    ds = load_split(args.dataset, args.split, args.config)
    cols = detect_columns(ds.features, args.bm_column, args.fr_column, need_audio=True)
    indices = sample_indices(len(ds), args.limit, args.seed)
    n = export(ds, cols, indices, args.out)
    print(f"{n} énoncés de {args.dataset} [{ds.split}] -> {args.out / 'testset.jsonl'}")


if __name__ == "__main__":
    main()
