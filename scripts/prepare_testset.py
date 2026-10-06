"""Prépare et valide le jeu de test : conversion audio + contrôles de qualité.

    python scripts/prepare_testset.py data/testset/testset.jsonl --convert

Vérifie que chaque audio existe, le convertit en WAV mono 16 kHz si demandé,
renseigne les durées manquantes et signale les problèmes de composition qui
fausseraient l'évaluation (trop peu de locuteurs, aucun énoncé spontané...).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.dataset import load_testset, describe, MIN_ITEMS, MIN_SPEAKERS  # noqa: E402
from bambara_voice.normalize import orthography_ratio  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("testset", type=Path)
    p.add_argument("--convert", action="store_true",
                   help="réécrire les audios en WAV mono 16 kHz")
    p.add_argument("--write-durations", action="store_true",
                   help="renseigner duration_s dans le JSONL")
    args = p.parse_args()

    items = load_testset(args.testset)
    root = args.testset.parent
    problems: list[str] = []

    for item in items:
        path = item.audio_path(root)
        if not path.exists():
            problems.append(f"{item.id}: audio introuvable ({path})")
            continue
        if args.convert or args.write_durations:
            import librosa
            import soundfile as sf

            wav, _ = librosa.load(str(path), sr=16_000, mono=True)
            item.duration_s = round(len(wav) / 16_000, 2)
            if args.convert:
                sf.write(str(path.with_suffix(".wav")), wav, 16_000, subtype="PCM_16")

    stats = describe(items)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # --- contrôles de composition ------------------------------------------
    if stats["n"] < MIN_ITEMS:
        problems.append(
            f"seulement {stats['n']} items : en dessous de {MIN_ITEMS}, "
            f"l'intervalle de confiance sur le WER est trop large pour conclure"
        )
    if stats["locuteurs"] < MIN_SPEAKERS:
        problems.append(
            f"seulement {stats['locuteurs']} locuteurs : viser au moins {MIN_SPEAKERS} "
            f"pour que le WER ne mesure pas une voix particulière"
        )
    if stats["part_spontane"] == 0:
        problems.append("aucun énoncé spontané : les scores seront optimistes")
    if stats["avec_traduction_fr"] < stats["n"]:
        problems.append(
            f"{stats['n'] - stats['avec_traduction_fr']} items sans traduction "
            f"française : la variante bout-en-bout ne pourra pas être évaluée dessus"
        )

    ratio = orthography_ratio([i.transcript_bm for i in items])
    print(f"\nPart de transcriptions en orthographe officielle : {ratio:.0%}")
    if ratio < 0.9:
        problems.append(
            f"seulement {ratio:.0%} des transcriptions utilisent ɛ/ɔ/ɲ/ŋ : "
            f"harmoniser avant d'évaluer, sinon le WER strict n'a pas de sens"
        )

    if args.write_durations:
        with args.testset.open("w", encoding="utf-8") as fh:
            for item in items:
                row = {k: v for k, v in vars(item).items() if k != "extra"}
                row.update(item.extra)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Durées écrites dans {args.testset}")

    if problems:
        print("\nÀ corriger :")
        for pb in problems:
            print(f"  - {pb}")
        sys.exit(1)
    print("\nJeu de test conforme.")


if __name__ == "__main__":
    main()
