"""Tableau comparatif des rapports d'évaluation.

Produit le tableau Markdown à coller dans le mémoire.

    python -m eval.compare eval/results/*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = [
    ("Architecture", lambda r: r.get("architecture", "?")),
    ("Items", lambda r: r.get("composition", {}).get("n", 0)),
    ("WER strict", lambda r: _pct(r.get("asr", {}).get("wer"))),
    ("WER relâché", lambda r: _pct(r.get("asr", {}).get("wer_folded"))),
    ("Écart orthographique", lambda r: _pct(r.get("asr", {}).get("orthographic_gap"))),
    ("CER strict", lambda r: _pct(r.get("asr", {}).get("cer"))),
    ("chrF++ (depuis ASR)", lambda r: _num(r.get("mt_in_depuis_asr", {}).get("chrf"))),
    ("chrF++ (texte de réf.)", lambda r: _num(r.get("mt_in_depuis_texte_de_reference", {}).get("chrf"))),
    ("Perte due à l'ASR", lambda r: _num(r.get("propagation_erreurs", {}).get("perte_absolue"))),
    ("BLEU (depuis ASR)", lambda r: _num(r.get("mt_in_depuis_asr", {}).get("bleu"))),
    # Références zero-shot de la traduction seule (eval.baselines mt).
    ("chrF++ bm→fr", lambda r: _num(r.get("mt_bm_fr", {}).get("chrf"))),
    ("chrF++ fr→bm", lambda r: _num(r.get("mt_fr_bm", {}).get("chrf"))),
    ("chrF++ fr→bm (replié)", lambda r: _num(r.get("mt_fr_bm_replie", {}).get("chrf"))),
    ("BLEU bm→fr", lambda r: _num(r.get("mt_bm_fr", {}).get("bleu"))),
    ("BLEU fr→bm", lambda r: _num(r.get("mt_fr_bm", {}).get("bleu"))),
    ("Sorties coupées bm→fr", lambda r: _pct(r.get("sorties_coupees", {}).get("bm_fr"))),
    ("Sorties coupées fr→bm", lambda r: _pct(r.get("sorties_coupees", {}).get("fr_bm"))),
    ("Phrases découpées (fr→bm)", lambda r: _pct(r.get("decoupe_fr", {}).get("phrases_decoupees"))),
    ("Latence totale (s)", lambda r: _num(r.get("latence", {}).get("total_moyen_s"))),
    ("RTF", lambda r: _num(r.get("latence", {}).get("rtf_moyen"))),
    ("Couverture gabarits", lambda r: _pct(r.get("gabarits", {}).get("couverture"))),
]


def _pct(v) -> str:
    return "—" if v is None else f"{v * 100:.1f} %"


def _num(v) -> str:
    return "—" if v is None else f"{v:.2f}"


def to_markdown(reports: list[dict]) -> str:
    names = [r.get("nom") or r.get("architecture", "?") for r in reports]
    lines = ["| Métrique | " + " | ".join(names) + " |",
             "|---" * (len(names) + 1) + "|"]
    for label, getter in ROWS:
        cells = [str(getter(r)) for r in reports]
        if all(c in ("—", "0") for c in cells):
            continue  # ne pas encombrer avec des lignes vides
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in args.reports]
    print(to_markdown(reports))


if __name__ == "__main__":
    main()
