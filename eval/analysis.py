"""Analyse d'erreurs à partir des sorties ligne à ligne des rapports.

    python -m eval.analysis eval/results/*.json

Les scores disent combien un système se trompe ; cette analyse dit comment.
Pour chaque rapport de eval.baselines, elle lit ses sorties ligne à ligne
(`.details.jsonl`) :

- **ASR** : erreurs décomposées en substitutions, suppressions et insertions ;
  part des substitutions purement orthographiques (bɛ → be) ; hallucinations
  (sortie beaucoup plus longue que la référence, boucles de répétition) ; WER
  selon la longueur de l'énoncé ; confusions de mots les plus fréquentes.
- **Traduction** : chrF++ selon la longueur ; sorties vides, trop courtes ou
  trop longues ; phrases recopiées sans être traduites.
- **Exemples** : meilleur, médian et pire énoncé, pour illustrer le mémoire.

Tout est calculé au niveau du corpus, comme les scores, et dans la même
normalisation (eval.metrics).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from bambara_voice.normalize import fold, normalize, strip_punctuation

from .significance import chrf_stats, corpus_score

LENGTH_BUCKETS = ((1, 3), (4, 7), (8, 15), (16, 10_000))
HALLUCINATION_RATIO = 2.0   # sortie au moins deux fois plus longue que la référence
REPEAT_NGRAM, REPEAT_MIN = 3, 3


def _strict(text: str) -> str:
    return strip_punctuation(normalize(text or ""))


def _bucket(n: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}–{hi}" if hi < 10_000 else f"{lo}+"
    return "0"


def has_loop(text: str, n: int = REPEAT_NGRAM, times: int = REPEAT_MIN) -> bool:
    """Vrai si un même n-gramme de mots revient au moins `times` fois : la
    signature des boucles de génération (« a bɛ a bɛ a bɛ… »)."""
    words = text.split()
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return bool(grams) and grams.most_common(1)[0][1] >= times


# --- ASR ----------------------------------------------------------------------

def asr_analysis(rows: list[dict], hyp_key: str = "hyp_bm", ref_key: str = "ref_bm",
                 top: int = 10) -> dict:
    import jiwer

    rows = [r for r in rows if _strict(r.get(ref_key, ""))]
    sub = dele = ins = hits = ortho_sub = 0
    confusions: Counter = Counter()
    deleted: Counter = Counter()
    inserted: Counter = Counter()
    per_bucket: dict[str, list[int]] = {}
    ratios, loops, empty, per_item = [], 0, 0, []

    for r in rows:
        ref, hyp = _strict(r[ref_key]), _strict(r.get(hyp_key, ""))
        ref_words, hyp_words = ref.split(), hyp.split()
        ratios.append(len(hyp_words) / len(ref_words))
        empty += not hyp_words
        loops += has_loop(hyp)
        if hyp_words:
            out = jiwer.process_words(ref, hyp)
            s, d, i, h = out.substitutions, out.deletions, out.insertions, out.hits
            for chunk in out.alignments[0]:
                rw = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
                hw = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
                if chunk.type == "substitute":
                    for a, b in zip(rw, hw):
                        confusions[(a, b)] += 1
                        ortho_sub += fold(a) == fold(b)
                elif chunk.type == "delete":
                    deleted.update(rw)
                elif chunk.type == "insert":
                    inserted.update(hw)
        else:
            s, d, i, h = 0, len(ref_words), 0, 0
            deleted.update(ref_words)
        sub, dele, ins, hits = sub + s, dele + d, ins + i, hits + h
        b = per_bucket.setdefault(_bucket(len(ref_words)), [0, 0, 0])
        b[0] += s + d + i
        b[1] += len(ref_words)
        b[2] += 1
        per_item.append(((s + d + i) / len(ref_words), r))

    n_ref = sub + dele + hits
    ratios = np.array(ratios) if ratios else np.zeros(1)
    return {
        "n": len(rows),
        "wer": (sub + dele + ins) / n_ref if n_ref else 0.0,
        "substitutions": sub / n_ref if n_ref else 0.0,
        "suppressions": dele / n_ref if n_ref else 0.0,
        "insertions": ins / n_ref if n_ref else 0.0,
        "substitutions_orthographiques": ortho_sub / sub if sub else 0.0,
        "sorties_vides": empty / len(rows) if rows else 0.0,
        "hallucinations": float(np.mean(ratios >= HALLUCINATION_RATIO)),
        "boucles": loops / len(rows) if rows else 0.0,
        "ratio_longueur_median": float(np.median(ratios)),
        "wer_par_longueur": {k: {"wer": v[0] / v[1], "n": v[2]}
                             for k, v in sorted(per_bucket.items(), key=_bucket_order)},
        "confusions": [(f"{a} → {b}", c) for (a, b), c in confusions.most_common(top)],
        "mots_supprimes": deleted.most_common(top),
        "mots_inseres": inserted.most_common(top),
        "exemples": _examples(per_item, ref_key, hyp_key),
    }


def _bucket_order(item):
    key = item[0]
    return int(key.split("–")[0].rstrip("+")) if key != "0" else 0


def _examples(per_item: list, ref_key: str, hyp_key: str, src_key: str | None = None) -> dict:
    """Meilleur, médian et pire énoncé : de quoi illustrer chaque score.
    `per_item` : (score, ligne), le plus petit score étant le meilleur."""
    if not per_item:
        return {}
    ordered = sorted(per_item, key=lambda x: x[0])
    picks = {"meilleur": ordered[0], "median": ordered[len(ordered) // 2], "pire": ordered[-1]}
    out = {}
    for name, (score, r) in picks.items():
        out[name] = {"score": round(score, 3), "reference": r[ref_key],
                     "sortie": r.get(hyp_key, "")}
        if src_key:
            out[name]["source"] = r[src_key]
    return out


# --- traduction ---------------------------------------------------------------

def mt_analysis(rows: list[dict], src_key: str, hyp_key: str, ref_key: str) -> dict:
    rows = [r for r in rows if (r.get(ref_key) or "").strip() and hyp_key in r]
    if not rows:
        return {"n": 0}
    hyps = [r[hyp_key] or "" for r in rows]
    refs = [r[ref_key] for r in rows]
    stats = chrf_stats(hyps, refs)

    per_bucket: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        per_bucket.setdefault(_bucket(len(r[src_key].split())), []).append(i)
    ratios = np.array([len(h.split()) / max(1, len(ref.split())) for h, ref in zip(hyps, refs)])
    # Recopie : la sortie reprend la phrase source au lieu de la traduire.
    copies = sum(fold(h) == fold(r[src_key]) and bool(h.strip()) for h, r in zip(hyps, rows))
    per_item = [(corpus_score("chrf", stats[i]), r) for i, r in enumerate(rows)]
    return {
        "n": len(rows),
        "chrf": corpus_score("chrf", stats.sum(axis=0)),
        "chrf_par_longueur_source": {
            k: {"chrf": corpus_score("chrf", stats[idx].sum(axis=0)), "n": len(idx)}
            for k, idx in sorted(per_bucket.items(), key=_bucket_order)},
        "sorties_vides": float(np.mean([not h.strip() for h in hyps])),
        "sorties_trop_courtes": float(np.mean(ratios < 0.5)),
        "sorties_trop_longues": float(np.mean(ratios >= HALLUCINATION_RATIO)),
        "boucles": float(np.mean([has_loop(_strict(h)) for h in hyps])),
        "recopies_de_la_source": copies / len(rows),
        # Score négatif : pour chrF++, le plus haut est le meilleur.
        "exemples": _examples([(-s, r) for s, r in per_item], ref_key, hyp_key, src_key),
    }


# --- lecture des rapports et présentation -------------------------------------

def analyse_report(path: Path) -> tuple[str, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in
            path.with_suffix(".details.jsonl").read_text(encoding="utf-8").splitlines() if line]
    name = report.get("nom") or path.stem
    maillon, arch = report.get("maillon"), report.get("architecture")
    out: dict = {}
    if maillon == "asr" and arch != "e2e":
        out["asr"] = asr_analysis(rows)
        if any("mt_depuis_asr" in r for r in rows):
            out["mt_depuis_asr"] = mt_analysis(rows, "ref_bm", "mt_depuis_asr", "ref_fr")
    elif maillon == "asr":
        out["e2e"] = mt_analysis(rows, "ref_bm", "hyp_fr", "ref_fr")
    elif maillon == "mt":
        out["bm_fr"] = mt_analysis(rows, "ref_bm", "hyp_fr", "ref_fr")
        out["fr_bm"] = mt_analysis(rows, "ref_fr", "hyp_bm", "ref_bm")
    return name, out


def _counts(items: list) -> str:
    return ", ".join(f"{w} ({n})" for w, n in items) or "—"


def _pct(v: float) -> str:
    return f"{100 * v:.1f} %"


def to_markdown(name: str, analysis: dict) -> str:
    lines = [f"### {name}", ""]
    if "asr" in analysis:
        a = analysis["asr"]
        lines += [
            f"**ASR** ({a['n']} énoncés) — WER {_pct(a['wer'])} = substitutions "
            f"{_pct(a['substitutions'])} + suppressions {_pct(a['suppressions'])} + "
            f"insertions {_pct(a['insertions'])}.",
            "",
            f"- Substitutions purement orthographiques : {_pct(a['substitutions_orthographiques'])}",
            f"- Sorties vides : {_pct(a['sorties_vides'])} ; hallucinations (sortie ≥ "
            f"{HALLUCINATION_RATIO:.0f}× la référence) : {_pct(a['hallucinations'])} ; "
            f"boucles de répétition : {_pct(a['boucles'])}",
            f"- Longueur sortie / référence, médiane : {a['ratio_longueur_median']:.2f}",
            "- WER selon la longueur de l'énoncé (mots) : " + " ; ".join(
                f"{k} : {_pct(v['wer'])} (n={v['n']})" for k, v in a["wer_par_longueur"].items()),
            "- Confusions fréquentes : " + _counts(a["confusions"]),
            "- Mots les plus supprimés : " + _counts(a["mots_supprimes"]),
            "- Mots les plus insérés : " + _counts(a["mots_inseres"]),
            "",
        ]
        lines += _example_lines(a["exemples"], "WER")
    for key, label in (("mt_depuis_asr", "Traduction de la sortie de l'ASR"),
                       ("e2e", "Traduction directe (bout-en-bout)"),
                       ("bm_fr", "bambara → français"), ("fr_bm", "français → bambara")):
        if key in analysis and analysis[key].get("n"):
            m = analysis[key]
            lines += [
                f"**{label}** ({m['n']} phrases) — chrF++ {m['chrf']:.1f}.",
                "",
                "- chrF++ selon la longueur de la source (mots) : " + " ; ".join(
                    f"{k} : {v['chrf']:.1f} (n={v['n']})"
                    for k, v in m["chrf_par_longueur_source"].items()),
                f"- Sorties vides : {_pct(m['sorties_vides'])} ; trop courtes (< moitié de la "
                f"référence) : {_pct(m['sorties_trop_courtes'])} ; trop longues : "
                f"{_pct(m['sorties_trop_longues'])} ; boucles : {_pct(m['boucles'])}",
                f"- Source recopiée sans traduction : {_pct(m['recopies_de_la_source'])}",
                "",
            ]
            lines += _example_lines(m["exemples"], "chrF++", negate=True)
    return "\n".join(lines)


def _example_lines(examples: dict, metric: str, negate: bool = False) -> list[str]:
    lines = []
    for name, ex in examples.items():
        score = -ex["score"] if negate else ex["score"]
        shown = f"{score:.1f}" if negate else _pct(score)
        lines.append(f"- *{name}* ({metric} {shown}) — "
                     + (f"source : « {ex['source']} » ; " if "source" in ex else "")
                     + f"référence : « {ex['reference']} » ; sortie : « {ex['sortie']} »")
    return lines + [""]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reports", nargs="+", type=Path)
    p.add_argument("--json", type=Path, default=None, help="écrire aussi l'analyse en JSON")
    args = p.parse_args(argv)

    results = {}
    for path in args.reports:
        if path.name.endswith(".details.jsonl"):
            continue
        name, analysis = analyse_report(path)
        if analysis:
            results[name] = analysis
            print(to_markdown(name, analysis))
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=list),
                             encoding="utf-8")


if __name__ == "__main__":
    main()
