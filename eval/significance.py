"""Intervalles de confiance et test apparié, à partir des rapports existants.

    python -m eval.significance eval/results/*.json \\
        --noms "cascade affinée" "bout-en-bout affiné"

Sur quelques centaines d'énoncés, un écart de un ou deux points peut n'être
que du bruit d'échantillonnage. On le vérifie par bootstrap (Koehn, 2004) :

- **intervalle de confiance** de chaque score : on retire au hasard, avec
  remise, autant d'énoncés qu'il y en a, et l'on recalcule le score, 1 000
  fois ; l'IC à 95 % est l'intervalle des percentiles 2,5 et 97,5 ;
- **test apparié** entre deux systèmes : les *mêmes* tirages servent aux
  deux, ce qui est possible parce que toutes les variantes sont évaluées sur
  les mêmes énoncés (cf. METHODE.md, §5). Si l'IC de l'écart contient zéro,
  la différence n'est pas établie.

Les scores sont recalculés au niveau du corpus à chaque tirage, à partir de
statistiques additives par énoncé (erreurs et mots de référence pour le WER ;
n-grammes de caractères et de mots pour chrF++). On ne fait donc pas de
moyenne de scores par phrase, qui donnerait une autre métrique.

**Unité de tirage.** Sur le jeu maison, chaque locuteur enregistre des
dizaines d'énoncés : ses énoncés se ressemblent (voix, débit, micro), ils ne
sont pas indépendants. Tirer les énoncés un par un ferait comme si l'on avait
autant de voix que d'énoncés, et donnerait des IC trop étroits. Avec
`--unite locuteur`, on tire donc des *locuteurs* avec remise, chacun avec
tous ses énoncés (bootstrap par grappes). C'est le choix par défaut (`auto`)
dès que toutes les sorties portent un locuteur. En dessous de 8 locuteurs,
ces IC sont eux-mêmes peu fiables : le tableau le signale.

Le premier système nommé sert de référence : chaque autre est comparé à lui.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bambara_voice.normalize import fold, normalize, strip_punctuation

LOWER_IS_BETTER = {"WER strict", "WER relâché"}
UNITS = ("auto", "enonce", "locuteur")
MIN_CLUSTERS = 8  # même seuil que eval.dataset.MIN_SPEAKERS


@dataclass
class Series:
    """Une métrique d'un rapport, décomposée en statistiques par énoncé."""

    label: str            # libellé de la ligne, comme dans eval.compare
    kind: str             # "wer" ou "chrf"
    ids: list[str]
    stats: np.ndarray     # (énoncés, k), additives
    speakers: list[str] | None = None  # locuteur de chaque énoncé, s'il est connu
    refs: list[str] | None = None      # référence de chaque énoncé

    def score(self, summed: np.ndarray) -> float:
        return corpus_score(self.kind, summed)

    @property
    def value(self) -> float:
        return self.score(self.stats.sum(axis=0))


# --- statistiques additives ---------------------------------------------------

def wer_stats(hyps: list[str], refs: list[str], folded: bool = False) -> np.ndarray:
    """(erreurs, mots de référence) par énoncé, avec la normalisation de
    eval.metrics.score_asr : la somme redonne exactement le WER du corpus."""
    import jiwer

    prep = fold if folded else (lambda t: strip_punctuation(normalize(t)))
    rows = []
    for h, r in zip(hyps, refs):
        h, r = prep(h), prep(r)
        if not r.strip():
            rows.append((0, 0))
            continue
        if not h.strip():
            rows.append((len(r.split()), len(r.split())))  # tout est supprimé
            continue
        o = jiwer.process_words(r, h)
        rows.append((o.substitutions + o.deletions + o.insertions,
                     o.substitutions + o.deletions + o.hits))
    return np.asarray(rows, dtype=np.float64)


def chrf_stats(hyps: list[str], refs: list[str]) -> np.ndarray:
    """Statistiques de n-grammes de chrF++ par énoncé (sacrebleu). Leur somme
    redonne exactement le chrF++ du corpus : c'est ainsi que sacrebleu fait
    lui-même ses tests de significativité."""
    from sacrebleu.metrics import CHRF

    return np.asarray(CHRF(word_order=2)._extract_corpus_statistics(hyps, [refs]),
                      dtype=np.float64)


def corpus_score(kind: str, summed: np.ndarray) -> float:
    if kind == "wer":
        return float(summed[0] / summed[1]) if summed[1] else 0.0
    from sacrebleu.metrics import CHRF

    return float(CHRF(word_order=2)._compute_score_from_stats(list(summed)).score)


# --- lecture des rapports -----------------------------------------------------

def _details(report_path: Path) -> list[dict]:
    path = report_path.with_suffix(".details.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : le rapport a besoin de ses sorties "
                                f"ligne à ligne")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _series(label, kind, rows, id_key, hyp_key, ref_key, folded=False) -> Series | None:
    rows = [r for r in rows if r.get(ref_key) and hyp_key in r]
    if not rows:
        return None
    hyps = [r[hyp_key] or "" for r in rows]
    refs = [r[ref_key] for r in rows]
    stats = wer_stats(hyps, refs, folded) if kind == "wer" else chrf_stats(
        [fold(h) for h in hyps] if folded else hyps, [fold(r) for r in refs] if folded else refs)
    speakers = [str(r.get("speaker") or "") for r in rows]
    return Series(label, kind, [str(r[id_key]) for r in rows], stats,
                  speakers if all(speakers) else None, refs)


def load_series(report_path: str | Path) -> tuple[str, dict[str, Series]]:
    """Séries disponibles dans un rapport de eval.baselines ou eval.run_eval,
    sous les mêmes libellés que dans eval.compare."""
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    name = report.get("nom") or report.get("architecture", report_path.stem)
    rows = _details(report_path)
    arch, maillon = report.get("architecture"), report.get("maillon")

    if maillon == "mt":
        specs = [("chrF++ bm→fr", "chrf", "id", "hyp_fr", "ref_fr", False),
                 ("chrF++ fr→bm", "chrf", "id", "hyp_bm", "ref_bm", False),
                 ("chrF++ fr→bm (replié)", "chrf", "id", "hyp_bm", "ref_bm", True)]
    elif maillon == "asr":
        fr_hyp = "hyp_fr" if arch == "e2e" else "mt_depuis_asr"
        specs = [("WER strict", "wer", "id", "hyp_bm", "ref_bm", False),
                 ("WER relâché", "wer", "id", "hyp_bm", "ref_bm", True),
                 ("chrF++ (depuis ASR)", "chrf", "id", fr_hyp, "ref_fr", False)]
    elif maillon is None and arch in ("cascade", "e2e"):  # eval.run_eval
        specs = [("WER strict", "wer", "item", "source_bm", "ref_transcript_bm", False),
                 ("WER relâché", "wer", "item", "source_bm", "ref_transcript_bm", True),
                 ("chrF++ (depuis ASR)", "chrf", "item", "source_fr", "ref_translation_fr",
                  False)]
        if arch == "e2e":
            specs = specs[2:]  # pas de transcription bambara en bout-en-bout
    else:
        specs = []

    series = {}
    for spec in specs:
        s = _series(spec[0], spec[1], rows, *spec[2:])
        if s is not None:
            series[s.label] = s
    return name, series


# --- bootstrap ----------------------------------------------------------------

def _percentiles(values: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _check_same_items(systems: list[Series], common: list[str]) -> None:
    """Le test apparié suppose les mêmes énoncés. Un même identifiant ne le
    garantit pas : entre deux exécutions, le corpus a pu changer sur le Hub."""
    known = [s for s in systems if s.refs]
    if len(known) < 2:
        return
    first = dict(zip(known[0].ids, known[0].refs))
    for s in known[1:]:
        other = dict(zip(s.ids, s.refs))
        differ = [i for i in common if other[i] != first[i]]
        if differ:
            raise ValueError(f"« {s.label} » : {len(differ)} énoncés portent le même identifiant "
                             f"mais une autre référence (ex. {differ[0]}) : ce ne sont pas les "
                             f"mêmes échantillons")


def _units(systems: list[Series], common: list[str], unit: str) -> tuple[str, np.ndarray]:
    """Unité de tirage effective, et l'unité de chaque énoncé commun."""
    speakers = systems[0].speakers
    if unit == "auto":
        unit = "locuteur" if all(s.speakers for s in systems) else "enonce"
    if unit == "enonce":
        return unit, np.arange(len(common))
    if not speakers:
        raise ValueError("tirage par locuteur impossible : les sorties ne disent pas qui parle "
                         "(champ speaker du jeu de test)")
    by_id = dict(zip(systems[0].ids, speakers))
    for s in systems[1:]:
        other = dict(zip(s.ids, s.speakers or []))
        if any(other.get(i, by_id[i]) != by_id[i] for i in common):
            raise ValueError(f"« {s.label} » n'attribue pas les énoncés aux mêmes locuteurs")
    labels = sorted({by_id[i] for i in common})
    index = {spk: k for k, spk in enumerate(labels)}
    return unit, np.array([index[by_id[i]] for i in common])


def bootstrap(systems: list[Series], n_boot: int = 1000, seed: int = 0,
              alpha: float = 0.05, unit: str = "auto") -> dict:
    """IC de chaque système et test apparié de chacun contre le premier, sur
    les énoncés communs à tous. `unit` : ce que l'on tire avec remise, les
    énoncés ou les locuteurs (« auto » : les locuteurs s'ils sont connus)."""
    if unit not in UNITS:
        raise ValueError(f"unité inconnue : {unit} ; possibles : {UNITS}")
    common = sorted(set.intersection(*(set(s.ids) for s in systems)))
    if not common:
        raise ValueError("aucun énoncé commun : les rapports portent sur des échantillons différents")
    _check_same_items(systems, common)
    positions = [np.array([s.ids.index(i) for i in common]) for s in systems]
    unit, unit_of = _units(systems, common, unit)
    n_units = int(unit_of.max()) + 1

    # Statistiques additives : celles d'un locuteur sont la somme de celles de
    # ses énoncés, et un tirage se calcule sur ces sommes.
    per_unit = []
    for s, pos in zip(systems, positions):
        agg = np.zeros((n_units, s.stats.shape[1]))
        np.add.at(agg, unit_of, s.stats[pos])
        per_unit.append(agg)

    rng = np.random.default_rng(seed)
    samples = rng.integers(0, n_units, size=(n_boot, n_units))
    boots = [np.array([s.score(agg[smp].sum(axis=0)) for smp in samples])
             for s, agg in zip(systems, per_unit)]
    observed = [s.score(agg.sum(axis=0)) for s, agg in zip(systems, per_unit)]
    out = {"n": len(common), "n_boot": n_boot, "unite": unit, "n_unites": n_units,
           "systemes": []}
    for i, s in enumerate(systems):
        entry = {"score": observed[i], "ic": _percentiles(boots[i], alpha)}
        if i > 0:
            diff = boots[i] - boots[0]
            delta = observed[i] - observed[0]
            # p bilatéral : part des tirages où l'écart s'annule ou s'inverse.
            flipped = np.mean(diff <= 0) if delta > 0 else np.mean(diff >= 0)
            entry.update({"ecart": delta, "ic_ecart": _percentiles(diff, alpha),
                          "p": float(min(1.0, 2 * flipped)) if delta else 1.0})
        out["systemes"].append(entry)
    return out


# --- présentation -------------------------------------------------------------

def _fmt(label: str, v: float) -> str:
    return f"{100 * v:.1f}" if label.startswith("WER") else f"{v:.1f}"


def to_markdown(names: list[str], results: dict[str, dict], alpha: float = 0.05) -> str:
    conf = f"{100 * (1 - alpha):.0f} %"
    lines = ["| Métrique | n | " + " | ".join(names) + " |",
             "|---" * (len(names) + 2) + "|"]
    notes = []
    for label, res in results.items():
        cells = []
        for i, entry in enumerate(res["systemes"]):
            lo, hi = entry["ic"]
            cell = f"{_fmt(label, entry['score'])} [{_fmt(label, lo)} ; {_fmt(label, hi)}]"
            if i > 0:
                dlo, dhi = entry["ic_ecart"]
                sign = "+" if entry["ecart"] >= 0 else ""
                significant = dlo > 0 or dhi < 0
                cell += (f"<br>écart {sign}{_fmt(label, entry['ecart'])} "
                         f"[{_fmt(label, dlo)} ; {_fmt(label, dhi)}], p = {entry['p']:.3f}"
                         + ("" if significant else " (n.s.)"))
            cells.append(cell)
        lines.append(f"| {label} | {res['n']} | " + " | ".join(cells) + " |")
        if label in LOWER_IS_BETTER:
            notes.append(label)
    lines.append("")
    text = f"Entre crochets : IC à {conf} par bootstrap ({res['n_boot']} tirages"
    if res["unite"] == "locuteur":
        text += f", en tirant les locuteurs ({res['n_unites']}) avec tous leurs énoncés)."
        if res["n_unites"] < MIN_CLUSTERS:
            text += (f" **Moins de {MIN_CLUSTERS} locuteurs : ces intervalles sont peu fiables "
                     f"(trop étroits).**")
    else:
        text += ", en tirant les énoncés)."
    if len(names) > 1:
        text += (f" Écarts mesurés par rapport à « {names[0]} », sur les mêmes énoncés ; "
                 f"n.s. : l'IC de l'écart contient zéro.")
    if notes:
        text += " WER en % : plus bas est meilleur."
    lines.append(text)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reports", nargs="+", type=Path)
    p.add_argument("--noms", nargs="+", default=None,
                   help="systèmes à comparer, dans l'ordre (le premier sert de référence)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unite", choices=UNITS, default="auto",
                   help="unité tirée avec remise : énoncés ou locuteurs (auto : locuteurs "
                        "si toutes les sorties en portent un)")
    args = p.parse_args(argv)

    loaded = [load_series(path) for path in args.reports
              if not path.name.endswith(".details.jsonl")]
    if args.noms:
        by_name = {name: series for name, series in loaded}
        missing = [n for n in args.noms if n not in by_name]
        if missing:
            raise SystemExit(f"introuvables : {missing} ; disponibles : {sorted(by_name)}")
        loaded = [(n, by_name[n]) for n in args.noms]
    if len(loaded) < 1:
        raise SystemExit("aucun rapport")

    names = [name for name, _ in loaded]
    labels = [label for label in loaded[0][1] if all(label in s for _, s in loaded)]
    if not labels:
        raise SystemExit("aucune métrique commune à tous les rapports")
    by_label = {label: [series[label] for _, series in loaded] for label in labels}
    try:
        results = {label: bootstrap(systems, args.n_boot, args.seed, unit=args.unite)
                   for label, systems in by_label.items()}
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(to_markdown(names, results))


if __name__ == "__main__":
    main()
