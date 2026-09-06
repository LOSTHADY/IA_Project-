"""Métriques d'évaluation.

Deux partis pris méthodologiques, à reprendre tels quels dans le mémoire :

1. **WER strict vs WER relâché.** Le relâché est calculé sur la forme repliée
   (ɛ→e, ɔ→o, ɲ→ny, ŋ→ng, sans ponctuation). L'écart entre les deux mesure la
   part d'erreur imputable à la seule variation orthographique — sans ça, un
   modèle qui écrit un bambara correct mais en ASCII paraît bien pire qu'il
   n'est.

2. **chrF++ plutôt que BLEU** comme métrique principale de traduction. BLEU
   est peu fiable sur langue morphologiquement riche et faible ressource ;
   chrF++ opère au niveau caractère et corrèle mieux avec le jugement humain.
   BLEU est tout de même rapporté, pour comparabilité avec la littérature.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from bambara_voice.normalize import fold, normalize


@dataclass
class ASRScores:
    wer: float
    cer: float
    wer_folded: float
    cer_folded: float
    n: int

    @property
    def orthographic_gap(self) -> float:
        """Points de WER attribuables à la variation orthographique seule."""
        return self.wer - self.wer_folded

    def to_dict(self) -> dict:
        d = asdict(self)
        d["orthographic_gap"] = round(self.orthographic_gap, 4)
        return d


@dataclass
class MTScores:
    chrf: float
    bleu: float
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


def score_asr(hypotheses: list[str], references: list[str]) -> ASRScores:
    """WER/CER strict et relâché sur un corpus aligné."""
    import jiwer

    if len(hypotheses) != len(references):
        raise ValueError(
            f"{len(hypotheses)} hypothèses pour {len(references)} références"
        )
    if not hypotheses:
        raise ValueError("corpus vide")

    strict_h = [normalize(h) for h in hypotheses]
    strict_r = [normalize(r) for r in references]
    fold_h = [fold(h) for h in hypotheses]
    fold_r = [fold(r) for r in references]

    # jiwer rejette les références vides : on écarte ces paires des deux côtés.
    def _drop_empty(hs: list[str], rs: list[str]) -> tuple[list[str], list[str]]:
        pairs = [(h, r) for h, r in zip(hs, rs) if r.strip()]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    strict_h, strict_r = _drop_empty(strict_h, strict_r)
    fold_h, fold_r = _drop_empty(fold_h, fold_r)
    if not strict_r:
        raise ValueError("toutes les références sont vides")

    return ASRScores(
        wer=jiwer.wer(strict_r, strict_h),
        cer=jiwer.cer(strict_r, strict_h),
        wer_folded=jiwer.wer(fold_r, fold_h),
        cer_folded=jiwer.cer(fold_r, fold_h),
        n=len(strict_r),
    )


def score_mt(hypotheses: list[str], references: list[str]) -> MTScores:
    """chrF++ (principal) et BLEU (comparabilité) au niveau corpus."""
    import sacrebleu

    if len(hypotheses) != len(references):
        raise ValueError(
            f"{len(hypotheses)} hypothèses pour {len(references)} références"
        )
    if not hypotheses:
        raise ValueError("corpus vide")

    refs = [references]  # sacrebleu attend une liste de jeux de références
    chrf = sacrebleu.corpus_chrf(hypotheses, refs, word_order=2)  # word_order=2 => chrF++
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    return MTScores(chrf=chrf.score, bleu=bleu.score, n=len(hypotheses))


def cascade_degradation(
    scores_from_gold_text: MTScores, scores_from_asr: MTScores
) -> dict[str, float]:
    """Propagation d'erreurs : ce que coûte l'ASR au reste de la chaîne.

    On évalue deux fois la même traduction — une fois depuis la transcription
    de référence, une fois depuis la sortie de l'ASR. La différence isole le
    coût de l'ASR, indépendamment de la qualité de la MT.
    """
    delta = scores_from_gold_text.chrf - scores_from_asr.chrf
    rel = delta / scores_from_gold_text.chrf if scores_from_gold_text.chrf else 0.0
    return {
        "chrf_texte_de_reference": round(scores_from_gold_text.chrf, 2),
        "chrf_depuis_asr": round(scores_from_asr.chrf, 2),
        "perte_absolue": round(delta, 2),
        "perte_relative": round(rel, 4),
    }
