"""Réponses gabarits validées par un locuteur natif.

Levier 3 : sur les intentions les plus fréquentes, on court-circuite la
traduction sortante et on renvoie du bambara *humain*. La MT ne sert plus que
de repli. Le mémoire doit rapporter le taux de couverture des gabarits en
regard du taux de repli — c'est cette proportion, pas la démo, qui est le
résultat.

Appariement par similarité de Dice sur trigrammes de caractères, calculée sur
la forme repliée (cf. normalize.fold) : robuste aux erreurs d'ASR et aux
variantes orthographiques, et surtout entièrement explicable — pas de boîte
noire supplémentaire dans la chaîne.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .config import TemplateConfig
from .normalize import fold

logger = logging.getLogger(__name__)


def _trigrams(text: str) -> set[str]:
    padded = f"  {text} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def dice(a: str, b: str) -> float:
    """Similarité de Dice sur trigrammes de caractères, dans [0, 1]."""
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))


@dataclass
class Template:
    """Une intention : des formulations bambara -> une réponse bambara validée."""

    id: str
    triggers_bm: list[str]
    response_bm: str
    response_fr: str = ""  # traduction de référence, pour l'évaluation
    note: str = ""

    @property
    def folded_triggers(self) -> list[str]:
        return [fold(t) for t in self.triggers_bm]


@dataclass
class TemplateMatch:
    template: Template
    score: float
    matched_trigger: str


class TemplateBank:
    """Banque de gabarits chargée depuis un JSON."""

    def __init__(self, templates: list[Template], threshold: float = 0.62):
        self.templates = templates
        self.threshold = threshold

    @classmethod
    def from_config(cls, config: TemplateConfig) -> "TemplateBank":
        if not config.enabled:
            return cls([], config.threshold)
        return cls.from_file(config.path, config.threshold)

    @classmethod
    def from_file(cls, path: str | Path, threshold: float = 0.62) -> "TemplateBank":
        path = Path(path)
        if not path.exists():
            logger.info("Aucun fichier de gabarits à %s — banque vide", path)
            return cls([], threshold)
        raw = json.loads(path.read_text(encoding="utf-8"))
        templates = [Template(**entry) for entry in raw.get("templates", [])]
        logger.info("%d gabarits chargés depuis %s", len(templates), path)
        return cls(templates, threshold)

    def match(self, utterance_bm: str) -> TemplateMatch | None:
        """Meilleur gabarit au-dessus du seuil, sinon None (repli sur la MT)."""
        if not self.templates or not utterance_bm.strip():
            return None
        folded = fold(utterance_bm)
        best: TemplateMatch | None = None
        for tpl in self.templates:
            for trigger, folded_trigger in zip(tpl.triggers_bm, tpl.folded_triggers):
                score = dice(folded, folded_trigger)
                if best is None or score > best.score:
                    best = TemplateMatch(tpl, score, trigger)
        if best is not None and best.score >= self.threshold:
            return best
        return None

    def __len__(self) -> int:
        return len(self.templates)
