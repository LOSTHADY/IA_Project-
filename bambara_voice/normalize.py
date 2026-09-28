"""Normalisation orthographique du bambara.

Le bambara s'écrit en alphabet latin étendu (ɛ, ɔ, ɲ, ŋ) mais une grande part
des textes disponibles — et à peu près tout ce qui est tapé sur un clavier
ordinaire — utilise des approximations ASCII (e, o, ny, n/ng). Les tons ne sont
quasiment jamais notés.

Deux usages distincts, à ne pas confondre :

1. `normalize()` — nettoyage d'entrée avant un modèle. Conservateur : on ne
   touche pas aux caractères spéciaux, on ne fait qu'unifier Unicode,
   apostrophes, espaces et ponctuation.
2. `fold()` — repli agressif vers un ASCII canonique, uniquement pour la
   *comparaison* en évaluation. Permet de calculer un WER « relâché » qui ne
   pénalise pas un modèle pour avoir écrit "ne" au lieu de "nɛ". La différence
   entre WER strict et WER relâché mesure directement la part de l'erreur
   imputable à la variation orthographique.
"""

from __future__ import annotations

import re
import unicodedata

# Caractères propres à l'orthographe officielle du bambara.
BAMBARA_SPECIALS = "ɛɔɲŋɛ́ɔ́"

# Repli vers ASCII, pour comparaison uniquement. Volontairement destructif.
_FOLD_MAP = {
    "ɛ": "e", "Ɛ": "e",
    "ɔ": "o", "Ɔ": "o",
    "ɲ": "ny", "Ɲ": "ny",
    "ŋ": "ng", "Ŋ": "ng",
}

# Variantes d'apostrophes rencontrées dans les corpus (n', k'a, b'a...).
_APOSTROPHES = "’ʼʾ`´"

# Ponctuation unifiée vers ASCII.
_PUNCT_MAP = {
    "–": "-", "—": "-", "−": "-",
    "“": '"', "”": '"', "„": '"',
    "‹": '"', "›": '"', "«": '"', "»": '"',
    "…": "...",
}

_WS_RE = re.compile(r"\s+")
_PUNCT_STRIP_RE = re.compile(r"[^\w\s'" + BAMBARA_SPECIALS + r"]", flags=re.UNICODE)


def _apply_map(text: str, mapping: dict[str, str]) -> str:
    for src, dst in mapping.items():
        text = text.replace(src, dst)
    return text


def normalize(text: str, *, lower: bool = True) -> str:
    """Nettoyage conservateur, à appliquer avant tout modèle.

    Ne modifie aucun caractère spécifique au bambara : un texte correctement
    orthographié le reste.
    """
    if not text:
        return ""
    # NFC : recompose les diacritiques pour que "ɛ" + accent == "ɛ́".
    text = unicodedata.normalize("NFC", text)
    text = _apply_map(text, _PUNCT_MAP)
    for ap in _APOSTROPHES:
        text = text.replace(ap, "'")
    text = _WS_RE.sub(" ", text).strip()
    if lower:
        text = text.lower()
    return text


def fold(text: str) -> str:
    """Forme canonique ASCII, pour la *comparaison* en évaluation.

    Replie ɛ→e, ɔ→o, ɲ→ny, ŋ→ng, retire ponctuation et apostrophes. Deux
    graphies concurrentes du même mot donnent la même sortie.
    """
    text = normalize(text, lower=True)
    text = _apply_map(text, _FOLD_MAP)
    # Retirer les diacritiques tonals résiduels.
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("'", "")
    text = _PUNCT_STRIP_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def strip_punctuation(text: str) -> str:
    """Retire la ponctuation mais garde lettres, diacritiques et apostrophes.

    Sert au WER strict : la ponctuation n'est pas prononcée et la littérature
    ne la compte pas. La retirer ici comme dans `fold()` garantit que l'écart
    entre WER strict et relâché ne mesure que l'orthographe.
    """
    return _WS_RE.sub(" ", _PUNCT_STRIP_RE.sub(" ", text)).strip()


def has_official_orthography(text: str) -> bool:
    """Vrai si le texte utilise au moins un caractère latin étendu bambara.

    Indicateur grossier mais utile pour trier un corpus : une ligne sans aucun
    de ces caractères est soit très courte, soit tapée en ASCII approximatif.
    """
    return any(ch in BAMBARA_SPECIALS for ch in text)


def orthography_ratio(lines: list[str]) -> float:
    """Part des lignes en orthographe officielle. Diagnostic de corpus."""
    if not lines:
        return 0.0
    return sum(has_official_orthography(l) for l in lines) / len(lines)
