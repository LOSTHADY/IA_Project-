"""Les nombres, par règle plutôt que par le traducteur.

NLLB traduit mal les nombres écrits en lettres : « compte jusqu'à vingt »
donnait « … kɔnɔntɔn, tan, tan, tan, tan ni fila … », sans 11 ni 19 (test au
micro du 15 octobre 2026). Or la numération bambara est régulière, et NLLB
recopie fidèlement les chiffres. D'où, autour de la traduction sortante :

1. `en_chiffres` : les nombres français en lettres deviennent des chiffres
   (« vingt et un » → « 21 ») ;
2. NLLB traduit et recopie les chiffres ;
3. `en_bambara` : les chiffres deviennent des nombres bambara (« 21 » →
   « mugan ni kelen »), que la synthèse vocale sait lire (elle saute les
   chiffres).

« un »/« une » restent des articles (« un homme ») sauf seuls devant une
ponctuation (« un, deux, trois ») ; « neuf » reste un adjectif (« un livre
neuf ») sauf dans une énumération ou devant un pluriel (« neuf ans »).
"""

from __future__ import annotations

import re

# --- bambara -----------------------------------------------------------------

UNITES = ("", "kelen", "fila", "saba", "naani", "duuru", "wɔɔrɔ", "wolonwula",
          "seegin", "kɔnɔntɔn")
ZERO = "fu"  # « fu » : rien, zéro


def _dizaines(n: int) -> str:
    dizaine, unite = divmod(n, 10)
    if dizaine == 0:
        base = ""
    elif dizaine == 1:
        base = "tan"
    elif dizaine == 2:
        base = "mugan"
    else:
        base = "bi" + UNITES[dizaine]  # bisaba (30) … bikɔnɔntɔn (90)
    if not base:
        return UNITES[unite]
    return f"{base} ni {UNITES[unite]}" if unite else base


def bambara(n: int) -> str:
    """Un entier positif en toutes lettres bambara : 21 → « mugan ni kelen »,
    125 → « kɛmɛ ni mugan ni duuru », 2026 → « waa fila ni mugan ni wɔɔrɔ »."""
    if n < 0:
        raise ValueError("nombre négatif")
    if n == 0:
        return ZERO
    parts = []
    for valeur, nom in ((1_000_000_000, "miliyari"), (1_000_000, "miliyɔn"), (1000, "waa")):
        q, n = divmod(n, valeur)
        if q:
            parts.append(f"{nom} {bambara(q)}")
    q, n = divmod(n, 100)
    if q:
        parts.append("kɛmɛ" if q == 1 else f"kɛmɛ {bambara(q)}")
    if n:
        parts.append(_dizaines(n))
    return " ni ".join(parts)


_CHIFFRES_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:[   ]\d{3})+(?![\d.,]\d)|"
                          r"(?<![\d.,])\d+(?![.,]\d)")


def en_bambara(text: str) -> str:
    """Remplace les nombres entiers écrits en chiffres par leur nom bambara.
    Les décimaux (« 1,5 ») restent tels quels."""
    def repl(m: re.Match) -> str:
        return bambara(int(re.sub(r"\D", "", m.group())))
    return _CHIFFRES_RE.sub(repl, text)


# --- français ----------------------------------------------------------------

_VALEURS = {
    "zéro": 0, "zero": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4,
    "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "onze": 11,
    "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15, "seize": 16,
    "vingt": 20, "vingts": 20, "trente": 30, "quarante": 40, "cinquante": 50,
    "soixante": 60, "cent": 100, "cents": 100, "mille": 1000,
    "million": 1_000_000, "millions": 1_000_000,
    "milliard": 1_000_000_000, "milliards": 1_000_000_000,
}
_MOT = "|".join(sorted(_VALEURS, key=len, reverse=True))
# Une suite de mots-nombres liés par des espaces ou des traits d'union, et
# par « et » seulement devant un/une/onze (« vingt et un », « soixante et
# onze ») : « un et deux » reste deux nombres.
_NOMBRE_RE = re.compile(
    rf"(?<![\w-])(?:{_MOT})(?:[\s-]+(?:et[\s-]+(?=(?:un|une|onze)(?![\w-])))?(?:{_MOT}))*"
    rf"(?![\w-])", re.IGNORECASE)
_PONCTUATION_APRES = re.compile(r"\s*(?:[,.;:!?)]|$)")


def _valeur(mots: list[str]) -> int:
    total = courant = 0
    precedent = None
    for mot in mots:
        v = _VALEURS[mot]
        if v == 20 and precedent == "quatre":
            courant += 4 * 20 - 4  # « quatre-vingt » : 80, et non 4 + 20
        elif v == 100:
            courant = (courant or 1) * 100
        elif v >= 1000:
            total += (courant or 1) * v
            courant = 0
        else:
            courant += v
        precedent = mot
    return total + courant


def _groupes(mots: list[str]) -> list[list[str]]:
    """« cinq six » est une énumération, pas 11 : deux unités qui se suivent
    sans multiplicateur ouvrent un nouveau nombre."""
    groupes: list[list[str]] = [[]]
    for mot in mots:
        prev = groupes[-1][-1] if groupes[-1] else None
        if prev and 1 <= _VALEURS[prev] <= 9 and 1 <= _VALEURS[mot] <= 16:
            groupes.append([])
        groupes[-1].append(mot)
    return groupes


def en_chiffres(text: str) -> str:
    """Écrit en chiffres les nombres français en lettres : « vingt et un » →
    « 21 », « quatre-vingt-dix-neuf » → « 99 », « un, deux, trois » →
    « 1, 2, 3 ». Garde « un »/« une » articles et « neuf » adjectif."""
    enumeration = len(_NOMBRE_RE.findall(text)) >= 3

    def repl(m: re.Match) -> str:
        mots = [w for w in re.split(r"[\s-]+", m.group().lower()) if w and w != "et"]
        if len(mots) == 1 and mots[0] in ("un", "une", "neuf"):
            apres = text[m.end():]
            seul = bool(_PONCTUATION_APRES.match(apres))
            if mots[0] == "neuf":
                suivant = re.match(r"\s+(\w+)", apres)
                pluriel = bool(suivant and re.search(r"[sx]$", suivant.group(1)))
                if not (pluriel or (seul and enumeration)):
                    return m.group()
            elif not seul:
                return m.group()
        return " ".join(str(_valeur(g)) for g in _groupes(mots))

    return _NOMBRE_RE.sub(repl, text)
