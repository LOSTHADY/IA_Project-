"""Le bambara écrit comme on l'entend en français, ramené à l'orthographe standard.

La plupart des locuteurs n'ont jamais appris à écrire le bambara : ils
l'écrivent avec les lettres du français (« tchè » pour « cɛ », « gnogon »
pour « ɲɔgɔn », « ou » pour « u », « é » ou « è » pour « e » et « ɛ »).
Un texte corrigé par eux dans l'atelier ne peut pas servir tel quel à
entraîner la reconnaissance, qui écrit, elle, en orthographe standard.

Chaque mot est ramené à une clé phonétique commune aux deux écritures
(digrammes du français remplacés, puis `normalize.fold`), et la clé désigne
le mot standard : d'abord parmi les mots que la reconnaissance a proposés
pour cette phrase (ce qui a été entendu), puis dans le vocabulaire du
bambara parlé (data/vocabulaire_bambara.tsv, le plus fréquent l'emporte),
enfin parmi les clés très proches (une lettre de différence). Un mot sans
équivalent (nom propre, mot rare) reste tel qu'il a été écrit, accents du
français retirés.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .normalize import fold, normalize

VOCABULAIRE = Path(__file__).resolve().parent.parent / "data" / "vocabulaire_bambara.tsv"

# Digrammes et lettres du français → sons du bambara, dans cet ordre.
_FRANCAIS = [
    (r"tch", "c"), (r"ch", "c"), (r"dj", "j"), (r"gn", "ny"), (r"ñ", "ny"),
    (r"qu", "k"), (r"ou(?=[aeiɛoɔ])", "w"), (r"ou", "u"), (r"ç", "s"),
]
_ACCENTS = str.maketrans({"é": "e", "è": "e", "ê": "e", "ë": "e", "à": "a", "â": "a",
                          "ô": "o", "ö": "o", "ù": "u", "û": "u", "ü": "u", "î": "i", "ï": "i"})


def sans_accents_francais(mot: str) -> str:
    return unicodedata.normalize("NFC", mot).lower().translate(_ACCENTS)


def cle(mot: str) -> str:
    """Forme commune à « tchè » et à « cɛ » (« ce »), à « gnogon » et à
    « ɲɔgɔn » (« nyogon »)."""
    m = sans_accents_francais(mot)
    for motif, son in _FRANCAIS:
        m = re.sub(motif, son, m)
    return fold(m).replace(" ", "")


def lire_vocabulaire(path: str | Path = VOCABULAIRE) -> dict[str, int]:
    out = {}
    for ligne in Path(path).read_text(encoding="utf-8").splitlines():
        if ligne and not ligne.startswith("#"):
            mot, _, n = ligne.partition("\t")
            out[mot] = int(n or 1)
    return out


@dataclass
class Correction:
    texte: str
    changements: list[tuple[str, str]] = field(default_factory=list)


class Correcteur:
    def __init__(self, vocabulaire: dict[str, int] | None = None):
        if vocabulaire is None:
            vocabulaire = lire_vocabulaire() if VOCABULAIRE.exists() else {}
        self._par_cle: dict[str, tuple[int, str]] = {}
        for mot, n in vocabulaire.items():
            k = cle(mot)
            if k and n > self._par_cle.get(k, (0, ""))[0]:
                self._par_cle[k] = (n, mot)
        self._par_initiale: dict[str, list[str]] = {}
        for k in self._par_cle:
            self._par_initiale.setdefault(k[0], []).append(k)

    def _standard(self, k: str, entendus: dict[str, str]) -> str | None:
        if k in entendus:
            return entendus[k]
        if k in self._par_cle:
            return self._par_cle[k][1]
        proches = difflib.get_close_matches(k, self._par_initiale.get(k[0], []), n=1,
                                            cutoff=0.8) if len(k) >= 4 else []
        return self._par_cle[proches[0]][1] if proches else None

    def corriger(self, texte: str, entendu: str | list[str] = "") -> Correction:
        """`texte` en orthographe standard ; `entendu` : ce que la
        reconnaissance a proposé pour cette phrase, prioritaire."""
        sources = [entendu] if isinstance(entendu, str) else entendu
        entendus = {cle(m): m for s in sources for m in normalize(s).split() if cle(m)}
        mots, changements = [], []
        for brut in normalize(texte).split():
            k = cle(brut)
            standard = (self._standard(k, entendus) if k else None) or sans_accents_francais(brut)
            if standard != brut:
                changements.append((brut, standard))
            mots.append(standard)
        return Correction(" ".join(mots), changements)
