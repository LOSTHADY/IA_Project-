"""Ce que l'assistant a appris de toi : retenu tout de suite, sans réentraînement.

Un réentraînement ne garantit jamais de gain : il déplace des millions de
poids, et ce qu'il gagne d'un côté peut se perdre ailleurs (docs/METHODE.md).
Une mémoire, si : ce qu'on y met sert dès la phrase suivante, et ne touche à
rien d'autre. C'est aussi ainsi que les grands assistants « se souviennent »
d'une conversation à l'autre : des notes relues, pas des poids modifiés.

Deux sortes de souvenirs :

- **des phrases** : ce que la reconnaissance a entendu → ce que tu voulais
  dire. Elles viennent de l'atelier (chaque phrase dite) et des corrections
  faites pendant la démo (« Ce n'est pas ce que j'ai dit »). Quand la
  reconnaissance entend une phrase assez proche d'un souvenir (similarité de
  Dice sur trigrammes, forme repliée, comme les gabarits), le français du
  souvenir remplace la traduction ;
- **des mots** : le lexique de l'atelier (argot, emprunts) avec leur
  équivalent en bambara courant, substitué avant la traduction pour que le
  traducteur reçoive un mot qu'il connaît (« wai » → « i ni ce »).

Une correction plus récente remplace un souvenir de la même phrase. Le seuil
se choisit sur le dev du jeu maison (eval.redire) : trop bas, une phrase
inconnue prendrait le sens d'une autre.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .normalize import fold
from .templates import dice


@dataclass
class Souvenir:
    bm: str           # ce que la reconnaissance a entendu
    fr: str           # ce que la phrase voulait dire
    source: str = ""  # « atelier », « correction »…
    date: str = ""


def _lignes(path: str | Path | None) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    return [json.loads(ligne) for ligne in Path(path).read_text(encoding="utf-8").splitlines()
            if ligne.strip()]


class Memoire:
    """Souvenirs lus dans `atelier` (lecture seule) puis `path` (les
    corrections, où `apprendre` écrit) ; mots lus dans `lexique`
    (`lexique.jsonl` de l'atelier)."""

    def __init__(self, path: str | Path | None = None, seuil: float = 0.85,
                 atelier: str | Path | None = None, lexique: str | Path | None = None):
        self.path, self.seuil = (Path(path) if path else None), seuil
        self._par_cle: dict[str, Souvenir] = {}
        for x in _lignes(atelier) + _lignes(self.path):
            self._retenir(Souvenir(**{k: x.get(k, "") for k in ("bm", "fr", "source", "date")}))
        self.mots: list[tuple[list[str], str]] = sorted(
            ((fold(m["forme"]).split(), m["equivalent_bm"].strip()) for m in _lignes(lexique)
             if (m.get("equivalent_bm") or "").strip() and fold(m.get("forme") or "")),
            key=lambda m: -len(m[0]))  # les expressions avant les mots qu'elles contiennent

    def _retenir(self, s: Souvenir) -> None:
        if fold(s.bm) and s.fr.strip():
            self._par_cle[fold(s.bm)] = s

    def __len__(self) -> int:
        return len(self._par_cle)

    def chercher(self, bm: str) -> tuple[Souvenir, float] | None:
        """Le souvenir le plus proche de ce qui a été entendu, s'il dépasse le
        seuil."""
        cle = fold(bm)
        if not cle or not self._par_cle:
            return None
        if cle in self._par_cle:
            return self._par_cle[cle], 1.0
        score, souvenir = max(((dice(cle, k), s) for k, s in self._par_cle.items()),
                              key=lambda x: x[0])
        return (souvenir, score) if score >= self.seuil else None

    def apprendre(self, bm: str, fr: str, source: str = "correction") -> Souvenir:
        """Retient que `bm` voulait dire `fr`, tout de suite et pour la suite
        (écrit dans `path` s'il y en a un)."""
        if not fold(bm) or not fr.strip():
            raise ValueError("il faut ce qui a été entendu et ce que cela voulait dire")
        s = Souvenir(bm.strip(), fr.strip(), source, datetime.now().isoformat(timespec="seconds"))
        self._retenir(s)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
        return s

    def remplacer_mots(self, bm: str) -> str:
        """Les mots du lexique remplacés par leur équivalent courant."""
        if not self.mots:
            return bm
        mots, cles, out, i = bm.split(), [fold(m) for m in bm.split()], [], 0
        while i < len(mots):
            for forme, equivalent in self.mots:
                if cles[i:i + len(forme)] == forme:
                    out.append(equivalent)
                    i += len(forme)
                    break
            else:
                out.append(mots[i])
                i += 1
        return " ".join(out)
