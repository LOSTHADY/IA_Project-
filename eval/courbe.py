"""La courbe d'apprentissage : combien rapporte chaque heure de lecture ?

En reconnaissance vocale, le taux d'erreur baisse à peu près d'autant à
chaque doublement des données : il suit une droite en fonction du logarithme
des heures apprises. Trois points (le même Soloni affiné sur un quart, la
moitié et toutes les lectures du pilote) suffisent à tracer cette droite et
à prédire, grossièrement, ce que donneraient 100 ou 300 heures. Une
prédiction n'est pas une mesure : plus on s'éloigne des heures mesurées,
moins elle vaut ; elle sert à décider si la campagne vaut la peine d'être
poursuivie.
"""

from __future__ import annotations

import math


def ajuster(points: list[tuple[float, float]]) -> tuple[float, float]:
    """(a, b) de WER = a + b·ln(heures), aux moindres carrés ; b < 0 si plus
    de lectures font moins d'erreurs."""
    pts = [(math.log(h), w) for h, w in points if h > 0]
    if len(pts) < 2:
        raise ValueError("il faut au moins deux points (heures > 0)")
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        raise ValueError("il faut des quantités d'heures différentes")
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    return my - b * mx, b


def predire(a: float, b: float, heures: float) -> float:
    return max(0.0, a + b * math.log(heures))


def to_markdown(points: list[tuple[float, float]], cibles=(100, 300)) -> str:
    a, b = ajuster(points)
    lignes = ["| Heures apprises | WER (voix nouvelles) |", "|---|---|"]
    lignes += [f"| {h:.1f} | {w:.1f} % |" for h, w in sorted(points)]
    gain = -b * math.log(2)
    lignes += ["", f"Chaque doublement des heures : {gain:+.1f} points de WER en moins."
               if gain > 0 else "Plus d'heures n'ont pas fait baisser les erreurs : la courbe "
               "est plate, ou remonte."]
    if gain > 0:
        lignes += [f"Prédiction (grossière) à {h} h : WER ≈ {predire(a, b, h):.0f} %"
                   for h in cibles]
    return "\n".join(lignes)
