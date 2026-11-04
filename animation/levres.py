"""Synchronisation labiale : quelle forme de bouche à chaque image.

Deux moyens :

- **Rhubarb Lip Sync**, s'il est installé (programme ``rhubarb`` dans le PATH,
  ou variable d'environnement ``RHUBARB``). Son mode phonétique ne dépend
  d'aucune langue : il convient au bambara.
- **Sinon, l'audio seul** : l'énergie dit combien la bouche s'ouvre, et le
  timbre (centroïde spectral) si elle s'arrondit (o, u). C'est plus fruste,
  mais on n'installe rien, le calcul est instantané, et à 30 images par
  seconde l'œil ne demande pas plus.

Les formes sont celles de Rhubarb (``personnage.VISEMES``) : X repos,
A lèvres closes (m, b, p), B entrouverte, C ouverte, D grande ouverte,
E arrondie, F avancée (u, w), G f/v, H l.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np


def energie(signal: np.ndarray, sr: int, fps: int) -> np.ndarray:
    """Énergie par image, ramenée entre 0 et 1 (1 : les passages forts)."""
    pas = sr / fps
    n = max(1, int(math.ceil(len(signal) / pas)))
    rms = np.empty(n)
    for i in range(n):
        a = int(i * pas - pas / 2)
        fenetre = signal[max(0, a): int(a + 2 * pas)]
        rms[i] = np.sqrt(np.mean(fenetre ** 2)) if len(fenetre) else 0.0
    ref = np.percentile(rms, 95) if np.any(rms > 0) else 1.0
    return np.clip(rms / (ref + 1e-9), 0.0, 1.0)


def _centroide(signal: np.ndarray, sr: int, fps: int, n: int) -> np.ndarray:
    pas = sr / fps
    taille = 1 << int(np.ceil(np.log2(2 * pas)))
    frequences = np.fft.rfftfreq(taille, 1 / sr)
    fenetre = np.hanning(taille)
    c = np.zeros(n)
    for i in range(n):
        a = int(i * pas - taille / 2 + pas / 2)
        bout = signal[max(0, a): a + taille]
        if len(bout) < taille:
            bout = np.pad(bout, (0, taille - len(bout)))
        spectre = np.abs(np.fft.rfft(bout * fenetre))
        c[i] = (spectre * frequences).sum() / (spectre.sum() + 1e-9)
    return c


def depuis_audio(signal: np.ndarray, sr: int, fps: int) -> list[str]:
    """Une forme de bouche par image, d'après l'audio seul."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    e = energie(signal, sr, fps)
    # lissage sur trois images : la bouche ne claque pas à chaque crête
    e = np.convolve(np.pad(e, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    c = _centroide(signal, sr, fps, len(e))
    voise = e > 0.25
    mediane = np.median(c[voise]) if voise.any() else 0.0
    formes = []
    for i, niveau in enumerate(e):
        if niveau < 0.08:
            f = "X"
        elif niveau < 0.25:
            f = "B"
        elif mediane and c[i] < 0.7 * mediane:  # timbre sombre : voyelle arrondie
            f = "E" if niveau >= 0.55 else "F"
        elif niveau < 0.55:
            f = "C"
        else:
            f = "D"
        formes.append(f)
    # brève fermeture entre deux syllabes : lèvres closes plutôt que repos
    for i in range(1, len(formes) - 1):
        if formes[i] == "X" and formes[i - 1] not in "XA" and formes[i + 1] not in "XA":
            formes[i] = "A"
    return _tenir(formes, 2)


def _tenir(formes: list[str], minimum: int) -> list[str]:
    """Chaque forme tient au moins ``minimum`` images : sinon, ça scintille."""
    sortie, actuelle, depuis = [], None, 0
    for f in formes:
        if f != actuelle and (actuelle is None or depuis >= minimum):
            actuelle, depuis = f, 0
        sortie.append(actuelle)
        depuis += 1
    return sortie


def rhubarb() -> str | None:
    return os.environ.get("RHUBARB") or shutil.which("rhubarb")


def depuis_rhubarb(wav: str, fps: int, duree: float, texte: str | None = None) -> list[str]:
    """Une forme par image, d'après Rhubarb (mode phonétique)."""
    commande = [rhubarb(), "-r", "phonetic", "-f", "json", "--extendedShapes", "GHX", wav]
    with tempfile.TemporaryDirectory() as tmp:
        if texte:
            dialogue = os.path.join(tmp, "dialogue.txt")
            with open(dialogue, "w", encoding="utf-8") as f:
                f.write(texte)
            commande[-1:-1] = ["-d", dialogue]
        sortie = subprocess.run(commande, check=True, capture_output=True, text=True).stdout
    indices = json.loads(sortie)["mouthCues"]
    n = max(1, int(math.ceil(duree * fps)))
    formes = ["X"] * n
    for indice in indices:
        for i in range(int(indice["start"] * fps), min(n, int(math.ceil(indice["end"] * fps)))):
            formes[i] = indice["value"]
    return formes


def formes(signal: np.ndarray, sr: int, fps: int, wav: str | None = None,
           texte: str | None = None, moteur: str = "auto") -> list[str]:
    """Formes de bouche : Rhubarb s'il est là (``moteur="auto"``), sinon l'audio."""
    if moteur == "rhubarb" or (moteur == "auto" and wav and rhubarb()):
        return depuis_rhubarb(wav, fps, len(signal) / sr, texte)
    return depuis_audio(signal, sr, fps)

