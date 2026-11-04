"""Un épisode : des personnages, des répliques enregistrées, une vidéo.

**L'image suit la voix.** Personne n'a à caler sa voix sur l'animation :
chaque réplique dure le temps de son enregistrement. Les répliques
s'enchaînent et le jeu des personnages se calcule seul. Celui qui parle
remue la bouche et hoche la tête sur ses syllabes fortes. Celui qui écoute
se tourne vers lui. Tous clignent des yeux et respirent. Une même scène
donne ainsi autant de vidéos qu'il y a de participants, sans retouche.

Description d'un épisode (``episode.json``, chemins relatifs à ce fichier) ::

    {
      "format": [1080, 1920], "fps": 30,
      "fond": {"haut": "#fde2b3", "bas": "#e9a66b"},
      "personnages": {
        "awa":    {"dossier": "../../personnages/awa", "x": 330, "y": 1750},
        "seydou": {"dossier": "../../personnages/seydou", "x": 760, "y": 1750}
      },
      "repliques": [
        {"qui": "awa", "texte": "I ni sɔgɔma !", "audio": "awa_1.wav"},
        {"qui": "seydou", "texte": "Nba, i ni sɔgɔma.", "audio": "seydou_1.wav",
         "mouvements": [{"t": 0.0, "lacet": -0.6}, {"t": 1.2, "lacet": 0.0}]}
      ]
    }

``mouvements`` (facultatif) impose des positions de tête au cours d'une
réplique (``t`` en secondes depuis son début ; ``lacet``, ``tangage``,
``roulis``). Le reste du temps, le jeu est automatique.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

from . import levres
from .personnage import Personnage, Pose, charger

SR = 44100
POLICES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


@dataclass
class Replique:
    qui: str
    texte: str
    audio: np.ndarray  # mono, à SR
    debut: float = 0.0
    formes: list[str] = field(default_factory=list)
    energie: np.ndarray | None = None
    mouvements: list[dict] = field(default_factory=list)

    @property
    def duree(self) -> float:
        return len(self.audio) / SR


@dataclass
class Acteur:
    """Un personnage sur scène, et l'état lissé de sa tête."""
    personnage: Personnage
    x: float
    y: float
    echelle: float = 1.0
    graine: int = 0
    lacet: float = 0.0
    tangage: float = 0.0
    roulis: float = 0.0
    clignements: set[int] = field(default_factory=set)
    phases: list[float] = field(default_factory=list)

    def derive(self, t: float, k: int) -> float:
        """Petit mouvement lent et jamais périodique (somme de sinus)."""
        a, b, c = self.phases[3 * k: 3 * k + 3]
        return (math.sin(0.37 * t + a) + 0.6 * math.sin(0.83 * t + b)
                + 0.3 * math.sin(1.9 * t + c)) / 1.9


def charger_audio(chemin: Path) -> np.ndarray:
    signal, sr = sf.read(str(chemin), dtype="float32", always_2d=True)
    signal = signal.mean(axis=1)
    if sr != SR:  # rééchantillonnage linéaire : suffit pour la piste du montage
        t = np.arange(int(len(signal) * SR / sr)) * sr / SR
        signal = np.interp(t, np.arange(len(signal)), signal).astype(np.float32)
    return signal


def _couleur(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _police(taille: int) -> ImageFont.ImageFont:
    for chemin in POLICES:
        if Path(chemin).exists():
            return ImageFont.truetype(chemin, taille)
    return ImageFont.load_default(taille)


def _degrade(taille: tuple[int, int], haut: str, bas: str) -> Image.Image:
    w, h = taille
    a, b = np.array(_couleur(haut), float), np.array(_couleur(bas), float)
    k = np.linspace(0, 1, h)[:, None]
    lignes = (a * (1 - k) + b * k).astype(np.uint8)
    return Image.fromarray(np.repeat(lignes[:, None, :], w, axis=1), "RGB").convert("RGBA")


def _interpoler(cles: list[dict], t: float, champ: str) -> float | None:
    points = [(c["t"], c[champ]) for c in cles if champ in c]
    if not points:
        return None
    points.sort()
    if t <= points[0][0]:
        return points[0][1]
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            u = u * u * (3 - 2 * u)  # départ et arrivée en douceur
            return v0 + (v1 - v0) * u
    return points[-1][1]


class Episode:
    def __init__(self, fichier: str | Path, moteur_levres: str = "auto"):
        fichier = Path(fichier)
        racine = fichier.parent
        d = json.loads(fichier.read_text(encoding="utf-8"))
        self.taille = tuple(d.get("format", (1080, 1920)))
        self.fps = int(d.get("fps", 30))
        fond = d.get("fond", {})
        if "image" in fond:
            self.fond = Image.open(racine / fond["image"]).convert("RGBA").resize(self.taille)
        else:
            self.fond = _degrade(self.taille, fond.get("haut", "#fde2b3"), fond.get("bas", "#e9a66b"))
        self.sous_titres = d.get("sous_titres", True)
        self.acteurs: dict[str, Acteur] = {}
        for i, (nom, a) in enumerate(d["personnages"].items()):
            r = random.Random(f"{nom}-{i}")
            self.acteurs[nom] = Acteur(
                personnage=charger(racine / a["dossier"]), x=a["x"], y=a["y"],
                echelle=a.get("echelle", 1.0), graine=i,
                phases=[r.uniform(0, 2 * math.pi) for _ in range(9)])
        pause, t = d.get("pause", 0.35), d.get("intro", 0.6)
        self.repliques = []
        for r in d["repliques"]:
            if r["qui"] not in self.acteurs:
                raise KeyError(f"réplique de « {r['qui']} », absent des personnages")
            wav = racine / r["audio"]
            rep = Replique(qui=r["qui"], texte=r.get("texte", ""), audio=charger_audio(wav),
                           debut=t, mouvements=r.get("mouvements", []))
            rep.formes = levres.formes(rep.audio, SR, self.fps, str(wav), rep.texte or None,
                                       moteur_levres)
            rep.energie = levres.energie(rep.audio, SR, self.fps)
            self.repliques.append(rep)
            t += rep.duree + pause
        self.duree = t - pause + d.get("fin", 0.8)
        self.n_images = int(math.ceil(self.duree * self.fps))
        for acteur in self.acteurs.values():  # un clignement toutes les 2 à 5 s
            r, i = random.Random(f"cligne-{acteur.graine}"), 0
            while i < self.n_images:
                i += int(r.uniform(2.0, 5.0) * self.fps)
                acteur.clignements.update(range(i, i + 4))

    # --- jeu des personnages ---

    def replique_a(self, t: float) -> Replique | None:
        for r in self.repliques:
            if r.debut <= t < r.debut + r.duree:
                return r
        return None

    def _pose(self, nom: str, acteur: Acteur, i: int, t: float) -> Pose:
        parle = self.replique_a(t)
        cible_lacet = 0.25 * acteur.derive(t, 0)
        cible_tangage = 0.15 * acteur.derive(t, 1)
        cible_roulis = 2.5 * acteur.derive(t, 2)
        bouche = "X"
        if parle is not None:
            locuteur = self.acteurs[parle.qui]
            j = int((t - parle.debut) * self.fps)
            if parle.qui == nom:
                bouche = parle.formes[min(j, len(parle.formes) - 1)]
                autres = [a.x for n, a in self.acteurs.items() if n != nom]
                if autres:  # parle vers les autres, sans les fixer
                    cible_lacet += 0.3 * np.sign(np.mean(autres) - acteur.x)
                e = parle.energie[min(j, len(parle.energie) - 1)]
                cible_tangage += 0.35 * e - 0.1  # hoche la tête sur les syllabes fortes
                cible_roulis += 3.0 * e * math.sin(0.9 * t + acteur.graine)
                imposes = {c: _interpoler(parle.mouvements, t - parle.debut, c)
                           for c in ("lacet", "tangage", "roulis")}
                cible_lacet = imposes["lacet"] if imposes["lacet"] is not None else cible_lacet
                cible_tangage = imposes["tangage"] if imposes["tangage"] is not None else cible_tangage
                cible_roulis = imposes["roulis"] if imposes["roulis"] is not None else cible_roulis
            else:  # écoute : se tourne vers celui qui parle
                cible_lacet = 0.55 * np.sign(locuteur.x - acteur.x) + 0.1 * acteur.derive(t, 0)
        # amortissement : la tête rejoint sa cible sans à-coup
        k = 1 - math.exp(-7.0 / self.fps)
        acteur.lacet += (cible_lacet - acteur.lacet) * k
        acteur.tangage += (cible_tangage - acteur.tangage) * k * 1.6
        acteur.roulis += (cible_roulis - acteur.roulis) * k
        lacet = max(-1.0, min(1.0, acteur.lacet))
        return Pose(lacet=lacet, tangage=max(-1.0, min(1.0, acteur.tangage)),
                    roulis=acteur.roulis, regard=(0.8 * lacet, 0.3 * acteur.tangage),
                    bouche=bouche, cligne=i in acteur.clignements,
                    respiration=(t / 3.6 + acteur.graine * 0.3) % 1.0)

    # --- rendu ---

    def _sous_titre(self, image: Image.Image, r: Replique) -> None:
        w, _ = self.taille
        police = _police(int(w * 0.058))
        dessin = ImageDraw.Draw(image)
        mots, lignes, ligne = r.texte.split(), [], ""
        for mot in mots:
            essai = f"{ligne} {mot}".strip()
            if dessin.textlength(essai, font=police) > w * 0.8 and ligne:
                lignes.append(ligne)
                ligne = mot
            else:
                ligne = essai
        if ligne:
            lignes.append(ligne)
        y = int(self.taille[1] * 0.16)
        for ligne in lignes:
            dessin.text((w / 2, y), ligne, font=police, anchor="ma", fill="white",
                        stroke_width=max(3, w // 200), stroke_fill=(40, 25, 20))
            y += int(police.size * 1.25)

    def image(self, i: int, poses: dict[str, Pose] | None = None) -> Image.Image:
        """L'image numéro ``i``. Les images doivent être demandées dans l'ordre :
        la tête de chaque personnage est lissée d'une image à la suivante."""
        t = i / self.fps
        image = self.fond.copy()
        for nom, acteur in self.acteurs.items():
            pose = poses[nom] if poses else self._pose(nom, acteur, i, t)
            rendu = acteur.personnage.rendre(pose)
            if acteur.echelle != 1.0:
                rendu = rendu.resize((round(rendu.width * acteur.echelle),
                                      round(rendu.height * acteur.echelle)), Image.LANCZOS)
            image.alpha_composite(rendu, (round(acteur.x - rendu.width / 2),
                                          round(acteur.y - rendu.height)))
        r = self.replique_a(t)
        if self.sous_titres and r is not None and r.texte:
            self._sous_titre(image, r)
        return image.convert("RGB")

    def piste(self) -> np.ndarray:
        piste = np.zeros(int(self.duree * SR) + 1, dtype=np.float32)
        for r in self.repliques:
            a = int(r.debut * SR)
            piste[a: a + len(r.audio)] += r.audio[: len(piste) - a]
        return np.clip(piste, -1, 1)

    def rendre(self, sortie: str | Path, echelle: float = 1.0) -> Path:
        """Vidéo MP4 (H.264 + AAC), prête pour TikTok en 1080 × 1920."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg introuvable : l'installer (apt install ffmpeg)")
        sortie = Path(sortie)
        sortie.parent.mkdir(parents=True, exist_ok=True)
        w, h = (round(self.taille[0] * echelle) // 2 * 2, round(self.taille[1] * echelle) // 2 * 2)
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "piste.wav"
            sf.write(str(wav), self.piste(), SR)
            ffmpeg = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(self.fps),
                 "-i", "-", "-i", str(wav),
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart",
                 str(sortie)], stdin=subprocess.PIPE)
            try:
                for i in range(self.n_images):
                    image = self.image(i)
                    if (w, h) != image.size:
                        image = image.resize((w, h), Image.BILINEAR)
                    ffmpeg.stdin.write(image.tobytes())
            finally:
                ffmpeg.stdin.close()
                if ffmpeg.wait() != 0:
                    raise RuntimeError("ffmpeg a échoué")
        return sortie
