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
import multiprocessing
import os
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
from . import toon as _toon

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
    personnage: "Personnage | _toon.Toon"
    x: float
    y: float
    echelle: float = 1.0
    graine: int = 0
    lacet: float = 0.0
    tangage: float = 0.0
    roulis: float = 0.0
    sourcils: float = 0.0
    balance: float = 0.0
    corps: float = 0.0  # personnages en volume : orientation du corps (degrés)
    orientation: float = 0.0  # orientation de repos, vers les autres
    bras_g: float = 0.0
    bras_d: float = 0.0
    clignements: set[int] = field(default_factory=set)
    phases: list[float] = field(default_factory=list)

    @property
    def volume(self) -> bool:
        return isinstance(self.personnage, _toon.Toon)

    @property
    def tete_y(self) -> float:
        """Hauteur du centre de la tête, dans la scène."""
        if self.volume:
            return self.y + self.personnage.m.tete_c[1] * self.echelle
        p = self.personnage
        return self.y - (p.taille[1] - p.pivot_cou[1] * 0.75) * self.echelle

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
        from PIL import ImageFilter
        self._fond_flou = self.fond.filter(ImageFilter.GaussianBlur(5))
        ombre = Image.new("L", (420, 90), 0)
        ImageDraw.Draw(ombre).ellipse([20, 10, 400, 80], fill=110)
        ombre = ombre.filter(ImageFilter.GaussianBlur(12))
        self._ombre = Image.merge("RGBA", (Image.new("L", ombre.size, 40), Image.new("L", ombre.size, 20),
                                           Image.new("L", ombre.size, 10), ombre))
        self.sous_titres = d.get("sous_titres", True)
        self.cadrage = d.get("cadrage", "auto")  # "auto" : plans larges et rapprochés ; "fixe"
        self.zoom_serre = d.get("zoom_serre", 1.7)
        self.fichier, self.moteur_levres = fichier, moteur_levres
        self.acteurs: dict[str, Acteur] = {}
        for i, (nom, a) in enumerate(d["personnages"].items()):
            r = random.Random(f"{nom}-{i}")
            if "toon" in a:  # personnage en volume (toon.py) : tour complet possible
                personnage = _toon.Toon(_toon.modele(a["toon"]))
            else:
                personnage = charger(racine / a["dossier"])
            self.acteurs[nom] = Acteur(
                personnage=personnage, x=a["x"], y=a["y"],
                echelle=a.get("echelle", 1.0), graine=i,
                phases=[r.uniform(0, 2 * math.pi) for _ in range(9)])
        for nom, acteur in self.acteurs.items():  # chacun se tourne un peu vers les autres
            autres = [b.x for n, b in self.acteurs.items() if n != nom]
            defaut = 28.0 * float(np.sign(np.mean(autres) - acteur.x)) if autres else 0.0
            acteur.orientation = acteur.corps = d["personnages"][nom].get("orientation", defaut)
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

    def _replique_active(self, t: float) -> Replique | None:
        """La réplique en cours, pause qui la suit comprise."""
        actives = [r for r in self.repliques if r.debut <= t]
        return actives[-1] if actives else None

    @staticmethod
    def _cles(r: Replique, nom: str) -> list[dict]:
        """Mouvements imposés à ``nom`` pendant la réplique ``r`` : ceux sans
        « qui » concernent celui qui parle."""
        return [c for c in r.mouvements if c.get("qui", r.qui) == nom]

    def etat(self, i: int) -> dict:
        """Calcule l'image ``i`` du jeu de tous les acteurs (dans l'ordre des images)."""
        t = i / self.fps
        etats = {}
        for nom, a in self.acteurs.items():
            pose = self._pose(nom, a, i, t)
            etats[nom] = {"pose": pose, "balance": a.balance, "corps": a.corps,
                          "bras_g": a.bras_g, "bras_d": a.bras_d, "x": a.x}
        return etats

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
                imposes = {c: _interpoler(self._cles(parle, nom), t - parle.debut, c)
                           for c in ("lacet", "tangage", "roulis")}
                cible_lacet = imposes["lacet"] if imposes["lacet"] is not None else cible_lacet
                cible_tangage = imposes["tangage"] if imposes["tangage"] is not None else cible_tangage
                cible_roulis = imposes["roulis"] if imposes["roulis"] is not None else cible_roulis
            else:  # écoute : se tourne vers celui qui parle
                cible_lacet = 0.55 * np.sign(locuteur.x - acteur.x) + 0.1 * acteur.derive(t, 0)
        # sourcils : levés sur les syllabes les plus fortes, et à la fin d'une question
        cible_sourcils = 0.0
        cible_balance = 0.8 * acteur.derive(t, 1)
        if parle is not None and parle.qui == nom:
            j = min(int((t - parle.debut) * self.fps), len(parle.energie) - 1)
            cible_sourcils = max(0.0, min(1.0, (parle.energie[j] - 0.75) * 4))
            if parle.texte.rstrip().endswith("?") and t - parle.debut > 0.6 * parle.duree:
                cible_sourcils = max(cible_sourcils, 0.9)
            cible_balance += 1.6 * parle.energie[j] * math.sin(1.3 * t + acteur.graine)
        elif parle is not None and t - parle.debut < 0.8:  # l'auditeur réagit à la réplique
            cible_sourcils = 0.45
        # gestes : celui qui parle lève un avant-bras sur les passages forts
        cible_g = cible_d = 0.0
        if parle is not None and parle.qui == nom:
            j = min(int((t - parle.debut) * self.fps), len(parle.energie) - 1)
            fen = parle.energie[max(0, j - 8): j + 1].mean()
            geste = max(0.0, min(0.85, fen * 1.5 - 0.35))
            if acteur.derive(t * 0.5, 2) > 0:
                cible_d = geste
            else:
                cible_g = geste
        # corps : orientation de repos, ou mouvements imposés (tour, déplacement)
        cible_corps = acteur.orientation + 360 * round((acteur.corps - acteur.orientation) / 360)
        actif = self._replique_active(t)
        if actif is not None:
            cles = self._cles(actif, nom)
            for champ in ("corps", "x", "bras_g", "bras_d"):
                v = _interpoler(cles, t - actif.debut, champ)
                if v is None:
                    continue
                if champ == "corps":
                    cible_corps = v
                elif champ == "x":
                    acteur.x = v
                elif champ == "bras_g":
                    cible_g = v
                else:
                    cible_d = v
        # amortissement : la tête rejoint sa cible sans à-coup
        k = 1 - math.exp(-7.0 / self.fps)
        acteur.corps += (cible_corps - acteur.corps) * k * 1.3
        acteur.bras_g += (cible_g - acteur.bras_g) * k * 1.1
        acteur.bras_d += (cible_d - acteur.bras_d) * k * 1.1
        acteur.sourcils += (cible_sourcils - acteur.sourcils) * k * 1.8
        acteur.balance += (cible_balance - acteur.balance) * k * 0.6
        acteur.lacet += (cible_lacet - acteur.lacet) * k
        acteur.tangage += (cible_tangage - acteur.tangage) * k * 1.6
        acteur.roulis += (cible_roulis - acteur.roulis) * k
        lacet = max(-1.0, min(1.0, acteur.lacet))
        return Pose(lacet=lacet, tangage=max(-1.0, min(1.0, acteur.tangage)),
                    roulis=acteur.roulis, regard=(0.8 * lacet, 0.3 * acteur.tangage),
                    bouche=bouche, cligne=i in acteur.clignements, sourcils=acteur.sourcils,
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

    # --- caméra ---

    def camera(self, t: float) -> tuple[float, float, float]:
        """(x, y, zoom) : le point de la scène au centre de l'écran, et le grossissement.

        Une réplique sur trois en plan large, les autres en plan rapproché sur
        celui qui parle, avec un léger travelling avant pendant le plan : c'est
        le montage des vidéos courtes. On coupe d'un plan à l'autre au début
        de chaque réplique."""
        w, h = self.taille
        large = (w / 2, h / 2, 1.0)
        if self.cadrage == "fixe":
            return large
        precedentes = [r for r in self.repliques if r.debut <= t]
        if not precedentes:
            return large
        k = len(precedentes) - 1
        r = precedentes[-1]
        avance = 1 + 0.035 * min(t - r.debut, r.duree + 0.5)
        if k % 3 == 2 or len(self.acteurs) == 1 and k % 2:
            return (w / 2, h / 2, 1.0 * avance)
        acteur = self.acteurs[r.qui]
        tete_y = acteur.tete_y
        zoom = self.zoom_serre * avance
        # le visage un peu au-dessous du milieu : les sous-titres sont en haut
        return (acteur.x, tete_y - 0.06 * h / zoom, zoom)

    def image(self, i: int, etats: dict | None = None) -> Image.Image:
        """L'image numéro ``i``. Sans ``etats``, les images doivent être demandées
        dans l'ordre : le jeu de chaque personnage est lissé d'une image à l'autre."""
        if etats is None:
            etats = self.etat(i)
        t = i / self.fps
        w, h = self.taille
        cx, cy, zoom = self.camera(t)
        # le fond, plus loin que les personnages : il bouge et grossit moins qu'eux
        # (parallaxe), et il est flou en plan rapproché (profondeur de champ)
        zf = 1 + (zoom - 1) * 0.45
        fx, fy = w / 2 + (cx - w / 2) * 0.45, h / 2 + (cy - h / 2) * 0.45
        x0, y0 = fx - w / 2 / zf, fy - h / 2 / zf
        fond = self._fond_flou if zoom > 1.25 else self.fond
        if abs(zf - 1) < 1e-3 and abs(x0) < 0.5 and abs(y0) < 0.5:
            image = fond.copy()
        else:
            image = fond.resize((w, h), Image.BILINEAR,
                                box=(max(0, x0), max(0, y0), min(w, x0 + w / zf), min(h, y0 + h / zf)))
        for nom, acteur in sorted(self.acteurs.items(), key=lambda kv: kv[1].y):
            e = etats[nom]
            pose = e["pose"]
            s = acteur.echelle * zoom
            ex, ey = (e["x"] - cx) * zoom + w / 2, (acteur.y - cy) * zoom + h / 2
            if acteur.volume:
                pt = _toon.PoseToon(
                    corps=e["corps"], tete=pose.lacet * 32, tangage=pose.tangage * 12,
                    roulis=pose.roulis, bouche=pose.bouche, cligne=pose.cligne, regard=pose.regard,
                    sourcils=pose.sourcils, bras_g=e["bras_g"], bras_d=e["bras_d"],
                    respiration=pose.respiration, balance=e["balance"] * 0.6)
                # ombre portée au sol : le personnage tient debout dans la scène
                ombre = self._ombre.resize((max(2, round(420 * s)), max(2, round(90 * s))), Image.BILINEAR)
                image.alpha_composite(ombre, (round(ex - ombre.width / 2), round(ey - ombre.height * 0.55)))
                rendu = acteur.personnage.rendre(pt, s)
                ax, ay = _toon.TAILLE[0] / 2 * s, (_toon.TAILLE[1] - _toon.SOL) * s
                image.alpha_composite(rendu, (round(ex - ax), round(ey - ay)))
                continue
            rendu = acteur.personnage.rendre(pose)
            if abs(e["balance"]) > 0.05:  # le corps se balance, pivot aux hanches
                rendu = rendu.rotate(e["balance"], resample=Image.BICUBIC,
                                     center=(rendu.width / 2, rendu.height))
            if abs(s - 1) > 1e-3:
                rendu = rendu.resize((round(rendu.width * s), round(rendu.height * s)), Image.BILINEAR)
            image.alpha_composite(rendu, (round(ex - rendu.width / 2), round(ey - rendu.height)))
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
                # le jeu se calcule dans l'ordre ; les images, en parallèle
                etats = [self.etat(i) for i in range(self.n_images)]
                global _EPISODE
                _EPISODE = self
                processus = max(1, min(4, os.cpu_count() or 1))
                taches = [(i, etats[i], (w, h)) for i in range(self.n_images)]
                if processus > 1:
                    with multiprocessing.get_context("fork").Pool(processus) as pool:
                        for octets in pool.imap(_image_brute, taches, chunksize=4):
                            ffmpeg.stdin.write(octets)
                else:
                    for tache in taches:
                        ffmpeg.stdin.write(_image_brute(tache))
            finally:
                ffmpeg.stdin.close()
                if ffmpeg.wait() != 0:
                    raise RuntimeError("ffmpeg a échoué")
        return sortie


_EPISODE: "Episode | None" = None


def _image_brute(tache) -> bytes:
    """Rendu d'une image dans un processus fils (l'épisode est hérité par fork)."""
    i, etats, (w, h) = tache
    image = _EPISODE.image(i, etats)
    if (w, h) != image.size:
        image = image.resize((w, h), Image.BILINEAR)
    return image.tobytes()
