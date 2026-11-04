"""Personnage en marionnette découpée : des calques, chacun à sa profondeur.

Pour tourner la tête, pas besoin de dessiner un profil. Chaque calque glisse
d'autant plus qu'il est près du spectateur (nez, bouche, yeux). Ceux du fond
(oreilles, cheveux de derrière) glissent en sens inverse et passent derrière
la tête. Avec un léger écrasement horizontal des traits, l'œil y voit une
rotation : c'est la parallaxe, le principe des logiciels de marionnettes 2D.

Un personnage est un dossier d'images PNG transparentes :

- soit avec un ``personnage.json`` qui décrit chaque partie ;
- soit sans, si les images suivent les noms convenus (``NOMS``). Chacune est
  alors un calque de la taille du dessin entier, exporté tel quel depuis
  Krita ou Photopea. La position de chaque partie se déduit de ce qui n'est
  pas transparent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

# Partie conventionnelle -> (groupe, profondeur, compression, rôle).
# Ordre = ordre d'empilement, du fond vers l'avant.
NOMS: dict[str, tuple[str, float, float, str]] = {
    "cheveux_arriere": ("tete", -0.55, 0.05, ""),
    "corps": ("corps", 0.0, 0.0, ""),
    "cou": ("corps", 0.0, 0.0, ""),
    "oreilles": ("tete", -0.25, 0.0, ""),
    "tete": ("tete", 0.05, 0.06, ""),
    "joues": ("tete", 0.55, 0.25, ""),
    "yeux": ("tete", 0.7, 0.28, "yeux"),
    "pupilles": ("tete", 0.78, 0.3, "pupilles"),
    "sourcils": ("tete", 0.75, 0.28, "sourcils"),
    "nez": ("tete", 1.0, 0.2, ""),
    "bouche": ("tete", 0.85, 0.3, "bouche"),
    "cheveux_avant": ("tete", 0.2, 0.08, ""),
    "coiffe": ("tete", 0.2, 0.08, ""),
}

# Formes de bouche de Rhubarb Lip Sync (planche de Preston Blair).
VISEMES = "XABCDEFGH"


@dataclass
class Partie:
    nom: str
    images: dict[str, Image.Image]  # variante -> image ; "defaut" s'il n'y en a qu'une
    position: tuple[float, float]  # centre de l'image, dans le repère du personnage
    groupe: str = "tete"  # "tete" (tourne avec elle) ou "corps"
    profondeur: float = 0.0  # > 0 vers le spectateur, < 0 derrière la tête
    compression: float = 0.0  # écrasement horizontal quand la tête est tournée à fond
    role: str = ""  # "bouche", "yeux", "pupilles", "sourcils" ou ""
    course: float = 0.0  # pupilles : déplacement maximal du regard, en pixels

    def image(self, pose: "Pose") -> Image.Image | None:
        if self.role == "bouche":
            return self.images.get(pose.bouche) or self.images.get("X") or _premiere(self.images)
        if self.role == "yeux":
            return self.images.get("fermes" if pose.cligne else "ouverts") or _premiere(self.images)
        if self.role == "pupilles" and pose.cligne:
            return None
        return self.images.get("defaut") or _premiere(self.images)


def _premiere(images: dict[str, Image.Image]) -> Image.Image:
    return next(iter(images.values()))


@dataclass
class Pose:
    lacet: float = 0.0  # tête tournée : -1 vers la gauche de l'écran, 1 vers la droite
    tangage: float = 0.0  # -1 lève la tête, 1 la baisse
    roulis: float = 0.0  # inclinaison, en degrés (positif : sens trigonométrique)
    regard: tuple[float, float] = (0.0, 0.0)  # pupilles, de -1 à 1
    bouche: str = "X"
    cligne: bool = False
    respiration: float = 0.0  # de 0 à 1


@dataclass
class Personnage:
    nom: str
    taille: tuple[int, int]
    parties: list[Partie]
    pivot_cou: tuple[float, float]
    amplitude_lacet: float = 55.0  # glissement, en pixels, d'une partie de profondeur 1
    amplitude_tangage: float = 28.0
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def centre_tete(self) -> float:
        tete = [p for p in self.parties if p.nom == "tete"]
        return tete[0].position[0] if tete else self.taille[0] / 2

    def _redimensionne(self, partie: Partie, image: Image.Image, sx: float) -> Image.Image:
        sx = round(sx, 2)  # quelques dizaines de tailles différentes au plus
        if sx >= 0.995:
            return image
        cle = (partie.nom, id(image), sx)
        if cle not in self._cache:
            largeur = max(1, round(image.width * sx))
            self._cache[cle] = image.resize((largeur, image.height), Image.LANCZOS)
        return self._cache[cle]

    def rendre(self, pose: Pose) -> Image.Image:
        """L'image RGBA du personnage dans cette pose, à la taille ``taille``.

        Les parties sont empilées dans l'ordre. Chaque suite de parties d'un
        même groupe forme un calque : les cheveux de derrière, qui tournent
        avec la tête, restent ainsi sous le corps."""
        cx = self.centre_tete
        respire = 2.5 * math.sin(2 * math.pi * pose.respiration)
        calques: list[tuple[str, Image.Image]] = []
        for p in self.parties:
            image = p.image(pose)
            if image is None:
                continue
            x, y = p.position
            if p.groupe == "tete":
                serre = 1 - 0.12 * abs(pose.lacet)  # les traits se rapprochent de l'axe
                x = cx + (x - cx) * serre + pose.lacet * p.profondeur * self.amplitude_lacet
                y += pose.tangage * p.profondeur * self.amplitude_tangage
                if p.role == "pupilles":
                    x += pose.regard[0] * p.course
                    y += pose.regard[1] * p.course * 0.6
                image = self._redimensionne(p, image, 1 - p.compression * abs(pose.lacet))
            y += respire if p.groupe == "corps" else respire * 0.6
            if not calques or calques[-1][0] != p.groupe:
                calques.append((p.groupe, Image.new("RGBA", self.taille, (0, 0, 0, 0))))
            calques[-1][1].alpha_composite(
                image, (round(x - image.width / 2), round(y - image.height / 2)))
        rendu = Image.new("RGBA", self.taille, (0, 0, 0, 0))
        for groupe, calque in calques:
            if groupe == "tete" and abs(pose.roulis) > 0.05:
                calque = calque.rotate(pose.roulis, resample=Image.BICUBIC, center=self.pivot_cou)
            rendu.alpha_composite(calque)
        return rendu


def _rogner(image: Image.Image) -> tuple[Image.Image, tuple[float, float]]:
    """Calque pleine taille -> (partie utile, centre de cette partie)."""
    boite = image.getchannel("A").getbbox()
    if boite is None:
        raise ValueError("calque entièrement transparent")
    x0, y0, x1, y1 = boite
    return image.crop(boite), ((x0 + x1) / 2, (y0 + y1) / 2)


def _ouvrir(chemin: Path) -> Image.Image:
    return Image.open(chemin).convert("RGBA")


def charger(dossier: str | Path) -> Personnage:
    """Charge un personnage depuis son dossier (voir le haut du module)."""
    dossier = Path(dossier)
    fichier = dossier / "personnage.json"
    if fichier.exists():
        return _depuis_json(dossier, json.loads(fichier.read_text(encoding="utf-8")))
    return _depuis_calques(dossier)


def _depuis_json(dossier: Path, d: dict) -> Personnage:
    parties = []
    for p in d["parties"]:
        groupe, profondeur, compression, role = NOMS.get(p["nom"], ("tete", 0.0, 0.0, ""))
        fichiers = p.get("images") or {"defaut": p["image"]}
        images = {v: _ouvrir(dossier / f) for v, f in fichiers.items()}
        if "position" in p:
            position = tuple(p["position"])
        else:  # calques pleine taille : on rogne et on déduit la position
            rognees = {v: _rogner(im) for v, im in images.items()}
            images = {v: im for v, (im, _) in rognees.items()}
            position = _premiere({v: c for v, (_, c) in rognees.items()})
        parties.append(Partie(
            nom=p["nom"], images=images, position=position,
            groupe=p.get("groupe", groupe), profondeur=p.get("profondeur", profondeur),
            compression=p.get("compression", compression), role=p.get("role", role),
            course=p.get("course", 12.0 if role == "pupilles" else 0.0)))
    return Personnage(
        nom=d.get("nom", dossier.name), taille=tuple(d["taille"]), parties=parties,
        pivot_cou=tuple(d.get("pivot_cou") or _pivot(parties, tuple(d["taille"]))),
        amplitude_lacet=d.get("amplitude_lacet", 55.0),
        amplitude_tangage=d.get("amplitude_tangage", 28.0))


def _depuis_calques(dossier: Path) -> Personnage:
    """Dossier d'images pleine taille aux noms convenus : ``tete.png``,
    ``yeux_ouverts.png``, ``yeux_fermes.png``, ``bouche_X.png`` … ``bouche_H.png``."""
    fichiers = {f.stem: f for f in dossier.glob("*.png")}
    if "tete" not in fichiers:
        raise FileNotFoundError(f"{dossier} : ni personnage.json ni tete.png")
    taille = _ouvrir(fichiers["tete"]).size
    description = []
    for nom in NOMS:
        variantes = {k.split("_", 1)[1] if k != nom else "defaut": f.name
                     for k, f in fichiers.items() if k == nom or k.startswith(nom + "_")}
        if variantes:
            description.append({"nom": nom, "images": variantes})
    return _depuis_json(dossier, {"nom": dossier.name, "taille": taille, "parties": description})


def _pivot(parties: list[Partie], taille: tuple[int, int]) -> tuple[float, float]:
    """Par défaut, la tête pivote au bas de son ovale."""
    for p in parties:
        if p.nom == "tete":
            im = _premiere(p.images)
            return (p.position[0], p.position[1] + im.height * 0.42)
    return (taille[0] / 2, taille[1] / 2)
