"""Personnage multi-vues : le tour complet avec les vrais dessins.

C'est la méthode de Cartoon Animator pour ses personnages qui tournent. Le
personnage est dessiné sous plusieurs angles (un *turnaround*) : de face, de
trois-quarts, de profil, de trois-quarts dos, de dos. Les angles négatifs
sont les mêmes dessins en miroir. À chaque image :

1. pour l'angle voulu, on prend les deux vues dessinées les plus proches ;
2. chacune est tournée en volume du petit angle qui la sépare de l'angle
   voulu : le corps comme un cylindre, la tête comme un ellipsoïde ;
3. on passe de l'une à l'autre par un fondu bref, à mi-chemin.

La tête est découpée à part, au cou : elle tourne indépendamment du corps
(regarder quelqu'un, se retourner), hoche et s'incline. La bouche du dessin
est remplacée par les 9 formes de la planche des bouches, à l'échelle de
chaque vue, écrasées de trois-quarts et de profil. Les yeux clignent.

Préparer un personnage depuis sa planche (5 vues de gauche à droite, fond
blanc) ::

    python -m animation tour fanta
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .decoupe import _liseres

ANGLES = (0, 45, 90, 135, 180)
CONTOUR = (45, 22, 12)
RACINE = Path(__file__).parent

# Bouche de chaque vue dessinée (de face, trois-quarts, profil), dans la
# planche : x0, y0, x1, y1. De dos, pas de bouche.
REPERES = {
    "fanta": {"bouches": [(207, 143, 246, 164), (620, 141, 654, 162), (996, 135, 1018, 159)],
              "yeux": [[(184, 100, 216, 122), (238, 100, 268, 122)],
                       [(592, 100, 625, 120), (645, 100, 669, 120)],
                       [(975, 100, 1001, 120)]]},
    "bakary": {"bouches": [(213, 153, 257, 175), (629, 153, 665, 174), (1034, 155, 1062, 179)],
               "yeux": [[(193, 107, 223, 125), (246, 107, 276, 125)],
                        [(605, 107, 638, 127), (656, 107, 673, 127)],
                        [(1011, 105, 1038, 125)]]},
}


@dataclass
class PoseTour:
    corps: float = 0.0  # orientation du corps, degrés (0 : de face ; 90 : profil, regard vers la droite)
    tete: float = 0.0  # rotation de la tête par rapport au corps, degrés
    tangage: float = 0.0  # degrés ; positif : la tête se baisse
    roulis: float = 0.0  # degrés
    bouche: str = "X"
    cligne: bool = False
    respiration: float = 0.0
    balance: float = 0.0


# --- préparation ---------------------------------------------------------

def _retoucher(rgba: np.ndarray, masque: np.ndarray, iterations: int = 120) -> None:
    """Remplit ``masque`` en prolongeant les couleurs qui l'entourent (diffusion),
    comme l'outil de retouche d'un logiciel de dessin. En place."""
    ys, xs = np.nonzero(masque)
    if not len(ys):
        return
    y0, y1, x0, x1 = max(0, ys.min() - 2), ys.max() + 3, max(0, xs.min() - 2), xs.max() + 3
    sous = rgba[y0:y1, x0:x1, :3].astype(np.float32)
    m = masque[y0:y1, x0:x1]
    connu = ~m & (rgba[y0:y1, x0:x1, 3] > 200)
    sous[m] = np.median(sous[connu], axis=0) if connu.any() else 128
    for _ in range(iterations):
        moy = (np.roll(sous, 1, 0) + np.roll(sous, -1, 0) + np.roll(sous, 1, 1) + np.roll(sous, -1, 1)) / 4
        sous[m] = moy[m]
    rgba[y0:y1, x0:x1, :3][m] = np.clip(sous[m], 0, 255).astype(np.uint8)



def _figures(image: np.ndarray, fond: np.ndarray) -> list[tuple[int, int]]:
    colonnes = (~fond).sum(0) > 3
    runs, debut = [], None
    for x, c in enumerate(list(colonnes) + [False]):
        if c and debut is None:
            debut = x
        if not c and debut is not None:
            if x - debut > 30:
                runs.append((debut, x))
            debut = None
    if len(runs) != 5:
        raise ValueError(f"{len(runs)} figures trouvées dans la planche, 5 attendues")
    return runs


def preparer(nom: str, source: Path | None = None, destination: Path | None = None) -> Path:
    source = source or RACINE / "brut" / nom
    destination = destination or RACINE / "personnages" / f"{nom}_tour"
    image = np.asarray(Image.open(source / "tour.webp").convert("RGB")).astype(np.float32)
    H, Wp = image.shape[:2]
    blanc = image.min(axis=2) > 225
    lab, n = ndimage.label(blanc)
    bords = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    fond = np.isin(lab, bords[bords > 0])
    runs = _figures(image, fond)
    haut = int(np.nonzero((~fond).any(1))[0].min())
    # poches de fond enfermées entre bras et corps : blanc pur, sous le cou
    tres_blanc = image.min(axis=2) > 244
    lab2, n2 = ndimage.label(tres_blanc & ~fond)
    if n2:
        tailles = ndimage.sum(tres_blanc, lab2, range(1, n2 + 1))
        ys = ndimage.center_of_mass(tres_blanc, lab2, range(1, n2 + 1))
        for k, (t, (cy, _)) in enumerate(zip(tailles, ys), start=1):
            if t > 25 and cy > haut + 0.30 * (H - haut):
                fond |= lab2 == k
    alpha = _liseres(image, fond)
    rgba = np.dstack([image, alpha * 255]).clip(0, 255).astype(np.uint8)
    reperes = REPERES.get(nom, {})
    marge = 30
    largeur = max(x1 - x0 for x0, x1 in runs) + 2 * marge + 60
    destination.mkdir(parents=True, exist_ok=True)
    for ancien in destination.glob("*.png"):
        ancien.unlink()
    cous = []
    for x0, x1 in runs:
        a = alpha[:, x0:x1] > 0.5
        ys = np.nonzero(a.any(1))[0]
        top, bas = ys.min(), ys.max()
        h = bas - top
        cou_y = min((a[y].sum(), y) for y in range(int(top + 0.13 * h), int(top + 0.32 * h)))[1]
        xs = np.nonzero(a[cou_y])[0]
        cous.append((x0 + (xs.min() + xs.max()) / 2, cou_y, h, top))
    # La planche « yeux fermés » de ChatGPT ne se superpose pas assez bien
    # (personnage redessiné) : les paupières sont faites ici, pour chaque vue.
    vues = []
    for k, (x0, x1) in enumerate(runs):
        a = alpha[:, x0:x1] > 0.5
        ys = np.nonzero(a.any(1))[0]
        top, bas = ys.min(), ys.max()
        h = bas - top
        # le cou : la ligne la plus étroite entre la tête et les épaules
        lignes = range(int(top + 0.13 * h), int(top + 0.32 * h))
        largeurs = [(a[y].sum(), y) for y in lignes]
        cou_y = min(largeurs)[1]
        xs = np.nonzero(a[cou_y])[0]
        cou_x = x0 + (xs.min() + xs.max()) / 2
        # la figure sur une toile commune, le cou au milieu
        decal = int(round(largeur / 2 - cou_x))
        toile = np.zeros((H, largeur, 4), np.uint8)
        a0, a1 = max(0, x0 - marge), min(Wp, x1 + marge)
        b0, b1 = a0 + decal, a1 + decal
        toile[:, max(0, b0):min(largeur, b1)] = rgba[:, a0 + max(0, -b0):a1 - max(0, b1 - largeur)]
        toile_rgb = toile[..., :3].astype(np.float32)
        tete = toile.copy()
        tete[cou_y + 3:] = 0
        corps = toile.copy()
        corps[:cou_y - 8] = 0
        info = {"angle": ANGLES[k], "cou": [largeur / 2, int(cou_y)], "bas": int(bas)}
        if k < 3 and "bouches" in reperes:  # bouche et yeux : vues de face, trois-quarts, profil
            bx0, by0, bx1, by1 = reperes["bouches"][k]
            bx0, bx1 = bx0 + decal, bx1 + decal
            peau = np.median(toile_rgb[by0 - 9:by0 - 4, (bx0 + bx1) // 2 - 3:(bx0 + bx1) // 2 + 4]
                             .reshape(-1, 3), axis=0)
            info["bouche"] = [bx0, by0, bx1, by1]
            info["peau"] = [int(c) for c in peau]
            if k < 2:  # de face et de trois-quarts : la bouche du dessin est effacée
                masque = np.zeros(tete.shape[:2], bool)
                yy, xx = np.ogrid[:tete.shape[0], :tete.shape[1]]
                cxm, cym = (bx0 + bx1) / 2, (by0 + by1) / 2
                masque |= (((xx - cxm) / ((bx1 - bx0) / 2 + 3)) ** 2 + ((yy - cym) / ((by1 - by0) / 2 + 3)) ** 2) <= 1
                tete = tete.copy()
                _retoucher(tete, masque & (tete[..., 3] > 200))
            # les yeux (repères notés sur la planche)
            yeux = [[ex0 + decal, ey0, ex1 + decal, ey1] for ex0, ey0, ex1, ey1 in reperes["yeux"][k]]
            info["yeux"] = yeux
            # les paupières fermées, prêtes à poser : planche des yeux fermés
            # si elle existe, sinon peau retouchée et un trait de paupière
            ferme = tete.copy()
            masque = np.zeros(tete.shape[:2], bool)
            for ex0, ey0, ex1, ey1 in yeux:
                masque[max(0, ey0 - 3):ey1 + 2, max(0, ex0 - 4):ex1 + 4] = True
            _retoucher(ferme, masque & (tete[..., 3] > 200))
            im = Image.fromarray(ferme, "RGBA")
            d = ImageDraw.Draw(im)
            for ex0, ey0, ex1, ey1 in yeux:
                d.arc([ex0 + 1, ey0 + (ey1 - ey0) * 0.05, ex1 - 1, ey1 - (ey1 - ey0) * 0.15], 15, 165,
                      fill=CONTOUR + (255,), width=3)
            ferme = np.asarray(im).copy()
            Image.fromarray(ferme, "RGBA").save(destination / f"tete_{ANGLES[k]:03d}_fermes.png")
        Image.fromarray(tete, "RGBA").save(destination / f"tete_{ANGLES[k]:03d}.png")
        Image.fromarray(corps, "RGBA").save(destination / f"corps_{ANGLES[k]:03d}.png")
        vues.append(info)
    # les bouches : celles du personnage découpé de face (même dessinateur)
    bouches = {}
    dossier_bouches = RACINE / "personnages" / nom
    for f in "XABCDEFGH":
        chemin = dossier_bouches / f"bouche_{f}.png"
        if chemin.exists():
            bouches[f] = chemin.name
            Image.open(chemin).save(destination / f"bouche_{f}.png")
    d = {"nom": nom, "taille": [largeur, H], "vues": vues, "bouches": bouches}
    (destination / "tour.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    return destination


# --- rendu ----------------------------------------------------------------

def _premul(im: Image.Image) -> np.ndarray:
    a = np.asarray(im).astype(np.float32) / 255
    a[..., :3] *= a[..., 3:4]
    return a


def _image(premul: np.ndarray) -> Image.Image:
    al = premul[..., 3:4]
    rgb = np.where(al > 1e-4, premul[..., :3] / np.maximum(al, 1e-4), 0)
    return Image.fromarray((np.dstack([rgb, al]).clip(0, 1) * 255).astype(np.uint8), "RGBA")


def _lisse(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


class _Vue:
    def __init__(self, dossier: Path, info: dict, miroir: bool, largeur: int):
        self.info = info
        self.miroir = miroir
        a = info["angle"]
        tete = Image.open(dossier / f"tete_{a:03d}.png").convert("RGBA")
        corps = Image.open(dossier / f"corps_{a:03d}.png").convert("RGBA")
        f = dossier / f"tete_{a:03d}_fermes.png"
        fermes = Image.open(f).convert("RGBA") if f.exists() else tete
        self.largeur = largeur
        if miroir:
            tete, corps, fermes = (im.transpose(Image.FLIP_LEFT_RIGHT) for im in (tete, corps, fermes))
        self.tete, self.corps, self.tete_fermes = tete, corps, fermes
        self.corps_p = _premul(corps)
        # rayons : demi-largeur du corps, boîte de la tête
        al = np.asarray(corps)[..., 3] > 128
        xs = np.nonzero(al.any(0))[0]
        self.R = max(largeur / 2 - xs.min(), xs.max() - largeur / 2) * 1.12
        bt = tete.getchannel("A").getbbox()
        self.tete_boite = bt
        self.A = (bt[2] - bt[0]) / 2 * 1.08
        self.B = (bt[3] - bt[1]) / 2 * 1.05
        self.tcx, self.tcy = (bt[0] + bt[2]) / 2, (bt[1] + bt[3]) / 2

    def _x(self, x: float) -> float:
        return self.largeur - x if self.miroir else x


class Tour:
    """Un personnage multi-vues, prêt à être rendu dans n'importe quel angle."""

    def __init__(self, dossier: str | Path):
        dossier = Path(dossier)
        d = json.loads((dossier / "tour.json").read_text(encoding="utf-8"))
        self.nom = d["nom"]
        self.taille = tuple(d["taille"])
        self.vues: dict[int, _Vue] = {}
        for info in d["vues"]:
            a = info["angle"]
            self.vues[a] = _Vue(dossier, info, False, self.taille[0])
            if 0 < a < 180:
                self.vues[-a] = _Vue(dossier, info, True, self.taille[0])
        self.vues[-180] = self.vues[180]
        self.bouches = {f: Image.open(dossier / fichier).convert("RGBA")
                        for f, fichier in d["bouches"].items()}
        info0 = d["vues"][0]
        self.cou = tuple(info0["cou"])
        self.pieds_y = info0["bas"]
        self._cache: dict = {}

    @property
    def tete_hauteur(self) -> float:
        """Du sol au centre de la tête, en pixels du dessin."""
        v = self.vues[0]
        return self.pieds_y - v.tcy

    # --- un étage (corps ou tête) dans un angle donné ---

    @staticmethod
    def _voisines(angle: float) -> tuple[int, int, float]:
        a = (angle + 180) % 360 - 180
        v1 = math.floor(a / 45) * 45
        return v1, v1 + 45 if v1 < 180 else 180, (a - v1) / 45

    @staticmethod
    def glissement(vue: _Vue, delta: float) -> float:
        """De combien glissent les détails au milieu du corps, pour ``delta`` degrés."""
        return 0.30 * vue.R * math.sin(math.radians(delta))

    def _corps(self, vue: _Vue, delta: float) -> np.ndarray:
        """Le corps de la vue, tourné de ``delta`` degrés : la silhouette reste en
        place, les détails (encolure, plis, bras) glissent vers le côté où il se
        tourne, d'autant plus qu'ils sont près du milieu. C'est la déformation
        qui, dans les logiciels de marionnettes, relie deux vues dessinées."""
        cle = ("corps", id(vue), round(delta))
        if cle in self._cache:
            return self._cache[cle]
        w = self.taille[0]
        u = np.arange(w, dtype=np.float32) + 0.5 - w / 2
        r = vue.R
        g = self.glissement(vue, round(delta))
        dep = g * np.clip(1 - (u / r) ** 2, 0, 1)
        us = u - dep + w / 2 - 0.5
        src = vue.corps_p
        i0 = np.clip(np.floor(us).astype(int), 0, w - 2)
        f = np.clip(us - i0, 0, 1)[None, :, None]
        out = src[:, i0] * (1 - f) + src[:, i0 + 1] * f
        if len(self._cache) > 600:
            self._cache.clear()
        self._cache[cle] = out
        return out

    def _tete(self, vue: _Vue, delta: float, tangage: float, bouche: str, cligne: bool):
        """La tête de la vue, avec sa bouche et ses yeux, tournée (ellipsoïde)."""
        cle = ("tete", id(vue), round(delta), round(tangage), bouche, cligne)
        if cle in self._cache:
            return self._cache[cle]
        im = (vue.tete_fermes if cligne else vue.tete).copy()
        info = vue.info
        if "bouche" in info:
            self._visage(im, vue, bouche, cligne)
        src = _premul(im)
        h, w = src.shape[:2]
        bx0, by0, bx1, by1 = vue.tete_boite
        m = 4
        x0, y0, x1, y1 = max(0, bx0 - m), max(0, by0 - m), min(w, bx1 + m), min(h, by1 + m)
        sous = src[y0:y1, x0:x1]
        u = np.arange(x0, x1, dtype=np.float32) + 0.5 - vue.tcx
        v = np.arange(y0, y1, dtype=np.float32) + 0.5 - vue.tcy
        U, V = np.meshgrid(u, v)
        A, B = vue.A, vue.B
        t = 1 - (U / A) ** 2 - (V / B) ** 2
        dedans = t > 0
        Z = A * np.sqrt(np.clip(t, 0, None))
        # la tête tourne moins que l'écart d'angle : le reste, c'est la vue voisine
        th, ph = math.radians(0.7 * round(delta)), math.radians(-round(tangage))
        y1_ = V * math.cos(ph) - Z * math.sin(ph)
        z1 = V * math.sin(ph) + Z * math.cos(ph)
        xs = U * math.cos(th) - z1 * math.sin(th)
        zs = U * math.sin(th) + z1 * math.cos(th)
        ok = dedans & (zs >= 0)
        cols = xs + vue.tcx - x0 - 0.5
        rows = y1_ + vue.tcy - y0 - 0.5
        out = np.empty_like(sous)
        for c in range(4):
            out[..., c] = ndimage.map_coordinates(sous[..., c], [rows, cols], order=1, cval=0.0)
        out *= ok[..., None]
        # hors de l'ellipsoïde (bords de la coiffe), on garde le dessin tel quel
        out = np.where(dedans[..., None], out, sous)
        res = (out, (x0, y0))
        if len(self._cache) > 600:
            self._cache.clear()
        self._cache[cle] = res
        return res

    def _visage(self, im: Image.Image, vue: _Vue, bouche: str, cligne: bool) -> None:
        info = vue.info
        bx0, by0, bx1, by1 = info["bouche"]
        if vue.miroir:
            bx0, bx1 = vue.largeur - bx1, vue.largeur - bx0
        angle = abs(info["angle"])
        if angle == 90:  # de profil : les lèvres du dessin, et seulement l'ouverture
            ouverture = {"B": 0.3, "C": 0.6, "D": 1.0, "E": 0.75, "F": 0.4, "G": 0.25, "H": 0.6}.get(bouche, 0)
            if ouverture:
                d = ImageDraw.Draw(im)
                largeur, hauteur = (bx1 - bx0) * 0.55, (by1 - by0) * 0.75 * ouverture
                cx = bx1 - (bx1 - bx0) * 0.38 if not vue.miroir else bx0 + (bx1 - bx0) * 0.38
                cy = (by0 + by1) / 2 + 1
                d.ellipse([cx - largeur / 2, cy - hauteur / 2, cx + largeur / 2, cy + hauteur / 2],
                          fill=(70, 20, 25, 255), outline=CONTOUR + (255,), width=1)
            return
        sprite = self.bouches.get(bouche) or self.bouches.get("X")
        if sprite is None:
            return
        ref = self.bouches["X"].width
        largeur = bx1 - bx0
        ecrase = {0: 1.0, 45: 0.85}.get(angle, 1.0)
        e = largeur / ref
        sw = max(2, round(sprite.width * e * ecrase))
        sh = max(2, round(sprite.height * e))
        s = sprite.resize((sw, sh), Image.LANCZOS)
        if vue.miroir and angle:
            s = s.transpose(Image.FLIP_LEFT_RIGHT)
        cx = (bx0 + bx1) / 2
        im.alpha_composite(s, (round(cx - sw / 2), round(by0 - 1)))

    # --- l'image ---

    def rendre(self, pose: PoseTour, echelle: float = 1.0) -> Image.Image:
        w, h = self.taille
        total = np.zeros((h, w, 4), np.float32)
        # le corps : deux vues voisines, fondu à mi-chemin
        v1, v2, f = self._voisines(pose.corps)
        poids2 = 1.0 if f >= 0.5 else 0.0  # changement de dessin instantané, comme Cartoon Animator
        respire = 1.5 * math.sin(2 * math.pi * pose.respiration)
        cou_dx = 0.0
        for va, poids in ((v1, 1 - poids2), (v2, poids2)):
            if poids <= 0.001:
                continue
            vue = self.vues[va]
            delta = ((pose.corps - va + 180) % 360) - 180
            total += self._corps(vue, delta) * poids
            cou_dx += poids * self.glissement(vue, delta)
        # la tête, à sa propre orientation, posée sur le cou
        angle_tete = pose.corps + pose.tete
        t1, t2, g = self._voisines(angle_tete)
        p2 = 1.0 if g >= 0.5 else 0.0
        tete = np.zeros((h, w, 4), np.float32)
        for va, poids in ((t1, 1 - p2), (t2, p2)):
            if poids <= 0.001:
                continue
            vue = self.vues[va]
            delta = ((angle_tete - va + 180) % 360) - 180
            arr, (x0, y0) = self._tete(vue, delta, pose.tangage, pose.bouche, pose.cligne)
            hh, ww = arr.shape[:2]
            dx = int(round(cou_dx))
            ys, xs = slice(max(0, y0), min(h, y0 + hh)), slice(max(0, x0 + dx), min(w, x0 + dx + ww))
            tete[ys, xs] += arr[ys.start - y0:ys.stop - y0, xs.start - x0 - dx:xs.stop - x0 - dx] * poids
        image = _image(total)
        tete_im = _image(tete)
        if abs(pose.roulis) > 0.05:
            tete_im = tete_im.rotate(pose.roulis, resample=Image.BICUBIC, center=self.cou)
        image.alpha_composite(tete_im, (0, int(round(respire * 0.5))))
        if abs(pose.balance) > 0.05:
            image = image.rotate(pose.balance, resample=Image.BICUBIC, center=(w / 2, self.pieds_y))
        if abs(echelle - 1) > 1e-3:
            image = image.resize((max(2, round(w * echelle)), max(2, round(h * echelle))), Image.LANCZOS)
        return image
