"""La tête en volume : le dessin plaqué sur un ellipsoïde qui tourne.

Faire glisser des calques plats (la parallaxe) donne une impression de
rotation, mais la tête reste une découpe de carton. Ici, le dessin de la
tête (visage, foulard, bonnet, barbe) est plaqué sur un ellipsoïde, qui
tourne en perspective. Le côté qui s'éloigne s'écrase vers le bord, celui
qui vient vers nous s'élargit, et le contour du visage glisse sur la forme.
C'est ce que font les têtes « 3D » des logiciels de marionnettes, avec un
seul dessin de face.

Les traits (yeux, nez, bouche, sourcils, oreilles) restent des images à part.
Chacun est un point de la surface, avec un relief propre : le nez dépasse,
les oreilles sont en arrière. Son image se déplace selon la rotation, et
s'écrase autant que la surface à cet endroit. L'œil du fond rétrécit, le nez
déborde sur la joue. Une ombre douce vers les bords du visage donne le
volume.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
from PIL import Image
from scipy import ndimage

# Parties plaquées sur la surface ; les autres parties de la tête sont des traits.
SURFACE = {"tete", "coiffe", "cheveux_avant", "cheveux_arriere", "joues"}
# Relief des traits, en fraction du rayon horizontal de la tête.
RELIEF = {"nez": 0.17, "bouche": 0.06, "sourcils": 0.03, "yeux": 0.0, "blanc_yeux": -0.01,
          "pupilles": 0.0, "oreilles": -0.35}


class Volume:
    """Géométrie de la tête d'un personnage, et rendu de ses parties en volume."""

    def __init__(self, parties, angle_lacet: float = 38.0, angle_tangage: float = 14.0):
        surfaces = [p for p in parties if p.groupe == "tete" and p.nom in SURFACE]
        if not surfaces:
            raise ValueError("pas de tête (partie « tete ») à mettre en volume")
        boites = []
        for p in surfaces:
            for im in p.images.values():
                x, y = p.position
                boites.append((x - im.width / 2, y - im.height / 2, x + im.width / 2,
                               y + im.height / 2))
        x0, y0 = min(b[0] for b in boites), min(b[1] for b in boites)
        x1, y1 = max(b[2] for b in boites), max(b[3] for b in boites)
        tete = next((p for p in surfaces if p.nom == "tete"), surfaces[0])
        self.cx = tete.position[0]
        self.cy = (y0 + y1) / 2
        self.A = max(self.cx - x0, x1 - self.cx) * 1.06
        self.B = max(self.cy - y0, y1 - self.cy) * 1.03
        self.C = self.A
        self.angle_lacet, self.angle_tangage = angle_lacet, angle_tangage
        # la toile du rendu : la boîte de l'ellipsoïde
        self.ox, self.oy = int(math.floor(self.cx - self.A)) - 2, int(math.floor(self.cy - self.B)) - 2
        self.w = int(math.ceil(2 * self.A)) + 5
        self.h = int(math.ceil(2 * self.B)) + 5
        u = np.arange(self.w, dtype=np.float32) + self.ox + 0.5 - self.cx
        v = np.arange(self.h, dtype=np.float32) + self.oy + 0.5 - self.cy
        self.u, self.v = np.meshgrid(u, v)
        t = 1 - (self.u / self.A) ** 2 - (self.v / self.B) ** 2
        self.dedans = t > 0
        self.zs = self.C * np.sqrt(np.clip(t, 0, None))  # surface vue de face
        nz = np.sqrt(np.clip(t, 0, None))
        # ombre douce vers les bords, un peu plus marquée en bas à droite
        lumiere = 0.80 + 0.20 * nz ** 0.6 - 0.04 * (self.u / self.A + self.v / self.B)
        self.ombre = np.clip(lumiere, 0.6, 1.05).astype(np.float32)
        self._sources: dict[tuple, np.ndarray] = {}
        self._rendus: OrderedDict = OrderedDict()

    # --- géométrie ---

    def angles(self, lacet: float, tangage: float) -> tuple[float, float]:
        return math.radians(lacet * self.angle_lacet), math.radians(tangage * self.angle_tangage)

    def z_surface(self, x: float, y: float) -> float:
        t = 1 - ((x - self.cx) / self.A) ** 2 - ((y - self.cy) / self.B) ** 2
        return self.C * math.sqrt(t) if t > 0 else 0.0

    def projeter(self, x: float, y: float, z: float, th: float, ph: float) -> tuple[float, float]:
        """Point de la tête (repère du personnage, z vers le spectateur) -> écran."""
        x, y = x - self.cx, y - self.cy
        x1 = x * math.cos(th) + z * math.sin(th)
        z1 = -x * math.sin(th) + z * math.cos(th)
        y2 = y * math.cos(ph) + z1 * math.sin(ph)
        return x1 + self.cx, y2 + self.cy

    def trait(self, nom: str, x: float, y: float, th: float, ph: float, relief=None):
        """Position à l'écran et écrasement (sx, sy) d'un trait centré en (x, y)."""
        r = (RELIEF.get(nom, 0.0) if relief is None else relief) * self.A
        z = self.z_surface(x, y) + r
        px, py = self.projeter(x, y, z, th, ph)
        d = 0.06 * self.A

        def p(dx, dy):
            return self.projeter(x + dx, y + dy, self.z_surface(x + dx, y + dy) + r, th, ph)

        (xa, _), (xb, _) = p(-d, 0), p(d, 0)
        (_, ya), (_, yb) = p(0, -d), p(0, d)
        sx = max(0.25, min(1.3, (xb - xa) / (2 * d)))
        sy = max(0.4, min(1.2, (yb - ya) / (2 * d)))
        return px, py, sx, sy

    # --- la surface ---

    def _source(self, parties, pose) -> np.ndarray:
        """Les parties plaquées, empilées sur la toile de l'ellipsoïde (prémultipliées)."""
        cle = tuple(p.nom for p in parties)
        if cle not in self._sources:
            toile = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
            for p in parties:
                im = p.image(pose)
                toile.alpha_composite(im, (round(p.position[0] - im.width / 2) - self.ox,
                                           round(p.position[1] - im.height / 2) - self.oy))
            a = np.asarray(toile).astype(np.float32) / 255
            a[..., :3] *= a[..., 3:4]
            self._sources[cle] = a
        return self._sources[cle]

    def surface(self, parties, pose, th: float, ph: float) -> tuple[Image.Image, tuple[int, int]]:
        """Les parties plaquées, tournées de (th, ph) : image et coin haut gauche."""
        cle = (tuple(p.nom for p in parties), round(math.degrees(th) * 3), round(math.degrees(ph) * 3))
        if cle in self._rendus:
            self._rendus.move_to_end(cle)
            return self._rendus[cle], (self.ox, self.oy)
        src = self._source(parties, pose)
        # écran (u, v, surface) -> tête de face : on défait le tangage, puis le lacet
        u, v, zs = self.u, self.v, self.zs
        y1 = v * math.cos(ph) - zs * math.sin(ph)
        z1 = v * math.sin(ph) + zs * math.cos(ph)
        xs = u * math.cos(th) - z1 * math.sin(th)
        zsrc = u * math.sin(th) + z1 * math.cos(th)
        visible = self.dedans & (zsrc >= 0)
        colonnes = xs + self.cx - self.ox - 0.5
        lignes = y1 + self.cy - self.oy - 0.5
        sortie = np.empty((self.h, self.w, 4), np.float32)
        for c in range(4):
            sortie[..., c] = ndimage.map_coordinates(src[..., c], [lignes, colonnes], order=1,
                                                     mode="constant", cval=0.0)
        sortie *= visible[..., None]
        alpha = sortie[..., 3:4]
        rgb = np.where(alpha > 1e-4, sortie[..., :3] / np.maximum(alpha, 1e-4), 0)
        rgb *= self.ombre[..., None]
        image = Image.fromarray(np.dstack([np.clip(rgb * 255, 0, 255),
                                           np.clip(alpha * 255, 0, 255)]).astype(np.uint8), "RGBA")
        self._rendus[cle] = image
        if len(self._rendus) > 400:
            self._rendus.popitem(last=False)
        return image, (self.ox, self.oy)
