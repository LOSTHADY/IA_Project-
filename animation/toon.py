"""Personnage en volume, rendu comme un dessin animé : tour complet à 360°.

Un dessin de face ne montre pas le dos d'une tête. Pour qu'un personnage
fasse un tour complet, il faut connaître sa forme. Ici, le personnage est
fait de volumes simples : crâne et menton, nez, foulard et son nœud ou
bonnet, cou, boubou (forme de révolution, plus large en bas), manches,
avant-bras, mains et pieds. Chaque image est calculée par lancer de rayons,
pixel par pixel. Le rendu est celui d'un dessin animé (*cel shading*) : deux
tons par matière, une lumière franche, des contours épais à la silhouette et
entre les matières.

Le visage est celui du dessin : yeux, pupilles, sourcils, nez, bouches,
découpés par ``decoupe.py``, plaqués sur l'avant de la tête. Il tourne avec
elle, l'œil du fond fuit, et de profil on voit la joue, l'oreille et le nez
qui dépasse. De dos, on voit le foulard et son nœud.

Repère : x vers la droite, y vers le bas, z vers le spectateur ; l'origine
est au sol, entre les pieds. Unité : le pixel du personnage à l'échelle 1.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

LUMIERE = np.array([-0.45, -0.55, 0.70], np.float32)
LUMIERE = LUMIERE / np.linalg.norm(LUMIERE)
CONTOUR = np.array([45, 22, 12], np.float32) / 255
TAILLE = (760, 1420)  # toile du personnage à l'échelle 1
SOL = 30  # marge sous les pieds


def _rgb(c) -> np.ndarray:
    if isinstance(c, str):
        c = c.lstrip("#")
        c = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    return np.asarray(c, np.float32) / 255


def _rot(lacet: float, tangage: float = 0.0) -> np.ndarray:
    """Rotation : lacet autour de y, puis tangage autour de x (radians)."""
    cy, sy, cx, sx = math.cos(lacet), math.sin(lacet), math.cos(tangage), math.sin(tangage)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rx @ ry


@dataclass
class PoseToon:
    corps: float = 0.0  # orientation du corps, en degrés (0 : de face, 90 : profil droit)
    tete: float = 0.0  # rotation de la tête par rapport au corps, en degrés
    tangage: float = 0.0  # degrés ; positif : la tête se baisse
    roulis: float = 0.0  # degrés
    bouche: str = "X"
    cligne: bool = False
    regard: tuple[float, float] = (0.0, 0.0)
    sourcils: float = 0.0
    bras_g: float = 0.0  # 0 : le long du corps ; 1 : avant-bras levé devant (geste)
    bras_d: float = 0.0
    respiration: float = 0.0
    balance: float = 0.0  # inclinaison du corps, en degrés


@dataclass
class Modele:
    """Dimensions et couleurs ; les valeurs par défaut font un adulte de 1300 px."""
    peau: np.ndarray
    habit: np.ndarray
    coiffe: np.ndarray
    motif_coiffe: np.ndarray
    pieds: np.ndarray = field(default_factory=lambda: _rgb((95, 60, 35)))
    tete_c: tuple = (0.0, -1105.0, 0.0)
    tete_r: tuple = (150.0, 170.0, 148.0)
    coiffe_type: str = "foulard"  # "foulard", "bonnet" ou "aucun"
    barbe: np.ndarray | None = None
    epaules: float = -915.0
    ourlet: float = -40.0
    largeur_epaules: float = 170.0
    largeur_ourlet: float = 230.0
    visage: "Visage | None" = None
    broderie: Image.Image | None = None  # motif de l'encolure, plaqué devant le boubou


class Visage:
    """Les traits du dessin (découpés par decoupe.py), à plaquer sur la tête."""

    def __init__(self, dossier: str | Path, rayon_tete: float):
        dossier = Path(dossier)
        d = json.loads((dossier / "personnage.json").read_text(encoding="utf-8"))
        parties = {p["nom"]: p for p in d["parties"]}
        tete = parties["tete"]
        im_tete = Image.open(dossier / tete["images"]["defaut"])
        self.echelle = 0.95 * rayon_tete / (im_tete.width / 2)
        self.cx, self.cy = tete["position"]
        self.cy += im_tete.height * 0.07  # centre des traits, un peu sous le milieu
        self.parties = {}
        for nom in ("blanc_yeux", "pupilles", "yeux", "sourcils", "nez", "bouche"):
            if nom in parties:
                p = parties[nom]
                self.parties[nom] = (p["position"],
                                     {v: Image.open(dossier / f).convert("RGBA")
                                      for v, f in p["images"].items()})
        self._cache: dict = {}

    def texture(self, pose: PoseToon, taille: int) -> np.ndarray:
        """Les traits vus de face, sur une texture carrée centrée sur le visage."""
        cle = (taille, pose.bouche, pose.cligne, round(pose.regard[0], 1),
               round(pose.regard[1], 1), round(pose.sourcils, 1))
        if cle in self._cache:
            return self._cache[cle]
        e = self.echelle
        toile = Image.new("RGBA", (taille, taille), (0, 0, 0, 0))
        for nom, (pos, images) in self.parties.items():
            if nom == "bouche":
                im = images.get(pose.bouche) or images["X"]
            elif nom in ("yeux", "blanc_yeux"):
                im = images.get("fermes" if pose.cligne else "ouverts")
            elif nom == "pupilles" and pose.cligne:
                continue
            else:
                im = images["defaut"]
            if im is None or im.getbbox() is None:
                continue
            im = im.resize((max(1, round(im.width * e)), max(1, round(im.height * e))), Image.LANCZOS)
            x = (pos[0] - self.cx) * e + taille / 2
            y = (pos[1] - self.cy) * e + taille / 2
            if nom == "pupilles":
                x += pose.regard[0] * 7 * e
                y += pose.regard[1] * 4 * e
            if nom == "sourcils":
                y -= pose.sourcils * 15 * e
            toile.alpha_composite(im, (round(x - im.width / 2), round(y - im.height / 2)))
        a = np.asarray(toile).astype(np.float32) / 255
        if len(self._cache) > 300:
            self._cache.clear()
        self._cache[cle] = a
        return a


def _echantillonner(tex: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.empty((*x.shape, 4), np.float32)
    for c in range(4):
        out[..., c] = ndimage.map_coordinates(tex[..., c], [y, x], order=1, mode="constant", cval=0.0)
    return out


class _Tampon:
    """Profondeur, couleur et matière de chaque pixel ; calculs par boîtes."""

    def __init__(self, w: int, h: int, echelle: float, ox: float, oy: float):
        self.w, self.h, self.e, self.ox, self.oy = w, h, echelle, ox, oy
        self.z = np.full((h, w), -np.inf, np.float32)
        self.c = np.zeros((h, w, 3), np.float32)
        self.id = np.zeros((h, w), np.int16)
        self.n = 1

    def boite(self, x0, y0, x1, y1, roulis=None):
        """Pixels couvrant la boîte (repère du modèle) : tranche et grilles U, V."""
        e = self.e
        i0, i1 = max(0, int((y0 + self.oy) * e) - 2), min(self.h, int((y1 + self.oy) * e) + 3)
        j0, j1 = max(0, int((x0 + self.ox) * e) - 2), min(self.w, int((x1 + self.ox) * e) + 3)
        if i0 >= i1 or j0 >= j1:
            return None
        v = (np.arange(i0, i1, dtype=np.float32) + 0.5) / e - self.oy
        u = (np.arange(j0, j1, dtype=np.float32) + 0.5) / e - self.ox
        U, V = np.meshgrid(u, v)
        if roulis is not None:  # roulis de la tête : on tourne la grille autour du cou
            a, py = roulis
            ca, sa = math.cos(a), math.sin(a)
            U, V = ca * U + sa * (V - py), -sa * U + ca * (V - py) + py
        return (slice(i0, i1), slice(j0, j1)), U, V

    def poser(self, sl, z, rgb, masque, meme_id: int | None = None) -> np.ndarray:
        zz = self.z[sl]
        devant = masque & (z > zz)
        zz[devant] = z[devant]
        self.c[sl][devant] = rgb[devant]
        if meme_id is None:
            meme_id = self.n
            self.n += 1
        self.id[sl][devant] = meme_id
        return devant


def _cel(base, normale, ombre=0.74):
    """Deux tons : clair face à la lumière, foncé sinon ; un liseré au bord éclairé."""
    n = normale / (np.linalg.norm(normale, axis=-1, keepdims=True) + 1e-6)
    l = (n * LUMIERE).sum(-1)
    k = np.where(l > 0.05, 1.0, ombre).astype(np.float32)
    k += np.where((n[..., 2] < 0.3) & (l > 0.05), 0.07, 0.0).astype(np.float32)
    return np.clip(base * k[..., None], 0, 1)


def _ellipsoide(U, V, centre, rayons, R):
    """Rayon orthographique (vers -z) contre un ellipsoïde tourné par R :
    (z de l'impact ou -inf, point d'impact local, normale dans la vue, masque)."""
    r = np.asarray(rayons, np.float32)
    o = np.stack([U - centre[0], V - centre[1], np.full_like(U, 5000.0 - centre[2])], -1) @ R.astype(np.float32)
    d = (R.T @ np.array([0, 0, -1.0])).astype(np.float32)
    oo, dd = o / r, d / r
    a = float((dd * dd).sum())
    b = 2 * (oo * dd).sum(-1)
    c = (oo * oo).sum(-1) - 1
    disc = b * b - 4 * a * c
    ok = disc >= 0
    t = (-b - np.sqrt(np.where(ok, disc, 0))) / (2 * a)
    loc = o + t[..., None] * d
    z = np.where(ok, 5000.0 - t, -np.inf).astype(np.float32)
    n = (loc / r ** 2) @ R.T.astype(np.float32)
    return z, loc, n, ok


class Toon:
    """Rendu d'un personnage en volume, dans une pose, à une échelle donnée."""

    def __init__(self, modele: Modele):
        self.m = modele

    @property
    def taille(self) -> tuple[int, int]:
        return TAILLE

    def rendre(self, pose: PoseToon, echelle: float = 1.0) -> Image.Image:
        m = self.m
        w, h = max(2, round(TAILLE[0] * echelle)), max(2, round(TAILLE[1] * echelle))
        ox, oy = TAILLE[0] / 2, TAILLE[1] - SOL
        T = _Tampon(w, h, echelle, ox, oy)
        a_corps = math.radians(pose.corps)
        Rc = _rot(a_corps)
        respire = 3.0 * math.sin(2 * math.pi * pose.respiration)
        y0, y1 = m.epaules + respire, m.ourlet

        # --- le boubou ---
        b = T.boite(-m.largeur_ourlet - 5, y0 - 5, m.largeur_ourlet + 5, y1 + 5)
        if b:
            sl, U, V = b
            tv = np.clip((V - y0) / (y1 - y0), 0, 1)
            rx = m.largeur_epaules + (m.largeur_ourlet - m.largeur_epaules) * tv ** 0.8
            dh = np.clip((V - y0) / 55, 0, 1)
            rx = rx * (0.55 + 0.45 * np.sqrt(1 - (1 - dh) ** 2))  # épaules arrondies
            rx = np.maximum(rx, 1.0)
            rz = rx * 0.55
            ca, sa = math.cos(a_corps), math.sin(a_corps)
            R = np.sqrt((rx * ca) ** 2 + (rz * sa) ** 2)
            tau = np.arctan2(rz * sa, rx * ca)
            dans = (np.abs(U) < R) & (V >= y0) & (V <= y1)
            ac = np.arccos(np.clip(U / R, -1, 1))
            t1, t2 = tau + ac, tau - ac
            z1 = -rx * np.cos(t1) * sa + rz * np.sin(t1) * ca
            z2 = -rx * np.cos(t2) * sa + rz * np.sin(t2) * ca
            t = np.where(z1 > z2, t1, t2)
            zb = np.maximum(z1, z2)
            lx, lz = rx * np.cos(t), rz * np.sin(t)
            n = np.stack([np.cos(t) * rz / rx, np.zeros_like(t), np.sin(t)], -1) @ Rc.T.astype(np.float32)
            base = np.broadcast_to(m.habit, (*U.shape, 3)).copy()
            plis = (np.sin(t * 9 + 0.6) > 0.94) & (tv > 0.3)
            base[plis] *= 0.84
            if m.broderie is not None:
                bro = np.asarray(m.broderie).astype(np.float32) / 255
                e = _echantillonner(bro, lx + bro.shape[1] / 2, (V - y0) + 8)
                a = e[..., 3:4] * (lz > 0.3 * rz)[..., None]
                base = base * (1 - a) + e[..., :3] * a
            T.poser(sl, np.where(dans, zb, -np.inf), _cel(base, n), dans)

        # --- bras : manche large jusqu'au coude, avant-bras, main ---
        for cote, geste in ((-1, pose.bras_g), (1, pose.bras_d)):
            ep = np.array([cote * (m.largeur_epaules - 18), y0 + 55, 0.0])
            du = np.array([cote * 0.22, 1.0, 0.05]) * (1 - geste) + np.array([cote * 0.3, 0.8, 0.55]) * geste
            du /= np.linalg.norm(du)
            df = np.array([cote * 0.12, 1.0, 0.08]) * (1 - geste) + np.array([-cote * 0.25, -0.35, 1.0]) * geste
            df /= np.linalg.norm(df)
            coude = ep + du * 255
            poignet = coude + df * 200
            main = poignet + df * 30
            p = [x @ Rc.T for x in (ep, coude, poignet, main)]
            self._capsule(T, p[0], p[1], 62, 50, m.habit)
            self._capsule(T, p[1] - (p[1] - p[0]) * 0.05, p[2], 38, 33, m.peau)
            self._sphere(T, p[3], 42, m.peau)

        # --- pieds ---
        for cote in (-1, 1):
            c = np.array([cote * 68, -20.0, 30.0]) @ Rc.T
            b = T.boite(c[0] - 80, c[1] - 30, c[0] + 80, c[1] + 30)
            if b:
                sl, U, V = b
                z, loc, n, ok = _ellipsoide(U, V, c, (52, 22, 72), Rc)
                T.poser(sl, z, _cel(np.broadcast_to(m.pieds, (*U.shape, 3)), n), ok)

        # --- cou ---
        c = np.array([0, y0 - 45, 0.0]) @ Rc.T
        b = T.boite(c[0] - 60, c[1] - 90, c[0] + 60, c[1] + 90)
        if b:
            sl, U, V = b
            z, loc, n, ok = _ellipsoide(U, V, c, (50, 85, 50), Rc)
            T.poser(sl, z, _cel(np.broadcast_to(m.peau, (*U.shape, 3)), n, 0.8), ok)

        # --- tête ---
        self._tete(T, pose, a_corps, respire)

        # --- contours : silhouette et changements de matière ---
        alpha = np.isfinite(T.z)
        ep = max(2, round(4.5 * echelle))
        bord = ndimage.binary_dilation(alpha, iterations=ep) & ~alpha
        k = max(1, round(1.6 * echelle))
        trait = np.zeros_like(alpha)
        for dy, dx in ((0, k), (k, 0), (k, k), (k, -k)):
            trait |= (np.roll(np.roll(T.id, dy, 0), dx, 1) != T.id)
        trait &= alpha
        T.c[trait] = CONTOUR
        rgba = np.zeros((h, w, 4), np.float32)
        rgba[..., :3] = T.c
        rgba[..., 3] = alpha
        rgba[bord, :3] = CONTOUR
        rgba[bord, 3] = 1
        rgba[..., 3] = ndimage.gaussian_filter(rgba[..., 3], 0.6)
        image = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA")
        if abs(pose.balance) > 0.05:
            image = image.rotate(pose.balance, resample=Image.BICUBIC,
                                 center=(ox * echelle, oy * echelle))
        return image

    def _sphere(self, T, c, r, rgb):
        b = T.boite(c[0] - r, c[1] - r, c[0] + r, c[1] + r)
        if not b:
            return
        sl, U, V = b
        d2 = r * r - (U - c[0]) ** 2 - (V - c[1]) ** 2
        ok = d2 > 0
        z = np.where(ok, c[2] + np.sqrt(np.clip(d2, 0, None)), -np.inf)
        n = np.stack([U - c[0], V - c[1], z - c[2]], -1)
        n[~ok] = (0, 0, 1)
        T.poser(sl, z, _cel(np.broadcast_to(rgb, (*U.shape, 3)), n, 0.8), ok)

    def _capsule(self, T, p, q, rp, rq, rgb):
        """Cône arrondi de p (rayon rp) à q (rayon rq), vu de face."""
        r = max(rp, rq)
        b = T.boite(min(p[0], q[0]) - r, min(p[1], q[1]) - r, max(p[0], q[0]) + r, max(p[1], q[1]) + r)
        if not b:
            return
        sl, U, V = b
        d = q - p
        d2 = d[0] ** 2 + d[1] ** 2 + 1e-6
        t = np.clip(((U - p[0]) * d[0] + (V - p[1]) * d[1]) / d2, 0, 1)
        cx, cy, cz = p[0] + t * d[0], p[1] + t * d[1], p[2] + t * d[2]
        rr = rp + (rq - rp) * t
        e2 = rr * rr - (U - cx) ** 2 - (V - cy) ** 2
        ok = e2 > 0
        z = np.where(ok, cz + np.sqrt(np.clip(e2, 0, None)), -np.inf)
        n = np.stack([U - cx, V - cy, z - cz], -1)
        n[~ok] = (0, 0, 1)
        T.poser(sl, z, _cel(np.broadcast_to(rgb, (*U.shape, 3)), n, 0.78), ok)

    def _tete(self, T, pose, a_corps, respire):
        m = self.m
        rx, ry, rz = m.tete_r
        Rt = _rot(a_corps + math.radians(pose.tete), math.radians(-pose.tangage))
        tc = np.array(m.tete_c) + np.array([0, respire * 0.6, 0])
        pivot_y = m.epaules - 30
        b = T.boite(tc[0] - rx * 1.6, tc[1] - ry * 1.55, tc[0] + rx * 1.6, tc[1] + ry * 1.45,
                    roulis=(math.radians(pose.roulis), pivot_y))
        if not b:
            return
        sl, U, V = b
        sh = U.shape

        def surface(c_loc, rayons):
            c = tc + Rt @ np.asarray(c_loc, float)
            z, loc, n, ok = _ellipsoide(U, V, c, rayons, Rt)
            return z, loc + np.asarray(c_loc, np.float32), n, ok

        peau = np.broadcast_to(m.peau, (*sh, 3))
        # oreilles
        for cote in (-1, 1):
            z, loc, n, ok = surface((cote * rx * 0.96, 10, -8), (24, 40, 20))
            T.poser(sl, z, _cel(peau, n, 0.8), ok)
        # nez : ne se voit qu'en dépassant du profil (de face, c'est celui du dessin)
        zn, _, nn, okn = surface((0, ry * 0.10, rz * 0.90), (19, 26, 32))
        # crâne et menton : une seule matière
        z1, l1, n1, ok1 = surface((0, 0, 0), (rx, ry, rz))
        z2, l2, n2, ok2 = surface((0, ry * 0.40, rz * 0.10), (rx * 0.74, ry * 0.64, rz * 0.82))
        prend2 = ok2 & (z2 > z1)
        z = np.where(prend2, z2, z1)
        loc = np.where(prend2[..., None], l2, l1)
        n = np.where(prend2[..., None], n2, n1)
        ok = ok1 | ok2
        base = peau.copy()
        if m.barbe is not None:  # le long de la mâchoire, la bouche dégagée
            bouche = ((loc[..., 0] / (rx * 0.34)) ** 2 + ((loc[..., 1] - ry * 0.47) / (ry * 0.16)) ** 2) < 1
            # une bande en U le long de la mâchoire, au bord festonné comme sur le dessin
            bx, by = loc[..., 0] / (rx * 0.80), (loc[..., 1] - ry * 0.05) / (ry * 0.62)
            feston = 0.06 * np.sin(np.arctan2(by, bx) * 22)
            machoire = (bx ** 2 + by ** 2 > 1.0 + feston) & (loc[..., 1] > ry * 0.22)
            # vers l'arrière, la barbe remonte en favori jusque sous l'oreille
            arriere = loc[..., 2] > rz * (0.25 - 0.9 * np.clip(loc[..., 1] / ry - 0.22, 0, 1))
            barbe = machoire & arriere & ~bouche
            base[barbe] = m.barbe
        if m.visage is not None:
            taille = int(2 * rx * 1.25)
            tex = m.visage.texture(pose, taille)
            e = _echantillonner(tex, loc[..., 0] + taille / 2, loc[..., 1] + taille / 2)
            face = np.clip((loc[..., 2] / rz - 0.22) / 0.2, 0, 1)
            a = e[..., 3:4] * face[..., None]
            base = base * (1 - a) + e[..., :3] * a
        id_tete = T.n
        T.poser(sl, z, _cel(base, n, 0.82), ok)
        T.poser(sl, np.where(okn, zn - 2000, -np.inf), _cel(peau, nn, 0.82), okn, meme_id=id_tete)
        # coiffe : une coque un peu plus grande que le crâne, sur le dessus et l'arrière
        if m.coiffe_type != "aucun":
            if m.coiffe_type == "foulard":
                rc, axe, seuil = (rx * 1.07, ry * 1.06, rz * 1.08), np.array([0, -0.9, -0.44]), 0.10
            else:  # bonnet rond, haut et droit
                rc, axe, seuil = (rx * 1.05, ry * 1.12, rz * 1.05), np.array([0, -1.0, -0.10]), 0.30
            zc, lc, nc, okc = surface((0, -4 if m.coiffe_type == "foulard" else -22, -4), rc)
            pn = lc / np.asarray(rc, np.float32)
            dans = okc & ((pn * axe).sum(-1) > seuil)
            th = np.arctan2(pn[..., 0], pn[..., 2])
            ph = np.arccos(np.clip(-pn[..., 1], -1, 1))
            if m.coiffe_type == "foulard":
                motif = (np.sin(th * 10) * np.sin(ph * 16) > 0.72)
            else:  # une frise de fleurs grises
                motif = (np.sin(th * 12) * np.sin(ph * 18) > 0.8) & (pn[..., 1] > -0.8)
            cc = np.where(motif[..., None], m.motif_coiffe, m.coiffe)
            T.poser(sl, np.where(dans, zc, -np.inf), _cel(cc, nc, 0.78), dans)
        if m.coiffe_type == "foulard":  # le nœud, derrière en haut
            zk, _, nk, okk = surface((rx * 0.22, -ry * 0.80, -rz * 0.62), (64, 54, 60))
            T.poser(sl, zk, _cel(np.broadcast_to(m.coiffe * 0.92, (*sh, 3)), nk, 0.78), okk)


# --- personnages construits d'après leurs dessins ---

def _broderie(image: Path, boite, fond, echelle: float) -> Image.Image:
    """Le motif de l'encolure du dessin : ce qui n'est pas la couleur de l'habit."""
    a = np.asarray(Image.open(image).convert("RGB")).astype(np.float32)
    x0, y0, x1, y1 = boite
    c = a[y0:y1, x0:x1]
    d = np.sqrt(((c - np.asarray(fond, np.float32)) ** 2).sum(-1))
    alpha = np.clip((d - 40) / 40, 0, 1)
    im = Image.fromarray(np.dstack([c, alpha * 255]).astype(np.uint8), "RGBA")
    return im.resize((round(im.width * echelle), round(im.height * echelle)), Image.LANCZOS)


def modele(nom: str, racine: Path | None = None) -> Modele:
    """Fanta et Bakary, d'après les dessins de ChatGPT (animation/brut)."""
    racine = racine or Path(__file__).parent
    brut, perso = racine / "brut" / nom, racine / "personnages" / nom
    if nom == "fanta":
        m = Modele(peau=_rgb((146, 70, 30)), habit=_rgb((0, 102, 88)), coiffe=_rgb((212, 16, 24)),
                   motif_coiffe=_rgb((250, 212, 30)), coiffe_type="foulard")
        m.broderie = _broderie(brut / "personnage.webp", (280, 640, 790, 1050), (0, 102, 88), 0.47)
    elif nom == "bakary":
        m = Modele(peau=_rgb((134, 64, 28)), habit=_rgb((18, 44, 118)), coiffe=_rgb((246, 246, 244)),
                   motif_coiffe=_rgb((172, 172, 174)), coiffe_type="bonnet", barbe=_rgb((150, 148, 146)),
                   largeur_epaules=180, largeur_ourlet=240)
        m.broderie = _broderie(brut / "personnage.webp", (330, 630, 730, 1030), (18, 44, 118), 0.47)
    else:
        raise KeyError(nom)
    if (perso / "personnage.json").exists():
        m.visage = Visage(perso, m.tete_r[0])
    return m
