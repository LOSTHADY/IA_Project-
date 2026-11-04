"""Découper un personnage dessiné d'une pièce (ChatGPT, etc.) en calques.

Un générateur d'images donne un personnage entier, de face, sur fond blanc.
Le moteur a besoin de calques : corps, oreilles, tête sans traits, sourcils,
yeux, pupilles, nez, bouches. Ce module les tire de l'image grâce à quelques
repères, notés une fois pour toutes dans ``reperes.json`` :

    {
      "image": "personnage.webp",            # le personnage, yeux ouverts, bouche fermée
      "yeux_fermes": "yeux_fermes.webp",     # facultatif : la même image, yeux fermés
      "bouches": "../bouches.webp",          # planche 3 × 3 des bouches (ordre du prompt 2)
      "peau": [430, 450],                    # un point de peau du visage (une joue)
      "tete": [[530, 125]],                  # un point dans chaque autre partie de la tête
                                             # (foulard, bonnet, barbe)
      "oreilles": [[310, 340], [745, 350]],  # un point de peau dans chaque oreille
      "coupe": 525,                          # y où commence le cou : au-dessus, c'est la tête
      "cou": [420, 640, 430],                # x gauche, x droit, y : peau à prolonger sous la tête
      "sourcils": [x0, y0, x1, y1],          # rectangles qui contiennent chaque trait,
      "yeux": [x0, y0, x1, y1],              # sans déborder sur le contour du visage
      "iris": [[450, 332, 30], [612, 332, 30]],  # centre et rayon de chaque iris
      "nez": [x0, y0, x1, y1],
      "bouche": [x0, y0, x1, y1],
      "pivot_cou": [530, 560],
      "echelle": 0.65                        # taille finale, par rapport à l'image
    }

Le style plat (aplats, aucune ombre) fait tout le travail : un trait, c'est
ce qui n'a pas la couleur de la peau. Sous chaque trait découpé, la peau est
repeinte. Sinon, quand les yeux glissent, on verrait leur ancienne place.

    python -m animation decouper animation/brut/fanta animation/personnages/fanta
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

CONTOUR = (50, 22, 10)
BLANC_YEUX = (250, 248, 244)
FORMES_PLANCHE = "XABCDEFGH"  # ordre des bouches dans la planche du prompt 2


def _rgb(chemin: Path) -> np.ndarray:
    return np.asarray(Image.open(chemin).convert("RGB")).astype(np.float32)


def _distance(image: np.ndarray, couleur) -> np.ndarray:
    return np.sqrt(((image - np.asarray(couleur, np.float32)) ** 2).sum(axis=2))


def _composante(masque: np.ndarray, x: int, y: int) -> np.ndarray:
    etiquettes, _ = ndimage.label(masque)
    e = etiquettes[y, x]
    if e == 0:
        raise ValueError(f"le point ({x}, {y}) ne tombe pas dans la zone attendue")
    return etiquettes == e


def _remplir(image: np.ndarray, x: int, y: int, tolerance: float = 45) -> np.ndarray:
    """La zone de couleur proche de celle du point (x, y), trous bouchés."""
    zone = _composante(_distance(image, image[y, x]) < tolerance, x, y)
    return ndimage.binary_fill_holes(zone)


def _fond(image: np.ndarray, garder: np.ndarray, coupe: int) -> np.ndarray:
    """Le fond blanc : le blanc qui touche les bords de l'image, et les petites
    poches de blanc enfermées entre deux contours de la tête (entre foulard et
    oreille), sauf dans ``garder`` (le blanc des yeux). Sous la coupe du cou,
    le blanc est celui des vêtements : on n'y touche pas."""
    blanc = image.min(axis=2) > 225
    etiquettes, n = ndimage.label(blanc)
    bords = np.unique(np.concatenate([etiquettes[0], etiquettes[-1], etiquettes[:, 0],
                                      etiquettes[:, -1]]))
    fond = np.isin(etiquettes, bords[bords > 0])
    tailles = ndimage.sum(blanc, etiquettes, range(1, n + 1))
    poches = np.isin(etiquettes, 1 + np.nonzero(tailles < 2500)[0]) & ~garder
    poches[coupe:] = False
    return fond | poches


def _liseres(image: np.ndarray, fond: np.ndarray) -> np.ndarray:
    """Alpha des bords : les pixels de transition avec le fond blanc deviennent
    du contour semi-transparent. Sinon, un liseré blanc apparaît dès qu'un
    calque glisse sur un autre."""
    bord = ndimage.binary_dilation(fond, iterations=3) & ~fond
    lum = image.mean(axis=2)
    alpha = np.where(bord, np.clip((250 - lum) / 170, 0, 1), 1.0) * ~fond
    image[bord] = CONTOUR
    return alpha


def _rect(forme: tuple[int, int], boite) -> np.ndarray:
    m = np.zeros(forme, bool)
    x0, y0, x1, y1 = boite
    m[y0:y1, x0:x1] = True
    return m


def _disque(forme: tuple[int, int], cx, cy, r) -> np.ndarray:
    yy, xx = np.ogrid[: forme[0], : forme[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _calque(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Image RGBA (uint8) à partir des couleurs et d'un alpha entre 0 et 1."""
    rgba = np.zeros((*image.shape[:2], 4), np.uint8)
    rgba[..., :3] = np.clip(image, 0, 255).astype(np.uint8)
    rgba[..., 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    return rgba


def _trait(image: np.ndarray, peau, zone: np.ndarray) -> np.ndarray:
    """Alpha d'un trait : ce qui, dans la zone, n'a pas la couleur de la peau."""
    return np.clip((_distance(image, peau) - 25) / 35, 0, 1) * zone


def _effacer(image: np.ndarray, peau, zone: np.ndarray) -> None:
    """Repeint la peau à la place des traits de la zone (en place)."""
    a = np.clip((_distance(image, peau) - 12) / 20, 0, 1) * zone
    image[:] = image * (1 - a[..., None]) + np.asarray(peau, np.float32) * a[..., None]


def _bouches(planche: Path, largeur_ref: float, peau_ref) -> dict[str, tuple[np.ndarray, int]]:
    """Les 9 bouches de la planche, détourées : forme -> (RGBA, haut des lèvres)."""
    p = _rgb(planche)
    h, w = p.shape[:2]
    bouches = {}
    for k, forme in enumerate(FORMES_PLANCHE):
        r, c = divmod(k, 3)
        case = p[r * h // 3:(r + 1) * h // 3, c * w // 3:(c + 1) * w // 3]
        pastille = case.min(axis=2) < 225  # la pastille de peau autour de la bouche
        etiquettes, n = ndimage.label(pastille)
        tailles = ndimage.sum(pastille, etiquettes, range(1, n + 1))
        pastille = etiquettes == (1 + int(np.argmax(tailles)))
        peau = np.median(case[pastille & (_distance(case, peau_ref) < 60)], axis=0)
        alpha = _trait(case, peau, ndimage.binary_erosion(pastille, iterations=3))
        alpha = np.maximum(alpha, ndimage.binary_fill_holes(alpha > 0.5))
        ys, xs = np.nonzero(alpha > 0.5)
        bouches[forme] = (_calque(case, alpha)[ys.min():ys.max() + 1, xs.min():xs.max() + 1],
                          xs.max() - xs.min() + 1)
    echelle = largeur_ref / bouches["X"][1]
    sortie = {}
    for forme, (rgba, _) in bouches.items():
        im = Image.fromarray(rgba, "RGBA")
        sortie[forme] = im.resize((max(1, round(im.width * echelle)),
                                   max(1, round(im.height * echelle))), Image.LANCZOS)
    return sortie


def decouper(source: str | Path, destination: str | Path) -> Path:
    source, destination = Path(source), Path(destination)
    r = json.loads((source / "reperes.json").read_text(encoding="utf-8"))
    image = _rgb(source / r["image"])
    forme = image.shape[:2]
    fond = _fond(image, _rect(forme, r["yeux"]), r["coupe"])
    opaque = ~fond
    couverture = _liseres(image, fond)  # avant toute découpe : vaut pour tous les calques
    px, py = r["peau"]
    peau = np.median(image[py - 3:py + 4, px - 3:px + 4].reshape(-1, 3), axis=0)

    # --- oreilles : leur peau et leur contour, pas celui du visage ---
    visage_peau = _remplir(image, px, py)
    pres_du_visage = ndimage.binary_dilation(visage_peau, iterations=10)
    oreilles = np.zeros(forme, bool)
    peaux_oreilles = []
    for x, y in r["oreilles"]:
        o = _remplir(image, x, y)
        peaux_oreilles.append((x, o))
        oreilles |= ndimage.binary_dilation(o, iterations=10) & opaque & ~pres_du_visage

    # --- la tête : tout ce qui est au-dessus de la coupe du cou, sauf les
    # oreilles ; en dessous, le menton (et la barbe) seulement ---
    coupe = r["coupe"]
    tete = opaque & _rect(forme, (0, 0, forme[1], coupe)) & ~oreilles
    menton = visage_peau.copy()
    for x, y in r.get("tete", []):
        menton |= _remplir(image, x, y)
    menton = ndimage.binary_dilation(ndimage.binary_fill_holes(menton), iterations=9) & opaque
    tete |= menton & ~_rect(forme, (0, 0, forme[1], coupe))
    tete = ndimage.binary_fill_holes(tete) & opaque & ~oreilles
    # le corps ne doit rien garder de la tête : pas de morceau de contour isolé
    reste = opaque & ~tete & ~oreilles & _rect(forme, (0, 0, forme[1], coupe))
    tete |= reste

    # --- oreilles prolongées sous la tête : en glissant, elles ne s'en détachent pas ---
    rallonge = np.zeros(forme, bool)
    centre = r["pivot_cou"][0]
    for x, o in peaux_oreilles:
        sens = 1 if x < centre else -1
        for d in range(2, 40, 2):
            rallonge |= np.roll(o, sens * d, axis=1)
    rallonge &= tete & ~oreilles

    # --- les traits du visage, puis le visage sans traits ---
    traits = {}
    visage = image.copy()
    for nom in ("sourcils", "nez", "bouche"):
        zone = _rect(forme, r[nom])
        alpha = _trait(image, peau, zone)
        if nom == "bouche":  # lèvres proches de la peau : tout l'intérieur du contour
            alpha = np.maximum(alpha, ndimage.binary_fill_holes(alpha > 0.5))
            visage[zone] = peau  # et toute la zone repeinte : aucune ombre des lèvres
        traits[nom] = _calque(image, alpha)
        _effacer(visage, peau, zone)
    zone_yeux = _rect(forme, r["yeux"])
    iris = np.zeros(forme, bool)
    for cx, cy, rayon in r["iris"]:
        iris |= _disque(forme, cx, cy, rayon)
    pas_peau = _trait(image, peau, zone_yeux)
    lum = image.mean(axis=2)
    blanc = (lum > 185) & zone_yeux
    traits["pupilles"] = _calque(image, pas_peau * iris)
    contour = pas_peau * ~iris * ~ndimage.binary_dilation(blanc, iterations=1)
    traits["yeux_ouverts"] = _calque(image, contour)
    blancs = ndimage.binary_fill_holes(ndimage.binary_dilation(blanc | (iris & (pas_peau > 0.5)),
                                                               iterations=2)) & zone_yeux
    traits["blanc_yeux"] = _calque(np.broadcast_to(np.float32(BLANC_YEUX), image.shape), blancs)
    _effacer(visage, peau, zone_yeux)
    if r.get("yeux_fermes"):
        fermes = _rgb(source / r["yeux_fermes"])
        traits["yeux_fermes"] = _calque(fermes, _trait(fermes, peau, zone_yeux))
    else:  # paupières dessinées : un trait courbe à la place de chaque œil
        dessin = Image.new("RGBA", (forme[1], forme[0]), (0, 0, 0, 0))
        d = ImageDraw.Draw(dessin)
        etiquettes, n = ndimage.label(ndimage.binary_dilation(contour > 0.5, iterations=3))
        for tranche in ndimage.find_objects(etiquettes):
            ys, xs = tranche
            if xs.stop - xs.start < 40:
                continue
            haut = ys.stop - ys.start
            d.arc([xs.start + 4, ys.start + haut * 0.05, xs.stop - 4, ys.start + haut * 0.75],
                  15, 165, fill=CONTOUR + (255,), width=6)
        traits["yeux_fermes"] = np.asarray(dessin)

    # --- le corps : tout le reste, et la peau du cou prolongée sous la tête ---
    x0, x1, y0 = r["cou"]
    sous_tete = tete & _rect(forme, (x0, y0, x1, forme[0]))
    corps = image.copy()
    corps[sous_tete] = peau
    calques = {
        "corps": _calque(corps, np.maximum(couverture * (~tete & ~oreilles), sous_tete)),
        "oreilles": _calque(np.where(rallonge[..., None], peau, image),
                            np.maximum(couverture * oreilles, rallonge)),
        "tete": _calque(visage, couverture * tete),
        **traits,
    }

    # --- bouches : celle du dessin au repos, les autres tirées de la planche ---
    bx0, by0, bx1, by1 = r["bouche"]
    a = traits["bouche"][..., 3] > 128
    ys, xs = np.nonzero(a)
    largeur, haut_levres, milieu = xs.max() - xs.min() + 1, ys.min(), (xs.min() + xs.max()) / 2
    bouches = {"X": Image.fromarray(traits.pop("bouche"), "RGBA")}
    planche = _bouches((source / r["bouches"]), largeur, peau)
    for f, im in planche.items():
        if f == "X":
            continue
        toile = Image.new("RGBA", (forme[1], forme[0]), (0, 0, 0, 0))
        toile.alpha_composite(im, (round(milieu - im.width / 2), int(haut_levres) - 2))
        bouches[f] = toile

    # --- écriture : chaque calque rogné, à l'échelle, avec sa position ---
    echelle = r.get("echelle", 1.0)
    destination.mkdir(parents=True, exist_ok=True)
    for ancien in destination.glob("*.png"):
        ancien.unlink()

    def ecrire(nom: str, im: Image.Image | np.ndarray) -> tuple[str, list[float]]:
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im, "RGBA")
        boite = im.getchannel("A").getbbox() or (0, 0, 2, 2)
        im = im.crop(boite)
        im = im.resize((max(2, round(im.width * echelle)), max(2, round(im.height * echelle))),
                       Image.LANCZOS)
        im.save(destination / f"{nom}.png")
        return f"{nom}.png", [(boite[0] + boite[2]) / 2 * echelle, (boite[1] + boite[3]) / 2 * echelle]

    def partie(nom, images, **reglages):
        fichiers, position = {}, None
        for variante, im in images.items():
            fichier, pos = ecrire(nom if variante == "defaut" else f"{nom}_{variante}", im)
            fichiers[variante] = fichier
            position = position or pos
        if len(fichiers) > 1:  # variantes de tailles différentes : toutes centrées pareil
            position = _centre_commun(destination, fichiers, echelle, images)
        return {"nom": nom, "images": fichiers, "position": position, **reglages}

    vide = np.zeros((*forme, 4), np.uint8)
    parties = [
        partie("corps", {"defaut": calques["corps"]}),
        partie("oreilles", {"defaut": calques["oreilles"]}, profondeur=-0.15),
        partie("tete", {"defaut": calques["tete"]}),
        partie("blanc_yeux", {"ouverts": calques["blanc_yeux"], "fermes": vide}, groupe="tete",
               role="yeux", profondeur=0.6, compression=0.25),
        partie("pupilles", {"defaut": calques["pupilles"]}, profondeur=0.62, compression=0.27,
               course=6 * echelle / 0.65),
        partie("yeux", {"ouverts": calques["yeux_ouverts"], "fermes": calques["yeux_fermes"]},
               profondeur=0.6, compression=0.25),
        partie("sourcils", {"defaut": calques["sourcils"]}, profondeur=0.62, compression=0.25),
        partie("nez", {"defaut": calques["nez"]}, profondeur=0.9, compression=0.2),
        partie("bouche", bouches, profondeur=0.75, compression=0.27),
    ]
    description = {"nom": destination.name, "taille": [round(forme[1] * echelle), round(forme[0] * echelle)],
                   "pivot_cou": [v * echelle for v in r["pivot_cou"]],
                   "amplitude_lacet": 70 * echelle, "amplitude_tangage": 36 * echelle,
                   "parties": parties}
    (destination / "personnage.json").write_text(
        json.dumps(description, indent=1, ensure_ascii=False), encoding="utf-8")
    return destination


def _centre_commun(destination, fichiers, echelle, images) -> list[float]:
    """Variantes d'une même partie (yeux ouverts / fermés, bouches) : on les
    recadre toutes sur la même boîte, l'union des leurs, pour qu'elles restent
    exactement à leur place quand on passe de l'une à l'autre."""
    ims = {v: (Image.fromarray(im, "RGBA") if isinstance(im, np.ndarray) else im)
           for v, im in images.items()}
    boites = [im.getchannel("A").getbbox() for im in ims.values()]
    boites = [b for b in boites if b]
    boite = (min(b[0] for b in boites), min(b[1] for b in boites),
             max(b[2] for b in boites), max(b[3] for b in boites))
    for v, im in ims.items():
        im = im.crop(boite)
        im = im.resize((max(2, round(im.width * echelle)), max(2, round(im.height * echelle))),
                       Image.LANCZOS)
        im.save(destination / fichiers[v])
    return [(boite[0] + boite[2]) / 2 * echelle, (boite[1] + boite[3]) / 2 * echelle]
