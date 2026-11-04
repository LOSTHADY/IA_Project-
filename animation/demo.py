"""Deux personnages de démonstration, dessinés par le programme.

Ils servent à essayer le moteur en attendant les vrais dessins : chaque
partie est une image PNG à part, exactement comme le seront celles des
illustrateurs (voir docs/ANIMATION.md). Pour obtenir les dossiers
``animation/personnages/awa`` et ``animation/personnages/seydou`` ::

    python -m animation demo

Le dessin se fait au double de la taille, puis il est réduit : c'est un
anticrénelage simple, et les contours sont nets.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

S = 2  # suréchantillonnage
TAILLE = (800, 1100)
TETE = (400, 400)  # centre de la tête
CONTOUR = (55, 32, 25)

STYLES = {
    "awa": {"peau": (139, 90, 60), "habit": (34, 120, 110), "motif": (245, 190, 60),
            "coiffe": "foulard", "coiffe_couleur": (214, 72, 52), "coiffe_motif": (250, 210, 90)},
    "seydou": {"peau": (110, 70, 48), "habit": (60, 80, 160), "motif": (240, 240, 230),
               "coiffe": "bonnet", "coiffe_couleur": (240, 236, 222), "coiffe_motif": (190, 150, 70)},
}


class Toile:
    """Une partie : image transparente de taille (w, h), centrée en ``centre``."""

    def __init__(self, w: int, h: int, centre: tuple[int, int]):
        self.w, self.h, self.centre = w, h, centre
        self.image = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.image)

    def p(self, x: float, y: float) -> tuple[float, float]:
        """Coordonnées du personnage -> coordonnées de la toile suréchantillonnée."""
        return ((x - self.centre[0] + self.w / 2) * S, (y - self.centre[1] + self.h / 2) * S)

    def boite(self, x0, y0, x1, y1):
        return [*self.p(x0, y0), *self.p(x1, y1)]

    def ellipse(self, cx, cy, rx, ry, fill, contour=CONTOUR, ep=5):
        self.d.ellipse(self.boite(cx - rx, cy - ry, cx + rx, cy + ry), fill=fill,
                       outline=contour, width=ep * S if contour else 0)

    def poly(self, points, fill, contour=CONTOUR, ep=5):
        self.d.polygon([self.p(x, y) for x, y in points], fill=fill,
                       outline=contour, width=ep * S if contour else 0)

    def ligne(self, points, couleur=CONTOUR, ep=6):
        self.d.line([self.p(x, y) for x, y in points], fill=couleur, width=ep * S, joint="curve")

    def arc(self, cx, cy, rx, ry, debut, fin, couleur=CONTOUR, ep=6):
        self.d.arc(self.boite(cx - rx, cy - ry, cx + rx, cy + ry), debut, fin,
                   fill=couleur, width=ep * S)

    def finir(self) -> Image.Image:
        return self.image.resize((self.w, self.h), Image.LANCZOS)


def _plus_sombre(c, k=0.8):
    return tuple(int(v * k) for v in c)


def corps(st) -> Toile:
    t = Toile(640, 520, (400, 840))
    # cou
    t.poly([(360, 560), (440, 560), (448, 660), (352, 660)], st["peau"])
    # boubou : épaules arrondies, évasé vers le bas
    t.poly([(400, 640), (250, 665), (165, 730), (120, 1100), (680, 1100), (635, 730), (550, 665)],
           st["habit"])
    t.ellipse(205, 720, 70, 60, st["habit"], contour=None)
    t.ellipse(595, 720, 70, 60, st["habit"], contour=None)
    # encolure et broderie
    t.poly([(350, 648), (450, 648), (400, 730)], st["peau"])
    t.ligne([(345, 650), (400, 740), (455, 650)], st["motif"], 10)
    for k in range(4):
        y = 770 + 70 * k
        t.ligne([(375, y), (400, y + 25), (425, y)], st["motif"], 7)
    return t


def oreilles(st) -> Toile:
    t = Toile(520, 140, (400, 420))
    for x in (192, 608):
        t.ellipse(x, 420, 34, 50, st["peau"])
        t.ellipse(x, 420, 15, 26, _plus_sombre(st["peau"]), contour=None)
    return t


def tete(st) -> Toile:
    t = Toile(440, 500, TETE)
    t.ellipse(400, 410, 205, 235, st["peau"], ep=6)
    return t


def joues(st) -> Toile:
    t = Toile(380, 80, (400, 495))
    rose = (196, 96, 86, 110)
    for x in (270, 530):
        t.ellipse(x, 495, 38, 22, rose, contour=None)
    return t


def yeux(st, ferme: bool) -> Toile:
    t = Toile(300, 110, (400, 420))
    for x in (325, 475):
        if ferme:
            t.arc(x, 405, 40, 22, 20, 160, ep=7)
        else:
            t.ellipse(x, 420, 42, 36, (252, 250, 245), ep=5)
    return t


def pupilles(st) -> Toile:
    t = Toile(260, 70, (400, 424))
    for x in (325, 475):
        t.ellipse(x, 424, 19, 21, (45, 28, 22), contour=None)
        t.ellipse(x + 7, 416, 6, 6, (255, 255, 255), contour=None)
    return t


def sourcils(st) -> Toile:
    t = Toile(300, 60, (400, 352))
    t.ligne([(285, 360), (325, 344), (365, 352)], CONTOUR, 11)
    t.ligne([(435, 352), (475, 344), (515, 360)], CONTOUR, 11)
    return t


def nez(st) -> Toile:
    t = Toile(110, 80, (400, 485))
    t.arc(400, 478, 34, 26, 20, 160, _plus_sombre(st["peau"], 0.6), 7)
    return t


LEVRES = (120, 45, 45)
INTERIEUR = (70, 20, 25)
DENTS = (250, 248, 240)
LANGUE = (210, 100, 105)


def bouche(st, forme: str) -> Toile:
    t = Toile(180, 120, (400, 560))
    cx, cy = 400, 560
    if forme == "X":  # repos : léger sourire
        t.arc(cx, cy - 22, 48, 30, 25, 155, LEVRES, 7)
    elif forme == "A":  # lèvres closes (m, b, p)
        t.ellipse(cx, cy, 44, 9, LEVRES, contour=None)
    elif forme in "BCDH":
        ry = {"B": 12, "C": 24, "D": 38, "H": 28}[forme]
        rx = {"B": 46, "C": 48, "D": 52, "H": 44}[forme]
        t.ellipse(cx, cy, rx, ry, INTERIEUR, LEVRES, 6)
        t.d.rectangle(t.boite(cx - rx * 0.62, cy - ry + 4, cx + rx * 0.62, cy - ry + 4 + min(12, ry)),
                      fill=DENTS)
        if forme in "CD":
            t.ellipse(cx, cy + ry * 0.55, rx * 0.5, ry * 0.35, LANGUE, contour=None)
        if forme == "H":  # l : langue levée derrière les dents
            t.ellipse(cx, cy - 2, rx * 0.45, ry * 0.45, LANGUE, contour=None)
    elif forme == "E":  # arrondie (o)
        t.ellipse(cx, cy, 32, 30, INTERIEUR, LEVRES, 8)
    elif forme == "F":  # avancée (u, w)
        t.ellipse(cx, cy, 20, 18, INTERIEUR, LEVRES, 9)
    elif forme == "G":  # f, v : dents sur la lèvre
        t.ellipse(cx, cy + 4, 42, 12, LEVRES, contour=None)
        t.d.rectangle(t.boite(cx - 28, cy - 12, cx + 28, cy + 2), fill=DENTS)
    return t


def coiffe_arriere(st) -> Toile | None:
    if st["coiffe"] != "foulard":
        return None
    t = Toile(560, 340, (420, 215))
    t.ellipse(400, 215, 215, 150, _plus_sombre(st["coiffe_couleur"]), ep=6)
    t.ellipse(560, 120, 85, 70, _plus_sombre(st["coiffe_couleur"]), ep=6)  # le nœud
    return t


def coiffe(st) -> Toile:
    c, m = st["coiffe_couleur"], st["coiffe_motif"]
    if st["coiffe"] == "foulard":
        t = Toile(470, 290, (400, 245))
        t.poly([(185, 330), (200, 220), (260, 140), (400, 110), (540, 140), (600, 220), (615, 330),
                (520, 300), (400, 290), (280, 300)], c, ep=6)
        for x in range(250, 560, 60):
            t.ellipse(x, 215 + abs(x - 400) * 0.25, 12, 12, m, contour=None)
        t.ligne([(200, 318), (280, 290), (400, 280), (520, 290), (600, 318)], _plus_sombre(c, 0.7), 6)
    else:  # bonnet rond brodé
        t = Toile(400, 200, (400, 220))
        t.poly([(225, 300), (240, 180), (300, 140), (400, 128), (500, 140), (560, 180), (575, 300)],
               c, ep=6)
        for k in range(3):
            y = 200 + 30 * k
            t.ligne([(250, y), (400, y - 18), (550, y)], m, 6)
    return t


def creer(nom: str, dossier: Path) -> Path:
    st = STYLES[nom]
    dossier.mkdir(parents=True, exist_ok=True)
    parties = []

    def ajouter(partie: str, toiles: dict[str, Toile]):
        images = {}
        for variante, t in toiles.items():
            fichier = f"{partie}.png" if variante == "defaut" else f"{partie}_{variante}.png"
            t.finir().save(dossier / fichier)
            images[variante] = fichier
        parties.append({"nom": partie, "images": images, "position": list(next(iter(toiles.values())).centre)})

    arriere = coiffe_arriere(st)
    if arriere:
        ajouter("cheveux_arriere", {"defaut": arriere})
    ajouter("corps", {"defaut": corps(st)})
    ajouter("oreilles", {"defaut": oreilles(st)})
    ajouter("tete", {"defaut": tete(st)})
    ajouter("joues", {"defaut": joues(st)})
    ajouter("yeux", {"ouverts": yeux(st, False), "fermes": yeux(st, True)})
    ajouter("pupilles", {"defaut": pupilles(st)})
    ajouter("sourcils", {"defaut": sourcils(st)})
    ajouter("nez", {"defaut": nez(st)})
    ajouter("bouche", {f: bouche(st, f) for f in "XABCDEFGH"})
    ajouter("coiffe", {"defaut": coiffe(st)})
    description = {"nom": nom, "taille": list(TAILLE), "pivot_cou": [400, 600], "parties": parties}
    (dossier / "personnage.json").write_text(json.dumps(description, indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    return dossier




# --- une voix factice, pour essayer sans enregistrer ---

def voix_factice(texte: str, f0: float, sr: int = 44100, graine: int = 0):
    """Un babil voisé qui suit les syllabes du texte : une voyelle par groupe
    de voyelles, des consonnes muettes entre elles. Ce n'est pas du bambara,
    seulement de quoi faire bouger la bouche avant les vrais enregistrements."""
    import numpy as np
    rng = np.random.default_rng(graine)
    formants = {"a": (750, 1250), "e": (450, 2000), "ɛ": (580, 1800), "i": (300, 2300),
                "o": (450, 850), "ɔ": (580, 900), "u": (320, 750)}
    morceaux, t_total = [], 0.0
    syllabes = [c for c in texte.lower() if c in formants or c in " ,.!?"]
    for c in syllabes:
        if c in " ,":
            d = 0.12 if c == " " else 0.25
            morceaux.append(np.zeros(int(d * sr)))
            continue
        if c in ".!?":
            morceaux.append(np.zeros(int(0.35 * sr)))
            continue
        d = rng.uniform(0.13, 0.22)
        n = int(d * sr)
        t = np.arange(n) / sr + t_total
        hauteur = f0 * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t))
        phase = 2 * np.pi * np.cumsum(hauteur) / sr
        source = sum(np.sin(k * phase) / k for k in range(1, 25))
        f1, f2 = formants[c]
        spectre = np.fft.rfft(source)
        freqs = np.fft.rfftfreq(n, 1 / sr)
        filtre = (np.exp(-((freqs - f1) / 120) ** 2) + 0.6 * np.exp(-((freqs - f2) / 160) ** 2)
                  + 0.05)
        voyelle = np.fft.irfft(spectre * filtre, n)
        enveloppe = np.sin(np.pi * np.arange(n) / n) ** 0.7
        morceaux.append(voyelle * enveloppe)
        morceaux.append(np.zeros(int(rng.uniform(0.03, 0.07) * sr)))
        t_total += d
    signal = np.concatenate(morceaux) if morceaux else np.zeros(sr // 2)
    return (0.5 * signal / (np.abs(signal).max() + 1e-9)).astype("float32")


EPISODE_DEMO = {
    "format": [1080, 1920], "fps": 30,
    "fond": {"haut": "#fbe3b6", "bas": "#e59a5c"},
    "personnages": {
        "awa": {"dossier": "../../personnages/awa", "x": 300, "y": 1820, "echelle": 0.8},
        "seydou": {"dossier": "../../personnages/seydou", "x": 780, "y": 1820, "echelle": 0.8},
    },
    "repliques": [
        {"qui": "awa", "texte": "I ni sɔgɔma, Seydou !", "audio": "awa_1.wav"},
        {"qui": "seydou", "texte": "Nba, i ni sɔgɔma. I ka kɛnɛ wa ?", "audio": "seydou_1.wav"},
        {"qui": "awa", "texte": "Tɔɔrɔ si tɛ. Somɔgɔw ka kɛnɛ wa ?", "audio": "awa_2.wav",
         "mouvements": [{"t": 0.0, "lacet": -0.7}, {"t": 0.9, "lacet": -0.7}, {"t": 1.6, "lacet": 0.4}]},
        {"qui": "seydou", "texte": "U ka kɛnɛ kosɛbɛ, Ala ka tanu.", "audio": "seydou_2.wav"},
    ],
}


def episode_demo(dossier: Path, noms: tuple[str, str] = ("awa", "seydou"),
                 echelle: float = 0.8) -> Path:
    """Épisode d'essai : deux personnages, quatre répliques en voix factice
    (une voix aiguë pour le premier, une grave pour le second)."""
    import copy
    import soundfile as sf
    dossier.mkdir(parents=True, exist_ok=True)
    d = copy.deepcopy(EPISODE_DEMO)
    a, b = noms
    d["personnages"] = {
        a: {"dossier": f"../../personnages/{a}", "x": 300, "y": 1820 if echelle == 0.8 else 1920,
            "echelle": echelle},
        b: {"dossier": f"../../personnages/{b}", "x": 780, "y": 1820 if echelle == 0.8 else 1920,
            "echelle": echelle},
    }
    for k, r in enumerate(d["repliques"]):
        r["qui"] = a if r["qui"] == "awa" else b
        r["texte"] = r["texte"].replace("Seydou", b.capitalize())
        r["audio"] = f"{r['qui']}_{k // 2 + 1}.wav"
        sf.write(str(dossier / r["audio"]),
                 voix_factice(r["texte"], 230.0 if r["qui"] == a else 125.0, graine=k), 44100)
    if echelle != 0.8:  # vrais personnages : un décor plutôt qu'un dégradé
        fond_cour(dossier / "fond.png")
        d["fond"] = {"image": "fond.png"}
        for p in d["personnages"].values():
            p["y"] = 1920
    fichier = dossier / "episode.json"
    fichier.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    return fichier


def fond_cour(sortie: Path, taille: tuple[int, int] = (1080, 1920)) -> Path:
    """Un décor d'essai : une cour, un mur en banco, une porte, un arbre."""
    import numpy as np
    from PIL import ImageFilter
    w, h = taille
    rng = np.random.default_rng(3)
    k = np.linspace(0, 1, h)[:, None, None]
    ciel = (np.array([120, 185, 225]) * (1 - k) + np.array([250, 215, 160]) * k)
    image = Image.fromarray(np.repeat(ciel, w, axis=1).astype(np.uint8), "RGB").convert("RGBA")
    d = ImageDraw.Draw(image)
    haut_mur, sol = int(h * 0.30), int(h * 0.72)
    # mur en banco, arrondi en haut, avec ses irrégularités
    banco, banco_sombre = (196, 128, 78), (168, 104, 60)
    d.rectangle([0, haut_mur, w, sol], fill=banco)
    for x in range(-40, w + 80, 120):
        d.ellipse([x, haut_mur - 40, x + 120, haut_mur + 40], fill=banco)
    texture = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    t = ImageDraw.Draw(texture)
    for _ in range(260):
        x, y = rng.uniform(0, w), rng.uniform(haut_mur, sol)
        rx, ry = rng.uniform(15, 60), rng.uniform(6, 18)
        t.ellipse([x - rx, y - ry, x + rx, y + ry], fill=banco_sombre + (int(rng.uniform(40, 110)),))
    image.alpha_composite(texture.filter(ImageFilter.GaussianBlur(6)))
    d = ImageDraw.Draw(image)
    # porte en bois
    px0, px1 = int(w * 0.62), int(w * 0.86)
    d.rectangle([px0, int(h * 0.42), px1, sol], fill=(110, 70, 40), outline=CONTOUR, width=8)
    for x in range(px0 + 30, px1, 40):
        d.line([(x, int(h * 0.42) + 10), (x, sol - 6)], fill=(90, 56, 30), width=5)
    d.ellipse([px1 - 50, int(h * 0.57), px1 - 30, int(h * 0.57) + 20], fill=(230, 190, 90))
    # sol de terre
    k2 = np.linspace(0, 1, h - sol)[:, None, None]
    terre = (np.array([214, 160, 104]) * (1 - k2) + np.array([186, 128, 78]) * k2)
    image.paste(Image.fromarray(np.repeat(terre, w, axis=1).astype(np.uint8), "RGB"), (0, sol))
    d = ImageDraw.Draw(image)
    d.rectangle([0, sol - 6, w, sol + 6], fill=(150, 95, 55))
    # arbre : tronc et feuillage qui dépasse du mur
    d.polygon([(70, sol), (110, haut_mur + 40), (150, haut_mur + 40), (190, sol)], fill=(95, 62, 40),
              outline=CONTOUR)
    feuillage = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    f = ImageDraw.Draw(feuillage)
    for _ in range(40):
        x, y, r = rng.uniform(-60, 420), rng.uniform(80, haut_mur + 120), rng.uniform(70, 140)
        vert = (int(rng.uniform(50, 80)), int(rng.uniform(110, 150)), int(rng.uniform(50, 70)), 255)
        f.ellipse([x - r, y - r, x + r, y + r], fill=vert)
    image.alpha_composite(feuillage)
    # un léger flou : le décor reste derrière les personnages
    image = image.filter(ImageFilter.GaussianBlur(2.5))
    image.convert("RGB").save(sortie)
    return sortie


def episode_volume(dossier: Path) -> Path:
    """Épisode d'essai avec Fanta et Bakary en volume : gestes, tour complet,
    Bakary qui se retourne vers la porte. Voix factice."""
    import copy
    import soundfile as sf
    dossier.mkdir(parents=True, exist_ok=True)
    d = copy.deepcopy(EPISODE_DEMO)
    d["personnages"] = {
        "fanta": {"toon": "fanta", "x": 330, "y": 1790, "echelle": 0.82},
        "bakary": {"toon": "bakary", "x": 760, "y": 1790, "echelle": 0.82},
    }
    qui = ["fanta", "bakary", "fanta", "bakary"]
    for k, r in enumerate(d["repliques"]):
        r["qui"] = qui[k]
        r["texte"] = r["texte"].replace("Seydou", "Bakary")
        r["audio"] = f"{r['qui']}_{k // 2 + 1}.wav"
        r.pop("mouvements", None)
        sf.write(str(dossier / r["audio"]),
                 voix_factice(r["texte"], 230.0 if r["qui"] == "fanta" else 125.0, graine=k), 44100)
    # Fanta fait un tour sur elle-même de joie ; Bakary se retourne vers la porte
    d["repliques"][2]["mouvements"] = [{"t": 0.2, "corps": 28}, {"t": 1.6, "corps": 388}]
    d["repliques"][3]["mouvements"] = [{"t": 0.0, "corps": -28}, {"t": 0.8, "corps": -150},
                                       {"t": 1.8, "corps": -150}, {"t": 2.6, "corps": -28}]
    fond_cour(dossier / "fond.png")
    d["fond"] = {"image": "fond.png"}
    fichier = dossier / "episode.json"
    fichier.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    return fichier


def episode_tour(dossier: Path) -> Path:
    """Épisode d'essai avec les vrais dessins multi-vues de Fanta et Bakary :
    Fanta fait un tour sur elle-même, Bakary se retourne vers la porte."""
    import copy
    import soundfile as sf
    dossier.mkdir(parents=True, exist_ok=True)
    d = copy.deepcopy(EPISODE_DEMO)
    d["personnages"] = {
        "fanta": {"tour": "../../personnages/fanta_tour", "x": 320, "y": 1800, "echelle": 1.4},
        "bakary": {"tour": "../../personnages/bakary_tour", "x": 770, "y": 1800, "echelle": 1.4},
    }
    d["zoom_serre"] = 1.5
    qui = ["fanta", "bakary", "fanta", "bakary"]
    for k, r in enumerate(d["repliques"]):
        r["qui"] = qui[k]
        r["texte"] = r["texte"].replace("Seydou", "Bakary")
        r["audio"] = f"{r['qui']}_{k // 2 + 1}.wav"
        r.pop("mouvements", None)
        sf.write(str(dossier / r["audio"]),
                 voix_factice(r["texte"], 230.0 if r["qui"] == "fanta" else 125.0, graine=k), 44100)
    d["repliques"][2]["mouvements"] = [{"t": 0.2, "corps": 28}, {"t": 1.8, "corps": 388}]
    d["repliques"][3]["mouvements"] = [{"t": 0.0, "corps": -28}, {"t": 0.9, "corps": -150},
                                       {"t": 1.9, "corps": -150}, {"t": 2.8, "corps": -28}]
    fond_cour(dossier / "fond.png")
    d["fond"] = {"image": "fond.png"}
    fichier = dossier / "episode.json"
    fichier.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    return fichier
