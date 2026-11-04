"""Ligne de commande de l'animation.

    python -m animation demo                      # personnages et épisode d'essai
    python -m animation planche animation/personnages/awa planche.png
    python -m animation rendre animation/episodes/demo/episode.json demo.mp4
    python -m animation rendre episode.json essai.mp4 --apercu   # moitié de taille, plus rapide
    python -m animation image episode.json 45 image.png          # une seule image, pour vérifier
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image, ImageDraw

from . import demo
from .personnage import VISEMES, Pose, charger
from .scene import Episode, _police

RACINE = Path(__file__).parent


def planche(dossier: str, sortie: str) -> Path:
    """Toutes les poses d'un personnage sur une image : rotations, bouches, clignement."""
    p = charger(dossier)
    poses = [(f"lacet {l:+.1f}", Pose(lacet=l)) for l in (-1, -0.5, 0, 0.5, 1)]
    poses += [("tangage -1", Pose(tangage=-1)), ("tangage +1", Pose(tangage=1)),
              ("roulis 8°", Pose(roulis=8)), ("clignement", Pose(cligne=True)),
              ("regard", Pose(regard=(1, 0)))]
    poses += [(f"bouche {v}", Pose(bouche=v)) for v in VISEMES]
    w, h = p.taille[0] // 2, p.taille[1] // 2
    colonnes = 5
    lignes = -(-len(poses) // colonnes)
    image = Image.new("RGB", (colonnes * w, lignes * (h + 40)), (250, 240, 225))
    dessin, police = ImageDraw.Draw(image), _police(26)
    for k, (titre, pose) in enumerate(poses):
        x, y = (k % colonnes) * w, (k // colonnes) * (h + 40)
        rendu = p.rendre(pose).resize((w, h), Image.LANCZOS)
        image.paste(rendu, (x, y + 40), rendu)
        dessin.text((x + w / 2, y + 8), titre, font=police, anchor="ma", fill=(60, 40, 30))
    image.save(sortie)
    return Path(sortie)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m animation", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sous = ap.add_subparsers(dest="commande", required=True)
    sous.add_parser("demo", help="dessine les personnages d'essai et crée l'épisode d'essai")
    pl = sous.add_parser("planche", help="toutes les poses d'un personnage sur une image")
    pl.add_argument("dossier")
    pl.add_argument("sortie")
    r = sous.add_parser("rendre", help="épisode -> vidéo MP4")
    r.add_argument("episode")
    r.add_argument("sortie")
    r.add_argument("--apercu", action="store_true", help="moitié de la taille, deux fois plus vite")
    r.add_argument("--levres", choices=("auto", "audio", "rhubarb"), default="auto",
                   help="synchronisation labiale (auto : Rhubarb s'il est installé)")
    im = sous.add_parser("image", help="une image de l'épisode, en PNG")
    im.add_argument("episode")
    im.add_argument("numero", type=int)
    im.add_argument("sortie")
    a = ap.parse_args(argv)

    if a.commande == "demo":
        for nom in demo.STYLES:
            print(demo.creer(nom, RACINE / "personnages" / nom))
        print(demo.episode_demo(RACINE / "episodes" / "demo"))
    elif a.commande == "planche":
        print(planche(a.dossier, a.sortie))
    elif a.commande == "rendre":
        debut = time.time()
        ep = Episode(a.episode, moteur_levres=a.levres)
        ep.rendre(a.sortie, echelle=0.5 if a.apercu else 1.0)
        print(f"{a.sortie} : {ep.duree:.1f} s de vidéo en {time.time() - debut:.0f} s")
    elif a.commande == "image":
        ep = Episode(a.episode)
        for i in range(a.numero + 1):  # dans l'ordre : la tête est lissée d'image en image
            image = ep.image(i)
        image.save(a.sortie)
        print(a.sortie)


if __name__ == "__main__":
    main()
