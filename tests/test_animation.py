"""Animation : calques, synchronisation labiale, épisode, vidéo."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from animation import demo, levres
from animation.personnage import VISEMES, Pose, charger
from animation.scene import SR, Episode


@pytest.fixture(scope="module")
def awa(tmp_path_factory):
    return demo.creer("awa", tmp_path_factory.mktemp("perso") / "awa")


def test_poses_differentes(awa):
    p = charger(awa)
    face = np.asarray(p.rendre(Pose()))
    assert face.shape == (p.taille[1], p.taille[0], 4)
    for pose in (Pose(lacet=0.8), Pose(tangage=0.8), Pose(roulis=6), Pose(cligne=True),
                 Pose(bouche="D")):
        assert np.abs(np.asarray(p.rendre(pose)).astype(int) - face).sum() > 0, pose


def test_parallaxe_nez_glisse_plus_que_les_oreilles(awa):
    p = charger(awa)
    nez = next(x for x in p.parties if x.nom == "nez")
    oreilles = next(x for x in p.parties if x.nom == "oreilles")
    assert nez.profondeur > 0 > oreilles.profondeur


def test_bouche_chaque_forme(awa):
    p = charger(awa)
    bouche = next(x for x in p.parties if x.role == "bouche")
    assert set(bouche.images) == set(VISEMES)


def test_calques_pleine_taille(tmp_path):
    """Sans personnage.json : des calques de la taille du dessin, aux noms convenus."""
    for nom, boite, couleur in (("tete", (100, 50, 300, 300), "sienna"),
                                ("yeux_ouverts", (150, 120, 250, 160), "white"),
                                ("yeux_fermes", (150, 135, 250, 145), "black"),
                                ("bouche_X", (170, 220, 230, 230), "maroon"),
                                ("bouche_D", (165, 205, 235, 245), "maroon")):
        im = Image.new("RGBA", (400, 500), (0, 0, 0, 0))
        im.paste(Image.new("RGBA", (boite[2] - boite[0], boite[3] - boite[1]), couleur), boite[:2])
        im.save(tmp_path / f"{nom}.png")
    p = charger(tmp_path)
    assert p.taille == (400, 500)
    yeux = next(x for x in p.parties if x.nom == "yeux")
    assert set(yeux.images) == {"ouverts", "fermes"} and yeux.position == (200, 140)
    assert np.any(np.asarray(p.rendre(Pose(bouche="D"))) != np.asarray(p.rendre(Pose())))


def test_levres_silence_et_voix():
    fps = 30
    voix = demo.voix_factice("a i a o u", 200.0, SR)
    signal = np.concatenate([np.zeros(SR), voix, np.zeros(SR)])
    formes = levres.depuis_audio(signal, SR, fps)
    assert len(formes) == int(np.ceil(len(signal) / (SR / fps)))
    assert set(formes) <= set(VISEMES)
    assert set(formes[:20]) == {"X"} and set(formes[-20:]) == {"X"}
    assert set(formes[fps + 3: fps + 3 + int(len(voix) / SR * fps) - 6]) - {"X"}


def test_tenir():
    import itertools
    sortie = levres._tenir(list("XDXDBDCX"), 2)
    series = [len(list(g)) for _, g in itertools.groupby(sortie)]
    assert all(n >= 2 for n in series[:-1])  # plus de forme d'une seule image
    assert levres._tenir(list("XXDDXX"), 2) == list("XXDDXX")


@pytest.fixture(scope="module")
def episode(tmp_path_factory):
    racine = tmp_path_factory.mktemp("ep")
    for nom in demo.STYLES:
        demo.creer(nom, racine / "personnages" / nom)
    dossier = racine / "episodes" / "e"
    dossier.mkdir(parents=True)
    for k, qui in enumerate(("awa", "seydou")):
        sf.write(str(dossier / f"{k}.wav"), demo.voix_factice("i ni sɔgɔma", 200.0, 16000), 16000)
    d = {"format": [270, 480], "fps": 10,
         "personnages": {"awa": {"dossier": "../../personnages/awa", "x": 80, "y": 470, "echelle": 0.2},
                         "seydou": {"dossier": "../../personnages/seydou", "x": 190, "y": 470,
                                    "echelle": 0.2}},
         "repliques": [{"qui": "awa", "texte": "I ni sɔgɔma", "audio": "0.wav"},
                       {"qui": "seydou", "texte": "Nba", "audio": "1.wav",
                        "mouvements": [{"t": 0, "lacet": -1.0}]}]}
    (dossier / "episode.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return dossier / "episode.json"


def test_l_image_suit_la_voix(episode):
    ep = Episode(episode, moteur_levres="audio")
    a, b = ep.repliques
    assert b.debut == pytest.approx(a.debut + a.duree + 0.35)
    assert ep.duree == pytest.approx(b.debut + b.duree + 0.8)
    assert len(ep.piste()) == pytest.approx(ep.duree * SR, abs=2)


def test_ecoute_et_mouvements_imposes(episode):
    ep = Episode(episode, moteur_levres="audio")
    a, b = ep.repliques
    fin_a = int((a.debut + a.duree) * ep.fps) - 1
    for i in range(fin_a):
        ep.image(i)
    assert ep.acteurs["seydou"].lacet < -0.3  # écoute Awa, à sa gauche
    for i in range(fin_a, int((b.debut + b.duree) * ep.fps)):
        ep.image(i)
    assert ep.acteurs["seydou"].lacet < -0.7  # mouvement imposé : lacet -1


def test_replique_d_un_inconnu(episode, tmp_path):
    d = json.loads(episode.read_text(encoding="utf-8"))
    d["repliques"][0]["qui"] = "personne"
    autre = episode.parent / "faux.json"
    autre.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(KeyError):
        Episode(autre)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg absent")
def test_video(episode, tmp_path):
    sortie = Episode(episode, moteur_levres="audio").rendre(tmp_path / "v.mp4")
    assert sortie.stat().st_size > 1000


def test_decoupe_des_dessins_de_chatgpt(tmp_path):
    """Les images brutes de Fanta : chaque calque est là, et la tête n'a plus de traits."""
    from pathlib import Path
    from animation.decoupe import decouper
    brut = Path(__file__).parent.parent / "animation" / "brut" / "fanta"
    p = charger(decouper(brut, tmp_path / "fanta"))
    noms = {x.nom for x in p.parties}
    assert {"corps", "oreilles", "tete", "blanc_yeux", "pupilles", "yeux", "sourcils", "nez",
            "bouche"} <= noms
    bouche = next(x for x in p.parties if x.nom == "bouche")
    assert set(bouche.images) == set(VISEMES)
    assert len({im.size for im in bouche.images.values()}) == 1  # toutes sur la même boîte
    # la tête sans traits : là où étaient les yeux, de la peau
    tete = next(x for x in p.parties if x.nom == "tete")
    yeux = next(x for x in p.parties if x.nom == "yeux")
    im = np.asarray(tete.images["defaut"].convert("RGB")).astype(int)
    x = round(yeux.position[0] - tete.position[0] + tete.images["defaut"].width / 2)
    y = round(yeux.position[1] - tete.position[1] + tete.images["defaut"].height / 2)
    assert im[y - 5:y + 5, x - 5:x + 5].std(axis=(0, 1)).max() < 12
    for pose in (Pose(lacet=1), Pose(roulis=8), Pose(cligne=True), Pose(bouche="D")):
        p.rendre(pose)
