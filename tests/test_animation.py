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


def test_volume_de_face_ne_change_rien(awa):
    """À angle nul, la tête plaquée sur l'ellipsoïde est le dessin d'origine."""
    p = charger(awa)
    plat = charger(awa)
    plat.volume = False
    a = np.asarray(p.rendre(Pose())).astype(int)
    b = np.asarray(plat.rendre(Pose())).astype(int)
    assert np.abs(a - b).mean() < 2.0


def test_volume_le_nez_tourne_plus_que_les_oreilles(awa):
    from animation.volume import Volume
    p = charger(awa)
    v = Volume(p.parties, 30, 12)
    th, ph = v.angles(1.0, 0.0)
    nez = next(x for x in p.parties if x.nom == "nez")
    oreilles = next(x for x in p.parties if x.nom == "oreilles")
    xn, _, sxn, _ = v.trait("nez", *nez.position, th, ph)
    xo, _, _, _ = v.trait("oreilles", *oreilles.position, th, ph)
    assert xn - nez.position[0] > 0.3 * v.A          # le nez part dans le sens du regard
    assert xo - oreilles.position[0] < xn - nez.position[0]
    # l'œil du fond se rétrécit, celui de devant s'élargit
    _, _, s_g, _ = v.trait("yeux", v.cx - 0.4 * v.A, v.cy, th, ph)
    _, _, s_d, _ = v.trait("yeux", v.cx + 0.4 * v.A, v.cy, th, ph)
    assert s_d < s_g


def test_camera_plan_rapproche_sur_celui_qui_parle(episode):
    ep = Episode(episode, moteur_levres="audio")
    a = ep.repliques[0]
    x, _, zoom = ep.camera(a.debut + 0.1)
    assert zoom > 1.5 and x == ep.acteurs["awa"].x
    assert ep.camera(0.0) == (ep.taille[0] / 2, ep.taille[1] / 2, 1.0)


# --- personnages en volume (toon.py) ---

def _toon():
    from animation.toon import Toon, modele
    return Toon(modele("fanta"))


def test_toon_tour_complet():
    from animation.toon import PoseToon
    t = _toon()
    blancs = {}
    for angle in (0, 90, 180):
        im = np.asarray(t.rendre(PoseToon(corps=angle), 0.35)).astype(int)
        assert im[..., 3].sum() > 0
        # le blanc des yeux : visible de face, plus de profil, pas du tout de dos
        blanc = (im[..., :3].min(-1) > 225) & (im[..., 3] > 200)
        blancs[angle] = blanc.sum()
    assert blancs[0] > blancs[90] > blancs[180] == 0


def test_toon_geste_et_bouche():
    from animation.toon import PoseToon
    t = _toon()
    repos = np.asarray(t.rendre(PoseToon(), 0.35)).astype(int)
    geste = np.asarray(t.rendre(PoseToon(bras_d=1.0), 0.35)).astype(int)
    bouche = np.asarray(t.rendre(PoseToon(bouche="D"), 0.35)).astype(int)
    assert np.abs(geste - repos).sum() > 0 and np.abs(bouche - repos).sum() > 0


def test_episode_en_volume_tour_impose(tmp_path):
    d = {"format": [270, 480], "fps": 10,
         "personnages": {"fanta": {"toon": "fanta", "x": 135, "y": 470, "echelle": 0.3}},
         "repliques": [{"qui": "fanta", "texte": "I ni sɔgɔma", "audio": "0.wav",
                        "mouvements": [{"t": 0.0, "corps": 0}, {"t": 0.5, "corps": 360}]}]}
    sf.write(str(tmp_path / "0.wav"), demo.voix_factice("i ni sɔgɔma a", 200.0, 16000), 16000)
    (tmp_path / "e.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    ep = Episode(tmp_path / "e.json", moteur_levres="audio")
    etats = [ep.etat(i) for i in range(ep.n_images)]
    angles = [e["fanta"]["corps"] for e in etats]
    assert max(angles) > 300  # elle a fait le tour
    assert ep.image(len(etats) // 2, etats[len(etats) // 2]).size == (270, 480)


# --- personnages multi-vues (vues.py) ---

@pytest.fixture(scope="module")
def bakary_tour(tmp_path_factory):
    from animation.vues import Tour, preparer
    return Tour(preparer("bakary", destination=tmp_path_factory.mktemp("tour") / "bakary_tour"))


def test_tour_cinq_vues_et_miroirs(bakary_tour):
    from animation.vues import PoseTour
    t = bakary_tour
    assert set(t.vues) == {0, 45, 90, 135, 180, -45, -90, -135, -180}
    face = np.asarray(t.rendre(PoseTour(corps=0)))
    dos = np.asarray(t.rendre(PoseTour(corps=180)))
    droite = np.asarray(t.rendre(PoseTour(corps=90)))
    gauche = np.asarray(t.rendre(PoseTour(corps=-90)))
    assert np.abs(face.astype(int) - dos).mean() > 3
    assert np.abs(droite[:, ::-1].astype(int) - gauche).mean() < 1  # le profil gauche est le miroir


def test_tour_bouche_clignement_tete(bakary_tour):
    from animation.vues import PoseTour
    t = bakary_tour
    repos = np.asarray(t.rendre(PoseTour())).astype(int)
    for pose in (PoseTour(bouche="D"), PoseTour(cligne=True), PoseTour(tete=40), PoseTour(corps=20)):
        assert np.abs(np.asarray(t.rendre(pose)).astype(int) - repos).sum() > 0, pose
    # de dos, ni bouche ni yeux : la forme de bouche ne change rien
    assert np.array_equal(np.asarray(t.rendre(PoseTour(corps=180, bouche="D"))),
                          np.asarray(t.rendre(PoseTour(corps=180))))
