"""Atelier de données : enregistrer et juger sans rien transcrire.

Quatre gestes, tous à la portée d'un locuteur qui parle le bambara sans
l'écrire (docs/ATELIER.md) :

- **dire** : une phrase française s'affiche, on la dit en bambara, avec ses
  mots, à un ami, à un inconnu ou à une personne âgée (on ne parle pas de
  la même façon aux trois) → audio bambara + traduction française de
  référence, tutoyée ou vouvoyée selon le cas. C'est un jeu de test de la
  chaîne (chrF++ depuis l'audio, `eval.run_eval`) sur la vraie façon de
  parler, et des données pour le bout-en-bout ;
- **lire** : une phrase bambara s'affiche, on la lit → audio + texte exact.
  Données de synthèse vocale (une voix) et de reconnaissance ;
- **juger** : deux traductions d'une même phrase, à l'aveugle → préférence.
  Évaluation humaine (v1 contre v2, par exemple) ;
- **lexique** : argot et mots d'emprunt, avec leur sens ;
- **discuter** (mode entraînement) : on parle librement à l'assistant, puis
  on choisit ou corrige ce qu'il a entendu (en bambara, orthographe ramenée
  au standard par `bambara_voice.orthographe`) et ce qu'il a compris (en
  français). Données de reconnaissance et de traduction, et mémoire de
  l'assistant, en conversant.

Toute la logique vit ici pour être testée sans interface ; l'application
(`app/atelier_app.py`) n'est qu'une façade. Le dossier de l'atelier n'est
jamais versionné (la voix est une donnée personnelle, le dépôt est public) :

- `locuteurs.jsonl` : contributeurs pseudonymisés, avec leur consentement ;
- `dire.jsonl`, `lire.jsonl` : une prise de son par ligne ; `audio/<contributeur>/*.wav` ;
- `passes.jsonl` : phrases passées (illisibles, gênantes…) ;
- `lecture.jsonl` : phrases bambara à lire (`preparer_lecture`) ;
- `lots/*.jsonl` : paires à juger (`preparer_lot`) ; `jugements.jsonl` ;
- `lexique.jsonl` ;
- `conversations.jsonl` : les tours du mode « discuter » ; `reponses.jsonl` :
  les réponses de l'assistant jugées justes ou fausses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from bambara_voice.normalize import normalize

from .collection import (SAMPLE_RATE, Speaker, _from_dict, _next_id, _read_jsonl,
                         _write_jsonl, check_audio, to_mono_16k)

ROOT = Path(__file__).resolve().parent.parent
PHRASES_A_DIRE = ROOT / "data" / "atelier" / "a_dire.json"

MODES = ("dire", "lire")
# À qui l'on parle : la même phrase ne se dit pas pareil à un ami, à un
# inconnu ou à une personne âgée (formules, respect, argot). Chaque phrase
# peut se dire aux trois ; le français de référence est vouvoyé pour les
# deux derniers.
INTERLOCUTEURS = {"ami": "un ami", "inconnu": "un inconnu", "aine": "une personne âgée"}
VOUVOIEMENT = ("inconnu", "aine")
CHOIX = {"A": "A meilleure", "B": "B meilleure",
         "egal_bon": "les deux bonnes", "egal_mauvais": "les deux mauvaises"}
TYPES_LEXIQUE = ("argot", "mot français", "mot anglais", "expression", "autre")
LECTURE_CIBLE_MIN = 60   # une voix de synthèse correcte demande au moins une heure

# Phrases bibliques ou coraniques, fréquentes dans Bayelemabaga : peu utiles
# pour une voix d'assistant, et pénibles à lire à la chaîne.
_RELIGIEUX = re.compile(r"\b(yesu|krista|krisita|matigi|kira|musa|isirayɛli|isirayeli|"
                        r"sali|alakɔrɔ|alakɔrɔlen|sitanɛ|sɛtanɛ|eleyisa|filisitiw?)\b",
                        re.IGNORECASE)


# --- phrases ------------------------------------------------------------------

def charger_phrases_a_dire(path: str | Path = PHRASES_A_DIRE) -> list[dict]:
    """Phrases françaises à dire, dans l'ordre : {id, rubrique, titre, fr},
    plus « vous » (la forme vouvoyée) pour une phrase qui s'adresse à
    quelqu'un. Dans le fichier, une phrase est une chaîne, ou {fr, vous}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    phrases = []
    for rub in data["rubriques"]:
        for k, entree in enumerate(rub["phrases"], 1):
            formes = {"fr": entree} if isinstance(entree, str) else dict(entree)
            if not (formes.get("fr") or "").strip():
                raise ValueError(f"{path} : phrase vide dans « {rub['titre']} »")
            phrase = {"id": f"{rub['id']}-{k:02d}", "rubrique": rub["id"],
                      "titre": rub["titre"], "fr": formes["fr"].strip()}
            if (formes.get("vous") or "").strip():
                phrase["vous"] = formes["vous"].strip()
            phrases.append(phrase)
    ids = [p["id"] for p in phrases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} : identifiants de rubrique en double")
    return phrases


def partition(prompt_id: str) -> str:
    """« dev » ou « test », fixé par la phrase : toutes les prises d'une même
    phrase (locuteurs, interlocuteurs, façons de la dire) vont du même côté.
    Le dev sert à choisir (assemblage, checkpoint) ; le test, que ce choix
    n'a jamais vu, décide si un modèle remplace le précédent
    (eval.promotion). Stable d'une session et d'une machine à l'autre."""
    return "dev" if hashlib.sha1(prompt_id.encode("utf-8")).digest()[0] % 2 == 0 else "test"


def partie_voix(cle: str) -> str:
    """Pour l'étage de la voix (scripts/affiner_soloni.py) : « apprentissage »
    (70 %), « choix » du checkpoint (15 %) ou « test » (15 %), fixé par la
    phrase lue (`lu:<id>`) ou par le tour du mode Discuter (`co:<id>`)."""
    b = hashlib.sha1(cle.encode("utf-8")).digest()[0] * 100 // 256
    return "test" if b < 15 else "choix" if b < 30 else "apprentissage"


def lecteur_reserve(speaker_id: str) -> bool:
    """Un contributeur sur huit (spk-08, spk-16…) est mis de côté pour le test
    des « voix nouvelles » : rien de ce qu'il enregistre n'est appris, et il ne
    lit que des phrases du test. Les sept premiers apprennent tous."""
    m = re.search(r"(\d+)$", speaker_id or "")
    return bool(m) and int(m.group(1)) % 8 == 0


def _chemin_audio(speaker: str, ident: str) -> str:
    """Un dossier par contributeur : Drive peine sur un dossier de dizaines de
    milliers de fichiers (une campagne en produit des centaines de milliers).
    Les prises déjà faites gardent leur chemin, `audio/<id>.wav`."""
    return f"audio/{speaker}/{ident}.wav"


def usage(prompt_id: str) -> str:
    """Ce qu'une phrase apporte quand on apprend de la voix des locuteurs :
    « test » (la moitié test de `partition`, jamais apprise : c'est elle qui
    juge), « choix » (un quart du dev, jamais appris non plus : il choisit le
    checkpoint) ou « apprentissage » (le reste du dev). Fixé par la phrase,
    comme `partition`, dont il ne change aucune moitié."""
    if partition(prompt_id) == "test":
        return "test"
    return "choix" if hashlib.sha1(prompt_id.encode("utf-8")).digest()[1] % 4 == 0 else "apprentissage"


def texte_fr(phrase: dict, interlocuteur: str = "ami") -> str:
    """Le français de référence pour cet interlocuteur : vouvoyé pour un
    inconnu ou une personne âgée, quand la phrase s'adresse à quelqu'un."""
    if interlocuteur in VOUVOIEMENT and phrase.get("vous"):
        return phrase["vous"]
    return phrase["fr"]


def _interlocuteur(row: dict) -> str:
    """Celui d'une prise ou d'un passage. Les prises d'avant le choix de
    l'interlocuteur étaient dites sur une phrase tutoyée : « ami »."""
    return row.get("interlocuteur") or "ami"


def rubriques(phrases: list[dict]) -> list[tuple[str, str]]:
    """(id, titre) des rubriques, dans l'ordre du fichier."""
    vues: dict[str, str] = {}
    for p in phrases:
        vues.setdefault(p["rubrique"], p["titre"])
    return list(vues.items())


def preparer_lecture(out_path: str | Path, dataset: str = "RobotsMaliAI/bayelemabaga",
                     n: int = 400, seed: int = 0, min_mots: int = 4,
                     max_mots: int = 14) -> int:
    """Choisit des phrases bambara à lire dans un corpus public : de longueur
    moyenne, sans chiffres (dits de façons diverses), sans textes religieux,
    sans doublons. Écrit `lecture.jsonl` ; renvoie le nombre de phrases."""
    from .corpora import detect_columns, load_any

    ds = load_any(dataset, split="train")
    cols = detect_columns(ds.features)
    if cols.audio:
        ds = ds.remove_columns([cols.audio])
    textes = ([row[cols.bm] for row in ds[cols.nested]] if cols.nested else ds[cols.bm])
    return _ecrire_lecture(textes, out_path, dataset, n, seed, min_mots, max_mots)


def _ecrire_lecture(textes, out_path, source, n, seed, min_mots, max_mots) -> int:
    choisies, vues = [], set()
    for t in textes:
        t = normalize(t or "", lower=False)
        cle = t.lower()
        if (not t or cle in vues or re.search(r"\d", t) or _RELIGIEUX.search(t)
                or not min_mots <= len(t.split()) <= max_mots):
            continue
        vues.add(cle)
        choisies.append(t)
    random.Random(seed).shuffle(choisies)
    lignes = [{"id": f"lu-{k:04d}", "bm": t, "source": source}
              for k, t in enumerate(choisies[:n], 1)]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out, lignes)
    return len(lignes)


# --- lots à juger -------------------------------------------------------------

def trouver_rapport(dossier: str | Path, nom: str) -> Path | None:
    """Le rapport de eval.baselines le plus récent qui porte ce nom."""
    trouves = []
    for p in sorted(Path(dossier).glob("*.json")):
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("nom") == nom:
                trouves.append(p)
        except ValueError:
            continue
    return trouves[-1] if trouves else None


def _details(rapport: Path) -> tuple[str, str, dict[str, dict]]:
    entete = json.loads(rapport.read_text(encoding="utf-8"))
    lignes = _read_jsonl(rapport.with_suffix(".details.jsonl"))
    return (entete.get("nom") or rapport.stem, entete.get("maillon", ""),
            {str(r["id"]): r for r in lignes})


def _hypothese(ligne: dict, sens: str) -> str | None:
    """La traduction d'une ligne de rapport. En bm2fr, celle d'un rapport de
    traduction (hyp_fr), du bout-en-bout (hyp_fr, depuis l'audio) ou de la
    cascade (mt_depuis_asr : la traduction de ce que l'ASR a entendu)."""
    if sens == "fr2bm":
        return ligne.get("hyp_bm")
    return ligne.get("hyp_fr", ligne.get("mt_depuis_asr"))


def preparer_lot(rapport_a: str | Path, rapport_b: str | Path, sens: str,
                 dossier: str | Path, n: int = 100, seed: int = 0,
                 nom: str | None = None) -> tuple[Path, dict]:
    """Paires de traductions à juger à l'aveugle, à partir de deux rapports de
    traduction (`eval.baselines mt`) sur les mêmes phrases.

    `sens` : « fr2bm » (on juge du bambara, la source est en français) ou
    « bm2fr ». En bm2fr, les rapports peuvent aussi être ceux de la chaîne
    depuis l'audio (`eval.baselines asr`) : cascade contre bout-en-bout,
    jugés sur la transcription de référence. Les phrases traduites à
    l'identique par les deux systèmes ne départagent rien : elles sont
    comptées, pas montrées. Le côté (A ou B) de chaque système est tiré au
    hasard à chaque phrase."""
    if sens not in ("fr2bm", "bm2fr"):
        raise ValueError("sens attendu : fr2bm ou bm2fr")
    src_key = "ref_fr" if sens == "fr2bm" else "ref_bm"
    nom_a, maillon_a, a = _details(Path(rapport_a))
    nom_b, maillon_b, b = _details(Path(rapport_b))
    if sens == "fr2bm" and "asr" in (maillon_a, maillon_b):
        raise ValueError("fr2bm se juge sur des rapports de traduction (eval.baselines mt) : "
                         "dans un rapport de reconnaissance, hyp_bm est une transcription")
    hyp_a = {i: _hypothese(r, sens) for i, r in a.items()}
    hyp_b = {i: _hypothese(r, sens) for i, r in b.items()}
    communs = [i for i in a if i in b and a[i].get(src_key) and hyp_a[i] is not None
               and hyp_b.get(i) is not None and a[i].get(src_key) == b[i].get(src_key)]
    differents = [i for i in communs if hyp_a[i].strip() != hyp_b[i].strip()]
    rng = random.Random(seed)
    choisis = sorted(rng.sample(differents, min(n, len(differents))), key=differents.index)

    nom = nom or _slug(f"{nom_a}-contre-{nom_b}-{sens}")
    items = []
    for k, i in enumerate(choisis, 1):
        sides = [(nom_a, hyp_a[i]), (nom_b, hyp_b[i])]
        if rng.random() < 0.5:
            sides.reverse()
        items.append({"id": f"{nom}-{k:03d}", "lot": nom, "sens": sens, "ref_id": i,
                      "source": a[i][src_key], "a": sides[0][1], "b": sides[1][1],
                      "systeme_a": sides[0][0], "systeme_b": sides[1][0]})
    out = Path(dossier) / "lots" / f"{nom}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out, items)
    return out, {"communes": len(communs), "identiques": len(communs) - len(differents),
                 "retenues": len(items)}


def _slug(text: str) -> str:
    text = re.sub(r"[^\w]+", "-", normalize(text).replace("→", "-"))
    return text.strip("-")[:80]


def p_signe(gagnees: int, perdues: int) -> float:
    """p bilatéral du test du signe : les égalités sont écartées."""
    n = gagnees + perdues
    if n == 0:
        return 1.0
    k = min(gagnees, perdues)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


# --- l'atelier ----------------------------------------------------------------

@dataclass
class Prise:
    id: str
    mode: str                    # "dire" ou "lire"
    audio: str                   # relatif au dossier de l'atelier
    speaker: str
    prompt_id: str
    texte: str                   # français pour « dire », bambara pour « lire »
    rubrique: str = ""
    interlocuteur: str = ""      # « dire » : une clé de INTERLOCUTEURS
    code_switching: bool = False
    argot: str = ""              # mots familiers employés, s'il y en a
    conditions: str = ""
    duration_s: float = 0.0
    recorded_at: str = ""


@dataclass
class Jugement:
    id: str
    lot: str
    item: str
    juge: str
    choix: str                   # une clé de CHOIX
    prefere: str = ""            # système préféré, en clair ; vide en cas d'égalité
    recorded_at: str = ""


# Paliers de la campagne de lecture (docs/CAMPAGNE.md) : heures lues et
# nombre de lecteurs.
PALIERS_CAMPAGNE = (("pilote", 20, 30), ("significatif", 100, 100), ("grand", 300, 200))

STATUTS_ENTENDU = ("juste", "corrige", "inaudible")
STATUTS_COMPRIS = ("juste", "corrige")


@dataclass
class Tour:
    """Un tour du mode « discuter » : la personne parle librement, puis dit ce
    que la reconnaissance aurait dû entendre et ce que la phrase voulait dire."""
    id: str
    speaker: str
    audio: str
    entendu_auto: list           # propositions de la reconnaissance, la première d'abord
    entendu: str = ""            # choisi ou corrigé par la personne, tel qu'elle l'a écrit
    entendu_std: str = ""        # le même, en orthographe standard
    entendu_statut: str = ""     # une valeur de STATUTS_ENTENDU
    compris_auto: str = ""       # le français proposé par l'assistant
    compris: str = ""            # le sens voulu, confirmé ou corrigé
    compris_statut: str = ""     # une valeur de STATUTS_COMPRIS
    conditions: str = ""
    duration_s: float = 0.0
    recorded_at: str = ""


@dataclass
class Mot:
    id: str
    forme: str
    sens_fr: str
    equivalent_bm: str = ""
    exemple: str = ""
    type: str = "argot"
    auteur: str = ""
    added_at: str = ""


class Atelier:
    """Dossier de l'atelier. Sûr en présence de plusieurs contributeurs en
    même temps (Gradio traite les événements dans des fils séparés)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    def _path(self, name: str) -> Path:
        return self.root / name

    def _append(self, name: str, row: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _write_jsonl(self._path(name), _read_jsonl(self._path(name)) + [row])

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    # --- contributeurs ------------------------------------------------------

    def locuteurs(self) -> list[Speaker]:
        return [_from_dict(Speaker, r) for r in _read_jsonl(self._path("locuteurs.jsonl"))]

    def locuteur(self, speaker_id: str) -> Speaker:
        for s in self.locuteurs():
            if s.id == speaker_id:
                return s
        raise KeyError(f"contributeur inconnu : {speaker_id}")

    def ajouter_locuteur(self, gender: str, **info) -> Speaker:
        if not info.get("consent"):
            raise ValueError("pas d'enregistrement sans consentement")
        if gender not in ("f", "m"):
            raise ValueError("genre attendu : 'f' ou 'm'")
        with self._lock:
            rows = _read_jsonl(self._path("locuteurs.jsonl"))
            spk = Speaker(id=_next_id([r["id"] for r in rows], "spk-", 2), gender=gender,
                          consent_date=info.pop("consent_date", "") or self._now()[:10],
                          **info)
            self._append("locuteurs.jsonl", asdict(spk))
        return spk

    # --- prises de son ------------------------------------------------------

    def prises(self, mode: str | None = None, speaker: str | None = None) -> list[Prise]:
        modes = [mode] if mode else list(MODES)
        out = []
        for m in modes:
            out += [_from_dict(Prise, r) for r in _read_jsonl(self._path(f"{m}.jsonl"))]
        return [p for p in out if speaker is None or p.speaker == speaker]

    def enregistrer(self, mode: str, speaker: str, phrase: dict, audio: np.ndarray,
                    sample_rate: int, code_switching: bool = False, argot: str = "",
                    conditions: str = "", interlocuteur: str = "ami"
                    ) -> tuple[Prise, list[str]]:
        """Écrit le WAV et sa ligne. Renvoie aussi les problèmes audio, pour
        proposer de recommencer tout de suite (« Annuler la dernière »).
        `interlocuteur` : à qui la phrase est dite (« dire » seulement)."""
        if mode not in MODES:
            raise ValueError(f"mode inconnu : {mode}")
        if mode == "dire" and interlocuteur not in INTERLOCUTEURS:
            raise ValueError(f"interlocuteur inconnu : {interlocuteur}")
        if not self.locuteur(speaker).consent:
            raise ValueError(f"{speaker} n'a pas donné son consentement")
        wav = to_mono_16k(audio, sample_rate)
        problemes = check_audio(wav)
        if "enregistrement vide" in problemes:
            raise ValueError("enregistrement vide : rien n'a été capté")

        import soundfile as sf

        texte = texte_fr(phrase, interlocuteur) if mode == "dire" else phrase["bm"]
        with self._lock:
            prefix = "di-" if mode == "dire" else "li-"
            rows = _read_jsonl(self._path(f"{mode}.jsonl"))
            pid = _next_id([r["id"] for r in rows], prefix, 5)
            rel = _chemin_audio(speaker, pid)
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(self.root / rel), wav, SAMPLE_RATE, subtype="PCM_16")
            prise = Prise(id=pid, mode=mode, audio=rel, speaker=speaker,
                          prompt_id=phrase["id"], texte=texte,
                          rubrique=phrase.get("rubrique", ""),
                          interlocuteur=interlocuteur if mode == "dire" else "",
                          code_switching=bool(code_switching), argot=argot.strip(),
                          conditions=conditions,
                          duration_s=round(len(wav) / SAMPLE_RATE, 2),
                          recorded_at=self._now())
            self._append(f"{mode}.jsonl", asdict(prise))
        return prise, problemes

    def annuler_derniere(self, mode: str, speaker: str) -> Prise | None:
        """Retire la dernière prise de ce contributeur dans ce mode (ligne et
        fichier) : la phrase revient en tête de la file."""
        with self._lock:
            rows = _read_jsonl(self._path(f"{mode}.jsonl"))
            idx = [k for k, r in enumerate(rows) if r["speaker"] == speaker]
            if not idx:
                return None
            row = rows.pop(idx[-1])
            _write_jsonl(self._path(f"{mode}.jsonl"), rows)
            (self.root / row["audio"]).unlink(missing_ok=True)
        return _from_dict(Prise, row)

    def passer(self, mode: str, speaker: str, prompt_id: str,
               interlocuteur: str = "ami") -> None:
        row = {"mode": mode, "speaker": speaker, "prompt_id": prompt_id, "at": self._now()}
        if mode == "dire":
            row["interlocuteur"] = interlocuteur
        with self._lock:
            self._append("passes.jsonl", row)

    def _faites(self, mode: str, speaker: str, interlocuteur: str = "ami") -> set[str]:
        """Phrases dites ou passées ; en « dire », pour cet interlocuteur."""
        garder = (lambda r: _interlocuteur(r) == interlocuteur) if mode == "dire" \
            else (lambda r: True)
        faites = {p.prompt_id for p in self.prises(mode, speaker) if garder(asdict(p))}
        faites |= {r["prompt_id"] for r in _read_jsonl(self._path("passes.jsonl"))
                   if r["mode"] == mode and r["speaker"] == speaker and garder(r)}
        return faites

    def prochaine(self, mode: str, speaker: str, phrases: list[dict],
                  rubrique: str | None = None, interlocuteur: str = "ami") -> dict | None:
        """Première phrase, dans l'ordre, que ce contributeur n'a ni dite ni
        passée (dans la rubrique demandée, ou dans toutes ; en « dire », à
        cet interlocuteur)."""
        faites = self._faites(mode, speaker, interlocuteur)
        for p in phrases:
            if p["id"] not in faites and (not rubrique or p.get("rubrique") == rubrique):
                return p
        return None

    def prochaine_lecture(self, speaker: str, phrases: list[dict],
                          max_lecteurs: int = 3) -> dict | None:
        """La prochaine phrase à lire pour ce contributeur, dans un ordre qui
        lui est propre (deux lecteurs ne lisent pas les mêmes phrases dans le
        même ordre) : d'abord celles que moins de `max_lecteurs` personnes ont
        lues (plus de phrases différentes, chacune lue par plusieurs voix),
        puis les autres. Un lecteur réservé au test (`lecteur_reserve`) ne lit
        que des phrases du test."""
        from collections import Counter

        faites = self._faites("lire", speaker)
        lectures = Counter(p.prompt_id for p in self.prises("lire"))
        reserve = lecteur_reserve(speaker)
        ordre = sorted(phrases, key=lambda p: hashlib.sha1(
            f"{speaker}:{p['id']}".encode("utf-8")).digest())
        for plafond in (max_lecteurs, None):
            for p in ordre:
                if p["id"] in faites or (reserve and partie_voix(f"lu:{p['id']}") != "test"):
                    continue
                if plafond is None or lectures[p["id"]] < plafond:
                    return p
        return None

    def deja_faite(self, mode: str, speaker: str, prompt_id: str,
                   interlocuteur: str = "ami") -> bool:
        return prompt_id in self._faites(mode, speaker, interlocuteur)

    def versions(self, speaker: str, prompt_id: str) -> dict[str, int]:
        """Combien de fois ce contributeur a dit cette phrase, à qui (une même
        personne peut l'entendre de plusieurs façons), dans l'ordre de
        INTERLOCUTEURS ; seulement ceux à qui il l'a dite."""
        dites = [_interlocuteur(asdict(p)) for p in self.prises("dire", speaker)
                 if p.prompt_id == prompt_id]
        return {i: dites.count(i) for i in INTERLOCUTEURS if i in dites}

    def autre_interlocuteur(self, speaker: str, prompt_id: str, actuel: str) -> str | None:
        """Le suivant, après `actuel`, à qui cette phrase reste à dire (ni dite
        ni passée) ; None si elle a été faite pour tous."""
        ordre = list(INTERLOCUTEURS)
        k = ordre.index(actuel) if actuel in ordre else -1
        for i in ordre[k + 1:] + ordre[:k + 1]:
            if i != actuel and prompt_id not in self._faites("dire", speaker, i):
                return i
        return None

    def progression(self, mode: str, speaker: str, phrases: list[dict],
                    interlocuteur: str = "ami") -> list[dict]:
        """Par rubrique : phrases faites (dites ou passées) sur le total ; en
        « dire », pour cet interlocuteur."""
        faites = self._faites(mode, speaker, interlocuteur)
        par_rubrique: dict[str, dict] = {}
        for p in phrases:
            r = par_rubrique.setdefault(p.get("rubrique", ""), {
                "rubrique": p.get("rubrique", ""), "titre": p.get("titre", "Lecture"),
                "faites": 0, "total": 0})
            r["total"] += 1
            r["faites"] += p["id"] in faites
        return list(par_rubrique.values())

    def par_interlocuteur(self, speaker: str | None = None) -> dict[str, int]:
        """Prises « dire » par interlocuteur (de ce contributeur, ou de tous)."""
        dites = [_interlocuteur(asdict(p)) for p in self.prises("dire", speaker)]
        return {i: dites.count(i) for i in INTERLOCUTEURS}

    def minutes(self, mode: str | None = None, speaker: str | None = None) -> float:
        return round(sum(p.duration_s for p in self.prises(mode, speaker)) / 60, 1)

    def phrases_a_lire(self) -> list[dict]:
        return _read_jsonl(self._path("lecture.jsonl"))

    # --- discuter (mode entraînement) ----------------------------------------

    def tours(self, speaker: str | None = None) -> list[Tour]:
        rows = [_from_dict(Tour, r) for r in _read_jsonl(self._path("conversations.jsonl"))]
        return [t for t in rows if speaker is None or t.speaker == speaker]

    def enregistrer_tour(self, speaker: str, audio: np.ndarray, sample_rate: int,
                         entendu_auto: list[str], entendu: str, entendu_std: str,
                         entendu_statut: str, compris_auto: str, compris: str,
                         compris_statut: str, conditions: str = "") -> Tour:
        """Écrit le WAV et la ligne d'un tour corrigé par la personne."""
        if entendu_statut not in STATUTS_ENTENDU or compris_statut not in STATUTS_COMPRIS:
            raise ValueError(f"statuts attendus : {STATUTS_ENTENDU}, {STATUTS_COMPRIS}")
        if not compris.strip():
            raise ValueError("il faut ce que la phrase voulait dire")
        if not self.locuteur(speaker).consent:
            raise ValueError(f"{speaker} n'a pas donné son consentement")
        wav = to_mono_16k(audio, sample_rate)
        if "enregistrement vide" in check_audio(wav):
            raise ValueError("enregistrement vide : rien n'a été capté")

        import soundfile as sf

        with self._lock:
            rows = _read_jsonl(self._path("conversations.jsonl"))
            tid = _next_id([r["id"] for r in rows], "co-", 5)
            rel = _chemin_audio(speaker, tid)
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(self.root / rel), wav, SAMPLE_RATE, subtype="PCM_16")
            tour = Tour(id=tid, speaker=speaker, audio=rel, entendu_auto=list(entendu_auto),
                        entendu=entendu.strip(), entendu_std=entendu_std.strip(),
                        entendu_statut=entendu_statut, compris_auto=compris_auto.strip(),
                        compris=compris.strip(), compris_statut=compris_statut,
                        conditions=conditions, duration_s=round(len(wav) / SAMPLE_RATE, 2),
                        recorded_at=self._now())
            self._append("conversations.jsonl", asdict(tour))
        return tour

    def juger_reponse(self, tour_id: str, juge: str, reponse_bm: str, juste: bool) -> None:
        """La réponse de l'assistant à ce tour était-elle juste ?"""
        with self._lock:
            self._append("reponses.jsonl", {"tour": tour_id, "juge": juge,
                                            "reponse_bm": reponse_bm, "juste": bool(juste),
                                            "at": self._now()})

    # --- jugements ----------------------------------------------------------

    def lots(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "lots").glob("*.jsonl"))

    def items(self, lot: str) -> list[dict]:
        return _read_jsonl(self.root / "lots" / f"{lot}.jsonl")

    def jugements(self, lot: str | None = None) -> list[Jugement]:
        rows = [_from_dict(Jugement, r) for r in _read_jsonl(self._path("jugements.jsonl"))]
        return [j for j in rows if lot is None or j.lot == lot]

    def prochain_item(self, lot: str, juge: str) -> dict | None:
        faits = {j.item for j in self.jugements(lot) if j.juge == juge}
        return next((i for i in self.items(lot) if i["id"] not in faits), None)

    def juger(self, lot: str, juge: str, item_id: str, choix: str) -> Jugement:
        if choix not in CHOIX:
            raise ValueError(f"choix inconnu : {choix}")
        item = next((i for i in self.items(lot) if i["id"] == item_id), None)
        if item is None:
            raise KeyError(f"phrase inconnue dans {lot} : {item_id}")
        prefere = {"A": item["systeme_a"], "B": item["systeme_b"]}.get(choix, "")
        with self._lock:
            rows = _read_jsonl(self._path("jugements.jsonl"))
            j = Jugement(id=_next_id([r["id"] for r in rows], "ju-", 5), lot=lot,
                         item=item_id, juge=juge, choix=choix, prefere=prefere,
                         recorded_at=self._now())
            self._append("jugements.jsonl", asdict(j))
        return j

    def resultats(self, lot: str) -> dict:
        """Préférences levées (quel système a été préféré), et test du signe
        sur les jugements tranchés."""
        items = self.items(lot)
        systemes = sorted({i["systeme_a"] for i in items} | {i["systeme_b"] for i in items})
        js = self.jugements(lot)
        gagnes = {s: sum(j.prefere == s for j in js) for s in systemes}
        res = {"lot": lot, "items": len(items), "jugements": len(js), "preferences": gagnes,
               "egal_bon": sum(j.choix == "egal_bon" for j in js),
               "egal_mauvais": sum(j.choix == "egal_mauvais" for j in js)}
        if len(systemes) == 2:
            res["p_signe"] = p_signe(gagnes[systemes[0]], gagnes[systemes[1]])
        return res

    # --- lexique ------------------------------------------------------------

    def lexique(self) -> list[Mot]:
        return [_from_dict(Mot, r) for r in _read_jsonl(self._path("lexique.jsonl"))]

    def ajouter_mot(self, forme: str, sens_fr: str, equivalent_bm: str = "",
                    exemple: str = "", type: str = "argot", auteur: str = "") -> Mot:
        """Ajoute un mot, ou complète celui qui a la même forme."""
        forme, sens_fr = forme.strip(), sens_fr.strip()
        if not forme or not sens_fr:
            raise ValueError("il faut le mot et son sens en français")
        if type not in TYPES_LEXIQUE:
            raise ValueError(f"type attendu : {', '.join(TYPES_LEXIQUE)}")
        with self._lock:
            rows = _read_jsonl(self._path("lexique.jsonl"))
            rows = [r for r in rows if r["forme"].lower() != forme.lower()]
            mot = Mot(id=_next_id([r["id"] for r in _read_jsonl(self._path("lexique.jsonl"))],
                                  "mot-", 4),
                      forme=forme, sens_fr=sens_fr, equivalent_bm=equivalent_bm.strip(),
                      exemple=exemple.strip(), type=type, auteur=auteur, added_at=self._now())
            _write_jsonl(self._path("lexique.jsonl"), rows + [asdict(mot)])
        return mot

    def retirer_mot(self, forme: str) -> bool:
        with self._lock:
            rows = _read_jsonl(self._path("lexique.jsonl"))
            reste = [r for r in rows if r["forme"].lower() != forme.strip().lower()]
            _write_jsonl(self._path("lexique.jsonl"), reste)
        return len(reste) < len(rows)

    # --- bilan et exports ---------------------------------------------------

    def bilan(self, phrases_dire: list[dict]) -> dict:
        lecture = self.phrases_a_lire()
        par_locuteur = []
        for s in self.locuteurs():
            par_qui = self.par_interlocuteur(s.id)
            par_locuteur.append({
                "locuteur": s.id, "genre": s.gender,
                "dites": sum(par_qui.values()),
                "par_interlocuteur": par_qui,
                "a_dire": len(phrases_dire),
                "lues": len(self.prises("lire", s.id)),
                "minutes_lues": self.minutes("lire", s.id),
                "tours": len(self.tours(s.id)),
                "telephone": s.telephone, "reserve": lecteur_reserve(s.id),
                "jugements": sum(j.juge == s.id for j in self.jugements()),
            })
        tours = self.tours()
        return {"locuteurs": par_locuteur,
                "tours": {"total": len(tours),
                          "entendu_corrige": sum(t.entendu_statut == "corrige" for t in tours),
                          "compris_corrige": sum(t.compris_statut == "corrige" for t in tours)},
                "par_interlocuteur": self.par_interlocuteur(),
                "minutes": {"dire": self.minutes("dire"), "lire": self.minutes("lire")},
                "phrases_a_lire": len(lecture), "lexique": len(self.lexique()),
                "lots": [self.resultats(lot) for lot in self.lots()]}

    def bilan_campagne(self) -> dict:
        """Où en est la campagne de lecture : heures et lecteurs (appris, et
        réservés au test des voix nouvelles), téléphones, lieux, phrases."""
        from collections import Counter

        lues = self.prises("lire")
        lecteurs = {p.speaker for p in lues}
        reserves = {s for s in lecteurs if lecteur_reserve(s)}
        par_id = {s.id: s for s in self.locuteurs()}
        heures = round(sum(p.duration_s for p in lues) / 3600, 2)
        out = {"heures": heures,
               "heures_reservees": round(sum(p.duration_s for p in lues
                                             if p.speaker in reserves) / 3600, 2),
               "lecteurs": len(lecteurs), "reserves": len(reserves),
               "femmes": sum(par_id[s].gender == "f" for s in lecteurs if s in par_id),
               "telephones": len({par_id[s].telephone.lower() for s in lecteurs
                                  if s in par_id and par_id[s].telephone}),
               "lieux": dict(Counter(p.conditions or "?" for p in lues)),
               "phrases_lues": len({p.prompt_id for p in lues}),
               "phrases_disponibles": len(self.phrases_a_lire())}
        out["palier"] = next((nom for nom, h, n in reversed(PALIERS_CAMPAGNE)
                              if heures >= h and len(lecteurs) >= n), None)
        return out

    def exporter(self) -> dict[str, Path]:
        """Fichiers prêts à servir, dans le dossier de l'atelier :

        - `testset-dire.jsonl` : jeu de test pour `eval.run_eval` (audio et
          traduction française ; pas de transcription bambara, donc pas de
          WER, mais le chrF++ depuis l'audio, la mesure centrale) ;
        - `lecture-export.jsonl` : audio et texte bambara exact (voix,
          reconnaissance) ;
        - `testset-lire.jsonl` : les mêmes lectures, jeu de test pour
          `eval.run_eval` (WER de la reconnaissance sur ces voix) ;
        - `conversations-export.jsonl` : les tours du mode « discuter » dont
          le bambara a été confirmé ou corrigé (audio, texte en orthographe
          standard, sens voulu) : jamais du test ;
        - `lexique.tsv`."""
        genres = {s.id: s.gender for s in self.locuteurs()}
        regions = {s.id: s.region for s in self.locuteurs()}
        out = {}
        test = [{"id": p.id, "audio": p.audio, "transcript_bm": "", "translation_fr": p.texte,
                 "speaker": p.speaker, "gender": genres.get(p.speaker, ""),
                 "region": regions.get(p.speaker, ""), "register": "dit",
                 "interlocuteur": _interlocuteur(asdict(p)),
                 "code_switching": p.code_switching, "duration_s": p.duration_s,
                 "rubrique": p.rubrique, "conditions": p.conditions, "argot": p.argot,
                 "prompt_id": p.prompt_id, "partition": partition(p.prompt_id)}
                for p in self.prises("dire")]
        out["testset"] = self.root / "testset-dire.jsonl"
        _write_jsonl(out["testset"], test)
        lecture = [{"audio": p.audio, "text": p.texte, "speaker": p.speaker,
                    "duration_s": p.duration_s} for p in self.prises("lire")]
        out["lecture"] = self.root / "lecture-export.jsonl"
        _write_jsonl(out["lecture"], lecture)
        # Les lectures comme jeu de test de la reconnaissance : le texte lu est
        # exact, d'où le WER de la reconnaissance sur ces voix.
        fr_de = {x["id"]: x.get("fr", "") for x in self.phrases_a_lire()}
        lu = [{"id": p.id, "audio": p.audio, "transcript_bm": p.texte,
               "translation_fr": fr_de.get(p.prompt_id, ""),
               "speaker": p.speaker, "gender": genres.get(p.speaker, ""),
               "region": regions.get(p.speaker, ""), "register": "lu",
               "conditions": p.conditions, "prompt_id": p.prompt_id,
               "duration_s": p.duration_s} for p in self.prises("lire")]
        out["testset_lire"] = self.root / "testset-lire.jsonl"
        _write_jsonl(out["testset_lire"], lu)
        out["conversations"] = self.root / "conversations-export.jsonl"
        _write_jsonl(out["conversations"], [
            {"audio": t.audio, "text": t.entendu_std, "fr": t.compris, "speaker": t.speaker,
             "duration_s": t.duration_s} for t in self.tours()
            if t.entendu_statut in ("juste", "corrige") and t.entendu_std])
        out["lexique"] = self.root / "lexique.tsv"
        lignes = ["forme\tsens_fr\tequivalent_bm\ttype\texemple"]
        lignes += ["\t".join([m.forme, m.sens_fr, m.equivalent_bm, m.type, m.exemple])
                   for m in self.lexique()]
        out["lexique"].write_text("\n".join(lignes) + "\n", encoding="utf-8")
        return out


# --- ligne de commande --------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--racine", type=Path, required=True, help="dossier de l'atelier")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("lecture", help="choisir les phrases bambara à lire")
    sp.add_argument("--dataset", default="RobotsMaliAI/bayelemabaga")
    sp.add_argument("-n", type=int, default=400)
    sp = sub.add_parser("lot", help="préparer des paires de traductions à juger")
    sp.add_argument("--resultats", type=Path, required=True,
                    help="dossier des rapports de eval.baselines")
    sp.add_argument("--noms", nargs=2, required=True, metavar=("NOM_A", "NOM_B"))
    sp.add_argument("--sens", choices=["fr2bm", "bm2fr"], required=True)
    sp.add_argument("-n", type=int, default=100)
    sub.add_parser("exporter", help="jeu de test, lecture et lexique")
    args = p.parse_args(argv)

    if args.cmd == "lecture":
        n = preparer_lecture(args.racine / "lecture.jsonl", args.dataset, args.n)
        print(f"{n} phrases à lire : {args.racine / 'lecture.jsonl'}")
    elif args.cmd == "lot":
        rapports = [trouver_rapport(args.resultats, nom) for nom in args.noms]
        if None in rapports:
            raise SystemExit(f"rapport introuvable dans {args.resultats} : "
                             f"{[n for n, r in zip(args.noms, rapports) if r is None]}")
        path, stats = preparer_lot(*rapports, args.sens, args.racine, args.n)
        print(f"{path} : {stats['retenues']} paires ({stats['identiques']} traductions "
              f"identiques écartées sur {stats['communes']})")
    else:
        for k, path in Atelier(args.racine).exporter().items():
            print(f"{k} : {path}")


if __name__ == "__main__":
    main()
