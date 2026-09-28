"""Configuration centrale du pipeline.

Tous les identifiants de modèles vivent ici, pas dans le code des composants :
changer de modèle ne doit jamais demander de toucher à la logique.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict, fields, replace
from pathlib import Path
from typing import Literal

import json

ROOT = Path(__file__).resolve().parent.parent

# Codes de langue NLLB.
BAM = "bam_Latn"
FRA = "fra_Latn"
ENG = "eng_Latn"

# Whisper n'a pas de token de langue pour le bambara. Le fine-tuning
# (scripts/finetune_whisper.py) détourne le token swahili ; l'inférence doit
# utiliser le même, sinon la tâche transcribe/translate n'est plus forcée et
# les deux architectures produisent la même sortie.
WHISPER_LANG_SLOT = "sw"


@dataclass
class ASRConfig:
    """Reconnaissance / traduction de la parole bambara.

    `task` distingue les deux architectures comparées dans ce projet :
      - "transcribe" : audio bm -> texte bm, puis MT séparée (cascade)
      - "translate"  : audio bm -> texte fr directement (bout-en-bout)

    `kind` choisit la famille de modèle : "whisper" (seq2seq, les deux tâches)
    ou "ctc" (wav2vec2 / MMS, transcription seule — cascade uniquement).
    """

    model_id: str = "openai/whisper-small"
    task: Literal["transcribe", "translate"] = "transcribe"
    language: str | None = WHISPER_LANG_SLOT  # None : détection automatique
    sample_rate: int = 16_000
    beam_size: int = 1  # 1 = greedy, suffisant et bien plus rapide sur CPU
    max_new_tokens: int = 200
    kind: Literal["whisper", "ctc"] = "whisper"
    target_lang: str | None = None  # adaptateur de langue MMS, ex. "bam"
    # "ctranslate2" : modèle converti par scripts/export_cpu.py (int8 sur CPU,
    # cf. docs/DEPLOIEMENT.md pour les gains mesurés). Whisper uniquement.
    backend: Literal["transformers", "ctranslate2"] = "transformers"
    compute_type: str = "int8"  # ctranslate2 : int8, int8_float32, float32...


@dataclass
class MTConfig:
    """Traduction bambara <-> français."""

    model_id: str = "facebook/nllb-200-distilled-600M"
    src_lang: str = BAM
    tgt_lang: str = FRA
    beam_size: int = 4
    max_new_tokens: int = 256
    backend: Literal["transformers", "ctranslate2"] = "transformers"
    compute_type: str = "int8"


@dataclass
class LLMConfig:
    """Modèle de dialogue. Traité comme une boîte noire remplaçable.

    Les contraintes de style sont le vrai levier : une réponse en français
    simple et courte se traduit nettement mieux vers le bambara qu'une réponse
    élégante. Cf. docs/METHODE.md, levier 2.
    """

    backend: Literal["transformers", "llamacpp", "echo"] = "transformers"
    model_id: str = "google/gemma-3-1b-it"
    gguf_path: str | None = None  # requis si backend == "llamacpp"
    max_new_tokens: int = 96
    temperature: float = 0.3
    max_words: int = 15  # contrainte dure sur la longueur des phrases produites
    system_prompt: str = (
        "Tu es un assistant vocal. Tes réponses sont traduites automatiquement "
        "vers le bambara, donc elles doivent être faciles à traduire.\n"
        "Règles strictes :\n"
        "- Réponds en français simple.\n"
        "- Phrases courtes, 15 mots maximum, une idée par phrase.\n"
        "- Pas de subordonnées, pas de tournures idiomatiques, pas de jargon.\n"
        "- Vocabulaire concret et courant.\n"
        "- 3 phrases maximum au total.\n"
        "- Pas de listes, pas de mise en forme, pas d'emoji."
    )


@dataclass
class TTSConfig:
    """Synthèse vocale bambara."""

    model_id: str = "facebook/mms-tts-bam"
    sample_rate: int = 16_000
    speaker: str | None = None  # utilisé par les modèles multi-locuteurs


@dataclass
class TemplateConfig:
    """Réponses gabarits validées par un locuteur natif.

    Court-circuite la traduction sortante quand la requête est reconnue : la
    sortie est alors du bambara humain, sans erreur de MT. Cf. levier 3.
    """

    enabled: bool = True
    path: Path = ROOT / "data" / "templates.json"
    threshold: float = 0.62  # similarité minimale pour accepter un gabarit


@dataclass
class PipelineConfig:
    """Configuration complète d'une variante du système."""

    name: str = "cascade"
    architecture: Literal["cascade", "e2e"] = "cascade"
    asr: ASRConfig = field(default_factory=ASRConfig)
    mt_in: MTConfig = field(default_factory=lambda: MTConfig(src_lang=BAM, tgt_lang=FRA))
    mt_out: MTConfig = field(default_factory=lambda: MTConfig(src_lang=FRA, tgt_lang=BAM))
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    templates: TemplateConfig = field(default_factory=TemplateConfig)
    device: str = os.environ.get("BV_DEVICE", "cpu")

    def __post_init__(self) -> None:
        if self.architecture == "e2e":
            if self.asr.kind == "ctc":
                raise ValueError("un modèle CTC ne traduit pas : la variante e2e exige Whisper")
            # En bout-en-bout, l'ASR produit déjà du français : pas de MT entrante.
            self.asr.task = "translate"

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


def cascade_config(**overrides) -> PipelineConfig:
    """Variante A : ASR bambara -> MT -> LLM -> MT -> TTS (4 maillons)."""
    return PipelineConfig(name="cascade", architecture="cascade", **overrides)


def e2e_config(**overrides) -> PipelineConfig:
    """Variante B : traduction vocale directe bm->fr (un maillon en moins)."""
    return PipelineConfig(name="e2e", architecture="e2e", **overrides)


def build_config(architecture: str = "cascade", path: str | Path | None = None) -> PipelineConfig:
    """Point d'entrée commun de la CLI, de la démo et de l'évaluation : les
    valeurs par défaut, ou un fichier de déploiement (`load_config`)."""
    if path:
        return load_config(path, architecture)
    return e2e_config() if architecture == "e2e" else cascade_config()


def split_device(device: str) -> tuple[str, int]:
    """"cuda:1" -> ("cuda", 1) : la forme qu'attend CTranslate2."""
    name, _, index = device.partition(":")
    return name, int(index or 0)


# Champs contenant un chemin : relatifs au fichier de configuration s'ils y
# existent, sinon laissés tels quels (identifiant du Hub).
_PATH_FIELDS = ("model_id", "gguf_path", "path")


def load_config(path: str | Path, architecture: str = "cascade") -> PipelineConfig:
    """Configuration de déploiement lue depuis un fichier JSON.

    Le fichier ne donne que ce qui change par rapport aux valeurs par défaut,
    section par section, et sert aux deux architectures :

        {"asr": {"model_id": "modeles-cpu/whisper-small-bm",
                 "backend": "ctranslate2"},
         "mt_out": {"model_id": "modeles-cpu/nllb-fr2bm", "backend": "ctranslate2"},
         "llm": {"backend": "llamacpp", "gguf_path": "modeles-cpu/llm.gguf"}}

    Une faute de frappe dans un nom de section ou de champ est une erreur, pas
    un réglage silencieusement ignoré.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    base_dir = path.resolve().parent
    defaults = PipelineConfig()
    sections = ("asr", "mt_in", "mt_out", "llm", "tts", "templates")

    unknown = set(data) - set(sections) - {"device", "_comment"}
    if unknown:
        raise ValueError(f"{path} : sections inconnues {sorted(unknown)} ; "
                         f"attendues : {list(sections)} et device")

    overrides: dict = {}
    for key in sections:
        if key not in data:
            continue
        current = getattr(defaults, key)
        valid = {f.name for f in fields(current)}
        bad = set(data[key]) - valid
        if bad:
            raise ValueError(f"{path} : champs inconnus dans '{key}' : {sorted(bad)} ; "
                             f"valides : {sorted(valid)}")
        values = dict(data[key])
        for name in _PATH_FIELDS:
            value = values.get(name)
            if isinstance(value, str) and not Path(value).is_absolute() \
                    and (base_dir / value).exists():
                values[name] = str(base_dir / value)
        if key == "templates" and "path" in values:
            values["path"] = Path(values["path"])
        # replace() garde les valeurs propres au rôle (sens de traduction de
        # mt_in et mt_out) que le fichier ne précise pas.
        overrides[key] = replace(current, **values)
    if "device" in data:
        overrides["device"] = data["device"]

    build = e2e_config if architecture == "e2e" else cascade_config
    return build(**overrides)
