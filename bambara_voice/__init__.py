"""Assistant vocal bambara — chaîne ASR / traduction / dialogue / synthèse."""

from .config import (
    PipelineConfig, ASRConfig, MTConfig, LLMConfig, TTSConfig, TemplateConfig,
    cascade_config, e2e_config, BAM, FRA, ENG,
)
from .normalize import normalize, fold
from .pipeline import VoicePipeline, TurnTrace

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig", "ASRConfig", "MTConfig", "LLMConfig", "TTSConfig",
    "TemplateConfig", "cascade_config", "e2e_config", "BAM", "FRA", "ENG",
    "normalize", "fold", "VoicePipeline", "TurnTrace", "__version__",
]
