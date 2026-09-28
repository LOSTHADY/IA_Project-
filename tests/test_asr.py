"""Tests de bambara_voice.asr sur de minuscules modèles aléatoires.

Aucun téléchargement : les modèles sont construits en mémoire. On ne teste
pas la qualité des sorties (aléatoires), mais la plomberie qui décide de ce
que le modèle fait — tâche forcée, langue, repli, famille CTC.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from bambara_voice.asr import SpeechRecognizer  # noqa: E402
from bambara_voice.config import ASRConfig, WHISPER_LANG_SLOT, e2e_config  # noqa: E402

AUDIO = np.zeros(16_000, dtype=np.float32)


def _tiny_whisper():
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    cfg = WhisperConfig(
        vocab_size=51865, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=16, decoder_ffn_dim=16, num_mel_bins=80,
        max_source_positions=1500, max_target_positions=32,
    )
    model = WhisperForConditionalGeneration(cfg).eval()
    gc = model.generation_config
    # Identifiants de openai/whisper-small.
    gc.lang_to_id = {"<|en|>": 50259, "<|fr|>": 50265, "<|sw|>": 50318}
    gc.task_to_id = {"transcribe": 50359, "translate": 50358}
    gc.is_multilingual = True
    gc.decoder_start_token_id = 50258
    gc.no_timestamps_token_id = 50363
    gc.forced_decoder_ids = None

    calls = []
    generate = model.generate

    def spy(*args, **kwargs):
        calls.append(dict(kwargs))
        return generate(*args, **kwargs)

    model.generate = spy
    return model, calls


class _WhisperProcessor:
    """Extracteur réel, tokenizer factice (le vocabulaire n'importe pas ici)."""

    def __init__(self):
        from transformers import WhisperFeatureExtractor

        self.fe = WhisperFeatureExtractor()

    def __call__(self, audio, sampling_rate, return_tensors):
        return self.fe(audio, sampling_rate=sampling_rate, return_tensors=return_tensors)

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["i ni ce"]


def _recognizer(**cfg):
    rec = SpeechRecognizer(ASRConfig(max_new_tokens=3, **cfg))
    rec._model, calls = _tiny_whisper()
    rec._processor = _WhisperProcessor()
    return rec, calls


def test_la_langue_par_defaut_est_celle_du_fine_tuning():
    # Doit rester alignée avec scripts/finetune_whisper.py.
    assert ASRConfig().language == WHISPER_LANG_SLOT == "sw"


@pytest.mark.parametrize("task", ["transcribe", "translate"])
def test_tache_et_langue_sont_forcees(task):
    rec, calls = _recognizer(task=task)
    res = rec.transcribe(AUDIO)
    assert calls[-1]["task"] == task and calls[-1]["language"] == "sw"
    assert res.language == ("fr" if task == "translate" else "bm")


def test_detection_automatique_si_langue_nulle():
    rec, calls = _recognizer(language=None)
    rec.transcribe(AUDIO)
    assert "language" not in calls[-1] and calls[-1]["task"] == "transcribe"


def test_langue_inconnue_repli_en_transcription(caplog):
    # Whisper n'a pas de token "bm" (transformers lèverait ValueError).
    rec, calls = _recognizer(language="bm")
    with caplog.at_level(logging.WARNING):
        rec.transcribe(AUDIO)
    assert len(calls) == 1 and "task" not in calls[0] and "language" not in calls[0]
    assert "<|bm|>" in caplog.text


def test_langue_inconnue_jamais_de_repli_en_traduction():
    # Sans tâche forcée, Whisper transcrirait : la variante e2e mesurerait
    # autre chose sans que rien ne le signale.
    rec, calls = _recognizer(language="bm", task="translate")
    with pytest.raises(ValueError, match="traduction directe impossible"):
        rec.transcribe(AUDIO)
    assert calls == []


def test_une_erreur_sans_rapport_n_est_pas_prise_pour_un_probleme_de_langue():
    # Ancien comportement : toute ValueError de generate déclenchait le
    # repli, et une erreur de longueur passait pour une langue inconnue.
    rec, calls = _recognizer(task="translate")
    rec.config.max_new_tokens = 10_000  # dépasse le décodeur du modèle
    with pytest.raises(ValueError) as err:
        rec.transcribe(AUDIO)
    assert "traduction directe impossible" not in str(err.value)
    assert len(calls) == 1


# --- CTC (MMS / wav2vec2) -----------------------------------------------------

def _tiny_ctc(tmp_path):
    from transformers import (
        Wav2Vec2Config, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC, Wav2Vec2Processor,
    )

    vocab = {"<pad>": 0, "<unk>": 1, "|": 2, "a": 3, "i": 4, "n": 5, "ɛ": 6}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    tok = Wav2Vec2CTCTokenizer(str(tmp_path / "vocab.json"))
    proc = Wav2Vec2Processor(feature_extractor=Wav2Vec2FeatureExtractor(), tokenizer=tok)
    cfg = Wav2Vec2Config(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, conv_dim=(16, 16),
        conv_stride=(5, 4), conv_kernel=(10, 8), num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
    )
    return Wav2Vec2ForCTC(cfg).eval(), proc


def test_ctc_transcrit(tmp_path):
    rec = SpeechRecognizer(ASRConfig(kind="ctc", language=None))
    rec._model, rec._processor = _tiny_ctc(tmp_path)
    res = rec.transcribe(AUDIO)
    assert isinstance(res.text, str) and res.language == "bm"


def test_ctc_refuse_la_traduction(tmp_path):
    rec = SpeechRecognizer(ASRConfig(kind="ctc", task="translate"))
    rec._model, rec._processor = _tiny_ctc(tmp_path)
    with pytest.raises(ValueError, match="CTC"):
        rec.transcribe(AUDIO)
    with pytest.raises(ValueError, match="CTC"):
        e2e_config(asr=ASRConfig(kind="ctc"))


def test_ctc_charge_l_adaptateur_mms(monkeypatch, tmp_path):
    model, proc = _tiny_ctc(tmp_path)
    seen = {}

    def fake_proc(model_id, **kw):
        seen["processor"] = kw
        return proc

    def fake_model(model_id, **kw):
        seen["model"] = kw
        return model

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", fake_proc)
    monkeypatch.setattr(transformers.Wav2Vec2ForCTC, "from_pretrained", fake_model)
    rec = SpeechRecognizer(ASRConfig(model_id="facebook/mms-1b-all", kind="ctc",
                                     target_lang="bam"))
    rec._ensure_loaded()
    assert seen["processor"] == {"target_lang": "bam"}
    assert seen["model"]["target_lang"] == "bam"
    assert seen["model"]["ignore_mismatched_sizes"] is True
