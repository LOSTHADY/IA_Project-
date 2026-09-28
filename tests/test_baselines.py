"""Tests de eval/baselines.py : calculs, rapports et ligne de commande, avec
des composants factices à la place des modèles."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("datasets")

from bambara_voice.asr import ASRResult  # noqa: E402
from bambara_voice.normalize import fold  # noqa: E402
from bambara_voice.tts import SpeechResult  # noqa: E402
from eval import baselines  # noqa: E402
from eval.compare import to_markdown  # noqa: E402
from eval.corpora import Example  # noqa: E402
from test_corpora import _bayelemabaga_like, _jeli_like  # noqa: E402

AUDIO = np.zeros(32_000, dtype=np.float32)


class FakeASR:
    """Renvoie la référence repliée en ASCII : erreur purement orthographique."""

    def __init__(self, *args, **kwargs):
        self.refs = {}

    def transcribe(self, audio):
        return ASRResult(text=self.refs.get(len(audio), "n be taa"), language="bm",
                         audio_seconds=len(audio) / 16_000)


class FakeMT:
    """Traduction « parfaite » vers le français ; vers le bambara, la sortie
    prévue par `outputs`, sinon le texte tel quel."""

    def __init__(self, config=None, device="cpu", outputs=None):
        self.config = config
        self.outputs = outputs or {}

    def translate(self, text):
        if self.config is None or self.config.tgt_lang.startswith("fra"):
            return "je pars"
        return self.outputs.get(text, text)


class FakeTTS:
    def __init__(self, *args, **kwargs):
        pass

    def synthesize(self, text):
        return SpeechResult(np.zeros(8_000, dtype=np.float32), 16_000)


def test_run_asr_et_propagation():
    exs = [Example(id="a", bm="n bɛ taa", fr="je pars", audio=AUDIO),
           Example(id="b", bm="", fr="", audio=AUDIO)]  # sans référence : ignoré
    report, rows = baselines.run_asr(exs, FakeASR(), FakeMT())
    assert len(rows) == 1
    assert report["asr"]["wer"] > 0 and report["asr"]["wer_folded"] == 0
    assert report["propagation_erreurs"]["perte_absolue"] == 0
    assert report["latence"]["rtf_moyen"] >= 0 and "asr" in report["latence"]["par_etape_s"]


def test_run_asr_sans_traduction_de_reference():
    report, _ = baselines.run_asr([Example(id="a", bm="n bɛ taa", audio=AUDIO)],
                                  FakeASR(), FakeMT())
    assert "propagation_erreurs" not in report


def test_run_mt_isole_l_ecart_orthographique():
    from bambara_voice.config import BAM, FRA, MTConfig

    exs = [Example(id="a", bm="n bɛ taa sɔgɔma", fr="je pars")]
    ascii_bm = fold("n bɛ taa sɔgɔma")  # bon bambara, mauvaise graphie
    report, _ = baselines.run_mt(exs, FakeMT(MTConfig(src_lang=BAM, tgt_lang=FRA)),
                                 FakeMT(MTConfig(src_lang=FRA, tgt_lang=BAM),
                                        outputs={"je pars": ascii_bm}))
    assert round(report["mt_bm_fr"]["chrf"]) == 100
    # Sortie bambara en ASCII : pénalisée en strict, pas en replié.
    assert report["mt_fr_bm"]["chrf"] < 100
    assert round(report["mt_fr_bm_replie"]["chrf"]) == 100


def test_run_tts_ecrit_audio_et_grille_mos(tmp_path):
    report, rows = baselines.run_tts([("tts-000", "i ni ce")], FakeTTS(), tmp_path)
    assert (tmp_path / "tts-000.wav").exists()
    with (tmp_path / "mos.csv").open(encoding="utf-8") as fh:
        grid = list(csv.DictReader(fh))
    assert grid[0]["texte_bm"] == "i ni ce" and grid[0]["note_1_a_5"] == ""
    assert "rtf_moyen" in report["latence"]


@pytest.fixture
def fakes(monkeypatch):
    import bambara_voice.asr
    import bambara_voice.mt
    import bambara_voice.tts

    monkeypatch.setattr(bambara_voice.asr, "SpeechRecognizer", FakeASR)
    monkeypatch.setattr(bambara_voice.mt, "Translator", FakeMT)
    monkeypatch.setattr(bambara_voice.tts, "SpeechSynthesizer", FakeTTS)


def _report(out: Path, prefix: str) -> dict:
    [path] = out.glob(f"{prefix}-*[0-9].json")
    assert path.with_suffix(".details.jsonl").exists()
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_asr_mt_tts_et_tableau(tmp_path, fakes):
    jeli = _jeli_like(tmp_path / "jeli")
    bayel = _bayelemabaga_like(tmp_path / "bayel")
    out = tmp_path / "results"
    common = ["--out", str(out), "--device", "cpu"]

    baselines.main(common + ["asr", "--dataset", str(jeli), "--with-mt", "--limit", "2"])
    asr = _report(out, "zero-shot-asr")
    assert asr["nom"] == "zero-shot whisper-small (sw)"
    assert asr["corpus"]["evaluees"] == 2 and asr["corpus"]["partition"] == "test"
    assert asr["environnement"]["appareil"] == "cpu" and "propagation_erreurs" in asr

    baselines.main(common + ["mt", "--dataset", str(bayel)])
    mt = _report(out, "zero-shot-mt")
    assert mt["corpus"]["partition"] == "validation" and mt["composition"]["n"] == 2

    baselines.main(common + ["tts", "--dataset", str(jeli), "--limit", "2"])
    tts = _report(out, "zero-shot-tts")
    assert tts["composition"]["n"] == 2 and Path(tts["tts"]["dossier"]).is_dir()

    table = to_markdown([asr, mt, tts])
    assert "zero-shot whisper-small (sw)" in table
    assert "WER relâché" in table and "chrF++ fr→bm (replié)" in table
