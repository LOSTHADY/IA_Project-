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
    # La traduction de la sortie de l'ASR fait partie du chemin de la cascade.
    assert list(report["latence"]["par_etape_s"]) == ["asr", "mt_in"]
    assert report["latence"]["rtf_moyen"] >= 0


def test_run_asr_bout_en_bout_note_le_francais_directement():
    exs = [Example(id="a", bm="n bɛ taa", fr="je pars", audio=AUDIO),
           Example(id="b", bm="n bɛ taa", fr="", audio=AUDIO)]  # sans français : ignoré

    class DirectFr:
        def transcribe(self, audio):
            return ASRResult(text="je pars", language="fr", audio_seconds=2.0)

    report, rows = baselines.run_asr(exs, DirectFr(), e2e=True)
    assert len(rows) == 1 and "asr" not in report
    assert round(report["mt_in_depuis_asr"]["chrf"]) == 100
    assert list(report["latence"]["par_etape_s"]) == ["asr"]


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


def test_run_mt_compte_les_sorties_coupees():
    from bambara_voice.config import BAM, FRA, MTConfig

    class Looping(FakeMT):
        """Vers le bambara, la seconde phrase s'arrête sur la longueur maximale."""

        def translate(self, text):
            out = super().translate(text)
            self.last_capped = text == "il tousse"
            return out

    exs = [Example(id="a", bm="n bɛ taa", fr="je pars"),
           Example(id="b", bm="a bɛ sɔgɔsɔgɔ", fr="il tousse")]
    report, rows = baselines.run_mt(exs, FakeMT(MTConfig(src_lang=BAM, tgt_lang=FRA)),
                                    Looping(MTConfig(src_lang=FRA, tgt_lang=BAM)))
    assert report["sorties_coupees"] == {"bm_fr": 0.0, "fr_bm": 0.5}
    assert [r["coupe_bm"] for r in rows] == [False, True]


def test_run_mt_decoupe_le_francais_avant_traduction():
    from bambara_voice.config import BAM, FRA, MTConfig

    class Echo(FakeMT):
        """Vers le bambara, rend chaque segment entre crochets."""

        def translate(self, text):
            self.calls = getattr(self, "calls", []) + [text]
            return f"[{text}]"

    fr = "Quand le chef est arrivé au grand marché du village, tous les commerçants se sont levés."
    exs = [Example(id="a", bm="n bɛ taa", fr=fr), Example(id="b", bm="i ni ce", fr="Bonjour.")]
    fr2bm = Echo(MTConfig(src_lang=FRA, tgt_lang=BAM))
    report, rows = baselines.run_mt(exs, FakeMT(MTConfig(src_lang=BAM, tgt_lang=FRA)), fr2bm,
                                    split_fr=10)
    assert fr2bm.calls == ["Quand le chef est arrivé au grand marché du village,",
                           "tous les commerçants se sont levés.", "Bonjour."]
    assert rows[0]["hyp_bm"] == ("[Quand le chef est arrivé au grand marché du village,] "
                                 "[tous les commerçants se sont levés.]")
    assert [r["segments_fr"] for r in rows] == [2, 1]
    assert report["decoupe_fr"] == {"max_mots": 10, "phrases_decoupees": 0.5,
                                    "segments_moyens": 1.5}


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
    asr = _report(out, "asr-cascade")
    assert asr["nom"] == "whisper-small + nllb-200-distilled-600M"
    assert asr["corpus"]["evaluees"] == 2 and asr["corpus"]["partition"] == "test"
    assert asr["environnement"]["appareil"] == "cpu" and "propagation_erreurs" in asr

    baselines.main(common + ["--label", "bout-en-bout", "asr", "--dataset", str(jeli),
                             "--task", "translate"])
    e2e = _report(out, "asr-e2e")
    assert e2e["nom"] == "bout-en-bout" and e2e["architecture"] == "e2e"
    assert "asr" not in e2e and e2e["composition"]["n"] == 3

    baselines.main(common + ["mt", "--dataset", str(bayel),
                             "--bm2fr-model", "ckpt/nllb-bm2fr", "--fr2bm-model", "ckpt/nllb-fr2bm"])
    mt = _report(out, "mt")
    assert mt["nom"] == "nllb-bm2fr + nllb-fr2bm / bayel"
    assert mt["corpus"]["partition"] == "validation" and mt["composition"]["n"] == 2

    baselines.main(common + ["tts", "--dataset", str(jeli), "--limit", "2"])
    tts = _report(out, "tts")
    assert tts["composition"]["n"] == 2 and Path(tts["tts"]["dossier"]).is_dir()

    table = to_markdown([asr, e2e, mt, tts])
    assert "| bout-en-bout |" in table.splitlines()[0]
    # Cascade et bout-en-bout sur la même ligne : la comparaison centrale.
    [row] = [line for line in table.splitlines() if line.startswith("| chrF++ (depuis ASR)")]
    assert row.count("—") == 2  # renseignée pour asr et e2e, vide pour mt et tts
    assert "WER relâché" in table and "chrF++ fr→bm (replié)" in table

    with pytest.raises(SystemExit):
        baselines.main(common + ["asr", "--dataset", str(jeli), "--task", "translate",
                                 "--with-mt"])
    with pytest.raises(SystemExit):
        baselines.main(common + ["asr", "--dataset", str(jeli), "--task", "translate",
                                 "--kind", "nemo"])


def test_cli_asr_nemo(tmp_path, fakes):
    jeli = _jeli_like(tmp_path / "jeli")
    out = tmp_path / "results"
    baselines.main(["--out", str(out), "--device", "cpu", "asr", "--dataset", str(jeli),
                    "--model", "RobotsMali/soloni-114m-tdt-ctc-v1", "--kind", "nemo",
                    "--nemo-decoder", "ctc", "--with-mt"])
    asr = _report(out, "asr-cascade")
    assert asr["nom"] == "soloni-114m-tdt-ctc-v1 + nllb-200-distilled-600M"
    assert asr["modeles"]["kind"] == "nemo" and asr["modeles"]["nemo_decoder"] == "ctc"
    assert asr["modeles"]["language"] is None  # sans objet hors de Whisper


def test_cli_plafond_de_longueur(tmp_path, fakes):
    bayel = _bayelemabaga_like(tmp_path / "bayel")
    out = tmp_path / "results"
    baselines.main(["--out", str(out), "--device", "cpu", "mt", "--dataset", str(bayel),
                    "--plafond", "2"])
    mt = _report(out, "mt")
    assert mt["nom"].endswith("[plafond ×2]")
    assert mt["modeles"]["plafond_longueur"] == 2.0
    assert set(mt["sorties_coupees"]) == {"bm_fr", "fr_bm"}

    # Sans l'option : le plafond par défaut, sans le signaler dans le nom.
    sans_option = tmp_path / "defaut"
    baselines.main(["--out", str(sans_option), "--device", "cpu", "mt", "--dataset", str(bayel)])
    mt = _report(sans_option, "mt")
    assert "plafond" not in mt["nom"] and mt["modeles"]["plafond_longueur"] == 2.0

    # --plafond 0 : comme en phase 1.
    phase1 = tmp_path / "phase1"
    baselines.main(["--out", str(phase1), "--device", "cpu", "mt", "--dataset", str(bayel),
                    "--plafond", "0"])
    mt = _report(phase1, "mt")
    assert mt["nom"].endswith("[sans plafond]") and mt["modeles"]["plafond_longueur"] is None


def test_cli_decoupe_et_phrases_longues(tmp_path, fakes):
    bayel = _bayelemabaga_like(tmp_path / "bayel")
    out = tmp_path / "results"
    baselines.main(["--out", str(out), "--device", "cpu", "mt", "--dataset", str(bayel),
                    "--decoupe", "10", "--min-mots", "3"])
    mt = _report(out, "mt")
    assert mt["nom"].endswith("[découpe ≤10 mots]")
    # Seule « il est là » a au moins trois mots.
    assert mt["composition"]["n"] == 1 and mt["corpus"]["filtre"].endswith("3 mots")
    assert mt["decoupe_fr"]["max_mots"] == 10
