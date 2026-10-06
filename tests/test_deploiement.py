"""Phase 5 : fichier de configuration, conversion CTranslate2 int8, et chaîne
complète sur CPU (run_eval), sur des modèles minuscules construits en local."""

import dataclasses
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bambara_voice.config import (  # noqa: E402
    BAM, FRA, WHISPER_LANG_SLOT, MTConfig, build_config, load_config,
)


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- fichier de configuration -------------------------------------------------

def test_config_ne_change_que_ce_qui_est_precise(tmp_path):
    (tmp_path / "modeles" / "nllb").mkdir(parents=True)
    path = _write(tmp_path / "config.json", {
        "device": "cpu",
        "mt_in": {"model_id": "modeles/nllb", "backend": "ctranslate2"},
        "llm": {"backend": "echo"},
    })
    cfg = load_config(path)
    # Chemin relatif au fichier, parce qu'il y existe...
    assert cfg.mt_in.model_id == str(tmp_path / "modeles" / "nllb")
    # ... et le sens de traduction propre à mt_in est conservé.
    assert (cfg.mt_in.src_lang, cfg.mt_in.tgt_lang) == (BAM, FRA)
    assert (cfg.mt_out.src_lang, cfg.mt_out.tgt_lang) == (FRA, BAM)
    assert cfg.mt_out.backend == "transformers" and cfg.llm.backend == "echo"
    assert cfg.asr.language == WHISPER_LANG_SLOT


def test_config_identifiant_du_hub_laisse_tel_quel(tmp_path):
    cfg = load_config(_write(tmp_path / "c.json", {"tts": {"model_id": "facebook/mms-tts-bam"}}))
    assert cfg.tts.model_id == "facebook/mms-tts-bam"


def test_config_sert_aux_deux_architectures(tmp_path):
    path = _write(tmp_path / "c.json", {"asr": {"model_id": "w"}})
    assert load_config(path, "cascade").asr.task == "transcribe"
    e2e = load_config(path, "e2e")
    assert e2e.architecture == "e2e" and e2e.asr.task == "translate"
    assert build_config("e2e").asr.task == "translate"
    assert build_config("cascade", path).asr.model_id == "w"


def test_config_les_fautes_de_frappe_sont_des_erreurs(tmp_path):
    with pytest.raises(ValueError, match="sections inconnues"):
        load_config(_write(tmp_path / "a.json", {"mt": {}}))
    with pytest.raises(ValueError, match="champs inconnus dans 'asr'"):
        load_config(_write(tmp_path / "b.json", {"asr": {"modele": "x"}}))


# --- modèles convertis --------------------------------------------------------

ct2 = pytest.importorskip("ctranslate2")
pytest.importorskip("transformers")
pytest.importorskip("sentencepiece")
pytest.importorskip("datasets")

import tiny_models  # noqa: E402
from bambara_voice.asr import SpeechRecognizer  # noqa: E402
from bambara_voice.mt import Translator  # noqa: E402

AUDIO = np.zeros(16_000, dtype=np.float32)


class Spy:
    """Enregistre les appels faits au modèle CTranslate2."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def generate(self, feats, prompts, **kw):
        self.calls.append(prompts[0])
        self.kwargs = kw
        return self.inner.generate(feats, prompts, **kw)

    def detect_language(self, feats):
        self.calls.append("detect")
        return self.inner.detect_language(feats)

    def translate_batch(self, source, **kw):
        self.calls.append(kw["target_prefix"][0])
        return self.inner.translate_batch(source, **kw)


def _script(name: str):
    """Les scripts ne sont pas un paquet : on les importe par leur chemin."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cpu(tmp_path_factory):
    export_cpu = _script("export_cpu")
    root = tmp_path_factory.mktemp("deploiement")
    whisper = tiny_models.whisper(root / "whisper-bm")
    nllb = tiny_models.nllb(root / "nllb")
    export_cpu.main(["--out", str(root / "cpu"), "--whisper", str(whisper),
                     "--nllb-bm2fr", str(nllb), "--nllb-fr2bm", str(nllb)])
    config = json.loads((root / "cpu" / "config.json").read_text(encoding="utf-8"))
    # Le LLM n'est pas l'objet du test : bouchon déterministe.
    config["llm"] = {"backend": "echo"}
    _write(root / "cpu" / "config.json", config)
    return root


def test_export_ecrit_modeles_int8_et_configuration(cpu):
    config = json.loads((cpu / "cpu" / "config.json").read_text(encoding="utf-8"))
    assert config["asr"] == {"model_id": "whisper-bm", "backend": "ctranslate2",
                             "compute_type": "int8"}
    converted = cpu / "cpu" / "whisper-bm"
    assert (converted / "model.bin").exists() and (converted / "tokenizer.json").exists()


def _asr(cpu, **kw):
    cfg = load_config(cpu / "cpu" / "config.json")
    for k, v in kw.items():
        setattr(cfg.asr, k, v)
    rec = SpeechRecognizer(cfg.asr, "cpu")
    rec._ensure_loaded()
    rec._model = Spy(rec._model)
    return rec


def _prompt_tokens(rec) -> list[str]:
    return rec._processor.tokenizer.convert_ids_to_tokens(rec._model.calls[-1])


@pytest.mark.parametrize("task", ["transcribe", "translate"])
def test_ct2_whisper_impose_langue_et_tache(cpu, task):
    rec = _asr(cpu, task=task)
    res = rec.transcribe(AUDIO)
    assert _prompt_tokens(rec) == ["<|startoftranscript|>", "<|sw|>", f"<|{task}|>",
                                   "<|notimestamps|>"]
    assert res.language == ("fr" if task == "translate" else "bm")


def test_ct2_whisper_meme_plafond_de_tokens_que_transformers(cpu):
    # CTranslate2 s'arrête à max_length // 2 tokens générés.
    rec = _asr(cpu, max_new_tokens=50)
    rec.transcribe(AUDIO)
    assert rec._model.kwargs["max_length"] == 100
    rec = _asr(cpu, max_new_tokens=400)  # au-delà du contexte : plafonné
    rec.transcribe(AUDIO)
    assert rec._model.kwargs["max_length"] == 448


def test_ct2_whisper_detection_automatique(cpu):
    rec = _asr(cpu, language=None)

    # CTranslate2 ne détecte la langue que sur un vocabulaire multilingue
    # complet (≥ 51 865 tokens) : on simule sa réponse.
    def detect(feats):
        rec._model.calls.append("detect")
        return [[("<|fr|>", 0.8), ("<|sw|>", 0.1)]]

    rec._model.detect_language = detect
    rec.transcribe(AUDIO)
    assert rec._model.calls[0] == "detect"
    # La langue détectée est bien celle qui entre dans le préfixe.
    assert _prompt_tokens(rec)[1:3] == ["<|fr|>", "<|transcribe|>"]


def test_ct2_whisper_langue_inconnue(cpu, caplog):
    rec = _asr(cpu, language="bm")
    with caplog.at_level(logging.WARNING):
        rec.transcribe(AUDIO)
    assert _prompt_tokens(rec) == ["<|startoftranscript|>", "<|notimestamps|>"]
    rec = _asr(cpu, language="bm", task="translate")
    with pytest.raises(ValueError, match="traduction directe impossible"):
        rec.transcribe(AUDIO)


def test_ct2_nllb_impose_la_langue_cible(cpu):
    cfg = load_config(cpu / "cpu" / "config.json")
    for mt, lang in ((cfg.mt_in, FRA), (cfg.mt_out, BAM)):
        tr = Translator(mt, "cpu")
        tr._ensure_loaded()
        tr._model = Spy(tr._model)
        assert isinstance(tr.translate("i ni ce"), str)
        assert tr._model.calls == [[lang]]


def test_plafond_de_longueur_calcul():
    tr = Translator(MTConfig(max_new_tokens=256, max_length_ratio=2.0, max_length_margin=10))
    assert tr.max_tokens(20) == 50
    assert tr.max_tokens(200) == 256  # jamais au-delà de max_new_tokens
    assert Translator(MTConfig(max_new_tokens=256)).max_tokens(20) == 256


class MTSpy:
    """Budget de longueur passé au modèle, et longueur effectivement produite."""

    def __init__(self, inner):
        self.inner = inner

    def generate(self, **kw):  # transformers
        self.budget = kw["max_new_tokens"]
        out = self.inner.generate(**kw)
        self.produced = out.shape[1] - 1  # sans le token de début du décodeur
        return out

    def translate_batch(self, source, **kw):  # CTranslate2
        self.budget = kw["max_decoding_length"]
        out = self.inner.translate_batch(source, **kw)
        self.produced = len(out[0].hypotheses[0])
        return out


@pytest.mark.parametrize("backend", ["transformers", "ctranslate2"])
def test_plafond_de_longueur_relatif_a_la_source(cpu, backend):
    """Le plafond borne la sortie et la coupure est signalée, sur les deux
    moteurs."""
    cfg = load_config(cpu / "cpu" / "config.json").mt_out
    if backend == "transformers":
        cfg = dataclasses.replace(cfg, model_id=str(cpu / "nllb"), backend="transformers")
    budgets = {}
    for ratio in (None, 1.0):
        tr = Translator(dataclasses.replace(cfg, max_new_tokens=40, max_length_ratio=ratio,
                                            max_length_margin=2), "cpu")
        tr._ensure_loaded()
        tr._model = MTSpy(tr._model)
        tr.translate("je vais au marché")
        budgets[ratio] = tr._model.budget
        assert tr._model.produced <= tr._model.budget
        assert tr.last_capped == (tr._model.produced >= tr._model.budget)
    n_source = len(tr._tokenizer("je vais au marché")["input_ids"])
    assert budgets[None] == 40
    assert budgets[1.0] == n_source + 2 < 40
    # Un seul jeton permis, le token de langue : la sortie est forcément coupée.
    one = Translator(dataclasses.replace(cfg, max_new_tokens=1), "cpu")
    one.translate("je vais au marché")
    assert one.last_capped
    one.translate("")
    assert not one.last_capped


def test_cli_tour_texte_avec_la_configuration(cpu, capsys):
    from bambara_voice import cli

    assert cli.main(["--config", str(cpu / "cpu" / "config.json"), "text", "n bɛ taa"]) == 0
    trace = json.loads(capsys.readouterr().out)
    assert trace["reply_source"] == "mt" and set(trace["timings"]) == {"mt_in", "llm", "mt_out"}


def test_chaine_complete_sur_cpu(cpu):
    """Échantillon de corpus exporté, puis run_eval avec la configuration de
    déploiement : c'est ainsi que se mesure la latence de bout en bout."""
    from eval.dataset import load_testset
    from eval.run_eval import evaluate

    sample = _script("export_corpus_sample")
    corpus = tiny_models.speech_dataset(cpu / "jeli")
    sample.main(["--dataset", str(corpus), "--limit", "3", "--out", str(cpu / "echantillon")])
    testset = cpu / "echantillon" / "testset.jsonl"
    items = load_testset(testset)
    assert len(items) == 3 and all(i.audio_path(testset.parent).exists() for i in items)

    config = cpu / "cpu" / "config.json"
    cascade = evaluate(testset, "cascade", cpu / "res", config_path=config)
    assert cascade["modeles"]["asr"].startswith("ctranslate2:")
    assert set(cascade["latence"]["par_etape_s"]) == {"asr", "mt_in", "llm", "mt_out"}
    assert "propagation_erreurs" in cascade and cascade["appareil"] == "cpu"

    e2e = evaluate(testset, "e2e", cpu / "res", config_path=config)
    assert set(e2e["latence"]["par_etape_s"]) == {"asr", "llm", "mt_out"}

    # Les rapports de run_eval se prêtent au test apparié.
    from eval.significance import bootstrap, load_series

    [c_path] = (cpu / "res").glob("cascade-*[0-9].json")
    [e_path] = (cpu / "res").glob("e2e-*[0-9].json")
    _, c_series = load_series(c_path)
    _, e_series = load_series(e_path)
    assert {"WER strict", "chrF++ (depuis ASR)"} <= set(c_series)
    res = bootstrap([c_series["chrF++ (depuis ASR)"], e_series["chrF++ (depuis ASR)"]], n_boot=20)
    assert res["n"] == 3 and res["unite"] == "enonce"  # échantillon sans locuteurs

    # Et à l'analyse d'erreurs.
    from eval.analysis import analyse_report

    assert {"asr", "mt_depuis_asr"} <= set(analyse_report(c_path)[1])
    assert "e2e" in analyse_report(e_path)[1]
