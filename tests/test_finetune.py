"""Chaîne complète des phases 3-4 sur des modèles minuscules : fine-tuning
Whisper (multi-tâche) et NLLB (deux sens), reprise après coupure, puis
évaluation cascade / bout-en-bout avec les vrais composants du pipeline.

Le plus lent des tests (~1 min sur CPU), mais le seul qui fasse tourner les
scripts d'entraînement : c'est lui qui détecte les ruptures d'API de
transformers ou datasets avant de perdre une session Colab.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")
pytest.importorskip("sentencepiece")
pytest.importorskip("datasets")

import tiny_models  # noqa: E402
from eval import baselines  # noqa: E402
from eval.compare import to_markdown  # noqa: E402


def _train(script: str, *args: str) -> str:
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--epochs", "1",
         "--batch-size", "4", "--grad-accum", "1", *args],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    return res.stdout


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    root = tmp_path_factory.mktemp("finetune")
    return {
        "speech": tiny_models.speech_dataset(root / "jeli"),
        "text": tiny_models.parallel_dataset(root / "bayel"),
        "whisper": tiny_models.whisper(root / "tiny-whisper"),
        "nllb": tiny_models.nllb(root / "tiny-nllb"),
        "root": root,
    }


@pytest.fixture(scope="module")
def whisper_ft(work):
    out = work["root"] / "whisper-ft"
    log = _train("finetune_whisper.py", "--dataset", str(work["speech"]),
                 "--model", str(work["whisper"]), "--task", "both", "--warmup", "0",
                 "--save-steps", "4", "--eval-samples", "3", "--num-workers", "0",
                 "--output", str(out))
    # 16 audios x 2 tâches, moins la validation prise dans train.
    assert "(transcribe + translate)" in log and "prise dans train" in log
    return out


@pytest.fixture(scope="module")
def nllb_ft(work):
    outs = {}
    for direction in ("bm2fr", "fr2bm"):
        outs[direction] = work["root"] / f"nllb-{direction}"
        log = _train("finetune_nllb.py", "--dataset", str(work["text"]),
                     "--model", str(work["nllb"]), "--direction", direction,
                     "--output", str(outs[direction]))
        assert "translation.bam" in log  # colonne imbriquée détectée
    return outs


def test_whisper_sauve_un_modele_rechargeable(whisper_ft):
    assert (whisper_ft / "model.safetensors").exists()
    assert list(whisper_ft.glob("checkpoint-*"))


def test_whisper_reprend_au_dernier_checkpoint(work, whisper_ft):
    log = _train("finetune_whisper.py", "--dataset", str(work["speech"]),
                 "--model", str(work["whisper"]), "--task", "both", "--warmup", "0",
                 "--save-steps", "4", "--eval-samples", "3", "--num-workers", "0",
                 "--epochs", "2", "--output", str(whisper_ft), "--resume")
    assert "Reprise depuis" in log and "checkpoint-" in log


def test_nllb_impose_la_langue_cible(nllb_ft):
    from transformers import AutoTokenizer

    for direction, lang in (("bm2fr", "fra_Latn"), ("fr2bm", "bam_Latn")):
        gen = json.loads((nllb_ft[direction] / "generation_config.json").read_text())
        tok = AutoTokenizer.from_pretrained(nllb_ft[direction])
        assert gen["forced_bos_token_id"] == tok.convert_tokens_to_ids(lang)


def test_evaluation_cascade_contre_bout_en_bout(work, whisper_ft, nllb_ft):
    out = work["root"] / "results"
    common = ["--out", str(out), "--device", "cpu"]
    baselines.main(common + ["--label", "cascade", "asr", "--dataset", str(work["speech"]),
                             "--model", str(whisper_ft), "--with-mt",
                             "--mt-model", str(nllb_ft["bm2fr"])])
    baselines.main(common + ["--label", "bout-en-bout", "asr",
                             "--dataset", str(work["speech"]),
                             "--model", str(whisper_ft), "--task", "translate"])
    baselines.main(common + ["mt", "--dataset", str(work["text"]),
                             "--bm2fr-model", str(nllb_ft["bm2fr"]),
                             "--fr2bm-model", str(nllb_ft["fr2bm"])])

    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*[0-9].json"))]
    by_arch = {r["architecture"]: r for r in reports}
    assert set(by_arch) == {"cascade", "e2e", "traduction"}
    assert "asr" in by_arch["cascade"] and "propagation_erreurs" in by_arch["cascade"]
    assert "asr" not in by_arch["e2e"] and "mt_in_depuis_asr" in by_arch["e2e"]
    assert list(by_arch["cascade"]["latence"]["par_etape_s"]) == ["asr", "mt_in"]

    table = to_markdown([by_arch["cascade"], by_arch["e2e"]])
    [row] = [line for line in table.splitlines() if line.startswith("| chrF++ (depuis ASR)")]
    assert "—" not in row  # les deux architectures sur la même mesure


def test_significativite_sur_les_rapports_d_eval_baselines(work, capsys):
    """Le test apparié lit les sorties ligne à ligne de vrais rapports."""
    from eval import significance

    reports = [str(p) for p in sorted((work["root"] / "results").glob("asr-*[0-9].json"))]
    significance.main(reports + ["--noms", "cascade", "bout-en-bout", "--n-boot", "50"])
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "| Métrique | n | cascade | bout-en-bout |"
    assert "| chrF++ (depuis ASR) | 6 |" in out and "écart" in out
