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
                     "--save-steps", "1", "--eval-samples", "1",
                     "--output", str(outs[direction]))
        assert "translation.bam" in log  # colonne imbriquée détectée
        # Checkpoints réguliers : une session coupée ne perd pas tout.
        checkpoints = list(outs[direction].glob("checkpoint-*"))
        assert checkpoints
        # Adafactor (seconds moments factorisés), pas AdamW : NLLB doit
        # tenir dans la mémoire d'un T4.
        import torch
        state = torch.load(checkpoints[0] / "optimizer.pt", weights_only=True)["state"]
        assert any("exp_avg_sq_row" in v for v in state.values())
        assert not any("exp_avg" in v for v in state.values())
    return outs


def test_whisper_sauve_un_modele_rechargeable(whisper_ft):
    assert (whisper_ft / "model.safetensors").exists()
    assert list(whisper_ft.glob("checkpoint-*"))


def test_le_notebook_n_a_pas_d_apostrophe_dans_les_commandes():
    # IPython ne remplace pas $VAR quand une apostrophe suit sur la ligne
    # (il la croit entre guillemets simples) : « --label "NLLB d'origine" »
    # a ainsi envoyé un rapport hors de $OUT, le 7 octobre 2026.
    nb = json.loads((ROOT / "notebooks" / "phase3_finetuning.ipynb").read_text(encoding="utf-8"))
    shell = [line for c in nb["cells"] if c["cell_type"] == "code"
             for line in "".join(c["source"]).splitlines() if line.lstrip().startswith("!")]
    assert shell and not [line for line in shell if "'" in line]


def test_le_notebook_estime_la_duree_d_apres_l_essai(whisper_ft, capsys):
    """La cellule « durée prévue » du notebook Colab lit trainer_state.json,
    que le script écrit en fin d'entraînement."""
    nb = json.loads((ROOT / "notebooks" / "phase3_finetuning.ipynb").read_text(encoding="utf-8"))
    [cell] = [c for c in nb["cells"] if "Durée prévue" in "".join(c["source"])]
    code = "".join(cell["source"]).replace("/content/essai", str(whisper_ft))
    exec(code, {"EPOCHS_WHISPER": 1, "EPOCHS_NLLB": 1})
    out = capsys.readouterr().out
    assert "s par étape" in out and "Whisper, 3 époque(s)" in out


def test_le_notebook_evalue_le_dernier_checkpoint(whisper_ft, tmp_path, capsys):
    """Session Colab coupée : sans modèle final, le notebook évalue le dernier
    checkpoint ; sans rien, le modèle d'origine s'il y en a un."""
    import shutil

    nb = json.loads((ROOT / "notebooks" / "phase3_finetuning.ipynb").read_text(encoding="utf-8"))
    [cell] = [c for c in nb["cells"] if "".join(c["source"]).startswith("# Modèle final")]
    ns: dict = {"CKPT": str(tmp_path / "drive"), "OUT": str(tmp_path / "resultats")}
    exec("".join(cell["source"]), ns)
    modele = ns["modele"]

    assert modele(str(whisper_ft)) == str(whisper_ft)  # modèle final
    coupe = tmp_path / "coupe"
    last = sorted(whisper_ft.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))[-1]
    shutil.copytree(last, coupe / last.name)
    assert modele(str(coupe)).endswith(last.name)
    assert "Entraînement inachevé" in capsys.readouterr().out
    assert modele(str(tmp_path / "rien"), repli="nllb") == "nllb"
    with pytest.raises(FileNotFoundError):
        modele(str(tmp_path / "rien"))

    # Après une coupure : Whisper sur Drive (checkpoint), NLLB pas encore là.
    shutil.copytree(last, tmp_path / "drive" / "whisper-small-bm" / last.name)
    whisper, bm2fr, fr2bm = ns["chemins"]()
    assert whisper.endswith(last.name)
    assert bm2fr == fr2bm == "facebook/nllb-200-distilled-600M"

    # Une évaluation déjà faite n'est pas refaite.
    (tmp_path / "resultats").mkdir()
    (tmp_path / "resultats" / "asr-e2e-1.json").write_text(json.dumps({"nom": "bout-en-bout affiné"}))
    assert ns["deja"]("bout-en-bout affiné") and not ns["deja"]("cascade affinée")

    # Le checkpoint se charge comme un modèle : l'évaluation peut s'en servir.
    from bambara_voice.asr import SpeechRecognizer
    from bambara_voice.config import ASRConfig
    import numpy as np

    rec = SpeechRecognizer(ASRConfig(model_id=str(coupe / last.name), max_new_tokens=5))
    assert isinstance(rec.transcribe(np.zeros(16_000, dtype=np.float32)).text, str)


def test_whisper_reprend_au_dernier_checkpoint_complet(work, whisper_ft):
    import shutil
    from transformers.trainer_utils import get_last_checkpoint

    # Copie vers Drive interrompue : le checkpoint le plus récent n'a pas son
    # plan de taux d'apprentissage. Le Trainer repartirait avec un plan neuf.
    last = Path(get_last_checkpoint(str(whisper_ft)))
    incomplete = whisper_ft / "checkpoint-99999"
    shutil.copytree(last, incomplete)
    (incomplete / "scheduler.pt").unlink()
    try:
        log = _train("finetune_whisper.py", "--dataset", str(work["speech"]),
                     "--model", str(work["whisper"]), "--task", "both", "--warmup", "0",
                     "--save-steps", "4", "--eval-samples", "3", "--num-workers", "0",
                     "--epochs", "2", "--output", str(whisper_ft), "--resume")
    finally:
        shutil.rmtree(incomplete, ignore_errors=True)
    assert "Checkpoint incomplet, ignoré : checkpoint-99999 (manque : scheduler.pt)" in log
    assert f"Reprise depuis {last}" in log


def test_reprise_sans_checkpoint_complet(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from reprise import last_complete_checkpoint

    assert last_complete_checkpoint(tmp_path / "absent") is None
    ck = tmp_path / "checkpoint-10"
    ck.mkdir()
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
        (ck / name).write_text("{}")
    assert last_complete_checkpoint(tmp_path) is None  # pas de poids
    (ck / "model.safetensors").write_text("")
    (tmp_path / "checkpoint-20").mkdir()  # vide : copie à peine commencée
    assert last_complete_checkpoint(tmp_path) == str(ck)


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
