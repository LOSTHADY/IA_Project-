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


def test_le_notebook_se_compile():
    """« Tout exécuter » s'arrête à la première erreur : chaque cellule doit
    au moins être du Python valide, commandes shell et magies mises à part."""
    nb = json.loads((ROOT / "notebooks" / "phase3_finetuning.ipynb").read_text(encoding="utf-8"))
    for k, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        lines, shell = [], False
        for line in "".join(c["source"]).splitlines():
            stripped = line.lstrip()
            if shell:  # suite d'une commande shell coupée par « \ »
                shell = line.rstrip().endswith("\\")
                continue
            if stripped.startswith(("!", "%")):
                lines.append(line[:len(line) - len(stripped)] + "pass")
                shell = line.rstrip().endswith("\\")
            else:
                lines.append(line)
        compile("\n".join(lines), f"cellule {k}", "exec")


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

    # Whisper réentraîné après une évaluation : le rapport est mis de côté et
    # l'évaluation refaite ; plus récent que le modèle, il est gardé.
    import os
    poids = tmp_path / "whisper" / "model.safetensors"
    poids.parent.mkdir()
    poids.write_text("")
    res = tmp_path / "resultats"
    for name, date in (("asr-cascade-1", 1e9), ("asr-cascade-2", 3e9)):
        (res / f"{name}.json").write_text(json.dumps({"nom": name, "modeles": {
            "asr": str(poids.parent), "mt_in": "facebook/nllb-200-distilled-600M",
            "plafond_longueur": 2.0}}))
        (res / f"{name}.details.jsonl").write_text("")
        os.utime(res / f"{name}.json", (date, date))
    os.utime(poids, (2e9, 2e9))
    assert not ns["deja"]("asr-cascade-1")
    assert sorted(f.name for f in res.glob("asr-cascade-1*")) == [
        "asr-cascade-1.details.jsonl.ancien", "asr-cascade-1.json.ancien"]
    assert not ns["deja"]("asr-cascade-1") and ns["deja"]("asr-cascade-2")
    assert "réentraîné depuis : refait" in capsys.readouterr().out

    # Rapport tronqué (copie vers Drive interrompue) : écarté, sans erreur.
    (res / "asr-coupe.json").write_text('{"nom": "cou')
    assert not ns["deja"]("cou") and (res / "asr-coupe.json.tronque").exists()

    # Étape à refaire tant que sa sortie est plus ancienne que les poids en
    # service (ici le checkpoint de Whisper sur Drive).
    sortie = tmp_path / "chaine" / "cascade-1.json"
    sortie.parent.mkdir()
    assert not ns["a_jour"](str(tmp_path / "chaine" / "*.json"))  # rien encore
    sortie.write_text("{}")
    os.utime(sortie, (1e9, 1e9))
    assert not ns["a_jour"](str(sortie))
    os.utime(sortie, (4e9, 4e9))
    assert ns["a_jour"](str(sortie))
    # Avec code=True, le dernier commit de bambara_voice compte aussi : le
    # dépôt courant en a un, postérieur à 1e9 et antérieur à 4e9.
    assert ns["a_jour"](str(sortie), code=True)
    os.utime(sortie, (1.5e9, 1.5e9))
    os.utime(next((tmp_path / "drive" / "whisper-small-bm").glob("*/model.safetensors")),
             (1e9, 1e9))
    assert ns["a_jour"](str(sortie)) and not ns["a_jour"](str(sortie), code=True)

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


def test_nllb_avec_jeli_asr_en_plus(work, tmp_path):
    """NLLB v2 : Bayelemabaga plus les paires transcription-traduction d'un
    jeu façon Jeli-ASR (audio ignoré), doublons retirés, validation mêlée."""
    out = tmp_path / "nllb-v2"
    log = _train("finetune_nllb.py", "--dataset", str(work["text"]),
                 "--extra-dataset", str(work["speech"]), "--model", str(work["nllb"]),
                 "--direction", "bm2fr", "--save-steps", "2", "--eval-samples", "4",
                 "--output", str(out))
    assert f"+ {work['speech']} : " in log and "bambara=bam, français=french" in log
    # 16 + 14 paires, mais seulement 4 phrases différentes dans ces jeux factices.
    assert "Entraînement : 4 paires (26 doublons retirés) ; validation : 4" in log
    assert (out / "model.safetensors").exists()


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


def test_moyenne_des_poids_de_deux_versions(tmp_path):
    """v1 et v2 de NLLB (même architecture, la v2 reprise de la v1) : leur
    moyenne se recharge, traduit, et vaut exactement la moyenne des poids."""
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    sys.path.insert(0, str(ROOT / "scripts"))
    from moyenne_poids import moyenne

    v1 = tiny_models.nllb(tmp_path / "v1")
    v2 = tmp_path / "v2"
    import shutil
    shutil.copytree(v1, v2)
    poids = load_file(str(v2 / "model.safetensors"))
    save_file({k: t + 1.0 for k, t in poids.items()}, str(v2 / "model.safetensors"),
              metadata={"format": "pt"})
    (v2 / "optimizer.pt").write_bytes(b"x")  # état d'entraînement : pas copié

    out = moyenne(v1, v2, tmp_path / "moyenne", poids=0.25)
    assert not (out / "optimizer.pt").exists() and (out / "config.json").exists()
    m = load_file(str(out / "model.safetensors"))
    k = next(iter(m))
    assert torch.allclose(m[k], load_file(str(v1 / "model.safetensors"))[k] + 0.25)

    model = AutoModelForSeq2SeqLM.from_pretrained(out)
    tok = AutoTokenizer.from_pretrained(out, src_lang="bam_Latn")
    model.generate(**tok("n bɛ taa", return_tensors="pt"), max_new_tokens=3)

    autre = tiny_models.nllb(tmp_path / "autre")
    cfg = json.loads((autre / "config.json").read_text())
    cfg["d_model"] = 8
    (autre / "config.json").write_text(json.dumps(cfg))
    from transformers import M2M100Config, M2M100ForConditionalGeneration
    M2M100ForConditionalGeneration(M2M100Config(**{k: v for k, v in cfg.items()
                                                   if k not in ("architectures",)})
                                   ).save_pretrained(autre)
    with pytest.raises(ValueError, match="formes différentes|architectures différentes"):
        moyenne(v1, autre, tmp_path / "x")
