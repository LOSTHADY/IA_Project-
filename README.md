# Assistant vocal bambara

Un système de dialogue vocal en bambara (bamanankan) : on parle en bambara, il
répond en bambara. Le raisonnement se fait en français, le bambara vit aux
extrémités de la chaîne.

```
audio bm → ASR → [texte bm] → traduction → français → LLM
                                                        ↓
audio bm ←  TTS  ←  texte bm  ←  traduction  ←  français simple
```

> **Ce que ce projet n'est pas** : un LLM à qui l'on aurait appris le bambara.
> Les données écrites disponibles sont inférieures d'environ trois ordres de
> grandeur à ce qu'exigerait un pré-entraînement. C'est un projet de
> **transduction vocale**, et le LLM y est un composant interchangeable.
> Voir [`docs/METHODE.md`](docs/METHODE.md).

## Deux architectures, comparées

| | Maillons | Transcription bambara | Gabarits |
|---|---|---|---|
| **cascade** | ASR(bm) → MT → LLM → MT → TTS | oui | oui |
| **bout-en-bout** | ASR(bm→fr) → LLM → MT → TTS | non | non |

Jeli-ASR fournit pour chaque audio la transcription bambara *et* sa traduction
française : un seul Whisper multi-tâche donne les deux variantes avec le même
encodeur. La comparaison porte donc bien sur l'architecture, et sur rien
d'autre. C'est la contribution centrale du travail.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

L'inférence tourne sur CPU. L'entraînement se fait sur Colab (cf. `scripts/`).

## Utilisation

```bash
# Vérifier que les composants se chargent
python -m bambara_voice.cli check

# Un tour de parole, en partant du texte (sans ASR)
python -m bambara_voice.cli text "i ni sɔgɔma"

# Un tour complet, avec réponse audio
python -m bambara_voice.cli audio enregistrement.wav --out reponse.wav

# En variante bout-en-bout
python -m bambara_voice.cli audio enregistrement.wav --arch e2e

# Démo interactive
python app/gradio_app.py
```

Pour itérer sur le pipeline sans charger de LLM :
`--llm-backend echo` renvoie une réponse bouchon déterministe.

## Collecte du jeu de test

```bash
python app/collect_app.py                  # enregistrer, transcrire, valider
python scripts/export_testset.py           # énoncés validés -> testset.jsonl
```

L'application suit une séance dans l'ordre :
1. consentement et pseudonyme du locuteur ;
2. consignes de parole spontanée et phrases à lire ;
3. contrôle audio immédiat ;
4. transcription avec clavier ɛ/ɔ/ɲ/ŋ, puis validation par une seconde
   personne ;
5. suivi de la composition face aux cibles.

Protocole et formulaire de consentement : [`docs/COLLECTE.md`](docs/COLLECTE.md).

## Références zero-shot (phase 1)

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LOSTHADY/IA_Project-/blob/claude/chat-ia-bambara-dho8oa/notebooks/phase1_zero_shot.ipynb)

Ce que donnent les modèles publics tels quels, maillon par maillon, sur
Jeli-ASR et Bayelemabaga : Whisper et MMS pour l'ASR, NLLB pour la
traduction dans les deux sens, MMS-TTS avec une grille MOS. Le notebook
[`notebooks/phase1_zero_shot.ipynb`](notebooks/phase1_zero_shot.ipynb)
enchaîne tout sur un GPU Colab, en environ 30 minutes.

```bash
python -m eval.baselines asr --model facebook/mms-1b-all --kind ctc --target-lang bam --with-mt
python -m eval.baselines mt --dataset RobotsMaliAI/bayelemabaga
python -m eval.compare eval/results/zero-shot-*.json
```

## Évaluation

```bash
python scripts/prepare_testset.py data/testset/testset.jsonl   # contrôles
python -m eval.run_eval --testset data/testset/testset.jsonl --arch cascade
python -m eval.run_eval --testset data/testset/testset.jsonl --arch e2e
python -m eval.compare eval/results/*.json                     # tableau Markdown
```

Le rapport produit : WER/CER strict **et relâché**, chrF++/BLEU, propagation
d'erreurs entre l'ASR et le reste de la chaîne, latence par étape, RTF, et
taux de couverture des gabarits.

L'écart entre WER strict et WER relâché mesure la part d'erreur imputable à la
seule variation orthographique (ɛ/ɔ/ɲ/ŋ contre approximations ASCII) — sans
cette distinction, un modèle produisant un bambara correct mais en ASCII
paraît bien pire qu'il n'est.

## Fine-tuning et comparaison des architectures (phases 3-4)

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LOSTHADY/IA_Project-/blob/claude/chat-ia-bambara-dho8oa/notebooks/phase3_finetuning.ipynb)

Le notebook [`notebooks/phase3_finetuning.ipynb`](notebooks/phase3_finetuning.ipynb)
entraîne les modèles, puis compare cascade et bout-en-bout avec le protocole
de la phase 1. Les checkpoints vont sur Drive, et `--resume` reprend après
une coupure de session.

```bash
# Whisper multi-tâche : transcription bambara + traduction française
python scripts/finetune_whisper.py --dataset RobotsMali/jeli-asr \
    --model openai/whisper-small --task both --output ckpt/whisper-bm --resume

# NLLB, un modèle par sens
python scripts/finetune_nllb.py --direction bm2fr --output ckpt/nllb-bm2fr --resume
python scripts/finetune_nllb.py --direction fr2bm --output ckpt/nllb-fr2bm --resume

# Même Whisper, deux architectures : seule la tâche change
python -m eval.baselines --label cascade asr --model ckpt/whisper-bm \
    --with-mt --mt-model ckpt/nllb-bm2fr
python -m eval.baselines --label bout-en-bout asr --model ckpt/whisper-bm --task translate
```

Les spectrogrammes sont calculés à la volée : les précalculer occuperait des
dizaines de Go pour Jeli-ASR.

## Structure

```
bambara_voice/     pipeline d'inférence
  config.py        tous les identifiants de modèles, en un seul endroit
  normalize.py     normalisation et repli orthographique du bambara
  asr.py           Whisper (transcription bm ou traduction directe fr), MMS (CTC)
  mt.py            NLLB bambara <-> français
  llm.py           modèle de dialogue + contrainte de style traduisible
  tts.py           synthèse vocale bambara
  templates.py     réponses validées par un locuteur natif
  pipeline.py      orchestration + trace chronométrée de chaque tour
eval/              métriques, jeu de test, références zero-shot, rapports comparatifs
scripts/           fine-tuning Colab, préparation du jeu de test
app/               démo Gradio, application de collecte
notebooks/         notebooks Colab (phases 1 et 3-4)
docs/METHODE.md    partis pris méthodologiques
docs/COLLECTE.md   protocole de collecte, formulaire de consentement
data/templates.json  banque de gabarits (à remplir)
data/testset/      jeu de test (à collecter — chemin critique), consignes
```

## État

| Phase | | |
|---|---|---|
| 0 | Squelette, harnais d'évaluation, normalisation | fait |
| 1 | Références zero-shot | notebook prêt, à exécuter sur Colab |
| 2 | **Jeu de test, 200–500 énoncés** | outillage prêt, collecte à faire — chemin critique |
| 3 | Fine-tuning Whisper + NLLB | scripts testés de bout en bout sur petits modèles, notebook prêt |
| 4 | Comparaison des deux architectures | prête sur Jeli-ASR ; sur le jeu maison après la phase 2 |
| 5 | Assemblage CPU, quantisation, démo | à faire |

La phase 2 conditionne tout : sans jeu de test, aucun chiffre n'est
défendable. Voir [`docs/COLLECTE.md`](docs/COLLECTE.md) pour le protocole et
[`data/testset/README.md`](data/testset/README.md) pour le format.

## Tests

```bash
python -m pytest tests/ -q
```

Couvrent la logique déterministe (normalisation, gabarits, simplification,
métriques, collecte, corpus) sans rien télécharger. Les parties à modèles
(ASR, fine-tuning, évaluation) tournent sur de minuscules modèles aléatoires
construits localement (`tests/tiny_models.py`). `tests/test_finetune.py`
entraîne réellement Whisper et NLLB puis compare les deux architectures, en
environ une minute : c'est lui qui détecte une rupture d'API de
transformers avant qu'elle ne coûte une session Colab. Les vrais modèles se
vérifient avec `python -m bambara_voice.cli check`.

## Licence

À définir avant toute diffusion de données audio — la décision engage aussi
les locuteurs enregistrés.
