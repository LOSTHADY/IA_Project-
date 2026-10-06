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

Tant que les modèles ne sont pas affinés (phase 3), utiliser la
configuration zero-shot : `--config configs/zero-shot.json`. Elle prend MMS
pour la reconnaissance, car Whisper non affiné n'écrit pas le bambara (WER
160 %, `docs/RESULTATS.md`). La qualité reste celle des modèles non
affinés : médiocre. Premier lancement : environ 8 Go de modèles à
télécharger.

```bash
# Vérifier que les composants se chargent
python -m bambara_voice.cli --config configs/zero-shot.json check
python -m bambara_voice.cli --config configs/zero-shot.json text "i ni sɔgɔma"
python app/gradio_app.py --config configs/zero-shot.json

# Configuration par défaut (Whisper) : pour les modèles affinés
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

Sans GPU ni rien à surveiller : le workflow GitHub Actions
[« Phase 1 — références zero-shot »](.github/workflows/phase1-zero-shot.yml)
fait les mêmes mesures sur les processeurs de GitHub, gratuitement pour un
dépôt public. Il faut compter quelques heures. Le tableau et les
intervalles de confiance s'affichent dans le résumé de l'exécution ; les
rapports et les audios à noter sont dans les artefacts.

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

```bash
# Un écart est-il réel ? IC à 95 % et test apparié (bootstrap). Sur le jeu
# maison, on tire les locuteurs plutôt que les énoncés (METHODE.md, §5).
python -m eval.significance eval/results/*.json --noms cascade e2e
# Comment chaque système se trompe, et sur quels sous-groupes
# (lu / spontané, code-switching, conditions, genre, locuteur)
python -m eval.analysis eval/results/*.json
# MOS de la synthèse, à partir des grilles remplies par les auditeurs
python -m eval.mos eval/results/tts-*/mos*.csv
```

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

## Déploiement sur CPU (phase 5)

```bash
# Whisper et NLLB en int8 (CTranslate2), et un fichier de configuration
python scripts/export_cpu.py --out modeles-cpu --whisper ckpt/whisper-bm \
    --nllb-bm2fr ckpt/nllb-bm2fr --nllb-fr2bm ckpt/nllb-fr2bm

python app/gradio_app.py --config modeles-cpu/config.json         # démo
python -m eval.run_eval --config modeles-cpu/config.json \
    --testset data/echantillons/jeli-test/testset.jsonl --synthesize  # latence
```

Modèles 4 fois plus légers. La traduction va 2 à 2,5 fois plus vite ;
Whisper 1,3 à 1,5 fois sur des énoncés courts, parce que son encodeur ne
profite pas de l'int8. Recette complète, perte de qualité à mesurer et
ordres de grandeur : [`docs/DEPLOIEMENT.md`](docs/DEPLOIEMENT.md).

## Structure

```
bambara_voice/     pipeline d'inférence
  config.py        identifiants de modèles et fichier de déploiement (--config)
  normalize.py     normalisation et repli orthographique du bambara
  asr.py           Whisper (transcription bm ou traduction directe fr), MMS (CTC)
  mt.py            NLLB bambara <-> français
  llm.py           modèle de dialogue + contrainte de style traduisible
  tts.py           synthèse vocale bambara
  templates.py     réponses validées par un locuteur natif
  pipeline.py      orchestration + trace chronométrée de chaque tour
eval/              métriques, jeu de test, références zero-shot, rapports comparatifs
scripts/           fine-tuning Colab, jeu de test, conversion CPU
app/               démo Gradio, application de collecte
notebooks/         notebooks Colab (phases 1 et 3-4)
docs/METHODE.md    partis pris méthodologiques
docs/COLLECTE.md   protocole de collecte, formulaire de consentement
docs/DEPLOIEMENT.md  déploiement CPU : conversion int8, GGUF, mesures
docs/RESULTATS.md  chiffres mesurés, phase par phase
data/templates.json  banque de gabarits (à remplir)
data/testset/      jeu de test (à collecter — chemin critique), consignes
```

## État

| Phase | | |
|---|---|---|
| 0 | Squelette, harnais d'évaluation, normalisation | fait |
| 1 | Références zero-shot | **fait** — [`docs/RESULTATS.md`](docs/RESULTATS.md) |
| 2 | **Jeu de test, 200–500 énoncés** | outillage prêt, collecte à faire — chemin critique |
| 3 | Fine-tuning Whisper + NLLB | scripts testés de bout en bout sur petits modèles, notebook prêt |
| 4 | Comparaison des deux architectures | prête sur Jeli-ASR ; sur le jeu maison après la phase 2 |
| 5 | Assemblage CPU, quantisation, démo | chaîne complète validée avec les vrais modèles (zero-shot, 33 s par tour en fp32) ; int8 à mesurer sur la machine cible |

La phase 2 conditionne tout : sans jeu de test, aucun chiffre n'est
défendable. Voir [`docs/COLLECTE.md`](docs/COLLECTE.md) pour le protocole et
[`data/testset/README.md`](data/testset/README.md) pour le format.

## Tests

```bash
for f in tests/test_*.py; do python -m pytest "$f" -q; done
```

Un processus par module : lancée d'un bloc (`pytest tests/`), la suite
plante par intermittence sur une corruption du tas native, toujours pendant
l'utilisation de CTranslate2, et jamais module par module. La cause n'est
pas identifiée. Les tests sont lancés automatiquement à chaque push par
GitHub Actions ([`tests.yml`](.github/workflows/tests.yml)).

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
