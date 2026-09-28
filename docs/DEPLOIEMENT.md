# Déploiement sur CPU (phase 5)

Objectif : faire tourner toute la chaîne sur un ordinateur ordinaire, sans
GPU, en push-to-talk. Le principe est posé dans [`METHODE.md`](METHODE.md),
§2 : l'entraînement a lieu sur GPU, l'inférence sur CPU, avec des modèles
quantifiés.

| Maillon | Moteur sur CPU | Format |
|---|---|---|
| ASR (Whisper) | CTranslate2 | int8 |
| Traduction (NLLB, deux sens) | CTranslate2 | int8 |
| LLM | llama.cpp | GGUF quantifié (Q4) |
| Synthèse (MMS-TTS, VITS) | PyTorch | float32 (pas d'équivalent CTranslate2) |

## 1. Convertir les modèles affinés

À la suite du fine-tuning (phase 3), sur Colab ou sur la machine cible :

```bash
pip install ctranslate2
python scripts/export_cpu.py --out modeles-cpu \
    --whisper ckpt/whisper-small-bm \
    --nllb-bm2fr ckpt/nllb-bm2fr --nllb-fr2bm ckpt/nllb-fr2bm
```

Le script convertit chaque modèle en int8 et copie à côté son tokenizer et
son extracteur. Il écrit aussi `modeles-cpu/config.json`, qui décrit le
déploiement et que la démo, la ligne de commande et l'évaluation lisent
avec `--config`. Le dossier est exclu de git : c'est un produit régénérable.

## 2. Le LLM en GGUF

Le LLM est une boîte noire figée (`METHODE.md`, §1) : on déploie **le même
modèle** que celui de l'évaluation, simplement quantifié.

```bash
pip install llama-cpp-python
```

Télécharger une version GGUF de `google/gemma-3-1b-it` depuis le Hub
(chercher « gemma-3-1b-it GGUF »), en quantification Q4_K_M ou Q4_0, et la
placer dans `modeles-cpu/`. Puis relancer la conversion avec
`--llm-gguf nom-du-fichier.gguf`, ou ajouter à `config.json` :

```json
"llm": {"backend": "llamacpp", "gguf_path": "nom-du-fichier.gguf"}
```

Les chemins relatifs se lisent depuis le dossier du fichier de
configuration.

## 3. Vérifier ce que la quantification coûte en qualité

L'int8 n'est pas gratuit. On mesure la perte sur **le même échantillon**
qu'aux phases 1 et 3, en comparant les deux moteurs :

```bash
python -m eval.baselines --device cpu --label "cascade float32" \
    asr --model ckpt/whisper-small-bm --with-mt --mt-model ckpt/nllb-bm2fr
python -m eval.baselines --device cpu --label "cascade int8" \
    asr --model modeles-cpu/whisper-small-bm --with-mt --mt-model modeles-cpu/nllb-bm2fr \
    --backend ctranslate2
python -m eval.compare eval/results/*.json
```

L'écart de WER et de chrF++ entre les deux colonnes est le prix de la
quantification. Il se rapporte dans le mémoire, à côté du gain de latence.

## 4. Mesurer la latence de bout en bout sur la machine cible

La latence n'a de sens que sur la machine de déploiement : les chiffres de
Colab (GPU) ne disent rien du CPU.

```bash
python scripts/export_corpus_sample.py --limit 50 --out data/echantillons/jeli-test
python -m eval.run_eval --testset data/echantillons/jeli-test/testset.jsonl \
    --config modeles-cpu/config.json --arch cascade --synthesize
python -m eval.run_eval --testset data/echantillons/jeli-test/testset.jsonl \
    --config modeles-cpu/config.json --arch e2e --synthesize
```

Le rapport donne le temps moyen par étape (ASR, MT, LLM, MT, TTS), le total
et le RTF. Une fois le jeu de test maison prêt (phase 2), refaire la mesure
dessus.

## 5. Lancer la démo

```bash
python app/gradio_app.py --config modeles-cpu/config.json
python app/gradio_app.py --config modeles-cpu/config.json --arch e2e
```

## 6. Ordres de grandeur mesurés

Mesures faites sur un Xeon à 4 cœurs, 2,1 GHz (AVX-512, AMX), avec des
modèles **aux dimensions exactes** de whisper-small (242 M de paramètres)
et de NLLB-600M (615 M), mais à poids aléatoires. Seuls les temps de calcul
sont significatifs. La longueur des sorties est imposée pour que les deux
moteurs fassent le même travail. Meilleur de trois essais, 4 threads.

| | PyTorch fp32 | CTranslate2 int8 | Gain |
|---|---|---|---|
| Taille de Whisper-small | 923 Mo | 236 Mo | ×3,9 |
| Taille de NLLB-600M | 2,3 Go | 593 Mo | ×3,9 |
| Whisper, encodeur (toujours 30 s d'audio) | 1,00 s | 0,94 s | ≈ ×1 |
| Whisper, décodage de 30 tokens | 1,33 s | 0,59 s | ×2,3 |
| Whisper, énoncé de 5 s au total | 1,9 à 2,3 s | 1,5 s | ×1,3 à 1,5 |
| NLLB, 20 → 25 tokens, glouton | 1,35 s | 0,70 s | ×1,9 |
| NLLB, 20 → 25 tokens, faisceau 4 (défaut) | 3,06 s | 1,22 s | ×2,5 |

Ce qu'il faut en retenir :

- **La traduction est le premier poste gagné.** Elle intervient deux fois
  par tour en cascade : environ 3,7 s gagnées par tour à faisceau 4.
- **L'encodeur de Whisper ne profite pas de l'int8** sur ce processeur, et
  il traite toujours une fenêtre de 30 s, même pour un énoncé de 2 s. Sur
  des énoncés courts, c'est lui qui fixe le plancher de l'ASR.
- **Sur un processeur sans AVX-512** (la plupart des portables), les
  rapports peuvent être différents : d'où la mesure du §4 sur la machine
  cible.
- **Le faisceau de NLLB est un levier** : passer de 4 à 1 divise le temps
  de traduction par environ 1,7 sous CTranslate2. On l'ajuste dans
  `config.json` (`"mt_out": {"beam_size": 1}`), après avoir mesuré la perte
  de chrF++ (§3).
- **Les gabarits sont aussi un levier de latence.** Un tour résolu par
  gabarit n'appelle ni le LLM ni la traduction sortante. Le taux de
  couverture des gabarits réduit donc aussi le temps de réponse moyen.
- **CTranslate2 génère au plus `max_length // 2` tokens avec Whisper**
  (mesuré : 40 → 20, 448 → 224), quelle que soit la longueur du préfixe.
  Le code en tient compte pour garder le même plafond qu'avec transformers.
