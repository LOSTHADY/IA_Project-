# Méthode

Ce document fixe les partis pris. Il est écrit pour être repris tel quel dans
le chapitre « méthodologie » du mémoire.

## 1. Ce que le projet n'est pas

Ce n'est **pas** un projet d'apprentissage du bambara par un LLM. Faire
acquérir une langue à un modèle par pré-entraînement continu demande de
l'ordre de 10⁹–10¹⁰ tokens dans cette langue. Le bambara écrit numérisé et
exploitable représente quelques millions de tokens : il manque environ trois
ordres de grandeur. C'est une limite structurelle, pas une limite de matériel.

Le projet est donc un projet de **transduction vocale** : le bambara vit aux
extrémités de la chaîne, le raisonnement se fait en français, le LLM est un
composant remplaçable. L'essentiel de la qualité perçue vient de l'ASR et de
la traduction — c'est là que va l'effort expérimental.

Conséquence pratique : le LLM est figé dès le départ et traité comme une
boîte noire. Aucun temps n'est investi à en comparer plusieurs.

## 2. Séparation entraînement / déploiement

- **Entraînement** : GPU, sur Colab. Aucun fine-tuning n'est réalisable sur CPU.
- **Déploiement** : CPU uniquement, modèles quantisés — CTranslate2 (int8)
  pour Whisper et NLLB, `llama.cpp` (GGUF) pour le LLM. Recette et mesures :
  [`DEPLOIEMENT.md`](DEPLOIEMENT.md).

Cette séparation est assumée et doit être énoncée explicitement : la
contrainte « sans GPU » porte sur l'inférence, pas sur l'entraînement.

## 3. Les deux architectures comparées

C'est la contribution centrale du travail.

**Variante A — cascade**

```
audio bm → ASR(bm) → texte bm → MT(bm→fr) → fr → LLM → fr → MT(fr→bm) → TTS
```

**Variante B — bout-en-bout à l'entrée**

```
audio bm → ASR(translate) ────────────────→ fr → LLM → fr → MT(fr→bm) → TTS
```

Jeli-ASR fournit, pour chaque audio, **la transcription bambara et sa
traduction française**. On peut donc entraîner un unique modèle Whisper
multi-tâche produisant l'une ou l'autre cible (`scripts/finetune_whisper.py
--task both`). Les deux variantes partagent alors exactement le même encodeur,
ce qui rend la comparaison méthodologiquement propre : la seule différence est
architecturale.

**Token de langue.** Whisper n'a pas de token pour le bambara ; on détourne
celui du swahili, à l'entraînement comme à l'inférence
(`WHISPER_LANG_SLOT`). C'est ce token, suivi du token de tâche, qui fait
produire au même modèle soit du bambara, soit du français. Si l'inférence
n'utilisait pas le même token, la tâche ne serait plus forcée, et les deux
variantes produiraient la même sortie sans que rien ne le signale.

**Pas de référence zero-shot pour B.** La tâche `translate` de Whisper
d'origine ne produit que de l'anglais. La variante bout-en-bout n'existe
qu'après fine-tuning, alors que la cascade a une référence zero-shot
(phase 1).

**Compromis à documenter** : la variante B supprime un maillon et une source
d'erreurs, mais ne produit aucune transcription bambara. Elle prive donc le
système de l'affichage du texte bambara *et* de l'appariement de gabarits
(levier 3), qui opère sur le bambara. Ce n'est pas un détail : c'est
précisément le genre d'arbitrage qu'un mémoire doit exposer plutôt que
trancher silencieusement.

## 4. Les trois leviers de qualité

### Levier 1 — supprimer un maillon (variante B)

Chaque maillon multiplie les erreurs. Passer de 4 à 3 modèles en série réduit
la propagation et la latence. Mesuré par `propagation_erreurs` dans le rapport
d'évaluation.

### Levier 2 — contraindre la sortie du LLM

Un LLM répond spontanément en français élégant : phrases longues,
subordonnées, vocabulaire abstrait. NLLB fine-tuné sur ~47k paires s'effondre
dessus. On impose donc des phrases déclaratives courtes et un vocabulaire
concret, par prompt système **et** par post-traitement dur
(`bambara_voice.llm.simplify_for_translation`) — un prompt seul n'est jamais
respecté à 100 %.

Coût nul, effet important sur la fidélité de la traduction sortante. À
quantifier : chrF++ fr→bm avec et sans contrainte, sur le même jeu.

### Levier 3 — réponses gabarits validées

Sur les intentions les plus fréquentes, la réponse bambara est pré-traduite et
**validée par un locuteur natif** (`data/templates.json`). L'appariement se
fait par similarité de Dice sur trigrammes de caractères, calculée sur la
forme repliée : robuste aux erreurs d'ASR, et entièrement explicable — aucune
boîte noire supplémentaire dans la chaîne.

Ce n'est pas de la triche, **à condition de le déclarer** : le rapport
d'évaluation publie le taux de couverture des gabarits et le taux de repli sur
la MT. C'est cette proportion, pas la démo, qui constitue le résultat.

## 5. Protocole de mesure

| Maillon | Métrique | Justification |
|---|---|---|
| ASR | WER / CER **strict** | comparabilité avec la littérature |
| ASR | WER / CER **relâché** | calculé sur la forme repliée (ɛ→e, ɔ→o, ɲ→ny, ŋ→ng) |
| ASR | **écart orthographique** | strict − relâché : part d'erreur due à la seule graphie |
| MT | **chrF++** (principal) | niveau caractère, fiable en faible ressource et morphologie riche |
| MT | BLEU (secondaire) | comparabilité uniquement |
| Chaîne | **propagation d'erreurs** | chrF++ depuis le texte de référence − depuis l'ASR |
| TTS | MOS sur échantillon | aucune métrique automatique ne vaut ici |
| Système | latence par étape, RTF | contrainte de déploiement CPU |

Le **WER relâché** mérite un paragraphe dans le mémoire : sans lui, un modèle
qui produit un bambara correct mais en ASCII paraît bien pire qu'il n'est, et
l'on conclut à tort sur la qualité acoustique.

La **propagation d'erreurs** est mesurée en évaluant deux fois la même
traduction — depuis la transcription de référence, puis depuis la sortie de
l'ASR. La différence isole le coût de l'ASR indépendamment de la qualité de la
traduction.

Deux règles garantissent que les chiffres se comparent d'une phase à
l'autre :

- **Le test ne sert qu'à mesurer.** Le meilleur checkpoint est choisi sur
  une validation, prise dans `train` si le jeu n'en fournit pas. Choisir sur
  le test puis y rapporter le score le rendrait optimiste.
- **Même échantillon partout.** Références zero-shot, modèles affinés,
  cascade et bout-en-bout sont évalués sur les mêmes énoncés : même corpus,
  même partition, même graine (`eval.baselines`). Chaque rapport consigne
  ces paramètres ainsi que les versions des bibliothèques.

**Significativité.** Aucun écart n'est annoncé sans son intervalle de
confiance. `eval.significance` rééchantillonne les énoncés (bootstrap, 1 000
tirages) et recalcule chaque score au niveau du corpus. Pour deux systèmes,
le test est apparié : les mêmes tirages servent aux deux, ce que permet
la règle précédente (Koehn, 2004). Un écart dont l'IC contient zéro n'est
pas établi, quelle que soit sa taille apparente. Pour la synthèse, le MOS
(`eval.mos`) a lui aussi son IC, par bootstrap sur les phrases, car les
notes d'une même phrase ne sont pas indépendantes.

**Tirer les locuteurs, pas les énoncés.** Le même raisonnement vaut pour le
jeu maison : chaque locuteur y enregistre une trentaine d'énoncés, qui
partagent sa voix, son débit et son micro. Les tirer un par un reviendrait à
supposer 300 voix indépendantes là où il n'y en a que 10, et donnerait des
IC trop étroits. Sur ce jeu, `eval.significance` tire donc des locuteurs avec
remise, chacun avec tous ses énoncés (bootstrap par grappes, Field et
Welsh, 2007). Le test reste apparié : un même tirage de locuteurs sert à
toutes les variantes. C'est automatique dès que les sorties portent un
locuteur (`--unite auto`). Sur un jeu synthétique où chaque voix a son
propre taux d'erreur (10 locuteurs, 12 énoncés chacun), l'IC tiré par
locuteur est 3,3 fois plus large que l'IC tiré par énoncé
(`tests/test_significance.py`). Les chiffres de la phase 1 (Jeli-ASR, sans
locuteur dans les sorties) restent tirés par énoncé.

Conséquence pour la collecte : à nombre d'énoncés égal, **plus de
locuteurs** resserre l'IC davantage que plus d'énoncés par locuteur (voir
`docs/COLLECTE.md`, §3).

**Sous-groupes.** `eval.analysis` découpe aussi les scores du jeu maison par
registre (lu / spontané), code-switching, conditions d'enregistrement, genre
et locuteur. Ces chiffres décrivent : deux sous-groupes n'ont ni les mêmes
énoncés ni les mêmes voix, et leur écart n'est pas un test. Un locuteur
nettement à part y apparaît aussi (micro défectueux, transcription à
revoir).

## 6. Spécificités du bambara à traiter

- **Langue à tons non notés.** L'orthographe n'écrit pas les tons : source
  d'homographes en traduction et d'ambiguïté prosodique en synthèse. C'est un
  objet d'analyse, pas seulement une gêne.
- **Variation orthographique.** ɛ/ɔ/ɲ/ŋ contre approximations ASCII,
  conventions concurrentes. Traitée par `bambara_voice.normalize`.
- **Code-switching bambara-français** massif à l'oral, surtout à Bamako.
  Doit figurer dans le jeu de test : l'exclure rendrait l'évaluation
  irréaliste.
- **Latence.** Trois à quatre modèles en série sur CPU : push-to-talk assumé,
  pas de duplex temps réel. Contrainte annoncée, pas subie.

## 7. Ordre de travail

| Phase | Contenu | Où |
|---|---|---|
| 0 | Squelette, harnais d'évaluation, normalisation | ce dépôt |
| 1 | Références zero-shot (Whisper, MMS, NLLB, MMS-TTS) | Colab |
| 2 | **Jeu de test maison, 200–500 énoncés** | terrain |
| 3 | Fine-tuning Whisper multi-tâche + NLLB deux sens | Colab |
| 4 | Comparaison cascade / bout-en-bout, propagation d'erreurs | Colab |
| 5 | Assemblage CPU, quantisation, démo | ce dépôt |

**La phase 2 est le chemin critique.** Elle prend plus de temps que tout le
code réuni, et sans elle aucun chiffre n'est défendable. Tout le reste peut
avancer en parallèle ; elle, non.

## 8. Références

- Bayelemabaga: Creating Resources for Bambara NLP — NAACL 2025 —
  <https://aclanthology.org/2025.naacl-long.602.pdf>
- Where Are We At with ASR for the Bambara Language? — AfricaNLP 2026 —
  <https://aclanthology.org/2026.africanlp-main.26.pdf>
- Jeli-ASR (RobotsMali) — <https://huggingface.co/datasets/RobotsMali/jeli-asr>
- Bayelemabaga (RobotsMaliAI) — <https://huggingface.co/datasets/RobotsMaliAI/bayelemabaga>
- Koehn, P. — *Statistical Significance Tests for Machine Translation
  Evaluation* — EMNLP 2004 — <https://aclanthology.org/W04-3250/>
- Field, C. A. et Welsh, A. H. — *Bootstrapping Clustered Data* — Journal
  of the Royal Statistical Society, série B, 69(3), 2007
- Kunnafonidilaw ka Cadeau, ASR dataset de bambara contemporain —
  <https://arxiv.org/html/2512.19400>
- Manding Language Tech Resources (An ka taa) —
  <https://www.ankataa.com/blog/2024/6/27/manding-language-tech-resources-and-initiatives>

> Les identifiants de modèles et de jeux de données doivent être **revérifiés**
> avant usage : l'écosystème bambara bouge vite et des modèles ont déjà été
> retirés (le `MALIBA-AI/bambara-tts` d'origine, par exemple).
