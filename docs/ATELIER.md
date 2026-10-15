# Atelier de données

Ce qu'un locuteur natif peut fournir **sans savoir écrire le bambara**, et
ce que chaque donnée apporte à chaque maillon de l'assistant.

| Onglet | Geste | Maillon servi | Ce qu'on obtient |
|---|---|---|---|
| 🗣️ Dire | lire une phrase française, la dire en bambara comme on parle | **toute la chaîne** (évaluation) | un jeu de test maison : chrF++ du français compris depuis l'audio, cascade contre bout-en-bout, IC par locuteur |
| 📖 Lire | lire à voix haute une phrase bambara déjà écrite | synthèse vocale, reconnaissance | audio + texte exact, sans transcrire |
| ⚖️ Juger | choisir la meilleure de deux traductions, à l'aveugle | traduction, chaîne complète | une évaluation humaine, avec un test du signe |
| 📚 Lexique | noter un mot d'argot ou un emprunt et son sens | entrée de la traduction | une table de normalisation (*wai* → salut) |

Le jeu « Dire » ne demande aucune transcription : la phrase française
affichée **est** la référence. On n'y mesure donc pas le WER, mais la
mesure centrale du mémoire, la compréhension depuis l'audio, sur de la
parole réelle : la vôtre, avec ses mélanges de français et son argot,
qui manquent à Jeli-ASR.

## Lancer l'atelier

### Sur Colab (recommandé : un lien pour tout le monde)

[`notebooks/atelier_collecte.ipynb`](../notebooks/atelier_collecte.ipynb),
[ouvrir dans Colab](https://colab.research.google.com/github/LOSTHADY/IA_Project-/blob/claude/chat-ia-bambara-dho8oa/notebooks/atelier_collecte.ipynb).

1. **Exécution → Modifier le type d'exécution → CPU** : pas besoin de GPU,
   le quota reste pour l'entraînement.
2. **Exécution → Tout exécuter**, autoriser Drive.
3. Ouvrir le lien `https://….gradio.live` affiché par la dernière cellule,
   et l'envoyer aux contributeurs.

Tout est écrit sur Drive (`MyDrive/bambara/atelier`) à chaque clic. Le lien
marche tant que la cellule tourne ; si la session se coupe, tout relancer
(nouveau lien) : chacun reprend où il en était. Garder l'onglet Colab
ouvert pendant les séances : Colab gratuit coupe une session inactive.

GitHub ne peut pas héberger l'atelier : GitHub Pages ne sert que des pages
fixes, sans programme pour recevoir l'audio.

### Sur un ordinateur

```bash
pip install gradio soundfile librosa datasets
python -m eval.atelier --racine data/atelier/collecte lecture   # une fois
python app/atelier_app.py                    # http://127.0.0.1:7862
python app/atelier_app.py --share            # plus un lien public, pour un téléphone
```

Le micro d'un navigateur ne marche qu'en `https` ou sur l'ordinateur
lui-même : un téléphone passe par le lien `--share`, pas par l'adresse
`192.168…` du réseau local.

## Le parcours d'un contributeur

1. **Première fois ? Inscris-toi ici** : genre, tranche d'âge, région, et
   deux cases. La première (usage pour la recherche) est obligatoire ; la
   seconde (publication un jour) est facultative. Aucun nom : l'atelier
   donne un pseudonyme (`spk-01`…). Le consentement suit
   [`COLLECTE.md`](COLLECTE.md), §2 : expliquer le projet en bambara avant,
   et garder les formulaires à part.
2. Choisir son pseudonyme dans **Qui es-tu ?**, et l'endroit dans **Où
   enregistres-tu ?** (salle calme, rue, voiture…).
3. **Dire** : choisir une rubrique, ou les faire dans l'ordre. Micro, parler,
   arrêter, **Enregistrer et suivante**. Une phrase qui ne parle pas :
   **Passer**. Une erreur : **Annuler la dernière** (la phrase revient).
4. **Lire**, **Juger**, **Lexique**, dans l'ordre qu'on veut.

Les barres de progression suivent chacun : au total, par rubrique, en
minutes lues, en paires jugées. **Bilan** montre l'ensemble.

### Les 300 phrases à dire

12 rubriques de 25 phrases, de la vie courante et des demandes qu'on ferait
à un assistant ([`data/atelier/a_dire.json`](../data/atelier/a_dire.json)) :

salutations et politesse · présentations et famille · santé · météo et
saisons · nombres, heures, dates et argent · marché et commerce ·
déplacements et directions · école et travail · téléphone, internet et
argent mobile · nourriture et cuisine · demandes à l'assistant · loisirs,
jeunes et vie quotidienne.

Une rubrique se modifie dans ce fichier ; les prises déjà faites gardent
leur phrase.

### Conseils de prise de son

- Le téléphone à 20–30 cm de la bouche, sans le couvrir.
- Écouter la première prise avant d'enchaîner : l'atelier signale une prise
  trop faible, saturée ou coupée, mais pas un bruit de fond.
- **Dire comme on parle**, pas mot à mot : si on dirait « n ka réunion
  bɛ… », c'est exactement ce qu'il faut. Cocher la case de mélange, et noter
  les mots familiers employés (*wai*, *yoh*…).
- Varier les voix : au moins 8 personnes, femmes et hommes, plusieurs
  âges et régions. Pour le jeu de test, 100 phrases de 6 personnes valent
  mieux que 300 de 2 : les IC se tirent par locuteur.

### Juger

Les lots sont préparés par le notebook depuis les rapports de la phase 3
(`resultats-phase3`), dans l'ordre :

1. **cascade contre bout-en-bout**, sur ce qui a été compris de l'audio :
   la question du mémoire, tranchée par des locuteurs natifs. On lit ce qui
   a été dit (la transcription de Jeli-ASR, en bambara), puis les deux
   français compris ;
2. et 3. **NLLB d'origine contre NLLB affiné**, dans chaque sens ;
4. et 5. **NLLB v1 contre v2**, dès que la v2 est évaluée.

Le côté de chaque système (A ou B) est tiré au hasard, et les traductions
identiques ne sont pas montrées. Juger d'abord le sens, puis le naturel ;
« les deux sont bonnes » et « les deux sont mauvaises » sont des réponses
utiles. **Bilan** donne les préférences et le p du test du signe (les
égalités écartées) : en dessous de 0,05, la préférence n'est pas due au
hasard.

## Ce qu'on fait des données

**Exporter** (onglet Bilan, ou la dernière cellule du notebook) écrit dans
le dossier de l'atelier :

- `testset-dire.jsonl` : le jeu de test maison. Le notebook de la phase 3
  l'évalue de lui-même (§9b), cascade contre bout-en-bout ; à la main :

  ```bash
  python -m eval.run_eval --testset "$ATELIER/testset-dire.jsonl" --arch cascade \
      --config assistant-affine.json --comprendre-seulement --out resultats/maison
  # idem avec --arch e2e et la configuration du bout-en-bout, puis :
  python -m eval.significance resultats/maison/cascade-<date>.json resultats/maison/e2e-<date>.json
  ```

  Avec plusieurs locuteurs, les IC sont tirés par locuteur (cf.
  `eval/significance.py`) : 8 locuteurs au moins pour qu'ils soient fiables.
- `lecture-export.jsonl` : audio et texte exact. Une heure (l'objectif
  affiché) est l'ordre de grandeur qu'il faut pour affiner une voix de
  synthèse MMS ; c'est aussi de l'entraînement pour la reconnaissance.
- `lexique.tsv` : à brancher avant la traduction entrante, pour que *wai*
  arrive à NLLB sous une forme qu'il connaît.

### Fichiers

Dans `MyDrive/bambara/atelier` (ou `data/atelier/collecte` en local) :

```
locuteurs.jsonl   pseudonymes, genre, âge, région, consentements
dire.jsonl        une prise « Dire » par ligne      audio/di-00001.wav …
lire.jsonl        une prise « Lire » par ligne      audio/li-00001.wav …
passes.jsonl      phrases passées
lecture.jsonl     les phrases bambara à lire
lots/*.jsonl      les paires à juger
jugements.jsonl   les choix
lexique.jsonl     le lexique
```

Audio en WAV mono 16 kHz. Écritures atomiques : un fichier n'est jamais à
moitié écrit, même si la session tombe pendant un clic.

## Données personnelles

La voix est une donnée personnelle, et le dépôt GitHub est **public** :
`data/atelier/collecte/` est exclu par `.gitignore` ; ne jamais le forcer
(`git add -f`), ni copier le dossier de Drive dans le dépôt. Seule la liste
des phrases à dire est versionnée.

- **Sauvegarde** : copier le dossier de l'atelier ailleurs après chaque
  séance (disque, second compte). Une corbeille de Drive vidée ne se
  rattrape pas.
- **Retrait** : supprimer les lignes du locuteur (`"speaker": "spk-XX"`)
  dans `dire.jsonl`, `lire.jsonl`, `passes.jsonl` et `jugements.jsonl`, sa
  ligne dans `locuteurs.jsonl` et ses fichiers audio, puis exporter de
  nouveau.
- **Publication** : seulement les locuteurs qui ont coché la seconde case
  (`public_release` dans `locuteurs.jsonl`), et seulement après l'avis de
  l'encadrant (cf. [`COLLECTE.md`](COLLECTE.md), §7).
