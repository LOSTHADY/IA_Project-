# La campagne de lecture

Des dizaines de lecteurs lisent à voix haute, sur leur téléphone, des phrases
bambara déjà écrites. Chaque lecture donne de l'audio avec son texte exact :
c'est ce qui apprend à Soloni à mieux entendre (`scripts/affiner_soloni.py`,
notebook `voix`). La reconnaissance est l'étage dont tout le reste dépend :
une phrase mal entendue sera mal traduite, quel que soit le traducteur.

Ce document dit quoi viser, comment s'y prendre, et comment savoir, dès le
pilote, si la campagne vaut les mois qu'elle demande.

## Ce qu'on vise

| Palier | Heures lues | Lecteurs | À quoi il sert |
|---|---|---|---|
| **pilote** | 20 h | 30 | vérifier que tout marche, et tracer la courbe d'apprentissage |
| **significatif** | 100 h | 100 | un gain qui se voit à l'usage |
| **grand** | 300 h | 200 | autant que les plus gros corpus lus publiés |

Les lecteurs comptent autant que les heures : 100 heures lues par dix
personnes apprennent dix voix ; Soloni doit entendre celles qu'il n'a jamais
entendues. Viser la moitié de femmes, tous les âges adultes, plusieurs régions
(Bamako, Ségou, Sikasso, Kayes, Koulikoro…), beaucoup de téléphones
différents.

**Le critère de réussite** se mesure sur les **voix nouvelles** (plus bas) :
au palier significatif, Soloni affiné doit faire au moins **30 % d'erreurs de
moins** que Soloni d'origine (par exemple 30 % de WER ramenés à 21 %). Ordres
de grandeur publiés, pour situer :

- RobotsMali, 45,6 h de livres lus (be-kalan) : WER de 42 % à 22 %, mais
  mesuré sur d'autres lectures des mêmes livres, donc optimiste ;
- Kunkado, environ 33 h de parole spontanée : de 36 % à 32 % ;
- Soloni v2, appris sur 100 h d'AfVoices : environ 29 %.

Ce sont d'autres données et d'autres mesures. Seule la courbe du pilote dira
ce que *nos* heures rapportent.

## Le pilote d'abord, et la courbe

Le pilote (20 h, 30 lecteurs) ne sert pas qu'à essayer l'outil. Avec
`COURBE = True`, le notebook `voix` affine Soloni trois fois : sur un quart,
sur la moitié, puis sur toutes les lectures du pilote. Il mesure à chaque fois
le WER des voix nouvelles. En reconnaissance, les erreurs baissent à peu près
d'autant à chaque doublement des heures ; `eval/courbe.py` trace cette droite
et prédit 100 et 300 heures.

- **La courbe descend nettement** (plusieurs points par doublement) : la
  campagne vaut la peine, continuer.
- **Elle est plate** : plus de la même chose ne rapportera pas plus. Chercher
  pourquoi avant de continuer : lectures de mauvaise qualité, phrases trop
  proches de ce que Soloni connaît déjà, lecteurs trop semblables.

C'est une extrapolation : plus on s'éloigne des heures mesurées, moins elle
vaut. Elle sert à décider, pas à promettre un chiffre.

## Les voix nouvelles : la mesure honnête

Un contributeur sur huit, dans l'ordre d'inscription (spk-08, spk-16,
spk-24…), est **réservé** : rien de ce qu'il enregistre n'est appris, et il ne
lit que des phrases du test, qu'aucun lecteur n'apprend non plus
(`eval.atelier.lecteur_reserve`, `partie_voix`). Ses lectures forment le test
des voix nouvelles : des voix et des phrases que Soloni n'a jamais entendues.
C'est exactement ce que fera l'assistant devant un inconnu. Le notebook `voix`
juge sur elles dès qu'il y en a 50, sinon sur les phrases mises de côté.

Ne choisis pas qui est réservé : l'ordre d'inscription le décide, ce qui évite
de mettre de côté, sans le vouloir, les voix les plus faciles. Et **une
personne, un identifiant** : un lecteur réservé qui se réinscrirait sous un
autre numéro serait appris, et le test ne vaudrait plus rien.

## Préparer la liste des phrases

Une fois, dans `notebooks/atelier_collecte.ipynb` (section 3, `CAMPAGNE =
True`), ou en ligne de commande :

```bash
python scripts/campagne.py liste --atelier ATELIER   # environ 30 000 phrases
python scripts/campagne.py bilan --atelier ATELIER   # où en est la campagne
```

- 22 000 phrases de **conversation** : les transcriptions relues d'AfVoices
  (RobotsMali, CC-BY-4.0), du bambara parlé de tous les jours ;
- 8 000 phrases **avec leur français** (Bayelemabaga) : la même campagne
  servira un jour à la traduction et au bout-en-bout ;
- de 4 à 14 mots, sans chiffres (qui se disent de plusieurs façons), sans
  texte religieux, sans doublon. Les phrases déjà dans la liste, déjà lues,
  sont gardées ; l'ancienne liste est copiée dans
  `lecture.avant-campagne.jsonl`.

L'onglet **Lire** distribue les phrases : chaque lecteur dans son propre
ordre, trois lecteurs au plus par phrase tant qu'il en reste (plus de phrases
différentes, chacune dite par plusieurs voix).

## Qui peut lire

**Il faut savoir lire le bambara couramment.** Une lecture hésitante,
syllabe par syllabe, ou qui dit autre chose que le texte, apprend à Soloni de
mauvaises choses. Le filtre (plus bas) en écarte une partie, pas toutes.
Les bons lecteurs se trouvent parmi les enseignants des écoles bilingues, les
alphabétiseurs, les étudiants en langues nationales, les animateurs de radio
en bambara, les associations d'alphabétisation.

Avant la première séance, chacun s'inscrit dans l'atelier : genre, âge,
région, **téléphone (marque et modèle)**, et le consentement
([COLLECTE.md](COLLECTE.md)), qu'il faut lui lire ou lui faire lire.
Seulement des adultes : pour un mineur, il faudrait aussi l'accord écrit des
parents. Un dédommagement (du crédit téléphonique par exemple) se pratique
dans ce genre de collecte et aide à garder les lecteurs.

## Consignes aux lecteurs

À leur envoyer avec le lien, ou à leur dire au début de la séance :

1. Ouvre le lien sur ton téléphone, choisis ton pseudonyme (spk-…) en haut de la page,
   onglet **📖 Lire**.
2. Lis la phrase une fois dans ta tête, puis enregistre-la **comme tu la
   dirais à quelqu'un**, ni trop lentement, ni en criant.
3. Tiens le téléphone comme pour un appel, à une vingtaine de centimètres de
   la bouche.
4. Tu t'es trompé ? Recommence l'enregistrement. Une phrase trop difficile,
   ou qui ne te paraît pas être du bon bambara : **Passer**.
5. Indique où tu es (« Où enregistres-tu ? ») : salle calme, intérieur bruyant,
   rue ou marché, véhicule… Lis aussi dans des endroits bruyants : l'assistant
   servira là aussi. Environ la moitié au calme, le reste ailleurs.
6. Des séances de 30 à 45 minutes : la fatigue fait les mauvaises lectures.

## Combien de temps, combien de place

- Une lecture dure 4 à 5 secondes : **une heure de parole, c'est 700 à
  1 000 lectures**. Une heure de séance donne environ 20 minutes de parole
  (lire, écouter, recommencer). Pour 20 h, il faut donc environ 60 heures de
  séance (deux par lecteur pour 30 lecteurs) ; pour 300 h, environ 900.
- Le son est gardé en WAV 16 kHz, environ **115 Mo par heure** : 20 h font
  2,3 Go, 100 h 11,5 Go, 300 h 35 Go. Ton Drive de 100 Go suffit, à condition
  de ne garder que les meilleurs checkpoints (un Soloni affiné pèse environ
  0,5 Go, un NLLB plusieurs Go).
- Les fichiers sont rangés par lecteur (`audio/spk-NN/`) : Drive peine sur
  un dossier de centaines de milliers de fichiers.

## Où faire tourner l'atelier

L'atelier n'a pas besoin de GPU. **Pour la campagne, lance-le sur ton
ordinateur** ([ATELIER.md](ATELIER.md), « Sur un ordinateur ») avec
`--share`, et range son dossier dans Drive grâce à l'application *Google
Drive pour ordinateur* : les notebooks d'entraînement le retrouvent dans
`MyDrive/bambara/atelier`. La FAQ de Colab exclut « les services web sans
rapport avec le calcul interactif » : une application de collecte ouverte des
semaines à des dizaines de personnes en relève probablement. Le notebook
`atelier_collecte` reste pratique pour préparer la liste et pour une séance
ponctuelle où tu es là.

Un seul atelier à la fois par dossier : deux ateliers qui écrivent dans le
même dossier (l'ordinateur et Colab, par exemple) se marcheraient dessus.

Le lien `gradio.live` change à chaque lancement : renvoie le nouveau au
groupe à chaque séance.

## Le filtre de qualité

Avant d'apprendre, `affiner_soloni.py` fait entendre chaque lecture à Soloni
d'origine et écarte celles dont il entend tout autre chose que le texte
(`--filtre 0.7` : plus de 70 % d'erreurs par caractère) : mots sautés,
silence, autre phrase lue. Le rapport (`rapport.json`)
en donne le nombre ; s'il dépasse quelques pour cent, réécouter quelques
lectures écartées pour comprendre (un lecteur, un téléphone, un lieu ?).

## Entraîner

- **Au pilote** : le notebook `voix` sur Colab (GPU T4), quelques heures
  pour les trois entraînements de la courbe.
- **À 100 h et plus**, un entraînement dépassera une séance Colab : il
  faudra pouvoir reprendre un entraînement coupé. Ce n'est pas encore écrit ;
  à faire avant le palier significatif. Kaggle donne aussi des heures de GPU
  chaque semaine, à vérifier sur son site.

Soloni affiné ne remplace celui en service que s'il fait moins d'erreurs sur
les voix nouvelles **et** que l'assistant comprend mieux le jeu maison
(`eval.promotion`) ; sinon rien ne change.

## Suivre la campagne

L'onglet **Bilan** de l'atelier, ou `python scripts/campagne.py bilan
--atelier ATELIER` : heures lues, lecteurs (dont réservés), femmes,
téléphones différents, lieux, phrases lues, palier atteint.
