# Résultats

Chiffres mesurés, dans l'ordre des phases. Chaque tableau indique d'où il
vient, pour être reproductible. Les rapports complets et les sorties ligne à
ligne sont dans les artefacts des exécutions GitHub Actions (conservés
quelques mois seulement : d'où ce fichier).

## Phase 1 — références zero-shot

Exécution : workflow « Phase 1 — références zero-shot », 28 septembre 2026
([exécution 36467549375](https://github.com/LOSTHADY/IA_Project-/actions/runs/36467549375)).
Machines GitHub standard (4 processeurs, sans GPU), PyTorch 2.14 CPU,
transformers 5.17, datasets 5.0. Partitions de test, échantillon tiré avec
la graine 0. Intervalles de confiance (IC) à 95 % par bootstrap, 1 000
tirages (`eval.significance`).

### Reconnaissance vocale — Jeli-ASR, 300 énoncés

| | Whisper-small | MMS-1b-all (adaptateur `bam`) |
|---|---|---|
| WER strict | 160,3 % [137,0 ; 186,6] | **58,2 % [54,6 ; 62,0]** |
| WER relâché | 159,1 % [135,9 ; 185,2] | 54,8 % [51,3 ; 58,7] |
| Écart orthographique | 1,2 point | 3,4 points |
| CER strict | 114,2 % | **25,7 %** |
| Latence moyenne | 4,2 s (RTF 1,64) | 8,3 s avec la traduction (RTF 2,59) |

Écart MMS − Whisper, apparié sur les mêmes énoncés : −102,1 points de WER
strict [−127,9 ; −79,0], p < 0,001.

- **Whisper d'origine ne transcrit pas le bambara.** Au-delà de 100 %, le
  WER compte plus d'erreurs que de mots de référence, ce qui suppose de
  nombreuses insertions : le modèle produit du texte qui n'est pas dans
  l'audio. Il ne connaît pas la langue, et le token « sw » qu'on lui impose
  l'oriente vers le swahili (voir l'analyse d'erreurs ci-dessous).
- **MMS est la vraie référence zero-shot.** Environ trois caractères sur
  quatre sont justes (CER 25,7 %), mais plus d'un mot sur deux est faux.
  C'est le seuil qu'un Whisper affiné doit battre (phase 3).
- **L'orthographe pèse peu dans l'erreur d'ASR** (3,4 points pour MMS) :
  les erreurs sont surtout acoustiques et lexicales, pas graphiques.

### Traduction — NLLB-200 distillé 600M

| chrF++ [IC 95 %] | Bayelemabaga (écrit, 500) | Jeli-ASR (oral, 491) |
|---|---|---|
| bambara → français | 21,7 [20,7 ; 22,7] | 22,4 [21,2 ; 23,4] |
| français → bambara | 25,2 [23,7 ; 26,6] | 28,9 [27,1 ; 30,5] |
| français → bambara, graphie repliée | 31,9 [30,1 ; 33,4] | 31,3 [29,3 ; 33,0] |
| BLEU bambara → français | 4,6 | 3,1 |
| BLEU français → bambara | 7,1 | 7,5 |

Latence : environ 2,6 s par phrase et par sens (faisceau 4, PyTorch fp32).
Sur Jeli-ASR, 9 des 500 phrases tirées n'avaient pas de paire complète.

- **La traduction zero-shot est faible dans les deux sens** (chrF++ entre
  22 et 29). Le fine-tuning sur Bayelemabaga (phase 3) est indispensable.
- **Vers le bambara, la graphie coûte jusqu'à 6,7 points de chrF++**
  (Bayelemabaga). NLLB n'écrit pas le bambara dans la même orthographe que
  les références. C'est exactement ce que la mesure « relâchée » isole.
- Les deux corpus ne portent pas sur les mêmes phrases : on ne peut pas les
  comparer par un test apparié. En français → bambara, les IC ne se
  chevauchent pas (25,2 contre 28,9), mais le chiffre replié les rapproche
  (31,9 contre 31,3). L'écart tiendrait donc surtout aux conventions
  graphiques des deux corpus, plus qu'au domaine.

### Chaîne d'entrée — de l'audio au français (MMS puis NLLB), Jeli-ASR

| | chrF++ |
|---|---|
| Traduction de la transcription de référence | 22,8 |
| Traduction de la sortie de MMS | 17,8 |
| **Perte due à l'ASR** | **5,0 points (22 %)** |

- **À ce stade, le maillon limitant est la traduction, pas l'ASR** : même
  avec une transcription parfaite, NLLB plafonne vers 23 de chrF++.
- **Seuil à battre en phase 4** : 17,8 de chrF++ de l'audio au français.
  La variante bout-en-bout (Whisper affiné qui traduit directement) doit
  être comparée à ce chiffre, puis à la cascade affinée.

### Synthèse vocale — MMS-TTS (`facebook/mms-tts-bam`)

20 phrases de Jeli-ASR synthétisées : 1,5 s par phrase, RTF 0,51, soit plus
rapide que le temps réel sur le processeur de GitHub. La qualité se juge à
l'oreille : les audios et la grille `mos.csv` sont dans l'artefact
`phase1-tts`, à faire noter par au moins trois auditeurs natifs, puis
`python -m eval.mos`.

### Analyse d'erreurs

Exécution : workflow « Analyse d'erreurs », 6 octobre 2026
([exécution 37397069671](https://github.com/LOSTHADY/IA_Project-/actions/runs/37397069671)),
`python -m eval.analysis` sur les sorties ligne à ligne de l'exécution
ci-dessus. Aucun modèle n'est relancé. Le détail est dans l'artefact
`analyse-erreurs`.

#### Reconnaissance vocale

Les trois types d'erreurs sont rapportés aux mots de référence ; leur somme
redonne le WER strict.

| | Whisper-small | MMS-1b-all |
|---|---|---|
| WER strict | 160,3 % | 58,2 % |
| dont substitutions | 56,0 % | 38,7 % |
| dont suppressions | 40,7 % | 10,3 % |
| dont insertions | 63,7 % | 9,2 % |
| Substitutions purement orthographiques (part des substitutions) | 0,8 % | 7,0 % |
| Énoncés en boucle de répétition | 7,7 % | 0,0 % |
| Énoncés hallucinés (sortie ≥ 2 × la référence) | 7,7 % | 2,0 % |
| Longueur sortie / référence, médiane | 0,60 | 1,00 |

| WER selon la longueur de l'énoncé | 1–3 mots (20) | 4–7 (103) | 8–15 (121) | 16 et plus (56) |
|---|---|---|---|---|
| Whisper-small | 93,6 % | 230,6 % | 158,4 % | 133,8 % |
| MMS-1b-all | 110,6 % | 56,0 % | 59,2 % | 56,2 % |

- **Whisper écrit du swahili.** Le mot qu'il insère le plus est « kwa »,
  mot grammatical très courant du swahili : 1 926 fois sur 300 énoncés, loin
  devant le suivant (« nga », 88 fois). Il remplace aussi les petits mots
  les plus fréquents du bambara (ka, a, ko, ye → kwa). C'est l'effet direct
  du token de langue « sw » imposé faute de token bambara (`METHODE.md`, §3).
  Le fine-tuning doit réaffecter ce token au bambara ; la disparition de
  « kwa » dans les sorties de la phase 3 en sera le premier signe.
- **Le WER au-delà de 100 % vient d'une minorité d'énoncés en boucle.**
  D'ordinaire, Whisper écrit trop peu et faux : la sortie médiane fait 60 %
  de la longueur de la référence, et l'énoncé médian a un WER de 100 %. Mais
  7,7 % des énoncés partent en boucle (« kwa kwa kwa… », jusqu'à 25 fois
  plus de mots que la référence), et ce sont eux qui font exploser les
  insertions.
- **MMS ne boucle pas** (aucun énoncé sur 300). Son décodage CTC n'est pas
  autorégressif : il ne réinjecte pas sa propre sortie, et la longueur de
  celle-ci reste bornée par la durée de l'audio. Ses erreurs sont aux deux
  tiers des substitutions (38,7 points sur 58,2).
- **Les confusions les plus fréquentes de MMS sont des élisions** (k'a → ka,
  11 fois ; n'i → ni ; t'a → ta ; k'o → ko), la longueur vocalique (bɛɛ →
  bɛ), la nasale finale (do → don) et o/ɔ (ko → kɔ). Ces confusions
  fréquentes ne sont pourtant qu'une petite part du total : 7 % seulement
  des substitutions sont purement orthographiques (élisions comprises, que
  le WER relâché compte justes). L'essentiel est une longue traîne d'erreurs
  lexicales, que seul le fine-tuning peut corriger.
- **Les énoncés très courts sont les plus durs pour MMS** (110,6 % de 1 à
  3 mots), mais sur 20 énoncés seulement, où un mot inséré dans un énoncé
  de deux mots pèse 50 %.

#### Traduction

| chrF++ selon la longueur de la source | 1–3 mots | 4–7 | 8–15 | 16 et plus |
|---|---|---|---|---|
| Bayelemabaga, bambara → français | 20,8 (45) | 22,4 (190) | 22,2 (201) | 20,2 (64) |
| Bayelemabaga, français → bambara | 33,9 (37) | 28,1 (142) | 25,7 (240) | 22,7 (81) |
| Jeli-ASR, bambara → français | 23,2 (43) | 23,3 (177) | 23,1 (179) | 21,1 (92) |
| Jeli-ASR, français → bambara | 13,4 (39) | 28,4 (170) | 29,0 (206) | 31,3 (76) |
| MMS puis NLLB, bambara → français | 14,8 (20) | 17,1 (102) | 18,0 (118) | 18,1 (53) |

Entre parenthèses : nombre de phrases.

| Sorties dégénérées | trop longues (≥ 2 × la référence) | en boucle |
|---|---|---|
| Bayelemabaga, bambara → français | 5,2 % | 1,6 % |
| Bayelemabaga, français → bambara | 10,4 % | 2,6 % |
| Jeli-ASR, bambara → français | 10,2 % | 2,9 % |
| Jeli-ASR, français → bambara | 6,7 % | 1,8 % |
| MMS puis NLLB | 11,9 % | 3,4 % |

- **NLLB boucle aussi** : 1,6 à 3,4 % des sorties, et 5 à 12 % font au
  moins deux fois la longueur de la référence. Pire cas : « Le crocodile
  hurla; il vira. » devient « Sɔgɔsɔgɔsɔgɔ… » sur des centaines de
  caractères. Piste peu coûteuse pour la phase 3 : plafonner la longueur
  générée en proportion de la source. Interdire les répétitions de
  n-grammes serait plus risqué, car le bambara redouble légitimement des
  mots (« a y'i munumunu », dans la référence de ce même exemple).
- **Vers le français, la longueur ne joue presque pas** (chrF++ entre 20 et
  23 dans toutes les tranches) : la faiblesse de NLLB y est lexicale, pas
  liée aux phrases longues.
- **Vers le bambara, l'effet de la longueur dépend du corpus.** Sur
  Bayelemabaga, le chrF++ baisse régulièrement avec la longueur de la phrase
  française (33,9 → 22,7). Cela va dans le sens de la contrainte de phrases
  courtes en sortie du LLM (`METHODE.md`, §4, levier 2). Sur Jeli-ASR, la
  tendance s'inverse (13,4 → 31,3) : les phrases françaises très courtes y
  sont des fragments d'oral (« ehh mais », référence « ɔɔ mɛ si »), difficiles
  pour une autre raison, et le chrF++ d'une phrase de deux mots est très
  bruité. La longueur se confond ici avec le type de contenu : l'effet du
  levier 2 reste à mesurer directement, sur le même contenu avec et sans
  simplification.

#### Ce que l'analyse change pour la suite

1. **Phase 3, Whisper affiné** : relancer la même analyse sur ses sorties.
   « kwa » et les boucles doivent disparaître. S'il reste des boucles, les
   traiter au décodage, comme le décodage d'origine de Whisper : détecter
   une sortie trop répétitive (taux de compression) et redécoder à
   température plus élevée (Radford et al., 2022).
2. **NLLB** : mesurer l'effet d'un plafond de longueur générée sur les
   sorties dégénérées, le chrF++ et la latence. Option `--plafond` de
   `eval.baselines` (désactivée par défaut), expérience dans le workflow
   « Plafond de longueur NLLB », sur les mêmes phrases que ci-dessus.
3. **Levier 2** : expérience directe, les longueurs observées ici ne
   suffisant pas à conclure. Mêmes phrases longues, traduites entières puis
   découpées en segments courts (`eval.baselines mt --decoupe`, workflow
   « Découpage avant traduction »).

### Limites

- **Latences mesurées sur les processeurs de GitHub, en PyTorch fp32** :
  indicatives seulement. Le budget de déploiement se mesure sur la machine
  cible, avec les modèles int8 (`docs/DEPLOIEMENT.md`).
- **Jeli-ASR n'est pas le jeu de test du mémoire.** Ses locuteurs et ses
  conditions d'enregistrement ne sont pas ceux de l'assistant, et il servira
  à l'entraînement en phase 3. Les chiffres définitifs se mesureront sur le
  jeu de test maison (phase 2).
