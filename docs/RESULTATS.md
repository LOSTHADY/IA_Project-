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
strict [−127,9 ; −79,0], p < 0,002 (aucun des 1 000 tirages n'annule l'écart).

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
| Bayelemabaga, français → bambara | 10,4 % | 3,0 % |
| Jeli-ASR, bambara → français | 10,2 % | 2,9 % |
| Jeli-ASR, français → bambara | 6,7 % | 2,0 % |
| MMS puis NLLB | 11,9 % | 3,4 % |

Une sortie est comptée en boucle quand un même groupe de trois mots y
revient au moins trois fois, ou qu'un motif de 2 à 12 caractères s'y répète
au moins quatre fois d'affilée (« Sɔgɔsɔgɔsɔgɔ… », un seul « mot » que le
premier critère manquait). Analyse refaite avec ce détecteur le 6 octobre
([exécution 37405802014](https://github.com/LOSTHADY/IA_Project-/actions/runs/37405802014)) :
seuls les deux taux vers le bambara ont changé.

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
2. **NLLB** : plafonner la longueur générée. Fait, et activé par défaut :
   voir la section suivante.
3. **Levier 2** : expérience directe, les longueurs observées ici ne
   suffisant pas à conclure. Faite : voir « Découpage avant traduction »
   ci-dessous.

### Plafond de longueur de NLLB

Exécution : workflow « Plafond de longueur NLLB », 6 octobre 2026
([exécution 37398523459](https://github.com/LOSTHADY/IA_Project-/actions/runs/37398523459)),
comparée phrase par phrase à la phase 1 avec `eval.changes`
([exécution 37405802049](https://github.com/LOSTHADY/IA_Project-/actions/runs/37405802049)).
Mêmes phrases que la phase 1 (même corpus, même partition, même graine).
Une seule valeur, fixée avant la mesure : la traduction compte au plus
2 × la longueur de la source + 10 jetons, contre 256 jetons fixes en phase 1.

| chrF++ [IC 95 %] | sans plafond (phase 1) | avec plafond | écart [IC 95 %] |
|---|---|---|---|
| Bayelemabaga, bambara → français | 21,7 [20,7 ; 22,7] | 21,8 [20,8 ; 22,8] | +0,1 [0,0 ; 0,3] |
| Bayelemabaga, français → bambara | 25,2 [23,7 ; 26,6] | 25,7 [24,3 ; 26,9] | +0,4 [0,1 ; 0,8] |
| Bayelemabaga, fr → bm, graphie repliée | 31,9 [30,1 ; 33,4] | 32,5 [30,9 ; 34,0] | +0,6 [0,1 ; 1,1] |
| Jeli-ASR, bambara → français | 22,4 [21,2 ; 23,4] | 22,6 [21,5 ; 23,6] | +0,2 [0,0 ; 0,5] |
| Jeli-ASR, français → bambara | 28,9 [27,1 ; 30,5] | 29,8 [28,1 ; 31,2] | +0,9 [0,3 ; 1,5] |
| Jeli-ASR, fr → bm, graphie repliée | 31,3 [29,3 ; 33,0] | 32,2 [30,4 ; 33,7] | +0,9 [0,3 ; 1,6] |

Test apparié sur les mêmes phrases (bootstrap, 1 000 tirages) : l'IC de
l'écart exclut zéro dans les six cas (sa borne basse s'arrondit à 0,0 vers
le français), p entre 0,002 et 0,026.

| Traductions | total | changées | dont arrêtées par le plafond |
|---|---|---|---|
| Bayelemabaga, bambara → français | 500 | 4 | 3 |
| Bayelemabaga, français → bambara | 500 | 6 | 5 |
| Jeli-ASR, bambara → français | 491 | 5 | 4 |
| Jeli-ASR, français → bambara | 491 | 8 | 8 |

- **Le plafond ne touche que des boucles.** Il arrête 21 traductions sur
  1 982 (0,6 à 1,6 % selon le corpus et le sens). Le détecteur reconnaît une
  boucle dans au moins 83 % d'entre elles, et tous les exemples affichés en
  sont (« Il m'a parlé, il m'a parlé… », « fiɲɛba fiɲɛba… »,
  « Sɔgɔsɔgɔ… »). Trois autres traductions ont changé sans être arrêtées :
  c'étaient aussi des boucles en phase 1, et la recherche en faisceau, bornée
  plus tôt, a retenu une hypothèse terminée (« Ne tun y'a fɔ ko ne tɛ pikiri
  kɛ tuguni, ko ne tɛ pikiri kɛ tuguni, … » devient « Ne tun y'a fɔ ko ne tɛ
  pikiri kɛ tuguni. »).
- **L'écart revient bien au plafond.** Entre les deux exécutions,
  transformers (5.17 → 5.18), torch (2.14.0 → 2.14.1) et datasets ont changé
  de version sans modifier aucune autre traduction.
- **Le gain est petit mais établi.** Il vient de ce qu'une boucle
  raccourcie apporte moins de caractères faux au chrF++. Le plafond ne guérit
  pas les boucles, il les raccourcit : leur part ne baisse presque pas
  (Bayelemabaga, français → bambara : 3,0 % → 2,8 %).
- **Latence** : non comparable ici, les deux exécutions ayant tourné sur des
  machines différentes (celle du 6 octobre, plus lente dans son ensemble).
  Le plafond borne surtout le pire cas : une boucle pouvait coûter 256
  étapes de décodage, même pour une phrase de trois mots.
- **Décision : plafond activé par défaut** (`MTConfig.max_length_ratio`),
  dans la chaîne comme dans l'évaluation. Une seule valeur, fixée avant la
  mesure, et un effet de même sens sur deux corpus : avoir décidé sur ces
  partitions de test n'introduit pas de biais de sélection notable, mais
  cela est signalé ici. Pour reproduire exactement les chiffres de la
  phase 1 : `--plafond 0`.

### Découpage avant traduction (levier 2)

Exécution : workflow « Découpage avant traduction (levier 2) »,
6 octobre 2026
([exécution 37399099065](https://github.com/LOSTHADY/IA_Project-/actions/runs/37399099065)) ;
changements phrase par phrase avec `eval.changes`
([exécution 37414453973](https://github.com/LOSTHADY/IA_Project-/actions/runs/37414453973)).
Bayelemabaga, partition de validation : 1 000 paires tirées (graine 0), dont
588 avec une phrase française d'au moins 12 mots. Chaque phrase est traduite
vers le bambara entière, puis découpée aux virgules, points-virgules et
deux-points en segments d'au plus 10 mots, traduits un à un et mis bout à
bout (`split_for_translation`). 62,6 % des phrases ont été découpées.
Paramètres fixés à l'avance ; pas de plafond de longueur dans les deux
variantes (expérience lancée avant son activation).

| Français → bambara, 588 phrases | entières | découpées | écart [IC 95 %] |
|---|---|---|---|
| chrF++ | 31,0 [29,9 ; 32,0] | 30,0 [29,0 ; 31,0] | −1,0 [−1,5 ; −0,5], p < 0,002 |
| chrF++, graphie repliée | 34,9 [33,8 ; 36,1] | 34,4 [33,4 ; 35,5] | −0,5 [−1,1 ; 0,1], n.s. |
| BLEU | 9,5 | 7,8 | |
| chrF++, phrases de 8 à 15 mots (183) | 27,9 | 27,7 | |
| chrF++, phrases de 16 mots et plus (405) | 31,8 | 30,6 | |

Contrôle : bambara → français, traduit de la même façon dans les deux
variantes, donne les mêmes 588 sorties et le même score (30,2), avec les
mêmes versions de bibliothèques. Seul le découpage distingue les deux
colonnes.

- **Découper ne fait pas mieux traduire, plutôt moins bien.** −1,0 point de
  chrF++, perte concentrée sur les phrases les plus longues, celles que le
  découpage touche le plus. En graphie repliée, l'écart n'est plus
  significatif.
- **Deux mécanismes, visibles sur les sorties** (366 traductions changées) :
  - *la forme* : chaque morceau reçoit sa majuscule et son point
    (« … i kosɔn. Ne bɛna … »). Le chrF++ strict et le BLEU le comptent, la
    forme repliée (sans casse ni ponctuation) non, d'où la différence entre
    les deux lignes ;
  - *le fond* : traduit seul, un fragment est complété en phrase, et NLLB y
    ajoute parfois des mots que la version entière n'avait pas (« O kɔ, … »
    en tête d'une traduction, « o dɔrɔn tɛ » à la place de « o de ye nin
    ye » ; gloses à faire vérifier par un locuteur natif).
- **Conséquence pour la chaîne.** `simplify_for_translation` coupait les
  phrases de plus de 15 mots et ajoutait un point : elle fabriquait ce même
  type de fragment, en perdant en plus la fin de la phrase. Elle ne coupe
  plus par défaut (`LLMConfig.max_words`). Le découpage n'est pas adopté non
  plus. Restent la consigne du prompt, le retrait de la mise en forme et la
  limite de trois phrases.
- **Ce que l'expérience ne dit pas.** Elle découpe après coup des phrases
  existantes ; elle ne teste pas un LLM qui écrirait d'emblée des phrases
  courtes et complètes, ni un NLLB affiné. Ces deux questions restent
  ouvertes (phase 4).

### Limites

- **Latences mesurées sur les processeurs de GitHub, en PyTorch fp32** :
  indicatives seulement. Le budget de déploiement se mesure sur la machine
  cible, avec les modèles int8 (`docs/DEPLOIEMENT.md`).
- **Jeli-ASR n'est pas le jeu de test du mémoire.** Ses locuteurs et ses
  conditions d'enregistrement ne sont pas ceux de l'assistant, et il servira
  à l'entraînement en phase 3. Les chiffres définitifs se mesureront sur le
  jeu de test maison (phase 2).
