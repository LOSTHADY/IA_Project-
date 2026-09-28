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
  ne suffit pas. Les sorties ligne à ligne (artefact `phase1-asr-whisper`)
  permettent d'illustrer ces erreurs dans le mémoire.
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

### Limites

- **Latences mesurées sur les processeurs de GitHub, en PyTorch fp32** :
  indicatives seulement. Le budget de déploiement se mesure sur la machine
  cible, avec les modèles int8 (`docs/DEPLOIEMENT.md`).
- **Jeli-ASR n'est pas le jeu de test du mémoire.** Ses locuteurs et ses
  conditions d'enregistrement ne sont pas ceux de l'assistant, et il servira
  à l'entraînement en phase 3. Les chiffres définitifs se mesureront sur le
  jeu de test maison (phase 2).
