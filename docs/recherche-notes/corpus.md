# Corpus parallèles et monolingues bambara (bm / bam_Latn) ↔ français (et anglais)

> **Limites d'accès de cette recherche (à lire d'abord).** Depuis cet environnement, huggingface.co, arxiv.org, aclanthology.org, opus.nlpl.eu, statmt.org, cormand.huma-num.fr, robotsmali.org et ankataa.com étaient injoignables (échec DNS ou refus du proxy). github.com et raw.githubusercontent.com étaient joignables. Conséquences :
> - **Mesures directes sur les fichiers** pour deux corpus seulement : Bayelemabaga (archive GitHub v1.0.1) et MAFAND fr-bam (GitHub lafand-mt). Ces chiffres sont signalés « mesuré ».
> - Pour tout ce qui est hébergé sur Hugging Face ou décrit dans des articles arXiv/ACL, les faits viennent des **extraits du moteur de recherche** (cartes de jeux et résumés). Ils sont plausibles mais non relus en entier. Je les signale comme tels.

---

## Inventaire récapitulatif : quels corpus sont utilisables pour bm→fr, et avec quelles caractéristiques ?

### Takeaway
En dehors de Bayelemabaga et de Jeli-ASR, il n'existe que **peu de bm–fr écrit par des humains, librement téléchargeable**. Le seul ajout sûr et immédiat est **MAFAND-MT fr-bam** (3 013 paires d'entraînement de presse malienne, graphie standard, CC-BY-NC). Le reste se range en trois groupes :
- des données humaines **pivotées par l'anglais** (NLLB-Seed, SMOL, Tapo 2020 en partie) ;
- des données **minées ou synthétiques** de qualité inégale (NLLB minées : 56 151 paires après nettoyage ; synthétique Google Translate d'oza75 : environ 5 M de lignes) ;
- des **jeux de test** (FLORES+).

Pour la langue parlée de tous les jours, la ressource majeure récente est de la **parole spontanée transcrite sans traduction française** (afvoices, 423 h ; Kunkado, 160 h). Elle sert donc à la rétro-traduction, pas à l'apprentissage direct.

### Cited Findings

**Tableau d'inventaire.** Une source est donnée pour chaque ligne. « ? » marque ce que je n'ai pas pu confirmer.

| Corpus | Taille | Langues | Domaine | Qualité de traduction | Graphie | Accès / licence | Recouvrement / risque |
|---|---|---|---|---|---|---|---|
| **Bayelemabaga v1.0.1** (déjà utilisé) | 46 976 paires ; 37 580 / 4 698 / 4 698 ; ~650 k mots bm (mesuré) | bm–fr | contes, épopées, presse Fasokan, NT + Coran (~25 %), cours de grammaire | humaine (éditions bilingues du Corpus Bambara de Référence), mais alignement bruité | mixte : ~29 % des lignes sans ɛ/ɔ/ɲ (mesuré) | libre ; CC-BY-SA-4.0 dans le README, CC-BY-4.0 dans la barre latérale GitHub — [GitHub](https://github.com/RobotsMali-AI/datasets) | découpage **séquentiel** par source (voir la question 2) |
| **MAFAND-MT fr-bam** | 3 013 / 1 500 / 1 500 paires ; ~66 k mots bm en train (mesuré) | fr–bm | presse malienne 2021 | traduction humaine à partir du français | standard (≤ 1,6 % des lignes sans ɛ/ɔ, mesuré) | libre ; CC-BY-4.0-NC — [lafand-mt](https://github.com/masakhane-io/lafand-mt) | 0 phrase commune avec Bayelemabaga (mesuré) |
| **FLORES+ bam_Latn** | 997 dev + 1 012 devtest | multi (fra_Latn aligné) | Wikinews/Wikivoyage/Wikibooks | professionnelle, avec contrôle qualité | ? (non vérifiée) | CC BY-SA 4.0 ; zip protégé par mot de passe ; interdit de l'inclure dans l'entraînement — [flores-OLDI](https://github.com/avidale/flores-OLDI) | **à réserver au test** ; NLLB-200 a été développé sur FLORES |
| **NLLB-Seed / OLDI Seed** | ~6 000 phrases × 39 langues, dont bam_Latn | eng–bam (pas de fr) | Wikipédia (« articles every Wikipedia should have ») | professionnelle, **sans** l'assurance qualité de FLORES ; critiquée pour bam (voir la question 3) | sans tons | libre — [NLLB Seed](https://github.com/facebookresearch/flores/tree/main/nllb_seed), version actuelle sur HF `openlanguagedata/oldi_seed` ([dépôt](https://github.com/openlanguagedata/seed)) | probablement déjà vu à l'entraînement de NLLB-200 (inférence) |
| **Google SMOL** (SmolSent, SmolDoc) | ligne « bm » du tableau : 863 et 260 (sens des colonnes non confirmé) | source anglaise | phrases et documents variés | professionnelle | ? | HF `google/smol` — [article](https://arxiv.org/pdf/2502.12301) ; licence ? | inconnu |
| **NLLB minées (allenai/nllb)** fra–bam | 56 151 paires après nettoyage (version burkimbia) | fr–bm | web | **automatique** (minage par encodeur de phrases) | ? | HF `allenai/nllb` ([paires](https://huggingface.co/datasets/allenai/nllb/blob/main/nllb_lang_pairs.py)) ; version nettoyée et décontaminée de FLORES+ : [burkimbia/nllb-burkina-parallel](https://huggingface.co/datasets/burkimbia/nllb-burkina-parallel) | NLLB-200 a été entraîné sur ces données (inférence) |
| **Tapo et al. 2020** | 2 146 paires bm–fr | bm–fr / en | phrases d'exemple du dictionnaire SIL Mali | humaine (lexicographes) | ? | [LoResMT 2020](https://aclanthology.org/2020.loresmt-1.3/) ; disponibilité ? | possible recouvrement avec le dictionnaire SIL cité par l'article Bayelemabaga |
| **Tapo et al. 2021** (domaine santé) | ? | bm–fr | guide de santé | humaine | ? | [arXiv 2104.00041](https://arxiv.org/abs/2104.00041) ; publication annoncée, non vérifiée | probable recouvrement avec les fichiers `dogotoro_*` de Bayelemabaga (inférence) |
| **djelia/bambara-mt-dataset** | 10 k–100 k entrées | bm–fr–en | manuels scolaires EGAFE + livre d'histoire (Hoffman) | humaine ? | ? | cc-by-sa-4.0 (étiquette) — [HF](https://huggingface.co/datasets/djelia/bambara-mt-dataset) | ? |
| **oza75/bambara-mt** | ~4,99 M lignes | bm↔fr, bm↔en | dictionnaire (Bamadaba), synthétique (Wikipédia), sous-titres Mali Pense | dictionnaire = humain ; synthétique = **Google Translate** ; sous-titres = humain | ? | [HF](https://huggingface.co/datasets/oza75/bambara-mt) ; licence ? | ? |
| **kalilouisangare/corpus-parallel-en-fr-bam** | ? | en–fr–bm | presse | bm **produit par NLLB-200** | — | **à accès restreint** (coordonnées à partager) — [HF](https://huggingface.co/datasets/kalilouisangare/corpus-parallel-en-fr-bam) | à éviter : sorties du modèle même qu'on affine |
| **Tatoeba** | bam-eng : 7 phrases de test ; « train » OPUS : 162 392 | bm–en | — | mixte | — | [Tatoeba-Challenge v2023-09-26](https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/data/README.md) | pas de paire bam-fra listée |
| **NTREX-128** | — | — | — | — | — | [LANGUAGES.tsv](https://github.com/MicrosoftTranslator/NTREX) | **pas de bambara** |
| **AfroLingu-MT** | agrège 43 jeux, 84 paires ; « bam » présent | ? | ? | ? | ? | HF `UBC-NLP/AfroLingu-MT` — [Toucan](https://github.com/UBC-NLP/Toucan) | **agrégat** : probablement MAFAND/FLORES (inférence), risque de contamination |
| **Jeli-ASR** (déjà utilisé) | 32,48 h, 33 643 segments (carte HF) | audio bm + transcription + fr | récits de griots | humaine | ? | cc-by-4.0 (carte en cache) — [HF](https://huggingface.co/datasets/RobotsMali/jeli-asr) | 24 % de phrases longues du test présentes en train (mesure du projet) |
| **bam-asr-early** | 37,41 h, 38 769 segments | audio bm + transcription + fr | mixte | semi-experte | normalisée en v1.0.1 | [HF](https://huggingface.co/datasets/RobotsMali/bam-asr-early) | **contient ~3,5 h de Jeli-ASR** |
| **afvoices** (African Next Voices) | 612 h brutes / 423 h segmentées ; 874 762 lignes | audio bm + transcription (pas de fr confirmé) | parole **spontanée** sur des sujets familiers | transcription semi-automatique | ? | cc-by-4.0 — [HF](https://huggingface.co/datasets/RobotsMali/afvoices) | — |
| **Kunkado** | ~160 h ; 118 925 segments ; sous-ensemble relu de 33,47 h | audio bm + transcription | archives radio, alternance codique | humaine (partielle) | ? | licence ? — [arXiv 2512.19400](https://arxiv.org/abs/2512.19400) | — |
| **Corpus Bambara de Référence** | ~11,8 M mots (juillet 2025), dont 2,1 M désambiguïsés | bm (glosé en fr) | tous genres | — | standard, avec tons | consultation en ligne ; licence et téléchargement ? — [cormand](https://cormand.huma-num.fr/) | **source de Bayelemabaga** |
| **MADLAD-400 bm** | clean 1,17 Mo gz ; noisy 15,1 Mo gz | bm | web | — | ? | ODC-BY — [HF](https://huggingface.co/datasets/allenai/MADLAD-400/tree/95bb50141e1945da136b7968b99e15065a803835/data/bm) | — |
| **Wikipédia bm** | ~930 articles | bm | encyclopédique | — | variable | CC BY-SA — [en.wikipedia](https://en.wikipedia.org/wiki/Bambara_Wikipedia) | — |

### Inferences
- **Ajout le plus rentable et le moins risqué pour bm→fr :** MAFAND fr-bam. Il est humain, en graphie standard, d'un domaine (presse 2021) absent de Bayelemabaga et sans recouvrement mesuré. Il reste petit (3 k paires) et sa licence interdit l'usage commercial.
- **NLLB-Seed et SMOL** n'ont pas de côté français. Pour s'en servir en bm→fr, il faut traduire le côté anglais en français par une MT en→fr de haute qualité (pivot synthétique, côté cible quasi humain), ou les garder pour un entraînement multi-cible bm→en/fr.
- **Données minées NLLB et sous-ensembles synthétiques d'oza75 :** NLLB-200-distilled-600M a très probablement déjà vu les premières, donc le gain attendu est faible. Dans les seconds, le **bambara est la sortie de Google Translate** alors que le français est authentique. En bm→fr, c'est exactement le schéma de la rétro-traduction (source synthétique, cible réelle). C'est acceptable en mélange mesuré, mais cela n'apprend pas le bambara parlé.

### Gaps
- Comptes exacts sur HF (lignes par sous-ensemble d'oza75/bambara-mt, de djelia/bambara-mt-dataset, de SMOL bm, de la paire fra–bam brute d'allenai/nllb) : non vérifiés, Hugging Face étant injoignable.
- Licences non confirmées : SMOL (CC-BY-4.0 selon mes souvenirs, non vérifié ici), oza75/bambara-mt, Kunkado, allenai/nllb (ODC-BY selon mes souvenirs, non vérifié).

---

## Bayelemabaga : sources, domaines, part religieuse, caractère conversationnel, qualité

### Takeaway
Bayelemabaga v1.0.1 est tiré du Corpus Bambara de Référence. Il réunit des contes, des épopées, des numéros du mensuel Fasokan, un cours de grammaire et un manuel, **le Nouveau Testament entier et des sourates du Coran (~25 % des lignes)**. Le jeu publié sur GitHub présente un **découpage séquentiel** (train, dev puis test dans l'ordre alphabétique des sources). En conséquence, le **test ne contient presque aucun texte religieux et est écrit à ~76 % en graphie ancienne sans ɛ/ɔ**, tandis que le train compte 27,6 % de texte religieux. Il y a un peu de dialogue (roman, manuels), mais ce n'est pas un corpus conversationnel.

### Cited Findings
- Le README annonce « 44562 aligned machine translation ready Bambara-French lines » issues du Corpus Bambara de Référence, extraites de 264 fichiers : « periodicals, books, short stories, blog posts, part of the Bible and the Quran ». L'instantané indique 46 976 lignes, 691 312 tokens français, 660 732 tokens bambara (daba), et un découpage train 37 580 / valid 4 698 / test 4 698 (80/10/10). Il note aussi : « We are working on resolving some last minute misalignment issues » ; le symbole ʃ est laissé dans les données ; la licence affichée est `CC-BY-SA-4.0` (alors que la barre latérale GitHub affiche CC-BY-4.0) ; version 1.0.1 — [README GitHub RobotsMali-AI/datasets](https://github.com/RobotsMali-AI/datasets)
- Le `meta.json` de l'archive indique une version antérieure de 44 160 lignes issues de 231 sources — [archive bayelemabaga.tar.gz](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/bayelemabaga.tar.gz)
- L'article NAACL 2025 (Tapo et al., « Bayelemabaga: Creating Resources for Bambara NLP », p. 12060–12070) parle de ~47 K paires issues de 231 sources, « from informal content to formal documents to religious literature (passages from the Bible and the Quran) ». Selon les extraits, les sources comprennent le projet Dokotoro, des livres religieux, des phrases du dictionnaire SIL et le Corpus Bambara de Référence. Les modèles ont aussi été évalués sur trois petits jeux hors domaine (Dictionary, Medical, News) — [ACL Anthology](https://aclanthology.org/2025.naacl-long.602/) (via extraits de recherche, PDF non ouvert)
- Lacuna Fund décrit le corpus comme l'extension et la correction d'un jeu bilingue antérieur, avec 46 976 paires issues du Corpus Bambara de Référence (LLACAN, INALCO) — [Lacuna Fund](https://lacunafund.org/?p=9200)
- **Composition par fichier** (264 entrées de `sources.txt`, comptées par préfixe) — [sources.txt](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/bayelemabaga/sources.txt) :
  - 73 numéros ou articles du journal **Fasokan** (2009–2014 : téléphone, internet, actualité), 2 de **Jɛkabaara**, 3 de « fakan » (dont un VOA) ;
  - 41 **contes** Görög-Meyer ;
  - **33 leçons du cours de grammaire de Vydrine**, et le manuel **Bird, Hutchison & Kanté, « An ka bamanankan kalan »** ;
  - **27 livres du Nouveau Testament** (« layidu_kura ») et **5 fichiers de Coran** (« kurane ») ;
  - 6 épisodes de l'épopée de Ségou (Dumestre), un Soundiata (kone-sunjata) ;
  - 8 Bailleul « sagesse bambara », 8 Oteri Keyita « sogobo », 8 Keita « folo kita » ;
  - 2 textes de santé (« dogotoro… dogotoro_te_sigida_min_na », soit *Là où il n'y a pas de docteur*), 1 « nye_keneya » ;
  - un roman à dialogues (« kuyate-sous_l_orage »), etc.
- **Part religieuse, mesurée** dans `text.fr`, qui suit l'ordre de `sources.txt` : le bloc Coran va des lignes 26 565 à 28 523 (1 959 lignes) et le bloc Nouveau Testament de 29 181 (« Ascendance de Jésus ») à 39 073 (« Que la grâce du Seigneur Jésus soit avec tous ! »), soit 9 893 lignes. Au total, **11 852 lignes ≈ 25,2 %** du corpus — mesure sur [l'archive GitHub](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/bayelemabaga.tar.gz)
- **Découpage séquentiel, mesuré** en retrouvant la position de chaque ligne des fichiers de partition dans `text.fr` :
  - train = lignes 1–37 580 ;
  - dev = lignes 37 581–42 278 ;
  - test = lignes 42 279–46 976 (à quelques doublons près).

  D'où un **texte religieux à 27,6 % en train (10 359 lignes), 31,8 % en dev (1 493 lignes de la fin du NT) et ~0 % en test**. Le test couvre la fin alphabétique des sources : contes (Sidibé, Sunbunu…), épopées historiques (Almoravides, princes peuls), texte ethnographique sur le mariage, cours de grammaire de Vydrine — mesure sur [l'archive GitHub](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/bayelemabaga.tar.gz)
- **Graphie, mesurée** (part des lignes bm de 6 mots ou plus sans aucun ɛ, ɔ ni ɲ, indice d'ancienne graphie ou de graphie aplatie) : train 23,2 %, dev 34,2 %, **test 76,3 %**, ensemble 29,0 %. Exemples du test : « walasa kow ka segin u cogo koro la » (pour kɔw… kɔrɔ), « Nk'olu dun ye mogo fila de ye » (pour mɔgɔ). Le corpus entier contient 95 540 ɛ, 83 065 ɔ, 11 102 ɲ, 505 ŋ et 63 ʃ — mesure sur [l'archive GitHub](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/bayelemabaga.tar.gz)
- **Bruit, mesuré** — sur [l'archive GitHub](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/bayelemabaga.tar.gz) :
  - 1 220 copies de paires en double ;
  - 638 paires dont le rapport de longueur fr/bm en caractères dépasse 3 ou tombe sous 1/3, et 3 705 au-delà de 2 ou sous 1/2 (indice de désalignement) ;
  - 3 434 lignes bm de 3 mots ou moins ;
  - test/train : 6 lignes françaises et 23 lignes bambara du test se retrouvent en train ;
  - longueur médiane de 11 mots bm (moyenne 13,8) ; 30 % des lignes ont 7 mots ou moins.
- **Caractère conversationnel, mesuré** : 2 071 lignes françaises commencent par un nom de locuteur (« Benfa: Sibiri, Sibiri, » / « Sibiri: Oui! », roman *Sous l'orage* ; dialogues des manuels) et 2 085 finissent par « ? ». Environ 20–25 % des lignes françaises contiennent « je/tu/nous/vous » — mesure sur [l'archive GitHub](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/bayelemabaga.tar.gz)
- Un article tiers de 2025 rapporte qu'un LLaMA 3 8B obtenait le meilleur score sur le test Bayelemabaga, avec BLEU 9,82 et chrF 19,00 — [arXiv 2509.12514](https://arxiv.org/html/2509.12514) (via extrait de recherche)

### Inferences
- **Le chrF++ de 27,8 sur le test Bayelemabaga mesure surtout la généralisation vers des contes, épopées et leçons de grammaire en graphie ancienne**, pas vers la langue de tous les jours. Un quart des données d'entraînement est biblique ou coranique, ce qui peut pousser le modèle vers un registre biblique (« Et il leur dit… »). Ce biais s'accorde avec le jugement du locuteur natif sur sa propre parole.
- **À vérifier avant toute conclusion :** la version Hugging Face reprend-elle bien ce découpage séquentiel ? La carte HF annonce 46 976 lignes, et 93 952 lignes au total sur la page, soit 2 × 46 976 ; c'est donc probablement les mêmes données, dans les deux sens. Contrôle simple dans le projet : (a) mesurer la part de lignes bm sans ɛ/ɔ dans le test HF (~76 % attendu si c'est la même partition) ; (b) chercher « Jésus » dans le test HF (~0 attendu).
- Pour une évaluation plus représentative, on peut **re-découper Bayelemabaga par source**, en écartant explicitement NT et Coran, et **normaliser la graphie** (o→ɔ, e→ɛ impossible de façon déterministe : il faut un lexique de type Bamadaba, ou exclure les lignes sans ɛ/ɔ de l'évaluation).
- Les fichiers `dogotoro_*` correspondent probablement au même manuel de santé que le corpus de domaine de Tapo et al. (2021) : il y a là un risque de recouvrement si l'on ajoute ce dernier.

### Gaps
- Je n'ai pas pu ouvrir le PDF NAACL 2025 : pas de tableau par source, et la mention « SIL dictionary sentences » ne se retrouve pas dans `sources.txt`. La version de l'article diffère peut-être de la v1.0.1 sur GitHub.
- Je n'ai pas pu vérifier si la version HF a été re-mélangée après 2023.
- Recouvrement avec FLORES non mesuré (FLORES+ est distribué en zip protégé). Par construction (Wikinews/Wikivoyage/Wikibooks traduits pour FLORES, contre des éditions bilingues maliennes pour Bayelemabaga), le recouvrement devrait être nul. C'est une inférence.

---

## Jeux internationaux : MAFAND/Lafand, FLORES-200/FLORES+, NLLB-Seed, SMOL/GATITOS, NTREX, Tatoeba, OPUS, WMT, AfroLingu-MT

### Takeaway
- **Utiles et libres :** MAFAND fr-bam (seul jeu humain fr↔bm de presse), FLORES+ (évaluation uniquement), NLLB-Seed (eng–bam, critiqué), SMOL (source anglaise).
- **Inutiles ou absents pour le bambara :** NTREX-128 (pas de bambara), Tatoeba (7 phrases), WMT22 (bambara non confirmé).
- **À manier avec prudence :** AfroLingu-MT, un agrégat qui risque de contenir des jeux de test.

### Cited Findings
- **MAFAND-MT / Lafand :**
  - Le dépôt `lafand-mt` crée « a newly created MT dataset in the news domain » pour 16 langues, dont « Bambara (bam) ». Licence : « CC-BY-4.0-NC ; the French/English source texts we translated from have different licenses depending on the news website license » — [masakhane-io/lafand-mt](https://github.com/masakhane-io/lafand-mt). « Lafand » est le nom du dépôt et MAFAND celui du jeu : c'est la même ressource. Le répertoire `data/json_files/fr-bam` existe — [GitHub](https://github.com/masakhane-io/lafand-mt/tree/main/data/json_files)
  - Mesuré : fr-bam compte **3 013 paires en train, 1 500 en dev et 1 500 en test**, soit 66 279 / 35 313 / 33 620 mots bm, avec une médiane de 19–20 mots par phrase. La graphie est standard (1,6 % / 0,8 % / 1,2 % des lignes sans ɛ/ɔ/ɲ). Les textes sont des articles de presse malienne de juin 2021 (ex. : visite du Premier ministre Choguel Kokalla Maïga à la Maison de la Presse). Aucune phrase en commun avec Bayelemabaga (comparaison exacte normalisée, bm et fr) ; 24 phrases bm du test se retrouvent en train. Fichiers : [test.json](https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/json_files/fr-bam/test.json), [train.json](https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/json_files/fr-bam/train.json)
- **FLORES+ (successeur de FLORES-200) :** 997 phrases en `dev` et 1 012 en `devtest` par langue ; bam_Latn figure dans la couverture. Licence CC BY-SA 4.0. Les données sont distribuées en zip protégé par mot de passe, avec la demande de ne pas les rediffuser en clair ni les inclure dans des données d'entraînement — [avidale/flores-OLDI (miroir)](https://github.com/avidale/flores-OLDI). Version officielle : HF `openlanguagedata/flores_plus`, à accès restreint (acceptation des conditions requise, selon mes souvenirs, non revérifié ici).
- **NLLB-Seed :** « professionally-translated sentences in the Wikipedia domain… around six thousand sentences in 39 languages… meant to be used for training… does not go through the human quality assurance process present in FLORES-200 ». Bambara | bam_Latn figure dans la liste. Le dépôt n'est plus mis à jour ; les nouvelles versions sont sur github.com/openlanguagedata/seed puis sur HF `openlanguagedata/oldi_seed` — [facebookresearch/flores nllb_seed](https://github.com/facebookresearch/flores/tree/main/nllb_seed) ; [openlanguagedata/seed](https://github.com/openlanguagedata/seed)
- **Critique de la qualité bam_Latn de NLLB-Seed :** des traducteurs N'ko qui ont relu les données bambara latines signalent « too much French vocabulary and too little Manding vocabulary, entries that do not match the English counterpart at all, and entries fully in French », ainsi que l'absence totale de tons — [Doumbouya et al., WMT 2023, MT pour le N'ko](https://aclanthology.org/2023.wmt-1.34.pdf) (via extrait de recherche)
- **Critique générale de FLORES+ :** une réévaluation de 2025 sur quatre autres langues (pas le bambara) trouve une qualité souvent inférieure au seuil annoncé de 90 % — [arXiv 2508.20511](https://arxiv.org/html/2508.20511v1) (via extrait)
- **Google SMOL (2025) :**
  - « Professionally translated parallel data for 115 under-represented languages » ; le tableau contient une ligne « bm | Bambara | Latn | Africa | 863 | 260 » — [arXiv 2502.12301](https://arxiv.org/pdf/2502.12301) ; publié à [WMT 2025](https://aclanthology.org/2025.wmt-1.85/)
  - SmolSent est décrit comme 863 phrases **anglaises** traduites (source anglaise) — [revue Slator](https://slator.com/google-expands-low-resource-ai-translation-with-smol-dataset/), [arXiv 2502.12301](https://arxiv.org/pdf/2502.12301)
  - La carte HF actuelle indique « SmolDoc: document-level translations into 106 [puis 130] language pairs » et « SmolSent: … 88 [puis 114] language pairs » ; une mise à jour d'avril 2026 cite « bm » — [HF google/smol](https://huggingface.co/datasets/google/smol)
  - La vérification par identification de langue n'a montré de confusions qu'entre dialectes, dont « bm/dyu » — [arXiv 2502.12301](https://arxiv.org/html/2502.12301v1)
  - L'article Omnilingual MT (2026) rapporte des résultats bam_Latn avec SmolDoc et SmolSent comme données d'amorçage — [arXiv 2603.16309](https://arxiv.org/pdf/2603.16309)
- **GATITOS :** a rejoint le dépôt HF de SMOL (« GATITOS has moved to Huggingface, to join its sister dataset SMOL ») — [google-research/url-nlp gatitos](https://github.com/google-research/url-nlp/tree/main/gatitos). GATITOS est un lexique de jetons et de phrases courtes, et SMOL « supplements the token-level translations from Gatitos » — [arXiv 2502.12301](https://arxiv.org/html/2502.12301v1)
- **NTREX-128 :** `LANGUAGES.tsv` liste 128 langues et plus, dont fuc (pulaar), wol, mey (hassaniyya), shi, mais **aucune entrée bam/bm**. Licence CC BY-SA 4.0 — [MicrosoftTranslator/NTREX](https://github.com/MicrosoftTranslator/NTREX)
- **Tatoeba Challenge v2023-09-26 :** « Bambara - English [bam-eng] | 7 (test) | — | 162392 (train) » et « bam-bam | 1 » ; aucune paire bam-fra n'est listée — [Tatoeba-Challenge data README](https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/data/README.md)
- **NLLB minées / OPUS :**
  - La liste NLLB_PAIRS d'allenai/nllb contient bam_Latn–fra_Latn — [nllb_lang_pairs.py](https://huggingface.co/datasets/allenai/nllb/blob/main/nllb_lang_pairs.py)
  - Version nettoyée : config fra-bam de **56 151 paires** (sans pivot), sur un total de 588 640 paires pour 14 couples. Nettoyage : suppression des déchets et copies non traduites, filtre de rapport de longueur, GlotLID, dédoublonnage 1-1, **décontamination FLORES+**, sans seuil sur le score d'alignement — [burkimbia/nllb-burkina-parallel](https://huggingface.co/datasets/burkimbia/nllb-burkina-parallel) (via extrait)
  - OPUS en a une copie — [OPUS NLLB](https://opus.nlpl.eu/legacy/NLLB.php)
- **WMT :** la tâche WMT'22 « Large-Scale MT Evaluation for African Languages » couvrait 24 langues africaines avec le français et l'anglais, évaluées sur FLORES — [Findings WMT22](https://aclanthology.org/2022.wmt-1.72). Un extrait affirmait que le bambara y figurait, mais la liste citée (« Amharic, Bambara, Ghomala, Ewe, Fon… ») est celle de **MAFAND**, pas de WMT22. Le système de Tencent liste afr, amh, fuv, kam, luo, hau, ibo, kin, lug, nso, nya… sans bam — [Tencent WMT22](https://arxiv.org/pdf/2210.09644). Le bambara à WMT22, WMT23 ou WMT24 n'est pas confirmé.
- **AfroLingu-MT (Toucan, ACL Findings 2024) :** « data from a total of 43 datasets, encompassing 84 unique language pairs derived from 46 different languages », plus un nouveau jeu traduit à la main dans le domaine gouvernemental ; « bam »: « Bambara » figure dans la table des langues ; HF `UBC-NLP/AfroLingu-MT` — [UBC-NLP/Toucan](https://github.com/UBC-NLP/Toucan)

### Inferences
- **MAFAND** est le complément naturel de Bayelemabaga : autre domaine (presse moderne, institutions, vocabulaire politique), graphie standard, traductions humaines, et licence NC acceptable pour un projet de recherche. Son **test de 1 500 phrases** peut servir de second jeu d'évaluation propre, non religieux. Attention : NLLB-200 n'a pas été entraîné sur MAFAND à ma connaissance (non vérifié).
- **FLORES+ devtest** (1 012 phrases, fra_Latn et bam_Latn alignés) est le jeu de test public de référence. Il ne doit pas servir à l'entraînement, et NLLB-200 a été sélectionné sur dev.
- Les **NLLB minées** (56 k fra–bam nettoyées) sont probablement déjà dans les données d'entraînement de NLLB-200. Le gain attendu en affinage est faible, et le bruit peut dégrader le modèle.
- **AfroLingu-MT** reprend des jeux existants. Il ne faut l'utiliser qu'après dédoublonnage contre MAFAND test et FLORES.

### Gaps
- Nombre exact d'entrées SMOL pour bm (sens des colonnes 863 / 260), présence de bm dans GATITOS, et licence SMOL : non vérifiés (HF et arXiv injoignables).
- Taille et provenance des 162 392 paires « train » bam-eng de Tatoeba Challenge (OPUS : probablement Bible, NLLB, wikimedia…) : non vérifiées.
- Liste des corpus OPUS contenant bm et leurs tailles : OPUS était injoignable.
- Sous-jeux bam d'AfroLingu-MT et leur caractère restreint ou non : non vérifiés.
- Je n'ai trouvé aucun jeu Masakhane bm–fr hors MAFAND.

---

## Tapo et al. et jeux maliens sur Hugging Face (RobotsMali, MALIBA-AI, Djelia, oza75, sudoping01, autres)

### Takeaway
Tapo et al. ont publié en 2020 un petit jeu tiré du **dictionnaire SIL** (2 146 paires bm–fr) et, en 2021, un jeu **de domaine santé** issu d'un guide de santé (taille non confirmée). Les jeux maliens plus récents sur HF sont surtout **synthétiques** (Google Translate, NLLB) ou **agrégés**. Exception : djelia/bambara-mt-dataset (manuels scolaires et livre d'histoire, trilingue) et le sous-ensemble de sous-titres d'oza75 (traduit à la main).

### Cited Findings
- **Tapo et al. 2020** (LoResMT) : « the first parallel data set for machine translation of Bambara into and from English and French ». La partie bm–fr compte 2 146 phrases parallèles — [ACL Anthology 2020.loresmt-1.3](https://aclanthology.org/2020.loresmt-1.3/) ; [arXiv 2011.05284](https://arxiv.org/pdf/2011.05284). Les données reposent sur les phrases d'exemple d'un dictionnaire SIL Mali, traduites en français, anglais et espagnol — [arXiv 2104.00041](https://arxiv.org/pdf/2104.00041) (via extrait)
- **Tapo et al. 2021**, « Domain-specific MT for Low-resource Languages: The case of Bambara-French » : présente « the first domain-specific parallel dataset for MT of Bambara into and from French ». Selon un extrait, les auteurs « built [their] own dataset from a health-care guide… and plan to release the data publicly » — [arXiv 2104.00041](https://arxiv.org/abs/2104.00041)
- **Yiri (agriculture et médecine) :** un article décrit des corpus français–bambara agriculture et médecine construits à partir du manuel Dokotoro, de Bayelemabaga, de MAFAND-MT, du corpus NLLB et d'un « Bambara Lexicon » — [NLP4TIA 2023, p. 39–47](https://aclanthology.org/2023.nlp4tia-1.6.pdf) (via extrait ; l'attribution exacte de ce PDF à « Yiri » n'est pas confirmée)
- **djelia/bambara-mt-dataset :** trilingue avec des champs `bm`, `fr`, `en` ; il s'appuie sur le projet de manuels scolaires **EGAFE** (RobotsMali) et sur « La Guerre des Griots de Kita 1985 » (Barbara G. Hoffman). Taille 10 K–100 K, étiquette cc-by-sa-4.0 — [HF djelia/bambara-mt-dataset](https://huggingface.co/datasets/djelia/bambara-mt-dataset)
- **oza75/bambara-mt :** ~4,99 M lignes, mis à jour le 7 décembre 2024. Sous-ensembles :
  - `fr-bm-dictionary` (Bamadaba) ;
  - synthétiques fr→bm, bm→fr, bm→en, en→bm, produits par **Google Translate** à partir de Wikipédia et d'autres jeux HF ;
  - `dataset_source` = transcription, avec des sous-titres « Mali Pense » traduits par des humains.

  Les entrées très courtes ou vides ont été retirées — [HF oza75/bambara-mt](https://huggingface.co/datasets/oza75/bambara-mt). Un jeu Kaggle apparenté existe : [ozaresearch1/bambara-french-parallel-dataset](https://www.kaggle.com/datasets/ozaresearch1/bambara-french-parallel-dataset)
- **djelia / oza75 bambara-lm-qa :** jeu d'instructions pour LLM, avec des tâches de traduction fr↔bm et en↔bm ; la partie Alpaca est « automatically translated using Google Translate » — [HF djelia/bambara-lm-qa](https://huggingface.co/datasets/djelia/bambara-lm-qa)
- **sudoping01 :**
  - `sudoping01/datasets-bambara` sur GitHub est un **fork de Bayelemabaga**, sans données nouvelles — [GitHub](https://github.com/sudoping01/datasets-bambara)
  - `sudoping01/maliba-llm` (MIT) a été affiné sur « a cleaned subset of the MALIBA-Instructions dataset (1M examples) » — [HF](https://huggingface.co/sudoping01/maliba-llm)
- **MALIBA-AI :** un blog tiers indique que MALIBA-AI a publié en 2025 un « bambara-mt-dataset » consolidant plusieurs sources, le plus grand à l'époque, et que son application BambaraTranslator repose sur NLLB. Le même billet note que des applications d'oza75 n'étaient plus disponibles en janvier 2026 — [An ka taa, « Manding Language Tech Resources »](https://www.ankataa.com/blog/2024/6/27/manding-language-tech-resources-and-initiatives) (via extrait). Je n'ai pas trouvé l'identifiant HF exact.
- **kalilouisangare/corpus-parallel-en-fr-bam :** à accès restreint (« You need to agree to share your contact information ») ; le côté français d'un corpus de presse en–fr y est traduit en bambara **avec NLLB-200**, et la carte dit « No professional human translations available » — [HF](https://huggingface.co/datasets/kalilouisangare/corpus-parallel-en-fr-bam)
- Autres jeux repérés, de contenu non vérifié : [OumarDicko/bambara](https://huggingface.co/datasets/OumarDicko/bambara) ; un jeu bambara d'analyse de sentiment — [arXiv 2108.02524](https://arxiv.org/pdf/2108.02524)

### Inferences
- **EGAFE** (manuels scolaires) est le seul ajout « humain » de ce groupe susceptible de contenir des phrases simples de la vie courante. Il faut vérifier sa taille, sa graphie et son recouvrement avec Bayelemabaga avant de l'utiliser.
- Les sous-ensembles **synthétiques Google Translate** d'oza75 utilisés en bm→fr fournissent du bambara artificiel en entrée et du français réel en sortie. Ils peuvent aider la fluidité du français, mais pas la compréhension du bambara parlé, et ils ont le style de Google Translate. **Le jeu de Kalilou Sangaré (bm généré par NLLB-200) est à exclure** : on réapprendrait au modèle ses propres sorties.
- Le **dictionnaire Bamadaba** (dans oza75/bambara-mt) et le dictionnaire SIL (Tapo 2020) fournissent des paires courtes, mot ou expression, utiles contre les hallucinations sur les énoncés de 1 à 3 mots, qui sont fréquents à l'oral.

### Gaps
- Taille et lien du jeu de santé de Tapo 2021, et disponibilité réelle de celui de 2020 : non confirmées.
- Identifiant HF et taille du « bambara-mt-dataset » de MALIBA-AI, existence publique de MALIBA-Instructions : non confirmés.
- Nombre de lignes par sous-ensemble d'oza75/bambara-mt, en particulier le volume des sous-titres Mali Pense traduits à la main : inconnu.

---

## Bambara conversationnel et quotidien (SMS, réseaux sociaux, radio, dialogues, guides de conversation, Peace Corps, Wikipédia)

### Takeaway
**Aucun corpus parallèle bm–fr conversationnel de taille significative n'a été trouvé.** Les meilleures sources de bambara quotidien sont :
- des **transcriptions de parole spontanée sans français** : afvoices (612 h brutes / 423 h segmentées, sujets familiers) et Kunkado (radio, 160 h) ;
- les **sous-titres Mali Pense** d'oza75 (traduits à la main, taille inconnue) ;
- les **dialogues des manuels** déjà présents dans Bayelemabaga.

Je n'ai trouvé aucune source pour les SMS, les réseaux sociaux ou les manuels Peace Corps.

### Cited Findings
- **afvoices (African Next Voices, RobotsMali) :** 612 h de parole spontanée collectées dans le sud du Mali, sur des sujets familiers aux locuteurs ; environ 626,32 h enregistrées, 612 h traitées ; transcription semi-automatique ; modèles ASR affinés et testés sur un sous-ensemble de 101 h — [arXiv 2511.18557](https://arxiv.org/abs/2511.18557) / [AfricaNLP 2026](https://aclanthology.org/2026.africanlp-main.1/). La carte HF indique 423 h segmentées, 874 762 lignes et cc-by-4.0 — [HF RobotsMali/afvoices](https://huggingface.co/datasets/RobotsMali/afvoices). Un blog tiers parle de 512 locuteurs — [Lanfrica](https://lanfrica.com/en/blog/insight/understanding-the-african-next-voices-datasets/)
- **Kunkado :** ~160 h issues d'archives radio maliennes, « spontaneous speech with code-switching and background noise » ; 33,47 h relues par des humains — [arXiv 2512.19400](https://arxiv.org/html/2512.19400v1)
- **oza75/bambara-mt**, sous-ensemble « transcription » : paires de sous-titres « Mali Pense » français–bambara, traduites par des humains — [HF oza75/bambara-mt](https://huggingface.co/datasets/oza75/bambara-mt) (via extrait)
- **Dialogues dans Bayelemabaga, mesurés :** 2 071 lignes avec nom de locuteur et 2 085 questions. Ils viennent notamment du roman *Sous l'orage* (Kuyate), du manuel Bird/Hutchison/Kanté « An ka bamanankan kalan » et des 33 leçons de Vydrine — [sources.txt](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/bayelemabaga/sources.txt), [archive](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/bayelemabaga.tar.gz)
- **Fasokan** (73 fichiers dans Bayelemabaga) aborde des sujets quotidiens : téléphone (« telefoni_yaalata_batiri »), internet (« enterineti_kalanni »), mototaxis, mangues, football — [sources.txt](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/bayelemabaga/sources.txt)
- **Wikipédia bm :** environ 930 articles ; lancée en 2005 — [Wikipedia, « Bambara Wikipedia »](https://en.wikipedia.org/wiki/Bambara_Wikipedia). Wikidata donne 864 pages d'articles — [Wikidata Q8559737](https://www.wikidata.org/wiki/Q8559737)

### Inferences
- Pour rapprocher la traduction de la parole du locuteur natif, la voie la plus prometteuse est la **rétro-traduction en sens inverse appliquée aux transcriptions d'afvoices et de Kunkado** : on traduit le bambara spontané en français avec le meilleur modèle disponible, puis on filtre, ou on fait relire un échantillon par un locuteur. Dans le schéma classique de rétro-traduction pour bm→fr, il faudrait plutôt du **français conversationnel authentique** (sous-titres français, corpus de dialogue) traduit **vers** le bambara, pour obtenir une source synthétique et une cible réelle. Mais un bambara synthétique produit par NLLB ne ressemble pas à la parole réelle.
- Le plus rentable à petite échelle est sans doute un **petit jeu bm–fr de phrases quotidiennes traduites à la main** : quelques centaines à quelques milliers de phrases, tirées des transcriptions d'afvoices ou de Kunkado et traduites par des locuteurs. Il servirait à la fois au test et à l'affinage. Le projet a déjà un outil de collecte (`app/collect_app.py`).

### Gaps
- Corpus SMS, WhatsApp ou réseaux sociaux en bambara : rien de trouvé. Le jeu de sentiment (arXiv 2108.02524) n'a pas été vérifié.
- Manuels Peace Corps Mali et guides de conversation numérisés et alignés : je n'ai trouvé aucune source fiable.
- Taille et droits du sous-ensemble Mali Pense : inconnus.
- Je n'ai pas vérifié si afvoices propose des traductions françaises ; la carte n'en mentionne pas dans les extraits.

---

## Texte monolingue bambara pour la rétro-traduction (Corpus Bambara de Référence, MADLAD-400, CC-100/Glot500/FineWeb-2, Kibaru, Jɛkabaara)

### Takeaway
La principale source monolingue de qualité est le **Corpus Bambara de Référence** (~11,8 M de mots en juillet 2025, graphie standard avec tons). Il est consultable en ligne, mais son téléchargement et sa licence restent à confirmer, et Bayelemabaga en est extrait, d'où un risque de recouvrement. Les corpus web (MADLAD-400 bm : quelques Mo compressés ; FineWeb-2 bam_Latn) sont petits et bruités. Pour la langue parlée, **les transcriptions d'afvoices et de Kunkado** sont la meilleure source monolingue.

### Cited Findings
- **Corpus Bambara de Référence :**
  - ~11,8 millions de mots au total à la mise à jour de juillet 2025, dont un sous-corpus désambiguïsé de 2,1 M de mots — [cormand.huma-num.fr](https://cormand.huma-num.fr/) (via extrait de recherche ; site injoignable directement)
  - En mars 2017 : 3 846 094 mots, dont ~700 000 désambiguïsés — [Vydrin 2017, HAL](https://halshs.archives-ouvertes.fr/halshs-01584112v1/file/2017vydrin_electronic_library.pdf)
  - Entièrement tonalisé, étiqueté en parties du discours et glosé en français ; décrit comme « probably the only one freely accessible on the Internet » parmi les corpus africains de plus d'un million de mots — [Bamana Reference Corpus (BRC), HAL](https://halshs.archives-ouvertes.fr/halshs-00908108)
  - Sous-corpus « corbama » brut, non désambiguïsé et désambiguïsé — [An ka taa, « bambara corpus how to »](https://www.ankataa.com/blog/bambara-corpus-how-to)
- **MADLAD-400 bm :** un fichier `bm_clean_0000.jsonl.gz` (~1,17 Mo) et un `bm_noisy_0000.jsonl.gz` (~15,1 Mo) — [HF allenai/MADLAD-400 data/bm](https://huggingface.co/datasets/allenai/MADLAD-400/tree/95bb50141e1945da136b7968b99e15065a803835/data/bm). Licence ODC-BY — [README MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400/blob/01ad5c885296970ecf9828c14218b8fd75436774/README.md). La version « noisy » n'a qu'un filtrage par identification de langue au niveau du document — [arXiv 2309.04662](https://arxiv.org/pdf/2309.04662)
- **FineWeb-2 :** 1 868 couples langue-écriture ; bam_Latn existe. Une carte de modèle d'identification de langue précise que **dyu_Latn (dioula) a été fusionné dans bam_Latn** et que les étiquettes FineWeb-2 viennent d'un identifiant automatique, avec des pages hors langue — [HF fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) ; [Horizon-Labs/language-detection-small](https://huggingface.co/Horizon-Labs/language-detection-small)
- **Wikipédia bm :** ~930 articles — [Wikipedia](https://en.wikipedia.org/wiki/Bambara_Wikipedia) ; une version HF prétraitée existe — [wikilangs/bm](https://huggingface.co/wikilangs/bm)
- **Jɛkabaara et Fasokan** sont déjà en partie dans Bayelemabaga (2 fichiers « jekabaara273_* », 73 « Fasokan… ») — [sources.txt](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/bayelemabaga/sources.txt)

### Inferences
- MADLAD bm clean pèse 1,17 Mo compressé, soit probablement **de l'ordre de quelques centaines de milliers de mots**. C'est une estimation par le volume, non mesurée ; elle est à confirmer en comptant les lignes après téléchargement. Elle est plus petite que le Corpus Bambara de Référence.
- La fusion dioula/bambara dans FineWeb-2 impose un **filtrage** (lexique ou GlotLID) avant toute rétro-traduction.
- Toute rétro-traduction à partir du Corpus Bambara de Référence doit **exclure les textes déjà dans Bayelemabaga**, en particulier ceux de son test, sous peine de contamination.

### Gaps
- Tailles de bm dans CC-100 (probablement absent), Glot500-c et FineWeb-2 bam_Latn : non trouvées.
- Kibaru (journal en bambara de l'AMAP) : je n'ai trouvé ni archive numérisée, ni taille, ni licence.
- Licence et possibilité de télécharger en masse le Corpus Bambara de Référence : non confirmées (site injoignable).

---

## Jeux de parole avec traductions françaises (Jeli-ASR, Kunkado, bam-asr-early, autres)

### Takeaway
Seuls **Jeli-ASR** (~30–32 h de récits de griots) et **bam-asr-early** (37,41 h) associent l'audio bambara à une traduction française. bam-asr-early **inclut ~3,5 h de Jeli-ASR**, ce qui crée un risque de fuite vers le test Jeli-ASR. Kunkado (radio, 160 h) et afvoices (spontané, 423 h segmentées) sont plus proches de la parole réelle, mais **sans traduction française confirmée**.

### Cited Findings
- **Jeli-ASR :**
  - La carte HF indique 32,48 h, 33 643 échantillons, avec transcriptions bambara et traductions françaises — [HF RobotsMali/jeli-asr](https://huggingface.co/datasets/RobotsMali/jeli-asr) ; le GitHub parle de ~30 h de récits et narrations de griots — [GitHub RobotsMali-AI/jeli-asr](https://github.com/RobotsMali-AI/jeli-asr)
  - Licence cc-by-4.0 dans une version en cache de la carte — [README HF (révision)](https://huggingface.co/datasets/RobotsMali/jeli-asr/blob/ccf54859d0a646c5e588d0fa92d7b25d35ce924f/README.md)
  - 30 narrations d'environ une heure, 23 hommes et 7 femmes, surtout en bambara de Bamako, avec un peu de maninka et de bambara de Ségou ; tous les narrateurs ne sont pas griots — [An ka taa, « The Griots and Tales of Jeli-ASR »](https://www.ankataa.com/blog/2022/10/13/the-griots-and-tales-of-jeli-asr)
- **bam-asr-early :** 37,41 h et 38 769 échantillons, avec transcriptions semi-expertes et traductions françaises. Composants : Bambara-ASR d'Oza (~29 h), **Jeli-ASR-RMAI (~3,5 h)**, oza-tts-mali-pense (~4 h), reading-tutor-data-collection (~1 h). La v1.0.1 a normalisé les transcriptions — [HF RobotsMali/bam-asr-early](https://huggingface.co/datasets/RobotsMali/bam-asr-early)
- **Kunkado :** ~160 h (161,15 h et 118 925 segments publiés), archives radio, parole spontanée avec alternance codique ; sous-ensemble relu de 33,47 h ; AfricaNLP 2026, p. 190–196 — [arXiv 2512.19400](https://arxiv.org/abs/2512.19400). La carte HF comporte une clause de partage à l'identique, mais la licence exacte n'est pas confirmée — [commit HF RobotsMali/kunkado](https://huggingface.co/datasets/RobotsMali/kunkado/commit/be43c51b54962c4083b3350913d12e9317d9a46b). Je n'ai trouvé aucune mention de traduction française.
- **afvoices :** 423 h segmentées (612 h brutes), cc-by-4.0, parole spontanée — [HF RobotsMali/afvoices](https://huggingface.co/datasets/RobotsMali/afvoices) ; [arXiv 2511.18557](https://arxiv.org/abs/2511.18557)
- Autres corpus de parole d'oza75 (`oza75/bambara-tts`, `oza75/bambara-asr`), distincts de bambara-mt — [HF oza75/bambara-tts](https://huggingface.co/datasets/oza75/bambara-tts)

### Inferences
- Le **côté français de bam-asr-early** (hors partie Jeli) ajoute ~33 h de paires bm–fr d'un autre registre : la partie Oza et la partie « mali-pense », proches des sous-titres et de vidéos de vulgarisation. Il faut **retirer la partie Jeli-ASR-RMAI** et dédoublonner contre le test Jeli-ASR avant de s'en servir.
- La forte hausse sur le test Jeli-ASR (+11) avec seulement +1 sur Bayelemabaga s'explique bien par la fuite déjà mesurée (24 % de phrases longues du test présentes en train). Pour évaluer proprement, il faudrait **découper Jeli-ASR par narrateur ou par récit**.
- Kunkado et afvoices, faute de traduction française, servent à l'ASR et, en traduction, de **source monolingue de bambara parlé** (voir la section précédente).

### Gaps
- Je n'ai pas pu confirmer si les traductions françaises de bam-asr-early sont humaines ou automatiques pour chaque composant (la carte parle de « semi-expert transcriptions and French translations »).
- Licence de bam-asr-early : non vérifiée.
- Je n'ai pas vérifié si afvoices ou Kunkado ont ajouté des traductions françaises après décembre 2025.
