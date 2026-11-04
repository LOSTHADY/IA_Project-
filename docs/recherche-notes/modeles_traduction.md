# Modèles et techniques de traduction automatique bambara ↔ français sous contrainte de calcul (Colab T4 16 Go, fp16, déploiement CPU)

> Note de méthode : depuis l'environnement de recherche, arXiv, ACL Anthology, Hugging Face, dl.fbaipublicfiles.com (fichiers `metrics.csv` de NLLB) et tinyurl étaient bloqués. Ont été **lus directement** : le PDF MAFAND-MT (copie Microsoft Research), les README GitHub de NLLB, FLORES-200, SeamlessM4T, MADLAD-400, Toucan, lafand-mt, AfroBench, CTranslate2, ainsi que le vocabulaire SentencePiece de MADLAD-400 (bucket GCS public). Tout le reste vient d'extraits de moteur de recherche et porte la mention « (extrait de recherche, non vérifié dans le PDF) ». Il faut revérifier ces chiffres avant de les citer dans le mémoire.

## Q1. Scores publiés des modèles candidats pour bam↔fra (ou bam↔eng)

### Takeaway
Pour bam↔fra, aucun score FLORES-200 n'a pu être récupéré : les fichiers de métriques de NLLB étaient inaccessibles, et pour les modèles denses (600M, 1.3B, 3.3B) ils ne contiennent de toute façon que les directions centrées sur l'anglais plus 200 directions tirées au hasard. Le seul chiffre publié et vérifiable pour bam↔fra est la comparaison sur le test MAFAND-MT (actualités, 1 600 phrases). Un M2M-100 418M fine-tuné y atteint 24,7 BLEU / 49,9 chrF (fr→bam) et 25,8 / 49,0 (bam→fr). NLLB-200 en zero-shot n'y obtient que 7,7 / 29,9 et 14,6 / 37,5. Aucun LLM généraliste (GPT, Gemini, Claude, Gemma, Llama, Aya, Qwen) n'a de score bambara publié retrouvé. SeamlessM4T v2 ne couvre pas le bambara. MADLAD-400 et Toucan le couvrent, mais sans chiffre bam↔fra retrouvé.

### Cited Findings

**NLLB-200 (Meta), licence CC-BY-NC 4.0**
- Cinq checkpoints publiés : MoE 54,5B, Dense 3,3B, Dense 1,3B, Distilled 1,3B et Distilled 600M. « All models are licensed under CC-BY-NC 4.0 ». Pour le MoE 54B, les métriques couvrent les 40 602 directions de FLORES-200 (chrF++, spBLEU SPM-200, spBLEU SPM-100). Pour les autres modèles, elles se limitent au chrF++ sur « all possible 402 English-centric direction + 200 sampled non English-centric directions ». Les liens tinyurl (`nllb200densedst600mmetrics`, `nllb200densedst1bmetrics`, `nllb200dense1bmetrics`, `nllb200dense3bmetrics`, `nllb200moe54bmetrics`) mènent à ces CSV — [README NLLB, fairseq/nllb](https://github.com/facebookresearch/fairseq/blob/nllb/examples/nllb/modeling/README.md)
- `bam_Latn`, `dyu_Latn` (dioula), `fuv_Latn`, `mos_Latn` et `wol_Latn` figurent dans les langues de FLORES-200/NLLB. On n'y trouve **aucun code maninka ou mandinka** (mnk, emk) — [langs.txt FLORES-200](https://github.com/facebookresearch/fairseq/blob/nllb/examples/nllb/modeling/scripts/flores200/langs.txt)
- Métrique officielle de FLORES-200 : chrF++ via `sacrebleu -m chrf --chrf-word-order 2` — [README FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)
- Tableau « Comparison to State-of-the-Art on MAFAND-MT's Test Set » de l'article NLLB, ligne `bam_Latn`, valeurs BLEU/chrF++ :
  - fr→bam : Adelani et al. 24,7/49,9 ; NLLB-200 **7,7/29,9**.
  - bam→fr : Adelani et al. 25,8/49,0 ; NLLB-200 **14,6/37,5**.
  - Le modèle NLLB évalué est très probablement le MoE 54B, mais ce n'est pas confirmé — [NLLB paper, arXiv 2207.04672](https://arxiv.org/pdf/2207.04672) (extrait de recherche). Les valeurs Adelani correspondent exactement à la ligne M2M-100 REL+NEWS→NEWS des tableaux 5 et 6 de MAFAND, vérifiés directement (voir ci-dessous), ce qui recoupe l'extrait.
- Article NLLB : pour un fine-tuning sur une seule tâche, LR = 5e-5 avec schéma inverse-sqrt — [NLLB paper](https://arxiv.org/pdf/2207.04672) (extrait de recherche).

**MAFAND-MT (Adelani et al., NAACL 2022), données bam/fr. Chiffres lus directement dans le PDF**
- Données bambara : actualités Maliweb.net, 3 302 / 1 484 / 1 600 phrases (train/dev/test), plus 28 K phrases de la Bible. Licence des données MT : CC-BY-NC 4.0 — [PDF MAFAND (copie MSR)](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf) ; [GitHub lafand-mt](https://github.com/masakhane-io/lafand-mt)
- Tableaux 3 et 4, entraînement sur les seules NEWS (3,3 K paires), métriques BLEU / chrF (chrF simple, pas chrF++) :

| Modèle | fr→bam BLEU | fr→bam chrF | bam→fr BLEU | bam→fr chrF |
|---|---|---|---|---|
| MT5-base | 1,5 | 10,0 | 2,5 | 19,4 |
| AfriMT5 | 2,1 | 14,0 | 6,4 | 27,7 |
| ByT5 | 9,5 | 27,8 | 10,0 | 31,2 |
| AfriByT5 | 11,4 | 31,4 | 13,8 | 34,8 |
| mBART50 | 18,6 | 42,3 | 6,8 | 26,0 |
| AfriMBART | 15,3 | 40,4 | 8,1 | 31,4 |
| **M2M-100 418M** | **22,7** | **48,2** | **22,1** | **45,9** |
| M2M-100 multilingue FR-centré | 18,5 | 43,4 | 22,1 | 45,6 |

  Source : [PDF MAFAND](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf)
- Tableaux 5 et 6, M2M-100 avec Bible (REL) et actualités (NEWS) :
  - fr→bam : REL+NEWS 23,0/46,8 ; REL→NEWS 20,3/44,1 ; **REL+NEWS→NEWS 24,7/49,9**.
  - bam→fr : REL+NEWS 24,0/47,1 ; REL→NEWS 20,3/44,5 ; **REL+NEWS→NEWS 25,8/49,0**.
  - Un Transformer bilingue entraîné de zéro plafonne à 8,5/27,4 (fr→bam) et 5,8/26,5 (bam→fr).
  - Source : [PDF MAFAND](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf)

**MADLAD-400 (Google), MT 3B / 7,2B / 7,2B-BT / 10,7B**
- Le dépôt publie quatre checkpoints MT : 3B, 7,2B, 7,2B fine-tuné sur back-translation, et 10,7B — [README MADLAD-400](https://github.com/google-research/google-research/tree/master/madlad_400)
- Le vocabulaire 256k utilisé par les modèles MT contient les balises de langue `<2bm>` (bambara) et `<2dyu>` (dioula), mais aucune balise maninka ou mandinka. Vérifié directement dans le fichier — [spm.vocab MADLAD (GCS)](https://storage.googleapis.com/madlad-400-checkpoints/vocabulary/256k_vocab/spm.vocab)
- CTranslate2 prend en charge MADLAD-400 — [README CTranslate2](https://github.com/OpenNMT/CTranslate2)
- Aucun score FLORES bam↔fra ou bam↔eng de MADLAD n'a été trouvé. L'article ([arXiv 2309.04662](https://arxiv.org/pdf/2309.04662)) n'a pas pu être lu.

**SeamlessM4T v2 (Meta)**
- Le modèle gère 101 langues en entrée parole, 96 en texte et 35 en sortie parole. La table officielle **ne contient pas `bam`**. Langues africaines voisines présentes : fuv, ibo, yor, lug, luo, kam, wol absent, etc. Vérifié directement — [docs m4t README](https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/README.md)

**Google Traduction**
- Le bambara a été ajouté le 11 mai 2022 avec 23 autres langues, pour un total de 133. Ces langues ont été apprises par « Zero-Shot Machine Translation », à partir de texte monolingue seulement. Google précise que la technologie « isn't perfect » — [blog Google](https://blog.google/products/translate/24-new-languages/) ; [9to5google](https://9to5google.com/2022/05/11/google-translate-new-languages/)
- Méthode décrite dans Bapna et al. 2022, « Building MT Systems for the Next Thousand Languages ». Aucun score bambara publié n'a été retrouvé — [arXiv 2205.03983](https://arxiv.org/abs/2205.03983v1)

**LLM généralistes (GPT-4o / 4.1 / 5, Gemini 2.x, Claude, Gemma 3, Llama 3/4, Aya)**
- Robinson et al. (WMT 2023) ont évalué GPT-3.5 et GPT-4 sur FLORES-200 (204 variétés). Les systèmes MT classiques battent ChatGPT sur 84,1 % des langues. En 5-shot, ChatGPT ne bat NLLB que sur environ 6 % des langues peu dotées. Google est le meilleur système global. La ligne bambara n'a pas été récupérée — [ACL Anthology 2023.wmt-1.40](https://aclanthology.org/2023.wmt-1.40/)
- AfroBench couvre 64 langues africaines, 15 tâches et 22 jeux de données, dont de la MT. GPT-4o et Gemini-1.5 Pro dépassent de +13 points Gemma 2 27B, le meilleur modèle ouvert. La présence du bambara dans les tâches MT et les chiffres associés n'ont pas été récupérés — [arXiv 2311.07978](https://arxiv.org/abs/2311.07978) ; [GitHub AfroBench](https://github.com/McGill-NLP/AfroBench)
- IrokoBench : le meilleur modèle est GPT-4o (moyenne 48,1). D'après l'extrait, le bambara ne figure pas dans ses tableaux — [IrokoBench NAACL 2025](https://aclanthology.org/2025.naacl-long.139.pdf) (extrait de recherche)
- Nsanku (2026, langues ghanéennes, donc **pas le bambara**) : gemini-2.5-flash, claude-sonnet-4-5 et gpt-4.1 forment le « Tier 1 ». Le chrF dépasse systématiquement le BLEU — [arXiv 2605.04208](https://arxiv.org/pdf/2605.04208) (extrait de recherche)
- African Languages Lab (2025) : 31 langues africaines, avec le bambara dans la table des données. Le chrF++ de base des LLM va de 2,00 (wolof) à 44,76 (afrikaans), moyenne 8,10. Le fine-tuning apporte +23,69 chrF++ en moyenne, sur un modèle de type Llama-3.2-1B d'après l'extrait — [arXiv 2510.05644](https://arxiv.org/pdf/2510.05644) (extrait de recherche, chiffre bambara non obtenu)
- Bayelemabaga (NAACL 2025) conclut que les modèles encodeur-décodeur (dont NLLB) restent meilleurs que les LLM décodeur-seul (Mistral, Open-Llama-7B, Llama3-8B) pour bam↔fr — [ACL Anthology 2025.naacl-long.602](https://aclanthology.org/2025.naacl-long.602/) (extrait de recherche)
- Bonfanti et al. (2025) : le meilleur Llama 3 8B fine-tuné obtient BLEU 9,82 et chrF 19,00 sur le test Bayelemabaga (fr→bam). Un petit Transformer entraîné de zéro donne environ 10 BLEU / 21 chrF sur Bayelemabaga, et 33,81 BLEU / 41 chrF sur leur jeu Yiri. Le corpus total compte 353 629 phrases alignées — [arXiv 2509.12514](https://arxiv.org/html/2509.12514v1) (extrait de recherche)

**Modèles africains**
- Toucan (UBC-NLP, ACL Findings 2024) : Cheetah-1.2B et Cheetah-3.7B, fine-tunés en Toucan pour 156 paires africaines. Le benchmark AfroLingu-MT compte 43 langues, et `"bam": "Bambara"` figure dans la liste des langues supportées. Toucan-1.2B dépasse NLLB-200-1.3B de 6,96 points spBLEU-1K **en moyenne**. Toucan (3,7B) bat Aya-13B sur 16 des 28 paires comparées. Modèles publiés : `UBC-NLP/toucan-base`, `UBC-NLP/toucan-1.2B`, `UBC-NLP/cheetah-base`, `UBC-NLP/cheetah-1.2B`. Vérifié directement — [README Toucan](https://github.com/UBC-NLP/Toucan) ; [ACL Findings 2024](https://aclanthology.org/2024.findings-acl.781/)
- AfriNLLB (AfricaNLP 2026) : NLLB-600M élagué (4 couches décodeur retirées, soit 548M) puis distillé depuis NLLB-3.3B. Qualité comparable au 600M, jusqu'à 20 % plus rapide, publié en Transformers et CTranslate2. **Il couvre 15 paires, sans le bambara** d'après les extraits — [ACL Anthology 2026.africanlp-main.30](https://aclanthology.org/2026.africanlp-main.30/) ; [arXiv 2602.09373](https://huggingface.co/papers/2602.09373) (extrait de recherche)

**Autres résultats bam↔fr**
- BFMT 2023, compétition Orange Silicon Valley (LoResMT 2023), évaluée au sacreBLEU. Les extraits citent 19,82 / 17,92 BLEU (bam→fr / fr→bam) pour un NLLB fine-tuné jugé « under-performing », et 24,55 / 25,56 BLEU pour un « distilled 1.3B non fine-tuné ». **L'attribution de ces chiffres est ambiguë dans les extraits** — [ACL Anthology 2023.loresmt-1.9](https://aclanthology.org/2023.loresmt-1.9/) (extrait de recherche)
- Tapo et al. 2020 (LoResMT) ont publié le premier jeu parallèle et le premier benchmark bambara↔français/anglais — [arXiv 2011.05284](https://arxiv.org/pdf/2011.05284)

### Inferences
- Les chiffres ne sont pas comparables entre eux. Le 27,8 chrF++ bm→fr du projet porte sur 500 phrases de Bayelemabaga. MAFAND rapporte du chrF (sans n-grammes de mots, donc structurellement plus haut que chrF++) sur des actualités. Le tableau NLLB/MAFAND mélange chrF++ côté NLLB et chrF côté Adelani. La seule comparaison honnête consiste à réévaluer tous les candidats sur **le même** test : test Bayelemabaga dédupliqué, FLORES-200 devtest `bam_Latn`↔`fra_Latn`, et un petit test « parole quotidienne ».
- Le zero-shot du projet (21,8 chrF++ bm→fr avec 600M) est cohérent avec la faiblesse de NLLB sur MAFAND (37,5 chrF++ bm→fr, probablement avec le MoE 54B, sur un autre domaine). Le fine-tuning reste indispensable.
- Le bambara n'est pas couvert par SeamlessM4T v2 ni par AfriNLLB, et les LLM n'ont aucun score publié retrouvé. Les seuls candidats réalistes sont donc la famille NLLB (600M ou 1.3B distillé), M2M-100 418M (licence MIT pour le modèle ; données MAFAND en NC), Toucan-1.2B (couvre bam, licence à vérifier) et MADLAD-3B (couvre `<2bm>`, mais 3B est lourd sur CPU et aucun score n'est publié).
- Au vu de Robinson et al. et d'AfroBench, Qwen2.5-1.5B, le LLM aval, ne doit pas être utilisé pour traduire le bambara. Il faut lui fournir du français.

### Gaps
- Les valeurs chrF++ exactes `bam_Latn-fra_Latn` et `fra_Latn-bam_Latn` des fichiers `metrics.csv` NLLB (600M, 1.3B distillé, 1.3B, 3.3B, 54B) n'ont pas été récupérées (hôte bloqué). Pour les modèles denses, la paire bam↔fra n'y figure probablement pas : seules les directions anglais-centrées sont garanties, avec 200 directions non anglaises tirées au hasard. Il faut les télécharger soi-même (`wget https://tinyurl.com/nllb200densedst600mmetrics`, etc.) et chercher `bam_Latn`. Seul le fichier du MoE 54B est garanti complet.
- Aucun score bambara n'a été trouvé pour MADLAD-400, GPT-4o, GPT-4.1, GPT-5, Gemini 2.x, Claude, Gemma 3, Llama 3/4, Aya ou Aya Expanse. La couverture du bambara par Aya-101, InkubaLM, Lugha-Llama et AfroLM n'a pas pu être vérifiée. D'après des connaissances antérieures non vérifiées ici, InkubaLM et Lugha-Llama ne couvrent pas le bambara.
- Les tableaux par langue de Toucan (AfroLingu-MT, ligne bam) et d'AfroBench (MT bambara) n'ont pas été lus.
- Les tableaux chiffrés de Bayelemabaga n'ont pas été lus. Le brouillon OpenReview cite « AfriM2M100 ~35,7 BLEU / 25,8 chrF++ », ce qui est incohérent (BLEU > chrF++). À vérifier dans la version finale.

## Q2. Modèles et ressources maliens ou communautaires sur Hugging Face (identifiants, scores, licences)

### Takeaway
Aucun modèle de traduction « officiel » RobotsMali ou MALIBA-AI avec des scores publiés n'a été trouvé. L'écosystème malien publie surtout des **données** (Bayelemabaga, Jeli-ASR, Kunkado, bam-asr-early) et des modèles **parole** (ASR, TTS). Le modèle MT communautaire le plus proche du projet est `Youssouf0123/bambara-french-nllb`, un NLLB-600M entraîné sur Jeli-ASR + Bayelemabaga. Ses scores déclarés sont faibles (sacreBLEU 16,5 bm→fr, 11,3 fr→bm), et il hérite de la licence CC-BY-NC de NLLB.

### Cited Findings
- `Youssouf0123/bambara-french-nllb` : NLLB-200-distilled-600M fine-tuné dans les deux sens sur 115 k paires (transcriptions Jeli-ASR + Bayelemabaga). SacreBLEU 16,5 en bam→fr et 11,3 en fr→bam. Le jeu de test n'est pas précisé dans l'extrait — [HF model card](https://huggingface.co/Youssouf0123/bambara-french-nllb) (extrait de recherche)
- `RobotsMaliAI/bayelemabaga` : 46 976 lignes bambara-français alignées, prêtes pour la MT — [HF dataset](https://huggingface.co/datasets/RobotsMaliAI/bayelemabaga). L'article (Tapo et al., NAACL 2025) décrit environ 47 K paires issues de 231 sources. Sept modèles y sont fine-tunés, dont NLLB-200, AfriMBART, AfriMT5, AfriM2M100, Mistral, Open-Llama-7B et Llama3-8B. Les gains vont jusqu'à +4,5 BLEU, +11,4 chrF++ et +0,27 AfriCOMET — [ACL Anthology](https://aclanthology.org/2025.naacl-long.602/) (extrait de recherche)
- `RobotsMali/jeli-asr` : environ 30 h, environ 11 533 fichiers audio, transcriptions bambara et traductions françaises (`french-manifests/`), licence CC-BY-4.0 — [HF README](https://huggingface.co/datasets/RobotsMali/jeli-asr/blob/ccf54859d0a646c5e588d0fa92d7b25d35ce924f/README.md) ; [commit v1.0.1](https://huggingface.co/datasets/RobotsMali/jeli-asr/commit/e034cfa22b3775983ba20f27c06527e8b5bf21c0). Le corpus compte 30 narrations d'environ 1 h chacune (7 femmes, 23 hommes), majoritairement en « Bamako Bambara » et 4 en **maninka**. Beaucoup de narrateurs « are not actual jeli » — [blog An ka taa](https://www.ankataa.com/blog/2022/10/13/the-griots-and-tales-of-jeli-asr)
- `RobotsMali/kunkado` (« Kunnafonidilaw ka Cadeau », EACL 2026) : 160 h d'archives de radios maliennes, parole spontanée actuelle. Le code-switching, les disfluences et le bruit sont **conservés volontairement**. Des modèles Parakeet fine-tunés sur 33,47 h révisées font passer le WER de 44,47 % à 37,12 % et de 36,07 % à 32,33 % — [arXiv 2512.19400](https://arxiv.org/abs/2512.19400) ; [HF dataset](https://huggingface.co/datasets/RobotsMali/kunkado). Un autre extrait parle d'une « 612-hour release » de RobotsMali fin 2025 : **c'est contradictoire avec les 160 h**, peut-être un autre corpus — [arXiv 2511.18557](https://arxiv.org/html/2511.18557v1) (extrait de recherche)
- `RobotsMali/bam-asr-early` : audio bambara avec transcriptions semi-expertes et traductions françaises — [HF dataset](https://huggingface.co/datasets/RobotsMali/bam-asr-early)
- MALIBA-AI et `sudoping01`, d'après les extraits de recherche :
  - Space « BambaraTranslator » : traduit bambara/français/anglais « using NLLB model », sans licence ni score publiés — [Ankataa resources](https://www.ankataa.com/blog/2024/6/27/manding-language-tech-resources-and-initiatives).
  - `sudoping01/maliba-llm` : modèle de chat sur base Gemma, licence MIT déclarée (les conditions Gemma s'appliquent aussi) — [HF](https://huggingface.co/sudoping01/maliba-llm).
  - `sudoping01/bambara-asr-v2` : Apache 2.0 — [HF](https://huggingface.co/sudoping01/bambara-asr-v2).
  - `MALIBA-AI/bambara-tts` : « MALIBA-AI Research Licence », usage non commercial — [HF](https://huggingface.co/MALIBA-AI/bambara-tts).
  - `sudoping01/bambara-tts` : CC BY-NC 4.0 — [HF](https://huggingface.co/sudoping01/bambara-tts).
  - Normaliseur de texte bambara — [GitHub MALIBA-AI/bambara-text-normalization](https://github.com/MALIBA-AI/bambara-text-normalization).
- Selon le blog An ka taa, certains outils bambara listés ne fonctionnaient pas lors d'un test le 22 janvier 2026 — [An ka taa](https://www.ankataa.com/blog/2024/6/27/manding-language-tech-resources-and-initiatives) (extrait de recherche)
- `masakhane/m2m100_418M-FR-NEWS` : M2M-100 fine-tuné sur MAFAND en mode français-centré (fr–{bam, bbj, ewe, fon, mos, wol}). Lu dans l'annexe du PDF — [PDF MAFAND](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf)
- Dioula :
  - Challenge Zindi « Melio MLOps » dyu→fr (2024) : environ 10 929 paires, extension de Common Voice FR v4 — [Zindi](https://zindi.africa/competitions/melio-mlops-competition).
  - Modèles communautaires `JuliusFx/dyula-french-joeynmt` et `JuliusFx/dyula-french-nmt-100/150`, sans scores publiés — [HF](https://huggingface.co/JuliusFx/dyula-french-joeynmt).
- Bambara ASR : comparatif « Where Are We At with ASR for the Bambara Language? » (AfricaNLP 2026) — [arXiv 2602.09785](https://arxiv.org/pdf/2602.09785) (titre seulement)

### Inferences
- Le modèle de Youssouf0123 (même recette que la V2 du projet : 600M + Jeli + Bayelemabaga) ne fournit pas de meilleure base. Ses scores BLEU déclarés restent modestes, et sans protocole de déduplication publié, ils ne sont pas interprétables.
- Kunkado (radio, code-switching, parole actuelle) est la ressource la plus proche de la « parole quotidienne » du locuteur natif. N'ayant pas de traductions françaises d'après les extraits, il peut servir de source monolingue bambara pour de la back-translation ou de la forward translation, et pour générer des sorties ASR bruitées.
- Jeli-ASR contient quelques locuteurs maninka et un registre narratif de griot. Cela peut expliquer la dérive constatée sur les salutations et les phrases courtes (voir Q3).

### Gaps
- Les licences exactes de Bayelemabaga, Kunkado et bam-asr-early, et la licence de `Youssouf0123/bambara-french-nllb`, n'ont pas été lues sur les fiches HF (site bloqué).
- Aucune liste exhaustive des modèles HF des organisations `RobotsMali` et `MALIBA-AI` n'a pu être obtenue. Il faut vérifier directement `https://huggingface.co/RobotsMali` et `https://huggingface.co/MALIBA-AI`.
- Aucun score publié n'a été trouvé pour le Space BambaraTranslator ni pour des modèles MT RobotsMali.

## Q3. Techniques et gains mesurés (langues africaines peu dotées)

### Takeaway
Le gain le mieux documenté pour le bambara est **l'adaptation de domaine avec un petit jeu in-domain**. Dans MAFAND, enchaîner « gros corpus hors domaine + in-domain » puis « in-domain seul » (REL+NEWS→NEWS) donne +2,0 BLEU (fr→bam) et +3,7 BLEU (bam→fr) par rapport au in-domain seul. Un entraînement hors domaine seul s'effondre (−93,5 % BLEU pour bam). C'est exactement le mécanisme qui explique la régression V2 sur la parole quotidienne. Les autres techniques ont des gains plus modestes ou peu documentés pour le bambara :
- back-translation, taguée ou non : de +0,8 chrF++ à plusieurs BLEU ;
- moyenne de poids : environ +1,7 chrF++ dans un cas NLLB-600M ;
- robustesse ASR : +1 à +3 BLEU sur entrée bruitée ;
- EWC et mélange des données contre l'oubli ;
- LoRA : proche du fine-tuning complet (≤ 0,85 BLEU d'écart dans un cas NLLB) ;
- CTranslate2 int8 : environ ×1,3 de débit et −38 % de mémoire par rapport à CT2 fp32, sans perte de BLEU.

### Cited Findings

**Adaptation de domaine (MAFAND, lu directement)**
- REL+NEWS→NEWS est la meilleure stratégie. En xx-en/fr, elle gagne 2,0 BLEU sur NEWS seul et 1,5 BLEU sur REL+NEWS. Pour bam : fr→bam passe de 22,7 (NEWS seul) à 24,7 BLEU (chrF 48,2 → 49,9), et bam→fr de 22,1 à 25,8 BLEU (chrF 45,9 → 49,0) — [PDF MAFAND](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf)
- Entraîné uniquement sur la Bible (REL), le modèle perd 93,5 % de BLEU sur les actualités bambara (28 k phrases REL). La paire fr-bam est citée comme ayant un **grand écart de domaine**, et la qualité augmente continûment de 0 à 31 K phrases in-domain. M2M-100 s'adapte plus vite que ByT5. Exemple : le modèle REL traduit « transition » par « la puissance du milieu » et « piliers » par « sauterelles » — [PDF MAFAND](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf)
- Le pré-entraînement continu africain (AfriByT5) améliore fr-bam de +1,9 BLEU par rapport à ByT5 — [PDF MAFAND](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf)

**Back-translation, taguée ou non, et forward translation**
- Tagged BT (Caswell et al., WMT 2019) : un simple jeton marquant la source synthétique fait mieux que la BT bruitée en En-Ro et l'égale en En-De — [ACL W19-5206](https://aclanthology.org/W19-5206) (extrait de recherche)
- Dans un cadre peu doté (Tag-less BT), la BT taguée gagne +6,81 BLEU contre +4,92 pour la BT standard en De-En, mais +2,83 contre +3,06 en En-Vi. Le tag n'aide donc pas toujours — [arXiv 1912.10514](https://arxiv.org/pdf/1912.10514) (extrait de recherche)
- NLLB-600M fine-tuné sur 10 langues amérindiennes (AmericasNLP 2023) : la BT apporte environ +0,008 chrF++, la moyenne de poids environ +0,017 (échelle 0–1 probable, soit environ +0,8 et +1,7 points) — [AmericasNLP 2023 PlayGround](https://preview.aclanthology.org/new-sigs/2023.americasnlp-1.19/) (extrait de recherche)
- Distillation par forward translation (AfriNLLB) : un enseignant NLLB-3.3B génère des données synthétiques, combinées aux données authentiques puis filtrées, pour fine-tuner un 600M (élagué) — [AfriNLLB](https://aclanthology.org/2026.africanlp-main.30/) (extrait de recherche)
- SMOL (Google, 2025) : données traduites professionnellement pour 115 à 124 langues peu dotées (SmolSent, SmolDoc), en complément de GATITOS (lexique au niveau des tokens). S'en servir pour prompter ou fine-tuner des LLM donne des « robust ChrF improvements » — [arXiv 2502.12301](https://arxiv.org/html/2502.12301v2) (extrait de recherche ; présence du bambara non vérifiée)

**Oubli catastrophique et entraînement continu**
- Thompson et al. (NAACL 2019) : l'EWC conserve l'essentiel de la performance du domaine général, perdue sinon par l'entraînement continu, sans nuire au domaine cible — [ACL N19-1209](https://aclanthology.org/N19-1209/) (extrait de recherche). Une étude multi-domaine EMNLP 2021 trouve l'EWC utile mais avec un compromis moins favorable — [EMNLP 2021](https://preview.aclanthology.org/fix_video/2021.emnlp-main.666.pdf) (extrait de recherche)
- Geler les paramètres importants pour la langue ou tâche source avant adaptation ramène la dégradation à environ 3,4 % (7B) et 2,8 % (13B), contre environ 20 % et 22 % en fine-tuning complet. Résultat obtenu sur des LLM, pas en MT — [arXiv 2512.04844](https://arxiv.org/pdf/2512.04844v1) (extrait de recherche)

**LoRA / QLoRA contre fine-tuning complet**
- Anglais-irlandais sur NLLB (SemiLoRA) : le fine-tuning complet est en général supérieur à LoRA en BLEU — [arXiv 2510.18725](https://arxiv.org/pdf/2510.18725) (extrait de recherche)
- Anglais-norvégien sur NLLB distillé (EAMT 2026) : écart maximal de 0,85 BLEU entre LoRA et fine-tuning complet — [EAMT 2026](https://aclanthology.org/2026.eamt-1.25.pdf) (extrait de recherche)
- Llama-3 en langues peu dotées : le fine-tuning complet dépasse LoRA d'environ 0,45 chrF++ — [arXiv 2504.01919](https://arxiv.org/pdf/2504.01919) (extrait de recherche ; attribution exacte incertaine)

**Robustesse aux erreurs d'ASR (pipeline ASR → MT)**
- Martucci et al. (Interspeech 2021) : bruiter les transcriptions propres avec un bruit lexical estimé sur de vraies sorties ASR donne presque les mêmes résultats qu'un entraînement sur ces sorties. Aucune perte sur le texte propre, et un transfert vers d'autres ASR et d'autres domaines — [ISCA](https://www.isca-archive.org/interspeech_2021/martucci21_interspeech.html) (extrait de recherche)
- Entraînement adversarial de stabilité : jusqu'à +2,83 BLEU sur sorties ASR bruitées (IWSLT2018) avec une performance quasi inchangée sur texte propre — [arXiv 1909.11430](https://ar5iv.labs.arxiv.org/html/1909.11430) (extrait de recherche)
- Augmentation par frontières de phrase (ponctuation ASR imparfaite) : +1 BLEU en moyenne — [arXiv 2010.11132](https://arxiv.org/pdf/2010.11132) (extrait de recherche)
- Vietnamien→anglais (FLEURS) : traduire des sorties ASR au lieu de transcriptions propres coûte de 6,79 à 10,64 BLEU, et le gain de robustesse dépend de l'adéquation entre le bruit injecté et les erreurs réelles — [arXiv 2606.12911](https://arxiv.org/pdf/2606.12911) (extrait de recherche)
- Un seul modèle peut traiter indifféremment une sortie ASR brute ou post-éditée — [Di Gangi et al., IWSLT 2019](https://preview.aclanthology.org/fix_video/2019.iwslt-1.32.pdf) (extrait de recherche)

**Décodage : MBR, faisceau, ensembles**
- Un MBR sur candidats issus d'epsilon-sampling bat le faisceau en évaluation humaine sur 4 paires, avec BLEURT comme fonction d'utilité — [arXiv 2305.09860](https://arxiv.org/pdf/2305.09860) (extrait de recherche). L'epsilon-sampling demande de reclasser beaucoup plus de candidats, ce qui coûte cher. Une alternative consiste à faire un MBR guidé par COMET sur les candidats du faisceau — [arXiv 2405.11937](https://arxiv.org/html/2405.11937v1) (extrait de recherche)
- MAFAND utilise un faisceau de taille 10 — [PDF MAFAND](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf)

**CTranslate2 int8 pour CPU (lu directement dans le README)**
- CTranslate2 prend en charge NLLB, M2M-100 et MADLAD-400, avec quantification FP16, BF16, INT16, INT8 et AWQ INT4 — [README CTranslate2](https://github.com/OpenNMT/CTranslate2)
- Benchmark CPU (Transformer-base WMT14, 4 threads, c5.2xlarge) :

| Configuration | tokens/s | mémoire max | BLEU |
|---|---|---|---|
| PyTorch (OpenNMT-py) | 275,8 | 2 012 Mo | 26,77 |
| CT2 fp32 | 658,8 | 849 Mo | 26,77 |
| **CT2 int8** | **860,2** | **529 Mo** | **26,78** |
| CT2 int8 + vmap | 1 126,2 | 598 Mo | — |

  Source : [README CTranslate2](https://github.com/OpenNMT/CTranslate2)
- NLLB-3.3B sous CT2 : int8 est le format le plus petit et le plus performant sur CPU, fp16 le plus rapide sur GPU, avec des écarts de qualité « marginal » — [arXiv 2412.05862](https://arxiv.org/pdf/2412.05862) (extrait de recherche). Une conversion int8 communautaire de NLLB-3.3B pèse environ 3,3 Go (licence CC-BY-NC) — [HF Napron](https://huggingface.co/Napron/nllb-200-3.3B-ct2-int8) (extrait de recherche)
- AfriNLLB publie des versions CT2 d'un NLLB-600M élagué jusqu'à 20 % plus rapide — [AfriNLLB](https://aclanthology.org/2026.africanlp-main.30/) (extrait de recherche)

**Transfert via les langues mandingues**
- NLLB et FLORES-200 incluent `dyu_Latn`. MADLAD inclut `<2dyu>`. Aucun des deux n'inclut le maninka ou le mandinka (vérifié directement) — [langs.txt](https://github.com/facebookresearch/fairseq/blob/nllb/examples/nllb/modeling/scripts/flores200/langs.txt) ; [spm.vocab MADLAD](https://storage.googleapis.com/madlad-400-checkpoints/vocabulary/256k_vocab/spm.vocab)
- Dans MAFAND, des langues absentes du pré-entraînement continu en bénéficient quand même, « possibly due to language similarity » — [PDF MAFAND](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf)

### Inferences
- **Diagnostic de la V2.** MAFAND montre que le bambara a un très grand écart de domaine et qu'un entraînement sans données du domaine cible s'effondre. La V2 a été poussée vers la narration de griot : phrases longues, registre littéraire, quelques locuteurs maninka. Il est donc attendu qu'elle se dégrade sur les salutations et le code-switching, même avec +1 chrF++ sur Bayelemabaga. Le « +11 » Jeli est en outre gonflé par le chevauchement (voir Q4).
- **Recette prioritaire (estimation de coût faible).**
  1. Constituer quelques centaines à 2–3 milliers de paires « parole quotidienne » (salutations, phrases courtes, code-switching bambara-français), traduites par le locuteur natif ou générées puis corrigées par lui.
  2. Appliquer le schéma MAFAND « (Bayelemabaga + Jeli) → in-domain » : second fine-tuning court sur le in-domain mélangé à environ 20–50 % de Bayelemabaga pour limiter l'oubli. La proportion est une heuristique, sans source spécifique au bambara.
- **Test quasi gratuit.** La V2 a été continuée depuis la V1 avec la même architecture. Une interpolation linéaire des poids, par exemple 0,5·V1 + 0,5·V2, ou une moyenne des derniers checkpoints est donc valide et peut être évaluée en quelques minutes. La seule preuve publiée ici est le gain d'environ +1,7 chrF++ de la moyenne de poids sur NLLB-600M (AmericasNLP).
- **Robustesse ASR.** Le pipeline réel est ASR → MT → Qwen. Il est probablement plus rentable de passer l'audio de Jeli-ASR et bam-asr-early dans l'ASR du projet et d'entraîner la MT sur (sortie ASR, traduction de référence) que d'ajouter des données propres. Gain publié attendu : +1 à +3 BLEU sur entrée bruitée, sans perte sur le texte propre.
- **Back-translation.** Kunkado (bambara spontané) peut fournir le texte monolingue. La BT taguée (`<bt>` en tête de source) est peu coûteuse, mais les gains publiés en peu doté vont de marginaux à quelques points. Il faut la tester sur le dev in-domain, pas sur Jeli.
- **Lexique et dictionnaire.** Ajouter des paires mot ou expression (salutations, formules) comme « phrases » d'entraînement est cohérent avec l'esprit de GATITOS, mais aucun gain chiffré pour le bambara n'a été trouvé.
- **LoRA sur 3.3B.** L'écart LoRA/fine-tuning complet est faible, mais passer du 600M au 3.3B multiplierait le coût CPU par environ 5 (estimation par nombre de paramètres). Pour un déploiement CPU, un 600M ou un 1.3B distillé bien adapté au domaine semble un meilleur compromis qu'un 3.3B LoRA.

### Gaps
- Aucune étude chiffrée n'a été trouvée pour la back-translation (taguée ou non), la forward translation, l'augmentation lexicale ou le transfert dioula/maninka **spécifiquement pour le bambara**.
- Aucun gain MBR mesuré n'a été trouvé pour une langue africaine peu dotée, ni avec chrF comme utilité. Aucune étude d'ensembling NLLB pour le bambara.
- Les chiffres EWC exacts de Thompson et al. n'ont pas été lus.
- Pas de benchmark CT2 publié spécifique à NLLB-600M ou 1.3B sur CPU (tokens/s, RAM). À mesurer.

## Q4. Évaluer correctement quand le test chevauche l'entraînement, et construire un petit dev in-domain

### Takeaway
La contamination source+cible gonfle fortement les scores : jusqu'à +30 BLEU dans une étude contrôlée. Elle peut aussi contaminer d'autres directions de traduction partageant la même cible. Le +11 chrF++ de la V2 sur Jeli-ASR (24 % de chevauchement) n'est donc pas interprétable tel quel. Il faut dédupliquer au niveau du segment, des deux côtés, après normalisation, et évaluer avec chrF++ (sacrebleu, word order 2) sur trois tests figés : Bayelemabaga dédupliqué, FLORES-200 devtest et un test « parole quotidienne ».

### Cited Findings
- Kocyigit et al. (ICML 2025) : une contamination portant à la fois sur la source et la cible gonfle substantiellement le BLEU, jusqu'à 30 points, avec un effet 2,5 fois plus grand pour 8B que pour 1B. Une contamination source seule ou cible seule a des effets plus faibles et moins réguliers — [arXiv 2501.18771](https://arxiv.org/abs/2501.18771) (extrait de recherche)
- Contamination inter-directions sur FLORES-200 : entraîner sur une paire peut gonfler les scores d'autres paires partageant la même langue cible — [arXiv 2601.20858](https://arxiv.org/pdf/2601.20858) (extrait de recherche)
- Dans un corpus peu doté (système d'écriture en danger), 32 % des cibles de test apparaissaient à l'identique dans l'entraînement. Une décontamination au niveau du document ne réduisait le BLEU contaminé que de 4,6 points, car 8 cibles sur 16 persistaient via d'autres documents. Une déduplication **au niveau de la cible** était nécessaire — [ACL Anthology, auteurs Harb/Toutou](https://aclanthology.org/people/abdelrahman-harb/unverified/) (extrait de recherche ; venue non vérifiée)
- La métrique de référence de FLORES-200 est chrF++ via `sacrebleu -m chrf --chrf-word-order 2` — [README FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)
- MAFAND : chrF « works better » que BLEU pour les langues morphologiquement riches, avec un écart entre directions bien plus faible en chrF qu'en BLEU — [PDF MAFAND](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf). Nsanku : chrF > BLEU systématiquement — [arXiv 2605.04208](https://arxiv.org/pdf/2605.04208) (extrait de recherche)
- Ordres de grandeur de MAFAND : 3 302 phrases in-domain suffisent pour un saut majeur, avec des dev et test de 1 484 et 1 600 phrases. « A few thousand translations go a long way » — [PDF MAFAND](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf)
- FLORES-200 contient `bam_Latn`. Son devtest, indépendant de Bayelemabaga et de Jeli, peut servir de test hors domaine neutre — [langs.txt](https://github.com/facebookresearch/fairseq/blob/nllb/examples/nllb/modeling/scripts/flores200/langs.txt)

### Inferences
- **Protocole de déduplication proposé (inférence méthodologique).**
  1. Normaliser : minuscules, NFC, suppression de la ponctuation et des espaces multiples, unification des caractères bambara ɛ/ɔ/ɲ/ŋ et de leurs variantes (e/o, ny, ng). Le normaliseur MALIBA-AI peut servir.
  2. Retirer du train toute paire dont la source **ou** la cible normalisée apparaît dans un test.
  3. Ajouter un filtre approximatif (par exemple chrF ou Jaccard sur 4-grammes de caractères > 0,8) pour les quasi-doublons, fréquents dans les narrations longues découpées en segments.
  4. Pour Jeli-ASR, découper train/test **par narrateur ou par narration**, pas par segment : les segments d'une même narration partagent vocabulaire et formules.
  5. Publier les scores sur le test « propre » **et** sur le sous-ensemble chevauchant, pour montrer l'écart.
- **Dev et test « parole quotidienne ».**
  - Viser environ 300 phrases de dev et environ 300 de test (estimation : assez pour détecter environ ±2 chrF++ par bootstrap ; `sacrebleu --paired-bs` permet de le vérifier).
  - Échantillonner par catégories : salutations et formules, questions courtes, commandes vocales typiques de l'assistant, code-switching bambara-français, chiffres et heures, noms propres maliens.
  - Les sources possibles sont l'enregistrement du locuteur natif et des extraits Kunkado transcrits.
  - Faire traduire par le locuteur natif, idéalement relu par une seconde personne, et figer le test avant toute expérience.
  - Ne jamais utiliser le test pour choisir des checkpoints : utiliser le dev.
  - Évaluer aussi sur la **sortie ASR** (pas seulement la transcription propre), puisque c'est l'entrée réelle.
- Un petit test humain (préférence A/B en aveugle V1 vs V2 sur 50 phrases du quotidien) est cohérent avec le constat du locuteur et facile à faire.

### Gaps
- Aucune étude n'a été trouvée sur la contamination dans Bayelemabaga ou Jeli-ASR (doublons internes, recouvrement avec FLORES ou MAFAND).
- Aucune recommandation publiée n'a été trouvée sur la taille minimale d'un test pour le bambara. Le chiffre d'environ 300 phrases est une estimation.

## Q5. Recettes pratiques : hyperparamètres de fine-tuning (NLLB, M2M sur langues africaines), faisabilité T4 et CPU

### Takeaway
Les recettes publiées convergent vers les réglages suivants :
- LR de 5e-5 (AdamW) à 1e-4 (constant après warmup) ;
- batch d'environ 10 à 16 ;
- longueur maximale de 128 à 200 tokens ;
- 3 époques sur des corpus moyens et environ 10 sur des corpus de quelques milliers de paires ;
- faisceau de 4 à 10.

Avec Adafactor et un LR manuel, il faut `scale_parameter=False` et `relative_step=False`. Sur un T4 16 Go en fp16, on peut fine-tuner complètement le 600M et, de justesse, le 1.3B. Le 3.3B ne passe qu'en LoRA ou QLoRA. Pour le déploiement CPU, la combinaison CT2 int8 + 600M ou 1.3B distillé est la plus raisonnable. Aucun temps d'entraînement publié sur T4 n'a été trouvé.

### Cited Findings
- MAFAND : learning rate 5e-5 (défaut HF), batch de 10, longueurs source et cible maximales de 200, faisceau de 10, 3 époques. Pour un entraînement sur NEWS seul (environ 3 k paires), 10 époques. Un seul GPU V100. Chaque étape de fine-tuning du schéma REL+NEWS→NEWS dure 3 époques — [PDF MAFAND](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf). Pour M2M-100 418M, les exemples du dépôt utilisent `per_device_train_batch_size=4`, `--num_beams 10` et `--forced_bos_token` — [GitHub lafand-mt](https://github.com/masakhane-io/lafand-mt)
- Article NLLB : fine-tuning sur une tâche avec LR 5e-5 et schéma inverse-sqrt — [arXiv 2207.04672](https://arxiv.org/pdf/2207.04672) (extrait de recherche)
- Fine-tuning NLLB pour des langues ultra peu dotées (nheengatu) : warmup linéaire de 0 à 1e-4 sur 1 000 pas, puis LR constant à 1e-4 — [arXiv 2511.22482](https://arxiv.org/pdf/2511.22482) (extrait de recherche)
- Limbum→anglais : AdamW, LR 5e-5, warmup linéaire de 500 pas — [arXiv 2608.07629](https://arxiv.org/pdf/2608.07629) (extrait de recherche)
- Adafactor (doc HF) : pour un LR externe, `scale_parameter=False` et `relative_step=False`. Un LR manuel avec `relative_step=True` lève une erreur. Pour T5, la doc recommande de ne pas combiner gradient clipping et Adafactor — [doc Transformers](https://huggingface.co/docs/transformers/main/main_classes/optimizer_schedules) (extrait de recherche)
- Le tutoriel de David Dale fine-tune NLLB-200 pour une nouvelle langue (russe→touvain) en 8 étapes : tokenisation, mise à jour optionnelle du vocabulaire, entraînement, évaluation, publication — [Medium](https://cointegrated.medium.com/a37fc706b865) (extrait de recherche ; hyperparamètres non confirmés par l'extrait)
- AFRIDOC-MT : le fine-tuning supervisé de NLLB-200 1.3B améliore la traduction de documents vers les langues africaines — [arXiv 2501.06374](https://arxiv.org/pdf/2501.06374) (extrait de recherche)
- CT2 int8 sur CPU : voir le tableau en Q3. Un débit environ 3,1 fois supérieur à PyTorch et une mémoire environ 3,8 fois plus faible pour un Transformer-base — [README CTranslate2](https://github.com/OpenNMT/CTranslate2)

### Inferences
- **Mémoire sur T4 (estimations, à mesurer).**
  - 600M : poids fp32 environ 2,4 Go, plus les gradients, plus Adafactor (états factorisés quasi négligeables), plus les activations. Large marge.
  - 1.3B distillé : environ 5,2 Go de poids et 5,2 Go de gradients en fp32 maître, plus les activations. Cela passe en batch 4–8 × 128 tokens avec gradient checkpointing et accumulation. C'est serré mais faisable avec Adafactor, et pas avec AdamW (+10,4 Go d'états).
  - 3.3B : fine-tuning complet impossible (13,2 Go de poids fp32 à eux seuls). En LoRA, les poids gelés en fp16 occupent environ 6,6 Go. En QLoRA 4 bits, environ 2 Go.
  - Le T4 ne gère pas bf16. En fp16 mixte, garder le loss scaling et surveiller les NaN/inf.
- **Point de départ proposé pour le deuxième fine-tuning in-domain (V3)**, déduit des recettes ci-dessus : partir de la V1 (pas de la V2), Adafactor avec LR 5e-5 (deux fois plus bas qu'un premier fine-tuning à 1e-4, pour limiter l'oubli), warmup de 200 à 500 pas, batch effectif d'environ 16, longueur max de 128, 3 à 10 époques sur le mélange in-domain + Bayelemabaga, sélection du checkpoint sur le dev « parole quotidienne », faisceau de 4 à 5 en production.
- **Temps sur T4.** Aucune source. À titre d'ordre de grandeur non vérifié, quelques itérations par seconde pour le 600M en batch 16 × 128 tokens en fp16, et environ 2 à 3 fois plus lent pour le 1.3B. Mieux vaut reprendre les temps réellement mesurés lors des runs V1 et V2 du projet.
- **CPU.** Convertir avec `ct2-transformers-converter --quantization int8` et tester `inter_threads`/`intra_threads`. Avec le faisceau à 1 ou 2 au lieu de 4 ou 5, la latence baisse fortement, pour un coût en qualité à mesurer sur le dev. Une variante élaguée de type AfriNLLB (décodeur réduit) est une piste supplémentaire de vitesse sur CPU.
- **Licence.** NLLB (CC-BY-NC 4.0) et les données MAFAND (CC-BY-NC) conviennent à un mémoire étudiant. Un usage commercial futur demanderait de basculer vers M2M-100 (MIT) ou MADLAD (licence à vérifier).

### Gaps
- Aucun temps d'entraînement publié n'a été trouvé pour NLLB-600M, 1.3B ou 3.3B sur un T4 Colab.
- Les hyperparamètres exacts du tutoriel de David Dale (optimiseur, LR, nombre de pas) n'ont pas été confirmés dans les sources lues. Les hyperparamètres de Bayelemabaga (NAACL 2025) n'ont pas été lus.
- Aucune étude n'a été trouvée sur la stabilité du fine-tuning de NLLB en fp16 pur (sans bf16) ni sur des NaN spécifiques à NLLB.
- La licence des checkpoints MADLAD-400 et de Toucan n'a pas été vérifiée.
