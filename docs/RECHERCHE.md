# Recentrer l'assistant bambara sur la parole quotidienne

La baisse ressentie au micro le 16 octobre s'explique d'abord par la manière dont les nouveaux modèles ont été choisis : **les deux tests qui ont servi à les départager ne ressemblent pas à la parole de tous les jours**. Le test de Bayelemabaga, découpé dans l'ordre des sources, ne contient presque que des contes, des épopées et des leçons de grammaire, écrits à 76 % en graphie ancienne. Celui de Jeli-ASR est fait de récits de griots, et un quart de ses phrases longues figurent aussi dans l'entraînement. Le +11,4 de chrF++ de NLLB v2 sur Jeli-ASR et le passage de Whisper à Soloni reposaient donc sur des mesures qui ne prédisaient pas votre voix au téléphone. S'y ajoute une cause visible : sur les énoncés courts, Soloni rend « <tag> », une balise de bruit. La littérature va dans le même sens. Pour le bambara, l'écart de domaine est énorme, et **quelques milliers de paires du bon domaine valent plus qu'un modèle plus gros** : dans MAFAND, finir l'entraînement sur le domaine cible fait gagner 3,7 BLEU en bambara→français. Le plan qui en découle tient sur un T4 gratuit, en trois temps. D'abord mesurer sans entraîner : §9a sur six assemblages, jugements à l'aveugle, décodage CTC de Soloni avec confiance et repli, détection de parole. Ensuite une V3 de NLLB-600M, qui repart de la v1 ou de la moyenne v1/v2 et se nourrit de paires produites par l'atelier sans écrire le bambara, des exemples du Bamadaba, du lexique GATITOS et de MAFAND, avec un rappel de Bayelemabaga. Plus tard seulement, le corpus de référence demandé au LLACAN et des modèles plus gros. Tout ce plan est permis pour un mémoire. En revanche, NLLB, MAFAND, MMS et le Bamadaba interdisent l'usage commercial, et FLORES ne doit jamais entrer dans l'entraînement.

> **Comment lire les niveaux de preuve.** Depuis leur bac à sable, les chercheurs n'ont pas pu joindre Hugging Face, arXiv, ACL Anthology, OPUS ni le site du Corpus Bambara de Référence. Seuls GitHub et un moteur de recherche répondaient. Les mentions suivantes indiquent l'origine de chaque fait :
>
> | Mention | Origine du fait |
> |---|---|
> | aucune | lu à la source : README GitHub, PDF de MAFAND, copies GitHub officielles des fiches Hugging Face de RobotsMali |
> | « mesuré » | compté par les chercheurs sur les fichiers téléchargés |
> | « non vérifié » | extrait de moteur de recherche, à relire avant de le citer dans le mémoire |
> | « projet » | sorties des notebooks du dépôt (`docs/RESULTATS.md`) |
> | « estimation » | déduction sans source publiée |

## La régression vient de tests qui ne ressemblent pas à votre parole

### Deux maillons changés sur la foi d'un corpus de griots

La démo du 15 octobre enchaînait Whisper-small affiné et NLLB v1, celle du 16 Soloni v3 et NLLB v2. Les deux remplacements avaient la même justification : un gain sur Jeli-ASR. Or, sur ce corpus, **l'avance de Soloni sur Whisper n'était même pas établie**. Le WER est de 45,5 % contre 50,3 %, mais l'écart de −4,8 points a un intervalle de confiance de [−18,6 ; 3,6] (projet). Seul l'écart entre Soloni et MMS est solide, et il tient sur les 211 énoncés jamais vus à l'entraînement : 46,4 % contre 61,0 % de WER (projet).

Pour la traduction, la v2 gagne **+1,0 point** de chrF++ bambara→français sur Bayelemabaga [0,3 ; 1,8], mais **+11,4** sur Jeli-ASR [10,0 ; 12,9] (projet). Repartie de la v1, elle a vu Bayelemabaga et les traductions de Jeli-ASR pendant une époque. Si toutes les paires de Jeli-ASR ont servi, les récits de griots (32 180 segments d'entraînement) forment près de la moitié de cette époque, à côté des 37 580 paires de Bayelemabaga (estimation). Comme deux maillons ont changé à la fois, la baisse ressentie ne peut être attribuée ni à l'un ni à l'autre. C'est ce que le nouveau §9a corrige.

Le jeu maison est la seule mesure proche de l'usage réel. Il établit déjà deux choses. D'abord, la cascade Soloni + NLLB v2 (chrF++ 26,7) dépasse nettement le bout-en-bout Whisper (12,9) : c'est un résultat central pour le mémoire. Ensuite, le bout-en-bout Whisper, qui obtenait 20,9 sur l'audio de Jeli-ASR (projet), s'effondre sur la parole quotidienne. Le Whisper du projet n'a vu que des griots ; c'est un indice qu'il généralise mal ailleurs. Ce n'est qu'un indice, car traduire directement depuis l'audio est plus dur que transcrire.

Les valeurs absolues du jeu maison ne se comparent pas à celles des autres tests. La référence y est la phrase française affichée, et le locuteur la dit « comme il parle » : une traduction juste mais formulée autrement y perd des points. Seules les comparaisons entre systèmes sur ce même jeu ont un sens.

### Le test de Bayelemabaga mesure des contes en graphie ancienne

Les chercheurs ont téléchargé Bayelemabaga v1.0.1 depuis GitHub et retrouvé la position de chaque ligne de ses partitions. **Le découpage est séquentiel** : l'entraînement couvre les lignes 1 à 37 580, la validation les suivantes, et le test les lignes 42 279 à 46 976, c'est-à-dire la fin alphabétique des sources (mesuré). Le corpus compte 11 852 lignes religieuses (Nouveau Testament et sourates du Coran), soit **25,2 %**. Elles représentent 27,6 % de l'entraînement et **environ 0 % du test** (mesuré, [archive Bayelemabaga](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/bayelemabaga.tar.gz)).

Le test contient surtout des contes (Sidibé, Sunbunu…), des épopées historiques (Almoravides, princes peuls), un texte ethnographique sur le mariage et les leçons de grammaire de Vydrine ([sources.txt](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/bayelemabaga/sources.txt)). Pour repérer la graphie ancienne, les chercheurs ont compté les lignes bambara de six mots ou plus qui ne contiennent ni ɛ, ni ɔ, ni ɲ. Elles forment **76,3 % du test**, contre 23,2 % de l'entraînement (mesuré). Exemple : « Nk'olu dun ye mogo fila de ye » pour *mɔgɔ*.

Le corpus est aussi bruité (mesuré) :

| Défaut | Nombre |
|---|---|
| copies de paires en double | 1 220 |
| paires au rapport de longueur suspect (au-delà de 2, ou sous 1/2) | 3 705 |
| lignes bambara de trois mots ou moins | 3 434 |
| lignes du test déjà présentes en entraînement | 6 côté français, 23 côté bambara |

Le dialogue y est rare et littéraire. On trouve 2 071 lignes ouvertes par un nom de locuteur, surtout le roman *Sous l'orage* et les manuels, et 2 085 questions. Le README annonce une version 1.0.1 et reconnaît des « last minute misalignment issues » ([RobotsMali-AI/datasets](https://github.com/RobotsMali-AI/datasets)).

Le +1,0 de la v2 mesure donc une généralisation vers des contes et des épopées en graphie ancienne, pas vers le téléphone. Une réserve compte : le projet charge la version Hugging Face (`RobotsMaliAI/bayelemabaga`), et rien ne garantit qu'elle reprenne ce découpage (non vérifié). Deux contrôles de quelques minutes suffisent : si c'est le même découpage, la part des lignes de test sans ɛ, ɔ ni ɲ approchera 76 %, et le mot « Jésus » y sera presque absent.

MAFAND montre jusqu'où peut aller un tel biais de domaine. Un modèle entraîné sur la seule Bible perd **93,5 % de son BLEU** sur la presse bambara. Il traduit « transition » par « la puissance du milieu » et « piliers » par « sauterelles » ([MAFAND, PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf)).

### Le +11 sur Jeli-ASR mélange progrès réel et mémoire

Le projet a compté que **62 des 256 phrases de test d'au moins cinq mots (24,2 %)** de Jeli-ASR figurent dans sa propre partition d'entraînement (87 sur 300 toutes longueurs confondues). La v2, qui a vu cet entraînement, est donc évaluée en partie sur des phrases qu'elle connaît (projet).

La littérature chiffre l'effet. Quand la source et la cible d'un test sont toutes deux contaminées, le BLEU peut gonfler de **jusqu'à 30 points** ([Kocyigit et al., arXiv 2501.18771](https://arxiv.org/abs/2501.18771), non vérifié). Dans un corpus peu doté, retirer les documents de test ne suffit pas : il faut dédoublonner phrase par phrase, au niveau de la cible ([Harb et al., ACL Anthology](https://aclanthology.org/people/abdelrahman-harb/unverified/), non vérifié).

Le registre pèse autant que la fuite. Jeli-ASR réunit une trentaine de narrations d'environ une heure chacune, par 23 hommes et 7 femmes. Elles sont surtout en bambara de Bamako, avec un peu de maninka et de bambara de Ségou ([An ka taa](https://www.ankataa.com/blog/2022/10/13/the-griots-and-tales-of-jeli-asr)). Les paires sont longues et narratives, à l'opposé d'un « i ni sɔgɔma » dit au téléphone.

Une contradiction des notes se tranche ici. L'une affirme que bam-asr-early ne contient qu'environ 3,5 h de Jeli-ASR. La copie GitHub de sa fiche dit au contraire qu'il contient **tout Jeli-ASR (~32,5 h), plus ~4 h de « mali-pense » et ~1 h de lecture**, avec un test de même taille, 1 463 échantillons ([fiche bam-asr-early](https://github.com/RobotsMali-AI/bambara-asr/blob/main/docs/data/RobotsMali/bam-asr-early.md)). Le projet le confirme : 63 phrases longues de test sur 256 s'y retrouvent, contre 62 dans l'entraînement de Jeli-ASR (projet).

Il en découle deux conséquences. D'abord, bam-asr-early n'ajouterait qu'environ 5 h de paires nouvelles. Ensuite, Soloni v0, l'ancêtre de toute la lignée, a appris sur ce corpus, tiré à environ 87 % de Jeli-ASR selon sa fiche ([fiche Soloni v0](https://github.com/RobotsMali-AI/bambara-asr/blob/main/docs/models/RobotsMali/soloni-114m-tdt-ctc-v0.md)). Le test de Jeli-ASR favorise donc à la fois votre Whisper et Soloni. Il ne doit plus décider seul, ni pour la reconnaissance ni pour la traduction.

### « <tag> » est une balise de bruit, pas un mot

Les corpus de RobotsMali annotent le non-verbal avec des balises : `<BRUITS>`, `<INCOMPRÉHENSIBLE>`, `<RIRES>`, `<MUSIQUE>`, `<CHEVAUCHEMENT>`, etc. Le français y est marqué autrement, entre `__doubles soulignés__` ([fiche Kunkado](https://github.com/RobotsMali-AI/bambara-asr/blob/main/docs/data/RobotsMali/kunkado.md)). Les scripts d'entraînement ramènent toutes ces balises à une balise générique `<T>` ([kunkado/scripts](https://github.com/RobotsMali-AI/bambara-asr/tree/main/kunkado/scripts)). Le tokeniseur de Soloni contient d'ailleurs les pièces `<`, `>` et `INCOMPRÉHENSIBLE` ([tokenizer.vocab](https://github.com/RobotsMali-AI/bambara-asr/blob/main/kunkado/tokenizers/soloni/tokenizer.vocab)). « <tag> » signale donc un signal que le modèle prend pour du bruit ou de la parole inaudible.

Ce défaut échappe aux WER publiés, car l'article Kunkado retire les balises avant de compter ([arXiv 2512.19400](https://arxiv.org/abs/2512.19400), non vérifié). Il frappe précisément les énoncés courts, et pour cause. AfVoices a exclu de l'annotation humaine **52 h de segments de moins d'une seconde**, qui n'ont que des étiquettes automatiques ([fiche AfVoices](https://github.com/RobotsMali-AI/bambara-asr/blob/main/docs/data/RobotsMali/afvoices.md)). Les segments relus de Kunkado durent de 0,6 à 15 s. Et aucun corpus public n'est enregistré au téléphone. Un « i ni sɔgɔma » bref, capté par un micro de smartphone, ressemble donc moins à ce que Soloni a appris qu'à un bruit.

Sur 2 des 8 énoncés affichés, NLLB traduisait « Tag » et le LLM répondait qu'il ne pouvait pas aider (projet). Un tel échec, très visible, suffit à dégrader l'impression d'ensemble. Le retrait des annotations et la demande de répéter, faits aujourd'hui, suppriment le contresens, mais pas l'échec : il reste à le rendre plus rare. La chaîne littérale « <tag> » n'apparaît dans aucune source publique. Il faudrait vérifier dans le tokeniseur du checkpoint v3 sur Hugging Face d'où elle vient.

### Trois hypothèses, que le §9a départage

Ces trois causes ne s'excluent pas. Les six assemblages du §9a (Whisper ou Soloni, NLLB v1, v2 ou leur moyenne) les séparent : à traducteur égal, on isole la reconnaissance ; à reconnaissance égale, on isole la traduction.

| Hypothèse | Indices actuels | Ce que le §9a montrerait | Conséquence |
|---|---|---|---|
| **H1. Soloni échoue sur les énoncés courts** | « <tag> » sur 2 énoncés affichés sur 8 ; segments de moins d'une seconde absents de l'annotation humaine | à traducteur égal, Soloni sous Whisper surtout dans les rubriques courtes (salutations), avec beaucoup de sorties vides | décodage CTC, détection de parole avec marges, repli vers un second modèle |
| **H2. NLLB v2 a dérivé vers la narration de griot** | près de la moitié de l'époque v2 tirée de Jeli-ASR (estimation) ; un seul point gagné sur Bayelemabaga ; grand écart de domaine du bambara (MAFAND) | à reconnaissance égale, v1 au moins égale à v2 ; la moyenne entre les deux, ou au-dessus | la V3 repart de la v1 ou de la moyenne, avec des données du quotidien |
| **H3. Le Whisper du projet généralise mal hors des griots** (contre-hypothèse) | bout-en-bout Whisper : 20,9 sur Jeli-ASR, 12,9 sur le jeu maison ; Whisper n'a vu que Jeli-ASR, Soloni v3 de la radio et de la conversation | à traducteur égal, Soloni au-dessus de Whisper sur l'ensemble du jeu maison | garder Soloni : la baisse vient de H1, de H2, ou des deux |

Le §9a ne couvre pas tout ce que vous avez entendu. Votre jugement au micro portait aussi sur les réponses, donc sur le sens français→bambara, que la v2 a également changé, et sur la synthèse. Les jugements à l'aveugle de l'atelier dans les deux sens complètent le §9a.

Deux précautions de méthode s'imposent aussi. D'abord, le §9a retient le meilleur des six assemblages sur les mêmes 555 phrases que celles qui servent à le mesurer : **le chiffre retenu est donc légèrement optimiste**. Pour le mémoire, il faut séparer un dev et un test, ou enregistrer un nouveau lot avec d'autres locuteurs. Ensuite, avec deux locuteurs seulement, les intervalles tirés par énoncé rendent compte de la variété des phrases, pas de celle des voix : les conclusions valent pour ces deux voix tant que d'autres ne sont pas venues s'ajouter.

## Les dictionnaires apportent du lexique court, des formules et une table d'orthographe

### Le Bamadaba est la ressource lexicale la plus directement exploitable

Le Bamadaba de Vydrin, Maslinsky, Méric et al. se télécharge librement sur GitHub au format Toolbox/MDF ([maslinych/bamadaba](https://github.com/maslinych/bamadaba)). Il est bâti sur le dictionnaire de Bailleul (édition de 2007), enrichi par Dumestre 2011 et par le dictionnaire manding-russe de Vydrin et Tomtchina ([README](https://raw.githubusercontent.com/maslinych/bamadaba/master/README.ru.md)). Le fichier principal contient (mesuré, [bamadaba.txt](https://raw.githubusercontent.com/maslinych/bamadaba/master/bamadaba.txt)) :

| Contenu | Nombre |
|---|---|
| entrées | 13 135 |
| gloses françaises lisibles | 22 400 |
| paires d'exemples bambara→français | **4 644** |
| variantes de forme | **5 537** |
| synonymes | 1 849 |
| interjections | 106 |

Tons et orthographe standard (ɛ, ɔ, ɲ, ŋ) y sont marqués. Le fichier `ETRGFRA.txt` recense **1 217 emprunts au français**, par exemple *batiri* → batterie ([ETRGFRA.txt](https://raw.githubusercontent.com/maslinych/bamadaba/master/ETRGFRA.txt)). D'autres fichiers listent 636 prénoms, 393 noms de clan et 953 toponymes.

Deux limites comptent. Le dépôt n'a pas bougé depuis le 6 décembre 2019 (version 0.5.4) ([commits](https://github.com/maslinych/bamadaba/commits/master)). Le dépôt GitHub n'a pas de fichier de licence, et la GPL v3 de l'outil daba ne couvre pas les données ([daba LICENSE](https://raw.githubusercontent.com/maslinych/daba/master/LICENSE.txt)). Mais **la page officielle du Bamadaba le place sous licence Creative Commons BY-NC-SA 3.0 France** : attribution, pas d'usage commercial, partage dans les mêmes conditions ([cormand.huma-num.fr/bamadaba.html](http://cormand.huma-num.fr/bamadaba.html), lu via un moteur de recherche, le site étant injoignable d'ici ; correction du 17 octobre).

Le Bamadaba sert à quatre choses. Ses 4 644 exemples sont des paires courtes et souvent orales, comme « Papa, merci ! » ou « le bonheur soit avec toi ! – merci ». Ils sont ajoutables à l'entraînement après suppression des tons, puisque Bayelemabaga n'en a pas. Ses variantes donnent une table « variante → forme standard ». Ses emprunts aident à reconnaître le mélange avec le français. Ses noms propres permettent de forcer leur traduction au décodage. Comme `sources.txt` ne liste aucun dictionnaire, ces exemples ne sont probablement pas déjà dans vos données (estimation).

### GATITOS donne un lexique bambara–français libre, salutations comprises

GATITOS (Google) traduit 4 000 segments anglais, à 93 % des mots isolés, dans 173 langues dont le bambara, sous **licence CC-BY-4.0** ([README GATITOS](https://raw.githubusercontent.com/google-research/url-nlp/3622039cf51f7eeffa58b957332a8e8c337981d7/gatitos/README.md)). Les fichiers `en_bm.tsv` et `en_fr.tsv` partagent 3 998 clés anglaises : en les joignant sur cette clé, on obtient **environ 4 000 paires bambara–français**, en orthographe standard (mesuré, [en_bm.tsv](https://raw.githubusercontent.com/google-research/url-nlp/3622039cf51f7eeffa58b957332a8e8c337981d7/gatitos/en_bm.tsv)). Elles incluent des formules courantes : « a' ni sɔgɔma », « i ka kɛnɛ wa », « i ni su », « kan bɛn », « barika ».

La jointure par l'anglais trompe parfois. *Aw ni baara* salue des gens au travail, ce n'est pas un « bonjour » générique, et *welcome* devient le verbe « accueillir » alors que *i danse* est une formule. Il faut donc relire. GATITOS a quitté GitHub pour le dépôt Hugging Face de SMOL ([google/smol](https://huggingface.co/datasets/google/smol)), et la version actuelle n'a pas été comparée.

Google rapporte que des lexiques curés comme GATITOS apportent **environ cinq fois plus de gain en chrF** que PanLex, plus gros et plus bruité, par substitution de mots dans la source ou par « lexical prompting » ([BiLex Rx, arXiv 2303.15265](https://arxiv.org/abs/2303.15265v1), non vérifié). Aucun chiffre propre au bambara n'a été trouvé.

### Des outils publiés corrigent l'orthographe, sauf l'écriture « e/o » de tous les jours

Le logiciel daba (GPL v3) contient plusieurs convertisseurs ([plugins daba](https://github.com/maslinych/daba/tree/master/daba/plugins)). Le plus utile, `bamlatinold`, convertit l'ancienne graphie : è→ɛ et ò→ɔ sont sûrs, tandis que ny→ɲ et ng→ŋ restent ambigus et doivent être départagés par le lexique ([bamlatinold.py](https://raw.githubusercontent.com/maslinych/daba/master/daba/plugins/bamlatinold.py)). S'y ajoutent un normaliseur d'apostrophes et la fonction `detone()`, qui retire les tons.

D'autres ressources complètent ces outils. Le correcteur Hunspell dérivé du Bamadaba (24 651 mots, licence de type BSD) intègre l'atelier d'orthographe tenu à Bamako en 2022 ([extension LibreOffice](https://extensions.libreoffice.org/en/extensions/show/21654), non vérifié). Vydrin (2022) propose un dictionnaire orthographique validé par des experts maliens ([Mandenkan 68](https://doaj.org/article/a16a2eb5ac564fc490eb276331c63a4d), non lu). MALIBA-AI publie aussi un normaliseur ([bambara-text-normalization](https://github.com/MALIBA-AI/bambara-text-normalization)).

Une limite vous concerne directement. Quand on écrit le bambara sans le connaître à l'écrit, on met souvent « e » et « o » à la place de « ɛ » et « ɔ ». **Aucune règle ne permet de revenir en arrière.** Il faut chercher la forme dans un lexique, avec une clé où ɛ et ɔ sont repliés sur e et o. C'est le principe de la variante `corbama-nul` du corpus ([corbama-build](https://raw.githubusercontent.com/maslinych/corbama-build/master/README.md)). Le projet mesure déjà le chrF++ « graphie repliée ».

### Le Corpus Bambara de Référence cache environ 27 000 phrases parallèles de plus

Les fichiers d'alignement publics du corpus de référence couvrent **569 documents et 67 776 phrases bambara** (mesuré, [corbama-bam-fra.prl](https://raw.githubusercontent.com/maslinych/corbama-build/master/corbama-bam-fra.prl)). En rapprochant les noms de fichiers, les chercheurs estiment qu'**environ 27 000 phrases** viennent de documents absents de Bayelemabaga v1.0.1. L'appariement par noms est approximatif. Parmi ces documents figurent les cours *An ka bamanankan kalan* et *Bamanankan maben*, les plus proches des dialogues de la vie courante.

Les textes eux-mêmes sont dans un dépôt privé ([README corbama-build](https://raw.githubusercontent.com/maslinych/corbama-build/master/README.md)), et aucune licence n'a été trouvée. Il faut demander un export à l'équipe du LLACAN (Vydrin, Maslinsky, Méric). Le site annoncerait environ 11,8 millions de mots en juillet 2025, mais un autre résumé du même site le contredit ([cormand.huma-num.fr](https://cormand.huma-num.fr/), non vérifié).

### Aucun lexique d'argot bamakois n'existe : le vôtre sera le premier

Les chercheurs n'ont trouvé aucun lexique publié de l'argot des jeunes de Bamako. Le sens urbain de « wai » et « yoh » n'est attesté par aucune source lexicographique. Le point de départ le plus proche est la liste des interjections du Bamadaba (mesuré) :

| Bambara | Glose |
|---|---|
| *yó / yóo* | « salut », vocatif insistant |
| *wàláyì* | « je le jure » |
| *wáyi* | « aïe », sans doute distinct du « wai » urbain |
| *nbá*, *nsé* | « salut » |
| *tɛ́sadì* | « n'est-ce pas » |
| *jàa* | « en fait » |

Un fil du forum An ka taa sur les abréviations SMS et l'argot n'a pas pu être lu ([forum](https://ankataa.discourse.group/c/language/16)). L'onglet Lexique de l'atelier est donc la seule source pour ce registre.

### Les ouvrages à citer dans le mémoire

| Ouvrage ou ressource | Ce qu'il apporte | Accès | Statut |
|---|---|---|---|
| Bailleul, *Dictionnaire bambara-français* (Donniya, 1996 ; 3ᵉ éd. 2007) et *français-bambara* (1998) | dictionnaire tonal de référence, base du Bamadaba | imprimé | [Glottolog](https://glottolog.org/resource/reference/id/140867) |
| Dumestre, *Dictionnaire bambara-français* (Karthala, 2011) | dictionnaire de l'INALCO, source du Bamadaba | imprimé | [An ka taa](https://www.ankataa.com/blog/tag/dictionaries), non vérifié |
| Kassim Kone, *Bamanankan Daɲɛgafe* | dictionnaire monolingue | PDF de l'ALMA | [ALMA](https://alma.matrix.msu.edu/assets/uploads/2020/01/Bambara_005-Danyegafe.pdf), non vérifié |
| Dictionnaire An ka taa (Donaldson, Fenayon) | bambara/dioula–anglais, index français, tons, emprunts | en ligne, sans licence de réutilisation trouvée | [dictionary.ankataa.com](https://dictionary.ankataa.com/lexicon.php), non vérifié |
| Leçon Kalan « Essential expressions » | salutations, deux formes de « oui » | en ligne | [kalan.ankataa.com](https://kalan.ankataa.com/bambara-beg/grammar-i/essential-expressions) |
| Bamba, *Bamanankan Lexicon* (LDC2016L01) | 5 978 entrées bambara–français et bambara–anglais | licence LDC | [catalogue LDC](https://catalog.ldc.upenn.edu/LDC2016L01), non vérifié |
| Marguets, *Lexique bambara-français-anglais* (SIL, 2010) | 3 512 entrées | annoncé sur bambara.org | [Webonary](https://www.webonary.org/dioula-bf/overview/foreword/?lang=en), non vérifié |
| Mbodj-Pouye et Van den Avenne (2007), « C'est bambara et français mélangés » | l'alternance bambara–français à l'écrit | revue | [Cairn](https://www.cairn.info/revue-langage-et-societe-2007-2-page-99.htm), non lu |
| Dumestre, « Emprunts syntaxiques du bambara au français » | emprunts dans le parler des journalistes bamakois | PDF | [iling-ran](https://iling-ran.ru/library/languageinafrica/1/LiA_4_1_Dumestre.pdf), non lu |

## Il n'existe presque pas de bambara–français quotidien : le corpus manquant est le vôtre

### Un seul corpus parallèle humain s'ajoute sans risque : MAFAND

En dehors de Bayelemabaga et de Jeli-ASR, le seul bambara–français écrit par des humains, libre et sûr à court terme, est **MAFAND-MT fr-bam**. C'est de la presse malienne de 2021, traduite à partir du français, en graphie standard : moins de 2 % des lignes sont sans ɛ ni ɔ.

Les tailles ne concordent pas selon la source. Les fichiers GitHub contiennent **3 013 / 1 500 / 1 500 paires** (mesuré), là où le PDF annonce 3 302 / 1 484 / 1 600. Les chercheurs n'ont trouvé **aucune phrase commune avec Bayelemabaga**, mais 24 phrases bambara du test reviennent dans son entraînement : il faudra les retirer. La licence est **CC-BY-4.0-NC** ([lafand-mt](https://github.com/masakhane-io/lafand-mt) ; [PDF MAFAND](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf)). Le reste se répartit entre des données pivotées par l'anglais, des données minées ou synthétiques, et des jeux de test.

| Corpus | Taille | Domaine et qualité | Licence | Verdict |
|---|---|---|---|---|
| Bayelemabaga v1.0.1 | 46 976 paires (mesuré) | contes, presse Fasokan, NT et Coran (25 %), grammaire ; humain, alignement bruité | CC-BY-SA-4.0 (README) ou CC-BY-4.0 (barre latérale) | déjà utilisé ; à garder en rappel, avec un test re-découpé ([GitHub](https://github.com/RobotsMali-AI/datasets)) |
| MAFAND fr-bam | 3 013 / 1 500 / 1 500 (mesuré) | presse 2021, humain, graphie standard | CC-BY-4.0-NC | **à ajouter en V3** ; son test sert de second test propre |
| Traductions de Jeli-ASR | 33 643 segments | récits de griots, humain | CC-BY-4.0 | à garder, dédoublonné et découpé par narrateur ([fiche](https://github.com/RobotsMali-AI/bambara-asr/blob/main/docs/data/RobotsMali/jeli-asr.md)) |
| bam-asr-early | 37,41 h, dont tout Jeli-ASR | + ~4 h « mali-pense », ~1 h de lecture | CC-BY-4.0 | n'apporte qu'environ 5 h nouvelles |
| djelia/bambara-mt-dataset | 10 k à 100 k entrées | manuels scolaires EGAFE, livre de Hoffman ; trilingue | étiquette cc-by-sa-4.0 | plus tard, après contrôle du recouvrement ([HF](https://huggingface.co/datasets/djelia/bambara-mt-dataset), non vérifié) |
| oza75/bambara-mt | ~4,99 M lignes | Bamadaba, synthétique Google Translate, sous-titres « Mali Pense » traduits à la main | inconnue | seul le sous-ensemble Mali Pense intéresse ([HF](https://huggingface.co/datasets/oza75/bambara-mt), non vérifié) |
| Tapo et al. 2020 | 2 146 paires | exemples du dictionnaire SIL Mali | disponibilité inconnue | à chercher ([LoResMT 2020](https://aclanthology.org/2020.loresmt-1.3/), non vérifié) |
| Tapo et al. 2021 | taille inconnue | guide de santé | publication annoncée | recouvre sans doute les fichiers `dogotoro` de Bayelemabaga ([arXiv 2104.00041](https://arxiv.org/abs/2104.00041), non vérifié) |
| NLLB-Seed / OLDI Seed | ~6 000 phrases anglais–bambara | Wikipédia ; jugé trop francisé, avec des entrées entièrement en français et sans tons | libre | faible priorité ; pivot par l'anglais ([NLLB Seed](https://github.com/facebookresearch/flores/tree/main/nllb_seed) ; critique : [Doumbouya et al.](https://aclanthology.org/2023.wmt-1.34.pdf), non vérifié) |
| SMOL (Google) | ligne « bm » : 863 et 260, sens des colonnes non confirmé | source anglaise, traduction professionnelle | non vérifiée | plus tard, par pivot ([arXiv 2502.12301](https://arxiv.org/pdf/2502.12301), non vérifié) |
| NLLB minées fra–bam | 56 151 paires après nettoyage | web, alignement automatique, décontaminé de FLORES+ | non vérifiée | gain faible : NLLB les a sans doute déjà vues ([burkimbia](https://huggingface.co/datasets/burkimbia/nllb-burkina-parallel), non vérifié) |
| kalilouisangare/corpus-parallel-en-fr-bam | taille inconnue | bambara **produit par NLLB-200** | accès restreint | **à exclure** : on réapprendrait au modèle ses propres sorties ([HF](https://huggingface.co/datasets/kalilouisangare/corpus-parallel-en-fr-bam), non vérifié) |
| AfroLingu-MT | agrégat de 43 jeux | contient probablement MAFAND et FLORES | — | seulement après dédoublonnage ([Toucan](https://github.com/UBC-NLP/Toucan)) |
| FLORES+ bam_Latn | 997 dev + 1 012 devtest | Wikinews, Wikivoyage, Wikibooks ; traduction professionnelle | CC BY-SA 4.0, zip protégé | **test seulement, jamais en entraînement** ([flores-OLDI](https://github.com/avidale/flores-OLDI)) |
| NTREX-128, Tatoeba | pas de bambara ; 7 phrases bam-eng | — | — | inutiles ([NTREX](https://github.com/MicrosoftTranslator/NTREX) ; [Tatoeba](https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/data/README.md)) |

### Le bambara monolingue existe surtout à l'écrit littéraire, sauf dans les transcriptions de parole

Plusieurs sources monolingues existent, mais peu ressemblent à la parole. Le **Corpus Bambara de Référence** compterait environ 11,8 M de mots (chiffre contradictoire, non vérifié), mais Bayelemabaga en est tiré, d'où un risque de recouvrement. **MADLAD-400 bm** ne pèse que 1,17 Mo compressé en version propre et 15,1 Mo en version bruitée, sous ODC-BY ([HF](https://huggingface.co/datasets/allenai/MADLAD-400/tree/95bb50141e1945da136b7968b99e15065a803835/data/bm), non vérifié). Dans **FineWeb-2 bam_Latn**, le dioula a été fusionné avec le bambara, ce qui impose un filtre ([HF](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), non vérifié). **Wikipédia** compte environ 930 articles ([Wikipedia](https://en.wikipedia.org/wiki/Bambara_Wikipedia)). Aucune archive numérisée du journal Kibaru n'a été trouvée.

Pour la langue parlée, la vraie source monolingue est faite des **transcriptions d'AfVoices et de Kunkado**. Elles n'ont pas de traduction française confirmée.

### Les corpus de parole sont conversationnels ou radiophoniques, jamais téléphoniques

| Corpus | Volume | Type de parole | Français | Licence |
|---|---|---|---|---|
| Jeli-ASR | 32,48 h ; 32 180 / 1 463 segments | récits, environnement contrôlé | oui | CC-BY-4.0 ([fiche](https://github.com/RobotsMali-AI/bambara-asr/blob/main/docs/data/RobotsMali/jeli-asr.md)) |
| Kunkado | 161,15 h, dont 39,27 h relues (33,47 h d'entraînement / 5,80 h de test) | radio et télévision de Bamako, alternance avec le français, bruit | non | CC-BY-SA-4.0 selon la copie de fiche, non confirmée par une autre note ([fiche](https://github.com/RobotsMali-AI/bambara-asr/blob/main/docs/data/RobotsMali/kunkado.md)) |
| AfVoices | 423 h segmentées sur 612 h brutes ; 159 h corrigées à la main ; 52 h de segments de moins d'une seconde, étiquetés automatiquement | conversation, sud du Mali | non confirmé | CC-BY-4.0 ([fiche](https://github.com/RobotsMali-AI/bambara-asr/blob/main/docs/data/RobotsMali/afvoices.md)) |
| Nyana Eval | 45 clips, ~3 min | Kunkado, interviews de rue, application de lecture ; 15 % d'alternance | non | CC-BY-4.0 |
| Bam_ASR_Eval_500 | 500 clips, ~37 min | — | non | CC-BY-4.0, mais « non-commercial research » sur la même fiche |
| Banc d'essai MALIBA | 1,075 h | Constitution lue en studio, sans alternance | non | ([README](https://github.com/MALIBA-AI/bambara-asr-leaderboard)) |
| an-be-kalan-bench | 45,6 h | lecture par des enfants | non | CC-BY-4.0 |
| oza75/bambara-tts ; WAXAL | ~30 107 exemples ; part bambara non chiffrée | synthèse vocale | — | CC-BY-SA-4.0 ([HF](https://huggingface.co/datasets/oza75/bambara-tts) ; [WAXAL](https://arxiv.org/abs/2602.02734), non vérifié) |

Les fiches de Nyana Eval, Bam_ASR_Eval_500 et an-be-kalan-bench sont regroupées dans [docs/data](https://github.com/RobotsMali-AI/bambara-asr/tree/main/docs/data/RobotsMali). Le bambara est **absent de Common Voice 27.0**, alors que le dioula y a 0,51 h ([cv-dataset](https://github.com/common-voice/cv-dataset/tree/main/datasets/scripted-speech)). Il est probablement absent de FLEURS ([arXiv 2205.12446](https://arxiv.org/abs/2205.12446), non vérifié). Aucun corpus téléphonique n'a été trouvé.

Corriger à la main la transcription d'une heure de bambara coûte environ 30 h de travail ([arXiv 2510.12781](https://arxiv.org/abs/2510.12781), non vérifié). Le choix de l'atelier, où la phrase française affichée sert de référence, évite ce coût et le besoin d'écrire le bambara. **Le corpus qui manque à toute la littérature, des paires bambara parlé–français du quotidien, ne peut venir que de là.**

## NLLB-600M reste le bon traducteur ; la concurrence se joue sur la reconnaissance

### Aucun traducteur ouvert n'a prouvé qu'il battait NLLB affiné sur le bambara

Aucun score FLORES bambara↔français n'a pu être récupéré. Les fichiers de métriques de NLLB étaient inaccessibles, et pour les modèles denses ils ne contiennent que les directions centrées sur l'anglais plus 200 directions tirées au hasard ([README NLLB](https://github.com/facebookresearch/fairseq/blob/nllb/examples/nllb/modeling/README.md)).

La seule comparaison publiée et recoupée vient du test de presse de MAFAND. Un **M2M-100 418M affiné atteint 25,8 BLEU / 49,0 chrF en bambara→français**, alors que NLLB-200 sans affinage n'y fait que **14,6 / 37,5 chrF++**, probablement avec le modèle 54B ([article NLLB](https://arxiv.org/pdf/2207.04672), non vérifié ; [PDF MAFAND](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf)). Ces chiffres mélangent chrF et chrF++ et portent sur un autre test : ils ne se comparent pas au 27,8 du projet.

Les grands modèles de langue n'ont pas fait mieux. L'article Bayelemabaga conclut que les encodeurs-décodeurs comme NLLB battent Mistral et Llama 3 8B ([NAACL 2025](https://aclanthology.org/2025.naacl-long.602/), non vérifié). Un Llama 3 8B affiné y plafonne à 9,82 BLEU / 19,00 chrF en français→bambara ([Bonfanti et al.](https://arxiv.org/html/2509.12514v1), non vérifié). Plus largement, les systèmes de traduction classiques battent ChatGPT sur 84,1 % des langues de FLORES-200 ([Robinson et al., WMT 2023](https://aclanthology.org/2023.wmt-1.40/), non vérifié). Aucun score bambara n'a été trouvé pour GPT-4o, Gemini, Claude, Gemma, Llama 3/4, Aya ou Qwen. Qwen2.5-1.5B doit donc continuer à ne recevoir que du français.

| Modèle | Bambara | Repère publié | Sur un T4 | Licence |
|---|---|---|---|---|
| **NLLB-200-distilled-600M** (actuel) | oui | sans affinage 21,8 → v1 27,8 chrF++ sur Bayelemabaga (projet) | affinage complet, large marge (estimation) | CC-BY-NC 4.0 |
| NLLB-200-distilled-1.3B | oui | — | affinage complet serré : Adafactor et *gradient checkpointing* (estimation) | CC-BY-NC 4.0 |
| NLLB-200 3.3B | oui | — | LoRA ou QLoRA seulement ; environ 5 fois plus lent sur CPU (estimation) | CC-BY-NC 4.0 |
| M2M-100 418M | via MAFAND | 25,8 BLEU / 49,0 chrF bambara→français sur la presse | facile | MIT selon les chercheurs (non vérifié) |
| Toucan-1.2B (UBC-NLP) | `bam` listé | +6,96 spBLEU en moyenne sur NLLB-1.3B, toutes paires confondues, pas pour le bambara | possible | non vérifiée ([Toucan](https://github.com/UBC-NLP/Toucan)) |
| MADLAD-400 3B | balise `<2bm>` | aucun score bambara | lourd | non vérifiée ([vocabulaire](https://storage.googleapis.com/madlad-400-checkpoints/vocabulary/256k_vocab/spm.vocab)) |
| Youssouf0123/bambara-french-nllb | oui | même recette que votre v2 ; 16,5 BLEU bambara→français, test inconnu | — | NC par héritage ([HF](https://huggingface.co/Youssouf0123/bambara-french-nllb), non vérifié) |

Deux candidats souvent cités ne couvrent pas le bambara : SeamlessM4T v2 ([table des langues](https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/README.md)) et AfriNLLB ([AfricaNLP 2026](https://aclanthology.org/2026.africanlp-main.30/), non vérifié). Le diagnostic plaide contre un changement de modèle : **le défaut vient des données, pas de la capacité**. M2M-100 sert surtout de comparaison, et de porte de sortie si un usage commercial devenait un objectif.

### Trois reconnaisseurs ouverts devancent Soloni sur le seul banc d'essai commun

Le classement MALIBA-AI réunit 48 systèmes, avec WER et CER en pourcentage ([leaderboard.csv](https://github.com/MALIBA-AI/bambara-asr-leaderboard/blob/main/leaderboard.csv)) :

| Modèle | WER | CER | Remarque |
|---|---|---|---|
| MALIBA-AI/bambara-asr-v3 | **45,73** | 13,45 | fiche non lue, licence inconnue |
| soloba-ctc-0.6b-v3 (RobotsMali) | 46,76 | 16,02 | |
| djelia/asr-v2 | 47,50 | 13,56 | propriétaire |
| soloni-v3 | 48,32 | 14,81 | CTC et TDT à égalité |
| MMS-1b-all | 61,06 | 14,71 | |
| Omnilingual LLM-7B | 62,57 | 15,08 | |
| whisper-small sans affinage | 109,97 | — | |

Ce banc d'essai est une Constitution lue en studio, sans alternance de langues. Les auteurs préviennent que les erreurs réelles seront plus élevées ([arXiv 2602.09785](https://arxiv.org/abs/2602.09785), non vérifié).

**Soloba-CTC-0,6B-v3** a un avantage précis pour vous : sa fiche dit qu'il « cannot produce acoustic event tags » ([fiche Soloba](https://github.com/RobotsMali-AI/bambara-asr/blob/main/docs/models/RobotsMali/soloba-ctc-0.6b-v3.md)). Pour Soloni v3, la branche CTC égale la branche TDT sur Kunkado et la bat sur Nyana Eval (**32,33 % contre 35,32 %**). C'est aussi la branche qui permet la confiance et la fusion avec un modèle de langue ([fiche Soloni v3](https://github.com/RobotsMali-AI/bambara-asr/blob/main/docs/models/RobotsMali/soloni-114m-tdt-ctc-v3.md)).

Omnilingual ASR de Meta couvre `bam_Latn` avec 15,1 h d'entraînement, sous **Apache-2.0**. Ses modèles CTC 300M et 1B demandent environ 2 à 3 Gio de mémoire graphique ([README Omnilingual](https://github.com/facebookresearch/omnilingual-asr)), mais il reste loin derrière au classement.

Le Whisper du projet a fait 3 781 pas en lot effectif de 16, soit 2 h 52 sur T4 (projet). C'est proche de la recette de référence de Hugging Face, 5 000 pas en lot de 16 ([blog fine-tune-whisper](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md)). Sa faiblesse tient donc plus à ses données, uniquement des griots, et aux hallucinations qu'à un manque d'entraînement. Whisper invente des phrases entières dans environ 1 % des transcriptions, surtout quand il y a de longues pauses ([Koenecke et al.](https://doi.org/10.1145/3630106.3658996), non vérifié).

La traduction parole→texte directe reste faible, même dans son domaine : le modèle st-soloni de RobotsMali atteint 24,18 BLEU sur le test de Jeli-ASR ([fiche](https://github.com/RobotsMali-AI/bambara-asr/blob/main/docs/models/RobotsMali/st-soloni-114m-tdt-ctc.md)). C'est cohérent avec le 12,9 du bout-en-bout sur votre jeu maison.

Pour la synthèse, comme vous n'enregistrerez pas votre voix, la seule piste est de comparer MMS-TTS à d'autres modèles existants, eux aussi non commerciaux : MALIBA-AI/bambara-tts et sudoping01/bambara-tts ([HF](https://huggingface.co/MALIBA-AI/bambara-tts), non vérifié). C'est une priorité basse, puisque le maillon faible est ailleurs.

## L'adaptation au domaine pèse plus que toutes les autres techniques réunies

### Quelques milliers de paires du bon domaine font l'essentiel du gain

La preuve la plus solide pour le bambara vient de MAFAND. Enchaîner « gros corpus hors domaine + domaine cible » puis « domaine cible seul » fait passer le bambara→français de **22,1 à 25,8 BLEU** (chrF de 45,9 à 49,0), par rapport au seul domaine cible. La qualité croît continûment de 0 à 31 000 phrases du domaine, et le titre de l'article résume la leçon : « A few thousand translations go a long way » ([PDF MAFAND](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/a_few_thousand_translations_go.pdf)). Votre v2 a fait l'inverse : elle a fini sur un domaine, les griots, qui n'est pas celui de l'assistant.

### Oublier moins coûte presque rien

Ce qu'une V3 doit éviter, c'est d'oublier Bayelemabaga en apprenant le quotidien, comme la v2 a peut-être oublié le quotidien en apprenant les griots. Le moyen le plus simple est de **repartir de la v1 avec un taux d'apprentissage bas** : les recettes publiées pour NLLB utilisent 5e-5 ([article NLLB](https://arxiv.org/pdf/2207.04672), non vérifié). On peut aussi **rappeler une part de Bayelemabaga dans le mélange** ; la proportion de 20 à 50 % est une heuristique, sans source propre au bambara. Pénaliser l'écart aux poids d'origine (EWC) conserve l'essentiel du domaine général sans nuire au domaine cible ([Thompson et al., NAACL 2019](https://aclanthology.org/N19-1209/), non vérifié), au prix d'un code plus complexe. Enfin, **moyenner les poids** est gratuit, et `scripts/moyenne_poids.py` le permet déjà puisque la v2 prolonge la v1. Le seul gain publié est d'environ +1,7 chrF++ pour un NLLB-600M affiné sur des langues amérindiennes ([AmericasNLP 2023](https://preview.aclanthology.org/new-sigs/2023.americasnlp-1.19/), non vérifié).

LoRA reste proche de l'affinage complet, à moins de 0,85 BLEU d'écart dans un cas NLLB ([EAMT 2026](https://aclanthology.org/2026.eamt-1.25.pdf), non vérifié). Sur le 600M, qui tient en entier sur un T4, l'affinage complet reste le plus simple.

### Entraîner la traduction sur des sorties de reconnaissance

Bruiter les transcriptions avec un bruit estimé sur de vraies sorties de reconnaissance donne presque le même résultat qu'un entraînement sur ces sorties. On ne perd rien sur le texte propre, et le gain se transfère à d'autres reconnaisseurs ([Martucci et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/martucci21_interspeech.html), non vérifié). Un entraînement adversarial apporte jusqu'à +2,83 BLEU sur entrée bruitée ([arXiv 1909.11430](https://ar5iv.labs.arxiv.org/html/1909.11430), non vérifié). À l'inverse, traduire des sorties de reconnaissance au lieu de transcriptions propres coûte de 6,79 à 10,64 BLEU en vietnamien→anglais ([arXiv 2606.12911](https://arxiv.org/pdf/2606.12911), non vérifié).

Dans votre chaîne, la reconnaissance coûte **3,2 points de chrF++ avec Soloni et 4,5 avec Whisper** (projet). Une partie de cet écart est récupérable. **Cette technique ne vaut que pour le sens bambara→français.** Il ne faut jamais apprendre au sens français→bambara à produire du bambara bruité.

### Lexique, orthographe, décodage, rétro-traduction : des gains plus minces

L'augmentation lexicale de type BiLex Rx remplace des mots de la source par leur traduction, et NLLB recopie volontiers les mots français. Elle offre donc une voie simple pour l'argot : remplacer « wai » par sa glose française avant de traduire. Cela reste à mesurer sur les phrases où l'atelier a noté de l'argot. Le décodage contraint (`force_words_ids` dans Transformers) impose la traduction des noms propres du Bamadaba.

La normalisation orthographique, appliquée à la fois aux données d'entraînement et à la sortie de la reconnaissance, enchaîne cinq opérations proposées par les chercheurs. On normalise d'abord l'Unicode (NFC) et les apostrophes, puis on retire les tons. On convertit ensuite è→ɛ et ò→ɔ, qui sont sûrs, mais ny→ɲ et ng→ŋ seulement si le mot obtenu figure dans le lexique. On ramène enfin chaque variante du Bamadaba à sa forme standard.

La rétro-traduction taguée, qui marque la source synthétique par un jeton, donne des résultats contrastés en peu doté. La version taguée gagne +6,81 BLEU contre +4,92 pour la version standard en allemand→anglais, mais +2,83 contre +3,06 en anglais→vietnamien ([arXiv 1912.10514](https://arxiv.org/pdf/1912.10514), non vérifié). Pour le bambara→français, elle a deux variantes. La rétro-traduction classique part de français quotidien traduit vers le bambara par NLLB, mais ce bambara synthétique ne ressemble pas à la parole. La traduction directe des transcriptions de Kunkado et d'AfVoices part au contraire d'un vrai bambara parlé, mais avec un français synthétique qui peut renforcer les erreurs du modèle. Les deux sont à garder pour plus tard.

Le décodage MBR bat le faisceau en évaluation humaine, mais il est coûteux et n'a pas été évalué sur une langue africaine ([arXiv 2305.09860](https://arxiv.org/pdf/2305.09860), non vérifié).

### Côté reconnaissance : couper, juger sa confiance, se rabattre

Trois techniques s'appliquent sans réentraîner le modèle acoustique. La **détection de parole** d'abord : Silero VAD (MIT) traite le 8 et le 16 kHz, et un fragment de 30 ms en moins d'une milliseconde sur CPU ([silero-vad](https://github.com/snakers4/silero-vad)). Il faut garder une marge de 200 à 300 ms autour de la parole, sinon les énoncés courts perdent leur attaque et basculent vers la balise de bruit (estimation). La **confiance** ensuite : NeMo calcule des confiances par mot en décodage glouton, et son tutoriel cite justement l'usage « demander de répéter » ([ASR_Confidence_Estimation](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_Confidence_Estimation.ipynb)). Le **modèle de langue** enfin : un n-gramme d'ordre 6 sur les jetons BPE, fusionné sur GPU (NGPU-LM) avec du *boosting* de phrases, corrige les formules courtes, pour seulement 20 % d'écart de vitesse entre faisceau et glouton selon NVIDIA ([doc NeMo](https://github.com/NVIDIA/NeMo/blob/main/docs/source/asr/asr_language_modeling_and_customization.rst) ; [ngram_utils](https://github.com/NVIDIA/NeMo/blob/main/docs/source/asr/asr_customization/ngram_utils.rst)). RobotsMali n'a publié aucun modèle de langue pour Soloni, ni de recommandation de faisceau.

L'affinage en domaine, enfin, réduit beaucoup le WER : de 42 % à 22 % pour Soloni affiné sur 45,6 h d'enfants ([fiche soloni-be-kalan](https://github.com/RobotsMali-AI/bambara-asr/blob/main/docs/models/RobotsMali/soloni-be-kalan-v0.md)).

### Évaluer proprement est une technique à part entière

La métrique de référence est le chrF++ de sacrebleu avec `--chrf-word-order 2` ([README FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)). MAFAND confirme que chrF convient mieux que BLEU aux langues à morphologie riche. Le protocole proposé par les chercheurs est une inférence méthodologique, sans étude dédiée au bambara. Il retire de l'entraînement toute paire dont la source **ou** la cible normalisée apparaît dans un test, puis écarte les quasi-doublons, par exemple au-delà de 0,8 de similarité sur les 4-grammes de caractères, fréquents quand une narration longue est découpée en segments. Il découpe Jeli-ASR par narrateur plutôt que par segment, et publie les scores à la fois sur la partie propre et sur la partie chevauchante, pour montrer l'écart. Pour la parole quotidienne, il vise environ 300 phrases de dev et 300 de test (estimation), comparées par bootstrap apparié.

## Plan d'action : mesurer sans entraîner, puis une V3 ciblée, puis agrandir

Trois règles traversent tout le plan. **Une seule chose change à chaque étape**, pour que chaque gain ou chaque perte ait une cause. **Chaque changement se juge sur la parole quotidienne** : le chrF++ du français obtenu depuis l'audio du jeu maison, ventilé par rubrique, par interlocuteur (ami, inconnu, aîné) et sur les sous-ensembles où l'atelier a coché « mélangé avec du français » ou noté des mots familiers, complété par le taux de « répétez ». Enfin, **on vérifie qu'on n'oublie rien** : Bayelemabaga, le test de MAFAND, FLORES+ et la partie propre de Jeli-ASR servent de contrôles de non-régression, jamais de critère de choix. Aucune étape prioritaire ne demande d'écrire le bambara.

### Étape 1 — cette semaine, sans entraînement

| Action | Coût sur T4 | Mesure sur la parole quotidienne |
|---|---|---|
| **1a.** Lancer le §9a : Whisper ou Soloni, avec NLLB v1, v2 ou la moyenne | inférence seule, moins d'une heure (estimation d'après les latences mesurées : 0,69 s par énoncé pour la cascade Soloni, 1,47 s avec Whisper) | chrF++ par assemblage, intervalles par énoncé, par rubrique ; taux de sorties vides après filtrage ; lecture avec le tableau des hypothèses H1 à H3 |
| **1b.** Couper le jeu maison en dev et test **par phrase**, puisque les deux locuteurs disent les mêmes phrases ; prévoir un nouveau lot avec d'autres locuteurs pour le chiffre final | aucun | le dev sert au choix, le test au mémoire |
| **1c.** Jugements à l'aveugle dans l'onglet Juger : v1 contre v2, puis contre la moyenne, **dans les deux sens** | quelques minutes d'inférence pour préparer les lots | test du signe, une cinquantaine de paires par sens (estimation) ; seul moyen de juger le sens français→bambara des réponses |
| **1d.** Soloni en décodage CTC, avec confiance NeMo ; repli vers Whisper ou Soloba si la sortie est vide, balisée ou peu confiante, sinon demander de répéter | aucun entraînement ; seuil réglé sur le dev | taux de balises et de « répétez », chrF++ sur les rubriques courtes |
| **1e.** Silero VAD avec 200 à 300 ms de marge, conversion en mono 16 kHz, normalisation du gain | CPU, négligeable | mêmes mesures, surtout sur les salutations |
| **1f.** Essayer des reconnaisseurs tout prêts : Soloba-CTC-0,6B-v3, MALIBA-AI/bambara-asr-v3 (licence à lire d'abord), Omnilingual CTC 1B | inférence ; 0,6 B tient sans peine ; Omnilingual en fp16 sur T4 non testé | chrF++ à traducteur fixe (v1) sur le dev maison ; WER sur Nyana Eval (45 clips transcrits, dont des interviews de rue) |
| **1g.** Mettre en place une évaluation propre : contrôler le découpage Bayelemabaga de Hugging Face (76 % de lignes sans ɛ/ɔ, « Jésus » absent) ; scores Jeli-ASR sur les 211 énoncés jamais vus ; ajouter le test MAFAND (1 500) et FLORES+ devtest (1 012) | quelques minutes d'inférence par modèle | non-régression seulement : ces tests sont écrits |

Le §9a donne directement les décisions. Si, à traducteur égal, Whisper bat Soloni de façon nette, gardez Whisper dans la démo et faites de Soloni le second avis, ou l'inverse. Si l'écart n'est pas établi, Soloni reste le meilleur pari : il est mieux classé sur les bancs indépendants et entraîné sur de la radio et de la conversation. Il doit alors être protégé par 1d et 1e.

Pour la traduction, si la v1 égale ou bat la v2, la V3 repart de la v1. Si la moyenne bat les deux, elle devient à la fois la démo et le point de départ, gratuitement. Si tout reste dans les intervalles, la réponse viendra des jugements à l'aveugle et d'un jeu maison agrandi à d'autres locuteurs.

### Étape 2 — la V3 de NLLB, puis la reconnaissance si le §9a l'exige

La V3 suit la recette de MAFAND : un entraînement court, au contenu proche du domaine, avec un rappel de l'ancien. Sa matière première est ce que l'atelier produit sans écrire le bambara.

Chaque enregistrement « Dire » d'une phrase **hors du test**, passé dans Soloni, donne une paire (sortie de la reconnaissance, phrase française). C'est à la fois du domaine quotidien et l'entraînement sur sorties de reconnaissance de Martucci et al. Les variantes « autre façon de la dire » et « à quelqu'un d'autre » multiplient les formulations. Visez **2 000 à 3 000 paires**, l'ordre de grandeur de MAFAND.

| Source | Volume | Sens | Licence | Précaution |
|---|---|---|---|---|
| Atelier : (sortie de la reconnaissance, phrase française) | 2 000 à 3 000 à collecter | **bambara→français seulement** | données propres, consentement selon `docs/COLLECTE.md` | jamais une phrase du test ; écarter les sorties vides ou balisées |
| Atelier : lexique d'argot, en phrases-gabarits ou par substitution de la glose française | quelques centaines | bambara→français | propres | prendre l'orthographe que la reconnaissance donne au mot ; garder une validation à part |
| Exemples du Bamadaba, tons retirés | 4 644 | les deux | **aucune** | usage de recherche, ne pas redistribuer |
| GATITOS joint par l'anglais | ~4 000, surtout des mots | les deux, poids faible | CC-BY-4.0 | relire la polysémie ; éviter un français→bambara télégraphique |
| MAFAND, entraînement | 3 013 | les deux | CC-BY-4.0-NC | retirer les 24 phrases communes avec son test |
| Jeli-ASR dédoublonné | une partie | les deux, transcriptions propres pour le français→bambara | CC-BY-4.0 | dédoublonner côté cible et quasi-doublons |
| Rappel de Bayelemabaga | 20 à 50 % du mélange (heuristique) | les deux | CC-BY-SA | réduire éventuellement la part NT et Coran |

Les réglages se déduisent des recettes publiées. On part de la v1 ou de la moyenne, selon le §9a, avec Adafactor (`scale_parameter=False`, `relative_step=False`), un taux de 5e-5 et 200 à 500 pas de préchauffage ([doc Transformers](https://huggingface.co/docs/transformers/main/main_classes/optimizer_schedules), non vérifié). Le lot effectif est de 16, la longueur maximale de 128 jetons, et trois époques suffisent au départ, comme chaque étape de MAFAND. L'entraînement se fait en **fp16 avec mise à l'échelle de la perte**, car le T4 ne gère pas bf16 ([forum PyTorch](https://discuss.pytorch.org/t/current-cuda-device-does-not-support-bfloat16-please-switch-dtype-to-float16/201564)). Le checkpoint se choisit sur le dev maison, et la démo décode avec un faisceau de 4 à 5. Un jeton `<asr>` en tête des sources bruitées, à la manière de la rétro-traduction taguée, est une variante à mesurer, pas un réglage par défaut.

Côté coût, la v1 (une époque par sens sur environ 37 600 paires) a demandé environ 1 h 30 de T4 (projet). Un mélange de 20 000 à 35 000 paires sur trois époques coûterait donc **de 3 à 6 h de T4** selon que ce chiffre couvrait un sens ou les deux (estimation), soit deux sessions Colab. Il faut des checkpoints sur Drive, une corbeille vidée et une reprise vérifiée. L'incident du Whisper en octobre l'a montré : le Drive plein ne prévient pas.

Pour le mémoire, l'ablation la plus parlante ajoute les sources une à une, chacune mesurée sur le dev maison : la v1 seule, puis avec les paires de l'atelier, puis avec le Bamadaba et GATITOS, puis avec MAFAND. Le Bamadaba, la seule source non commerciale à partage identique, apparaît ainsi comme un gain chiffré et isolable. Si le temps de T4 manque, on fusionne les deux dernières étapes.

La reconnaissance n'a sa piste d'entraînement que si le §9a montre qu'elle limite la qualité. La première option est la moins chère. On entraîne un n-gramme d'ordre 6 sur les jetons de Soloni, à partir des transcriptions de Kunkado et d'AfVoices et des formules du Bamadaba et de GATITOS. Le script `train_kenlm.py` le fait sur CPU, sans GPU. On le fusionne ensuite par NGPU-LM, avec du *boosting* des salutations.

La deuxième option affine Soloni à partir de l'onglet Lire, qui fournit l'audio et le texte exact sans transcription. Il faut garder le même tokeniseur, sinon les décodeurs repartent de zéro, et remplacer le `bf16-mixed` de la recette de RobotsMali par `16-mixed` ([YAML Soloni v3](https://github.com/RobotsMali-AI/bambara-asr/blob/main/kunkado/v3-config/soloni/soloni-v3.4.0.yaml)). Les lots passent de 32 à 8 ou 16, le taux d'apprentissage doit être bien plus bas que celui de la recette, et une part de Kunkado et d'AfVoices rejouée limite l'oubli (estimation des chercheurs). Aucune durée n'est publiée pour cet affinage. La mesure est le WER sur les lectures de locuteurs tenus à l'écart, puis le chrF++ maison à traducteur fixe.

La troisième option, réentraîner Whisper avec Kunkado en plus de Jeli-ASR, coûterait environ le double des 2 h 52 de la première fois (estimation). Elle n'a de sens que si Whisper gagne au §9a.

### Étape 3 — plus tard, quand la mesure est fiable

| Action | Coût | Condition ou mesure |
|---|---|---|
| Écrire au LLACAN (Vydrin, Maslinsky, Méric) pour l'export parallèle (~27 000 phrases) et la licence du Bamadaba ; à C. Donaldson pour An ka taa | aucun | le plus gros gain en volume humain disponible |
| Traduire vers le français les transcriptions de Kunkado et d'AfVoices, filtrer, ou rétro-traduction taguée | inférence de quelques heures, puis un entraînement | uniquement si le gain se voit sur le dev maison |
| NLLB-1.3B distillé (serré sur T4), M2M-100 418M, Toucan-1.2B, entraînés sur le même mélange | un entraînement chacun, 2 à 3 fois plus long pour le 1.3B (estimation) | même jeu maison, même ablation |
| Déploiement CTranslate2 int8, faisceau 1–2 contre 4–5 | conversion en minutes | latence CPU, chrF++ maison |
| Synthèse : MMS-TTS contre les modèles MALIBA-AI et sudoping01 | inférence | note d'opinion moyenne par l'atelier (`eval/mos.py`) |
| djelia/EGAFE, sous-titres Mali Pense, SMOL et NLLB-Seed par pivot anglais→français | variable | après contrôle du recouvrement |
| Agrandir le jeu maison à huit locuteurs ou plus | temps humain | intervalles tirés par locuteur |

Pour donner un ordre d'idée de CTranslate2 : sur un Transformer-base, int8 traite 860 tokens par seconde contre 276 pour PyTorch, avec 529 Mo de mémoire contre 2 012, sans perte de BLEU ([CTranslate2](https://github.com/OpenNMT/CTranslate2)).

## Licences : tout ce plan est permis pour un mémoire, pas pour un produit

Trois signaux méritent un paragraphe du mémoire.

**Le non-commercial est partout dans la traduction.** NLLB-200 est sous CC-BY-NC 4.0 pour tous ses checkpoints, donc vos v1, v2 et V3 aussi ([README NLLB](https://github.com/facebookresearch/fairseq/blob/nllb/examples/nllb/modeling/README.md)). Les données MAFAND sont CC-BY-4.0-NC, et MMS est CC-BY-NC 4.0 ([README MMS](https://github.com/facebookresearch/fairseq/blob/main/examples/mms/README.md)). Ce dernier point est constaté pour MMS-1B-all ; il reste à confirmer pour MMS-TTS.

**Le Bamadaba est sous CC BY-NC-SA 3.0 France** (page officielle ; le dépôt GitHub, lui, n'a pas de fichier de licence). L'usage pour un mémoire est permis, avec attribution. Le « partage dans les mêmes conditions » vaut pour les œuvres dérivées : qu'un modèle entraîné dessus en soit une n'est pas tranché. Il est prudent de ne pas publier un tel modèle sans en parler aux auteurs.

**FLORES+ est réservé au test.** Sa licence est CC BY-SA 4.0, mais le zip est protégé par mot de passe et ses auteurs demandent de ne pas l'inclure dans des données d'entraînement ([flores-OLDI](https://github.com/avidale/flores-OLDI)). Les agrégats comme AfroLingu-MT ou les données minées brutes de NLLB peuvent le contenir : il faut les dédoublonner avant usage.

| Ressource | Licence | Statut | Conséquence |
|---|---|---|---|
| NLLB-200 et vos modèles affinés | CC-BY-NC 4.0 | lu | mémoire et recherche oui ; pas d'usage commercial |
| MAFAND-MT (données) | CC-BY-4.0-NC | lu | idem ; citer Adelani et al. 2022 |
| MMS | CC-BY-NC 4.0 | lu pour MMS-1B-all | remplaçable par Soloni ou Soloba (CC-BY-4.0) ou Omnilingual (Apache-2.0) |
| Bamadaba, ETRGFRA, listes de noms | CC BY-NC-SA 3.0 France (page officielle) ; aucun fichier de licence sur GitHub | page officielle lue via un moteur de recherche | mémoire oui, avec attribution ; publier un modèle : en parler aux auteurs |
| Corpus Bambara de Référence | aucune page trouvée ; textes privés | non vérifié | export à demander |
| FLORES+ | CC BY-SA 4.0, interdit à l'entraînement | lu (miroir) | test seulement |
| Bayelemabaga | CC-BY-SA-4.0 ou CC-BY-4.0 selon l'endroit | lu, contradictoire | le traiter comme à partage identique |
| Kunkado | CC-BY-SA-4.0 | lu sur la copie de fiche, non confirmé ailleurs | partage identique pour les dérivés du corpus ; effet sur des poids non tranché |
| Jeli-ASR, bam-asr-early, AfVoices, Nyana Eval | CC-BY-4.0 | lu | libre, avec attribution |
| Bam_ASR_Eval_500 | CC-BY-4.0, mais « non-commercial research » | lu, contradictoire | évaluation de recherche seulement |
| Soloni, Soloba | CC-BY-4.0 | lu | libre, avec attribution |
| Omnilingual ASR | Apache-2.0 | lu | libre |
| GATITOS | CC-BY-4.0 | lu | libre, avec attribution |
| daba | GPL v3 (code) | lu | ne couvre pas les données |
| Silero VAD | MIT | lu | libre |
| sudoping01/bambara-asr-v2 | Apache-2.0 | non vérifié | libre |
| MALIBA-AI/bambara-tts ; sudoping01/bambara-tts | licence de recherche MALIBA-AI ; CC BY-NC 4.0 | non vérifié | recherche seulement |
| M2M-100 418M | MIT selon les chercheurs | non vérifié | porte de sortie commerciale à confirmer |
| MALIBA-AI/bambara-asr-v3, Toucan, MADLAD-400, oza75/bambara-mt, SMOL | inconnue | non vérifié | lire la fiche avant usage |
| LDC2016L01 | licence LDC | non vérifié | hors de portée sans abonnement |
| An ka taa | consultation libre, sans licence de réutilisation trouvée | non vérifié | vérification manuelle ; pas d'aspiration |
| kalilouisangare/corpus-parallel-en-fr-bam | accès restreint ; bambara généré par NLLB | non vérifié | à exclure |

## Conclusion

Pour le bambara, **le choix du test décide du choix du modèle**. Les deux tests publics du projet favorisent les registres narratifs et littéraires, et donc les modèles entraînés dessus : le test de Bayelemabaga par son découpage séquentiel, celui de Jeli-ASR par son recouvrement et son registre. Le jeu maison montre déjà ce qu'ils cachent : le bout-en-bout, à moins de trois points de la cascade sur Jeli-ASR (20,9 contre 23,7), tombe à la moitié de la cascade sur la parole réelle (12,9 contre 26,7). Ce constat peut devenir une contribution du mémoire en soi : un test de compréhension construit sans écrire le bambara, qui classe les systèmes autrement que les tests publics. La régression du 16 octobre en est la première illustration.

La ressource rare n'est ni le calcul ni la taille du modèle, mais les paires du quotidien. Personne ne les publie, et l'atelier sait les produire à partir de votre voix et de celle de vos proches. D'où l'ordre du plan : des mesures gratuites, puis quelques heures de T4 bien dirigées, et seulement ensuite des modèles plus gros. Les licences ne bloquent rien pour le mémoire, mais elles dessinent l'avenir du projet. La reconnaissance (Soloni, Soloba, Omnilingual), le lexique GATITOS et vos propres données sont réutilisables librement. La traduction, elle, reste enfermée dans le non-commercial (NLLB, MAFAND) ou dans le non-licencié (Bamadaba) tant qu'un M2M-100 affiné sur des données libres n'a pas fait ses preuves.
