# Dictionnaires, lexiques et ouvrages de référence bambara–français utilisables pour la traduction automatique (bm↔fr)

> **Note de méthode (à lire d'abord).** Depuis l'environnement de recherche, seuls GitHub (pages et `raw.githubusercontent.com`) étaient accessibles directement. Les sites cormand.huma-num.fr, HAL, ACL Anthology, arXiv, Hugging Face, ankataa.com, LDC, Lexilogos étaient bloqués (refus du proxy ou échec DNS). Conséquences :
> - Tout ce qui est hébergé sur GitHub (Bamadaba, daba, corbama-build, GATITOS, Bayelemabaga) a été **téléchargé et compté directement** (comptages marqués « vérifié par téléchargement »).
> - Les chiffres venant d'autres sites proviennent de **résumés de moteur de recherche** de ces pages. Ils sont signalés « (non lu directement) » et doivent être revérifiés avant d'être cités dans un rapport final.

## 1. Bamadaba (Vydrin et al., LLACAN/INALCO) : contenu, taille, gloses, exemples, téléchargement, licence

### Takeaway
Le Bamadaba est **téléchargeable librement en format Toolbox/MDF sur GitHub** (`maslinych/bamadaba`). Il compte 13 135 entrées avec gloses françaises, environ 4 644 paires d'exemples bambara→français et 5 537 variantes orthographiques. Un fichier séparé recense 1 217 emprunts au français. Pour un étudiant, c'est la ressource lexicale la plus directement exploitable. **Aucune licence explicite** n'est attachée aux données, donc leur statut juridique reste flou.

### Cited Findings
- **Dépôt public.** Le dépôt GitHub `maslinych/bamadaba` est décrit comme « Bamadaba — Bamara-French dictionary ». On y trouve 396 commits et les fichiers `bamadaba.txt`, `bamadaba-fr-split.txt`, `bamadaba-disamb-syn.txt`, `ETRGFRA.txt`, `enciclop.txt`, `jamuw.txt`, `togow.txt`, `yorow.txt`, `bamana.gram.txt`, plus des documents (`fields_bamadaba.docx`, `bamadaba_pres.docx`) — [GitHub maslinych/bamadaba](https://github.com/maslinych/bamadaba)
- **Dernier commit : 6 décembre 2019** (« minor modifications ») ; l'en-tête du fichier indique `\ver 0.5.4`. Le dépôt GitHub n'est donc plus mis à jour depuis 2019 — [Historique des commits](https://github.com/maslinych/bamadaba/commits/master)
- **Auteurs et filiation** (README en russe). Référence : « Bailleul Ch., Davydov A.V., Erman A.V., Maslinsky K.A., Méric J.J., Vydrin V.F. *Bamadaba : Dictionnaire électronique bambara-français, avec un index français-bambara*. 2011–2014 ». Le dictionnaire est bâti sur la version 2007 du dictionnaire de Charles Bailleul (Donniya, 476 p.). En 2010–2011, il a été retravaillé : doublons supprimés, variantes et parties du discours ordonnées, gloses françaises standardisées. Il est ensuite enrichi pendant la désambiguïsation du corpus, avec comme autres sources Dumestre 2011 (Karthala) et Vydrin & Tomtchina 1999 (dictionnaire manding-russe) — [README.ru.md](https://raw.githubusercontent.com/maslinych/bamadaba/master/README.ru.md)
- **Exemples.** Le README précise que les exemples illustratifs viennent pour l'essentiel du dictionnaire de Bailleul et que **l'équipe n'en ajoute plus**. Pour d'autres exemples, il renvoie au corpus — [README.ru.md](https://raw.githubusercontent.com/maslinych/bamadaba/master/README.ru.md)
- **Deux formats sur le site du CBR.**
  - Lexique-Pro (web) : `http://cormand.huma-num.fr/lexique/lexicon/a.htm`. Cette version inclut aussi les dictionnaires de noms propres.
  - Toolbox (zip) : `http://cormand.huma-num.fr/dicos/bamadaba.zip`.

  Ces deux URL sont citées par le README ; je n'ai pas pu les ouvrir — [README.ru.md](https://raw.githubusercontent.com/maslinych/bamadaba/master/README.ru.md)
- **Format** (vérifié par téléchargement). `bamadaba.txt` (2,5 Mo) est un fichier Toolbox, en-tête `\_sh v3.0 400 MDF 4.0`, `\lang bam`. Comptage des marqueurs de champ :

  | Marqueur | Contenu | Nombre |
  |---|---|---|
  | `\lx` | entrées | 13 135 |
  | `\gf` | glose française compacte, ex. `Dieu.merci` | 13 162 |
  | `\gvf` | glose française lisible | 22 400 |
  | `\xv` | exemples bambara | 4 877 |
  | `\xf` | traductions françaises des exemples | 4 699 |
  | — | paires `\xv` immédiatement suivies d'un `\xf` | **4 644** |
  | `\va` | variantes de forme | 5 537 |
  | `\ve` | marque dialectale de la variante, ex. `(bk,bn)` | 947 |
  | `\mm`/`\mmm` | décomposition morphémique | — |
  | `\sy` | synonymes | 1 849 |

  — [bamadaba.txt](https://raw.githubusercontent.com/maslinych/bamadaba/master/bamadaba.txt)
- **Parties du discours** (vérifié) : 8 947 noms, 2 532 verbes, 658 adjectifs, 396 adverbes, 106 interjections, 66 onomatopées — [bamadaba.txt](https://raw.githubusercontent.com/maslinych/bamadaba/master/bamadaba.txt)
- **Orthographe** (vérifié). Le dictionnaire suit l'orthographe malienne standard (ɛ, ɔ, ɲ, ŋ) **avec tons marqués** sur les entrées et les exemples. Exemple : `\xv à y'á d'á mà` / `\xf il le lui a donné`. Les dates `\dt` vont de 2006 à 2019 (4 689 entrées datées de 2006, 3 676 de 2019) — [bamadaba.txt](https://raw.githubusercontent.com/maslinych/bamadaba/master/bamadaba.txt)
- **`ETRGFRA.txt` : lexique des emprunts au français** (vérifié). 1 217 entrées marquées `\smf ETRG.FRA`, avec le mot français source en glose. Exemples : `batiri` (var. `bateri`) → *batterie*, `bàto` → *bateau*, `bazi` → *base*, `bazin` (var. `bazɛn`). Il contient 1 244 phrases-exemples tirées du corpus, presque toutes **sans traduction** (127 `\xf`), certaines en ancienne orthographe (« Dènmisènnin ninnu ka yòrò bè… », *Faso kumakan*, 1987) — [ETRGFRA.txt](https://raw.githubusercontent.com/maslinych/bamadaba/master/ETRGFRA.txt)
- **Autres fichiers** (vérifié) :

  | Fichier | Contenu | Taille |
  |---|---|---|
  | `bamadaba-fr-split.txt` | variante à gloses éclatées | 12 386 `\lx`, 5 143 `\xv` |
  | `bamadaba-disamb-syn.txt` | version enrichie pour la désambiguïsation, morphèmes inclus | 23 025 `\lx`, 9 267 `\xv` |
  | `togow.txt` | prénoms | 636 |
  | `jamuw.txt` | noms de clan | 393 |
  | `yorow.txt` | toponymes | 953 |
  | `enciclop.txt` | noms encyclopédiques, glosés en français (ex. *Abarahamu* → Abraham) | 332 |

  — [dépôt bamadaba](https://github.com/maslinych/bamadaba)
- **Licence.** Le dépôt ne contient **aucun fichier LICENSE** (`LICENSE` et `LICENSE.txt` renvoient 404) et le README ne mentionne aucune licence. Le logiciel daba, lui, est sous **GPL v3** (« (c) 2010-2011 Kirill Maslinsky … GNU General Public License … version 3 »), mais cela ne couvre pas les données du dictionnaire — [daba LICENSE.txt](https://raw.githubusercontent.com/maslinych/daba/master/LICENSE.txt)
- **Correcteur orthographique dérivé.** Un correcteur Hunspell/LibreOffice dérivé du Bamadaba est diffusé par Jean-Jacques Méric (mali-pense.net). Selon les résultats de recherche, il compte 24 651 mots sous licence de type BSD, et il est crédité « BAMADABA … Valentin Vydrin, INALCO, LLACAN – 2010-2022 » (non lu directement) — [LibreOffice extension](https://extensions.libreoffice.org/en/extensions/show/21654) ; [OpenOffice extension](https://extensions.openoffice.org/fr/node/17698)
- **Publication.** Depuis janvier 2014, le Bamadaba est publié au format Lexique-Pro sur le site du Corpus Bambara de Référence, avec mises à jour régulières. Il a été construit sur le dictionnaire Bailleul 2007, remanié et uniformisé (article TALAf 2014, non lu directement) — [Paper TALAf-O.2 (HAL)](https://halshs.archives-ouvertes.fr/halshs-01134070v1/file/Paper_TALAf-O.2.pdf)
- **Usage.** Un article de 2017 décrit le Bamadaba comme librement accessible sur `cormand.huma-num.fr/bamadaba.html`, utilisé par les étudiants et mis à jour en continu (non lu directement) — [Vydrin 2017, HAL](https://halshs.archives-ouvertes.fr/halshs-01714522v1/file/2017vydrin_corpus_driven_dict.pdf)
- **À ne pas confondre.** Le *Manding-English Dictionary* de Vydrin (MeaBooks, 2015, vol. 1, maninka-bamana) est un ouvrage imprimé distinct, glosé en anglais — [African Books Collective](https://www.africanbookscollective.com/books/manding-english-dictionary.-maninka-bamana-vol.-1)

### Inferences
- **Données parallèles supplémentaires.** Les ~4 644 paires `\xv`/`\xf` de `bamadaba.txt` donnent environ 4,6 k paires bm→fr courtes, de style « phrase de dictionnaire » (souvent orales et quotidiennes : « Papa, merci ! », « remercie-le pour son cadeau ! »). On peut ajouter environ 13 k paires mot→glose (`\lx` → `\gvf`) en format « mot isolé ». Les sources de Bayelemabaga ne listent aucun dictionnaire (voir section 2), donc ces exemples ne sont probablement pas déjà dans les données d'entraînement de l'étudiant.
- **Pré-traitement indispensable :**
  1. retirer les tons avec `daba.orthography.detone()`, car Bayelemabaga est sans tons ;
  2. couper les exemples qui contiennent `=` ou `...` ;
  3. pour la direction fr→bm, ne pas sur-pondérer les entrées mot-à-mot (risque de sorties télégraphiques).
- **Normalisation.** Les champs `\va` (5 537 variantes) donnent une table « variante → forme standard » prête à l'emploi. On peut l'appliquer à la sortie de l'ASR avant la traduction.
- **Code-switching.** `ETRGFRA.txt` sert à reconnaître les emprunts francisés dans la transcription (ex. *batiri* ↔ batterie) et à les recopier ou les « re-franciser » en sortie.
- **Juridique.** En l'absence de licence, l'usage pour entraîner un modèle de recherche non commercial est une zone grise. Il est prudent de ne pas redistribuer le fichier ou un modèle entraîné dessus sans l'accord de V. Vydrin ou K. Maslinsky (LLACAN). Une demande par courriel est conseillée.

### Gaps
- Je n'ai pas pu vérifier si `http://cormand.huma-num.fr/dicos/bamadaba.zip` est encore en ligne ni s'il est plus récent que la version GitHub de 2019 (site injoignable).
- La licence officielle des données Bamadaba est introuvable. Le chiffre « 24 651 mots, BSD » du correcteur vient d'un résumé de recherche, non d'une lecture directe.
- Les champs de `fields_bamadaba.docx` n'ont pas été lus.

## 2. Corpus Bambara de Référence (CBR / Corbama) : taille, sous-corpus glosé, téléchargement, licence, textes parallèles bm–fr

### Takeaway
Le CBR contient un **sous-corpus parallèle bambara–français aligné phrase à phrase** d'environ 68 k phrases. Bayelemabaga (déjà utilisé par l'étudiant) en est **extrait**. D'après la liste des documents alignés, environ 27 k phrases bambara proviendraient de documents absents de la liste de sources de Bayelemabaga v1.0.1. Les textes eux-mêmes ne sont **pas téléchargeables** : le dépôt source est privé et l'accès se fait par l'interface de concordance NoSketchEngine. Aucune licence publique n'a été trouvée.

### Cited Findings
- **Structure** (README de construction). Corbama a deux sous-corpus : le sous-corpus désambiguïsé manuellement (`corbama-net*`, ≈ 0,35 M mots) et le corpus complet (`corbama-brut`, `corbama-nul`, ≈ 2,1 M mots). Variantes : `net-non-tonal` (sans tons, comme les sources), `net-tonal` (tons ajoutés automatiquement), `brut` et `nul` (orthographe simplifiée, voyelles ouvertes remplacées par o, e) — [corbama-build README](https://raw.githubusercontent.com/maslinych/corbama-build/master/README.md)
- **Annotation** (même README). Champs : `word` (forme normalisée en nouvelle orthographe), `lemma`, `tag`, **`gloss` (glose française ou standardisée)**, `parts`, **`original` (forme d'origine non normalisée)**, `tonal`, `polisemy` — [corbama-build README](https://raw.githubusercontent.com/maslinych/corbama-build/master/README.md)
- **Accès aux textes.** Les textes sont dans « `corbama` — a corpus repository (private, not shown) » ; le lexique vient du dépôt public `bamadaba` — [corbama-build README](https://raw.githubusercontent.com/maslinych/corbama-build/master/README.md)
- **Alignements publics** (vérifié par téléchargement). `corbama-bam-fra.prl` et `corbama-fra-bam.prl` donnent les alignements de **569 documents**. En développant les plages, j'ai compté **67 776 phrases bambara** et **66 715 phrases françaises**. Les documents incluent :
  - les cours *An ka bamanankan kalan* (Bird, Hutchison, Kanté) ;
  - *Bamanankan maben* ;
  - des chansons et des contes (Diarra *nsiirinw*) ;
  - *Dɔgɔtɔrɔ tɛ sigida min na* ;
  - les textes de Bailleul et Dumestre.

  Le dernier commit du dépôt date du 23 juillet 2025 ; le dernier commit qui mentionne `corbama-bam-fra.prl` date du 26 mai 2023 — [corbama-bam-fra.prl](https://raw.githubusercontent.com/maslinych/corbama-build/master/corbama-bam-fra.prl) ; [commits](https://github.com/maslinych/corbama-build/commits/master)
- **Lien avec Bayelemabaga.** Bayelemabaga est « a collection of 44562 aligned machine translation ready Bambara-French lines, originating from Corpus Bambara de Reference… 264 text files ». Le tableau d'instantané du README donne 46 976 lignes, sous licence **CC-BY-SA-4.0** — [RobotsMali-AI/datasets README](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/README.md). L'article NAACL 2025 indique « 47K Bambara-French parallel sentences curated from 231 data sources » (231 contre 264 sources selon la version) — [Bayelemabaga, NAACL 2025](https://aclanthology.org/2025.naacl-long.602/)
- **Recoupement fait ici.** J'ai comparé les 569 noms de documents de `corbama-bam-fra.prl` aux 264 entrées de [bayelemabaga/sources.txt](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/bayelemabaga/sources.txt) :
  - 204 correspondent exactement ;
  - les autres, hors chapitres de Bird/Hutchison/Kanté (listés globalement dans Bayelemabaga), totalisent ≈ 26 977 phrases bambara.

  Cet appariement est approximatif, car il repose sur les noms de fichiers.
- **Taille** (résumé du site, non lu directement). Une mise à jour de **juillet 2025** donnerait ≈ 11,8 M mots au total, dont ≈ 2,1 M désambiguïsés. Corpus parallèle : **864 406 mots (69 286 phrases) côté bambara et 890 286 mots (66 606 phrases) côté français** — [cormand.huma-num.fr](https://cormand.huma-num.fr/) ; [Anciennes mises à jour](http://cormand.huma-num.fr/majs.html). Ce chiffre est **contradictoire** avec un autre résumé du même site, qui ne trouvait rien après mai 2023 et donnait pour octobre 2021 225 273 mots bm (17 964 phrases) et 240 829 mots fr. Mon comptage des alignements (67 776 phrases bm) appuie l'ordre de grandeur de 2025.
- **Croissance du parallèle** (résumé, non lu directement). Taille initiale : 99 163 mots bm et 116 177 mots fr. Une mise à jour de mai 2023 mentionne un financement **Lacuna Fund** ayant permis d'agrandir fortement le corpus parallèle, et un « sous-corpus parallèle ajusté ». La recherche se fait via deux interfaces : *corbamafara* (requête par formes bambara) et *corfarabama* (requête par formes françaises) — [Guide d'utilisation](http://cormand.huma-num.fr/guide.html) ; [majs.html](http://cormand.huma-num.fr/majs.html)
- **Taille en 2017.** Vydrin (2017) indique 3 846 094 mots en mars 2017, dont ≈ 700 000 désambiguïsés (non lu directement) — [Vydrin 2017, HAL](https://halshs.archives-ouvertes.fr/halshs-01584112v1/file/2017vydrin_electronic_library.pdf)
- **Origine.** Le corpus est en ligne depuis avril 2012. Il est décrit comme « probably the only one freely accessible on the Internet » pour l'Afrique subsaharienne à plus d'un million de mots (non lu directement) — [Bamana Reference Corpus (BRC), HAL](https://halshs.archives-ouvertes.fr/halshs-00908108)
- **Bibliographie bilingue.** Le site héberge une bibliographie des publications bilingues (« BAMBARABIBLIO – Publications bilingues »), utile pour repérer des livres bm–fr à aligner soi-même (non lu directement) — [pubilingues.jsp](https://cormand.huma-num.fr/biblio/pubilingues.jsp)

### Inferences
- Pour l'étudiant, le gain potentiel du CBR **au-delà de Bayelemabaga** est la partie parallèle ajoutée depuis 2022 (≈ 20–27 k phrases selon mon recoupement). Elle n'est accessible qu'en demandant un export à l'équipe (Vydrin, Maslinsky, Méric) ou une version plus récente de Bayelemabaga ; elle ne se télécharge pas directement.
- Les cours *An ka bamanankan kalan* et *Bamanankan maben* présents dans le parallèle sont plus proches de la langue quotidienne (dialogues, salutations) que la presse. Ce sont des cibles prioritaires si l'on peut les obtenir.
- Le champ `original` (forme non normalisée) face à `word` (forme normalisée) est en principe une **table de normalisation orthographique dérivée du corpus réel**, mais seulement si l'on obtient les fichiers `.vert`.

### Gaps
- Aucune page de licence ou de conditions d'utilisation du CBR n'a été trouvée ; le site était injoignable.
- Il n'est pas confirmé que l'interface NoSketchEngine permette d'exporter des concordances parallèles en masse.
- Les chiffres de 2025 (11,8 M mots ; 69 286 phrases parallèles) viennent d'un résumé de moteur de recherche, contredit par un autre résumé. À revérifier sur la page d'accueil du site.

## 3. Autres dictionnaires et lexiques (Bailleul, Dumestre, Kone, An ka taa, LDC, SIL, Wiktionnaire, Glosbe, PanLex, Tatoeba) : numérisation et usage légal

### Takeaway
En dehors du Bamadaba et de GATITOS, **aucun grand dictionnaire bambara–français n'est disponible en données ouvertes et réutilisables**. Bailleul et Dumestre sont des livres imprimés sous droits, en partie absorbés dans le Bamadaba. An ka taa est consultable gratuitement en ligne, sans licence de réutilisation trouvée. Le lexique LDC (Bamba, 5 978 entrées, bm–fr/en) est sous licence LDC. Les lexiques SIL et Rigdon sont petits.

### Cited Findings
- **Bailleul, dictionnaires Donniya.** *Dictionnaire bambara-français*, Bamako, Donniya, 1996, 480 p. ([Glottolog](https://glottolog.org/resource/reference/id/140867)). *Dictionnaire français-bambara*, Donniya, 1998, 389 p. ([Glottolog](https://glottolog.org/resource/reference/id/140870)). 3ᵉ édition de 2007 chez Donniya ([Webonary, avant-propos dioula](https://www.webonary.org/dioula-bf/overview/foreword/?lang=en)). Le dictionnaire marque intégralement les tons et sépare les sens avec des exemples grammaticaux (blog An ka taa, non lu directement) — [An ka taa, « Print Dictionaries »](https://www.ankataa.com/blog/tag/dictionaries). L'édition 2007 est la base du Bamadaba (voir section 1) — [README Bamadaba](https://raw.githubusercontent.com/maslinych/bamadaba/master/README.ru.md)
- **Dumestre, *Dictionnaire bambara-français suivi d'un index abrégé français-bambara*,** Paris, Karthala, 2011 (notice de catalogue, non lue directement). Le blog An ka taa le présente comme l'œuvre finale d'un chercheur de l'INALCO, de nature distincte d'un dictionnaire classique comme celui de Bailleul — [An ka taa, tag dictionaries](https://www.ankataa.com/blog/tag/dictionaries). C'est une source d'enrichissement du Bamadaba — [README Bamadaba](https://raw.githubusercontent.com/maslinych/bamadaba/master/README.ru.md)
- **Kassim Kone.** *Bamanankan Daɲɛgafe* est un dictionnaire **monolingue** bambara–bambara (Mother Tongue Editions, 1ʳᵉ éd. 1994, 2ᵉ éd. 2009/2010). Il existe aussi un recueil de 1 177 proverbes bamana-maninka-anglais (1995). Un PDF « Bambara_005-Danyegafe » est hébergé par l'African Language Materials Archive (MSU) (non lu directement) — [ALMA PDF](https://alma.matrix.msu.edu/assets/uploads/2020/01/Bambara_005-Danyegafe.pdf) ; [page Kassim Kone, SUNY Cortland](https://www2.cortland.edu/departments/sociology/kassim-kone) ; [Mother Tongue Editions](https://alma.matrix.msu.edu/assets/uploads/2009/04/mother_tongue_editions.pdf)
- **An ka taa** (Coleman Donaldson et Antoine Fenayon). Dictionnaire en ligne gratuit bambara/dioula–anglais avec index inverses anglais et **français**. Il marque les tons, donne des exemples, des notes bambara/jula, l'étymologie et les emprunts. Lancé en mars 2019, il est présenté comme un travail en cours (non lu directement) — [dictionary.ankataa.com](https://dictionary.ankataa.com/lexicon.php) ; [annonce 2019](https://www.ankataa.com/blog/2019/3/12/bambara-english-dictionary-released)
- **Kalan (An ka taa).** Cours en ligne avec une leçon « Essential expressions » (salutations, deux formes de « oui ») — [kalan.ankataa.com](https://kalan.ankataa.com/bambara-beg/grammar-i/essential-expressions)
- **LDC « Bamanankan Lexicon » (LDC2016L01).** Auteur Moussa Bamba, publié le 15 décembre 2016. **5 978 entrées**, en deux lexiques bamanankan–anglais et bamanankan–**français** ; troisième volume d'une série mandingue après Mawukakan (2005) et Maninkakan (2013) (non lu directement) — [LDC catalog](https://catalog.ldc.upenn.edu/LDC2016L01)
- **SIL Bamako.** Richard Marguets (2010), *Lexique Bambara-français-anglais*, SIL Bamako, **3 512 entrées**, version numérique annoncée sur bambara.org (non lu directement). Une page Wiktionnaire attribuerait le même lexique à Richard Nci Diarra : auteur non résolu — [Webonary, avant-propos](https://www.webonary.org/dioula-bf/overview/foreword/?lang=en)
- **Rigdon (2016),** *Bambara / French / English Dictionary* (Words R Us). 8 000 mots anglais et français pour plus de 5 600 termes bambara ; livre imprimé — [Cambridge Language Centre](https://www.langcen.cam.ac.uk/resources/langb/data/ba_110.html)
- **Scan Archive.org « DictionnaireBamabaraFrancais ».** Numérisation Google d'un dictionnaire ancien, présenté comme hors droits. L'auteur n'est pas identifié ; candidats : Bazin 1906 ([Glottolog](https://glottolog.org/resource/reference/id/10018)) ou Molin 1955 ([Glottolog](https://glottolog.org/resource/reference/id/98123)) — [archive.org texte](https://archive.org/stream/DictionnaireBamabaraFrancais/bambara_djvu.txt)
- **Wiktionnaire francophone.** Une catégorie « bambara » (code bm) existe, avec des sous-catégories d'exemples, d'expressions, de proverbes et d'onomatopées (non lu directement) — [fr.wiktionary Catégorie:bambara](https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:bambara)
- **Données à éviter comme référence.** Un jeu Hugging Face « corpus-parallel-en-fr-bam » a un côté bambara **généré par machine, non vérifié** (non lu directement) — [HF kalilouisangare](https://huggingface.co/datasets/kalilouisangare/corpus-parallel-en-fr-bam)

### Inferences
- **Utilisable légalement et gratuitement :**
  - GATITOS (CC-BY-4.0, section 4) ;
  - Bayelemabaga (CC-BY-SA-4.0) ;
  - le correcteur Bamadaba (BSD selon la page d'extension) ;
  - le Bamadaba GitHub (accessible, mais sans licence explicite) ;
  - le Wiktionnaire (CC-BY-SA, à confirmer).
- **Imprimé ou sous droits :** Bailleul et Dumestre (Karthala). Pas de numérisation légale trouvée. Leur contenu est déjà partiellement dans le Bamadaba.
- **Consultable mais non téléchargeable :** An ka taa. Un *scraping* massif sans accord serait contraire à l'esprit du projet ; mieux vaut écrire à C. Donaldson.
- **Payant ou sous licence :** LDC2016L01 (tarif non vérifié).

### Gaps
- **Glosbe :** taille du bm–fr et conditions d'utilisation non vérifiées (site non consulté).
- **PanLex :** nombre d'expressions bambara non vérifié. Le seul repère est que GATITOS donne environ 5 fois plus de gain chrF que PanLex dans les expériences de Google (section 7).
- **Tatoeba :** nombre de phrases bambara (code `bam`) non vérifié.
- Le nombre d'entrées des dictionnaires Bailleul 2007 et Dumestre 2011 n'a pas été trouvé ; le nombre de pages de Dumestre 2011 non plus.
- L'existence d'un dictionnaire bilingue bambara–français ou bambara–anglais de Kassim Kone n'a pas été confirmée : seul le dictionnaire monolingue est attesté.
- Les mentions « Mandenkan/Kalan resources » n'ont rien donné de plus précis. La revue *Mandenkan* (OpenEdition) publie des études lexicales, mais pas de lexique téléchargeable identifié.

## 4. Google GATITOS : inclut-il le bambara, et avec quelle langue pivot ?

### Takeaway
**Oui.** GATITOS contient `en_bm.tsv` : 4 000 segments anglais traduits en bambara (orthographe standard ɛ/ɔ/ɲ, presque sans tons), sous **CC-BY-4.0**, avec l'**anglais comme pivot**. Le français étant aussi couvert (`en_fr.tsv`), on obtient par jointure sur la clé anglaise environ **4 000 paires lexicales bm–fr**, avec des salutations courantes.

### Cited Findings
- **Description** (README, vérifié). GATITOS = « 4,000 English segments (4,500 tokens) … translated into each of 173 languages », traduits « directly from English ». Il comprend 93 % de mots isolés et doit être « best thought of as a multilingual lexicon ». Le tableau des langues contient `bm | Bambara | Bámánánkán | Latn` et `bm-Nkoo` (N'Ko). Licence : « This data is released with the `CC-BY-4.0` license » — [README GATITOS (commit 3622039)](https://raw.githubusercontent.com/google-research/url-nlp/3622039cf51f7eeffa58b957332a8e8c337981d7/gatitos/README.md)
- **Déménagement.** GATITOS a été déplacé sur Hugging Face (« has moved to Huggingface, to join its sister dataset SMOL ») ; le dépôt GitHub est archivé — [README GATITOS actuel](https://raw.githubusercontent.com/google-research/url-nlp/main/gatitos/README.md) ; [HF google/smol](https://huggingface.co/datasets/google/smol)
- **Contenu** (vérifié par téléchargement). `en_bm.tsv` contient 4 000 lignes. `en_fr.tsv` contient 3 999 clés anglaises, dont 3 998 communes avec `en_bm`. Caractères dominants : ɛ (1 724), ɔ (1 206), ɲ (266). Quelques diacritiques tonals épars (77 accents graves, 63 aigus). — [en_bm.tsv](https://raw.githubusercontent.com/google-research/url-nlp/3622039cf51f7eeffa58b957332a8e8c337981d7/gatitos/en_bm.tsv) ; [en_fr.tsv](https://raw.githubusercontent.com/google-research/url-nlp/3622039cf51f7eeffa58b957332a8e8c337981d7/gatitos/en_fr.tsv)
- **Exemples de salutations** extraits du fichier :

  | Anglais | Bambara | Français (jointure) |
  |---|---|---|
  | thank you | a' ni cɛ | merci |
  | good morning | a' ni sɔgɔma | bonjour |
  | how are you | i ka kɛnɛ wa | comment allez-vous |
  | Hello | aw ni baara | Bonjour |
  | welcome | i danse | accueillir |
  | good evening | i ni su | — |
  | goodbye | kan bɛn | — |
  | Sorry | Hakɛto | — |
  | thanks | barika | — |
  | Yes | Awɔ | — |
  | No | Ayi | — |
  | money | wari | — |
  | friend | teri | — |

  — [en_bm.tsv](https://raw.githubusercontent.com/google-research/url-nlp/3622039cf51f7eeffa58b957332a8e8c337981d7/gatitos/en_bm.tsv)

### Inferences
- La jointure en→bm / en→fr donne un lexique bm–fr gratuit et légalement réutilisable (attribution exigée). Il faut contrôler la polysémie à la main. Exemple : *aw ni baara* est la salutation adressée à des gens au travail, pas un « bonjour » générique ; *welcome* → « accueillir » est un verbe, alors que *i danse* est une formule de bienvenue.
- C'est la ressource idéale pour l'augmentation lexicale à la BiLex Rx (code-switching de mots dans les phrases sources, ou « lexical prompting », section 7).

### Gaps
- Je n'ai pas pu vérifier la version Hugging Face actuelle (`google/smol`, configuration `gatitos`), qui peut différer de la version GitHub de 2024. Je n'ai pas non plus vérifié si SMOL (SmolSent/SmolDoc) contient du bambara.

## 5. Salutations, formules de politesse, argot et bambara urbain (jeunes de Bamako, alternance avec le français)

### Takeaway
**Aucun lexique publié d'argot urbain bamakois** (« wai », « yoh », etc.) n'a été trouvé. Le plus utile est ce qui existe déjà dans d'autres ressources :
- les **106 interjections** du Bamadaba, dont *yó/yóo* « salut », *wáyi* « aïe », *wàláyì* « je le jure », *àbarika* « merci », *màrahaba*, *nbá/nsé* ;
- son **lexique de 1 217 emprunts au français** ;
- les salutations de GATITOS ;
- quelques études sociolinguistiques de l'alternance bambara–français.

### Cited Findings
- **Interjections du Bamadaba** (vérifié). Entrées `\ps intj` avec leur glose `\gf` :

  | Bambara | Glose |
  |---|---|
  | àbarika | merci |
  | àyí | non |
  | áyìwa | bon |
  | bìsímila | au nom de Dieu ! |
  | ìhí | hein |
  | íyo | oui |
  | màrahaba | salut |
  | nbá | salut |
  | nsé | salut |
  | nàamù | oui |
  | ɔ̀wɔ́ | oui |
  | ǹǹń | oui |
  | sáhà | merci |
  | sálamalekun | salutation |
  | sànbɛsànbɛ | bonne année |
  | wàláyì | je le jure |
  | wáyi | aïe |
  | yó (var. yóo) | salut — « interjection vocative exprimant l'insistance » |
  | hɛ̀n | hein |
  | jàa | en fait |
  | tɛ́sadì | n'est-ce pas |

  Le dictionnaire contient aussi les entrées *hɛ́rɛ* et *tɔ́ɔrɔ*, utilisées dans les formules de salutation — [bamadaba.txt](https://raw.githubusercontent.com/maslinych/bamadaba/master/bamadaba.txt)
- **Exemples de politesse dans le Bamadaba** (vérifié) : « Papa, merci ! », « merci à Dieu ! », « faire les salutations de départ », « toi et quelques jours ! (réponse à une salutation) », « le bonheur soit avec toi ! – merci » — [bamadaba.txt](https://raw.githubusercontent.com/maslinych/bamadaba/master/bamadaba.txt)
- **Emprunts au français.** 1 217 entrées étiquetées ETRG.FRA avec le mot français d'origine (voir section 1) — [ETRGFRA.txt](https://raw.githubusercontent.com/maslinych/bamadaba/master/ETRGFRA.txt)
- **Écrit mêlé.** Mbodj-Pouye et Van den Avenne (2007), « C'est bambara et français mélangés », *Langage et société*. Analyse de cahiers villageois maliens, avec des alternances de syntagmes et des emprunts, souvent quand le mot français est mal connu (non lu directement) — [Cairn](https://www.cairn.info/revue-langage-et-societe-2007-2-page-99.htm)
- **Emprunts syntaxiques.** Dumestre, « Emprunts syntaxiques du bambara au français : un premier repérage », dans *Language in Africa*. Corpus oral et écrit, dont des productions de journalistes bamakois (non lu directement) — [PDF iling-ran.ru](https://iling-ran.ru/library/languageinafrica/1/LiA_4_1_Dumestre.pdf)
- **Influence lexicale.** « French lexical influence in Bamana », IULC Working Papers (non lu directement) — [IU ScholarWorks](https://scholarworks.iu.edu/journals/index.php/iulcwp/article/view/25990)
- **Travaux universitaires** (non lus directement) : une thèse de l'Université d'Orléans (2022) comporte une section sur l'alternance codique bamanankan–français ([PDF](https://theses.univ-orleans.fr/public/2022ORLE1025_va.pdf)) ; un mémoire algérien porte sur le *code-switching* bambara–français dans la conversation quotidienne ([PDF](https://bucket.theses-algerie.com/files/repositories-dz/6653301544129487.pdf)).
- **Forum An ka taa.** Il héberge un fil sur les abréviations SMS, l'argot et l'orthographe (janvier 2025, ~210 vues), dont le contenu n'a pas pu être lu — [ankataa.discourse.group, catégorie Language](https://ankataa.discourse.group/c/language/16)

### Inferences
- Pour « wai », « yoh » et les marqueurs de discours urbains, il faudra **constituer son propre mini-lexique**, par exemple 200 à 500 paires écrites ou dictées par le développeur, qui est locuteur natif. Les entrées existantes du Bamadaba peuvent servir de point de départ. Exemples : *yó/yóo* (vocatif insistant, « salut »), *wáyi* (« aïe », à distinguer probablement de « wai » urbain), *wàláyì* (« walahi », « je le jure »).
- On peut ensuite fabriquer des phrases synthétiques par gabarits (salutation + question + réponse) pour le fine-tuning. Il faut alors les garder dans un jeu de validation séparé pour mesurer le gain réel.

### Gaps
- Aucune étude ni aucun lexique publié sur l'argot des jeunes de Bamako n'a été trouvé (pas de dictionnaire « bambara urbain », pas d'équivalent du nouchi ivoirien documenté pour Bamako dans les résultats).
- Le sens exact de « wai » et « yoh » en parler urbain n'est confirmé par aucune source lexicographique.

## 6. Orthographe : norme malienne (ɛ, ɔ, ɲ, ŋ) contre graphies anciennes ou informelles (e, o, ny, è) ; tables de normalisation publiées

### Takeaway
Il existe des **tables de conversion publiées dans le code de daba** (GPL v3) : è→ɛ, ò→ɔ, èe→ɛɛ, òo→ɔɔ, ny→ɲ, ng→ŋ (ces deux dernières sont ambiguës), ainsi qu'un normaliseur d'apostrophes et une fonction de suppression des tons. Le Bamadaba fournit 5 537 variantes (`\va`), et Vydrin (2022) a publié un projet de **dictionnaire orthographique** standardisé à partir du corpus.

### Cited Findings
- **Convertisseur `bamlatinold`** de daba, décrit comme « Convertor from old latin Bambara orthography (ambiguous) ». Table : `{'è':['ɛ'], 'ò':['ɔ'], 'èe':['ɛɛ'], 'òo':['ɔɔ'], 'ng':['ng','ŋ'], 'ny':['ny','ɲ']}`. Il produit toutes les candidates ; une forme ambiguë doit être départagée par le lexique — [bamlatinold.py](https://raw.githubusercontent.com/maslinych/daba/master/daba/plugins/bamlatinold.py)
- **Autres plugins de daba** (vérifié) :
  - `apostrophe.py` remplace l'apostrophe Unicode U+2019 par l'apostrophe ASCII ;
  - `bailleul.py` « Convert Bailleul's tonal orthography into corbama standard » ;
  - `vydrine.py` et `nko.py` font d'autres conversions ;
  - `daba/orthography.py` contient `detone()`, qui supprime les tons.

  — [répertoire plugins](https://github.com/maslinych/daba/tree/master/daba/plugins) ; [orthography.py](https://raw.githubusercontent.com/maslinych/daba/master/daba/orthography.py) ; [apostrophe.py](https://raw.githubusercontent.com/maslinych/daba/master/daba/plugins/apostrophe.py)
- **Variantes du Bamadaba** (vérifié) : 5 537 `\va`. Exemple : *bɛ́ɛn* a pour variantes *bɛ́rɛn, béren, bélen, bɛ́lɛn* ; *bɛ̀ɛrɛbɛɛrɛ* a pour variante *bèrebere*. Elles couvrent les graphies avec e/o au lieu de ɛ/ɔ et les variantes dialectales — [bamadaba.txt](https://raw.githubusercontent.com/maslinych/bamadaba/master/bamadaba.txt)
- **Variante `corbama-nul` du corpus** : « simplified orthography (open vowels replaced with their closed counterparts o,e) ». Le corpus garde aussi la forme originale (`original`) à côté de la forme normalisée (`word`) — [corbama-build README](https://raw.githubusercontent.com/maslinych/corbama-build/master/README.md)
- **Vydrin (2022), « Vers un dictionnaire orthographique bambara »,** *Mandenkan* 68, p. 59–82. Méthode : confronter les variantes du Bamadaba aux fréquences du corpus, fixer des formes standard, puis faire valider la liste par un groupe d'experts d'institutions maliennes ; les règles générales d'orthographe sont aussi révisées (non lu directement) — [DOAJ, Mandenkan déc. 2022](https://doaj.org/article/a16a2eb5ac564fc490eb276331c63a4d)
- **Correcteur LibreOffice v2.6.** Il intégrerait les conclusions d'un atelier tenu à Bamako en mai–juin 2022 sur le dictionnaire orthographique (non lu directement) — [LibreOffice extension](https://extensions.libreoffice.org/en/extensions/show/21654)
- **Débat sur la norme.** Une note de Dumestre (*Mandenkan*, 2014) discuterait les propositions orthographiques de Konta et Vydrin, notamment le choix de ɔ, ɛ, ɲ plutôt que des digrammes ou des lettres accentuées (non lu directement, attribution à confirmer) — [DOAJ, Mandenkan déc. 2014](https://doaj.org/article/8c7e50f0d627452b9c28a23fd360f491)
- **Faible standardisation de l'écrit.** Le bambara se caractérise par un faible niveau de standardisation écrite, d'où l'attention portée aux variantes dans l'annotation du corpus (non lu directement) — [TALN 2014](https://talnarchives.atala.org/ateliers/2014/TALAf/3.pdf)
- **Graphies non standard dans Bayelemabaga.** Le README signale que « ʃ » (son sh) est laissé dans le jeu de données alors qu'il n'appartient pas à l'orthographe bambara — [RobotsMali-AI/datasets README](https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/README.md)

### Inferences
- **Chaîne de normalisation recommandée** pour les données d'entraînement et la sortie de l'ASR :
  1. NFC + apostrophes (plugin `apostrophe`) ;
  2. suppression des tons (`detone`) ;
  3. è→ɛ et ò→ɔ (règle sûre) ;
  4. ny/ng → ɲ/ŋ **uniquement si la forme convertie figure dans le lexique Bamadaba ou dans le dictionnaire Hunspell**, la règle étant ambiguë ;
  5. table `\va` → `\lx` (après suppression des tons) pour les variantes lexicales.
- Les graphies informelles e/o, utilisées à la place de ɛ/ɔ par les gens qui ne savent pas écrire le bambara, sont **non réversibles par règle**. Il faut une recherche dans le lexique, par exemple en faisant correspondre la forme « e/o » à une forme « ɛ/ɔ » grâce à une clé de comparaison où ɛ→e et ɔ→o (c'est l'idée de `corbama-nul`).

### Gaps
- Le texte intégral de Vydrin 2022 (liste des variantes retenues) n'a pas pu être lu ; je n'ai pas vérifié s'il existe une table téléchargeable issue de l'atelier de 2022.
- Aucune table publiée ne couvre les graphies SMS ou réseaux sociaux (abréviations, « è » pour ɛ, « dj » pour j, etc.).

## 7. Travaux publiés ayant utilisé des dictionnaires pour améliorer la TA bambara (et effet mesuré)

### Takeaway
Le seul cas documenté pour le bambara est **Tapo et al. (2020)**, dont le jeu de données initial (~2 146 paires bm–fr) venait des **phrases-exemples d'un dictionnaire SIL Mali**. Aucune étude n'a mesuré l'apport des exemples du Bamadaba ajoutés à Bayelemabaga. Au niveau multilingue, Google (BiLex Rx / GATITOS) montre que l'augmentation lexicale par lexique curé améliore nettement le chrF, environ 5 fois plus qu'avec PanLex.

### Cited Findings
- **Tapo et al. (2020)**, « Neural Machine Translation for Extremely Low-Resource African Languages: A Case Study on Bambara », LoResMT 2020 : 2 146 phrases parallèles bm–fr et 2 158 bm–en. Le jeu de données le plus prometteur venait d'un dictionnaire de **SIL Mali**, avec des phrases d'exemple en bambara, français, anglais et espagnol (non lu directement) — [ACL Anthology](https://aclanthology.org/2020.loresmt-1.3/) ; [arXiv 2011.05284](https://arxiv.org/abs/2011.05284v1)
- **Thèse RIT associée** : « a BLEU score of 17.5. The results confirm that MT for Bambara, despite our small data set, is viable ». La direction de traduction n'a pas pu être vérifiée (non lu directement) — [RIT thesis](https://repository.rit.edu/theses/10584)
- **Tapo et al. (2021)**, « Domain-specific MT for Low-resource Languages: The case of Bambara–French » : premier jeu parallèle spécialisé ; l'alignement a représenté environ 90 % des heures de travail (non lu directement) — [arXiv 2104.00041](https://arxiv.org/abs/2104.00041)
- **Bayelemabaga (NAACL 2025)** : 47 k paires, 231 sources. Le fine-tuning de 7 modèles donne des gains allant jusqu'à +4,5 BLEU, +11,4 chrF++ et +0,27 AfriCOMET ; les modèles encodeur-décodeur restent les meilleurs. Un résumé de recherche indique que des phrases du dictionnaire SIL font partie des sources, ce que je n'ai pas vérifié — [ACL Anthology](https://aclanthology.org/2025.naacl-long.602/)
- **Bonfanti et al. (2025)**, comparaison de pipelines fr→bm : un transformer simple fait mieux que LLaMA3 fine-tuné et que la distillation LaBSE. Résultats : 10 % BLEU et 21 % chrF sur Bayelemabaga ; 33,81 BLEU et 41 chrF sur le jeu Yiri. Aucun usage de dictionnaire n'a été mentionné (non lu directement) — [arXiv 2509.12514](https://arxiv.org/html/2509.12514v1)
- **BiLex Rx (Google, 2023)** : « lexica like GATITOS show about 5x larger CHRF improvement than larger, noisier lexica like Panlex ». Les techniques d'augmentation lexicale (*codeswitching*, *lexical prompting*) donnent des gains similaires et cumulables, dans des modèles de 200+ langues, en traduction non supervisée (non lu directement) — [arXiv 2303.15265](https://arxiv.org/abs/2303.15265v1) ; [Google Research](https://research.google/pubs/bilex-rx-lexical-data-augmentation-for-massively-multilingual-machine-translation/)
- **GATITOS (EMNLP 2023)** : les expériences ont d'abord porté sur 24 langues avant l'extension à 170 ; on ignore si le bambara était parmi ces 24 (non lu directement) — [GATITOS EMNLP 2023](https://preview.aclanthology.org/setup/2023.emnlp-main.26)
- **Analyseur daba (Maslinsky 2014).** Il produit un texte bambara annoté (POS, lemmes, **gloses françaises**) à partir du dictionnaire et des règles morphologiques (non lu directement) — [ACL W14-6502](https://aclanthology.org/W14-6502/)

### Inferences
**Inventaire et usages concrets pour le pipeline NLLB-600M sur Colab T4** (ordre de priorité) :

| # | Ressource | Contenu | Licence | URL | Usage MT conseillé |
|---|---|---|---|---|---|
| 1 | Bamadaba `bamadaba.txt` | 13 135 entrées, gloses fr, ~4 644 paires d'exemples bm→fr, 5 537 variantes, tons marqués | aucune licence explicite (zone grise) | https://github.com/maslinych/bamadaba | (a) +4,6 k paires d'exemples (sans tons) ajoutées au fine-tuning ; (b) table de normalisation `\va` ; (c) lexique de contrôle |
| 2 | GATITOS `en_bm.tsv` + `en_fr.tsv` | 4 000 segments, pivot anglais, salutations | CC-BY-4.0 | https://huggingface.co/datasets/google/smol (copie GitHub au commit 3622039) | lexique bm–fr par jointure ; augmentation par *code-switching* lexical (BiLex Rx) ; jeu de test de salutations |
| 3 | Bamadaba `ETRGFRA.txt` | 1 217 emprunts au français | idem 1 | même dépôt | reconnaître et recopier les mots français dans le discours mixte |
| 4 | Interjections Bamadaba + mini-lexique maison | ~106 interjections + argot écrit par le développeur | — / propre | — | phrases synthétiques par gabarits (salutations, « wai », « yoh ») |
| 5 | Plugins daba (`bamlatinold`, `apostrophe`, `detone`) | règles de normalisation | GPL v3 | https://github.com/maslinych/daba | normaliser la sortie de l'ASR et les corpus avant la traduction |
| 6 | Sous-corpus parallèle du CBR (au-delà de Bayelemabaga) | ~68 k phrases alignées, dont ~20–27 k hors Bayelemabaga v1.0.1 | non trouvée ; textes privés | https://cormand.huma-num.fr/ | à demander à l'équipe LLACAN ; gain potentiel le plus important en volume |
| 7 | An ka taa, LDC2016L01, SIL (Marguets), Bailleul, Dumestre | dictionnaires | consultation libre / LDC / imprimé | voir section 3 | vérification manuelle ; pas d'ingestion massive sans autorisation |

- **Décodage contraint par lexique.** Une fois les termes trouvés via le Bamadaba ou GATITOS, on peut imposer leurs traductions avec `force_words_ids` / *constrained beam search* de Hugging Face sur NLLB. C'est surtout utile pour les noms propres (fichiers `togow`, `jamuw`, `yorow`) et les termes métier. Le coût en calcul est faible sur un T4, mais le décodage ralentit.
- **Effet attendu non mesuré pour le bambara.** Il faut prévoir une ablation simple : Bayelemabaga seul, puis avec les exemples du Bamadaba, puis avec GATITOS. L'évaluation se ferait sur un petit jeu « parole quotidienne » transcrit par le développeur, plutôt que sur le test de Bayelemabaga, qui est surtout de l'écrit.

### Gaps
- Aucune étude publiée ne mesure l'effet d'ajouter les exemples du Bamadaba à un modèle bm↔fr.
- Les chiffres BLEU exacts de Tapo 2020 par direction et les gains GATITOS spécifiques au bambara n'ont pas pu être lus (arXiv et ACL étaient inaccessibles).
- La disponibilité actuelle du dictionnaire SIL Mali utilisé par Tapo 2020 (bambara.org) n'a pas été vérifiée.
