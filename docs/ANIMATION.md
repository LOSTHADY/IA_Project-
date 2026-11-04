# Animation participative : les gens doublent les personnages

Des dessins animés courts, pour TikTok, où les personnages parlent avec la
voix des participants. Chaque participant reçoit sa vidéo et la publie. On
obtient deux choses d'un coup :

- **des données** : chaque réplique est un enregistrement dont le texte
  bambara est exact (le script), dit par une voix chaque fois différente,
  avec de l'expression. C'est exactement ce qu'il faut pour apprendre à
  Soloni des voix nouvelles (`docs/CAMPAGNE.md`) ;
- **du recrutement** : les vidéos font connaître le projet.

## Le principe : l'image suit la voix

Dans un doublage classique, le comédien cale sa voix sur l'image. C'est
difficile, ça décourage les amateurs, et ça déforme la parole (les gens
accélèrent, coupent leurs phrases), donc ça abîme les données. Ici, c'est
l'inverse :

1. une scène est décrite **une seule fois** : personnages, décor, répliques ;
2. le participant enregistre ses répliques **à son rythme**, une par une ;
3. le programme fabrique sa vidéo : chaque réplique dure le temps de son
   enregistrement, les bouches suivent l'audio, et le jeu se fait seul. Celui
   qui parle hoche la tête sur ses syllabes fortes et celui qui écoute se
   tourne vers lui. Tous clignent des yeux et respirent.

Une scène donne autant de vidéos que de participants, sans aucune retouche.

## Le moteur : `animation/`

Le moteur suit le principe des marionnettes 2D (Cartoon Animator, Live2D en
plus simple). Chaque personnage est un empilement de calques : corps,
oreilles, tête, yeux, pupilles, sourcils, nez, bouche, coiffe. Quand la tête
tourne, chaque calque glisse **selon sa profondeur**. Le nez et la bouche
glissent beaucoup dans le sens du regard. Les oreilles et les cheveux de
derrière glissent en sens inverse et passent sous la tête. Les traits
s'écrasent un peu en largeur. L'œil y voit une rotation, sans qu'on ait
dessiné de profil.

```bash
python -m animation demo        # deux personnages d'essai + un épisode en voix factice
python -m animation planche animation/personnages/awa planche.png   # toutes les poses
python -m animation rendre animation/episodes/demo/episode.json demo.mp4
python -m animation rendre episode.json essai.mp4 --apercu          # moitié de taille
python -m animation image episode.json 45 image.png                 # une image, pour vérifier
```

- **Sortie** : MP4 vertical 1080 × 1920, 30 images par seconde, H.264 + AAC,
  sous-titres bambara en haut de l'image (le bas est caché par l'interface
  de TikTok). Seul ffmpeg est requis.
- **Bouches** : 9 formes, celles de Rhubarb Lip Sync (X repos, A m/b/p,
  B entrouverte, C ouverte, D grande ouverte, E o, F u/w, G f/v, H l).
  Sans rien installer, la forme vient de l'audio : l'énergie dit combien la
  bouche s'ouvre, le timbre dit si elle s'arrondit. Si le programme
  [Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync) est
  installé (dans le PATH, ou variable `RHUBARB`), il est utilisé à la place,
  en mode phonétique, qui ne dépend d'aucune langue.
- **Vitesse** : environ 8 fois le temps réel sur un processeur (une vidéo de
  18 s en 2 min 20). Ça suffit pour un prototype ; un rendu sur plusieurs
  cœurs viendra si les participants affluent.
- **Mouvements imposés** : une réplique peut fixer la tête à certains
  instants (`"mouvements": [{"t": 0.5, "lacet": -0.7}]`). Le reste du temps,
  le jeu est automatique. Le format complet est décrit en tête de
  `animation/scene.py`.

Dans `animation/personnages/`, Awa et Seydou sont dessinés par le programme
(`animation/demo.py`) : ce sont des personnages d'essai. Fanta et Bakary
sont dessinés par ChatGPT, puis découpés automatiquement (voir plus bas).
`python -m animation demo` crée aussi l'épisode d'essai
`animation/episodes/fanta_bakary`.

## Fournir de vrais personnages

Un personnage est un dossier d'images PNG **transparentes, toutes de la même
taille** (celle du dessin entier, par exemple 800 × 1100). Chaque image ne
contient qu'une partie, à sa place. C'est ce qu'on obtient en exportant les
calques d'un dessin depuis Krita, Photopea (gratuit, dans le navigateur) ou
Photoshop. Aucun fichier de description n'est nécessaire si les noms
suivent la convention :

| Fichier | Contenu |
|---|---|
| `cheveux_arriere.png` | ce qui dépasse derrière la tête (chignon, nœud du foulard) — facultatif |
| `corps.png` | buste, cou et vêtements |
| `oreilles.png` | les deux oreilles, **dessinées en entier** (la tête les recouvre en partie) |
| `tete.png` | l'ovale du visage, **sans aucun trait** : ni yeux, ni nez, ni bouche |
| `joues.png` | rougeurs, scarifications, taches de rousseur — facultatif |
| `yeux_ouverts.png`, `yeux_fermes.png` | le blanc des yeux, contour compris ; les yeux fermés (deux traits) |
| `pupilles.png` | iris et pupilles, seuls (ils suivent le regard) |
| `sourcils.png` | les deux sourcils |
| `nez.png` | le nez |
| `bouche_X.png` … `bouche_H.png` | les 9 bouches, **toutes au même endroit** |
| `coiffe.png` ou `cheveux_avant.png` | foulard, bonnet, frange : ce qui est devant le front |

Avec un `personnage.json` (voir celui d'Awa), on règle en plus la profondeur
et l'écrasement de chaque partie, et le pivot du cou. Pour vérifier un
personnage, on fait sa planche : `python -m animation planche <dossier> planche.png`.

### Découper automatiquement un dessin fait d'une pièce

C'est la méthode qui a servi pour **Fanta** et **Bakary**, dessinés par
ChatGPT (`animation/brut/`). On n'a pas besoin de Photopea :
`animation/decoupe.py` tire les calques de l'image entière, grâce à quelques
repères notés une fois dans `reperes.json`. Ces repères sont un point de peau,
un point dans chaque oreille et dans chaque partie de la tête (foulard,
bonnet, barbe), les rectangles des traits, le centre des iris et le haut du
cou. Le style plat fait le reste : un trait, c'est ce qui n'a pas la couleur
de la peau. La peau est repeinte sous les traits, et les bouches viennent de
la planche du prompt 2, mises à l'échelle de celle du dessin.

```bash
python -m animation decouper animation/brut/fanta animation/personnages/fanta
python -m animation planche animation/personnages/fanta planche.png   # vérifier
```

Pour un nouveau personnage, copier le `reperes.json` de Fanta et ajuster les
coordonnées : on les lit sur l'image en pixels, avec n'importe quel logiciel
d'image. Sans image des yeux fermés, les paupières sont dessinées par le
programme (c'est le cas de Bakary).

### Les dessins avec un générateur d'images (ChatGPT, etc.)

Un générateur d'images ne sait pas produire des calques alignés. La méthode
qui marche se fait en trois temps :

1. **Générer le personnage entier**, de face, avec une **expression neutre et
   la bouche fermée**.
2. **Le découper en calques** dans Photopea. Ensuite, **reboucher le
   visage** sous les traits, c'est-à-dire repeindre la peau à leur place
   (l'outil « remplissage selon le contenu » le fait en un clic). Sinon,
   quand les yeux glissent, on voit leur ancienne place.
3. **Générer à part la planche des 9 bouches et les yeux fermés**, dans le
   même style, puis les découper et les placer sur le visage.

**Prompt 1 — le personnage** (à adapter : âge, vêtements, coiffe) :

> Personnage de dessin animé 2D, style illustration vectorielle plate,
> contours bruns foncés épais et réguliers, couleurs franches sans dégradé ni
> ombre portée. Une jeune femme malienne, peau brun foncé, foulard (musoroba)
> rouge à petits motifs jaunes noué derrière la tête, boubou vert canard à
> broderie jaune. Vue strictement de face, symétrique, buste jusqu'à la
> taille, bras le long du corps. Expression neutre et calme, **bouche
> fermée**, yeux grands ouverts regardant droit devant. Les deux oreilles
> visibles. Fond uni blanc. Pas de texte. Format portrait 800 × 1100.

**Prompt 2 — les bouches** (joindre l'image du personnage) :

> Dans exactement le même style que ce personnage (mêmes contours, mêmes
> couleurs de lèvres et de peau), une planche de 9 bouches de dessin animé,
> vues de face, à la même échelle, alignées sur une grille 3 × 3, sur fond
> uni blanc, sans texte. Dans l'ordre :
> 1. au repos, lèvres closes, léger sourire ;
> 2. lèvres pincées comme pour « m » ;
> 3. à peine entrouverte, dents visibles, comme pour « s » ou « t » ;
> 4. ouverte, comme pour « è » ;
> 5. grande ouverte, comme pour « a », langue visible ;
> 6. arrondie, comme pour « o » ;
> 7. petite et avancée, comme pour « ou » ;
> 8. dents du haut sur la lèvre inférieure, comme pour « f » ;
> 9. ouverte, langue levée derrière les dents du haut, comme pour « l ».

**Prompt 3 — les yeux fermés** (joindre l'image du personnage) :

> La même image exactement, à l'identique, mais les yeux fermés, paupières
> dessinées comme deux traits courbes. Rien d'autre ne change.

**Pour un second personnage**, reprendre le prompt 1 en joignant l'image du
premier : « dans exactement le même style que ce personnage ». Les
personnages d'une série doivent avoir l'air de venir du même dessin animé.

**Les personnages doivent être originaux.** Pas de personnage existant, même
renommé : TikTok supprime les vidéos qui enfreignent des droits, et le
projet a besoin de pouvoir réutiliser ses dessins librement.

## Ce qui reste à faire

- **Brancher l'atelier** : un onglet « Doubler » qui affiche la réplique, le
  personnage et l'émotion, enregistre chaque prise, puis rend la vidéo.
  Chaque prise entre dans les données comme une lecture : texte exact,
  contrôle par Soloni (`--filtre`), même consentement.
- **Le consentement** doit couvrir deux usages : la publication de la vidéo
  sur TikTok *et* l'entraînement des modèles. Le public de TikTok est jeune :
  pour les mineurs, il faut un accord parental, ou bien les exclure de la
  collecte. `docs/COLLECTE.md` est à compléter avant le pilote.
- **Les scripts** doivent être écrits ou relus par un locuteur natif, et il
  faut **beaucoup d'épisodes différents**. Les mêmes phrases dites par cent
  voix apprennent les voix, pas la langue.
- **Le pilote** : un épisode de 30 s, deux personnages, publié. On regarde
  combien de gens le doublent avant d'investir dans une série.
