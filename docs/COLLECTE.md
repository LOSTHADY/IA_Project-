# Protocole de collecte du jeu de test

La phase 2 est le chemin critique du projet (cf. [`METHODE.md`](METHODE.md),
§7) : sans jeu de test, aucun chiffre du mémoire n'est défendable. Ce
document décrit comment le constituer, de la signature du consentement à
l'export. L'outil est `app/collect_app.py` ; le format final est décrit dans
[`data/testset/README.md`](../data/testset/README.md).

```bash
python app/collect_app.py                      # http://127.0.0.1:7861
python scripts/export_testset.py               # énoncés validés -> testset.jsonl
python scripts/prepare_testset.py data/testset/testset.jsonl
```

## 1. Avant la première séance

- **Matériel** : un ordinateur portable et un micro-casque USB. Des
  enregistrements au téléphone sont bienvenus (ils varient les conditions) :
  enregistrer avec le dictaphone du téléphone, puis importer le fichier dans
  l'onglet « Enregistrer ».
- **Lancer l'application en local** (`python app/collect_app.py`). Elle
  n'écoute que sur la machine elle-même. Ne pas l'exposer avec
  `share=True` : les voix transiteraient par un serveur tiers.
- **Imprimer le formulaire de consentement** (§8), deux exemplaires par
  locuteur : un pour lui, un pour le dossier.
- **Fixer les conventions de transcription** (§4) avec le transcripteur et
  le valideur *avant* de transcrire quoi que ce soit.

## 2. Consentement

Rien n'est enregistré sans consentement : l'application refuse de créer un
locuteur si la case n'est pas cochée.

1. Expliquer le projet **en bambara** : à quoi servent les enregistrements,
   qui y aura accès, ce qui pourra être publié, et le droit de retrait.
2. Faire signer le formulaire. Si le locuteur ne lit pas, le consentement
   oral est recueilli devant un témoin, qui signe.
3. Deux accords distincts : l'usage pour la recherche (condition pour
   participer) et la **diffusion publique** (facultative). Le second est
   reporté dans l'application et décide de ce que `--public-only` exporte.
4. L'application attribue un pseudonyme (`spk-01`...). L'écrire sur le
   formulaire. **Aucun nom** dans l'application ni dans le dépôt : les
   formulaires signés sont conservés à part (papier, ou dossier chiffré).

**Composition visée** : au moins 8 locuteurs, moitié de femmes, âges et
régions variés. Un panel composé uniquement d'étudiants de Bamako donnerait
des scores qui ne disent rien du reste du pays.

## 3. Séance d'enregistrement

Compter 30 à 40 minutes par locuteur, pour environ **20 énoncés spontanés
et 10 lus**. Dix locuteurs donnent ainsi environ 300 énoncés, au milieu de
la cible de 200 à 500. Aucun locuteur ne doit dépasser 20 % du total :
l'onglet « Avancement » le signale.

**Si l'on peut recruter davantage, mieux vaut plus de locuteurs que plus
d'énoncés par locuteur.** Les énoncés d'une même voix se ressemblent : les
intervalles de confiance se calculent en tirant les locuteurs, pas les
énoncés (`docs/METHODE.md`, §5). À 300 énoncés, 15 locuteurs × 20 énoncés
donnent donc des intervalles plus serrés que 10 × 30. Ordre de grandeur, avec
l'effet de grappe classique (1 + (m − 1) ρ, où m est le nombre d'énoncés par
locuteur et ρ leur corrélation, supposée ici à 0,1 pour l'illustration) :
300 énoncés valent environ 77 énoncés indépendants avec 10 locuteurs, 103
avec 15, 125 avec 20. La valeur réelle de ρ se lira sur le jeu une fois
collecté.

**Spontané.** L'application propose une situation (`data/testset/consignes.json`) :
saluer, demander un prix, signaler une panne de réseau, etc. L'opérateur
l'explique, en bambara si possible, et le locuteur **parle à l'assistant
avec ses propres mots**. Il ne faut pas lui faire traduire la consigne
française : on obtiendrait des calques et un code-switching artificiel. Le
code-switching naturel, lui, est bienvenu et ne doit pas être corrigé.

**Lu.** Les phrases à lire sont les transcriptions *validées* d'énoncés
spontanés d'autres locuteurs, proposées automatiquement. Deux conséquences :

- on n'invente pas de bambara ;
- le même contenu existe en version spontanée et en version lue, ce qui
  permet de mesurer l'effet du registre à contenu égal (à exploiter dans le
  mémoire).

Au tout début, aucune phrase n'est encore validée : commencer par le
spontané. L'option `--phrases fichier.txt` ajoute des phrases (une par
ligne) rédigées ou validées par un locuteur natif.

**Conditions.** Mettre à jour le champ « Conditions » à chaque changement
(salle calme, intérieur bruyant, extérieur, téléphone), et les varier d'un
locuteur à l'autre.

**Contrôle immédiat.** Après chaque énoncé, l'application signale un
enregistrement coupé, trop long, saturé ou trop faible. Dans ce cas,
« Supprimer et réenregistrer » repropose la même consigne. Réenregistrer
tout de suite coûte dix secondes ; s'en apercevoir à la transcription coûte
l'énoncé.

## 4. Conventions de transcription

La cohérence compte plus que le choix de telle ou telle convention. Une
convention changée en cours de route oblige à reprendre les transcriptions
déjà faites.

- **Orthographe officielle** : ɛ, ɔ, ɲ, ŋ (clavier d'appoint sous le champ
  de transcription), tons non notés. C'est ce qui donne un sens au WER
  strict (cf. `METHODE.md`, §5).
- **Écrire ce qui est dit**, pas ce qui aurait dû être dit : aucune
  correction grammaticale.
- **Mots français** (code-switching) : orthographe française standard, et
  cocher « code-switching ».
- **Nombres** en toutes lettres, dans la langue où ils sont prononcés.
- **Élisions** notées comme prononcées : `k'a`, `b'a`, `n'a`.
- **Hésitations** (« ee », « hmm ») et bruits : non transcrits. Les
  répétitions et faux départs sont transcrits tels quels.
- **Passage incompréhensible** : ne pas deviner. Le signaler en note et ne
  pas valider l'énoncé, qui reste alors hors du jeu de test.
- **Traduction française** : fidèle au sens, simple, sans embellissement.
  C'est la référence de la variante bout-en-bout, et elle est obligatoire
  pour que l'énoncé serve aux deux architectures.

## 5. Validation

Une seconde personne, locutrice native, écoute chaque énoncé, corrige la
transcription et la traduction si besoin, puis valide. L'application
refuse qu'une même personne transcrive et valide le même énoncé, et
n'exporte par défaut que les énoncés validés. Les initiales du
transcripteur et du valideur sont conservées pour la traçabilité.

Pour les énoncés lus, le texte affiché sert de première transcription. La
validation vérifie que le locuteur a bien dit ce qui était écrit :
corriger toute différence (mot sauté, reformulation).

## 6. Export et contrôle

```bash
python scripts/export_testset.py                  # validés -> data/testset/testset.jsonl
python scripts/prepare_testset.py data/testset/testset.jsonl
```

`prepare_testset.py` vérifie la taille, le nombre de locuteurs, la
présence de spontané et de traductions, et l'usage de l'orthographe
officielle. L'option `--include-unvalidated` sert à tester la chaîne
d'évaluation avant la fin de la collecte, jamais à produire les chiffres du
mémoire.

## 7. Données : dépôt, sauvegarde, diffusion

- **Le dépôt GitHub est public.** Les fichiers audio, `locuteurs.jsonl`,
  `collecte.jsonl` et les `testset*.jsonl` exportés sont exclus par
  `.gitignore`. Ne jamais les forcer (`git add -f`).
- **Sauvegarde.** Comme ces fichiers ne sont pas dans git, copier tout le
  dossier `data/testset/` sur un second support (disque externe, stockage
  privé) **après chaque séance**. Perdre ce dossier, c'est perdre des
  semaines de terrain.
- **Diffusion.** Seul l'export `--public-only` (locuteurs ayant accepté la
  diffusion publique) peut être publié, sous la licence choisie avant la
  collecte. Un jeu de test bambara documenté et publié est une contribution
  en soi.
- **Droit de retrait.** Si un locuteur se retire, supprimer ses lignes
  (`"speaker": "spk-XX"`) dans `collecte.jsonl`, sa ligne dans
  `locuteurs.jsonl` et ses fichiers audio, puis refaire l'export.
- **Cadre légal.** La voix est une donnée personnelle. Au Mali, la loi
  n° 2013-015 relative à la protection des données à caractère personnel
  s'applique, sous le contrôle de l'APDP. Vérifier avec l'encadrant les
  démarches exigées (déclaration, avis d'un comité d'éthique de
  l'université).

## 8. Modèle de formulaire de consentement

À adapter, puis à faire relire par l'encadrant. Le texte est à expliquer
oralement en bambara.

---

**Formulaire de consentement — enregistrement de voix en bambara**

*Projet* : assistant vocal en bambara (mémoire de ………………, université ………………).
*Responsable* : ……………… — contact : ………………

*Ce que l'on vous demande.* Parler en bambara pendant 30 à 40 minutes :
répondre à des situations de la vie courante et lire quelques phrases. Vos
paroles sont enregistrées.

*À quoi servent les enregistrements.* À mesurer la qualité d'un système qui
comprend et parle le bambara. Ils sont écoutés et transcrits par l'équipe
du projet.

*Anonymat.* Votre nom n'apparaît nulle part : vous êtes désigné(e) par un
code (`spk-……`). Seuls votre genre, votre tranche d'âge et votre région sont
notés. Ne donnez pas d'informations personnelles pendant l'enregistrement.

*Participation volontaire.* Vous pouvez arrêter à tout moment. Vous pouvez
demander la suppression de vos enregistrements jusqu'au ……………… en
contactant le responsable, sans avoir à vous justifier.

*Dédommagement* (le cas échéant) : ………………

Cochez ce que vous acceptez :

- [ ] J'accepte que ma voix soit enregistrée et utilisée **pour la recherche**
  dans le cadre de ce mémoire (obligatoire pour participer).
- [ ] J'accepte que mes enregistrements et leur transcription soient
  **publiés** pour d'autres chercheurs, sans mon nom, sous la licence
  ……………… (facultatif).

Code locuteur : `spk-……` Date : ……………… Lieu : ………………

Signature du participant : ……………… Signature du responsable : ………………

Si le participant ne lit pas — témoin (nom et signature) : ………………

---
