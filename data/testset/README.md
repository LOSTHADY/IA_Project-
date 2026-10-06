# Jeu de test maison

C'est le chemin critique du projet : sans ce jeu, aucun chiffre du mémoire
n'est défendable. Tout le reste du code peut avancer en parallèle, mais la
collecte doit démarrer en premier.

La collecte se fait avec `python app/collect_app.py`, selon le protocole de
[`docs/COLLECTE.md`](../../docs/COLLECTE.md). Le fichier `testset.jsonl`
décrit ci-dessous en est l'export (`python scripts/export_testset.py`).

| Fichier | Contenu | Versionné |
|---|---|---|
| `consignes.json` | situations de parole spontanée | oui |
| `testset.example.jsonl` | trois lignes d'exemple du format | oui |
| `locuteurs.jsonl` | locuteurs pseudonymisés et leur consentement | **non** |
| `collecte.jsonl` | enregistrements, transcriptions, statuts | **non** |
| `audio/` | WAV mono 16 kHz | **non** |
| `testset*.jsonl` | exports | **non** |

Le dépôt est public : les données de collecte restent hors de git tant que
les locuteurs n'ont pas accepté leur diffusion. **Les sauvegarder
ailleurs après chaque séance.**

## Format

Un fichier JSONL, une ligne par énoncé, encodé en UTF-8 :

```json
{
  "id": "bv-0001",
  "audio": "audio/bv-0001.wav",
  "transcript_bm": "transcription bambara, orthographe officielle",
  "translation_fr": "traduction française de référence",
  "expected_reply_bm": "réponse attendue en bambara (facultatif)",
  "speaker": "spk-03",
  "gender": "f",
  "region": "Bamako",
  "register": "spontane",
  "code_switching": true,
  "duration_s": 3.8
}
```

Seuls `id`, `audio` et `transcript_bm` sont obligatoires. `translation_fr` est
nécessaire pour évaluer la variante bout-en-bout — sans lui, la comparaison
des deux architectures est impossible.

## Cible

**200 à 500 énoncés.** En dessous de 200, les intervalles de confiance sur le
WER sont trop larges pour conclure quoi que ce soit.

Composition à viser, à documenter dans le mémoire :

- au moins **8 locuteurs différents**, équilibrés en genre ;
- un mélange **lu / spontané** — le spontané est beaucoup plus dur, ne
  garder que du lu gonflerait artificiellement les scores ;
- une part assumée de **code-switching bambara-français** (très fréquent à
  l'oral, surtout à Bamako) : l'exclure rendrait l'évaluation irréaliste ;
- des durées variées, 2 à 15 secondes ;
- des conditions d'enregistrement variées (téléphone, salle calme, extérieur).

## Audio

WAV mono, 16 kHz, 16 bits. `scripts/prepare_testset.py` convertit et vérifie.

## Éthique et licence

À traiter explicitement dans le mémoire :

- consentement éclairé de chaque locuteur, par écrit, mentionnant l'usage
  (recherche académique) et la diffusion prévue ;
- pas de données personnelles identifiantes dans les énoncés ;
- licence de diffusion choisie à l'avance (CC BY-SA ou CC BY-NC, à décider
  avec le·s locuteur·rice·s, pas après coup).

Un jeu de test bambara propre, documenté et publié est en soi une
contribution réutilisable — potentiellement la partie la plus durable du
travail.
