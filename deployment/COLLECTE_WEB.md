# Collecte web · lectures en bambara

L’interface statique se trouve dans `docs/collecte/`. Le mode essai ne transmet rien. Pour une collecte réelle, le projet Supabase privé et les phrases vérifiées sont nécessaires.

## Activer le pilote

1. Dans un projet Supabase appartenant à l’équipe, exécuter une fois dans SQL Editor, dans cet ordre : `deployment/collection.sql` (tables, fonctions d’accès, bucket privé `bambara-readings`), `deployment/open-collection.sql` (participation directe, sans invitation), puis `deployment/collecte-v2.sql`. Ce dernier script peut être rejoué sans risque. Il apporte :
   - des phrases différentes pour chaque participant (les moins lues d’abord) ;
   - le bouton « Lire 10 phrases de plus », jusqu’à 200 phrases par participant ;
   - le genre, facultatif ;
   - le numéro de phrase et le genre dans l’export.
2. Dans Authentication, activer les connexions anonymes et le fournisseur e-mail. Ajouter l’URL HTTPS exacte de l’interface avec `?equipe=1` aux URL de redirection. Garder l’envoi de liens d’équipe limité aux comptes Auth existants. Si l’envoi SMTP par défaut est restreint aux membres du projet, configurer un SMTP ou utiliser les invitations Auth pour le pilote.
3. Attribuer les accès de transcription et validation à deux comptes Auth distincts. Pour chacun, insérer son UUID depuis Authentication dans `public.reading_team`.
4. Remplir `public.reading_prompts`. Pour un grand nombre de phrases, `python scripts/collecte_web.py phrases --n 500 > phrases.sql` en tire de Bayelemabaga, avec leur français : 4 à 14 mots, sans chiffres, sans texte religieux, sans doublon. Ce script se lance sur Colab ou sur un ordinateur, car il télécharge le corpus. On colle ensuite `phrases.sql` dans SQL Editor. Ces phrases viennent d’un corpus public : les faire relire par une personne native avant une grande campagne, et désactiver (`active = false`) celles qui sont fautives. Chaque participant reçoit les dix phrases actives les moins lues, puis dix autres à chaque « 10 de plus ». Pour corriger une phrase déjà attribuée, désactiver l’ancienne et ajouter une nouvelle ligne, afin de préserver les références des sessions en cours.
5. Dans `docs/collecte/config.js`, renseigner l’URL publique Supabase, la clé **publishable** (ou la clé historique **anon**) et l’adresse de contact de l’équipe. Ne placer aucune clé secrète dans ce fichier. Laisser `publicNotice` vide tant que le texte d’accord et la licence de diffusion ne sont pas finalisés.
6. Sélectionner **GitHub Actions** dans Settings → Pages, puis lancer **Publier la collecte Bambara** (Actions → Run workflow) sur la branche contenant ces fichiers, aujourd’hui `claude/focused-goldberg-uvcvdt`. La page publiée est celle de la branche choisie au lancement : relancer la publication après chaque modification de `docs/collecte`. La page fonctionne avant la migration `collecte-v2.sql` (le genre n’est alors pas enregistré), mais le bouton « 10 de plus » a besoin de cette migration. Le workflow publie uniquement les fichiers de l’interface. Si Pages héberge déjà un autre site sur ce dépôt, utiliser un dépôt dédié pour la publication ou intégrer ce dossier à son workflow existant.
7. Générer une invitation par locuteur. Partager chaque lien uniquement avec son destinataire. Les invitations sont valides 14 jours et liées au premier navigateur après acceptation. Réouvrir le même lien dans le même navigateur reprend les lectures déjà reçues.

### Opérations de départ

```sql
-- Remplacer la valeur par l’UUID d’un compte Authentication de l’équipe.
insert into public.reading_team (user_id) values ('UUID_DU_COMPTE');

-- Remplacer le texte par une phrase vérifiée et sa traduction.
insert into public.reading_prompts (sentence_bm, translation_fr)
values ('PHRASE_RELUE', 'TRADUCTION_VALIDEE');

-- Une ligne par participant. Ajouter #invitation=TOKEN au lien de collecte.
insert into public.reading_invites default values returning token, speaker, expires_at;
```

Le lien d’invitation contient un code d’accès. L’interface enlève ce code de l’adresse visible après l’avoir lu. En cas de perte de navigateur, désactiver l’ancienne invitation et en générer une nouvelle. Pour un retrait, repérer le code locuteur, révoquer son invitation, retirer les WAV via Storage et retirer les lignes correspondantes selon le protocole du projet. Les invitations, voix et exports ne vont jamais dans le dépôt Git. Pour une ouverture large, activer une protection contre les créations anonymes abusives et adapter l’interface si cette protection exige un challenge.

## Recette du pilote

- Tester l’essai depuis un téléphone en HTTPS, puis une invitation réelle avec une voix de test.
- Tester le refus de microphone, un échec d’envoi suivi d’une relance, et la reprise par le même lien après fermeture.
- Vérifier qu’un participant ne peut pas lire les fichiers et qu’un compte équipe ne peut pas approuver sa propre transcription. Vérifier qu’une autre personne peut la valider.
- Exporter le ZIP depuis l’espace équipe. Il contient `testset.jsonl` et `audio/*.wav`. `python scripts/collecte_web.py importer bambara-lectures.zip --atelier ATELIER` le verse dans le dossier de l’atelier (sur Drive : `MyDrive/bambara/atelier`). Chaque lecteur y devient un contributeur, et chaque lecture une prise du mode « lire ». Le notebook `voix` les utilise alors pour affiner Soloni. Réimporter le même ZIP n’ajoute rien en double. L’export normal est interne. Le filtre de diffusion ne garde que les lectures validées avec l’accord correspondant.
- Le champ `register` vaut `lu`. Ce lot peut servir à l’évaluation des lectures. Il ne couvre pas la parole spontanée que le protocole historique prévoit également.

## Tests

`tests/test_collecte_web.py` vérifie l’import dans l’atelier. Il joue aussi tout le parcours SQL sur un vrai PostgreSQL, quand `COLLECTE_PG_DSN` est défini, ce que fait la CI : phrases les moins lues, « 10 de plus », reprise, double validation par l’équipe, refus d’un genre inconnu. Les schémas `auth` et `storage` de Supabase y sont imités.
