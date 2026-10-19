"""Atelier de données — dire, lire, juger, enrichir le lexique, discuter.

    python app/atelier_app.py                        # sur cet ordinateur
    python app/atelier_app.py --racine /content/drive/MyDrive/bambara/atelier --share
    python app/atelier_app.py --racine ... --config assistant.json --share   # + Discuter

Pour contribuer sans rien transcrire : on dit en bambara une phrase affichée
en français, on lit une phrase bambara, on choisit la meilleure de deux
traductions, on note des mots familiers. Chacun reprend là où il s'était
arrêté ; plusieurs contributeurs peuvent travailler en même temps.

Avec `--config` (les modèles de l'assistant, sur GPU de préférence), un onglet
« Discuter » s'ajoute : le mode entraînement. On parle librement ; on choisit
ou corrige ce que l'assistant a entendu et ce qu'il a compris ; il le retient
(mémoire) et chaque tour devient une donnée d'entraînement.

Toute la logique est dans `eval/atelier.py` ; ce fichier n'est qu'une façade.
Mode d'emploi : docs/ATELIER.md. Avec --share, Gradio ouvre un lien public
(gradio.live) utilisable depuis un téléphone : ne le donner qu'aux
contributeurs, chacun ayant donné son consentement.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.atelier import (  # noqa: E402
    INTERLOCUTEURS, LECTURE_CIBLE_MIN, PHRASES_A_DIRE, TYPES_LEXIQUE, Atelier,
    charger_phrases_a_dire, rubriques, texte_fr,
)
from eval.collection import CONDITIONS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

AIDE_DIRE = """\
**Comment faire.** Choisis **à qui tu parles** (un ami, un inconnu, une personne
âgée), lis la phrase française, puis dis-la **en bambara, comme tu la lui
dirais** : avec tes mots, pas mot à mot. Si tu mélanges avec du français ou de
l'anglais, c'est très bien : coche la case. Appuie sur le micro, parle, arrête,
écoute si tu veux, puis « Enregistrer et suivante ».

**Plusieurs versions sont les bienvenues** : « Une autre façon » si tu la dirais
aussi autrement à la même personne, « À quelqu'un d'autre » pour la redire à
un inconnu ou à une personne âgée. Une phrase qui ne te parle pas : « Passer »."""

AIDE_LIRE = """\
**Comment faire.** Lis la phrase bambara **à voix haute, naturellement**, au
calme, toujours à la même distance du micro. Cette voix servira à la synthèse
vocale : une seule personne, au moins une heure au total, idéalement.
Une phrase difficile à lire : « Passer »."""

AIDE_JUGER = """\
**Comment faire.** Une phrase, puis deux traductions produites par deux
versions de l'assistant, dans un ordre tiré au hasard. Choisis la meilleure
**pour le sens d'abord**, puis pour le naturel. Ne cherche pas la perfection :
ton premier avis compte."""

AIDE_DISCUTER = """\
**Comment faire.** Parle à l'assistant **librement**, comme à quelqu'un :
une question, une demande, une salutation. Puis aide-le à apprendre :

1. **Ce que j'ai entendu** : choisis la bonne proposition, ou **corrige
   seulement les mots faux** dans la case. Écris le bambara comme tu
   l'entends, même avec les lettres du français (« tchè », « gnogon ») :
   l'atelier le remet en orthographe standard.
2. **Ce que j'ai compris** : corrige le français s'il le faut, puis valide.
3. **Ma réponse** : dis si elle était juste.

Ce que tu corriges, l'assistant le retient tout de suite ; et chaque tour
sert à l'entraîner."""

AIDE_LEXIQUE = """\
**Les mots que les corpus ne connaissent pas** : argot (« wai »), mots
français ou anglais glissés dans le bambara, expressions de jeunes. Donne le
sens en français, et un équivalent en bambara courant si tu en connais un."""


# --- affichage ----------------------------------------------------------------

def barre(faites: int, total: int, legende: str = "") -> str:
    """Barre de progression en HTML (styles en ligne : lisible en thème clair
    comme sombre)."""
    pct = 100 * faites / total if total else 0
    return (f'<div style="margin:4px 0">'
            f'<div style="font-size:0.9em;margin-bottom:2px">{html.escape(legende)} '
            f'<b>{faites}/{total}</b> ({pct:.0f} %)</div>'
            f'<div style="height:10px;border-radius:5px;background:rgba(128,128,128,0.25)">'
            f'<div style="height:10px;border-radius:5px;width:{pct:.1f}%;'
            f'background:#2e7d32"></div></div></div>')


def citation(texte: str, langue: str) -> str:
    return f'### « {texte} »\n<small>{langue}</small>'


def candidat(lettre: str, texte: str) -> str:
    return (f'<div style="padding:10px;border:1px solid rgba(128,128,128,0.4);'
            f'border-radius:8px"><b>{lettre}</b><br>'
            f'<span style="font-size:1.3em">{html.escape(texte)}</span></div>')


def build_interface(atelier: Atelier, phrases_dire: list[dict], pipeline=None):
    """`pipeline` (bambara_voice.pipeline.VoicePipeline) : avec lui, l'onglet
    « Discuter » (mode entraînement)."""
    import gradio as gr

    titres = dict(rubriques(phrases_dire))

    # --- contributeurs -------------------------------------------------------

    def choix_locuteurs():
        return [(f"{s.id} ({'femme' if s.gender == 'f' else 'homme'}"
                 f"{', ' + s.region if s.region else ''})", s.id) for s in atelier.locuteurs()]

    def ajouter_locuteur(genre, age, region, consent, public):
        if not genre:
            return "⚠️ Indique le genre.", gr.Dropdown()
        try:
            spk = atelier.ajouter_locuteur(genre, age_range=age or "", region=region.strip(),
                                           consent=bool(consent), public_release=bool(public))
        except ValueError as exc:
            return f"⚠️ {exc}.", gr.Dropdown()
        return (f"✅ Bienvenue, **{spk.id}** ! Tu peux commencer.",
                gr.Dropdown(choices=choix_locuteurs(), value=spk.id))

    # --- dire ------------------------------------------------------------------

    def rubriques_dd(spk, qui="ami"):
        prog = {r["rubrique"]: r for r in atelier.progression("dire", spk, phrases_dire,
                                                              qui or "ami")} if spk else {}
        choix = [("Toutes, dans l'ordre", "")]
        for rid, titre in titres.items():
            r = prog.get(rid)
            choix.append((f"{titre} — {r['faites']}/{r['total']}" if r else titre, rid))
        return choix

    def progression_dire(spk, qui="ami", courante=None):
        """Pour l'interlocuteur choisi : le total et la rubrique en cours ; le
        détail, repliable ; et les prises de toutes les versions."""
        if not spk:
            return ""
        prog = atelier.progression("dire", spk, phrases_dire, qui)
        total = sum(r["total"] for r in prog)
        faites = sum(r["faites"] for r in prog)
        lignes = [barre(faites, total, f"À {INTERLOCUTEURS[qui]}, au total :")]
        lignes += [barre(r["faites"], r["total"], r["titre"] + " :")
                   for r in prog if r["rubrique"] == courante]
        par_qui = atelier.par_interlocuteur(spk)
        toutes = " · ".join(f"à {INTERLOCUTEURS[i]} : {n}" for i, n in par_qui.items())
        lignes.append(f'<div style="font-size:0.9em;margin-top:4px">Toutes tes prises : '
                      f'{html.escape(toutes)}</div>')
        detail = "".join(barre(r["faites"], r["total"], r["titre"]) for r in prog)
        lignes.append(f'<details style="margin-top:6px"><summary>Progression par rubrique '
                      f'(à {html.escape(INTERLOCUTEURS[qui])})</summary>{detail}</details>')
        return "".join(lignes)

    def montrer_dire(spk, rubrique, qui="ami", courante=None, forcer=False):
        """La phrase en cours si elle reste à dire à cet interlocuteur (ou si
        `forcer` : une autre façon de la dire), sinon la suivante."""
        qui = qui or "ami"
        if not spk:
            return None, "_Choisis d'abord qui tu es, en haut de la page._", ""
        p = None
        if courante and (forcer or not atelier.deja_faite("dire", spk, courante["id"], qui)):
            p = courante
        if p is None:
            p = atelier.prochaine("dire", spk, phrases_dire, rubrique or None, qui)
        if p is None:
            fin = "cette rubrique" if rubrique else "toutes les phrases"
            return (None, f"🎉 Tu as terminé {fin} pour {INTERLOCUTEURS[qui]}. Merci ! "
                          f"Choisis une autre rubrique, ou une autre personne.",
                    progression_dire(spk, qui))
        rang = [q["id"] for q in phrases_dire if q["rubrique"] == p["rubrique"]].index(p["id"]) + 1
        n = sum(q["rubrique"] == p["rubrique"] for q in phrases_dire)
        legende = f"{p['titre']} · phrase {rang} sur {n} · à dire en bambara à {INTERLOCUTEURS[qui]}"
        deja = atelier.versions(spk, p["id"])
        if deja:
            legende += " · déjà dite : " + ", ".join(
                f"à {INTERLOCUTEURS[i]}" + (f" ({k} fois)" if k > 1 else "")
                for i, k in deja.items())
        return p, citation(texte_fr(p, qui), legende), progression_dire(spk, qui, p["rubrique"])

    def changer_qui(spk, rubrique, qui, courante):
        """Changer de personne garde la phrase en cours, si elle lui reste à dire."""
        return (*montrer_dire(spk, rubrique, qui, courante),
                gr.Dropdown(choices=rubriques_dd(spk, qui), value=rubrique))

    def apres_dire(spk, rubrique, qui, msg, garder=None):
        p, texte, prog = montrer_dire(spk, rubrique, qui, garder, forcer=garder is not None)
        return (msg, None, p, texte, prog,
                gr.Dropdown(choices=rubriques_dd(spk, qui), value=rubrique), False, "", qui)

    def _enregistrer_dire(spk, rubrique, qui, phrase, audio, cs, argot, conditions, suite):
        qui = qui or "ami"
        if not spk or not phrase:
            return apres_dire(spk, rubrique, qui, "⚠️ Choisis qui tu es, puis une phrase.")
        if audio is None:
            return (("⚠️ Rien d'enregistré : appuie sur le micro, parle, puis arrête.",)
                    + (gr.update(),) * 8)
        sr, data = audio
        try:
            prise, problemes = atelier.enregistrer("dire", spk, phrase, data, sr, cs, argot,
                                                   conditions, qui)
        except ValueError as exc:
            return (f"⚠️ {exc}.",) + (gr.update(),) * 8
        msg = f"✅ Enregistré ({prise.duration_s:.1f} s, à {INTERLOCUTEURS[qui]})."
        if problemes:
            msg += " ⚠️ " + " ; ".join(problemes) + " — « Annuler la dernière » pour recommencer."
        if suite == "autre_facon":
            return apres_dire(spk, rubrique, qui, msg + " Dis-la maintenant d'une autre façon.",
                              garder=phrase)
        if suite == "autre_personne":
            autre = atelier.autre_interlocuteur(spk, phrase["id"], qui)
            if autre:
                return apres_dire(spk, rubrique, autre,
                                  msg + f" Maintenant, dis-la à {INTERLOCUTEURS[autre]}.",
                                  garder=phrase)
            msg += " Elle est dite à tout le monde : phrase suivante."
        return apres_dire(spk, rubrique, qui, msg)

    def enregistrer_dire(spk, rubrique, qui, phrase, audio, cs, argot, conditions):
        return _enregistrer_dire(spk, rubrique, qui, phrase, audio, cs, argot, conditions,
                                 "suivante")

    def enregistrer_autre_facon(spk, rubrique, qui, phrase, audio, cs, argot, conditions):
        return _enregistrer_dire(spk, rubrique, qui, phrase, audio, cs, argot, conditions,
                                 "autre_facon")

    def enregistrer_autre_personne(spk, rubrique, qui, phrase, audio, cs, argot, conditions):
        return _enregistrer_dire(spk, rubrique, qui, phrase, audio, cs, argot, conditions,
                                 "autre_personne")

    def passer_dire(spk, rubrique, qui, phrase):
        qui = qui or "ami"
        if spk and phrase:
            atelier.passer("dire", spk, phrase["id"], qui)
        return apres_dire(spk, rubrique, qui, f"⏭️ Phrase passée (pour {INTERLOCUTEURS[qui]}).")

    def annuler_dire(spk, rubrique, qui):
        prise = atelier.annuler_derniere("dire", spk) if spk else None
        if prise is None:
            return apres_dire(spk, rubrique, qui or "ami", "Rien à annuler.")
        # La phrase annulée revient, avec la personne à qui elle était dite.
        phrase = next((p for p in phrases_dire if p["id"] == prise.prompt_id), None)
        qui = prise.interlocuteur or "ami"
        return apres_dire(spk, rubrique, qui,
                          f"↩️ Dernière prise supprimée : « {prise.texte} » revient.",
                          garder=phrase)

    # --- lire ------------------------------------------------------------------

    def progression_lire(spk):
        if not spk:
            return ""
        lues = len(atelier.prises("lire", spk))
        total = len(atelier.phrases_a_lire())
        minutes = atelier.minutes("lire", spk)
        return (barre(lues, total, "Phrases lues :")
                + barre(int(minutes), LECTURE_CIBLE_MIN, "Minutes de voix (objectif) :"))

    def montrer_lire(spk):
        phrases = atelier.phrases_a_lire()
        if not phrases:
            return None, ("_Pas encore de phrases à lire : lancer la cellule « Phrases à "
                          "lire » du notebook (ou `python -m eval.atelier --racine … "
                          "lecture`)._"), ""
        if not spk:
            return None, "_Choisis d'abord qui tu es, en haut de la page._", ""
        p = atelier.prochaine("lire", spk, phrases)
        if p is None:
            return None, "🎉 Tu as lu toutes les phrases. Merci !", progression_lire(spk)
        return p, citation(p["bm"], "à lire à voix haute"), progression_lire(spk)

    def apres_lire(spk, msg):
        p, texte, prog = montrer_lire(spk)
        return msg, None, p, texte, prog

    def enregistrer_lire(spk, phrase, audio, conditions):
        if not spk or not phrase:
            return apres_lire(spk, "⚠️ Choisis qui tu es.")
        if audio is None:
            return ("⚠️ Rien d'enregistré.",) + (gr.update(),) * 4
        sr, data = audio
        try:
            prise, problemes = atelier.enregistrer("lire", spk, phrase, data, sr,
                                                   conditions=conditions)
        except ValueError as exc:
            return (f"⚠️ {exc}.",) + (gr.update(),) * 4
        msg = f"✅ Enregistré ({prise.duration_s:.1f} s)."
        if problemes:
            msg += " ⚠️ " + " ; ".join(problemes) + " — « Annuler la dernière » pour recommencer."
        return apres_lire(spk, msg)

    def passer_lire(spk, phrase):
        if spk and phrase:
            atelier.passer("lire", spk, phrase["id"])
        return apres_lire(spk, "⏭️ Phrase passée.")

    def annuler_lire(spk):
        prise = atelier.annuler_derniere("lire", spk) if spk else None
        return apres_lire(spk, "↩️ Dernière prise supprimée." if prise else "Rien à annuler.")

    # --- juger -----------------------------------------------------------------

    def choix_lots(spk):
        out = []
        for lot in atelier.lots():
            n = len(atelier.items(lot))
            faits = sum(j.juge == spk for j in atelier.jugements(lot)) if spk else 0
            out.append((f"{lot} — {faits}/{n}", lot))
        return out

    def montrer_item(spk, lot):
        vide = (None, "", "", "", "")
        if not atelier.lots():
            return vide[:1] + ("_Pas encore de lot à juger : lancer la cellule « Lots à "
                               "juger » du notebook._", "", "", "")
        if not spk or not lot:
            return vide[:1] + ("_Choisis qui tu es, puis un lot._", "", "", "")
        item = atelier.prochain_item(lot, spk)
        n = len(atelier.items(lot))
        faits = sum(j.juge == spk for j in atelier.jugements(lot))
        prog = barre(faits, n, "Jugées :")
        if item is None:
            return None, "🎉 Lot terminé. Merci ! Résultats dans l'onglet « Bilan ».", "", "", prog
        langue = "français" if item["sens"] == "fr2bm" else "bambara"
        cible = "bambara" if item["sens"] == "fr2bm" else "français"
        source = citation(item["source"], f"phrase en {langue} · traductions en {cible}")
        return (item, source, candidat("A", item["a"]), candidat("B", item["b"]), prog)

    def juger(spk, lot, item, choix):
        if spk and lot and item:
            atelier.juger(lot, spk, item["id"], choix)
        return montrer_item(spk, lot) + (gr.Dropdown(choices=choix_lots(spk), value=lot),)

    # --- lexique ---------------------------------------------------------------

    def table_lexique():
        return [[m.forme, m.sens_fr, m.equivalent_bm, m.type, m.exemple]
                for m in atelier.lexique()] or [["", "", "", "", ""]]

    def ajouter_mot(spk, forme, sens, equiv, type_, exemple):
        try:
            mot = atelier.ajouter_mot(forme, sens, equiv, exemple, type_, spk or "")
        except ValueError as exc:
            return f"⚠️ {exc}.", gr.update(), forme, sens, equiv, exemple
        return f"✅ « {mot.forme} » ajouté ({len(atelier.lexique())} mots).", \
            table_lexique(), "", "", "", ""

    def retirer_mot(forme):
        ok = atelier.retirer_mot(forme or "")
        return (f"🗑️ « {forme} » retiré." if ok else "Mot introuvable."), table_lexique()

    # --- bilan -----------------------------------------------------------------

    # --- discuter (mode entraînement) ---------------------------------------

    correcteur = None

    def _correcteur():
        nonlocal correcteur
        if correcteur is None:
            from bambara_voice.orthographe import Correcteur
            correcteur = Correcteur()
        return correcteur

    def ecouter(spk, audio):
        """Ce que l'assistant a entendu : ses propositions à choisir ou corriger."""
        import gradio as gr
        if not spk:
            return gr.Radio(), "", "Choisis d'abord qui tu es, en haut de la page.", None
        if audio is None:
            return gr.Radio(), "", "Rien d'enregistré : appuie sur le micro, puis parle.", None
        from eval.collection import to_mono_16k
        sr, data = audio
        wav = to_mono_16k(data, sr)
        propositions = pipeline.asr.alternatives(wav)
        souvenir = pipeline.memoire.chercher(propositions[0])
        if souvenir and souvenir[0].bm not in propositions:
            propositions.append(souvenir[0].bm)
        etat = {"audio": (16_000, wav), "propositions": propositions}
        return (gr.Radio(choices=propositions, value=propositions[0]), propositions[0],
                "**1. Ce que j'ai entendu.** Choisis la bonne proposition, ou corrige les "
                "mots faux dans la case, puis valide.", etat)

    def choisir_proposition(choix):
        return choix or ""

    def valider_entendu(etat, texte, inaudible=False):
        """Le bambara confirmé (en orthographe standard), puis le français que
        l'assistant en comprend, à confirmer ou corriger."""
        if not etat:
            return "Parle d'abord à l'assistant.", "", etat
        propositions = etat["propositions"]
        if inaudible or not (texte or "").strip():
            statut, std, changements = "inaudible", "", []
            base = propositions[0]
        else:
            from bambara_voice.normalize import normalize, strip_punctuation
            c = _correcteur().corriger(texte, propositions)
            std, changements = c.texte, c.changements
            forme = lambda t: strip_punctuation(normalize(t))  # noqa: E731
            statut = "juste" if forme(std) in {forme(x) for x in propositions} else "corrige"
            base = std
        trouve = pipeline.memoire.chercher(base)
        compris = trouve[0].fr if trouve else pipeline.mt_in.translate(
            pipeline.memoire.remplacer_mots(base))
        etat = {**etat, "entendu": (texte or "").strip(), "entendu_std": std,
                "entendu_statut": statut, "compris_auto": compris}
        ecrit = (f"Écrit en orthographe standard : **{html.escape(std)}**"
                 + (f" ({', '.join(f'{a} → {b}' for a, b in changements)})" if changements else "")
                 if std else "Pas de bambara retenu pour la reconnaissance.")
        return (f"{ecrit}\n\n**2. Ce que j'ai compris.** Corrige le français s'il le faut, "
                "puis valide.", compris, etat)

    def valider_compris(spk, conditions, etat, compris):
        """Le tour est enregistré, retenu par la mémoire, puis l'assistant répond."""
        if not etat or "entendu_statut" not in etat:
            return "Commence par l'étape 1.", None, None, etat, ""
        if not (compris or "").strip():
            return "Écris ce que ta phrase voulait dire.", None, None, etat, compris
        sr, wav = etat["audio"]
        statut = "juste" if compris.strip() == etat["compris_auto"].strip() else "corrige"
        tour = atelier.enregistrer_tour(
            spk, wav, sr, etat["propositions"], etat["entendu"], etat["entendu_std"],
            etat["entendu_statut"], etat["compris_auto"], compris, statut, conditions)
        # La mémoire : ce que la reconnaissance entend, et la forme corrigée.
        for bm in {etat["propositions"][0], etat["entendu_std"]} - {""}:
            pipeline.memoire.apprendre(bm, compris, "conversation")
        from bambara_voice.pipeline import TurnTrace
        trace = TurnTrace(architecture="cascade", source_bm=etat["entendu_std"]
                          or etat["propositions"][0], source_fr=compris.strip())
        pipeline.respond(trace)
        son = pipeline.speak(trace)
        etat = {"tour": tour.id, "reponse_bm": trace.reply_bm}
        msg = (f"✅ Retenu ({tour.id}). **3. Ma réponse** : {html.escape(trace.reply_bm)} "
               f"(« {html.escape(trace.reply_fr)} »). Était-elle juste ?")
        audio = (son.sample_rate, son.audio) if son is not None and len(son.audio) else None
        return msg, audio, None, etat, ""

    def noter_reponse(spk, etat, juste):
        if not etat or "tour" not in etat:
            return "Rien à noter pour l'instant.", etat
        atelier.juger_reponse(etat["tour"], spk, etat["reponse_bm"], juste)
        return ("Merci ! Parle à nouveau quand tu veux.", None)

    def reponse_juste(spk, etat):
        return noter_reponse(spk, etat, True)

    def reponse_fausse(spk, etat):
        return noter_reponse(spk, etat, False)

    def bilan_markdown():
        b = atelier.bilan(phrases_dire)
        qui = " · ".join(INTERLOCUTEURS.values())
        lignes = ["### Contributeurs", "",
                  f"| Contributeur | Prises dites ({qui}) | Phrases lues | Minutes lues "
                  f"| Jugements |", "|---|---|---|---|---|"]
        for r in b["locuteurs"]:
            dites = " · ".join(str(n) for n in r["par_interlocuteur"].values())
            lignes.append(f"| {r['locuteur']} ({r['genre']}) | {dites} | "
                          f"{r['lues']} | {r['minutes_lues']} | {r['jugements']} |")
        if not b["locuteurs"]:
            lignes.append("| — | | | | |")
        versions = " · ".join(f"à {INTERLOCUTEURS[i]} : {n}"
                              for i, n in b["par_interlocuteur"].items())
        lignes += ["", f"**Prises dites** : {versions}", "",
                   f"**Voix enregistrée** : {b['minutes']['dire']} min dites, "
                       f"{b['minutes']['lire']} min lues · **lexique** : {b['lexique']} mots · "
                       f"**phrases à lire disponibles** : {b['phrases_a_lire']}", ""]
        if b["tours"]["total"]:
            t = b["tours"]
            lignes += [f"**Discussions** : {t['total']} tours, dont {t['entendu_corrige']} "
                       f"où le bambara entendu a été corrigé et {t['compris_corrige']} où le "
                       "sens a été corrigé.", ""]
        if b["lots"]:
            lignes += ["### Jugements (versions révélées)", "",
                       "| Lot | Jugés | Préférences | Les deux bonnes | Les deux mauvaises "
                       "| p (test du signe) |", "|---|---|---|---|---|---|"]
            for r in b["lots"]:
                prefs = " ; ".join(f"{s} : {n}" for s, n in r["preferences"].items())
                p = f"{r['p_signe']:.3f}" if "p_signe" in r else "—"
                lignes.append(f"| {r['lot']} | {r['jugements']}/{r['items']} | {prefs} | "
                              f"{r['egal_bon']} | {r['egal_mauvais']} | {p} |")
        return "\n".join(lignes)

    def exporter():
        chemins = atelier.exporter()
        return "✅ Fichiers écrits :\n\n" + "\n".join(f"- `{p}`" for p in chemins.values())

    # --- interface -------------------------------------------------------------

    with gr.Blocks(title="Atelier de données bambara") as demo:
        gr.Markdown("# Atelier de données — assistant vocal bambara\n"
                    "Merci de contribuer ! Ici, **pas besoin d'écrire le bambara** : tu "
                    "parles, tu lis, tu choisis. Tout est sauvegardé à chaque clic ; tu peux "
                    "t'arrêter et reprendre quand tu veux.")
        with gr.Row():
            spk = gr.Dropdown(choix_locuteurs(), value=None, label="Qui es-tu ?", scale=2,
                              min_width=260)
            conditions = gr.Dropdown(list(CONDITIONS), value=CONDITIONS[0],
                                     label="Où enregistres-tu ?", scale=1,
                                     min_width=200)
        with gr.Accordion("Première fois ? Inscris-toi ici", open=False):
            gr.Markdown("Aucun nom n'est demandé : tu reçois un pseudonyme (spk-01…). "
                        "Ta voix ne sera utilisée que pour ce projet, et jamais publiée "
                        "sans ton accord.")
            with gr.Row():
                genre = gr.Radio([("femme", "f"), ("homme", "m")], label="Genre")
                age = gr.Dropdown(["moins de 18", "18-25", "26-40", "41-60", "60+"],
                                  value=None, label="Âge")
                region = gr.Textbox(label="Ville ou région d'origine")
            consent = gr.Checkbox(label="J'accepte que ma voix soit enregistrée et utilisée "
                                        "pour ce projet de recherche (formulaire : "
                                        "docs/COLLECTE.md)")
            public = gr.Checkbox(label="J'accepte aussi qu'elle soit publiée, un jour, dans un "
                                       "corpus ouvert (facultatif)")
            inscrire = gr.Button("M'inscrire", variant="primary")
            inscription_msg = gr.Markdown()

        with gr.Tab("🗣️ Dire") as tab_dire:
            gr.Markdown(AIDE_DIRE)
            with gr.Row():
                a_qui = gr.Radio([(label, cle) for cle, label in INTERLOCUTEURS.items()],
                                 value="ami", label="À qui tu le dis ?", min_width=260)
                rubrique = gr.Dropdown(rubriques_dd(None), value="", label="Rubrique",
                                       min_width=200)
            phrase_dire = gr.State(None)
            texte_dire = gr.Markdown("_Choisis d'abord qui tu es, en haut de la page._")
            micro_dire = gr.Audio(sources=["microphone"], type="numpy", label="Ta voix")
            with gr.Row():
                cs = gr.Checkbox(label="J'ai mélangé avec du français ou de l'anglais")
                argot = gr.Textbox(label="Mots familiers employés (facultatif)",
                                   placeholder="ex. wai")
            ok_dire = gr.Button("✅ Enregistrer et suivante", variant="primary")
            with gr.Row():
                autre_facon = gr.Button("➕ Enregistrer, puis une autre façon de la dire")
                autre_personne = gr.Button("🔁 Enregistrer, puis la dire à quelqu'un d'autre")
            with gr.Row():
                passer_d = gr.Button("⏭️ Passer")
                annuler_d = gr.Button("↩️ Annuler la dernière")
            msg_dire = gr.Markdown()
            prog_dire = gr.HTML()

        with gr.Tab("📖 Lire") as tab_lire:
            gr.Markdown(AIDE_LIRE)
            phrase_lire = gr.State(None)
            texte_lire = gr.Markdown()
            micro_lire = gr.Audio(sources=["microphone"], type="numpy", label="Ta voix")
            with gr.Row():
                ok_lire = gr.Button("✅ Enregistrer et suivante", variant="primary")
                passer_l = gr.Button("⏭️ Passer")
                annuler_l = gr.Button("↩️ Annuler la dernière")
            msg_lire = gr.Markdown()
            prog_lire = gr.HTML()

        with gr.Tab("⚖️ Juger") as tab_juger:
            gr.Markdown(AIDE_JUGER)
            lot = gr.Dropdown(choix_lots(None), value=None, label="Lot à juger")
            item = gr.State(None)
            source = gr.Markdown()
            with gr.Row():
                cand_a = gr.HTML()
                cand_b = gr.HTML()
            with gr.Row():
                btn_a = gr.Button("A est meilleure", variant="primary")
                btn_b = gr.Button("B est meilleure", variant="primary")
            with gr.Row():
                btn_bon = gr.Button("Les deux sont bonnes")
                btn_mauvais = gr.Button("Les deux sont mauvaises")
            prog_juger = gr.HTML()

        with gr.Tab("📚 Lexique") as tab_lexique:
            gr.Markdown(AIDE_LEXIQUE)
            with gr.Row():
                forme = gr.Textbox(label="Mot ou expression", placeholder="wai")
                sens = gr.Textbox(label="Sens en français", placeholder="salut")
                equiv = gr.Textbox(label="Équivalent en bambara courant (facultatif)",
                                   placeholder="i ni ce")
            with gr.Row():
                type_ = gr.Dropdown(list(TYPES_LEXIQUE), value="argot", label="Type")
                exemple = gr.Textbox(label="Exemple de phrase (facultatif)")
            ajouter = gr.Button("Ajouter au lexique", variant="primary")
            msg_lexique = gr.Markdown()
            table = gr.Dataframe(table_lexique(), headers=["mot", "sens", "équivalent",
                                                           "type", "exemple"],
                                 interactive=False, wrap=True)
            with gr.Row():
                a_retirer = gr.Textbox(label="Retirer un mot (tape-le)")
                retirer = gr.Button("Retirer")

        if pipeline is not None:
            with gr.Tab("💬 Discuter"):
                gr.Markdown(AIDE_DISCUTER)
                etat_discuter = gr.State(None)
                micro_discuter = gr.Audio(sources=["microphone"], type="numpy",
                                          label="Parle à l'assistant")
                envoyer = gr.Button("Envoyer", variant="primary")
                etape = gr.Markdown()
                propositions = gr.Radio([], label="Ce que j'ai entendu (propositions)")
                entendu = gr.Textbox(label="Ce que tu as dit, en bambara (corrige les mots faux)")
                with gr.Row():
                    ok_entendu = gr.Button("✅ C'est ce que j'ai dit", variant="primary")
                    inaudible = gr.Button("🔇 Rien de juste dans ce bambara")
                compris_txt = gr.Textbox(label="Ce que ta phrase voulait dire, en français")
                ok_compris = gr.Button("✅ C'est ce que je voulais dire", variant="primary")
                reponse_audio = gr.Audio(label="Réponse de l'assistant", autoplay=True)
                with gr.Row():
                    rep_juste = gr.Button("👍 Réponse juste")
                    rep_fausse = gr.Button("👎 Réponse fausse")

        with gr.Tab("📊 Bilan") as tab_bilan:
            bilan = gr.Markdown(bilan_markdown())
            with gr.Row():
                actualiser = gr.Button("Actualiser")
                exporter_btn = gr.Button("Préparer les fichiers d'export")
            export_msg = gr.Markdown()

        # --- événements --------------------------------------------------------

        inscrire.click(ajouter_locuteur, [genre, age, region, consent, public],
                       [inscription_msg, spk])

        sorties_dire = [msg_dire, micro_dire, phrase_dire, texte_dire, prog_dire, rubrique,
                        cs, argot, a_qui]
        entrees_dire = [spk, rubrique, a_qui, phrase_dire, micro_dire, cs, argot, conditions]
        ok_dire.click(enregistrer_dire, entrees_dire, sorties_dire)
        autre_facon.click(enregistrer_autre_facon, entrees_dire, sorties_dire)
        autre_personne.click(enregistrer_autre_personne, entrees_dire, sorties_dire)
        passer_d.click(passer_dire, [spk, rubrique, a_qui, phrase_dire], sorties_dire)
        annuler_d.click(annuler_dire, [spk, rubrique, a_qui], sorties_dire)
        rubrique.input(montrer_dire, [spk, rubrique, a_qui], [phrase_dire, texte_dire, prog_dire])
        a_qui.input(changer_qui, [spk, rubrique, a_qui, phrase_dire],
                    [phrase_dire, texte_dire, prog_dire, rubrique])

        sorties_lire = [msg_lire, micro_lire, phrase_lire, texte_lire, prog_lire]
        ok_lire.click(enregistrer_lire, [spk, phrase_lire, micro_lire, conditions],
                      sorties_lire)
        passer_l.click(passer_lire, [spk, phrase_lire], sorties_lire)
        annuler_l.click(annuler_lire, [spk], sorties_lire)

        sorties_juger = [item, source, cand_a, cand_b, prog_juger, lot]
        for btn, choix in ((btn_a, "A"), (btn_b, "B"), (btn_bon, "egal_bon"),
                           (btn_mauvais, "egal_mauvais")):
            btn.click(lambda s, l, i, c=choix: juger(s, l, i, c), [spk, lot, item],
                      sorties_juger)
        lot.input(montrer_item, [spk, lot], [item, source, cand_a, cand_b, prog_juger])

        ajouter.click(ajouter_mot, [spk, forme, sens, equiv, type_, exemple],
                      [msg_lexique, table, forme, sens, equiv, exemple])
        retirer.click(retirer_mot, a_retirer, [msg_lexique, table])

        if pipeline is not None:
            envoyer.click(ecouter, [spk, micro_discuter],
                          [propositions, entendu, etape, etat_discuter])
            propositions.input(choisir_proposition, propositions, entendu)
            ok_entendu.click(valider_entendu, [etat_discuter, entendu],
                             [etape, compris_txt, etat_discuter])
            inaudible.click(lambda e, t: valider_entendu(e, t, inaudible=True),
                            [etat_discuter, entendu], [etape, compris_txt, etat_discuter])
            ok_compris.click(valider_compris, [spk, conditions, etat_discuter, compris_txt],
                             [etape, reponse_audio, micro_discuter, etat_discuter, compris_txt])
            rep_juste.click(reponse_juste, [spk, etat_discuter], [etape, etat_discuter])
            rep_fausse.click(reponse_fausse, [spk, etat_discuter], [etape, etat_discuter])

        actualiser.click(bilan_markdown, None, bilan)
        exporter_btn.click(exporter, None, export_msg)

        # Changer de contributeur, ou revenir sur un onglet, remet chaque mode
        # à sa prochaine phrase : les autres contributeurs avancent aussi.
        def tout_rafraichir(s, r, q, l):
            return (*montrer_dire(s, r, q), gr.Dropdown(choices=rubriques_dd(s, q), value=r),
                    *montrer_lire(s)[:3], gr.Dropdown(choices=choix_lots(s), value=l),
                    *montrer_item(s, l))

        tout = [phrase_dire, texte_dire, prog_dire, rubrique, phrase_lire, texte_lire,
                prog_lire, lot, item, source, cand_a, cand_b, prog_juger]
        spk.input(tout_rafraichir, [spk, rubrique, a_qui, lot], tout)
        for tab in (tab_dire, tab_lire, tab_juger):
            tab.select(tout_rafraichir, [spk, rubrique, a_qui, lot], tout)
        tab_lexique.select(table_lexique, None, table)
        tab_bilan.select(bilan_markdown, None, bilan)
        demo.load(lambda: gr.Dropdown(choices=choix_locuteurs()), None, spk)
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--racine", type=Path, default=ROOT / "data" / "atelier" / "collecte",
                   help="dossier de l'atelier (contributeurs, prises, audio/, lots/)")
    p.add_argument("--phrases", type=Path, default=PHRASES_A_DIRE,
                   help="phrases françaises à dire, par rubrique")
    p.add_argument("--share", action="store_true",
                   help="lien public gradio.live (Colab, téléphone)")
    p.add_argument("--port", type=int, default=7862)
    p.add_argument("--config", type=Path, default=None,
                   help="modèles de l'assistant (fichier de déploiement JSON) : ajoute "
                        "l'onglet Discuter, le mode entraînement")
    args = p.parse_args()

    atelier = Atelier(args.racine)
    pipeline = None
    if args.config:
        from bambara_voice.config import build_config
        from bambara_voice.pipeline import VoicePipeline
        pipeline = VoicePipeline(build_config("cascade", args.config))
    demo = build_interface(atelier, charger_phrases_a_dire(args.phrases), pipeline)
    demo.queue().launch(server_name="0.0.0.0" if args.share else "127.0.0.1",
                        server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
