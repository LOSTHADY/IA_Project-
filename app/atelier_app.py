"""Atelier de données — dire, lire, juger, enrichir le lexique.

    python app/atelier_app.py                        # sur cet ordinateur
    python app/atelier_app.py --racine /content/drive/MyDrive/bambara/atelier --share

Pour contribuer sans rien transcrire : on dit en bambara une phrase affichée
en français, on lit une phrase bambara, on choisit la meilleure de deux
traductions, on note des mots familiers. Chacun reprend là où il s'était
arrêté ; plusieurs contributeurs peuvent travailler en même temps.

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
    LECTURE_CIBLE_MIN, PHRASES_A_DIRE, TYPES_LEXIQUE, Atelier,
    charger_phrases_a_dire, rubriques,
)
from eval.collection import CONDITIONS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

AIDE_DIRE = """\
**Comment faire.** Lis la phrase française, puis dis-la **en bambara, comme tu
la dirais à un ami** : avec tes mots, pas mot à mot. Si tu mélanges avec du
français ou de l'anglais, c'est très bien : coche la case. Appuie sur le micro,
parle, arrête, écoute si tu veux, puis « Enregistrer et suivante ».
Une phrase qui ne te parle pas : « Passer »."""

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


def build_interface(atelier: Atelier, phrases_dire: list[dict]):
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

    def rubriques_dd(spk):
        prog = {r["rubrique"]: r for r in atelier.progression("dire", spk, phrases_dire)} \
            if spk else {}
        choix = [("Toutes, dans l'ordre", "")]
        for rid, titre in titres.items():
            r = prog.get(rid)
            choix.append((f"{titre} — {r['faites']}/{r['total']}" if r else titre, rid))
        return choix

    def progression_dire(spk, courante=None):
        """Le total et la rubrique en cours ; le détail, repliable."""
        if not spk:
            return ""
        prog = atelier.progression("dire", spk, phrases_dire)
        total = sum(r["total"] for r in prog)
        faites = sum(r["faites"] for r in prog)
        lignes = [barre(faites, total, "Au total :")]
        lignes += [barre(r["faites"], r["total"], r["titre"] + " :")
                   for r in prog if r["rubrique"] == courante]
        detail = "".join(barre(r["faites"], r["total"], r["titre"]) for r in prog)
        lignes.append(f'<details style="margin-top:6px"><summary>Progression par rubrique'
                      f'</summary>{detail}</details>')
        return "".join(lignes)

    def montrer_dire(spk, rubrique):
        if not spk:
            return None, "_Choisis d'abord qui tu es, en haut de la page._", ""
        p = atelier.prochaine("dire", spk, phrases_dire, rubrique or None)
        if p is None:
            fin = "cette rubrique" if rubrique else "toutes les phrases"
            return None, f"🎉 Tu as terminé {fin}. Merci ! Choisis une autre rubrique.", \
                progression_dire(spk)
        rang = [q["id"] for q in phrases_dire if q["rubrique"] == p["rubrique"]].index(p["id"]) + 1
        n = sum(q["rubrique"] == p["rubrique"] for q in phrases_dire)
        texte = citation(p["fr"], f"{p['titre']} · phrase {rang} sur {n} · à dire en bambara")
        return p, texte, progression_dire(spk, p["rubrique"])

    def apres_dire(spk, rubrique, msg):
        p, texte, prog = montrer_dire(spk, rubrique)
        return (msg, None, p, texte, prog,
                gr.Dropdown(choices=rubriques_dd(spk), value=rubrique), False, "")

    def enregistrer_dire(spk, rubrique, phrase, audio, cs, argot, conditions):
        if not spk or not phrase:
            return apres_dire(spk, rubrique, "⚠️ Choisis qui tu es, puis une phrase.")
        if audio is None:
            return (("⚠️ Rien d'enregistré : appuie sur le micro, parle, puis arrête.",)
                    + (gr.update(),) * 7)
        sr, data = audio
        try:
            prise, problemes = atelier.enregistrer("dire", spk, phrase, data, sr, cs, argot,
                                                   conditions)
        except ValueError as exc:
            return (f"⚠️ {exc}.",) + (gr.update(),) * 7
        msg = f"✅ Enregistré ({prise.duration_s:.1f} s)."
        if problemes:
            msg += " ⚠️ " + " ; ".join(problemes) + " — « Annuler la dernière » pour recommencer."
        return apres_dire(spk, rubrique, msg)

    def passer_dire(spk, rubrique, phrase):
        if spk and phrase:
            atelier.passer("dire", spk, phrase["id"])
        return apres_dire(spk, rubrique, "⏭️ Phrase passée.")

    def annuler_dire(spk, rubrique):
        prise = atelier.annuler_derniere("dire", spk) if spk else None
        msg = (f"↩️ Dernière prise supprimée : « {prise.texte} » revient."
               if prise else "Rien à annuler.")
        return apres_dire(spk, rubrique, msg)

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

    def bilan_markdown():
        b = atelier.bilan(phrases_dire)
        lignes = ["### Contributeurs", "",
                  "| Contributeur | Phrases dites | Phrases lues | Minutes lues | Jugements |",
                  "|---|---|---|---|---|"]
        for r in b["locuteurs"]:
            lignes.append(f"| {r['locuteur']} ({r['genre']}) | {r['dites']}/{r['a_dire']} | "
                          f"{r['lues']} | {r['minutes_lues']} | {r['jugements']} |")
        if not b["locuteurs"]:
            lignes.append("| — | | | | |")
        lignes += ["", f"**Voix enregistrée** : {b['minutes']['dire']} min dites, "
                       f"{b['minutes']['lire']} min lues · **lexique** : {b['lexique']} mots · "
                       f"**phrases à lire disponibles** : {b['phrases_a_lire']}", ""]
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
            rubrique = gr.Dropdown(rubriques_dd(None), value="", label="Rubrique")
            phrase_dire = gr.State(None)
            texte_dire = gr.Markdown("_Choisis d'abord qui tu es, en haut de la page._")
            micro_dire = gr.Audio(sources=["microphone"], type="numpy", label="Ta voix")
            with gr.Row():
                cs = gr.Checkbox(label="J'ai mélangé avec du français ou de l'anglais")
                argot = gr.Textbox(label="Mots familiers employés (facultatif)",
                                   placeholder="ex. wai")
            with gr.Row():
                ok_dire = gr.Button("✅ Enregistrer et suivante", variant="primary")
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
                        cs, argot]
        ok_dire.click(enregistrer_dire,
                      [spk, rubrique, phrase_dire, micro_dire, cs, argot, conditions],
                      sorties_dire)
        passer_d.click(passer_dire, [spk, rubrique, phrase_dire], sorties_dire)
        annuler_d.click(annuler_dire, [spk, rubrique], sorties_dire)
        rubrique.input(montrer_dire, [spk, rubrique], [phrase_dire, texte_dire, prog_dire])

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

        actualiser.click(bilan_markdown, None, bilan)
        exporter_btn.click(exporter, None, export_msg)

        # Changer de contributeur, ou revenir sur un onglet, remet chaque mode
        # à sa prochaine phrase : les autres contributeurs avancent aussi.
        def tout_rafraichir(s, r, l):
            return (*montrer_dire(s, r), gr.Dropdown(choices=rubriques_dd(s), value=r),
                    *montrer_lire(s)[:3], gr.Dropdown(choices=choix_lots(s), value=l),
                    *montrer_item(s, l))

        tout = [phrase_dire, texte_dire, prog_dire, rubrique, phrase_lire, texte_lire,
                prog_lire, lot, item, source, cand_a, cand_b, prog_juger]
        spk.input(tout_rafraichir, [spk, rubrique, lot], tout)
        for tab in (tab_dire, tab_lire, tab_juger):
            tab.select(tout_rafraichir, [spk, rubrique, lot], tout)
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
    args = p.parse_args()

    atelier = Atelier(args.racine)
    demo = build_interface(atelier, charger_phrases_a_dire(args.phrases))
    demo.queue().launch(server_name="0.0.0.0" if args.share else "127.0.0.1",
                        server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
