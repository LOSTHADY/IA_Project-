"""Application de collecte du jeu de test — enregistrer, transcrire, valider.

    python app/collect_app.py
    python app/collect_app.py --root data/testset --phrases phrases_lues.txt

Quatre onglets, dans l'ordre d'une séance : enregistrer le locuteur (et son
consentement), enregistrer ses énoncés, transcrire, suivre l'avancement.
Toute la logique est dans `eval/collection.py` ; ce fichier n'est qu'une
façade. Protocole complet : docs/COLLECTE.md.

L'application écoute uniquement sur la machine locale : les voix enregistrées
sont des données personnelles, elles ne transitent par aucun service tiers.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bambara_voice.normalize import has_official_orthography  # noqa: E402
from eval.collection import (  # noqa: E402
    CONDITIONS, STATUSES, Collection, load_prompts, next_prompt, progress,
    read_candidates,
)

ROOT = Path(__file__).resolve().parent.parent

# Clavier d'appoint : ces lettres sont absentes des claviers courants, et une
# orthographe incohérente fausse le WER strict (cf. docs/METHODE.md, §5).
SPECIAL_CHARS = ["ɛ", "ɔ", "ɲ", "ŋ", "Ɛ", "Ɔ", "Ɲ", "Ŋ"]

_INSERT_JS = """
() => {
  const ta = document.querySelector('#bv-transcript textarea');
  if (!ta) return;
  const start = ta.selectionStart ?? ta.value.length;
  const end = ta.selectionEnd ?? start;
  ta.setRangeText('%s', start, end, 'end');
  ta.dispatchEvent(new Event('input', {bubbles: true}));
  ta.focus();
}
"""

STATUS_LABELS = {
    "a_transcrire": "à transcrire",
    "transcrit": "transcrit (à valider)",
    "valide": "validé",
}


def build_interface(col: Collection, consignes: list[dict], extra_sentences: list[str]):
    import gradio as gr

    # --- aides ---------------------------------------------------------------

    def speaker_choices() -> list[tuple[str, str]]:
        return [(f"{s.id} — {s.gender}, {s.age_range or '?'}, {s.region or '?'}", s.id)
                for s in col.speakers() if s.consent]

    def speaker_summary(speaker_id: str | None) -> str:
        if not speaker_id:
            return ""
        recs = [r for r in col.recordings() if r.speaker == speaker_id]
        spont = sum(r.register == "spontane" for r in recs)
        return f"{speaker_id} : {spont} spontanés, {len(recs) - spont} lus"

    def candidates(register: str, speaker_id: str, recordings) -> list[dict]:
        if register == "lu":
            return read_candidates(recordings, speaker_id, extra_sentences)
        return consignes

    def show_prompt(prompt: dict | None, register: str) -> str:
        if prompt is None:
            if register == "lu":
                return ("_Aucune phrase à lire pour ce locuteur. Les phrases lues "
                        "viennent des énoncés spontanés **validés** d'autres "
                        "locuteurs, ou du fichier `--phrases`._")
            return "_Ce locuteur a fait toutes les consignes._"
        if register == "lu":
            return f"### Lisez à voix haute :\n\n# {prompt['consigne']}"
        return (f"### Situation ({prompt['theme']}) :\n\n# {prompt['consigne']}\n\n"
                "_Expliquez la situation au locuteur ; il parle avec ses propres "
                "mots, sans traduire la consigne._")

    def recording_choices(status_filter: str) -> list[tuple[str, str]]:
        out = []
        for r in col.recordings():
            if status_filter != "tous" and r.status != status_filter:
                continue
            out.append((f"{r.id} — {r.speaker}, {r.register}, "
                        f"{STATUS_LABELS[r.status]}", r.id))
        return out

    def progress_markdown() -> str:
        rows = progress(col.speakers(), col.recordings())
        recs = col.recordings()
        lines = ["| | critère | valeur | cible |", "|---|---|---|---|"]
        for row in rows:
            mark = "✅" if row["ok"] else "⏳"
            lines.append(f"| {mark} | {row['critere']} | {row['valeur']} | {row['cible']} |")
        counts = {s: sum(r.status == s for r in recs) for s in STATUSES}
        lines += ["", "**Statuts** : " + " · ".join(
            f"{STATUS_LABELS[s]} : {n}" for s, n in counts.items())]
        return "\n".join(lines)

    # --- 1. locuteur ---------------------------------------------------------

    def add_speaker(gender, age_range, region, first_language, consent,
                    consent_date, public_release, notes):
        try:
            spk = col.add_speaker(
                gender=gender, age_range=age_range or "", region=region.strip(),
                first_language=first_language.strip(), consent=bool(consent),
                consent_date=consent_date.strip(), public_release=bool(public_release),
                notes=notes.strip(),
            )
        except ValueError as exc:
            return f"❌ {exc}", gr.Dropdown()
        msg = (f"✅ Locuteur **{spk.id}** enregistré. Reportez cet identifiant sur "
               f"son formulaire de consentement (jamais son nom dans l'application).")
        return msg, gr.Dropdown(choices=speaker_choices(), value=spk.id)

    # --- 2. enregistrement ---------------------------------------------------

    def new_prompt(speaker_id, register):
        if not speaker_id:
            return None, "_Choisissez d'abord un locuteur._", ""
        recs = col.recordings()
        prompt = next_prompt(candidates(register, speaker_id, recs), recs, speaker_id)
        return prompt, show_prompt(prompt, register), speaker_summary(speaker_id)

    def save(speaker_id, register, conditions, prompt, audio, last):
        error = None
        if not speaker_id:
            error = "❌ Choisissez un locuteur."
        elif prompt is None:
            error = "❌ Affichez d'abord une consigne."
        elif audio is None:
            error = "❌ Aucun audio reçu."
        if error:
            return error, audio, prompt, gr.update(), last, gr.update()
        sample_rate, data = audio
        rec, problems = col.add_recording(
            speaker_id, data, sample_rate, register, prompt, conditions or "")
        msg = f"✅ **{rec.id}** enregistré ({rec.duration_s:.1f} s)."
        if problems:
            msg += ("\n\n⚠️ " + " ; ".join(problems)
                    + "\n\nSi besoin : **Supprimer et réenregistrer**.")
        nxt, text, summary = new_prompt(speaker_id, register)
        return msg, None, nxt, text, rec.id, summary

    def redo(last_id, speaker_id, prompt):
        if not last_id:
            return "Rien à supprimer.", None, prompt, gr.update(), gr.update()
        try:
            rec = col.recording(last_id)
        except KeyError:
            return f"{last_id} n'existe plus.", None, prompt, gr.update(), gr.update()
        if rec.status == "valide":
            # Entre-temps, quelqu'un a pu transcrire et valider cet énoncé.
            return (f"❌ {last_id} est déjà validé : il ne sera pas supprimé.",
                    None, prompt, gr.update(), gr.update())
        col.delete_recording(last_id)
        by_id = {c["id"]: c for c in consignes}
        prompt = by_id.get(rec.prompt_id) or {
            "id": rec.prompt_id, "consigne": rec.prompt, "theme": "lecture"}
        return (f"🗑️ {last_id} supprimé : réenregistrez la même consigne.",
                None, prompt, show_prompt(prompt, rec.register),
                speaker_summary(speaker_id))

    # --- 3. transcription ----------------------------------------------------

    def load_recording(rec_id):
        if not rec_id:
            return None, "", "", "", False, ""
        rec = col.recording(rec_id)
        header = (f"**{rec.id}** · {rec.speaker} · {rec.register} · "
                  f"{rec.duration_s:.1f} s · {STATUS_LABELS[rec.status]}\n\n"
                  f"Consigne : _{rec.prompt}_")
        if rec.transcriber:
            header += f"\n\nTranscrit par : {rec.transcriber}"
        return (str(col.root / rec.audio), header, rec.transcript_bm,
                rec.translation_fr, rec.code_switching, rec.notes)

    def save_transcript(rec_id, status_filter, transcript, translation,
                        code_switching, notes, person, validate):
        if not rec_id:
            return "❌ Choisissez un enregistrement.", gr.update()
        if not person.strip():
            return "❌ Indiquez vos initiales (traçabilité de la transcription).", gr.update()
        changes = dict(transcript_bm=transcript, translation_fr=translation,
                       code_switching=bool(code_switching), notes=notes.strip())
        if validate:
            changes.update(status="valide", validator=person.strip())
        else:
            changes.update(status="transcrit", transcriber=person.strip())
        try:
            rec = col.update_recording(rec_id, **changes)
        except ValueError as exc:
            return f"❌ {exc}", gr.update()

        msg = f"✅ {rec.id} : {STATUS_LABELS[rec.status]}."
        if len(rec.transcript_bm.split()) >= 4 and not has_official_orthography(rec.transcript_bm):
            msg += ("\n\n⚠️ Aucun ɛ, ɔ, ɲ ou ŋ : vérifier que la transcription "
                    "suit bien l'orthographe officielle.")
        if not rec.translation_fr:
            msg += "\n\n⚠️ Pas de traduction française : l'item ne servira pas en bout-en-bout."
        # Passer au suivant dans la liste filtrée (les identifiants bv-NNNN
        # se trient dans l'ordre d'enregistrement).
        choices = recording_choices(status_filter)
        ids = [c[1] for c in choices if c[1] != rec_id]
        nxt = next((i for i in ids if i > rec_id), ids[0] if ids else None)
        return msg, gr.Dropdown(choices=choices, value=nxt)

    # --- interface -----------------------------------------------------------

    with gr.Blocks(title="Collecte — jeu de test bambara") as demo:
        gr.Markdown("# Collecte du jeu de test bambara\n"
                    "Protocole : `docs/COLLECTE.md`. Les fichiers sont écrits dans "
                    f"`{col.root}`.")

        with gr.Tab("1. Locuteur"):
            with gr.Row():
                gender = gr.Radio([("femme", "f"), ("homme", "m")], label="Genre")
                # value=None explicite : sinon Gradio présélectionne le premier
                # choix, et une tranche d'âge non renseignée passerait pour vraie.
                age = gr.Dropdown(["18-25", "26-40", "41-60", "60+"], value=None,
                                  label="Âge")
                region = gr.Textbox(label="Région / ville d'origine")
                first_lang = gr.Textbox(value="bambara", label="Langue première")
            consent = gr.Checkbox(label="Le formulaire de consentement est signé "
                                        "(ou le consentement oral attesté par un témoin)")
            with gr.Row():
                consent_date = gr.Textbox(value=date.today().isoformat(),
                                          label="Date du consentement")
                public = gr.Checkbox(label="Accepte la diffusion publique de ses enregistrements")
            spk_notes = gr.Textbox(label="Notes (sans nom ni information identifiante)")
            add_btn = gr.Button("Enregistrer le locuteur", variant="primary")
            spk_msg = gr.Markdown()

        with gr.Tab("2. Enregistrer") as record_tab:
            with gr.Row():
                spk = gr.Dropdown(speaker_choices(), value=None, label="Locuteur")
                register = gr.Radio([("spontané", "spontane"), ("lu", "lu")],
                                    value="spontane", label="Registre")
                conditions = gr.Dropdown(list(CONDITIONS), value=CONDITIONS[0],
                                         label="Conditions")
            spk_count = gr.Markdown()
            prompt_state = gr.State(None)
            last_id = gr.State(None)
            next_btn = gr.Button("Consigne suivante")
            prompt_md = gr.Markdown("_Choisissez un locuteur puis « Consigne suivante »._")
            mic = gr.Audio(sources=["microphone", "upload"], type="numpy",
                           label="Énoncé")
            with gr.Row():
                save_btn = gr.Button("Enregistrer cet énoncé", variant="primary")
                redo_btn = gr.Button("Supprimer et réenregistrer")
            rec_msg = gr.Markdown()

        with gr.Tab("3. Transcrire et valider") as transcribe_tab:
            with gr.Row():
                status_filter = gr.Radio(
                    [(STATUS_LABELS[s], s) for s in STATUSES] + [("tous", "tous")],
                    value="a_transcrire", label="Afficher")
                rec_dd = gr.Dropdown(recording_choices("a_transcrire"), value=None,
                                     label="Enregistrement")
            player = gr.Audio(type="filepath", interactive=False, label="Écouter")
            header = gr.Markdown()
            transcript = gr.Textbox(label="Transcription bambara (orthographe officielle, "
                                          "sans tons)", elem_id="bv-transcript", lines=2)
            with gr.Row():
                char_btns = [gr.Button(ch, size="sm", min_width=40) for ch in SPECIAL_CHARS]
            translation = gr.Textbox(label="Traduction française (fidèle, pas élégante)",
                                     lines=2)
            code_switching = gr.Checkbox(label="Contient des mots français (code-switching)")
            tr_notes = gr.Textbox(label="Notes (bruit, hésitation, doute...)")
            person = gr.Textbox(label="Vos initiales")
            with gr.Row():
                tr_btn = gr.Button("Enregistrer la transcription", variant="primary")
                val_btn = gr.Button("Valider (relecture par une 2ᵉ personne)")
            tr_msg = gr.Markdown()

        with gr.Tab("4. Avancement") as progress_tab:
            progress_md = gr.Markdown(progress_markdown())
            refresh_btn = gr.Button("Actualiser")

        # --- événements --------------------------------------------------------

        add_btn.click(add_speaker,
                      [gender, age, region, first_lang, consent, consent_date, public, spk_notes],
                      [spk_msg, spk])

        for trigger in (next_btn.click, spk.change, register.change):
            trigger(new_prompt, [spk, register], [prompt_state, prompt_md, spk_count])
        save_btn.click(save, [spk, register, conditions, prompt_state, mic, last_id],
                       [rec_msg, mic, prompt_state, prompt_md, last_id, spk_count])
        redo_btn.click(redo, [last_id, spk, prompt_state],
                       [rec_msg, last_id, prompt_state, prompt_md, spk_count])

        status_filter.change(lambda s: gr.Dropdown(choices=recording_choices(s), value=None),
                             status_filter, rec_dd)
        rec_dd.change(load_recording, rec_dd,
                      [player, header, transcript, translation, code_switching, tr_notes])
        transcribe_tab.select(lambda s: gr.Dropdown(choices=recording_choices(s)),
                              status_filter, rec_dd)
        for ch, btn in zip(SPECIAL_CHARS, char_btns):
            btn.click(None, js=_INSERT_JS % ch)
        tr_inputs = [rec_dd, status_filter, transcript, translation, code_switching,
                     tr_notes, person]
        tr_btn.click(lambda *a: save_transcript(*a, validate=False), tr_inputs,
                     [tr_msg, rec_dd])
        val_btn.click(lambda *a: save_transcript(*a, validate=True), tr_inputs,
                      [tr_msg, rec_dd])

        refresh_btn.click(progress_markdown, None, progress_md)
        progress_tab.select(progress_markdown, None, progress_md)

        # Les listes sont calculées à la construction de l'interface : on les
        # relit à chaque chargement de page et à l'ouverture de l'onglet, sinon
        # une page rechargée ignore les locuteurs ajoutés entre-temps.
        def refresh_speakers():
            return gr.Dropdown(choices=speaker_choices())

        record_tab.select(refresh_speakers, None, spk)
        demo.load(refresh_speakers, None, spk)
        demo.load(lambda s: gr.Dropdown(choices=recording_choices(s)), status_filter, rec_dd)
        demo.load(progress_markdown, None, progress_md)
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT / "data" / "testset",
                   help="dossier de collecte (locuteurs, collecte.jsonl, audio/)")
    p.add_argument("--consignes", type=Path,
                   default=ROOT / "data" / "testset" / "consignes.json")
    p.add_argument("--phrases", type=Path, default=None,
                   help="phrases bambara à faire lire, une par ligne, "
                        "validées par un locuteur natif")
    p.add_argument("--port", type=int, default=7861)
    args = p.parse_args()

    extra = []
    if args.phrases:
        extra = args.phrases.read_text(encoding="utf-8").splitlines()
    col = Collection(args.root)
    demo = build_interface(col, load_prompts(args.consignes), extra)
    demo.launch(server_name="127.0.0.1", server_port=args.port,
                allowed_paths=[str(col.root.resolve())])


if __name__ == "__main__":
    main()
