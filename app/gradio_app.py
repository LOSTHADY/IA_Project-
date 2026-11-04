"""Démo Gradio — interface de soutenance.

    python app/gradio_app.py --arch cascade

Push-to-talk assumé : la chaîne compte quatre modèles en série, le duplex
temps réel n'est pas l'objectif. Le panneau de droite affiche la trace
complète du tour (texte intermédiaire, source de la réponse, temps par étape)
— c'est ce qui rend la démo *démonstrative* plutôt que magique.

Quand l'assistant a mal compris, le locuteur écrit ce qu'il voulait dire et
clique « Retiens-le » : la phrase entre dans la mémoire (bambara_voice.memoire)
et sera comprise la prochaine fois qu'elle sera dite, sans réentraînement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bambara_voice.config import build_config  # noqa: E402
from bambara_voice.pipeline import VoicePipeline  # noqa: E402


def _messages_format(gr) -> dict:
    """Gradio 4-5 attend type="messages" ; Gradio 6 a retiré le paramètre,
    ce format y étant le seul."""
    import inspect

    if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
        return {"type": "messages"}
    return {}


def retenir(pipeline: VoicePipeline, trace, voulu: str) -> str:
    """« Retiens-le » : le dernier tour voulait dire `voulu`."""
    if trace is None or not trace.source_bm.strip():
        return "Rien à retenir : dis d'abord une phrase."
    if not (voulu or "").strip():
        return "Écris en français ce que tu voulais dire."
    s = pipeline.apprendre(trace, voulu)
    return (f"Retenu : « {s.bm} » veut dire « {s.fr} ». "
            f"La mémoire compte {len(pipeline.memoire)} phrases.")


def build_interface(pipeline: VoicePipeline):
    import gradio as gr

    def respond(audio_path, history):
        if not audio_path:
            return history, None, "Aucun audio reçu.", None
        trace, speech = pipeline.run(audio_path)

        heard = trace.source_bm or trace.asr_text
        history = (history or []) + [
            {"role": "user", "content": heard},
            {"role": "assistant", "content": trace.reply_bm},
        ]

        out_path = None
        if speech is not None and len(speech.audio):
            out_path = str(Path(gr.utils.abspath(".")) / "reponse.wav")
            speech.save(out_path)

        # Deux espaces en fin de ligne : saut de ligne Markdown, sans quoi
        # tout le panneau s'affiche en un seul paragraphe.
        details = "  \n".join([
            f"**Architecture** : {trace.architecture}",
            f"**Entendu (bm)** : {trace.source_bm or '—'}",
            f"**Compris (fr)** : {trace.source_fr}"
            + (f" (mémoire, score {trace.memoire_score})" if trace.compris_par == "memoire" else ""),
            f"**Réponse (fr)** : {trace.reply_fr}",
            f"**Réponse (bm)** : {trace.reply_bm}",
            f"**Source** : {trace.reply_source}"
            + (f" ({trace.template_id}, score {trace.template_score})"
               if trace.template_id else ""),
            "\n**Temps par étape (s)**\n",
            *[f"- {k} : {v:.2f}" for k, v in trace.timings.items()],
            f"- **total** : {trace.total_seconds:.2f} (RTF {trace.rtf:.2f})",
        ])
        return history, out_path, details, trace

    with gr.Blocks(title="Assistant vocal bambara") as demo:
        gr.Markdown(
            "# Assistant vocal bambara\n"
            "Parlez en bambara, l'assistant répond en bambara. "
            "Le panneau de droite montre ce qui se passe à chaque maillon."
        )
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Conversation", height=380, **_messages_format(gr))
                mic = gr.Audio(sources=["microphone", "upload"], type="filepath",
                               label="Appuyez pour parler")
                player = gr.Audio(label="Réponse audio", autoplay=True)
                gr.ClearButton([chat, mic, player], value="Effacer")
            with gr.Column(scale=2):
                details = gr.Markdown("_En attente d'un énoncé._")
                voulu = gr.Textbox(label="Mal compris ? Écris ce que tu voulais dire (en français)")
                bouton = gr.Button("Retiens-le")
                retenu = gr.Markdown()
        dernier = gr.State(None)

        mic.stop_recording(respond, [mic, chat], [chat, player, details, dernier])
        mic.upload(respond, [mic, chat], [chat, player, details, dernier])
        bouton.click(lambda trace, texte: retenir(pipeline, trace, texte), [dernier, voulu], [retenu])
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=["cascade", "e2e"], default="cascade")
    p.add_argument("--config", default=None,
                   help="fichier de déploiement JSON (cf. scripts/export_cpu.py)")
    p.add_argument("--llm-backend", choices=["transformers", "llamacpp", "echo"],
                   default=None)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    config = build_config(args.arch, args.config)
    if args.llm_backend:
        config.llm.backend = args.llm_backend
    build_interface(VoicePipeline(config)).launch(share=args.share)


if __name__ == "__main__":
    main()
