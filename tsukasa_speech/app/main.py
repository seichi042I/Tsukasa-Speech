"""Application entry point: builds the full Gradio app with both tabs."""

import gradio as gr

from tsukasa_speech.app.inference_tab import build_inference_tab
from tsukasa_speech.app.training_tab import build_training_tab


def create_app():
    """Create and return the Gradio Blocks application."""
    with gr.Blocks(title="Tsukasa Speech") as app:
        gr.Markdown("# Tsukasa Speech")
        with gr.Tabs():
            with gr.Tab("音声合成"):
                build_inference_tab()
            with gr.Tab("学習"):
                build_training_tab()
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = create_app()
    app.launch(server_name=args.host, server_port=args.port)
