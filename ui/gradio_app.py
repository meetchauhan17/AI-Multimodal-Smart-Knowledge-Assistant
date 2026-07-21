"""
ui/gradio_app.py — Gradio User Interface for the AI Multimodal Smart Knowledge Assistant.

Ties all backend modules together under a tabbed blocks layout.
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr

from core.logger import logger
from core.assistant import MultimodalAssistant

# Domain dropdown mapping
DOMAINS_MAP = {
    "Auto-detect": None,
    "College": "college",
    "Tourism": "tourism",
    "Healthcare": "healthcare",
    "Agriculture": "agriculture",
    "Library": "library",
    "Museums": "museums",
    "Historical Monuments": "monuments"
}
DOMAIN_CHOICES = list(DOMAINS_MAP.keys())


# ── Module-level singleton ────────────────────────────────────────────────────
# Instantiate once so the Whisper model (and other expensive resources) are
# loaded into memory only on first use, then reused for every subsequent call.
_assistant: MultimodalAssistant | None = None


def _get_assistant() -> MultimodalAssistant:
    """Lazy-init the shared assistant instance."""
    global _assistant
    if _assistant is None:
        logger.info("[UI] Lazy-initializing MultimodalAssistant singleton...")
        _assistant = MultimodalAssistant()
    return _assistant


def handle_ask(question: str, audio_path: str | None, domain_label: str) -> tuple[str, str, str | None, str, str]:
    """Handle text or audio queries on Tab 1.
    
    Returns:
        (question_textbox_update, text_answer_output, audio_answer_path, sources_output, provider_output)
    """
    logger.info(f"[UI] Query submitted. Text: {question!r}, Audio: {audio_path!r}, Domain: {domain_label!r}")
    
    assistant = _get_assistant()
    domain = DOMAINS_MAP.get(domain_label)

    # Prefer voice input if provided
    if audio_path:
        res = assistant.ask_voice(audio_path, domain=domain)
        transcribed = res.get("transcribed_question", "")
        answer = res.get("answer", "")
        audio_out = res.get("answer_audio_path")
        sources = res.get("sources", [])
        provider = res.get("provider_used", "")
        err = res.get("error")

        if err:
            return transcribed, f"Error: {err}", None, "None", ""

        sources_str = ", ".join(sources) if sources else "None"
        return transcribed, answer, audio_out, sources_str, provider
    else:
        if not question.strip():
            return "", "Please write a question or record voice audio.", None, "None", ""

        res = assistant.ask_text(question, domain=domain)
        answer = res.get("answer", "")
        sources = res.get("sources", [])
        provider = res.get("provider_used", "")
        err = res.get("error")

        if err:
            return question, f"Error: {err}", None, "None", ""

        # Generate speech synthesis for the text answer
        audio_out = None
        if answer and not answer.startswith("Error"):
            try:
                from modules.text_to_speech import speak_text
                audio_out = speak_text(answer)
            except Exception as e:
                logger.warning(f"[UI] TTS Synthesis failed for answer: {e}")

        sources_str = ", ".join(sources) if sources else "None"
        return question, answer, audio_out, sources_str, provider


def handle_caption(image_path: str | None, style: str) -> tuple[str, str, str, str]:
    """Handle image captioning on Tab 2.
    
    Returns:
        (caption_output, provider_output, latency_output, error_output)
    """
    logger.info(f"[UI] Image captioning request. Style: {style!r}")
    
    if not image_path:
        return "", "", "0.0ms", "Please upload an image."

    assistant = _get_assistant()
    res = assistant.caption_uploaded_image(image_path, style=style, follow_up=False)

    if res.get("error"):
        return "", "", f"{res['latency_ms']:.1f}ms", f"Error: {res['error']}"

    return res["caption"], res["provider_used"], f"{res['latency_ms']:.1f}ms", ""


def handle_caption_followup(caption: str) -> tuple[str, str]:
    """Handle RAG follow-up for the generated image caption.
    
    Returns:
        (follow_up_answer_output, follow_up_sources_output)
    """
    logger.info(f"[UI] RAG follow-up requested for caption: {caption!r}")
    
    if not caption.strip():
        return "No caption text to search.", "None"

    assistant = _get_assistant()
    query = f"Tell me more about this: {caption}"
    res = assistant.ask_text(query, domain=None)

    sources = ", ".join(res.get("sources", [])) if res.get("sources") else "None"
    return res.get("answer", ""), sources


def handle_image_gen(prompt: str) -> tuple[str | None, str, str]:
    """Handle text-to-image generation on Tab 3.
    
    Returns:
        (image_output_path, provider_output, error_output)
    """
    logger.info(f"[UI] Image generation request. Prompt: {prompt!r}")
    
    if not prompt.strip():
        return None, "", "Prompt cannot be empty."

    assistant = _get_assistant()
    res = assistant.generate_image_from_prompt(prompt)

    if res.get("error"):
        return None, res.get("provider_used", ""), f"Error: {res['error']}"

    return res.get("image_path"), res.get("provider_used", ""), ""


# ── Premium Theme Configuration ───────────────────────────────────────────────
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Outfit")
)


def build_app() -> gr.Blocks:
    # NOTE: In Gradio 4.x, theme was in gr.Blocks().
    # In Gradio 6.x, it moved back to launch(). Keeping it here for compatibility
    # with whichever version is installed — gradio will warn but still apply it.
    with gr.Blocks(title="AI Multimodal Assistant") as app:
        
        # Premium Header Panel
        gr.HTML(
            """
            <div style="text-align: center; margin-bottom: 25px; padding: 20px; 
                        background: linear-gradient(135deg, #4f46e5, #06b6d4); 
                        border-radius: 14px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
                <h1 style="margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.5px;">🤖 AI Multimodal Smart Knowledge Assistant</h1>
                <p style="margin: 6px 0 0 0; font-size: 1.1rem; opacity: 0.9; font-weight: 400;">
                    Capstone Project — Knowledge Retrieval & Multimodal Synthesizers
                </p>
            </div>
            """
        )

        with gr.Tabs():
            
            # ── Tab 1: Ask (Text & Voice) ─────────────────────────────────────
            with gr.Tab("💬 Ask (Text & Voice)"):
                gr.Markdown("### Submit questions via text query or recorded microphone voice.")
                with gr.Row():
                    with gr.Column(scale=1):
                        text_in = gr.Textbox(
                            lines=3,
                            placeholder="Type your question here...",
                            label="Typed Question"
                        )
                        audio_in = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Voice Recording"
                        )
                        domain_dd = gr.Dropdown(
                            choices=DOMAIN_CHOICES,
                            value="Auto-detect",
                            label="Search Domain Filters"
                        )
                        ask_btn = gr.Button("Submit Question", variant="primary")
                    
                    with gr.Column(scale=1):
                        text_out = gr.Textbox(
                            lines=6,
                            label="Answer Text",
                            interactive=False
                        )
                        audio_out = gr.Audio(
                            label="Voice Answer Spoken",
                            interactive=False
                        )
                        
                        with gr.Row():
                            provider_out = gr.Textbox(
                                label="LLM Provider",
                                interactive=False
                            )
                            sources_out = gr.Textbox(
                                label="Documents Used",
                                interactive=False
                            )

                # Bind Tab 1 event
                ask_btn.click(
                    fn=handle_ask,
                    inputs=[text_in, audio_in, domain_dd],
                    outputs=[text_in, text_out, audio_out, sources_out, provider_out],
                    api_name="handle_ask"
                )

            # ── Tab 2: Image Captioning ───────────────────────────────────────
            with gr.Tab("🖼️ Image Captioning"):
                gr.Markdown("### Upload an image and generate natural language descriptions.")
                with gr.Row():
                    with gr.Column(scale=1):
                        img_in = gr.Image(
                            type="filepath",
                            label="Upload Image File"
                        )
                        style_radio = gr.Radio(
                            choices=["descriptive", "short", "detailed"],
                            value="descriptive",
                            label="Description Style"
                        )
                        caption_btn = gr.Button("Generate Caption", variant="primary")

                    with gr.Column(scale=1):
                        caption_out = gr.Textbox(
                            label="Caption Result",
                            interactive=False
                        )
                        with gr.Row():
                            vision_provider = gr.Textbox(
                                label="Vision Provider",
                                interactive=False
                            )
                            vision_latency = gr.Textbox(
                                label="Latency",
                                interactive=False
                            )
                        vision_error = gr.Textbox(
                            label="Error Trace (if any)",
                            interactive=False,
                            visible=True
                        )

                        # Follow up panel
                        gr.Markdown("---")
                        gr.Markdown("#### Follow-up Analysis")
                        followup_btn = gr.Button("Learn More about this Caption via RAG")
                        followup_ans = gr.Textbox(
                            lines=4,
                            label="RAG Context Analysis",
                            interactive=False
                        )
                        followup_sources = gr.Textbox(
                            label="RAG Sources",
                            interactive=False
                        )

                # Bind Tab 2 Events
                caption_btn.click(
                    fn=handle_caption,
                    inputs=[img_in, style_radio],
                    outputs=[caption_out, vision_provider, vision_latency, vision_error],
                    api_name="handle_caption"
                )
                followup_btn.click(
                    fn=handle_caption_followup,
                    inputs=[caption_out],
                    outputs=[followup_ans, followup_sources],
                    api_name="handle_caption_followup"
                )

            # ── Tab 3: Image Generation ───────────────────────────────────────
            with gr.Tab("🎨 Image Generation"):
                gr.Markdown("### Generate dynamic artwork from descriptive text prompts.")
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_in = gr.Textbox(
                            lines=3,
                            placeholder="E.g. A gorgeous watercolor painting of a red rose on a desk...",
                            label="Image Prompt"
                        )
                        gen_btn = gr.Button("Generate Image", variant="primary")
                    
                    with gr.Column(scale=1):
                        gen_img_out = gr.Image(
                            label="Generated Image",
                            interactive=False
                        )
                        gen_provider = gr.Textbox(
                            label="Gen Provider",
                            interactive=False
                        )
                        gen_error = gr.Textbox(
                            label="Error Trace (if any)",
                            interactive=False
                        )

                # Bind Tab 3 event
                gen_btn.click(
                    fn=handle_image_gen,
                    inputs=[prompt_in],
                    outputs=[gen_img_out, gen_provider, gen_error],
                    api_name="handle_image_gen"
                )

    return app


if __name__ == "__main__":
    import uvicorn
    from pathlib import Path
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware

    logger.info("Starting AI Multimodal Smart Knowledge Assistant on http://127.0.0.1:7871...")

    server = FastAPI(title="AI Multimodal Smart Knowledge Assistant")

    server.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 1. Mount Gradio API backend at /gradio
    blocks = build_app()
    server = gr.mount_gradio_app(
        server,
        blocks,
        path="/gradio",
        app_kwargs={"default_config": blocks.config},
    )

    # 2. Serve static web interface directly at root /
    notebook_dir = Path(__file__).parent / "web_modern"
    if notebook_dir.is_dir():
        server.mount(
            "/",
            StaticFiles(directory=str(notebook_dir), html=True),
            name="web_ui",
        )
        logger.info(f"Web interface mounted directly at / from {notebook_dir}")

    logger.info("═══════════════════════════════════════════════════════════")
    logger.info("  AI Multimodal Smart Knowledge Assistant  →  http://127.0.0.1:7871")
    logger.info("═══════════════════════════════════════════════════════════")

    uvicorn.run(server, host="127.0.0.1", port=7871)
