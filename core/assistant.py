"""
core/assistant.py — Coherent orchestrator class tying all multimodal modules together.

Provides a unified interface (MultimodalAssistant) for:
- Speech-to-Text (transcribe_audio)
- RAG Q&A (answer_with_rag)
- Image Captioning (caption_image)
- Image Generation (generate_image)
- Text-to-Speech (speak_text)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, List

from core.logger import logger
from modules.speech_to_text import transcribe_audio
from modules.text_to_speech import speak_text
from modules.image_captioning import caption_image
from modules.image_generation import generate_image
from modules.rag.retriever import answer_with_rag


class MultimodalAssistant:
    """Ties every system module into one orchestrator for the Gradio UI and CLI."""

    def __init__(self) -> None:
        logger.info("Initializing MultimodalAssistant orchestrator...")

    def ask_text(self, question: str, domain: str | None = None) -> Dict[str, Any]:
        """Answer a text question using RAG over the knowledge base.

        Args:
            question: The natural-language query.
            domain:   Target domain collection, or None to auto-detect.

        Returns:
            Dictionary with keys: {answer, sources, domain_detected, provider_used, error}
        """
        logger.info(f"[STAGE 2/3: RAG] Processing text query: {question!r} (domain: {domain})")
        start_time = time.perf_counter()
        
        try:
            result = answer_with_rag(question, domain=domain)
            latency = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                f"[STAGE 2/3: RAG Complete] Domain detected: {result['domain_detected']!r} "
                f"using {result['provider_used']!r} in {latency:.2f}ms"
            )
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "domain_detected": result["domain_detected"],
                "provider_used": result["provider_used"],
                "error": None
            }
        except Exception as exc:
            err_msg = f"RAG query pipeline failed: {exc}"
            logger.exception(err_msg)
            return {
                "answer": "",
                "sources": [],
                "domain_detected": domain or "all",
                "provider_used": "",
                "error": err_msg
            }

    def ask_voice(self, audio_filepath: str | Path, domain: str | None = None) -> Dict[str, Any]:
        """Transcribe spoken audio, query the RAG pipeline, and synthesize the answer to speech.

        Args:
            audio_filepath: Path to the input voice recording (.mp3, .wav, etc.).
            domain:         Optional domain collection filter.

        Returns:
            Dictionary with ask_text result fields plus:
                - transcribed_question: The text transcription of the audio.
                - answer_audio_path: Path to the generated spoken response file.
        """
        logger.info(f"[STAGE 1: STT] Transcribing voice input from: {audio_filepath}")
        stt_start = time.perf_counter()
        
        # 1. Transcribe audio to text
        stt_res = transcribe_audio(audio_filepath)
        stt_latency = (time.perf_counter() - stt_start) * 1000.0
        
        if stt_res.get("error"):
            logger.warning(f"[STAGE 1 Failed] STT Transcription failed: {stt_res['error']}")
            return {
                "answer": "",
                "sources": [],
                "domain_detected": domain or "all",
                "provider_used": "",
                "transcribed_question": "",
                "answer_audio_path": "",
                "error": stt_res["error"]
            }

        transcribed_text = stt_res["text"]
        logger.info(f"[STAGE 1 Complete] Transcribed text: {transcribed_text!r} in {stt_latency:.2f}ms")

        # 2. Query RAG
        rag_res = self.ask_text(transcribed_text, domain=domain)
        
        if rag_res.get("error"):
            return {
                "answer": "",
                "sources": [],
                "domain_detected": rag_res["domain_detected"],
                "provider_used": "",
                "transcribed_question": transcribed_text,
                "answer_audio_path": "",
                "error": rag_res["error"]
            }

        answer_text = rag_res["answer"]

        # 3. Synthesize response to speech
        logger.info(f"[STAGE 4: TTS] Synthesizing response: {answer_text[:50]}...")
        tts_start = time.perf_counter()
        
        try:
            audio_path = speak_text(answer_text)
            tts_latency = (time.perf_counter() - tts_start) * 1000.0
            logger.info(f"[STAGE 4 Complete] Generated audio saved to {audio_path} in {tts_latency:.2f}ms")
            
            return {
                "answer": answer_text,
                "sources": rag_res["sources"],
                "domain_detected": rag_res["domain_detected"],
                "provider_used": rag_res["provider_used"],
                "transcribed_question": transcribed_text,
                "answer_audio_path": audio_path,
                "error": None
            }
        except Exception as exc:
            err_msg = f"TTS Synthesis failed: {exc}"
            logger.warning(err_msg)
            return {
                "answer": answer_text,
                "sources": rag_res["sources"],
                "domain_detected": rag_res["domain_detected"],
                "provider_used": rag_res["provider_used"],
                "transcribed_question": transcribed_text,
                "answer_audio_path": "",
                "error": f"Answer generated, but TTS failed: {exc}"
            }

    def caption_uploaded_image(
        self,
        image_path: str | Path | bytes,
        style: str = "descriptive",
        follow_up: bool = False
    ) -> Dict[str, Any]:
        """Generate a caption for an image, with an optional RAG search follow-up.

        Args:
            image_path: Path to the uploaded image.
            style:      Style of description ('descriptive', 'short', 'detailed').
            follow_up:  If True, queries RAG for extra details about the caption.

        Returns:
            Dictionary containing:
                - caption: The text caption.
                - provider_used: LLM provider name.
                - latency_ms: Total call latency.
                - follow_up_answer: Optional RAG follow-up text.
                - follow_up_sources: Optional RAG sources.
                - error: Optional error description string.
        """
        logger.info(f"[STAGE: Vision] Captioning image: {image_path} (style: {style})")
        start_time = time.perf_counter()
        
        cap_res = caption_image(image_path, style=style)
        latency = (time.perf_counter() - start_time) * 1000.0

        if cap_res.get("error"):
            logger.error(f"[STAGE: Vision Failed] {cap_res['error']}")
            return {
                "caption": "",
                "provider_used": "",
                "latency_ms": latency,
                "follow_up_answer": None,
                "follow_up_sources": [],
                "error": cap_res["error"]
            }

        caption_text = cap_res["caption"]
        logger.info(f"[STAGE: Vision Complete] Caption: {caption_text!r}")

        follow_up_answer = None
        follow_up_sources = []

        # Trigger RAG query about the caption if requested
        if follow_up and caption_text:
            logger.info(f"[STAGE: RAG Follow-up] Querying details about caption: {caption_text!r}")
            query = f"Tell me more about this: {caption_text}"
            rag_res = self.ask_text(query, domain=None)
            follow_up_answer = rag_res.get("answer")
            follow_up_sources = rag_res.get("sources", [])

        return {
            "caption": caption_text,
            "provider_used": cap_res["provider_used"],
            "latency_ms": latency,
            "follow_up_answer": follow_up_answer,
            "follow_up_sources": follow_up_sources,
            "error": None
        }

    def generate_image_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """Wrap image_generation module directly."""
        logger.info(f"[STAGE: Image Gen] Requesting image for prompt: {prompt!r}")
        return generate_image(prompt)

    def full_pipeline_demo(
        self,
        question: str | None = None,
        audio_filepath: str | None = None,
        image_filepath: str | None = None
    ) -> Dict[str, Any]:
        """Execute text, voice, and/or image inputs through a single convenience interface.

        Args:
            question:       Direct text question.
            audio_filepath: Voice query recording path.
            image_filepath: Uploaded image file path.

        Returns:
            A combined result dictionary containing all execution outputs.
        """
        logger.info("Executing MultimodalAssistant full pipeline demo...")
        
        result = {
            "transcribed_question": None,
            "answer_text": None,
            "answer_audio_path": None,
            "sources": [],
            "domain_detected": None,
            "caption": None,
            "provider_used": None,
            "error": None
        }

        try:
            active_question = question

            # 1. Process voice audio if provided
            if audio_filepath:
                voice_res = self.ask_voice(audio_filepath)
                result["transcribed_question"] = voice_res.get("transcribed_question")
                result["answer_text"] = voice_res.get("answer")
                result["answer_audio_path"] = voice_res.get("answer_audio_path")
                result["sources"] = voice_res.get("sources", [])
                result["domain_detected"] = voice_res.get("domain_detected")
                result["provider_used"] = voice_res.get("provider_used")
                result["error"] = voice_res.get("error")
                active_question = result["transcribed_question"]

            # 2. Process text question (if RAG answer is not already computed by voice)
            if active_question and not result["answer_text"]:
                text_res = self.ask_text(active_question)
                result["answer_text"] = text_res.get("answer")
                result["sources"] = text_res.get("sources", [])
                result["domain_detected"] = text_res.get("domain_detected")
                result["provider_used"] = text_res.get("provider_used")
                if text_res.get("error"):
                    result["error"] = text_res.get("error")

                # Synthesize TTS for text answers
                if result["answer_text"] and not result["answer_text"].startswith("Error"):
                    logger.info("[STAGE: TTS] Synthesizing speech for text-only pipeline answer...")
                    try:
                        result["answer_audio_path"] = speak_text(result["answer_text"])
                    except Exception as e:
                        logger.warning(f"TTS synthesis failed in pipeline: {e}")

            # 3. Process image upload if provided
            if image_filepath:
                cap_res = self.caption_uploaded_image(image_filepath)
                result["caption"] = cap_res.get("caption")
                if cap_res.get("error"):
                    result["error"] = cap_res.get("error")

        except Exception as exc:
            err_msg = f"Pipeline execution failed: {exc}"
            logger.error(err_msg)
            result["error"] = err_msg

        return result


if __name__ == "__main__":
    # Visual check runner
    print("MultimodalAssistant CLI runner.")
    assistant = MultimodalAssistant()
    
    # Run a text query end-to-end
    q = "What is the course code for Introduction to Computer Science?"
    print(f"\nRunning end-to-end query: {q!r}")
    res = assistant.ask_text(q)
    
    print("\n=== Pipeline Demo Output ===")
    print("Answer:       ", res["answer"])
    print("Sources:      ", res["sources"])
    print("Domain:       ", res["domain_detected"])
    print("Provider:     ", res["provider_used"])
    print("Error Status: ", res["error"])
    print("=============================")
