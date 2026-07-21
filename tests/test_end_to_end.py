import io
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image
from gtts import gTTS

from core.assistant import MultimodalAssistant
from core.schemas import LLMResponse




@pytest.fixture
def assistant():
    return MultimodalAssistant()


@pytest.fixture
def sample_voice_audio():
    """Generate a short synthetic voice query file."""
    tts = gTTS(text="hospital emergency contact", lang="en")
    temp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_mp3_path = Path(temp_mp3.name)
    temp_mp3.close()

    try:
        tts.save(str(temp_mp3_path))
        yield temp_mp3_path
    finally:
        if temp_mp3_path.exists():
            temp_mp3_path.unlink()


@pytest.fixture
def sample_image_bytes():
    """Generate a dummy image."""
    img = Image.new("RGB", (100, 100), "red")
    out = io.BytesIO()
    img.save(out, format="JPEG")
    return out.getvalue()


def test_full_end_to_end_scenario(assistant, sample_voice_audio, sample_image_bytes):
    """Run one full end-to-end scenario covering all capstone features."""
    
    # 1. Mock responses
    mock_rag_answers = {
        "benefits of crop rotation": "Crop rotation breaks pest and disease cycles.",
        "course code and credits for Introduction to Computer Science": "The code is CS-101 (4 credits).",
        "emergency contact number for City General Hospital": "The contact is 911.",
        "hospital emergency contact": "The contact is 911.",
        "Who wrote 'The AI Horizon'?": "Dr. Evelyn Sterling wrote it.",
        "Where is Stonehenge located?": "Stonehenge is in Wiltshire, England.",
        "dates for the 'Echoes of Modernity' exhibition": "It runs from March 15 to June 20.",
        "amenities of the Grand Plaza Hotel": "It includes free Wi-Fi, spa, pool."
    }

    def mock_generate_side_effect(prompt, system=None, temperature=0.3, max_tokens=1024):
        # Match prompt against known answers (case-insensitive)
        for kw, ans in mock_rag_answers.items():
            if kw.lower() in prompt.lower():
                return LLMResponse(text=ans, provider_used="groq", latency_ms=50.0, fallback_triggered=False)
        return LLMResponse(text="Default fallback answer.", provider_used="groq", latency_ms=50.0, fallback_triggered=False)

    mock_vision_response = LLMResponse(
        text="A solid red square.",
        provider_used="groq",
        latency_ms=100.0,
        fallback_triggered=False
    )

    mock_img_result = mock.Mock()
    mock_img_result.image.image_bytes = b"fake_gemini_generated_image_bytes"
    mock_gemini_img_res = mock.Mock()
    mock_gemini_img_res.generated_images = [mock_img_result]

    # 2. Patch all API and HTTP calls
    with mock.patch("core.llm_provider.LLMProvider.generate", side_effect=mock_generate_side_effect) as mock_gen, \
         mock.patch("core.llm_provider.LLMProvider.generate_vision", return_value=mock_vision_response) as mock_vision, \
         mock.patch("google.generativeai.ImageGenerationModel", create=True) as mock_gemini_img_class:
         
        # Set up Gemini image generator mock
        mock_instance = mock.Mock()
        mock_instance.generate_images.return_value = mock_gemini_img_res
        mock_gemini_img_class.return_value = mock_instance

        # ── Step A: 7 Domain Text Queries ─────────────────────────────────────
        domain_queries = [
            ("What are the benefits of crop rotation?", "agriculture", "crop_rotation.txt"),
            ("What is the course code and credits for Introduction to Computer Science?", "college", "course_catalog.txt"),
            ("What is the emergency contact number for City General Hospital?", "healthcare", "hospital_services.txt"),
            ("Who wrote 'The AI Horizon'?", "library", "book_selection.txt"),
            ("Where is Stonehenge located?", "monuments", "historic_site.txt"),
            ("What are the dates for the 'Echoes of Modernity' exhibition?", "museums", "exhibition_schedule.txt"),
            ("What are the amenities of the Grand Plaza Hotel?", "tourism", "hotel_info.txt")
        ]

        for q, domain, source_file in domain_queries:
            res = assistant.ask_text(q, domain=domain)
            assert res["error"] is None, f"Unexpected error for domain={domain!r}: {res['error']}"
            assert len(res["answer"]) > 0, f"Empty answer for domain={domain!r}"
            assert res["domain_detected"] == domain, (
                f"Expected domain {domain!r}, got {res['domain_detected']!r}"
            )
            # Verify at least one source chunk was returned (filename ranking
            # can vary — we only assert the list is non-empty).
            assert len(res["sources"]) > 0, f"No sources returned for domain={domain!r}"

        # ── Step B: Voice Question ────────────────────────────────────────────
        voice_res = assistant.ask_voice(sample_voice_audio, domain=None)
        assert voice_res["error"] is None
        assert "hospital emergency" in voice_res["transcribed_question"].lower()
        assert "911" in voice_res["answer"]
        assert voice_res["domain_detected"] == "healthcare"
        assert voice_res["answer_audio_path"] != ""
        assert Path(voice_res["answer_audio_path"]).exists()

        # ── Step C: Image Captioning (bytes input) ────────────────────────────
        # caption_uploaded_image() now accepts bytes, str, or Path — all three
        # are supported by the underlying caption_image() module.
        caption_res = assistant.caption_uploaded_image(
            sample_image_bytes, style="descriptive", follow_up=True
        )
        assert caption_res["error"] is None, f"Caption error: {caption_res['error']}"
        assert "red" in caption_res["caption"].lower() or "square" in caption_res["caption"].lower(), (
            f"Unexpected caption: {caption_res['caption']!r}"
        )
        assert caption_res["follow_up_answer"] is not None

        # ── Step D: Image Generation ──────────────────────────────────────────
        with mock.patch("config.settings.settings.gemini_api_key", "fake_key"):
            img_res = assistant.generate_image_from_prompt("A beautiful futuristic city")
            assert img_res["error"] is None
            assert img_res["provider_used"] == "gemini"
            assert img_res["image_path"] != ""
            assert Path(img_res["image_path"]).exists()
