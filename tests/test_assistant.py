import pytest
from unittest import mock
from pathlib import Path

from core.assistant import MultimodalAssistant
from core.schemas import LLMResponse


@pytest.fixture
def assistant():
    return MultimodalAssistant()


def test_ask_text_rag_flow(assistant):
    """Test that ask_text routes query to RAG retriever and LLM successfully."""
    mock_llm_response = LLMResponse(
        text="Stonehenge is a famous prehistoric standing stones ring in Wiltshire, England.",
        provider_used="groq",
        latency_ms=150.0,
        fallback_triggered=False
    )

    with mock.patch("core.llm_provider.LLMProvider.generate", return_value=mock_llm_response) as mock_gen:
        # Querying monuments
        result = assistant.ask_text("Where is Stonehenge?", domain="monuments")
        
        assert result["error"] is None
        assert "Stonehenge" in result["answer"]
        assert "Wiltshire" in result["answer"]
        assert result["domain_detected"] == "monuments"
        assert result["provider_used"] == "groq"
        
        # Verify the sources list contains historic_site.txt
        assert any("historic_site.txt" in s for s in result["sources"])
        
        # Verify the LLM was called with the context and system instruction
        mock_gen.assert_called_once()
        called_kwargs = mock_gen.call_args[1]
        assert "provided Context" in called_kwargs["system"]


def test_caption_uploaded_image_with_followup(assistant):
    """Test that captioning works and optional follow-up RAG query resolves."""
    mock_caption_response = LLMResponse(
        text="A high-altitude shot of an ancient structure resembling Stonehenge.",
        provider_used="groq",
        latency_ms=110.0,
        fallback_triggered=False
    )
    
    mock_rag_response = LLMResponse(
        text="This refers to the prehistoric monument consisting of standing stones in Wiltshire.",
        provider_used="groq",
        latency_ms=90.0,
        fallback_triggered=False
    )

    # Use a valid dummy image bytes as image input
    from PIL import Image
    import io
    img = Image.new("RGB", (100, 100), "red")
    out = io.BytesIO()
    img.save(out, format="JPEG")
    dummy_image = out.getvalue()

    with mock.patch("core.llm_provider.LLMProvider.generate_vision", return_value=mock_caption_response) as mock_vision, \
         mock.patch("core.llm_provider.LLMProvider.generate", return_value=mock_rag_response) as mock_gen:
        
        # Run with follow_up=True
        result = assistant.caption_uploaded_image(
            dummy_image,
            style="descriptive",
            follow_up=True
        )
        
        assert result["error"] is None
        assert "Stonehenge" in result["caption"]
        assert result["provider_used"] == "groq"
        
        # Verify follow-up answer was resolved from the RAG search
        assert result["follow_up_answer"] == mock_rag_response.text
        assert len(result["follow_up_sources"]) > 0
        
        # Assert both endpoints were called correctly
        mock_vision.assert_called_once()
        mock_gen.assert_called_once()
        assert "Tell me more about this" in mock_gen.call_args[1]["prompt"]
