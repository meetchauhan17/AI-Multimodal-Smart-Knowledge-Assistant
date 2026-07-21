import os
from pathlib import Path
import pytest

from modules.text_to_speech import speak_text, _chunk_text, _split_into_sentences


def test_speak_text_short():
    """Verify that a short sentence is successfully synthesized and cached."""
    text = "This is a short test of the text to speech conversion."
    filepath = speak_text(text)
    
    assert filepath != ""
    path = Path(filepath)
    assert path.exists()
    assert path.stat().st_size > 0
    
    # Test Cache Hit
    filepath_cached = speak_text(text)
    assert filepath_cached == filepath


def test_chunking_logic():
    """Test that text is chunked correctly based on character limits."""
    short_text = "Hello world."
    chunks_short = _chunk_text(short_text, max_chars=100)
    assert len(chunks_short) == 1
    assert chunks_short[0] == "Hello world."

    # Long text made of multiple sentences
    sentence = "The quick brown fox jumps over the lazy dog."
    long_text = " ".join([sentence] * 15)  # ~670 characters
    
    chunks_long = _chunk_text(long_text, max_chars=200)
    assert len(chunks_long) > 1
    for chunk in chunks_long:
        assert len(chunk) <= 200


def test_speak_text_long():
    """Verify that a very long text is chunked, synthesized, and merged without error."""
    sentence = "Artificial intelligence is transforming how we build modern software applications."
    # Generate a string of ~2400 characters (30 sentences)
    long_text = " ".join([sentence] * 30)
    
    assert len(long_text) > 2000
    
    filepath = speak_text(long_text)
    assert filepath != ""
    path = Path(filepath)
    assert path.exists()
    assert path.stat().st_size > 0
