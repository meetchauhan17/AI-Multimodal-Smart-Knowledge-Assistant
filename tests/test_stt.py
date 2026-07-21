import os
import tempfile
from pathlib import Path

import pytest
from gtts import gTTS
from pydub import AudioSegment

from modules.speech_to_text import transcribe_audio


@pytest.fixture(scope="module")
def sample_phrase_audio():
    """Generate a short synthetic speech audio file containing 'hello world'."""
    text = "hello world"
    # Generate MP3 using gTTS
    tts = gTTS(text=text, lang="en")
    
    # Save to a temporary file
    temp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_mp3_path = Path(temp_mp3.name)
    temp_mp3.close()

    try:
        tts.save(str(temp_mp3_path))
        yield temp_mp3_path
    finally:
        # Cleanup
        if temp_mp3_path.exists():
            os.remove(temp_mp3_path)


@pytest.fixture
def silent_audio():
    """Generate a 1-second silent audio file."""
    # 1000ms of silence
    silence = AudioSegment.silent(duration=1000)
    
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = Path(temp_wav.name)
    temp_wav.close()

    try:
        silence.export(str(temp_wav_path), format="wav")
        yield temp_wav_path
    finally:
        if temp_wav_path.exists():
            os.remove(temp_wav_path)


@pytest.fixture
def too_short_audio():
    """Generate a 200ms audio file (under the 500ms limit)."""
    short_segment = AudioSegment.silent(duration=200)
    
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = Path(temp_wav.name)
    temp_wav.close()

    try:
        short_segment.export(str(temp_wav_path), format="wav")
        yield temp_wav_path
    finally:
        if temp_wav_path.exists():
            os.remove(temp_wav_path)


def test_transcribe_success(sample_phrase_audio):
    """Test that valid speech is transcribed correctly and roughly matches what was said."""
    result = transcribe_audio(sample_phrase_audio)
    
    assert result["error"] is None
    assert "hello" in result["text"].lower()
    assert "world" in result["text"].lower()
    assert result["detected_language"] == "en"
    assert result["confidence_estimate"] > 0.5


def test_transcribe_silent_audio(silent_audio):
    """Test that silent audio is detected and returns an error response."""
    result = transcribe_audio(silent_audio)
    
    assert result["text"] == ""
    assert result["error"] is not None
    assert "silent" in result["error"].lower()


def test_transcribe_too_short_audio(too_short_audio):
    """Test that audio that is too short is caught and returns an error response."""
    result = transcribe_audio(too_short_audio)
    
    assert result["text"] == ""
    assert result["error"] is not None
    assert "too short" in result["error"].lower()
