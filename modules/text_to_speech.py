"""
modules/text_to_speech.py — Text-to-speech conversion using gTTS.

Exposes speak_text() to convert natural language text into speech audio (.mp3).
Includes caching, sentence chunking for long text, and robust audio merging.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any, List

from gtts import gTTS
from pydub import AudioSegment

from core.logger import logger

# Directory to save synthesized audio files.
# Use an absolute path anchored to the project root so the output directory
# is the same regardless of the working directory (e.g. when running tests).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMP_AUDIO_DIR = _PROJECT_ROOT / "temp_audio"
TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_key(text: str, lang: str, slow: bool) -> str:
    """Generate a unique MD5 hash for the combination of text, lang, and speed."""
    raw_key = f"{text.strip()}|{lang.lower()}|{slow}"
    return hashlib.md5(raw_key.encode("utf-8")).hexdigest()


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentence-level chunks using punctuation boundaries."""
    # Split by periods, question marks, and exclamation marks followed by spaces or newlines
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(text: str, max_chars: int = 500) -> List[str]:
    """Group sentences into larger text blocks that do not exceed max_chars.
    
    If a single sentence is longer than max_chars, it will be split by words.
    """
    sentences = _split_into_sentences(text)
    chunks = []
    current_chunk: List[str] = []
    current_length = 0

    for sentence in sentences:
        # Handle exceptionally long sentences by splitting into words
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            words = sentence.split(" ")
            sub_chunk: List[str] = []
            sub_len = 0
            for word in words:
                if sub_len + len(word) + 1 > max_chars:
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                    sub_chunk = [word]
                    sub_len = len(word)
                else:
                    sub_chunk.append(word)
                    sub_len += len(word) + 1
            if sub_chunk:
                current_chunk = sub_chunk
                current_length = sub_len
        else:
            # Check if adding the next sentence exceeds chunk limit
            space_added = 1 if current_chunk else 0
            if current_length + len(sentence) + space_added > max_chars:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence) + space_added

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _merge_audio_files(filepaths: List[Path], output_path: Path) -> None:
    """Concatenate multiple audio files into a single destination file.
    
    Tries pydub first, falling back to a raw binary join if ffmpeg is missing.
    """
    try:
        # Try merging via pydub
        logger.debug(f"Attempting to merge {len(filepaths)} files using pydub")
        combined = AudioSegment.empty()
        for fp in filepaths:
            segment = AudioSegment.from_file(str(fp))
            combined += segment
        combined.export(str(output_path), format="mp3")
        logger.debug("Successfully merged audio files using pydub.")
    except Exception as exc:
        # Fall back to raw binary concatenation of MP3 frames
        logger.info(
            f"pydub merge failed (likely missing ffmpeg): {exc}. "
            "Falling back to raw binary concatenation."
        )
        try:
            with open(output_path, "wb") as outfile:
                for fp in filepaths:
                    with open(fp, "rb") as infile:
                        outfile.write(infile.read())
            logger.debug("Successfully merged audio files using raw binary concatenation.")
        except Exception as bin_exc:
            raise RuntimeError(f"Failed to merge audio files: {bin_exc}") from exc


def speak_text(text: str, lang: str = "en", slow: bool = False) -> str:
    """Convert text into spoken audio and return the filepath.

    Uses gTTS for synthesis. Splits long text into chunks, synthesizes each,
    merges them, and caches output based on parameters.

    Args:
        text: The text string to speak.
        lang: BCP-47 language code (e.g. 'en', 'fr').
        slow: If True, synthesizes speech at a slower speed.

    Returns:
        The string path to the generated MP3 file.
    """
    if not text.strip():
        # Synthesize a short empty space or throw? Return empty WAV
        logger.warning("Empty text passed to speak_text().")
        return ""

    # Check cache
    cache_key = _get_cache_key(text, lang, slow)
    cached_path = TEMP_AUDIO_DIR / f"tts_{cache_key}.mp3"
    
    if cached_path.exists() and cached_path.stat().st_size > 0:
        logger.info(f"TTS Cache hit: {cached_path.name}")
        return str(cached_path)

    # Split text if it exceeds 500 characters
    text_chunks = _chunk_text(text, max_chars=500)
    logger.info(f"Synthesizing text ({len(text)} chars) in {len(text_chunks)} chunk(s)")

    chunk_files: List[Path] = []
    try:
        for idx, chunk in enumerate(text_chunks):
            # Synthesize single chunk
            logger.debug(f"Calling gTTS for chunk {idx+1}/{len(text_chunks)}: {chunk[:50]}...")
            tts = gTTS(text=chunk, lang=lang, slow=slow)
            
            # Save chunk to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_path = Path(temp_file.name)
            temp_file.close()
            
            tts.save(str(temp_path))
            chunk_files.append(temp_path)

        # Merge synthesized chunks into the final cached path
        if len(chunk_files) == 1:
            # Move/Rename directly
            os.replace(chunk_files[0], cached_path)
            # Clear list so we don't try to delete it in finally
            chunk_files = []
        else:
            _merge_audio_files(chunk_files, cached_path)

        logger.info(f"TTS synthesis successful: {cached_path}")
        return str(cached_path)

    except Exception as exc:
        err_msg = f"Failed to synthesize text to speech: {exc}"
        logger.exception(err_msg)
        raise RuntimeError(err_msg) from exc

    finally:
        # Cleanup temporary chunk files
        for fp in chunk_files:
            if fp.exists():
                try:
                    os.remove(fp)
                except Exception as cleanup_exc:
                    logger.warning(f"Failed to delete temp chunk file {fp}: {cleanup_exc}")


# ── Scaffold Class Wrapper for Backwards Compatibility ──────────────────────

class TextToSpeech:
    """Converts text to an audio file using gTTS (wrapped class)."""

    def __init__(
        self,
        lang: str = "en",
        slow: bool = False,
        output_dir: str | Path = "temp_audio",
    ) -> None:
        self.lang = lang
        self.slow = slow
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text: str, filename: str | None = None) -> Any:
        """Synthesize text using the speak_text wrapper."""
        from core.schemas import TTSResult
        
        filepath = speak_text(text, lang=self.lang, slow=self.slow)
        
        # If a custom filename was requested, copy/move it
        if filename and filepath:
            dest_path = self.output_dir / filename
            os.replace(filepath, dest_path)
            filepath = str(dest_path)

        return TTSResult(audio_path=filepath, text=text)


if __name__ == "__main__":
    # Main block verification
    print("TextToSpeech module CLI runner.")
    test_phrase = "Hello! This is a test of the text to speech synthesis module."
    print(f"Synthesizing: '{test_phrase}'")
    out = speak_text(test_phrase)
    print(f"Resulting file path: {out}")
    if out and os.path.exists(out):
        print(f"File size: {os.path.getsize(out)} bytes")
