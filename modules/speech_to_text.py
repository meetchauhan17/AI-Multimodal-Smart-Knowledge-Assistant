"""
modules/speech_to_text.py — Speech-to-text conversion using local Whisper model.

This module implements local transcription from audio files to text.
We use `faster-whisper` for low-latency, CPU-optimised inference.
It uses `pydub` for audio format conversion and validation, with a graceful
fallback to `av` (PyAV) if system-wide ffmpeg binaries are missing on the PATH.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Any

from pydub import AudioSegment
from faster_whisper import WhisperModel

from config.settings import settings
from core.logger import logger

# Module-level cache to keep the model in memory across calls
_model_cache: Dict[str, WhisperModel] = {}


def get_whisper_model(model_size: str) -> WhisperModel:
    """Retrieve the cached Whisper model or load it if not cached.
    
    Using cpu device with int8 quantization for optimal local CPU performance.
    """
    global _model_cache
    if model_size not in _model_cache:
        logger.info(f"Loading Whisper model '{model_size}' into memory (CPU, int8)...")
        _model_cache[model_size] = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root=None,
        )
    return _model_cache[model_size]


def _get_audio_info(audio_path: Path) -> Dict[str, Any]:
    """Inspect audio file properties using pydub, falling back to av (PyAV) if ffmpeg is missing."""
    try:
        # Try parsing with pydub first
        audio = AudioSegment.from_file(str(audio_path))
        duration_ms = len(audio)
        # dBFS < -60.0 or rms == 0 represents silent audio
        is_silent = audio.rms == 0 or audio.dBFS < -60.0
        return {"duration_ms": duration_ms, "is_silent": is_silent, "method": "pydub"}
    except Exception as exc:
        # If ffmpeg is missing, from_file will throw FileNotFoundError or RuntimeWarning
        logger.info(f"pydub failed to inspect audio (likely missing ffmpeg): {exc}. Falling back to PyAV.")
        try:
            import av
            import numpy as np

            container = av.open(str(audio_path))
            stream = container.streams.audio[0]

            # Calculate duration
            if stream.duration and stream.time_base:
                duration_sec = float(stream.duration * stream.time_base)
                duration_ms = int(duration_sec * 1000)
            else:
                duration_ms = int((container.duration or 0) / 1000)

            # Analyze a sample of frames to estimate silence
            rms_values = []
            frame_count = 0
            for frame in container.decode(audio=0):
                if frame_count > 50:
                    break
                samples = frame.to_ndarray()
                rms = np.sqrt(np.mean(samples**2))
                rms_values.append(rms)
                frame_count += 1

            container.close()

            # If no frames decoded or max rms is extremely low, treat as silent
            is_silent = len(rms_values) == 0 or max(rms_values) < 1.0e-4
            return {"duration_ms": duration_ms, "is_silent": is_silent, "method": "av"}

        except Exception as av_exc:
            raise RuntimeError(
                f"Both pydub (ffmpeg) and PyAV failed to parse audio: {av_exc}"
            ) from exc


def transcribe_audio(audio_filepath: str | Path, language: str | None = None) -> Dict[str, Any]:
    """Convert spoken audio from a file into text.

    Handles format conversion using pydub, pre-flight checks for duration and silence,
    and uses the cached faster-whisper model.

    Args:
        audio_filepath: Absolute or relative path to the audio file.
        language:       Optional language override (e.g. 'en', 'fr').

    Returns:
        A dictionary containing:
            - text: The transcribed text.
            - detected_language: Detected language code or None.
            - confidence_estimate: Confidence probability of the language detection.
            - error: Optional error description string if any validation failed.
    """
    audio_path = Path(audio_filepath)
    if not audio_path.exists():
        err_msg = f"Audio file not found: {audio_path}"
        logger.error(err_msg)
        return {"text": "", "detected_language": None, "confidence_estimate": 0.0, "error": err_msg}

    temp_wav_path = None
    try:
        # Pre-flight validation checks using fallback chain
        info = _get_audio_info(audio_path)
        
        if info["duration_ms"] < 500:
            err_msg = f"Audio is too short ({info['duration_ms']}ms). Minimum required is 500ms."
            logger.warning(err_msg)
            return {
                "text": "",
                "detected_language": None,
                "confidence_estimate": 0.0,
                "error": err_msg,
            }

        if info["is_silent"]:
            err_msg = "Audio is silent."
            logger.warning(err_msg)
            return {
                "text": "",
                "detected_language": None,
                "confidence_estimate": 0.0,
                "error": err_msg,
            }

        transcribe_path = str(audio_path)

        # Convert to WAV with pydub if ffmpeg is available
        if info["method"] == "pydub":
            try:
                audio = AudioSegment.from_file(str(audio_path))
                temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_wav_path = Path(temp_wav.name)
                temp_wav.close()
                audio.export(str(temp_wav_path), format="wav")
                transcribe_path = str(temp_wav_path)
            except Exception as conv_exc:
                logger.warning(f"Could not convert format with pydub: {conv_exc}. Passing original file directly.")

        # Load Whisper model and transcribe
        model_size = settings.whisper_model
        model = get_whisper_model(model_size)

        logger.info(f"Starting Whisper transcription for {audio_path.name} using '{model_size}' model")
        
        # transcribe() returns a generator of segments and a transcription info object
        segments, trans_info = model.transcribe(transcribe_path, language=language)
        
        # Consume segments generator to get the full transcript text
        text_parts = [segment.text for segment in segments]
        full_text = "".join(text_parts).strip()

        logger.info(
            f"Transcription complete. Language={trans_info.language!r} (prob={trans_info.language_probability:.4f})"
        )

        return {
            "text": full_text,
            "detected_language": trans_info.language,
            "confidence_estimate": float(trans_info.language_probability),
            "error": None,
        }

    except Exception as exc:
        err_msg = f"Failed to transcribe audio: {exc}"
        logger.exception(err_msg)
        return {"text": "", "detected_language": None, "confidence_estimate": 0.0, "error": err_msg}

    finally:
        # Clean up temporary WAV file
        if temp_wav_path and temp_wav_path.exists():
            try:
                os.remove(temp_wav_path)
                logger.debug(f"Temporary WAV file deleted: {temp_wav_path}")
            except Exception as exc:
                logger.warning(f"Failed to delete temporary file {temp_wav_path}: {exc}")


if __name__ == "__main__":
    # Quick visual verification (scaffold test)
    print("SpeechToText module CLI runner.")
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Transcribing: {path}")
        res = transcribe_audio(path)
        print("Result:", res)
    else:
        print("No test file provided. Usage: python speech_to_text.py <audio_file_path>")
