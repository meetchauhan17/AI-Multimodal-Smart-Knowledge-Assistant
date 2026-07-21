"""
core/schemas.py — Shared Pydantic models (request / response contracts).

This file is a scaffold. Add domain-specific models as pipeline modules
are implemented.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Generic I/O ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Represents a user query sent to the assistant."""
    text: str = Field(..., description="The natural-language query from the user.")
    domain: Optional[str] = Field(None, description="Optional knowledge domain (e.g. 'healthcare').")
    modality: Optional[str] = Field(None, description="Input modality: 'text' | 'audio' | 'image'.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Represents the assistant's response to a query."""
    answer: str = Field(..., description="The generated answer text.")
    sources: List[str] = Field(default_factory=list, description="Source document references.")
    provider_used: Optional[str] = Field(None, description="Which LLM provider produced this answer.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Represents the output from the LLM provider router."""
    text: str = Field(..., description="The generated text response.")
    provider_used: str = Field(..., description="The name of the provider that successfully generated the response.")
    latency_ms: float = Field(..., description="Response latency in milliseconds.")
    fallback_triggered: bool = Field(..., description="Whether any fallback provider was called.")


# ── RAG ───────────────────────────────────────────────────────────────────────

class Document(BaseModel):
    """A knowledge-base document chunk used by the RAG pipeline."""
    content: str
    source: str
    domain: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Multimodal ────────────────────────────────────────────────────────────────

class ImageCaptionResult(BaseModel):
    caption: str
    confidence: Optional[float] = None


class SpeechTranscriptResult(BaseModel):
    transcript: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None


class TTSResult(BaseModel):
    audio_path: str
    text: str
