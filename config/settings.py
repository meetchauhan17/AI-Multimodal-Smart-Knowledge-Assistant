"""
config/settings.py — Centralised project configuration via Pydantic BaseSettings.

All settings are loaded from environment variables (or a .env file).
See .env.example for the full list of variables and where to get each key.
"""

from __future__ import annotations

from typing import Any, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Groq ─────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_vision_model: str = Field(default="llama-3.2-11b-vision-preview", alias="GROQ_VISION_MODEL")

    # ── Google Gemini ─────────────────────────────────────────────────────────
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.5-flash", alias="GEMINI_MODEL")
    gemini_vision_model: str = Field(default="gemini-3.5-flash", alias="GEMINI_VISION_MODEL")

    # ── LLM provider routing ──────────────────────────────────────────────────
    primary_provider: str = Field(default="gemini", alias="PRIMARY_PROVIDER")
    # Stored as a comma-separated string in .env; parsed to a list below.
    fallback_order: Any = Field(default=["groq", "gemini"], alias="FALLBACK_ORDER")

    # ── Image generation ──────────────────────────────────────────────────────
    gemini_image_model: str = Field(
        default="imagen-3.0-generate-002",
        alias="GEMINI_IMAGE_MODEL",
    )
    image_gen_fallback_url: str = Field(
        default="https://image.pollinations.ai/prompt/{prompt}",
        alias="IMAGE_GEN_FALLBACK_URL",
    )

    # ── Speech ────────────────────────────────────────────────────────────────
    whisper_model: str = Field(default="base", alias="WHISPER_MODEL")

    # ── Paths ─────────────────────────────────────────────────────────────────
    knowledge_base_dir: str = Field(default="data/knowledge_base", alias="KNOWLEDGE_BASE_DIR")
    chroma_persist_dir: str = Field(default="data/chroma_db", alias="CHROMA_PERSIST_DIR")
    log_dir: str = Field(default="logs", alias="LOG_DIR")

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("fallback_order", mode="before")
    @classmethod
    def _parse_fallback_order(cls, v: object) -> List[str]:
        """Accept either a list or a comma-separated string from .env."""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return ["groq", "gemini"]


# Module-level singleton — import and use directly:
#   from config.settings import settings
settings = Settings()
