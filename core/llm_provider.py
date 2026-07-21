"""
core/llm_provider.py — LLM provider router with Gemini → Groq fallback chain.

All calls route through the single LLMProvider class.
"""

from __future__ import annotations

import base64
import os
import time
from typing import List

from tenacity import retry, stop_after_attempt, retry_if_exception, wait_exponential

from config.settings import settings
from core.logger import logger
from core.schemas import LLMResponse


class PermanentProviderError(Exception):
    """An error that represents a permanent failure (e.g. missing CLI, invalid keys).
    
    This error will bypass tenacity retries and trigger immediate fallback.
    """
    pass


def is_transient_error(exception: Exception) -> bool:
    """Return True if the exception is transient and should be retried."""
    if isinstance(exception, PermanentProviderError):
        return False
    
    if isinstance(exception, FileNotFoundError):
        return False

    # Check for Groq API status codes
    try:
        from groq import APIStatusError
        if isinstance(exception, APIStatusError):
            # 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found are permanent
            if exception.status_code in (400, 401, 403, 404):
                return False
    except ImportError:
        pass

    # Check for Gemini API status codes
    try:
        from google.api_core.exceptions import GoogleAPICallError
        if isinstance(exception, GoogleAPICallError):
            # 4xx client errors are generally permanent, except for 429 Too Many Requests
            if exception.code and 400 <= exception.code < 500 and exception.code != 429:
                return False
    except ImportError:
        pass

    # By default, unexpected exceptions (network timeout, 5xx, etc.) are transient
    return True


class LLMProvider:
    """Routes LLM requests through an ordered fallback chain with retries."""

    def __init__(self) -> None:
        self.fallback_order = settings.fallback_order
        logger.debug(f"LLMProvider initialised with fallback order: {self.fallback_order}")

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a response using the first available provider from the fallback chain.

        Args:
            prompt:      The user prompt.
            system:      Optional system prompt.
            temperature: Sampling temperature.
            max_tokens:  Maximum tokens to generate.

        Returns:
            An LLMResponse containing the text, provider used, latency, and fallback status.

        Raises:
            RuntimeError: If all providers in the fallback chain fail.
        """
        errors = []
        first_attempted = True
        fallback_triggered = False

        for provider in self.fallback_order:
            provider = provider.strip().lower()
            logger.info(f"Attempting text generation with provider: {provider}")
            
            start_time = time.perf_counter()
            try:
                text = self._call_with_retry(provider, prompt, system, temperature, max_tokens)
                latency = (time.perf_counter() - start_time) * 1000.0
                
                logger.info(f"Successfully generated response using {provider} in {latency:.2f}ms")
                return LLMResponse(
                    text=text,
                    provider_used=provider,
                    latency_ms=latency,
                    fallback_triggered=fallback_triggered,
                )
            except Exception as exc:
                latency = (time.perf_counter() - start_time) * 1000.0
                logger.warning(
                    f"Provider {provider} failed after {latency:.2f}ms. Error: {exc}"
                )
                errors.append((provider, str(exc)))
                first_attempted = False
                fallback_triggered = True

        # If we got here, all providers failed
        err_msg = f"All LLM providers failed. Errors: {errors}"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    def generate_vision(
        self,
        image_bytes: bytes,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a response for multimodal vision prompts.

        Routes strictly through the configured primary and fallback providers.

        Args:
            image_bytes: The raw bytes of the image.
            prompt:      The text prompt accompanying the image.
            system:      Optional system prompt.
            temperature: Sampling temperature.
            max_tokens:  Maximum tokens to generate.

        Returns:
            An LLMResponse containing the text, provider used, latency, and fallback status.

        Raises:
            RuntimeError: If all vision providers fail.
        """
        vision_order = ["groq", "gemini"]
        errors = []
        first_attempted = True
        fallback_triggered = False

        for provider in vision_order:
            logger.info(f"Attempting vision generation with provider: {provider}")
            
            start_time = time.perf_counter()
            try:
                text = self._call_vision_with_retry(
                    provider, image_bytes, prompt, system, temperature, max_tokens
                )
                latency = (time.perf_counter() - start_time) * 1000.0
                
                logger.info(f"Successfully generated vision response using {provider} in {latency:.2f}ms")
                return LLMResponse(
                    text=text,
                    provider_used=provider,
                    latency_ms=latency,
                    fallback_triggered=fallback_triggered,
                )
            except Exception as exc:
                latency = (time.perf_counter() - start_time) * 1000.0
                logger.warning(
                    f"Vision provider {provider} failed after {latency:.2f}ms. Error: {exc}"
                )
                errors.append((provider, str(exc)))
                first_attempted = False
                fallback_triggered = True

        err_msg = f"All LLM vision providers failed. Errors: {errors}"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # ── Tenacity-wrapped execution blocks ────────────────────────────────────

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception(is_transient_error),
        reraise=True,
    )
    def _call_with_retry(
        self,
        provider: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Helper that routes to the specific client call and is wrapped in retries."""
        if provider == "groq":
            return self._call_groq(prompt, system, temperature, max_tokens)
        elif provider == "gemini":
            return self._call_gemini(prompt, system, temperature, max_tokens)
        else:
            raise PermanentProviderError(f"Unknown LLM provider: {provider}")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception(is_transient_error),
        reraise=True,
    )
    def _call_vision_with_retry(
        self,
        provider: str,
        image_bytes: bytes,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Helper that routes to the specific vision client call and is wrapped in retries."""
        if provider == "groq":
            return self._call_groq_vision(image_bytes, prompt, system, temperature, max_tokens)
        elif provider == "gemini":
            return self._call_gemini_vision(image_bytes, prompt, system, temperature, max_tokens)
        else:
            raise PermanentProviderError(f"Unknown LLM vision provider: {provider}")

    # ── Concrete Provider Implementations ────────────────────────────────────



    def _call_groq(
        self, prompt: str, system: str | None, temperature: float, max_tokens: int
    ) -> str:
        """Call Groq API using the groq python library."""
        api_key = settings.groq_api_key
        if not api_key:
            raise PermanentProviderError("GROQ_API_KEY is not configured in settings.")

        from groq import Groq

        client = Groq(api_key=api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content

    def _call_gemini(
        self, prompt: str, system: str | None, temperature: float, max_tokens: int
    ) -> str:
        """Call Gemini API using the new google-genai SDK."""
        api_key = settings.gemini_api_key
        if not api_key:
            raise PermanentProviderError("GEMINI_API_KEY is not configured in settings.")

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=config,
        )
        return response.text

    def _call_groq_vision(
        self,
        image_bytes: bytes,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call Groq API with multimodal image input."""
        api_key = settings.groq_api_key
        if not api_key:
            raise PermanentProviderError("GROQ_API_KEY is not configured in settings.")

        from groq import Groq

        client = Groq(api_key=api_key)
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        )

        completion = client.chat.completions.create(
            model=settings.groq_vision_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content

    def _call_gemini_vision(
        self,
        image_bytes: bytes,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call Gemini API with multimodal image input using the new google-genai SDK."""
        api_key = settings.gemini_api_key
        if not api_key:
            raise PermanentProviderError("GEMINI_API_KEY is not configured in settings.")

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        response = client.models.generate_content(
            model=settings.gemini_vision_model,
            contents=[image_part, prompt],
            config=config,
        )
        return response.text
