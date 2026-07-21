import os
import subprocess
from unittest import mock

import pytest

from config.settings import settings
from core.llm_provider import LLMProvider, PermanentProviderError


@pytest.fixture(autouse=True)
def mock_settings():
    """Ensure API keys and debug settings are mocked to allow tests to run without actual keys."""
    with mock.patch.object(settings, "bobshell_api_key", "mock-bob-key"), \
         mock.patch.object(settings, "bob_api_key", ""), \
         mock.patch.object(settings, "groq_api_key", "mock-groq-key"), \
         mock.patch.object(settings, "gemini_api_key", "mock-gemini-key"), \
         mock.patch.object(settings, "fallback_order", ["bob", "groq", "gemini"]):
        yield


def test_bob_success():
    """Test the happy path where Bob Shell CLI succeeds."""
    provider = LLMProvider()

    # Mock subprocess.run to return successful CompletedProcess
    mock_process = mock.Mock()
    mock_process.stdout = "Hello from Bob"
    mock_process.returncode = 0

    with mock.patch("subprocess.run", return_value=mock_process) as mock_run:
        response = provider.generate("Hi")
        
        assert response.text == "Hello from Bob"
        assert response.provider_used == "bob"
        assert response.fallback_triggered is False
        mock_run.assert_called_once()


def test_bob_transient_failure_falls_to_groq():
    """Test that a transient failure in Bob triggers retries and falls back to Groq."""
    provider = LLMProvider()

    # Mock subprocess.run to raise CalledProcessError with returncode 1 (transient)
    # Tenacity is configured for 2 attempts, so it will fail after 2 tries and fallback.
    error = subprocess.CalledProcessError(returncode=1, cmd="bob", stderr="SSO session expired")
    
    # Mock Groq client response
    mock_groq_client = mock.Mock()
    mock_completion = mock.Mock()
    mock_message = mock.Mock()
    mock_message.content = "Hello from Groq"
    mock_completion.choices = [mock.Mock(message=mock_message)]
    mock_groq_client.chat.completions.create.return_value = mock_completion

    with mock.patch("subprocess.run", side_effect=error) as mock_run, \
         mock.patch("groq.Groq", return_value=mock_groq_client) as mock_groq:
        
        response = provider.generate("Hi")

        assert response.text == "Hello from Groq"
        assert response.provider_used == "groq"
        assert response.fallback_triggered is True
        # Tenacity stops after 2 attempts
        assert mock_run.call_count == 2
        mock_groq.assert_called_once()


def test_bob_missing_command_falls_to_groq_immediately():
    """Test that a missing Bob CLI (FileNotFoundError) triggers immediate fallback (no retries)."""
    provider = LLMProvider()

    # Mock FileNotFoundError (missing CLI on PATH)
    error = FileNotFoundError(2, "No such file or directory")

    # Mock Groq client response
    mock_groq_client = mock.Mock()
    mock_completion = mock.Mock()
    mock_message = mock.Mock()
    mock_message.content = "Hello from Groq"
    mock_completion.choices = [mock.Mock(message=mock_message)]
    mock_groq_client.chat.completions.create.return_value = mock_completion

    with mock.patch("subprocess.run", side_effect=error) as mock_run, \
         mock.patch("groq.Groq", return_value=mock_groq_client) as mock_groq:
        
        response = provider.generate("Hi")

        assert response.text == "Hello from Groq"
        assert response.provider_used == "groq"
        assert response.fallback_triggered is True
        # FileNotFoundError is permanent, so it shouldn't retry (only 1 run call)
        assert mock_run.call_count == 1
        mock_groq.assert_called_once()


def test_bob_exit_code_9009_falls_to_groq_immediately():
    """Test that an exit code of 9009 (Windows command not found) triggers immediate fallback (no retries)."""
    provider = LLMProvider()

    # Mock CalledProcessError with exit code 9009
    error = subprocess.CalledProcessError(returncode=9009, cmd="bob", stderr="not recognized as an internal command")

    # Mock Groq client response
    mock_groq_client = mock.Mock()
    mock_completion = mock.Mock()
    mock_message = mock.Mock()
    mock_message.content = "Hello from Groq"
    mock_completion.choices = [mock.Mock(message=mock_message)]
    mock_groq_client.chat.completions.create.return_value = mock_completion

    with mock.patch("subprocess.run", side_effect=error) as mock_run, \
         mock.patch("groq.Groq", return_value=mock_groq_client) as mock_groq:
        
        response = provider.generate("Hi")

        assert response.text == "Hello from Groq"
        assert response.provider_used == "groq"
        assert response.fallback_triggered is True
        # 9009 is treated as permanent, so no retries (1 attempt only)
        assert mock_run.call_count == 1
        mock_groq.assert_called_once()


def test_fallback_chain_all_way_to_gemini():
    """Test that if Bob and Groq fail, the router falls back to Gemini."""
    provider = LLMProvider()

    # Bob fails with FileNotFoundError
    bob_error = FileNotFoundError(2, "No such file or directory")
    # Groq fails with a permanent exception (missing API key or Bad Request)
    groq_error = PermanentProviderError("Groq key missing")

    # Mock Gemini response
    mock_model = mock.Mock()
    mock_response = mock.Mock()
    mock_response.text = "Hello from Gemini"
    mock_model.generate_content.return_value = mock_response

    with mock.patch("subprocess.run", side_effect=bob_error), \
         mock.patch("groq.Groq", side_effect=groq_error), \
         mock.patch("google.generativeai.GenerativeModel", return_value=mock_model) as mock_gemini:
        
        response = provider.generate("Hi")

        assert response.text == "Hello from Gemini"
        assert response.provider_used == "gemini"
        assert response.fallback_triggered is True
        mock_gemini.assert_called_once()
