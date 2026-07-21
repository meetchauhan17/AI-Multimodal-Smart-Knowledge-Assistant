import os
from pathlib import Path
from unittest import mock

import pytest
import requests

from modules.image_generation import generate_image


@pytest.fixture(autouse=True)
def clean_generated_images():
    """Ensure generated_images directory is empty or cleaned up after tests."""
    yield
    # Clean up gen_*.png files inside generated_images/
    directory = Path("generated_images")
    if directory.exists():
        for f in directory.glob("gen_*.png"):
            try:
                f.unlink()
            except OSError:
                pass


def test_safety_check_rejections():
    """Verify empty prompts or safety-violating terms are rejected client-side."""
    # Empty prompt
    res_empty = generate_image("   ")
    assert res_empty["image_path"] == ""
    assert "cannot be empty" in res_empty["error"]

    # Restricted keyword
    res_harm = generate_image("generate a violent gore scene")
    assert res_harm["image_path"] == ""
    assert "safety guidelines" in res_harm["error"]
    assert "gore" in res_harm["error"]


@mock.patch("google.generativeai.ImageGenerationModel", create=True)
def test_generate_image_gemini_success(mock_model_class):
    """Test successful image generation path using Google Gemini."""
    # Setup mock return structure
    mock_img = mock.Mock()
    mock_img.image.image_bytes = b"fake_gemini_image_data"
    
    mock_result = mock.Mock()
    mock_result.generated_images = [mock_img]
    
    mock_model_instance = mock.Mock()
    mock_model_instance.generate_images.return_value = mock_result
    mock_model_class.return_value = mock_model_instance

    # We mock gemini_api_key in settings to guarantee the Gemini code path runs
    with mock.patch("config.settings.settings.gemini_api_key", "fake_key"):
        result = generate_image("A cute puppy")
        
        assert result["error"] is None
        assert result["provider_used"] == "gemini"
        assert result["image_path"] != ""
        
        saved_path = Path(result["image_path"])
        assert saved_path.exists()
        assert saved_path.read_bytes() == b"fake_gemini_image_data"


@mock.patch("google.generativeai.ImageGenerationModel", create=True)
@mock.patch("requests.get")
def test_generate_image_gemini_failure_pollinations_success(mock_get, mock_model_class):
    """Test that when Gemini fails, it falls back to Pollinations.ai GET request."""
    # Gemini raises exception
    mock_model_instance = mock.Mock()
    mock_model_instance.generate_images.side_effect = Exception("Gemini Quota Exceeded")
    mock_model_class.return_value = mock_model_instance

    # Mock requests.get response for Pollinations
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = b"fake_pollinations_image_data"
    mock_get.return_value = mock_response

    with mock.patch("config.settings.settings.gemini_api_key", "fake_key"):
        result = generate_image("A cute kitten")
        
        # Verify fallback triggered successfully
        assert result["error"] is None
        assert result["provider_used"] == "pollinations"
        assert result["image_path"] != ""
        
        saved_path = Path(result["image_path"])
        assert saved_path.exists()
        assert saved_path.read_bytes() == b"fake_pollinations_image_data"
        
        # Verify requests.get was called with the prompt
        mock_get.assert_called_once()
        called_url = mock_get.call_args[0][0]
        assert "pollinations" in called_url


@mock.patch("google.generativeai.ImageGenerationModel", create=True)
@mock.patch("requests.get")
def test_generate_image_all_fail(mock_get, mock_model_class):
    """Verify that when both providers fail, an error dictionary is returned gracefully."""
    # Gemini fails
    mock_model_instance = mock.Mock()
    mock_model_instance.generate_images.side_effect = Exception("Gemini fails")
    mock_model_class.return_value = mock_model_instance

    # Pollinations fails
    mock_get.side_effect = requests.RequestException("Pollinations timeout")

    with mock.patch("config.settings.settings.gemini_api_key", "fake_key"):
        result = generate_image("A beautiful sunset")
        
        assert result["image_path"] == ""
        assert result["provider_used"] == ""
        assert result["error"] is not None
        assert "All image generation providers failed" in result["error"]
