import pytest
from pathlib import Path

from modules.rag.ingestion import ingest_all
from modules.rag.retriever import retrieve, answer_with_rag




def test_retrieval_agriculture():
    """Verify retrieval for agriculture domain queries."""
    results = retrieve("benefits of crop rotation")
    assert len(results) > 0
    # The top result should be from agriculture and mention crop rotation
    top_result = results[0]
    assert top_result["domain"] == "agriculture"
    assert "crop_rotation.txt" in top_result["source_file"]
    assert "disease" in top_result["text"].lower() or "pest" in top_result["text"].lower()


def test_retrieval_college():
    """Verify retrieval for college domain queries."""
    results = retrieve("Introduction to Computer Science course code credits")
    assert len(results) > 0
    top_result = results[0]
    assert top_result["domain"] == "college"
    assert "course_catalog.txt" in top_result["source_file"]
    assert "cs-101" in top_result["text"].lower()


def test_retrieval_healthcare():
    """Verify retrieval for healthcare domain queries."""
    results = retrieve("hospital emergency contact number")
    assert len(results) > 0
    top_result = results[0]
    assert top_result["domain"] == "healthcare"
    assert "hospital_services.txt" in top_result["source_file"]
    assert "911" in top_result["text"]


def test_retrieval_library():
    """Verify retrieval for library domain queries."""
    results = retrieve("Evelyn Sterling AI Horizon recommended reading")
    assert len(results) > 0
    top_result = results[0]
    assert top_result["domain"] == "library"
    assert "book_selection.txt" in top_result["source_file"]
    assert "sterling" in top_result["text"].lower()


def test_retrieval_monuments():
    """Verify retrieval for monuments domain queries."""
    results = retrieve("Stonehenge standing stones Wiltshire England")
    assert len(results) > 0
    top_result = results[0]
    assert top_result["domain"] == "monuments"
    assert "historic_site.txt" in top_result["source_file"]
    assert "stonehenge" in top_result["text"].lower()


def test_retrieval_museums():
    """Verify retrieval for museums domain queries."""
    results = retrieve("Cretaceous Giants natural history exhibit")
    assert len(results) > 0
    top_result = results[0]
    assert top_result["domain"] == "museums"
    assert "history_exhibit.txt" in top_result["source_file"]
    assert "cretaceous" in top_result["text"].lower()


def test_retrieval_tourism():
    """Verify retrieval for tourism domain queries."""
    results = retrieve("Grand Plaza downtown downtown amenities wifi")
    assert len(results) > 0
    top_result = results[0]
    assert top_result["domain"] == "tourism"
    assert "hotel_info.txt" in top_result["source_file"]
    assert "wi-fi" in top_result["text"].lower() or "service" in top_result["text"].lower()


def test_cross_domain_isolation():
    """Assert that a college query does not pull tourism chunks."""
    # Searching for admission requirements should classify as college or retrieve college info
    results = retrieve("minimum GPA undergraduate admission requirements", domain=None)
    
    assert len(results) > 0
    # Verify that no retrieved document chunk belongs to the tourism domain
    for r in results:
        assert r["domain"] != "tourism", f"Tourism chunk wrongly retrieved: {r['source_file']}"
        assert r["domain"] == "college"  # Should strictly match college domain
