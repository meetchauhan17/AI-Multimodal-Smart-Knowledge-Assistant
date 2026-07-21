import os
from pathlib import Path

import chromadb
import pytest

from config.settings import settings
from modules.rag.ingestion import ingest_all


def _db_is_populated() -> bool:
    """Return True if the ChromaDB 'all' collection already has documents."""
    chroma_dir = Path(settings.chroma_persist_dir)
    if not chroma_dir.exists():
        return False
    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        col = client.get_collection("all")
        return col.count() > 0
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def setup_vector_db():
    """Ensure vector database is populated exactly once per test session.

    Rebuilds from scratch when:
      - ChromaDB directory / 'all' collection does not exist yet, OR
      - The FORCE_INGEST=1 environment variable is set.

    Skips rebuild on subsequent runs to keep test startup fast.
    """
    force_refresh = os.environ.get("FORCE_INGEST", "0") == "1"
    already_populated = _db_is_populated()

    if force_refresh or not already_populated:
        print(f"\n[Test Setup] Building vector database (force={force_refresh})...")
        total_chunks = ingest_all(refresh=True)
        print(f"[Test Setup] Ingestion complete. Persisted {total_chunks} chunk(s).")
        assert total_chunks > 0, "No chunks were ingested from the knowledge base."
    else:
        print("\n[Test Setup] ChromaDB already populated — skipping rebuild. "
              "Set FORCE_INGEST=1 to force a rebuild.")

    yield
