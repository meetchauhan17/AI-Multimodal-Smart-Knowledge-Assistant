"""
modules/rag/ingestion.py — Knowledge base document loader and vector store ingester.

Reads text/md/pdf files from data/knowledge_base/, chunks them, tags with metadata,
embeds using sentence-transformers, and stores in ChromaDB collections.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Any

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from core.logger import logger


def load_file(file_path: Path) -> List[Any]:
    """Load a document from the filesystem based on extension, returning langchain Documents."""
    ext = file_path.suffix.lower()
    try:
        if ext in (".txt", ".md"):
            # TextLoader handles text/markdown. Specifying UTF-8 is critical on Windows.
            loader = TextLoader(str(file_path), encoding="utf-8")
            return loader.load()
        elif ext == ".pdf":
            try:
                loader = PyPDFLoader(str(file_path))
                return loader.load()
            except ImportError:
                logger.warning(
                    f"pypdf package is not available. Skipping PDF file: {file_path.name}"
                )
                return []
        else:
            logger.warning(f"Unsupported file extension {ext} for {file_path}")
            return []
    except Exception as exc:
        logger.error(f"Error loading file {file_path.name}: {exc}")
        return []


class DocumentIngester:
    """Ingests documents from the knowledge base into separate ChromaDB collections."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

    def __init__(
        self,
        domain: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.domain = domain
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kb_dir = Path(settings.knowledge_base_dir)
        self.chroma_dir = Path(settings.chroma_persist_dir)
        
        # Initialize the Chroma client and embedding function
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        logger.debug(f"DocumentIngester ready (kb_dir={self.kb_dir}, chroma_dir={self.chroma_dir})")

    def ingest(self, refresh: bool = False) -> int:
        """Load, chunk, embed, and persist documents for the configured domain (or all if None)."""
        if refresh:
            logger.info("Resetting ChromaDB collections before ingestion...")
            try:
                for col in self.chroma_client.list_collections():
                    # chromadb 0.5.x returns Collection objects (use .name)
                    # chromadb 1.x returns plain strings directly
                    col_name = col if isinstance(col, str) else col.name
                    self.chroma_client.delete_collection(col_name)
            except Exception as e:
                logger.warning(f"Failed to clear collections programmatically: {e}. Re-initializing client.")

        # Determine target domains
        if self.domain:
            target_domains = [self.domain]
        else:
            if not self.kb_dir.exists():
                logger.error(f"Knowledge base directory does not exist: {self.kb_dir}")
                return 0
            target_domains = [d.name for d in self.kb_dir.iterdir() if d.is_dir()]

        logger.info(f"Identified domains for ingestion: {target_domains}")

        # Text splitter configuration
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        total_chunks = 0

        # Combine all collections into a main "all" collection
        all_collection = self.chroma_client.get_or_create_collection(
            name="all",
            embedding_function=self.emb_fn,
        )

        for domain in target_domains:
            domain_dir = self.kb_dir / domain
            files = [
                p for p in domain_dir.rglob("*")
                if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
            
            if not files:
                logger.info(f"No documents found for domain '{domain}'")
                continue

            logger.info(f"Ingesting {len(files)} file(s) for domain '{domain}'")
            domain_collection = self.chroma_client.get_or_create_collection(
                name=domain,
                embedding_function=self.emb_fn,
            )

            for file_path in files:
                documents = load_file(file_path)
                if not documents:
                    continue

                chunks = splitter.split_documents(documents)
                if not chunks:
                    continue

                ids = []
                docs_content = []
                metadatas = []

                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{domain}_{file_path.stem}_{idx}"
                    ids.append(chunk_id)
                    docs_content.append(chunk.page_content)
                    metadatas.append({
                        "domain": domain,
                        "source_file": file_path.name
                    })

                # Add to domain-specific collection
                domain_collection.add(
                    ids=ids,
                    documents=docs_content,
                    metadatas=metadatas
                )

                # Add to the unified combined collection
                all_collection.add(
                    ids=ids,
                    documents=docs_content,
                    metadatas=metadatas
                )

                total_chunks += len(chunks)
                logger.debug(f"Stored {len(chunks)} chunks from {file_path.name}")

        logger.info(f"Ingestion complete. Total chunks stored: {total_chunks}")
        return total_chunks


def ingest_all(refresh: bool = False) -> int:
    """Convenience helper to trigger full knowledge base ingestion."""
    ingester = DocumentIngester(domain=None)
    return ingester.ingest(refresh=refresh)


if __name__ == "__main__":
    # Visual check runner
    print("DocumentIngester CLI runner. Initiating full ingestion...")
    total = ingest_all(refresh=True)
    print(f"Full Ingestion Completed. Persisted {total} chunks in ChromaDB.")
