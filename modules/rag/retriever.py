"""
modules/rag/retriever.py — Semantic document chunk retriever and context-based RAG answering.

Queries ChromaDB collections using sentence-transformer embeddings.
Provides lightweight query classification to select the correct domain automatically.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, List

import chromadb
from chromadb.utils import embedding_functions

from config.settings import settings
from core.logger import logger
from core.llm_provider import LLMProvider
from modules.web_search import search_web

# Lightweight keyword maps for domain classification
DOMAIN_KEYWORDS = {
    "college": ["admission", "course", "degree", "class", "tuition", "major", "campus", "academic", "student", "faculty", "program", "college", "university"],
    "tourism": ["tourist", "travel", "visit", "hotel", "attraction", "destination", "guide", "sightseeing", "vacation", "hike", "beach", "spot"],
    "healthcare": ["health", "medical", "doctor", "hospital", "patient", "clinic", "treatment", "disease", "medicine", "wellness", "symptom", "pain", "nurse"],
    "agriculture": ["farm", "crop", "soil", "harvest", "plant", "agriculture", "farmer", "fertilizer", "irrigation", "wheat", "rice", "livestock"],
    "library": ["book", "library", "borrow", "catalog", "reading", "member", "journal", "checkout", "librarian", "borrowing"],
    "museums": ["museum", "exhibit", "art", "artifact", "gallery", "display", "collection", "curator", "sculpture", "painting"],
    "monuments": ["monument", "historic", "heritage", "statue", "ruins", "castle", "ancient", "tomb", "temple", "fort", "landmark"]
}


def classify_domain_by_keywords(query: str) -> str | None:
    """Return the domain name if query contains clear keyword matches, else None."""
    query_lower = query.lower()
    matches = {domain: 0 for domain in DOMAIN_KEYWORDS}
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            # Substring match with boundary helpers
            if kw in query_lower:
                matches[domain] += 1
                
    max_matches = max(matches.values())
    if max_matches > 0:
        winners = [dom for dom, count in matches.items() if count == max_matches]
        if len(winners) == 1:
            logger.info(f"Classified query to domain {winners[0]!r} via keyword matches.")
            return winners[0]
            
    return None


def retrieve(query: str, domain: str | None = None, top_k: int = 4) -> List[Dict[str, Any]]:
    """Retrieve the most relevant document chunks for the query.

    If domain is not specified, classifies the query using keyword heuristics.
    If classification has low confidence, falls back to the combined "all" collection.

    Args:
        query:  The search string.
        domain: Target knowledge domain, or None to classify.
        top_k:  Number of chunks to return.

    Returns:
        A list of dictionaries containing:
            - text: Chunk content.
            - source_file: Basename of original document.
            - domain: Knowledge domain of chunk.
            - relevance_score: Score between 0.0 and 1.0.
    """
    chroma_dir = Path(settings.chroma_persist_dir)
    if not chroma_dir.exists():
        logger.warning(f"ChromaDB directory does not exist: {chroma_dir}. Returning empty list.")
        return []

    # Get Chroma client
    client = chromadb.PersistentClient(path=str(chroma_dir))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Classify domain if not provided
    resolved_domain = domain
    if not resolved_domain:
        resolved_domain = classify_domain_by_keywords(query)
        if not resolved_domain:
            resolved_domain = "all"
            logger.info("Classification returned low confidence. Querying unified 'all' collection.")

    logger.info(f"Retrieving from ChromaDB collection {resolved_domain!r} (top_k={top_k})")
    
    try:
        collection = client.get_collection(name=resolved_domain, embedding_function=emb_fn)
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
    except Exception as exc:
        logger.warning(f"Failed to query collection {resolved_domain!r}: {exc}. Trying fallback collection 'all'...")
        try:
            collection = client.get_collection(name="all", embedding_function=emb_fn)
            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )
            resolved_domain = "all"
        except Exception as fallback_exc:
            logger.error(f"Failed to query fallback collection 'all': {fallback_exc}")
            return []

    # Parse and structure results
    retrieved_docs = []
    
    # Chroma returns lists of lists since it supports batch queries
    if not results or not results["documents"] or len(results["documents"][0]) == 0:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
    distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)

    for doc, meta, dist in zip(documents, metadatas, distances):
        # Convert distance to relevance score. Cosine distance 0 means identical.
        # score = 1 / (1 + d) guarantees a range of 0.0 to 1.0.
        score = 1.0 / (1.0 + float(dist))
        
        retrieved_docs.append({
            "text": doc,
            "source_file": meta.get("source_file", "unknown"),
            "domain": meta.get("domain", resolved_domain),
            "relevance_score": score
        })

    return retrieved_docs


def answer_with_rag(query: str, domain: str | None = None) -> Dict[str, Any]:
    """Retrieve relevant chunks and generate an answer using the LLM provider.

    If retrieved chunks have high relevance scores the answer is grounded in
    the knowledge-base context (RAG mode).  If the best score is below the
    relevance threshold the query is off-topic for the knowledge base and the
    LLM answers from its general knowledge (General mode), satisfying Module 1
    requirement (c): "The assistant shall answer the question in text."

    Args:
        query:  The search/question string.
        domain: Specific domain collection filter, or None to auto-detect.

    Returns:
        A dictionary containing:
            - answer: Natural-language response.
            - sources: Unique list of source files used (empty for general answers).
            - domain_detected: The domain searched.
            - provider_used: The LLM model/platform that generated the answer.
    """
    # Threshold below which we consider RAG context irrelevant and use general LLM.
    # score = 1/(1+distance); on-topic queries score ~0.70+, off-topic ~0.50-0.55.
    RELEVANCE_THRESHOLD = 0.62

    # 1. Retrieve chunks
    chunks = retrieve(query, domain=domain, top_k=4)

    # Detect domain from first chunk or fall back
    domain_detected = chunks[0]["domain"] if chunks else (domain or "all")
    sources = list(set(c["source_file"] for c in chunks if c.get("source_file")))

    # 2. Decide mode: RAG-grounded vs General Knowledge
    best_score = max((c["relevance_score"] for c in chunks), default=0.0)
    use_rag_context = chunks and best_score >= RELEVANCE_THRESHOLD

    if use_rag_context:
        # Build context block from relevant chunks
        context_parts = []
        for idx, c in enumerate(chunks):
            context_parts.append(f"Document chunk {idx+1} (Source: {c['source_file']}):\n{c['text']}")
        context_text = "\n\n".join(context_parts)

        system_instruction = (
            "You are an expert, highly accurate multimodal knowledge assistant. "
            "Answer the user's question explicitly based on the provided Context block. "
            "Rules:\n"
            "1. Rely strictly on the context facts.\n"
            "2. If the context does not contain the answer, you must clearly state that you do not know instead of guessing.\n"
            "3. Do not invent or hallucinate information.\n"
            "4. Keep your response factual, concise, and highly accurate."
        )
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
        logger.info(f"RAG mode: best relevance score {best_score:.3f} >= {RELEVANCE_THRESHOLD}. Using knowledge-base context.")
    else:
        # No relevant context in local knowledge base — perform live web search for real-time news
        logger.info(f"General mode: best relevance score {best_score:.3f} < {RELEVANCE_THRESHOLD}. Fetching live web search...")
        web_results = search_web(query, max_results=4)
        sources = []  # no local KB sources used

        if web_results:
            web_context_parts = []
            for idx, item in enumerate(web_results):
                web_context_parts.append(f"Web Source {idx+1} ({item['title']}):\n{item['snippet']}")
            web_context_text = "\n\n".join(web_context_parts)

            system_instruction = (
                "You are an expert, highly accurate assistant with live web search capabilities. "
                "Answer the user's question using ONLY the live web search results provided below. "
                "If the search results do not provide a clear answer or if the query is nonsensical, state that you cannot find a reliable answer. "
                "Be direct, informative, and summarize the key facts clearly without hallucinations."
            )
            user_prompt = f"Live Web Search Context:\n{web_context_text}\n\nQuestion: {query}"
            logger.info(f"Live web search context added with {len(web_results)} snippets.")
        else:
            system_instruction = (
                "You are a highly accurate, expert knowledge assistant. "
                "Answer the user's question factually and concisely using your general knowledge. "
                "If the question is gibberish or you are unsure, state that you do not understand or do not know. "
                "Do not hallucinate or guess."
            )
            user_prompt = query

    # 3. Generate answer via LLMProvider
    try:
        provider = LLMProvider()
        response = provider.generate(
            prompt=user_prompt,
            system=system_instruction,
            temperature=0.0  # zero temperature for maximum accuracy and zero hallucinations
        )
        return {
            "answer": response.text,
            "sources": sources,
            "domain_detected": domain_detected,
            "provider_used": response.provider_used,
        }
    except Exception as exc:
        err_msg = f"Failed to generate RAG response: {exc}"
        logger.error(err_msg)
        return {
            "answer": f"Error: {err_msg}",
            "sources": sources,
            "domain_detected": domain_detected,
            "provider_used": "",
        }



# ── Scaffold Class Wrapper for Backwards Compatibility ──────────────────────

class RAGRetriever:
    """Retrieves relevant document chunks (wrapped class)."""

    def __init__(
        self,
        domain: str | None = None,
        top_k: int = 5,
        embed_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "knowledge_base",
    ) -> None:
        self.domain = domain
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Any]:
        """Scaffold method mapping to retrieve function returning Document schemas."""
        from core.schemas import Document
        
        raw_results = retrieve(query, domain=self.domain, top_k=self.top_k)
        docs = []
        for r in raw_results:
            docs.append(Document(
                content=r["text"],               # schema field is 'content', not 'page_content'
                source=r["source_file"],         # schema field is 'source', not inside metadata
                domain=r["domain"],
            ))
        return docs
