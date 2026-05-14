# RAG pipeline — retrieves context from FAISS and generates a Sydney insider tip via GPT-4o.
# Features:
#   - Similarity score thresholding (prevents hallucination on low-confidence retrievals)
#   - Confidence scoring (tracks retrieval quality)
#   - Metadata TTL validation (ensures data freshness)

import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, List, Optional, Tuple, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import SecretStr

from app.config import CHAT_MODEL, OPENAI_API_KEY, OPENAI_TIMEOUT, TOP_K_RESULTS
from app.rag.retriever import get_vector_store

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Minimum similarity score required; below this triggers "no relevant context" response
MIN_SIMILARITY_THRESHOLD: float = 0.70

# TTL rules: how long before data is considered stale (in seconds).
TTL_RULES: dict[str, int] = {
    "alert": 24 * 3600,
    "alerts": 24 * 3600,
    "event": 7 * 24 * 3600,
    "events": 7 * 24 * 3600,
}

# Confidence thresholds for categorizing retrieval quality
CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "high": 0.85,  # avg score > 0.85, multiple relevant docs
    "medium": 0.70,  # avg score > 0.70 or mixed quality
    "low": 0.0,  # below threshold, fallback used
}


class RetrievalResult(TypedDict):
    """Typed result of a retrieval with confidence metadata."""
    documents: List[Document]
    scores: List[float]
    confidence: str  # "high", "medium", "low"
    avg_score: float
    warning: Optional[str]  # e.g., "Data is 3 days old"


class PipelineResult(TypedDict):
    """API-safe result returned by the RAG pipeline."""
    tip: str
    confidence: str
    confidence_score: float
    sources: List[dict[str, Any]]
    warning: Optional[str]

# ============================================================================
# LangChain chain (built once at import time, reused across requests)
# ============================================================================

_PROMPT = PromptTemplate(
    input_variables=["context", "query", "vibe_context", "confidence_notice"],
    template=(
        "You are the Aura Health Companion, a local Sydney expert. "
        "Answer the user using ONLY the provided Context. "
        'Add a warm, conversational "Insider Tip" vibe. '
        "If the context mentions the user's current vibe ({vibe_context}), tailor the advice. "
        "{confidence_notice}"
        "\n\nContext: {context}\n\nQuery: {query}"
    ),
)


@lru_cache(maxsize=1)
def _get_chain():
    api_key = SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.7,
        api_key=api_key,
        timeout=OPENAI_TIMEOUT,
    )
    return _PROMPT | llm | StrOutputParser()


# ============================================================================
# RETRIEVAL HARDENING: Validation & Thresholding
# ============================================================================

def _ttl_key_for_metadata(metadata: dict[str, Any]) -> Optional[str]:
    source_type = str(
        metadata.get("type")
        or metadata.get("source_type")
        or metadata.get("category")
        or ""
    ).lower()
    if source_type in TTL_RULES:
        return source_type
    if "alert" in source_type:
        return "alerts"
    if "event" in source_type:
        return "events"
    return None


def is_data_stale(doc: Document) -> Tuple[bool, Optional[str]]:
    """
    Check if a document has exceeded its TTL.

    Returns (is_stale: bool, warning: Optional[str])
    """
    try:
        ttl_key = _ttl_key_for_metadata(doc.metadata)
        if ttl_key is None:
            return False, None

        scraped_at = doc.metadata.get("scraped_at")
        if not scraped_at:
            return True, "scraped_at metadata missing"

        scraped_datetime = datetime.fromisoformat(scraped_at.replace("Z", "+00:00"))
        if scraped_datetime.tzinfo is None:
            scraped_datetime = scraped_datetime.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)

        ttl_seconds = TTL_RULES[ttl_key]
        age_seconds = (now - scraped_datetime).total_seconds()

        if age_seconds > ttl_seconds:
            age_human = _format_age(age_seconds)
            return True, f"Data is {age_human} old (TTL: {ttl_seconds // 3600}h)"

        # Warn if approaching TTL
        if age_seconds > ttl_seconds * 0.8:
            age_human = _format_age(age_seconds)
            return False, f"Data is {age_human} old (refresh soon)"

        return False, None

    except Exception as exc:
        logger.warning("Failed to parse scraped_at: %s", exc)
        return True, "scraped_at parsing error"


def _format_age(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m"
    elif seconds < 86400:
        return f"{int(seconds / 3600)}h"
    else:
        return f"{int(seconds / 86400)}d"


def _relevance_to_similarity(raw_score: float) -> float:
    """
    Clamp vector-store relevance scores into a 0..1 similarity value.

    LangChain's `similarity_search_with_relevance_scores` already returns higher
    scores for better matches. Clamping protects the API contract if a backend
    returns a value just outside the expected range.
    """
    return max(0.0, min(1.0, raw_score))


def _distance_to_similarity(distance: float) -> float:
    """Convert a raw vector distance where 0 is best into 0..1 similarity."""
    if distance < 0:
        return 0.0
    return 1.0 / (1.0 + distance)


def retrieve_with_scoring(query: str, top_k: int = TOP_K_RESULTS) -> RetrievalResult:
    """
    Retrieve documents with similarity scores and validate for staleness.

    Returns RetrievalResult with:
    - documents: filtered list of non-stale docs passing threshold
    - scores: cosine similarity scores [0..1]
    - confidence: "high" / "medium" / "low"
    - avg_score: average similarity of returned docs
    - warning: optional staleness warning
    """
    store = get_vector_store()
    if store is None:
        return {
            "documents": [],
            "scores": [],
            "confidence": "low",
            "avg_score": 0.0,
            "warning": "Vector store not initialized",
        }

    try:
        raw_results = store.similarity_search_with_relevance_scores(query, k=top_k)
        results = [
            (doc, _relevance_to_similarity(float(score)))
            for doc, score in raw_results
        ]
    except AttributeError:
        try:
            raw_results = store.similarity_search_with_score(query, k=top_k)
            results = [
                (doc, _distance_to_similarity(float(score)))
                for doc, score in raw_results
            ]
        except AttributeError:
            logger.warning("Vector store doesn't expose scored search; using conservative fallback")
            results = [(doc, 0.0) for doc in store.similarity_search(query, k=top_k)]

    raw_docs = [doc for doc, _ in results]
    raw_scores = [score for _, score in results]

    # Filter out stale documents
    filtered_docs = []
    filtered_scores = []
    staleness_warnings = []

    for doc, score in zip(raw_docs, raw_scores):
        is_stale, warning = is_data_stale(doc)
        if is_stale:
            staleness_warnings.append(warning)
            continue  # Skip stale documents

        filtered_docs.append(doc)
        filtered_scores.append(score)

    # Check if any docs pass threshold
    valid_docs = []
    valid_scores = []

    for doc, score in zip(filtered_docs, filtered_scores):
        similarity = _relevance_to_similarity(float(score))
        if similarity > MIN_SIMILARITY_THRESHOLD:
            valid_docs.append(doc)
            valid_scores.append(similarity)

    # Calculate confidence
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    confidence = "low"

    if len(valid_docs) >= 2 and avg_score >= CONFIDENCE_THRESHOLDS["high"]:
        confidence = "high"
    elif len(valid_docs) >= 1 and avg_score >= CONFIDENCE_THRESHOLDS["medium"]:
        confidence = "medium"

    # Compile warning
    warning_msg = None
    if staleness_warnings and not valid_docs:
        warning_msg = f"Retrieved docs are stale: {staleness_warnings[0]}"
    elif staleness_warnings:
        logger.info("Filtered out stale docs: %s", staleness_warnings)

    return {
        "documents": valid_docs,
        "scores": valid_scores,
        "confidence": confidence,
        "avg_score": avg_score,
        "warning": warning_msg,
    }


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def _source_metadata(doc: Document, score: float) -> dict[str, Any]:
    return {
        "source": doc.metadata.get("source"),
        "name": doc.metadata.get("name"),
        "type": doc.metadata.get("type") or doc.metadata.get("category"),
        "scraped_at": doc.metadata.get("scraped_at"),
        "similarity": round(score, 4),
    }


def generate_insider_tip(query: str, vibe_context: Optional[str] = None) -> PipelineResult:
    """
    Full RAG cycle:
      1. Load the FAISS vector store
      2. Retrieve the top-k most relevant chunks via similarity search
      3. Inject context + vibe into the Aura prompt
      4. Stream through ChatOpenAI (gpt-4o) and return the string response

    Raises openai.APITimeoutError / APIConnectionError / RateLimitError on
    upstream failures — callers (routes.py) map these to HTTP status codes.
    """
    retrieval = retrieve_with_scoring(query)
    docs = retrieval["documents"]
    scores = retrieval["scores"]

    if not docs:
        warning = retrieval["warning"] or "No fresh context passed the similarity threshold."
        return {
            "tip": (
                "I do not have enough fresh, relevant Sydney context to answer that safely right now. "
                "Try a more specific query or refresh the knowledge base."
            ),
            "confidence": "low",
            "confidence_score": 0.0,
            "sources": [],
            "warning": warning,
        }

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    confidence_notice = (
        f"Retrieval confidence is {retrieval['confidence']} "
        f"({retrieval['avg_score']:.2f}). If evidence is incomplete, say so briefly."
    )

    try:
        tip = _get_chain().invoke(
            {
                "context": context,
                "query": query,
                "vibe_context": vibe_context or "general",
                "confidence_notice": confidence_notice,
            }
        )
        return {
            "tip": tip,
            "confidence": retrieval["confidence"],
            "confidence_score": round(retrieval["avg_score"], 4),
            "sources": [_source_metadata(doc, score) for doc, score in zip(docs, scores)],
            "warning": retrieval["warning"],
        }
    except RateLimitError:
        logger.warning("OpenAI rate limit hit for query: %.60s…", query)
        raise
    except APITimeoutError:
        logger.error("OpenAI request timed out for query: %.60s…", query)
        raise
    except APIConnectionError:
        logger.error("OpenAI connection error for query: %.60s…", query)
        raise
