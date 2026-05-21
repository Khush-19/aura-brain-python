"""End-to-end RAG pipeline for Aura Brain."""

from __future__ import annotations

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

from app.config import CHAT_MODEL, MIN_SIMILARITY_THRESHOLD, OPENAI_API_KEY, OPENAI_TIMEOUT, TOP_K_RESULTS
from app.rag.retriever import retrieve_documents

logger = logging.getLogger(__name__)

TTL_RULES: dict[str, int] = {
    "alert": 24 * 3600,
    "alerts": 24 * 3600,
    "event": 7 * 24 * 3600,
    "events": 7 * 24 * 3600,
}

CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "high": 0.85,
    "medium": MIN_SIMILARITY_THRESHOLD,
    "low": 0.0,
}


class RetrievalResult(TypedDict):
    documents: List[Document]
    scores: List[float]
    confidence: str
    avg_score: float
    warning: Optional[str]


class PipelineResult(TypedDict):
    tip: str
    confidence: str
    confidence_score: float
    sources: List[dict[str, Any]]
    warning: Optional[str]


_PROMPT = PromptTemplate(
    input_variables=["context", "query", "vibe_context", "confidence_notice"],
    template=(
        "You are Aura Brain, a local Sydney insider tips engine. "
        "Answer using only the provided context. If the context does not support "
        "a claim, say briefly that you do not have enough fresh evidence. "
        "Keep the tone warm, concise, and useful. "
        "Tailor the advice to this user context when relevant: {vibe_context}. "
        "{confidence_notice}\n\n"
        "Context:\n{context}\n\n"
        "Query: {query}"
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


def _format_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h"
    return f"{int(seconds / 86400)}d"


def is_data_stale(doc: Document) -> Tuple[bool, Optional[str]]:
    try:
        ttl_key = _ttl_key_for_metadata(doc.metadata)
        if ttl_key is None:
            return False, None

        scraped_at = doc.metadata.get("scraped_at")
        if not scraped_at:
            return True, "scraped_at metadata missing"

        scraped_datetime = datetime.fromisoformat(str(scraped_at).replace("Z", "+00:00"))
        if scraped_datetime.tzinfo is None:
            scraped_datetime = scraped_datetime.replace(tzinfo=timezone.utc)

        ttl_seconds = TTL_RULES[ttl_key]
        age_seconds = (datetime.now(timezone.utc) - scraped_datetime).total_seconds()

        if age_seconds > ttl_seconds:
            return True, f"Data is {_format_age(age_seconds)} old (TTL: {ttl_seconds // 3600}h)"

        if age_seconds > ttl_seconds * 0.8:
            return False, f"Data is {_format_age(age_seconds)} old (refresh soon)"

        return False, None
    except Exception as exc:
        logger.warning("Failed to parse scraped_at metadata: %s", exc)
        return True, "scraped_at parsing error"


def retrieve_with_scoring(query: str, top_k: int = TOP_K_RESULTS) -> RetrievalResult:
    # 1. Fetch results
    results: List[Document] = retrieve_documents(query, top_k=top_k)
    
    valid_docs: List[Document] = []
    valid_scores: List[float] = []
    staleness_warnings: List[str] = []
    threshold_warnings = 0

    # 2. Process each document
    for doc in results:
        # Pylance now recognizes 'doc' as Document and 'score' as float
        is_stale, warning = is_data_stale(doc)
        
        if is_stale:
            if warning:
                staleness_warnings.append(warning)
            continue

        # Ensure similarity score is a valid float
        try:
            score = doc.metadata.get("similarity_score", 0.0)
            similarity = max(0.0, min(1.0, float(score)))
        except (ValueError, TypeError):
            similarity = 0.0

        if similarity >= MIN_SIMILARITY_THRESHOLD:
            valid_docs.append(doc)
            valid_scores.append(similarity)
        else:
            threshold_warnings += 1

    # 3. Calculate metrics
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    # 4. Determine confidence level
    confidence = "low"
    if len(valid_docs) >= 2 and avg_score >= CONFIDENCE_THRESHOLDS["high"]:
        confidence = "high"
    elif len(valid_docs) >= 1 and avg_score >= CONFIDENCE_THRESHOLDS["medium"]:
        confidence = "medium"

    # 5. Handle warnings
    warning_msg = None
    if staleness_warnings and not valid_docs:
        warning_msg = f"Retrieved docs are stale: {staleness_warnings[0]}"
    elif threshold_warnings and not valid_docs:
        warning_msg = "Retrieved docs did not meet the similarity threshold."
    elif staleness_warnings:
        logger.info("Filtered out stale docs: %s", staleness_warnings)

    return {
        "documents": valid_docs,
        "scores": valid_scores,
        "confidence": confidence,
        "avg_score": avg_score,
        "warning": warning_msg,
    }

def _source_metadata(doc: Document, score: float) -> dict[str, Any]:
    return {
        "source": doc.metadata.get("source"),
        "name": doc.metadata.get("name"),
        "type": doc.metadata.get("type") or doc.metadata.get("category"),
        "scraped_at": doc.metadata.get("scraped_at"),
        "similarity": round(score, 4),
    }


def generate_insider_tip(query: str, vibe_context: Optional[str] = None) -> PipelineResult:
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
                "vibe_context": vibe_context or "balanced",
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
        logger.warning("OpenAI rate limit hit for query: %.60s", query)
        raise
    except APITimeoutError:
        logger.error("OpenAI request timed out for query: %.60s", query)
        raise
    except APIConnectionError:
        logger.error("OpenAI connection error for query: %.60s", query)
        raise
