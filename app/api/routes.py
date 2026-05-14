# API route definitions for Aura Brain.
# /api/v1/query  — generate a personalised Sydney insider tip
# /api/v1/ingest — scrape sources and populate the vector store

import logging
import re
import time
from collections import defaultdict, deque
from threading import Lock
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from openai import APIConnectionError, APITimeoutError, OpenAIError, RateLimitError
from pydantic import BaseModel, Field, field_validator

from app.rag.ingestion import SYDNEY_SOURCES, Scraper, SourceConfig
from app.rag.pipeline import generate_insider_tip
from app.rag.retriever import ingest_documents

logger = logging.getLogger(__name__)

router = APIRouter()

_SAFE_VIBE_RE = re.compile(r"[^A-Za-z0-9 ]+")
_PROMPT_CONTROL_RE = re.compile(
    r"\b(ignore|instruction|system|developer|prompt|override|jailbreak|roleplay|assistant|script)\b",
    re.IGNORECASE,
)


def sanitize_vibe(vibe: str) -> str:
    """Allow only alphanumeric characters and spaces in vibe text."""
    if _PROMPT_CONTROL_RE.search(vibe or ""):
        return "balanced"
    sanitized = _SAFE_VIBE_RE.sub("", vibe or "")
    sanitized = " ".join(sanitized.split())
    if _PROMPT_CONTROL_RE.search(sanitized):
        return "balanced"
    return sanitized[:60] or "balanced"


class InMemoryRateLimiter:
    """Simple per-user sliding-window limiter for a single API process."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, user_id: str) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            hits = self._hits[user_id]
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - hits[0])))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)},
                )

            hits.append(now)


query_rate_limiter = InMemoryRateLimiter(limit=30, window_seconds=60)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="Natural-language question")
    user_id: str = Field(min_length=1, max_length=128, description="Unique user identifier")
    aura_score: float = Field(default=50.0, ge=0.0, le=100.0)
    vibe: str = Field(default="balanced", max_length=80, description="Alphanumeric vibe descriptor")

    @field_validator("vibe")
    @classmethod
    def validate_vibe(cls, value: str) -> str:
        return sanitize_vibe(value)


class QueryResponse(BaseModel):
    tip: str
    user_id: str
    confidence: str
    confidence_score: float
    sources: List[dict]
    warning: Optional[str] = None


class IngestRequest(BaseModel):
    # Optional list of custom URLs; falls back to the curated SYDNEY_SOURCES list
    urls: Optional[List[str]] = Field(
        default=None,
        description="Custom URLs to scrape. Leave empty to use default Sydney sources.",
    )


class IngestResponse(BaseModel):
    chunks_ingested: int
    sources_scraped: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "service": "Aura Brain"}


@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    summary="Generate a personalised Sydney insider tip",
    tags=["rag"],
)
def query_insider_tip(payload: QueryRequest):
    query_rate_limiter.check(payload.user_id)

    try:
        result = generate_insider_tip(payload.query, vibe_context=payload.vibe)
    except APITimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="OpenAI request timed out — please retry.",
        )
    except APIConnectionError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not reach OpenAI — please retry.",
        )
    except RateLimitError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="OpenAI rate limit reached — slow down and retry.",
        )
    except OpenAIError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI is not configured or temporarily unavailable.",
        )
    except Exception as exc:
        logger.exception("Unexpected error in query pipeline: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )

    return QueryResponse(
        tip=result["tip"],
        user_id=payload.user_id,
        confidence=result["confidence"],
        confidence_score=result["confidence_score"],
        sources=result["sources"],
        warning=result["warning"],
    )


@router.post(
    "/api/v1/ingest",
    response_model=IngestResponse,
    summary="Scrape Sydney sources and populate the vector store",
    tags=["rag"],
)
def ingest(payload: IngestRequest):
    scraper = Scraper()

    if payload.urls:
        sources = [
            SourceConfig(url=url, name=url, source_type="custom") for url in payload.urls
        ]
    else:
        sources = SYDNEY_SOURCES

    try:
        docs = scraper.scrape_all(sources)
    except Exception as exc:
        logger.exception("Scraping failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch content from one or more sources.",
        )

    if not docs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No usable content scraped. Check source URLs and try again.",
        )

    try:
        chunks = ingest_documents(docs)
    except Exception as exc:
        logger.exception("Vector store ingestion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Content scraped but failed to embed and store.",
        )

    return IngestResponse(chunks_ingested=chunks, sources_scraped=len(docs))
