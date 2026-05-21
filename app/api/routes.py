# API route definitions for Aura Brain.
# /api/v1/query  - generate a personalised Sydney insider tip
# /api/v1/ingest - scrape sources and populate the vector store

import logging
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from openai import APIConnectionError, APITimeoutError, OpenAIError, RateLimitError
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from app.config import VECTOR_STORE
from app.rag.ingestion import SYDNEY_SOURCES, Scraper, SourceConfig
from app.rag.pipeline import generate_insider_tip
from app.rag.retriever import ingest_documents
from app.squad_up.icebreaker import generate_icebreaker_prompt
from app.utils.validation import sanitize_vibe

logger = logging.getLogger(__name__)

router = APIRouter()

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
    sources: List[dict[str, Any]]
    warning: Optional[str] = None


class IngestRequest(BaseModel):
    urls: Optional[List[HttpUrl]] = Field(
        default=None,
        description="Custom URLs to scrape. Leave empty to use default Sydney sources.",
    )


class IngestResponse(BaseModel):
    chunks_ingested: int
    sources_scraped: int


class IcebreakerRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    event_id: UUID = Field(alias="eventId", description="Squad-Up event identifier")
    member_count: int = Field(alias="memberCount", ge=2, le=50)
    member_ids: List[str] = Field(alias="memberIds", min_length=2, max_length=50)
    context_hints: List[str] = Field(default_factory=list, alias="contextHints", max_length=12)

    @field_validator("member_ids")
    @classmethod
    def validate_member_ids(cls, value: List[str]) -> List[str]:
        cleaned = [member_id.strip() for member_id in value if member_id.strip()]
        if len(cleaned) < 2:
            raise ValueError("memberIds must contain at least two non-empty member IDs")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("memberIds must not contain duplicates")
        return cleaned

    @field_validator("context_hints", mode="before")
    @classmethod
    def default_context_hints(cls, value: Any) -> Any:
        if value is None:
            return []
        return value

    @field_validator("context_hints")
    @classmethod
    def validate_context_hints(cls, value: List[str]) -> List[str]:
        cleaned: List[str] = []
        for hint in value:
            normalized = " ".join(hint.strip().split())
            if normalized:
                cleaned.append(normalized[:160])
        return cleaned

    @model_validator(mode="after")
    def validate_member_count_matches_ids(self) -> "IcebreakerRequest":
        if self.member_count != len(self.member_ids):
            raise ValueError("memberCount must match the number of memberIds")
        return self


class IcebreakerResponse(BaseModel):
    prompt: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "service": "Aura Brain", "vector_store": VECTOR_STORE}


@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    summary="Generate a personalised Sydney insider tip",
    tags=["rag"],
)
def query_insider_tip(payload: QueryRequest):
    query_rate_limiter.check(payload.user_id)

    try:
        vibe_context = f"vibe={payload.vibe}; aura_score={payload.aura_score:.0f}/100"
        result = generate_insider_tip(payload.query, vibe_context=vibe_context)
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
    except RuntimeError as exc:
        logger.exception("Runtime configuration error in query pipeline: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG backend is not configured or temporarily unavailable.",
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
            SourceConfig(url=str(url), name=str(url), source_type="custom") for url in payload.urls
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
    except RuntimeError as exc:
        logger.exception("Vector store configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Vector store ingestion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Content scraped but failed to store in the configured vector backend.",
        )

    return IngestResponse(chunks_ingested=chunks, sources_scraped=len(docs))


@router.post(
    "/api/v1/squad-up/icebreaker",
    response_model=IcebreakerResponse,
    summary="Generate a Squad-Up icebreaker prompt",
    tags=["squad-up"],
)
def create_squad_up_icebreaker(payload: IcebreakerRequest):
    try:
        prompt = generate_icebreaker_prompt(
            event_id=payload.event_id,
            member_count=payload.member_count,
            member_ids=payload.member_ids,
            context_hints=payload.context_hints,
        )
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
        logger.exception("OpenAI failed while generating Squad-Up icebreaker.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Icebreaker generation is temporarily unavailable.",
        )
    except Exception as exc:
        logger.exception("Unexpected error while generating Squad-Up icebreaker: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )

    return IcebreakerResponse(prompt=prompt)


@router.post(
    "/api/v1/icebreaker",
    response_model=IcebreakerResponse,
    include_in_schema=False,
)
def create_icebreaker(payload: IcebreakerRequest):
    return create_squad_up_icebreaker(payload)
