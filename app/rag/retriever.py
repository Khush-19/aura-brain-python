"""Vector store layer for Aura Brain.

Production uses Pinecone integrated inference only: raw text records go to
Pinecone via upsert_records(), and queries go through search_records(). The
local FAISS fallback is the only path that imports OpenAIEmbeddings.
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from app.config import (
    FAISS_INDEX_PATH,
    PINECONE_API_KEY,
    PINECONE_AUTO_CREATE_INDEX,
    PINECONE_BATCH_SIZE,
    PINECONE_CLOUD,
    PINECONE_EMBED_MODEL,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_REGION,
    PINECONE_TEXT_FIELD,
    TOP_K_RESULTS,
    VECTOR_STORE,
)
from app.rag.embeddings import get_embeddings, split_documents
from app.utils.validation import pinecone_safe_metadata

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS

ScoredDocuments = List[Tuple[Document, float]]


def _faiss_class():
    from langchain_community.vectorstores import FAISS

    return FAISS


def _document_hash(doc: Document) -> str:
    normalized = " ".join(doc.page_content.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _stable_record_id(doc: Document, chunk_index: int = 0) -> str:
    metadata = doc.metadata or {}
    seed = "|".join(
        [
            str(metadata.get("source", "")),
            str(metadata.get("content_hash") or _document_hash(doc)),
            str(chunk_index),
            doc.page_content,
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _ensure_content_hashes(docs: Sequence[Document]) -> List[Document]:
    normalized_docs: List[Document] = []
    for doc in docs:
        metadata = dict(doc.metadata or {})
        metadata.setdefault("content_hash", _document_hash(doc))
        normalized_docs.append(Document(page_content=doc.page_content, metadata=metadata))
    return normalized_docs


def _existing_content_hashes(store: Optional["FAISS"]) -> set[str]:
    if store is None:
        return set()

    hashes: set[str] = set()
    docstore = getattr(store, "docstore", None)
    raw_docs = getattr(docstore, "_dict", {}) if docstore is not None else {}
    for doc in raw_docs.values():
        content_hash = getattr(doc, "metadata", {}).get("content_hash")
        if content_hash:
            hashes.add(str(content_hash))
    return hashes


def _dedupe_against_store(docs: Sequence[Document], store: Optional["FAISS"]) -> List[Document]:
    existing_hashes = _existing_content_hashes(store)
    seen_hashes = set(existing_hashes)
    unique_docs: List[Document] = []

    for doc in _ensure_content_hashes(docs):
        content_hash = str(doc.metadata["content_hash"])
        if content_hash in seen_hashes:
            logger.info(
                "Skipping duplicate document hash %s from %s",
                content_hash[:12],
                doc.metadata.get("source"),
            )
            continue
        seen_hashes.add(content_hash)
        unique_docs.append(doc)

    return unique_docs


def _score_to_similarity(raw_score: float) -> float:
    return max(0.0, min(1.0, raw_score))


def _distance_to_similarity(distance: float) -> float:
    if distance < 0:
        return 0.0
    return 1.0 / (1.0 + distance)


def _batched(items: Sequence[dict[str, str]], size: int) -> Iterable[Sequence[dict[str, str]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _get_attr_or_key(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _get_hits(response: Any) -> list[Any]:
    result = _get_attr_or_key(response, "result", default={})
    return list(_get_attr_or_key(result, "hits", default=[]) or [])


def _pinecone_record_from_chunk(doc: Document, chunk_index: int, chunk_count: int) -> dict[str, str]:
    record_id = _stable_record_id(doc, chunk_index)
    metadata = pinecone_safe_metadata(
        {
            **(doc.metadata or {}),
            "chunk_id": record_id,
            "document_id": str((doc.metadata or {}).get("content_hash") or _document_hash(doc)),
            "chunk_index": str(chunk_index),
            "chunk_count": str(chunk_count),
        }
    )
    metadata.pop(PINECONE_TEXT_FIELD, None)
    return {
        "_id": record_id,
        PINECONE_TEXT_FIELD: doc.page_content,
        **metadata,
    }


def _document_from_pinecone_hit(hit: Any) -> Tuple[Document, float]:
    fields = dict(_get_attr_or_key(hit, "fields", default={}) or {})
    record_id = str(_get_attr_or_key(hit, "_id", "id", default=""))
    raw_score = float(_get_attr_or_key(hit, "_score", "score", default=0.0) or 0.0)

    page_content = str(
        fields.get(PINECONE_TEXT_FIELD)
        or fields.get("text")
        or fields.get("chunk_text")
        or ""
    )
    metadata = pinecone_safe_metadata(
        {
            key: value
            for key, value in fields.items()
            if key not in {PINECONE_TEXT_FIELD, "text", "chunk_text", "_id", "id"}
        }
    )
    if record_id:
        metadata["id"] = record_id

    return Document(page_content=page_content, metadata=metadata), _score_to_similarity(raw_score)


# ---------------------------------------------------------------------------
# Functional API for the local FAISS seed path
# ---------------------------------------------------------------------------

_faiss_cache: Optional["FAISS"] = None


def get_vector_store() -> Optional["FAISS"]:
    """Load the persisted FAISS index for local development."""
    global _faiss_cache
    if _faiss_cache is not None:
        return _faiss_cache

    if not os.path.exists(FAISS_INDEX_PATH):
        logger.warning(
            "FAISS index not found at '%s'. Run `python scripts/seed_data.py` to create it.",
            FAISS_INDEX_PATH,
        )
        return None

    try:
        faiss = _faiss_class()
        _faiss_cache = faiss.load_local(
            FAISS_INDEX_PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded FAISS index from '%s'", FAISS_INDEX_PATH)
        return _faiss_cache
    except Exception as exc:
        logger.error("Failed to load FAISS index: %s", exc)
        return None


def save_documents_to_store(docs: Sequence[Document]) -> int:
    """Create and persist a fresh local FAISS index."""
    global _faiss_cache

    if not docs:
        raise ValueError("docs list is empty; nothing to ingest.")

    chunks = split_documents(_ensure_content_hashes(docs))
    embeddings = get_embeddings()
    faiss = _faiss_class()

    store = faiss.from_documents(chunks, embeddings)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    store.save_local(FAISS_INDEX_PATH)

    _faiss_cache = store
    logger.info("Saved %d chunks to FAISS index at '%s'", len(chunks), FAISS_INDEX_PATH)
    return len(chunks)


# ---------------------------------------------------------------------------
# Class-based API used by routes, scheduler, and pipeline
# ---------------------------------------------------------------------------

class BaseVectorStore(ABC):
    @abstractmethod
    def ingest_documents(self, docs: Sequence[Document]) -> int:
        raise NotImplementedError

    @abstractmethod
    def similarity_search_with_scores(self, query: str, top_k: int = TOP_K_RESULTS) -> ScoredDocuments:
        raise NotImplementedError

    def similarity_search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_scores(query, top_k)]


class FAISSStore(BaseVectorStore):
    def __init__(self) -> None:
        self._embeddings = get_embeddings()
        self._store: Optional["FAISS"] = get_vector_store()

    def ingest_documents(self, docs: Sequence[Document]) -> int:
        global _faiss_cache

        unique_docs = _dedupe_against_store(docs, self._store)
        if not unique_docs:
            logger.info("No new documents to ingest after content-hash deduplication")
            return 0

        chunks = split_documents(unique_docs)
        if self._store is None:
            faiss = _faiss_class()
            self._store = faiss.from_documents(chunks, self._embeddings)
        else:
            self._store.add_documents(chunks)

        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        self._store.save_local(FAISS_INDEX_PATH)
        _faiss_cache = self._store
        logger.info("Ingested %d chunks into FAISS", len(chunks))
        return len(chunks)

    def similarity_search_with_scores(self, query: str, top_k: int = TOP_K_RESULTS) -> ScoredDocuments:
        if self._store is None:
            return []

        try:
            raw_results = self._store.similarity_search_with_relevance_scores(query, k=top_k)
            return [(doc, _score_to_similarity(float(score))) for doc, score in raw_results]
        except AttributeError:
            raw_results = self._store.similarity_search_with_score(query, k=top_k)
            return [(doc, _distance_to_similarity(float(distance))) for doc, distance in raw_results]


class PineconeStore(BaseVectorStore):
    def __init__(self) -> None:
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is required when VECTOR_STORE=pinecone.")
        if not PINECONE_INDEX_NAME:
            raise RuntimeError("PINECONE_INDEX_NAME is required when VECTOR_STORE=pinecone.")

        from pinecone import Pinecone

        self._pc = Pinecone(api_key=PINECONE_API_KEY)
        self._ensure_index()
        self._index = self._pc.Index(PINECONE_INDEX_NAME)
        self._namespace = PINECONE_NAMESPACE

    def _ensure_index(self) -> None:
        if not PINECONE_AUTO_CREATE_INDEX:
            return

        if self._pc.has_index(PINECONE_INDEX_NAME):
            return

        from pinecone import IndexEmbed

        self._pc.create_index_for_model(
            name=PINECONE_INDEX_NAME,
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
            embed=IndexEmbed(
                model=PINECONE_EMBED_MODEL,
                field_map={"text": PINECONE_TEXT_FIELD},
            ),
        )
        logger.info(
            "Created Pinecone integrated-inference index '%s' with model '%s'",
            PINECONE_INDEX_NAME,
            PINECONE_EMBED_MODEL,
        )

    def ingest_documents(self, docs: Sequence[Document]) -> int:
        normalized_docs = _ensure_content_hashes(docs)
        chunks = split_documents(normalized_docs)
        if not chunks:
            return 0

        records = [
            _pinecone_record_from_chunk(chunk, chunk_index=index, chunk_count=len(chunks))
            for index, chunk in enumerate(chunks)
        ]

        for batch in _batched(records, PINECONE_BATCH_SIZE):
            self._index.upsert_records(namespace=self._namespace, records=list(batch))

        logger.info(
            "Upserted %d raw-text records into Pinecone index '%s' namespace '%s'",
            len(records),
            PINECONE_INDEX_NAME,
            self._namespace,
        )
        return len(records)

    def similarity_search_with_scores(self, query: str, top_k: int = TOP_K_RESULTS) -> ScoredDocuments:
        from pinecone import SearchQuery

        response = self._index.search_records(
            namespace=self._namespace,
            query=SearchQuery(
                inputs={"text": query},
                top_k=top_k,
            ),
            fields=["*"],
        )

        results = [_document_from_pinecone_hit(hit) for hit in _get_hits(response)]
        return [(doc, score) for doc, score in results if doc.page_content.strip()]


_store: Optional[BaseVectorStore] = None


def _get_store() -> BaseVectorStore:
    global _store
    if _store is None:
        _store = PineconeStore() if VECTOR_STORE == "pinecone" else FAISSStore()
    return _store


def ingest_documents(docs: Sequence[Document]) -> int:
    return _get_store().ingest_documents(docs)


def retrieve_documents_with_scores(query: str, top_k: int = TOP_K_RESULTS) -> ScoredDocuments:
    return _get_store().similarity_search_with_scores(query, top_k)


def retrieve_documents(query: str, top_k: int = TOP_K_RESULTS) -> List[Document]:
    return _get_store().similarity_search(query, top_k)
