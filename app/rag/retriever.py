# Vector store layer — FAISS (local) or Pinecone (production).
#
# Public functional API (used by pipeline.py and scripts):
#   get_vector_store()         → load FAISS from disk, returns None if missing
#   save_documents_to_store()  → chunk, embed, persist a fresh FAISS index
#
# Class-based API (used by HTTP routes for incremental ingestion):
#   ingest_documents()
#   retrieve_documents()

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import (
    FAISS_INDEX_PATH,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    TOP_K_RESULTS,
    VECTOR_STORE,
)
from app.rag.embeddings import get_embeddings, split_documents

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Functional API  (pipeline + seed script)
# ---------------------------------------------------------------------------

_faiss_cache: Optional[FAISS] = None


def get_vector_store() -> Optional[FAISS]:
    """
    Load the persisted FAISS index from disk and return it.
    Returns None (with a warning) if the index has not been created yet.
    Subsequent calls return the cached in-memory instance.
    """
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
        _faiss_cache = FAISS.load_local(
            FAISS_INDEX_PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded FAISS index from '%s'", FAISS_INDEX_PATH)
        return _faiss_cache
    except Exception as exc:
        logger.error("Failed to load FAISS index: %s", exc)
        return None


def save_documents_to_store(docs: List[Document]) -> int:
    """
    Chunk the supplied Documents, embed them with OpenAI, build a fresh FAISS
    index, persist it to FAISS_INDEX_PATH, and update the in-memory cache.
    Returns the number of chunks stored.
    """
    global _faiss_cache

    if not docs:
        raise ValueError("docs list is empty — nothing to ingest.")

    chunks = split_documents(docs)
    embeddings = get_embeddings()

    store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    store.save_local(FAISS_INDEX_PATH)

    _faiss_cache = store
    logger.info("Saved %d chunks to FAISS index at '%s'", len(chunks), FAISS_INDEX_PATH)
    return len(chunks)


# ---------------------------------------------------------------------------
# Class-based API  (HTTP ingest + retrieve routes)
# ---------------------------------------------------------------------------

class BaseVectorStore(ABC):
    @abstractmethod
    def ingest_documents(self, docs: List[Document]) -> int: ...

    @abstractmethod
    def similarity_search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Document]: ...


class FAISSStore(BaseVectorStore):
    def __init__(self) -> None:
        self._embeddings = get_embeddings()
        self._store: Optional[FAISS] = get_vector_store()  # reuse cached load

    def ingest_documents(self, docs: List[Document]) -> int:
        chunks = split_documents(docs)
        if self._store is None:
            self._store = FAISS.from_documents(chunks, self._embeddings)
        else:
            self._store.add_documents(chunks)
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        self._store.save_local(FAISS_INDEX_PATH)
        logger.info("Ingested %d chunks via HTTP route", len(chunks))
        return len(chunks)

    def similarity_search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Document]:
        if self._store is None:
            return []
        return self._store.similarity_search(query, k=top_k)


class PineconeStore(BaseVectorStore):
    def __init__(self) -> None:
        from pinecone import Pinecone
        from langchain_pinecone import PineconeVectorStore

        pc = Pinecone(api_key=PINECONE_API_KEY)
        self._embeddings = get_embeddings()
        self._store = PineconeVectorStore(
            index=pc.Index(PINECONE_INDEX_NAME),
            embedding=self._embeddings,
        )

    def ingest_documents(self, docs: List[Document]) -> int:
        chunks = split_documents(docs)
        self._store.add_documents(chunks)
        return len(chunks)

    def similarity_search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Document]:
        return self._store.similarity_search(query, k=top_k)


_store: Optional[BaseVectorStore] = None


def _get_store() -> BaseVectorStore:
    global _store
    if _store is None:
        _store = PineconeStore() if VECTOR_STORE == "pinecone" else FAISSStore()
    return _store


def ingest_documents(docs: List[Document]) -> int:
    return _get_store().ingest_documents(docs)


def retrieve_documents(query: str, top_k: int = TOP_K_RESULTS) -> List[Document]:
    return _get_store().similarity_search(query, top_k)
