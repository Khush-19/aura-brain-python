"""Local FAISS helpers.

OpenAIEmbeddings is imported lazily inside get_embeddings() so the production
Pinecone integrated-inference path never imports or initializes client-side
embedding code.
"""

from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, OPENAI_API_KEY


@lru_cache(maxsize=1)
def get_embeddings():
    """Return OpenAI embeddings for the local FAISS fallback only."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required when VECTOR_STORE=faiss.")

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=SecretStr(OPENAI_API_KEY),
    )


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks without embedding them."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
