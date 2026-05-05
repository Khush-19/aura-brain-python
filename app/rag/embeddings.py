# Embedding layer — single source of truth for the OpenAI embedding model and text splitter.

from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, OPENAI_API_KEY


def get_embeddings() -> OpenAIEmbeddings:
    """Return a configured OpenAIEmbeddings instance (text-embedding-3-small)."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks of CHUNK_SIZE with CHUNK_OVERLAP overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
