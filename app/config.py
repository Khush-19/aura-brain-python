# Central config - loads .env once and exposes typed constants to the rest of the app.

import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


# --- OpenAI ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o")

# --- Vector store ---
# Production defaults to Pinecone integrated inference. Set VECTOR_STORE=faiss
# only for local development with OPENAI_API_KEY-backed FAISS embeddings.
VECTOR_STORE: str = os.getenv("VECTOR_STORE", "pinecone").strip().lower()
if VECTOR_STORE not in {"faiss", "pinecone"}:
    raise ValueError("VECTOR_STORE must be either 'faiss' or 'pinecone'")

FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")

# --- Pinecone (only required when VECTOR_STORE=pinecone) ---
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "aura-brain")
PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "aura-brain")
PINECONE_TEXT_FIELD: str = os.getenv("PINECONE_TEXT_FIELD", "text")
PINECONE_EMBED_MODEL: str = os.getenv("PINECONE_EMBED_MODEL", "llama-text-embed-v2")
PINECONE_BATCH_SIZE: int = _env_int("PINECONE_BATCH_SIZE", 96)

# Optional helper for ephemeral/dev environments. Production should usually
# create the index explicitly with the desired cloud/region/schema.
PINECONE_AUTO_CREATE_INDEX: bool = _env_bool("PINECONE_AUTO_CREATE_INDEX", False)
PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")

# --- RAG tuning ---
TOP_K_RESULTS: int = _env_int("TOP_K_RESULTS", 4)
CHUNK_SIZE: int = _env_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP: int = _env_int("CHUNK_OVERLAP", 150)
MIN_SIMILARITY_THRESHOLD: float = _env_float(
    "MIN_SIMILARITY_THRESHOLD",
    0.10 if VECTOR_STORE == "pinecone" else 0.70,
)

# --- Timeouts (seconds) ---
SCRAPER_TIMEOUT: int = _env_int("SCRAPER_TIMEOUT", 10)
OPENAI_TIMEOUT: float = _env_float("OPENAI_TIMEOUT", 30.0)

# --- Background jobs ---
ENABLE_SCHEDULER: bool = _env_bool("ENABLE_SCHEDULER", True)
REFRESH_INTERVAL_MINUTES: int = _env_int("REFRESH_INTERVAL_MINUTES", 15)
