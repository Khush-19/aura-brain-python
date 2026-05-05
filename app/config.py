# Central config — loads .env once and exposes typed constants to the rest of the app.

import os
from dotenv import load_dotenv

load_dotenv()

# --- OpenAI ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = "text-embedding-3-small"
CHAT_MODEL: str = "gpt-4o"

# --- Vector store ---
# Set VECTOR_STORE=pinecone in .env to switch; default is FAISS (no extra infra)
VECTOR_STORE: str = os.getenv("VECTOR_STORE", "faiss")
FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")

# --- Pinecone (only required when VECTOR_STORE=pinecone) ---
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "aura-brain")

# --- RAG tuning ---
TOP_K_RESULTS: int = 4
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 150

# --- Timeouts (seconds) ---
SCRAPER_TIMEOUT: int = int(os.getenv("SCRAPER_TIMEOUT", "10"))
OPENAI_TIMEOUT: float = float(os.getenv("OPENAI_TIMEOUT", "30"))
