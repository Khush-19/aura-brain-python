# RAG pipeline — retrieves context from FAISS and generates a Sydney insider tip via GPT-4o.

import logging
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError

from app.config import OPENAI_API_KEY, OPENAI_TIMEOUT
from app.rag.retriever import get_vector_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangChain chain  (built once at import time, reused across requests)
# ---------------------------------------------------------------------------

_PROMPT = PromptTemplate(
    input_variables=["context", "query", "vibe_context"],
    template=(
        "You are the Aura Health Companion, a local Sydney expert. "
        "Answer the user using ONLY the provided Context. "
        'Add a warm, conversational "Insider Tip" vibe. '
        "If the context mentions the user's current vibe ({vibe_context}), tailor the advice. "
        "Context: {context} | Query: {query}"
    ),
)

_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    api_key=OPENAI_API_KEY,
    timeout=OPENAI_TIMEOUT,
)

_chain = _PROMPT | _llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def generate_insider_tip(query: str, vibe_context: Optional[str] = None) -> str:
    """
    Full RAG cycle:
      1. Load the FAISS vector store
      2. Retrieve the top-3 most relevant chunks via similarity search
      3. Inject context + vibe into the Aura prompt
      4. Stream through ChatOpenAI (gpt-4o) and return the string response

    Raises openai.APITimeoutError / APIConnectionError / RateLimitError on
    upstream failures — callers (routes.py) map these to HTTP status codes.
    """
    store = get_vector_store()
    if store is None:
        return (
            "The Aura knowledge base is empty. "
            "Run `python scripts/seed_data.py` from the project root to populate it."
        )

    docs = store.similarity_search(query, k=3)
    if not docs:
        return "No relevant Sydney context found — try a different query or re-run the seed script."

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    try:
        return _chain.invoke(
            {
                "context": context,
                "query": query,
                "vibe_context": vibe_context or "general",
            }
        )
    except RateLimitError:
        logger.warning("OpenAI rate limit hit for query: %.60s…", query)
        raise
    except APITimeoutError:
        logger.error("OpenAI request timed out for query: %.60s…", query)
        raise
    except APIConnectionError:
        logger.error("OpenAI connection error for query: %.60s…", query)
        raise
