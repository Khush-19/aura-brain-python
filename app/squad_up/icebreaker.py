"""Icebreaker prompt generation for the Squad-Up feature."""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import List, Optional
from uuid import UUID

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config import CHAT_MODEL, OPENAI_API_KEY, OPENAI_TIMEOUT

logger = logging.getLogger(__name__)


_ICEBREAKER_PROMPT = PromptTemplate(
    input_variables=["member_count", "context_hints"],
    template=(
        "You generate one short, friendly icebreaker prompt for a real-time "
        "Squad-Up group chat. The group has {member_count} members. "
        "Use the context hints if they are useful, but do not mention user IDs, "
        "private data, or that hints were provided. Avoid romance, sensitive "
        "attributes, divisive topics, and anything that needs personal disclosure. "
        "Return only the icebreaker prompt as one sentence, under 28 words.\n\n"
        "Context hints:\n{context_hints}"
    ),
)

_FALLBACK_PROMPTS = [
    "What is one low-stakes thing everyone should try together before the day ends?",
    "What is the most underrated way to make this moment more fun?",
    "If this group had to pick a tiny shared mission right now, what should it be?",
    "What is one local spot or activity you would recommend to the whole squad?",
    "What is a quick win this group could pull off together in the next hour?",
]


@lru_cache(maxsize=1)
def _get_chain():
    if not OPENAI_API_KEY:
        return None

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.8,
        api_key=SecretStr(OPENAI_API_KEY),
        timeout=OPENAI_TIMEOUT,
    )
    return _ICEBREAKER_PROMPT | llm | StrOutputParser()


def _format_hints(context_hints: List[str]) -> str:
    if not context_hints:
        return "- General social meetup"
    return "\n".join(f"- {hint}" for hint in context_hints)


def _fallback_prompt(event_id: UUID, member_ids: List[str], context_hints: List[str]) -> str:
    seed = f"{event_id}:{','.join(member_ids)}:{'|'.join(context_hints)}"
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return _FALLBACK_PROMPTS[digest[0] % len(_FALLBACK_PROMPTS)]


def _clean_generated_prompt(prompt: str) -> str:
    cleaned = " ".join(prompt.strip().split())
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def generate_icebreaker_prompt(
    *,
    event_id: UUID,
    member_count: int,
    member_ids: List[str],
    context_hints: Optional[List[str]] = None,
) -> str:
    """Generate a single Squad-Up icebreaker prompt."""
    hints = context_hints or []
    chain = _get_chain()

    if chain is None:
        logger.info("OPENAI_API_KEY is not configured; using deterministic icebreaker fallback.")
        return _fallback_prompt(event_id, member_ids, hints)

    prompt = chain.invoke(
        {
            "member_count": member_count,
            "context_hints": _format_hints(hints),
        }
    )
    cleaned = _clean_generated_prompt(str(prompt))

    if not cleaned:
        return _fallback_prompt(event_id, member_ids, hints)

    return cleaned
