"""Validation helpers for Pinecone-safe metadata."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SAFE_VIBE_RE = re.compile(r"[^A-Za-z0-9 ]+")
_PROMPT_CONTROL_RE = re.compile(
    r"\b(ignore|instruction|system|developer|prompt|override|jailbreak|roleplay|assistant|script)\b",
    re.IGNORECASE,
)


def _stringify_metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)
    return str(value)


class AuraMetadata(BaseModel):
    """Flat string metadata schema accepted by Pinecone integrated records."""

    source: str = ""
    name: str = ""
    type: str = "custom"
    source_type: str = "custom"
    category: str = ""
    location: str = ""
    content_hash: str = ""
    scraped_at: str = ""
    document_id: str = ""
    chunk_id: str = ""
    chunk_index: str = Field(default="0")
    chunk_count: str = Field(default="1")

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def coerce_flat_strings(cls, raw: Any) -> dict[str, str]:
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError("metadata must be a mapping")

        cleaned: dict[str, str] = {}
        for key, value in raw.items():
            if key is None or value is None:
                continue

            normalized_key = str(key).strip()
            if not normalized_key:
                continue

            cleaned[normalized_key] = _stringify_metadata_value(value)

        source_type = cleaned.get("source_type") or cleaned.get("type") or cleaned.get("category") or "custom"
        cleaned.setdefault("type", source_type)
        cleaned.setdefault("source_type", source_type)
        cleaned.setdefault("category", source_type)
        return cleaned


def pinecone_safe_metadata(metadata: Mapping[str, Any] | None) -> dict[str, str]:
    """Return metadata as a flat string dict with no null values."""
    return AuraMetadata.model_validate(metadata or {}).model_dump(exclude_none=True)


def sanitize_vibe(vibe: str) -> str:
    """Allow only simple vibe text through to the prompt."""
    if _PROMPT_CONTROL_RE.search(vibe or ""):
        return "balanced"
    sanitized = _SAFE_VIBE_RE.sub("", vibe or "")
    sanitized = " ".join(sanitized.split())
    if _PROMPT_CONTROL_RE.search(sanitized):
        return "balanced"
    return sanitized[:60] or "balanced"
