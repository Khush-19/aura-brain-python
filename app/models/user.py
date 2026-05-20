# UserContext is threaded through the pipeline to personalise the insider tip.

from pydantic import BaseModel, Field, field_validator

from app.utils.validation import sanitize_vibe


class UserContext(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    aura_score: float = Field(default=50.0, ge=0.0, le=100.0)
    vibe: str = Field(default="balanced", max_length=80)

    @field_validator("vibe")
    @classmethod
    def validate_vibe(cls, value: str) -> str:
        return sanitize_vibe(value)
