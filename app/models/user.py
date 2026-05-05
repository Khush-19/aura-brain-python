# UserContext is threaded through the pipeline to personalise the insider tip.

from pydantic import BaseModel, Field


class UserContext(BaseModel):
    user_id: str
    aura_score: float = Field(default=50.0, ge=0.0, le=100.0)
    # Vibe drives the tone of the generated tip (chill, energetic, zen, adventurous, social)
    vibe: str = Field(default="balanced")
