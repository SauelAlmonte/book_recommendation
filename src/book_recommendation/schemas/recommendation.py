"""HTTP request/response models (public API surface only)."""

from enum import StrEnum

from pydantic import BaseModel, Field


class EmotionalTone(StrEnum):
    ALL = "All"
    HAPPY = "Happy"
    SURPRISING = "Surprising"
    ANGRY = "Angry"
    SUSPENSEFUL = "Suspenseful"
    SAD = "Sad"


class RecommendationRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language description of desired book content.",
    )
    category: str = Field(
        default="All",
        max_length=200,
        description="Book category; use All or a value from the dataset simple_categories.",
    )
    tone: EmotionalTone = Field(
        default=EmotionalTone.ALL,
        description="Emotional tone ranking; All disables tone-based re-sorting.",
    )
    limit: int = Field(
        default=16,
        ge=1,
        le=50,
        description="Maximum number of recommendations to return.",
    )


class BookRecommendation(BaseModel):
    isbn13: int | None
    title: str
    authors: str
    description_preview: str
    thumbnail_url: str


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    status: str = "ready"
