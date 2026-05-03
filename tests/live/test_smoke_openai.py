"""Optional smoke tests against OpenAI + Chroma (gated by env)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from book_recommendation.services.recommendation import RecommendationService

pytestmark = pytest.mark.live

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "live_minimal"
_BOOKS = _FIXTURE_DIR / "books_with_emotions.csv"
_TAGGED = _FIXTURE_DIR / "tagged_description.txt"


@pytest.fixture(scope="module")
def live_openai_key() -> str:
    if os.environ.get("RUN_LIVE") != "1":
        pytest.skip("Set RUN_LIVE=1 to run live OpenAI/Chroma tests.")
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        pytest.skip("OPENAI_API_KEY is not set.")
    if key == "test-openai-key-not-for-production":
        pytest.skip("OPENAI_API_KEY appears to be a test placeholder.")
    return key


def test_live_create_catalog_and_search(live_openai_key: str) -> None:
    service = RecommendationService.create(
        books_csv_path=str(_BOOKS),
        tagged_description_path=str(_TAGGED),
        openai_api_key=live_openai_key,
    )
    assert not service.books.empty
    assert "All" in service.categories
    assert "Fiction" in service.categories

    frame = service.retrieve_ranked(
        "noir detective rain mystery investigation",
        category="All",
        tone="All",
        initial_top_k=10,
        final_top_k=3,
    )
    assert not frame.empty
    top_three = frame.head(3)["isbn13"].astype(int).tolist()
    assert 9781111111111 in top_three
