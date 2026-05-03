"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Generator
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from book_recommendation.core.settings import get_settings
from book_recommendation.main import create_app
from book_recommendation.services.recommendation import RecommendationService


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Generator[None, None, None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-not-for-production")
    monkeypatch.setenv("ENVIRONMENT", "dev")

    mock_service = MagicMock()
    mock_service.categories = ["All", "Fiction"]

    ordered = pd.DataFrame(
        {
            "isbn13": [303, 101, 202],
            "title": ["Third", "First", "Second"],
            "authors": ["A;B", "A;B", "A;B"],
            "description": [
                "one two three four",
                "one two three four",
                "one two three four",
            ],
            "large_thumbnail": [
                "http://example.com/t.jpg",
                "http://example.com/t.jpg",
                "http://example.com/t.jpg",
            ],
        }
    )
    mock_service.books = ordered

    def _items(frame: pd.DataFrame) -> list[dict[str, str | int | None]]:
        rows: list[dict[str, str | int | None]] = []
        for _, row in frame.iterrows():
            rows.append(
                {
                    "isbn13": int(row["isbn13"]),
                    "title": str(row["title"]),
                    "authors": "Author",
                    "description_preview": "prev",
                    "thumbnail_url": str(row["large_thumbnail"]),
                }
            )
        return rows

    mock_service.retrieve_ranked.return_value = ordered
    mock_service.to_recommendation_items.side_effect = _items

    def _fake_create(
        cls: type[RecommendationService],
        *args: object,
        **kwargs: object,
    ) -> object:
        return mock_service

    monkeypatch.setattr(
        RecommendationService,
        "create",
        classmethod(_fake_create),
    )

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client