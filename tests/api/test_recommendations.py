from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from book_recommendation.main import create_app
from book_recommendation.services.recommendation import RecommendationService


def test_recommendations_preserves_service_row_order(client: TestClient) -> None:
    response = client.post(
        "/v1/recommendations",
        json={
            "query": "space",
            "category": "All",
            "tone": "All",
            "limit": 10,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert [item["isbn13"] for item in payload] == [303, 101, 202]
    assert payload[0]["title"] == "Third"


def test_recommendations_invalid_category(client: TestClient) -> None:
    response = client.post(
        "/v1/recommendations",
        json={
            "query": "space",
            "category": "Not-a-real-category",
            "tone": "All",
            "limit": 5,
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid category for this catalog."


def test_recommendations_empty_query_validation(client: TestClient) -> None:
    response = client.post(
        "/v1/recommendations",
        json={
            "query": "",
            "category": "All",
            "tone": "All",
            "limit": 5,
        },
    )
    assert response.status_code == 422


def test_internal_error_body_is_generic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ENVIRONMENT", "dev")
    mock_service = MagicMock(
        spec=["categories", "retrieve_ranked", "to_recommendation_items"]
    )
    mock_service.categories = ["All", "Fiction"]

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("secret_provider_token_xyz")

    mock_service.retrieve_ranked.side_effect = _boom

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

    from book_recommendation.core.settings import get_settings

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/recommendations",
            json={
                "query": "adventure",
                "category": "All",
                "tone": "All",
                "limit": 3,
            },
        )
    assert response.status_code == 500
    body = response.json()
    assert body == {"detail": "Internal server error"}
    assert "secret_provider" not in response.text
