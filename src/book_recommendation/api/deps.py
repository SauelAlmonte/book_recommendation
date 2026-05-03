"""FastAPI dependencies."""

from typing import Annotated

from fastapi import Depends, Request

from book_recommendation.services.recommendation import RecommendationService


def get_recommendation_service(request: Request) -> RecommendationService:
    return request.app.state.recommendation_service


RecommendationServiceDep = Annotated[
    RecommendationService,
    Depends(get_recommendation_service),
]
