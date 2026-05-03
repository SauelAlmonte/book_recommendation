"""Book recommendation API."""

from fastapi import APIRouter, HTTPException, status

from book_recommendation.api.deps import RecommendationServiceDep
from book_recommendation.schemas.recommendation import (
    BookRecommendation,
    RecommendationRequest,
)

router = APIRouter(tags=["recommendations"])


@router.post(
    "/recommendations",
    response_model=list[BookRecommendation],
    status_code=status.HTTP_200_OK,
)
def post_recommendations(
    body: RecommendationRequest,
    service: RecommendationServiceDep,
) -> list[BookRecommendation]:
    if body.category not in service.categories:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid category for this catalog.",
        )

    initial_k = max(50, min(body.limit * 4, 200))

    frame = service.retrieve_ranked(
        body.query.strip(),
        category=body.category,
        tone=body.tone.value,
        initial_top_k=initial_k,
        final_top_k=body.limit,
    )
    payload = service.to_recommendation_items(frame)
    return [BookRecommendation.model_validate(row) for row in payload]
