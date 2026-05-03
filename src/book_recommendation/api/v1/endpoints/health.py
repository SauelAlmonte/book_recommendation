"""Liveness and readiness endpoints."""

from fastapi import APIRouter, HTTPException, Request, status

from book_recommendation.schemas.recommendation import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    service = getattr(request.app.state, "recommendation_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service is not initialized.",
        )
    books = getattr(service, "books", None)
    categories = getattr(service, "categories", None)
    try:
        books_empty = bool(books.empty)  # type: ignore[attr-defined]
    except AttributeError:
        books_empty = True
    if books_empty or not categories:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Catalog not loaded.",
        )
    return ReadyResponse()
