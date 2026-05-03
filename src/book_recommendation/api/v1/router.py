"""Version one API routes."""

from fastapi import APIRouter

from book_recommendation.api.v1.endpoints import recommendations

api_router = APIRouter()
api_router.include_router(recommendations.router)
