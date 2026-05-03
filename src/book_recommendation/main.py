"""Application factory and HTTP middleware."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from starlette.exceptions import HTTPException as StarletteHTTPException

from book_recommendation import __version__
from book_recommendation.api.v1.endpoints import health
from book_recommendation.api.v1.router import api_router
from book_recommendation.core.settings import get_settings
from book_recommendation.services.catalog_validation import CatalogValidationError
from book_recommendation.services.recommendation import RecommendationService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    try:
        service = RecommendationService.create(
            books_csv_path=settings.books_csv_path,
            tagged_description_path=settings.tagged_description_path,
            openai_api_key=settings.openai_api_key,
        )
    except CatalogValidationError:
        logger.exception("Catalog validation failed")
        raise
    except OSError:
        logger.exception("Failed to load catalog data files")
        raise
    except Exception:
        logger.exception("Failed to initialize recommendation service")
        raise
    app.state.recommendation_service = service
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    docs_kwargs = {}
    if settings.is_production:
        docs_kwargs = {
            "docs_url": None,
            "redoc_url": None,
            "openapi_url": None,
        }

    app = FastAPI(
        title="Book recommendation API",
        version=__version__,
        lifespan=lifespan,
        **docs_kwargs,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settings.trusted_hosts_list:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.trusted_hosts_list,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        if isinstance(exc, StarletteHTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
            )
        logger.exception("Unhandled server error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    app.include_router(health.router)
    app.include_router(api_router, prefix="/v1")
    return app
