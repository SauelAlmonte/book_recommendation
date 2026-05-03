"""Application settings loaded from environment (secrets never exposed via API)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


@lru_cache
def discover_project_root() -> Path | None:
    """Directory containing this package's pyproject.toml, or None if not found (e.g. non-editable install)."""
    for directory in Path(__file__).resolve().parents:
        marker = directory / "pyproject.toml"
        if not marker.is_file():
            continue
        try:
            head = marker.read_text(encoding="utf-8")[:4000]
        except OSError:
            continue
        if 'name = "book-recommendation"' in head:
            return directory
    return None


def _parse_csv_list(value: str) -> list[str]:
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(validation_alias="OPENAI_API_KEY")

    books_csv_path: str = Field(
        default="books_with_emotions.csv",
        validation_alias="BOOKS_CSV_PATH",
    )
    tagged_description_path: str = Field(
        default="tagged_description.txt",
        validation_alias="TAGGED_DESCRIPTION_PATH",
    )

    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        validation_alias="CORS_ORIGINS",
    )

    environment: Literal["dev", "prod"] = Field(
        default="dev",
        validation_alias="ENVIRONMENT",
    )

    trusted_hosts: str = Field(
        default="",
        validation_alias="TRUSTED_HOSTS",
    )

    @field_validator("books_csv_path", "tagged_description_path", mode="after")
    @classmethod
    def resolve_catalog_paths_relative_to_repo(cls, value: str) -> str:
        """Relative paths are resolved against the project root (directory with pyproject.toml)."""
        p = Path(value)
        if p.is_absolute():
            return str(p.resolve())
        root = discover_project_root()
        base = root if root is not None else Path.cwd()
        return str((base / p).resolve())

    @property
    def cors_origins_list(self) -> list[str]:
        return _parse_csv_list(self.cors_origins)

    @property
    def trusted_hosts_list(self) -> list[str]:
        return _parse_csv_list(self.trusted_hosts)

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def strip_api_key(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v

    @model_validator(mode="after")
    def require_openai_key(self) -> Self:
        if not self.openai_api_key:
            msg = "OPENAI_API_KEY must be set to a non-empty value"
            raise ValueError(msg)
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()
