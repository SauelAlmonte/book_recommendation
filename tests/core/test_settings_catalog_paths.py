"""Settings resolve relative catalog paths against the repository root."""

from __future__ import annotations

from pathlib import Path

import pytest

from book_recommendation.core.settings import Settings, discover_project_root


def test_relative_catalog_paths_resolve_to_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("BOOKS_CSV_PATH", "books_with_emotions.csv")
    monkeypatch.setenv("TAGGED_DESCRIPTION_PATH", "nested/tagged_description.txt")

    root = discover_project_root()
    assert root is not None
    assert (root / "pyproject.toml").is_file()

    s = Settings()
    assert s.books_csv_path == str((root / "books_with_emotions.csv").resolve())
    assert s.tagged_description_path == str(
        (root / "nested" / "tagged_description.txt").resolve()
    )


def test_absolute_catalog_paths_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    books = tmp_path / "b.csv"
    tagged = tmp_path / "t.txt"
    books.write_text("x", encoding="utf-8")
    tagged.write_text("y", encoding="utf-8")
    monkeypatch.setenv("BOOKS_CSV_PATH", str(books))
    monkeypatch.setenv("TAGGED_DESCRIPTION_PATH", str(tagged))

    s = Settings()
    assert s.books_csv_path == str(books.resolve())
    assert s.tagged_description_path == str(tagged.resolve())
