"""Tests for catalog file validation (no LangChain / OpenAI)."""

from __future__ import annotations

import pytest

from book_recommendation.services.catalog_validation import (
    CatalogValidationError,
    validate_catalog_paths,
)

_MINIMAL_HEADER = (
    "isbn13,thumbnail,simple_categories,authors,description,title,"
    "joy,surprise,anger,fear,sadness\n"
)
_MINIMAL_ROW = (
    "9780000000001,http://example.com/t.jpg,Fiction,Author One,"
    "A cold war spy story.,The First Book,0.2,0.1,0.0,0.1,0.1\n"
)


def _write_tagged(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_validate_missing_books_file(tmp_path) -> None:
    books = tmp_path / "missing.csv"
    tagged = tmp_path / "tagged.txt"
    _write_tagged(tagged, '9780000000001 "hello world"\n')

    with pytest.raises(CatalogValidationError) as excinfo:
        validate_catalog_paths(
            books_csv_path=str(books),
            tagged_description_path=str(tagged),
        )
    assert "not found" in str(excinfo.value).lower()
    assert "missing.csv" in str(excinfo.value)


def test_validate_books_empty_file(tmp_path) -> None:
    books = tmp_path / "books.csv"
    books.write_text("", encoding="utf-8")
    tagged = tmp_path / "tagged.txt"
    _write_tagged(tagged, '9780000000001 "hello"\n')

    with pytest.raises(CatalogValidationError) as excinfo:
        validate_catalog_paths(
            books_csv_path=str(books),
            tagged_description_path=str(tagged),
        )
    assert "empty" in str(excinfo.value).lower()


def test_validate_missing_csv_columns(tmp_path) -> None:
    books = tmp_path / "books.csv"
    books.write_text(
        "isbn13,title\n9780000000001,Only these cols\n", encoding="utf-8"
    )
    tagged = tmp_path / "tagged.txt"
    _write_tagged(tagged, '9780000000001 "hello"\n')

    with pytest.raises(CatalogValidationError) as excinfo:
        validate_catalog_paths(
            books_csv_path=str(books),
            tagged_description_path=str(tagged),
        )
    msg = str(excinfo.value)
    assert "missing required columns" in msg
    assert "thumbnail" in msg


def test_validate_books_no_data_rows(tmp_path) -> None:
    books = tmp_path / "books.csv"
    books.write_text(_MINIMAL_HEADER, encoding="utf-8")
    tagged = tmp_path / "tagged.txt"
    _write_tagged(tagged, '9780000000001 "hello"\n')

    with pytest.raises(CatalogValidationError) as excinfo:
        validate_catalog_paths(
            books_csv_path=str(books),
            tagged_description_path=str(tagged),
        )
    assert "no data rows" in str(excinfo.value).lower()


def test_validate_tagged_empty(tmp_path) -> None:
    books = tmp_path / "books.csv"
    books.write_text(_MINIMAL_HEADER + _MINIMAL_ROW, encoding="utf-8")
    tagged = tmp_path / "tagged.txt"
    tagged.write_text("", encoding="utf-8")

    with pytest.raises(CatalogValidationError) as excinfo:
        validate_catalog_paths(
            books_csv_path=str(books),
            tagged_description_path=str(tagged),
        )
    assert "empty" in str(excinfo.value).lower()


def test_validate_tagged_bad_first_line(tmp_path) -> None:
    books = tmp_path / "books.csv"
    books.write_text(_MINIMAL_HEADER + _MINIMAL_ROW, encoding="utf-8")
    tagged = tmp_path / "tagged.txt"
    _write_tagged(tagged, "not-an-isbn rest of line\n")

    with pytest.raises(CatalogValidationError) as excinfo:
        validate_catalog_paths(
            books_csv_path=str(books),
            tagged_description_path=str(tagged),
        )
    assert "isbn" in str(excinfo.value).lower()


def test_validate_success_minimal_catalog(tmp_path) -> None:
    books = tmp_path / "books.csv"
    books.write_text(_MINIMAL_HEADER + _MINIMAL_ROW, encoding="utf-8")
    tagged = tmp_path / "tagged.txt"
    _write_tagged(
        tagged,
        '9780000000001 "Tagged description line one."\n'
        '9780000000002 "Another line for a second book."\n',
    )

    validate_catalog_paths(
        books_csv_path=str(books),
        tagged_description_path=str(tagged),
    )
