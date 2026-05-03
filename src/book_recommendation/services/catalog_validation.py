"""Pre-flight checks for book catalog files before expensive embedding work."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_BOOKS_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "isbn13",
        "thumbnail",
        "simple_categories",
        "authors",
        "description",
        "title",
        "joy",
        "surprise",
        "anger",
        "fear",
        "sadness",
    }
)


class CatalogValidationError(ValueError):
    """Raised when CSV or tagged_description files do not match the expected contract."""


def parse_isbn_prefix(page_content: str) -> int:
    """Parse leading ISBN token from a tagged description line (same rules as vector chunks)."""
    stripped = page_content.strip().strip('"')
    token = stripped.split(maxsplit=1)[0]
    return int(token)


def validate_catalog_paths(
    *,
    books_csv_path: str,
    tagged_description_path: str,
) -> None:
    """Ensure files exist and are non-empty; validate CSV headers and at least one row."""
    books_p = Path(books_csv_path)
    tagged_p = Path(tagged_description_path)

    if not books_p.is_file():
        raise CatalogValidationError(
            f"Books catalog file not found or not a file: {books_csv_path!r}\n"
            f"Set BOOKS_CSV_PATH or place the file relative to the process working directory."
        )
    if books_p.stat().st_size == 0:
        raise CatalogValidationError(
            f"Books catalog file is empty: {books_csv_path!r}"
        )

    if not tagged_p.is_file():
        raise CatalogValidationError(
            f"Tagged description file not found or not a file: {tagged_description_path!r}\n"
            f"Set TAGGED_DESCRIPTION_PATH or place the file relative to the process working directory."
        )
    if tagged_p.stat().st_size == 0:
        raise CatalogValidationError(
            f"Tagged description file is empty: {tagged_description_path!r}"
        )

    try:
        header_df = pd.read_csv(books_p, nrows=0)
    except Exception as exc:
        raise CatalogValidationError(
            f"Could not read CSV header from {books_csv_path!r}: {exc}"
        ) from exc

    missing = sorted(_BOOKS_REQUIRED_COLUMNS.difference(header_df.columns))
    if missing:
        raise CatalogValidationError(
            f"Books CSV {books_csv_path!r} is missing required columns: {missing}\n"
            f"Required: {sorted(_BOOKS_REQUIRED_COLUMNS)}"
        )

    try:
        sample = pd.read_csv(books_p, nrows=1)
    except Exception as exc:
        raise CatalogValidationError(
            f"Could not read at least one data row from {books_csv_path!r}: {exc}"
        ) from exc

    if sample.empty:
        raise CatalogValidationError(
            f"Books CSV {books_csv_path!r} has headers but no data rows."
        )

    _validate_tagged_description_first_nonempty_line(tagged_description_path)


def _validate_tagged_description_first_nonempty_line(tagged_path: str) -> None:
    path_obj = Path(tagged_path)
    try:
        text = path_obj.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogValidationError(
            f"Could not read tagged description file {tagged_path!r}: {exc}"
        ) from exc

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parse_isbn_prefix(stripped)
        except (ValueError, IndexError) as exc:
            preview = stripped[:120] + ("..." if len(stripped) > 120 else "")
            raise CatalogValidationError(
                f"Tagged description {tagged_path!r}: first non-empty line must begin "
                f"with a numeric ISBN token (UTF-8). Problem line starts with: {preview!r}"
            ) from exc
        return

    raise CatalogValidationError(
        f"Tagged description file {tagged_path!r} has no non-empty lines."
    )
