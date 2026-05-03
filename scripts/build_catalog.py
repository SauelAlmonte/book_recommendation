#!/usr/bin/env python3
"""Download the Kaggle dataset and reproduce the notebook pipeline to emit API catalog files.

Outputs (by default next to pyproject.toml):
  - books_with_emotions.csv
  - tagged_description.txt

Install build deps:
  pip install -e ".[catalog-build]"

Usage:
  python scripts/build_catalog.py
  python scripts/build_catalog.py --max-books 200   # smoke run
  python scripts/build_catalog.py --dataset-root /path/to/books.csv/dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

CATEGORY_MAPPING: dict[str, str] = {
    "Fiction": "Fiction",
    "Juvenile Fiction": "Children's Fiction",
    "Biography & Autobiography": "Nonfiction",
    "History": "Nonfiction",
    "Literary Criticism": "Nonfiction",
    "Philosophy": "Nonfiction",
    "Religion": "Nonfiction",
    "Comics & Graphic Novels": "Fiction",
    "Drama": "Fiction",
    "Juvenile Nonfiction": "Children's Nonfiction",
    "Science": "Nonfiction",
    "Poetry": "Fiction",
}

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
FICTION_VS_NONFICTION = ["Fiction", "Nonfiction"]


def _repo_root() -> Path:
    p = REPO_ROOT
    if not (p / "pyproject.toml").is_file():
        raise SystemExit(
            "Run this script from the book-recommendation clone; "
            f"expected pyproject.toml at {p / 'pyproject.toml'}."
        )
    return p


def download_kaggle_dataset() -> Path:
    import kagglehub

    path_str = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
    return Path(path_str)


def load_raw_books(dataset_root: Path) -> pd.DataFrame:
    csv_path = dataset_root / "books.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path} (expected Kaggle layout).")
    return pd.read_csv(csv_path)


def clean_to_book_missing_25_words(books: pd.DataFrame) -> pd.DataFrame:
    """Match notebooks/data-exploration.ipynb up to books_cleaned columns."""
    books = books.copy()
    books["missing_description"] = np.where(books["description"].isna(), 1, 0)
    books["age_of_book"] = 2024 - books["published_year"]

    book_missing = books[
        ~(books["description"].isna())
        & ~(books["num_pages"].isna())
        & ~(books["average_rating"].isna())
        & ~(books["published_year"].isna())
    ]
    book_missing = book_missing.copy()
    book_missing["words_in_description"] = book_missing["description"].str.split().str.len()
    book_missing_25_words = book_missing[book_missing["words_in_description"] >= 25].copy()

    def _title_and_subtitle(row: pd.Series) -> str:
        title = row["title"]
        sub = row["subtitle"]
        if pd.isna(sub):
            return str(title)
        return f"{title}: {sub}"

    book_missing_25_words["title_and_subtitle"] = book_missing_25_words.apply(
        _title_and_subtitle, axis=1
    )
    def _tagged_description_row(row: pd.Series) -> str:
        isbn = row["isbn13"]
        isbn_part = str(int(isbn)) if pd.notna(isbn) else ""
        return f"{isbn_part} {row['description']}"

    book_missing_25_words["tagged_description"] = book_missing_25_words.apply(
        _tagged_description_row, axis=1
    )

    cleaned = (
        book_missing_25_words.drop(
            ["subtitle", "missing_description", "age_of_book", "words_in_description"],
            axis=1,
        )
    ).copy()
    return cleaned


def apply_simple_categories(books: pd.DataFrame) -> pd.DataFrame:
    """Match notebooks/text-classification.ipynb mapping + zero-shot fill."""
    from tqdm import tqdm
    from transformers import pipeline

    books = books.copy()
    books["simple_categories"] = books["categories"].map(CATEGORY_MAPPING)

    pipe = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,
    )

    def predict_one(sequence: str) -> str:
        out = pipe(sequence, FICTION_VS_NONFICTION)
        idx = int(np.argmax(out["scores"]))
        return str(out["labels"][idx])

    missing = books.loc[books["simple_categories"].isna(), ["isbn13", "description"]].reset_index(
        drop=True
    )
    isbns: list[int | np.integer] = []
    predicted: list[str] = []
    for i in tqdm(range(len(missing)), desc="categories (zero-shot)"):
        sequence = str(missing["description"].iloc[i])
        isbns.append(missing["isbn13"].iloc[i])
        predicted.append(predict_one(sequence))

    if missing.shape[0]:
        missing_df = pd.DataFrame({"isbn13": isbns, "predicted_categories": predicted})
        books = pd.merge(books, missing_df, on="isbn13", how="left")
        books["simple_categories"] = np.where(
            books["simple_categories"].isna(),
            books["predicted_categories"],
            books["simple_categories"],
        )
        books = books.drop(columns=["predicted_categories"])
    return books


def _normalize_emotion_batch(raw: list) -> list[list[dict]]:
    """HF pipeline may return one list of dicts for a single input string."""
    if not raw:
        return []
    if isinstance(raw[0], dict):
        return [raw]
    return raw


def calculate_max_emotion_scores(predictions: list[list[dict]], emotion_labels: list[str]):
    per_emotion_scores = {label: [] for label in emotion_labels}
    for prediction in predictions:
        sorted_predictions = sorted(prediction, key=lambda x: x["label"])
        for index, label in enumerate(emotion_labels):
            per_emotion_scores[label].append(sorted_predictions[index]["score"])
    return {label: float(np.max(scores)) for label, scores in per_emotion_scores.items()}


def apply_emotion_scores(books: pd.DataFrame) -> pd.DataFrame:
    """Match notebooks/sentiment-analysis.ipynb (per-book max over sentences)."""
    from tqdm import tqdm
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=-1,
    )

    isbn_col: list[int | np.integer] = []
    emotion_scores = {label: [] for label in EMOTION_LABELS}

    for i in tqdm(range(len(books)), desc="emotions"):
        isbn_col.append(books["isbn13"].iloc[i])
        desc = str(books["description"].iloc[i])
        sentences = [s.strip() for s in desc.split(".") if s.strip()]
        if not sentences:
            sentences = [desc]
        predictions = _normalize_emotion_batch(classifier(sentences))
        max_scores = calculate_max_emotion_scores(predictions, EMOTION_LABELS)
        for label in EMOTION_LABELS:
            emotion_scores[label].append(max_scores[label])

    emotions_df = pd.DataFrame(emotion_scores)
    emotions_df["isbn13"] = isbn_col
    return pd.merge(books, emotions_df, on="isbn13", how="left")


def write_outputs(books: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "books_with_emotions.csv"
    tagged_path = output_dir / "tagged_description.txt"
    books.to_csv(csv_path, index=False)
    books["tagged_description"].to_csv(
        tagged_path,
        sep="\n",
        index=False,
        header=False,
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {tagged_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV and txt (default: repository root).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Folder containing books.csv from the Kaggle dataset (skip download).",
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=None,
        help="Process only the first N rows after cleaning (for quick smoke tests).",
    )
    ns = parser.parse_args()
    root = _repo_root()
    out = (ns.output_dir or root).resolve()

    try:
        import kagglehub  # noqa: F401
        import tqdm  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print(
            "Missing dependencies. Install with:\n"
            '  pip install -e ".[catalog-build]"',
            file=sys.stderr,
        )
        return 1

    if ns.dataset_root is not None:
        ds = ns.dataset_root.resolve()
        print(f"Using existing dataset at {ds}")
    else:
        print("Downloading dataset via kagglehub…")
        ds = download_kaggle_dataset()
        print(f"Dataset path: {ds}")

    books = load_raw_books(ds)
    cleaned = clean_to_book_missing_25_words(books)
    if ns.max_books is not None:
        cleaned = cleaned.iloc[: ns.max_books].copy()
        print(f"Truncated to --max-books={ns.max_books}")

    print("Applying categories (BART-MNLI; may take a while on CPU)…")
    with_categories = apply_simple_categories(cleaned)
    print("Scoring emotions (DistilRoBERTa; may take a while on CPU)…")
    final_df = apply_emotion_scores(with_categories)
    write_outputs(final_df, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
