"""Book catalog loading and semantic retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from book_recommendation.services.catalog_validation import (
    parse_isbn_prefix,
    validate_catalog_paths,
)

logger = logging.getLogger(__name__)

_DEFAULT_THUMB_FALLBACK = "static/cover-not-found.jpg"
_TONE_SORT_COLUMNS: dict[str, str] = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness",
}


@dataclass
class RecommendationService:
    """Stateful service: book metadata + Chroma vector store."""

    books: pd.DataFrame
    vectorstore: Any
    categories: list[str]

    @classmethod
    def create(
        cls,
        *,
        books_csv_path: str,
        tagged_description_path: str,
        openai_api_key: str,
    ) -> RecommendationService:
        from langchain_chroma import Chroma
        from langchain_community.document_loaders import TextLoader
        from langchain_openai import OpenAIEmbeddings
        from langchain_text_splitters import CharacterTextSplitter

        validate_catalog_paths(
            books_csv_path=books_csv_path,
            tagged_description_path=tagged_description_path,
        )

        books = pd.read_csv(books_csv_path)
        books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
        books["large_thumbnail"] = np.where(
            books["large_thumbnail"].isna(),
            _DEFAULT_THUMB_FALLBACK,
            books["large_thumbnail"],
        )

        raw_documents = TextLoader(tagged_description_path).load()
        # Large chunk_size with newline separator yields one chunk per line for typical rows.
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=12_000,
            chunk_overlap=0,
        )
        documents = text_splitter.split_documents(raw_documents)
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vectorstore = Chroma.from_documents(documents, embeddings)

        categories = ["All"] + sorted(
            books["simple_categories"].dropna().unique().tolist(),
            key=str,
        )
        return cls(books=books, vectorstore=vectorstore, categories=categories)

    def retrieve_ranked(
        self,
        query: str,
        *,
        category: str | None,
        tone: str | None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
    ) -> pd.DataFrame:
        if category is None or category == "":
            category = "All"
        recs = self.vectorstore.similarity_search(query, k=initial_top_k)

        ordered_isbns: list[int] = []
        seen: set[int] = set()
        for rec in recs:
            try:
                isbn = parse_isbn_prefix(rec.page_content)
            except (ValueError, IndexError):
                logger.warning("Skipping chunk with unparsable ISBN prefix.")
                continue
            if isbn not in seen:
                seen.add(isbn)
                ordered_isbns.append(isbn)

        if not ordered_isbns:
            return pd.DataFrame()

        indexed = self.books.set_index("isbn13")
        present = [i for i in ordered_isbns if i in indexed.index]
        if not present:
            return pd.DataFrame()

        book_recs = indexed.loc[present].reset_index()

        if category != "All":
            book_recs = book_recs[book_recs["simple_categories"] == category]
        book_recs = book_recs.head(final_top_k)

        sort_col = _TONE_SORT_COLUMNS.get(tone or "", "")
        if sort_col and sort_col in book_recs.columns:
            book_recs = book_recs.sort_values(by=sort_col, ascending=False)

        return book_recs.reset_index(drop=True)

    def to_recommendation_items(
        self,
        frame: pd.DataFrame,
        *,
        description_preview_words: int = 30,
    ) -> list[dict[str, str | int | None]]:
        rows: list[dict[str, str | int | None]] = []
        for _, row in frame.iterrows():
            authors_raw = row["authors"]
            if pd.isna(authors_raw):
                authors_str = ""
            else:
                authors_split = str(authors_raw).split(";")
                if len(authors_split) == 2:
                    authors_str = f"{authors_split[0]} and {authors_split[1]}"
                elif len(authors_split) > 2:
                    authors_str = (
                        f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
                    )
                else:
                    authors_str = str(authors_raw)

            desc_raw = row["description"]
            if pd.isna(desc_raw):
                preview = ""
            else:
                words = str(desc_raw).split()
                preview = (
                    " ".join(words[:description_preview_words]) + "..."
                    if words
                    else ""
                )

            title = row["title"]
            title_str = "" if pd.isna(title) else str(title)

            thumb = row["large_thumbnail"]
            thumb_str = _DEFAULT_THUMB_FALLBACK if pd.isna(thumb) else str(thumb)

            isbn = row["isbn13"]
            isbn_int = int(isbn) if pd.notna(isbn) else None

            rows.append(
                {
                    "isbn13": isbn_int,
                    "title": title_str,
                    "authors": authors_str,
                    "description_preview": preview,
                    "thumbnail_url": thumb_str,
                }
            )
        return rows
