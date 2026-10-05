"""Loading and cleaning helpers for the Amazon product reviews dataset.

The raw Kaggle file has a few known quality problems (see README):
  * ``name`` sometimes holds two unrelated product names glued together;
  * the same ``id`` is sometimes reused for genuinely different products.

The approach here is the one the brief suggests: keep the first product-name
segment and treat the cleaned name -- not ``id`` -- as the product key.
"""

from __future__ import annotations

import re

import pandas as pd

from .config import (
    CATEGORIES_COL,
    DATE_COL,
    NAME_COL,
    RATING_COL,
    RATING_TO_SENTIMENT,
    RAW_REVIEWS_CSV,
    TEXT_COL,
    TITLE_COL,
)

# Two product names glued together show up as a run of commas between them.
_NAME_SEPARATOR = re.compile(r",{2,}")
_WHITESPACE = re.compile(r"\s+")


def load_raw(path=RAW_REVIEWS_CSV, **read_csv_kwargs) -> pd.DataFrame:
    """Read the raw CSV with no cleaning applied."""
    return pd.read_csv(path, low_memory=False, **read_csv_kwargs)


def clean_product_name(name) -> str | None:
    """Keep the first product-name segment and normalise whitespace.

    ``"Kindle Paperwhite,,,Echo Dot"`` -> ``"Kindle Paperwhite"``.
    Returns ``None`` for missing values.
    """
    if not isinstance(name, str):
        return None
    first = _NAME_SEPARATOR.split(name)[0]
    first = _WHITESPACE.sub(" ", first).strip().strip(",").strip()
    return first or None


def rating_to_sentiment(rating) -> str | None:
    """Map a 1-5 star rating to negative / neutral / positive."""
    try:
        return RATING_TO_SENTIMENT[int(rating)]
    except (TypeError, ValueError, KeyError):
        return None


def clean_review_text(text) -> str | None:
    """Light normalisation: collapse whitespace, drop empty strings.

    Deliberately conservative -- transformer tokenizers want the original
    casing and punctuation. Do heavier stripping (lowercasing, stopwords,
    lemmatising) in the TF-IDF pipeline only.
    """
    if not isinstance(text, str):
        return None
    cleaned = _WHITESPACE.sub(" ", text).strip()
    return cleaned or None


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw frame into the tidy frame the notebooks work from.

    Returns columns: ``product``, ``categories``, ``title``, ``text``,
    ``rating``, ``sentiment``, ``date``. Rows without usable text or a
    mappable rating are dropped, and exact duplicate reviews are removed.
    """
    out = pd.DataFrame(
        {
            "product": df[NAME_COL].map(clean_product_name),
            "categories": df.get(CATEGORIES_COL),
            "title": df.get(TITLE_COL),
            "text": df[TEXT_COL].map(clean_review_text),
            "rating": pd.to_numeric(df[RATING_COL], errors="coerce"),
        }
    )
    if DATE_COL in df.columns:
        out["date"] = pd.to_datetime(df[DATE_COL], errors="coerce", utc=True)

    out["sentiment"] = out["rating"].map(rating_to_sentiment)

    out = out.dropna(subset=["text", "sentiment"])
    out = out.drop_duplicates(subset=["product", "text", "rating"])
    return out.reset_index(drop=True)
