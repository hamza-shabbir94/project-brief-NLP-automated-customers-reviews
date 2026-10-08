"""Project-wide paths and constants.

Import these instead of hardcoding paths in notebooks, so a notebook runs the
same whatever directory it is launched from.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
ARTICLES_DIR = OUTPUTS_DIR / "articles"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# The Kaggle "Consumer Reviews of Amazon Products" file the brief recommends.
RAW_REVIEWS_CSV = RAW_DIR / "1429_1.csv"

# ---------------------------------------------------------------- columns

# Column names as they appear in the Kaggle CSV.
TEXT_COL = "reviews.text"
TITLE_COL = "reviews.title"
RATING_COL = "reviews.rating"
NAME_COL = "name"
CATEGORIES_COL = "categories"
DATE_COL = "reviews.date"

# ---------------------------------------------------------------- labels

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Star rating -> sentiment class, per the project brief.
RATING_TO_SENTIMENT = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}

# ---------------------------------------------------------------- misc

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
