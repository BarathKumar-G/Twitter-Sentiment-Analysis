"""
preprocessing.py - Data loading and text preprocessing for Twitter Sentiment Analysis.

Handles:
- Loading train/test CSVs with automatic column detection
- Cleaning tweets (lowercase, remove URLs, mentions, special chars)
- Handling missing values
"""

import re
import pandas as pd


# ─── Column Detection ────────────────────────────────────────────────────────

TEXT_ALIASES = ["sentence", "text", "tweet", "content", "message"]
LABEL_ALIASES = ["sentiment", "label", "target", "class", "polarity"]


def detect_columns(df: pd.DataFrame):
    """
    Auto-detect the text and label column names from a DataFrame.
    Checks lowercase column names against known aliases.
    Returns (text_col, label_col) or raises ValueError if not found.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    text_col = None
    for alias in TEXT_ALIASES:
        if alias in cols_lower:
            text_col = cols_lower[alias]
            break

    label_col = None
    for alias in LABEL_ALIASES:
        if alias in cols_lower:
            label_col = cols_lower[alias]
            break

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not detect text/label columns. Available columns: {list(df.columns)}\n"
            f"Text aliases tried: {TEXT_ALIASES}\n"
            f"Label aliases tried: {LABEL_ALIASES}"
        )

    return text_col, label_col


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_data(filepath: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load a CSV file and return a cleaned DataFrame.

    Args:
        filepath:    Path to the CSV file.
        sample_size: If provided, randomly sample this many rows (for large files).

    Returns:
        DataFrame with `text` and `label` columns.
    """
    print(f"  Loading: {filepath}")
    df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
    print(f"  Raw shape: {df.shape}")

    # Auto-detect columns
    text_col, label_col = detect_columns(df)
    print(f"  Detected columns — text: '{text_col}', label: '{label_col}'")

    # Standardise column names
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

    # Drop rows with missing values
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing values.")

    # Ensure label is integer
    df["label"] = df["label"].astype(int)

    # Optional sampling (used for large training sets on limited hardware)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"  Sampled {sample_size} rows from training data.")

    print(f"  Final shape: {df.shape}")
    return df


# ─── Text Cleaning ───────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean a single tweet string.

    Steps:
    1. Lowercase
    2. Remove URLs (http/https/www)
    3. Remove @mentions
    4. Remove hashtag symbol (keep the word)
    5. Remove special characters / punctuation (keep alphanumeric + spaces)
    6. Collapse multiple spaces
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # Remove URLs
    text = re.sub(r"@\w+", "", text)                    # Remove @mentions
    text = re.sub(r"#", "", text)                       # Remove # symbol
    text = re.sub(r"[^a-z0-9\s]", "", text)             # Keep only alphanumeric
    text = re.sub(r"\s+", " ", text).strip()            # Collapse whitespace
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to the entire DataFrame in-place (new 'clean_text' column).
    """
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)

    # Drop rows where clean_text is empty after cleaning
    before = len(df)
    df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
    if len(df) < before:
        print(f"  Removed {before - len(df)} empty rows after cleaning.")

    return df
