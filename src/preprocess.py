"""
preprocess.py — Text cleaning and preprocessing functions.

All functions are importable and independently testable.
"""

import re
import string
import pandas as pd
import nltk

# Download stopwords on first use
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lowercase text, remove punctuation, and strip extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """Remove English stopwords from whitespace-tokenised text."""
    tokens = text.split()
    filtered = [w for w in tokens if w not in STOPWORDS]
    return " ".join(filtered)


def preprocess_pipeline(df: pd.DataFrame, text_col: str = "Text") -> pd.DataFrame:
    """
    Apply full preprocessing to the text column in-place.

    Steps: clean_text → remove_stopwords
    Drops rows where the text column is null before processing.
    """
    df = df.copy()
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str).apply(clean_text).apply(remove_stopwords)
    return df


def load_and_binarize(csv_path: str) -> pd.DataFrame:
    """
    Load Reviews.csv, binarize labels, and drop ambiguous Score==3 rows.

    Returns a DataFrame with columns: Text, label
      - label: 'positive' (Score >= 4) or 'negative' (Score <= 2)
    """
    df = pd.read_csv(csv_path, usecols=["Score", "Text"])
    df = df.dropna(subset=["Score", "Text"])
    df = df[df["Score"] != 3].copy()
    df["label"] = df["Score"].apply(lambda s: "positive" if s >= 4 else "negative")
    df = df[["Text", "label"]].reset_index(drop=True)
    return df
