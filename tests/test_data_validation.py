"""
test_data_validation.py — Validate data assumptions for the review dataset.

These tests check the integrity of:
  - The raw dataset after binarization
  - The processed outputs after cleaning
  - The feature matrix shapes
"""

import pathlib

import pandas as pd
import pytest

from preprocess import load_and_binarize, preprocess_pipeline
from features import build_tfidf_features

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "Reviews.csv"

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """Load and binarize a 5 000-row sample for fast tests."""
    df = load_and_binarize(str(RAW_CSV))
    return df.sample(n=min(5_000, len(df)), random_state=42).reset_index(drop=True)


@pytest.fixture(scope="module")
def processed_df(raw_df):
    return preprocess_pipeline(raw_df, text_col="Text")


# ── Raw data validation ────────────────────────────────────────────────────────

def test_raw_csv_exists():
    assert RAW_CSV.exists(), f"Reviews.csv not found at {RAW_CSV}"


def test_no_score_3_after_binarize(raw_df):
    """load_and_binarize must drop Score==3 rows and not expose the Score column."""
    assert "Score" not in raw_df.columns


def test_labels_are_binary(raw_df):
    assert set(raw_df["label"].unique()).issubset({"positive", "negative"})


def test_class_balance(raw_df):
    """Positive/negative ratio should not exceed 85/15."""
    counts = raw_df["label"].value_counts(normalize=True)
    majority_share = counts.max()
    assert majority_share <= 0.85, f"Class imbalance too severe: {majority_share:.2%}"


def test_no_null_text_after_binarize(raw_df):
    assert raw_df["Text"].isnull().sum() == 0


# ── Processed data validation ──────────────────────────────────────────────────

def test_no_null_text_after_preprocessing(processed_df):
    assert processed_df["Text"].isnull().sum() == 0


def test_no_empty_strings_after_cleaning(processed_df):
    empty = (processed_df["Text"].str.strip() == "").sum()
    assert empty == 0, f"{empty} rows became empty strings after cleaning"


def test_all_text_lowercase(processed_df):
    mixed_case = processed_df["Text"].apply(lambda t: t != t.lower()).sum()
    assert mixed_case == 0


# ── Feature matrix validation ──────────────────────────────────────────────────

def test_feature_matrix_shape(processed_df):
    """Train and test feature matrices must have the same number of columns."""
    split = int(len(processed_df) * 0.8)
    train_texts = processed_df["Text"].iloc[:split].tolist()
    test_texts = processed_df["Text"].iloc[split:].tolist()
    X_train, X_test, _ = build_tfidf_features(train_texts, test_texts)
    assert X_train.shape[1] == X_test.shape[1]


def test_feature_matrix_row_counts(processed_df):
    split = int(len(processed_df) * 0.8)
    train_texts = processed_df["Text"].iloc[:split].tolist()
    test_texts = processed_df["Text"].iloc[split:].tolist()
    X_train, X_test, _ = build_tfidf_features(train_texts, test_texts)
    assert X_train.shape[0] == len(train_texts)
    assert X_test.shape[0] == len(test_texts)
