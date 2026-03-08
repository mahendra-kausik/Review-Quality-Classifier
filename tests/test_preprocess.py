"""test_preprocess.py — Unit tests for src/preprocess.py functions."""

import pandas as pd

from preprocess import clean_text, remove_stopwords, preprocess_pipeline


# ── clean_text ─────────────────────────────────────────────────────────────────

def test_clean_text_lowercases():
    assert clean_text("Hello WORLD") == "hello world"


def test_clean_text_removes_punctuation():
    assert clean_text("great!!!") == "great"


def test_clean_text_strips_whitespace():
    assert clean_text("  lots   of   spaces  ") == "lots of spaces"


def test_clean_text_handles_mixed():
    result = clean_text("  Fantastic Product!!! BUY Now.  ")
    assert result == "fantastic product buy now"


# ── remove_stopwords ───────────────────────────────────────────────────────────

def test_remove_stopwords_removes_common_words():
    result = remove_stopwords("this is a great product")
    # 'this', 'is', 'a' are stopwords; 'great' and 'product' should remain
    assert "great" in result
    assert "product" in result
    assert "this" not in result


def test_remove_stopwords_returns_string():
    assert isinstance(remove_stopwords("the best"), str)


# ── preprocess_pipeline ────────────────────────────────────────────────────────

def test_preprocess_pipeline_no_nulls():
    df = pd.DataFrame({"Text": ["Good product", "Bad item"]})
    result = preprocess_pipeline(df, text_col="Text")
    assert result["Text"].isnull().sum() == 0


def test_preprocess_pipeline_drops_null_rows():
    df = pd.DataFrame({"Text": ["Good product", None, "Terrible"]})
    result = preprocess_pipeline(df, text_col="Text")
    assert len(result) == 2


def test_preprocess_pipeline_lowercases_output():
    df = pd.DataFrame({"Text": ["AMAZING QUALITY"]})
    result = preprocess_pipeline(df, text_col="Text")
    assert result["Text"].iloc[0] == result["Text"].iloc[0].lower()
