"""test_features.py — Unit tests for src/features.py."""

import scipy.sparse
from features import build_tfidf_features


TRAIN_TEXTS = [
    "great product love it",
    "amazing quality highly recommend",
    "terrible waste of money",
    "horrible experience never buying again",
    "fast shipping good packaging",
]

TEST_TEXTS = [
    "really good product",
    "bad quality disappointed",
]


# ── build_tfidf_features ───────────────────────────────────────────────────────

def test_returns_three_items():
    result = build_tfidf_features(TRAIN_TEXTS, TEST_TEXTS)
    assert len(result) == 3


def test_train_row_count():
    X_train, _, _ = build_tfidf_features(TRAIN_TEXTS, TEST_TEXTS)
    assert X_train.shape[0] == len(TRAIN_TEXTS)


def test_test_row_count():
    _, X_test, _ = build_tfidf_features(TRAIN_TEXTS, TEST_TEXTS)
    assert X_test.shape[0] == len(TEST_TEXTS)


def test_train_and_test_same_column_count():
    X_train, X_test, _ = build_tfidf_features(TRAIN_TEXTS, TEST_TEXTS)
    assert X_train.shape[1] == X_test.shape[1]


def test_output_is_sparse():
    X_train, X_test, _ = build_tfidf_features(TRAIN_TEXTS, TEST_TEXTS)
    assert scipy.sparse.issparse(X_train)
    assert scipy.sparse.issparse(X_test)


def test_vectorizer_vocabulary_not_empty():
    _, _, vectorizer = build_tfidf_features(TRAIN_TEXTS, TEST_TEXTS)
    assert len(vectorizer.vocabulary_) > 0


def test_max_features_respected():
    X_train, X_test, _ = build_tfidf_features(TRAIN_TEXTS, TEST_TEXTS, max_features=5)
    assert X_train.shape[1] <= 5
