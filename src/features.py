"""
features.py — TF-IDF feature engineering.

Fits the vectorizer on training data only to avoid data leakage.
"""

from typing import Tuple

import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_features(
    train_texts,
    test_texts,
    max_features: int = 50_000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, TfidfVectorizer]:
    """
    Fit TF-IDF on train_texts, transform both splits.

    Parameters
    ----------
    train_texts : array-like of str
    test_texts  : array-like of str
    max_features : int
        Vocabulary cap (default 50 000).
    ngram_range : tuple
        Unigrams + bigrams by default.

    Returns
    -------
    X_train : sparse matrix
    X_test  : sparse matrix
    vectorizer : fitted TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer
