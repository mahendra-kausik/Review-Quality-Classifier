"""
predict.py — Load saved artifacts and run inference.

Run:
    python src/predict.py

Or import and call predict() directly:
    from predict import predict
    labels = predict(["This product is amazing!", "Terrible quality."])
"""

import pathlib
from typing import List

import joblib

from preprocess import clean_text, remove_stopwords

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def _load_artifacts():
    """Load classifier and vectorizer from models/."""
    clf = joblib.load(MODELS_DIR / "classifier.pkl")
    vec = joblib.load(MODELS_DIR / "vectorizer.pkl")
    return clf, vec


def predict(texts: List[str]) -> List[str]:
    """
    Classify a list of raw review texts.

    Parameters
    ----------
    texts : list of str
        Raw review strings (no preprocessing needed by caller).

    Returns
    -------
    list of str
        'positive' or 'negative' for each input text.
    """
    clf, vec = _load_artifacts()

    # Preprocess
    cleaned = [remove_stopwords(clean_text(t)) for t in texts]

    # Featurize & predict
    X = vec.transform(cleaned)
    preds = clf.predict(X)

    # Map numeric back to labels if stored as 0/1
    label_map = {0: "negative", 1: "positive", "0": "negative", "1": "positive"}
    return [label_map.get(p, str(p)) for p in preds]


if __name__ == "__main__":
    samples = [
        "Absolutely love this product! Would buy again.",
        "Worst purchase I have ever made. Total waste of money.",
        "Great taste and fast delivery.",
        "The packaging was damaged and the item was stale.",
    ]
    results = predict(samples)
    for text, label in zip(samples, results):
        print(f"[{label.upper():8s}]  {text[:70]}")
