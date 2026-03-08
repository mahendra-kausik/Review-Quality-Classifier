"""
train.py — Model training, evaluation, and artifact saving.

Run:
    python src/train.py

Saves:
    models/classifier.pkl
    models/vectorizer.pkl
    data/processed/train.csv
    data/processed/test.csv
"""

import pathlib

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from preprocess import load_and_binarize, preprocess_pipeline
from features import build_tfidf_features

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "Reviews.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(name: str, model, X_test, y_test, label_encoder) -> float:
    """Print classification report + confusion matrix. Return ROC-AUC."""
    y_pred = model.predict(X_test)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=label_encoder))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=label_encoder, ax=ax, colorbar=False
    )
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(DATA_PROCESSED / f"confusion_{name.replace(' ', '_')}.png", dpi=100)
    plt.close(fig)

    # ROC-AUC (binary: positive class = 1)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        return float("nan")

    auc = roc_auc_score(y_test, y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DATA_PROCESSED / f"roc_{name.replace(' ', '_')}.png", dpi=100)
    plt.close(fig)

    print(f"  ROC-AUC: {auc:.4f}")
    return auc


def main() -> None:
    # ── 1. Load & preprocess ───────────────────────────────────────────────────
    print("Loading data…")
    df = load_and_binarize(str(DATA_RAW))
    print(f"  Rows after binarization: {len(df):,}")
    print(f"  Class counts:\n{df['label'].value_counts()}\n")

    df = preprocess_pipeline(df, text_col="Text")

    # ── 2. Encode labels (positive=1, negative=0) ──────────────────────────────
    df["label_enc"] = (df["label"] == "positive").astype(int)

    # ── 3. Train / test split ──────────────────────────────────────────────────
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_enc"]
    )
    train_df.to_csv(DATA_PROCESSED / "train.csv", index=False)
    test_df.to_csv(DATA_PROCESSED / "test.csv", index=False)
    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # ── 4. TF-IDF features ────────────────────────────────────────────────────
    print("\nBuilding TF-IDF features…")
    X_train, X_test, vectorizer = build_tfidf_features(
        train_df["Text"].tolist(), test_df["Text"].tolist()
    )
    y_train = train_df["label_enc"].values
    y_test = test_df["label_enc"].values
    label_names = ["negative", "positive"]

    # ── 5. Train models ────────────────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=42
        ),
        # CalibratedClassifierCV wraps LinearSVC so it gains predict_proba
        # via Platt scaling — ensures ROC curves for both models use the same
        # probability scale rather than mixing probabilities with raw SVM margins.
        "Linear SVC (calibrated)": CalibratedClassifierCV(
            LinearSVC(C=0.5, max_iter=2000, random_state=42)
        ),
    }

    results: dict[str, float] = {}
    for name, clf in models.items():
        print(f"\nFitting {name}…")
        clf.fit(X_train, y_train)
        auc = evaluate_model(name, clf, X_test, y_test, label_names)
        results[name] = auc

    # ── 6. Pick best model by AUC ──────────────────────────────────────────────
    best_name = max(results, key=results.get)
    best_model = models[best_name]
    print(f"\nBest model: {best_name} (AUC={results[best_name]:.4f})")

    # ── 7. Save artifacts ──────────────────────────────────────────────────────
    joblib.dump(best_model, MODELS_DIR / "classifier.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")
    print(f"Saved classifier  → {MODELS_DIR / 'classifier.pkl'}")
    print(f"Saved vectorizer  → {MODELS_DIR / 'vectorizer.pkl'}")


if __name__ == "__main__":
    main()
