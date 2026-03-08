"""
drift_report.py — Evidently AI drift and classification performance report.

Run (after training):
    python monitoring/drift_report.py

Outputs:
    monitoring/drift_report.html

Design: why the test-set split approach
----------------------------------------
This project uses a static dataset (Amazon Fine Food Reviews), so there is
no live stream of new reviews to monitor against.  To demonstrate the
monitoring infrastructure meaningfully, we simulate the production pattern:

  Production reality (what this mimics)
  ──────────────────────────────────────
  1.  Model is trained and deployed.  Training/validation data becomes the
      "reference" — the expected distribution.
  2.  New reviews arrive over time (daily / weekly batches).
  3.  Each batch is the "current" window.
  4.  Evidently compares current vs reference to detect:
        - Data drift   : input feature distributions have shifted
        - Model decay  : precision / recall has degraded on the new batch
  5.  If drift is detected, a retraining pipeline is triggered.

  What we do here (static dataset simulation)
  ─────────────────────────────────────────────
  We take the held-out test set (never seen during training) and split it
  in half chronologically:
    - First half  → reference  (simulates the "baseline" window at deploy time)
    - Second half → current    (simulates a later batch of incoming reviews)

  Because both halves come from the same distribution, drift metrics will be
  low — which is the correct and expected result for a stable dataset.  The
  value is in having the pipeline in place: in a real deployment you would
  replace these two DataFrames with your actual reference and production data.

Feature strategy
----------------
The raw TF-IDF matrix (50 000+ sparse dimensions) is not suitable for
Evidently's statistical drift tests.  Instead we engineer four interpretable
numerical features from the review text:
  - text_length       character count
  - word_count        number of whitespace tokens
  - avg_word_length   mean characters per token
  - unique_word_ratio type-token ratio (vocabulary diversity)

These proxy features capture meaningful distributional properties of the text
(e.g., a sudden influx of very short / spam reviews would show up here) while
remaining compatible with Evidently's per-column drift detectors.
"""

import pathlib
import sys

import joblib
import pandas as pd

# Allow running from repo root or monitoring/ directory
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import clean_text, remove_stopwords  # noqa: E402

MODELS_DIR = ROOT / "models"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORT_PATH = pathlib.Path(__file__).resolve().parent / "drift_report.html"


# ── Feature engineering helpers ────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, text_col: str = "Text") -> pd.DataFrame:
    """Add numerical features derived from the review text."""
    df = df.copy()
    df["text_length"] = df[text_col].str.len()
    df["word_count"] = df[text_col].str.split().str.len()
    df["avg_word_length"] = df[text_col].apply(
        lambda t: (sum(len(w) for w in t.split()) / max(len(t.split()), 1))
    )
    df["unique_word_ratio"] = df[text_col].apply(
        lambda t: len(set(t.split())) / max(len(t.split()), 1)
    )
    return df


def build_report_df(test_csv: pathlib.Path) -> pd.DataFrame:
    """Load test.csv, predict, and return a report-ready DataFrame."""
    clf = joblib.load(MODELS_DIR / "classifier.pkl")
    vec = joblib.load(MODELS_DIR / "vectorizer.pkl")

    df = pd.read_csv(test_csv)

    # Clean text for vectorizer
    df["Text_clean"] = df["Text"].astype(str).apply(clean_text).apply(remove_stopwords)

    # Predict
    X = vec.transform(df["Text_clean"].tolist())
    preds = clf.predict(X)

    # Map numeric labels back to strings if necessary
    label_map = {0: "negative", 1: "positive", "0": "negative", "1": "positive"}
    df["prediction"] = [label_map.get(p, str(p)) for p in preds]

    # Target column (use string labels for ClassificationPreset)
    df["target"] = df["label"].astype(str)

    # Engineer numerical features on raw text
    df = engineer_features(df, text_col="Text")

    feature_cols = ["text_length", "word_count", "avg_word_length", "unique_word_ratio",
                    "target", "prediction"]
    return df[feature_cols].reset_index(drop=True)


def main() -> None:
    test_csv = DATA_PROCESSED / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(
            f"{test_csv} not found. Run `python src/train.py` first."
        )

    print("Building report DataFrame…")
    report_df = build_report_df(test_csv)

    # Split into reference / current.
    # In production: replace reference_df with your training-period data and
    # current_df with the latest batch of incoming reviews.  Here we use
    # two halves of the test set as a stand-in for that production pattern.
    mid = len(report_df) // 2
    reference_df = report_df.iloc[:mid].reset_index(drop=True)   # "baseline"
    current_df = report_df.iloc[mid:].reset_index(drop=True)      # "new batch"
    print(f"Reference rows: {len(reference_df):,}  |  Current rows: {len(current_df):,}")

    # ── Evidently report ───────────────────────────────────────────────────────
    # evidently 0.7.x restructured its API:
    #   - Report/BinaryClassification/Dataset/DataDefinition are top-level
    #   - Presets are under evidently.presets
    #   - DataFrames must be wrapped in Dataset.from_pandas() with a DataDefinition
    #     so Evidently knows which columns are target / prediction labels
    #   - report.run() returns a Snapshot; call snapshot.save_html() to persist
    try:
        from evidently import Report, BinaryClassification, Dataset, DataDefinition
        from evidently.presets import DataDriftPreset, ClassificationPreset
    except ImportError as e:
        print(f"Evidently import failed: {e}\nRun: pip install evidently")
        return

    data_def = DataDefinition(
        classification=[
            BinaryClassification(
                target="target",
                prediction_labels="prediction",
                pos_label="positive",
            )
        ]
    )
    reference_ds = Dataset.from_pandas(reference_df, data_definition=data_def)
    current_ds = Dataset.from_pandas(current_df, data_definition=data_def)

    report = Report(
        metrics=[
            DataDriftPreset(),
            ClassificationPreset(),
        ]
    )
    snapshot = report.run(current_data=current_ds, reference_data=reference_ds)
    snapshot.save_html(str(REPORT_PATH))
    print(f"\nDrift report saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
