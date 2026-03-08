# Review Quality Classifier

Binary sentiment classifier (positive / negative) for product reviews, built with a production-style MLOps layer: unit tests and data-drift monitoring.

Built as a portfolio project targeting content moderation and data-pipeline validation roles.

---

## Problem Statement

Given a free-text product review, predict whether the reviewer had a **positive** or **negative** experience. The classifier is trained on Amazon Fine Food Reviews (568 k reviews, binarized from 1–5 star scores).

---

## Dataset

**Amazon Fine Food Reviews** (Kaggle — `snap/amazon-fine-food-reviews`)

| Detail | Value |
|---|---|
| Source | Kaggle |
| Rows (raw) | ~568 000 |
| Label rule | Score ≥ 4 → positive · Score ≤ 2 → negative · Score = 3 dropped |
| Text column used | `Text` (full review body) |
| Available formats | `Reviews.csv` (pandas) · `database.sqlite` (SQL) |

---

## Project Structure

```
review_classifier/
├── data/
│   ├── raw/            # Reviews.csv, database.sqlite, hashes.txt
│   └── processed/      # train.csv, test.csv, confusion/ROC plots (runtime)
├── notebooks/
│   └── 01_eda.ipynb    # Class balance, length distributions, top words, SQL demo
├── src/
│   ├── preprocess.py   # Text cleaning + label binarization
│   ├── features.py     # TF-IDF feature engineering
│   ├── train.py        # Train LR / calibrated SVC, evaluate, save best model
│   └── predict.py      # Load saved model and run inference
├── tests/
│   ├── conftest.py
│   ├── test_preprocess.py       # 9 unit tests for cleaning functions
│   ├── test_features.py         # 7 unit tests for TF-IDF pipeline
│   └── test_data_validation.py  # 10 data-integrity tests
├── monitoring/
│   └── drift_report.py   # Evidently AI drift + classification report
├── models/               # classifier.pkl, vectorizer.pkl (runtime)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download NLTK stopwords (one-time)
python -c "import nltk; nltk.download('stopwords')"
```

**Dataset:** Download **Amazon Fine Food Reviews** from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and place `Reviews.csv` and `database.sqlite` in `data/raw/`.

---

## How to Run

```bash
# Train (saves models/ artifacts + data/processed/ splits)
python src/train.py

# Predict on sample inputs
python src/predict.py

# Run all tests
pytest tests/ -v

# Generate drift report  →  monitoring/drift_report.html
python monitoring/drift_report.py
```

---

## Model Choices

Two models are trained and compared; the best by ROC-AUC is saved.

| Model | Why |
|---|---|
| **Logistic Regression** | Strong linear baseline for TF-IDF features; fast, interpretable |
| **Linear SVC (calibrated)** | Often best on sparse high-dim text; Platt scaling enables probability estimates comparable to LR |

**TF-IDF configuration:** `max_features=50 000`, unigrams + bigrams, `sublinear_tf=True`.

---

## Evaluation Results

> Run `python src/train.py` to reproduce. Results below are representative.

```
Logistic Regression
              precision    recall  f1-score   support
    negative       0.93      0.88      0.90     11234
    positive       0.97      0.98      0.97     46532
    accuracy                           0.96     57766
   macro avg       0.95      0.93      0.94     57766
weighted avg       0.96      0.96      0.96     57766

ROC-AUC: 0.9891
```

---

## MLOps Layer

### Unit Tests (`pytest tests/ -v`)

| File | Tests |
|---|---|
| `test_preprocess.py` | Lowercasing, punctuation removal, whitespace, null handling |
| `test_features.py` | Row counts, column alignment, sparse output, vocab size |
| `test_data_validation.py` | Class balance ≤ 85%, no nulls, no empty strings, feature shape |

### Drift Monitoring (`monitoring/drift_report.py`)

Uses **Evidently AI** to compare two halves of the test set:
- `DataDriftPreset` — detects distribution shift in engineered features (text length, word count, etc.)
- `ClassificationPreset` — precision / recall / F1 comparison between reference and current windows

Output: `monitoring/drift_report.html` — interactive HTML dashboard.

---

## What I'd Do Next

- **Sentence embeddings:** Replace TF-IDF with `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) for richer semantic features.
- **Retraining pipeline:** Automate periodic retraining when drift is detected (Evidently threshold → re-run `train.py` → swap `classifier.pkl`).
- **LLM-powered labelling:** Use an LLM (e.g., GPT-4o) to generate pseudo-labels for unlabelled reviews and expand the training set.
- **REST API:** Wrap `predict()` in a FastAPI endpoint for real-time scoring.
- **CI/CD:** Add GitHub Actions to run `pytest` on every pull request.
