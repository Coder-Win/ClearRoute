"""
TF-IDF Machine Learning Baseline
Trains n-gram TF-IDF vectorizers and binary classifiers over labelled descriptions.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings("ignore")

# Paths
INPUT_CSV = Path("annotations/semantic_annotation_pilot_labelled.csv")
OUTPUT_PRED_CSV = Path("reports/phase0/tfidf_baseline_predictions.csv")
OUTPUT_METRICS_CSV = Path("reports/phase0/tfidf_baseline_metrics.csv")

TARGET_FIELDS = [
    "ann_heavy_vehicle",
    "ann_rollover",
    "ann_spill",
    "ann_fire",
    "ann_debris",
    "ann_lanes_blocked",
    "ann_full_road_blockage",
    "ann_ramp_involvement",
    "ann_disabled_vehicle",
    "ann_emergency_response_mentioned",
]


def normalize_labels(series: pd.Series) -> pd.Series:
    """Standardizes target values to clean binary integer series (0 or 1)."""
    return series.astype(str).str.lower().isin(["true", "1", "yes", "confirmed"]).astype(int)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found at: {INPUT_CSV}")

    OUTPUT_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    texts = df["Description"].fillna("").astype(str).tolist()

    # 1. Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000,
        stop_words="english",
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)

    predictions_df = pd.DataFrame(index=df.index)
    metrics_list = []

    print("\n" + "=" * 65)
    print(f"{'Field':<35} | {'Precision':<8} | {'Recall':<8} | {'F1-Score':<8}")
    print("=" * 65)

    # 2. Train and evaluate a classifier per semantic target
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for col in TARGET_FIELDS:
        clean_name = col.replace("ann_", "")
        pred_col_name = f"tfidf_{clean_name}"

        if col not in df.columns:
            continue

        y = normalize_labels(df[col])
        positive_count = int(y.sum())

        # SAFETY CHECKS FOR SMALL PILOT DATASETS
        if positive_count == 0 or positive_count == len(y):
            # Only one class present (all 0s or all 1s). Cannot train a model.
            y_pred = np.zeros(len(y), dtype=int) if positive_count == 0 else np.ones(len(y), dtype=int)
        elif positive_count < 5:
            # Too few samples for 5-fold CV, train and predict on the full set
            clf = LogisticRegression(class_weight="balanced", random_state=42)
            clf.fit(X, y)
            y_pred = clf.predict(X)
        else:
            # Normal 5-fold cross validation
            clf = LogisticRegression(class_weight="balanced", random_state=42)
            y_pred = cross_val_predict(clf, X, y, cv=cv)

        predictions_df[pred_col_name] = y_pred

        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        metrics_list.append({
            "field": clean_name,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "support_positive_true": positive_count,
            "predicted_positive": int(y_pred.sum()),
        })

        print(f"{clean_name:<35} | {precision:<8.4f} | {recall:<8.4f} | {f1:<8.4f}")

    # Combine original data with TF-IDF predictions
    df_results = pd.concat([df, predictions_df], axis=1)
    df_results.to_csv(OUTPUT_PRED_CSV, index=False)
    print(f"[OK] TF-IDF predictions saved to: {OUTPUT_PRED_CSV}")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(OUTPUT_METRICS_CSV, index=False)
    print("=" * 65)
    print(f"[OK] TF-IDF summary metrics written to: {OUTPUT_METRICS_CSV}\n")


if __name__ == "__main__":
    main()