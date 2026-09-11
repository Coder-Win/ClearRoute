"""
Keyword Extraction Baseline
Evaluates rule-based / regex extraction against labelled pilot data.
"""

from pathlib import Path
import re
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Paths
INPUT_CSV = Path("annotations/semantic_annotation_pilot_labelled.csv")
OUTPUT_PRED_CSV = Path("reports/phase0/keyword_baseline_predictions.csv")
OUTPUT_METRICS_CSV = Path("reports/phase0/keyword_baseline_metrics.csv")


def extract_keywords(text: str) -> dict[str, bool | str]:
    """
    Extracts semantic features from raw incident text using keyword matching & regex rules.
    """
    text_lower = str(text).lower()

    # Rule definitions
    heavy_vehicle_patterns = r"\b(semi|truck|tractor|trailer|tanker|18-wheeler|hauler|bus|dump truck)\b"
    rollover_patterns = r"\b(overturned|rollover|rolled over|flipped|on its side|on roof)\b"
    spill_patterns = r"\b(spill|spilled|leaking|leak|diesel fuel|hazardous material|hazmat|chemical)\b"
    fire_patterns = r"\b(fire|flames|burning|smoke|engulfed)\b"
    debris_patterns = r"\b(debris|scattered|obstruction|blocked by object|object on road|tire)\b"
    injury_patterns = r"\b(injury|injuries|injured|fatal|fatality|hospital|paramedics|ambulance)\b"
    ramp_patterns = r"\b(ramp|exit ramp|entrance ramp|on-ramp|off-ramp)\b"
    full_blockage_patterns = r"\b(all lanes closed|road closed|closed both directions|fully blocked|all traffic)\b"
    emergency_patterns = r"\b(police|trooper|fire department|ems|paramedic|sheriff|patrol)\b"
    disabled_patterns = r"\b(disabled|stalled|breakdown|flat tire|broken down)\b"

    # Lane regex detection (e.g., '2 lanes blocked', 'left lane closed')
    lanes_match = re.search(r"(\d+)\s+lane", text_lower)
    lane_blocked = bool(lanes_match) or ("lane closed" in text_lower) or ("lanes blocked" in text_lower)

    # Incident type heuristic
    if bool(re.search(rollover_patterns, text_lower)):
        incident_type = "overturned_vehicle"
    elif bool(re.search(fire_patterns, text_lower)):
        incident_type = "fire"
    elif bool(re.search(disabled_patterns, text_lower)):
        incident_type = "disabled_vehicle"
    elif "accident" in text_lower or "crash" in text_lower or "collision" in text_lower:
        incident_type = "crash"
    elif bool(re.search(debris_patterns, text_lower)):
        incident_type = "hazard"
    else:
        incident_type = "unknown"

    return {
        "kw_incident_type": incident_type,
        "kw_heavy_vehicle": bool(re.search(heavy_vehicle_patterns, text_lower)),
        "kw_rollover": bool(re.search(rollover_patterns, text_lower)),
        "kw_spill": bool(re.search(spill_patterns, text_lower)),
        "kw_fire": bool(re.search(fire_patterns, text_lower)),
        "kw_debris": bool(re.search(debris_patterns, text_lower)),
        "kw_injury": bool(re.search(injury_patterns, text_lower)),
        "kw_lanes_blocked": lane_blocked,
        "kw_full_road_blockage": bool(re.search(full_blockage_patterns, text_lower)),
        "kw_ramp_involvement": bool(re.search(ramp_patterns, text_lower)),
        "kw_disabled_vehicle": bool(re.search(disabled_patterns, text_lower)),
        "kw_emergency_response_mentioned": bool(re.search(emergency_patterns, text_lower)),
    }


def normalize_truth_series(series: pd.Series) -> pd.Series:
    """
    Standardizes boolean annotations to clean binary (True/False).
    """
    return series.astype(str).str.lower().isin(["true", "1", "yes", "confirmed"])


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found at: {INPUT_CSV}")

    OUTPUT_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading silver ground truth from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # 1. Run keyword extraction over all rows
    print("Running keyword extraction rules...")
    predictions = df["Description"].apply(extract_keywords).apply(pd.Series)
    df_results = pd.concat([df, predictions], axis=1)

    # Save detailed prediction file
    df_results.to_csv(OUTPUT_PRED_CSV, index=False)
    print(f"[OK] Keyword predictions saved to: {OUTPUT_PRED_CSV}")

    # 2. Evaluate performance against silver standard
    target_fields = [
        ("ann_heavy_vehicle", "kw_heavy_vehicle"),
        ("ann_rollover", "kw_rollover"),
        ("ann_spill", "kw_spill"),
        ("ann_fire", "kw_fire"),
        ("ann_debris", "kw_debris"),
        ("ann_lanes_blocked", "kw_lanes_blocked"),
        ("ann_full_road_blockage", "kw_full_road_blockage"),
        ("ann_ramp_involvement", "kw_ramp_involvement"),
        ("ann_disabled_vehicle", "kw_disabled_vehicle"),
        ("ann_emergency_response_mentioned", "kw_emergency_response_mentioned"),
    ]

    metrics_list = []

    print("\n" + "=" * 65)
    print(f"{'Field':<35} | {'Precision':<8} | {'Recall':<8} | {'F1-Score':<8}")
    print("=" * 65)

    for true_col, pred_col in target_fields:
        if true_col not in df.columns or pred_col not in df_results.columns:
            continue

        y_true = normalize_truth_series(df_results[true_col])
        y_pred = df_results[pred_col].astype(bool)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        metrics_list.append({
            "field": true_col.replace("ann_", ""),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "support_positive_true": int(y_true.sum()),
            "predicted_positive": int(y_pred.sum()),
        })

        print(f"{true_col.replace('ann_', ''):<35} | {precision:<8.4f} | {recall:<8.4f} | {f1:<8.4f}")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(OUTPUT_METRICS_CSV, index=False)
    print("=" * 65)
    print(f"[OK] Summary metrics written to: {OUTPUT_METRICS_CSV}\n")


if __name__ == "__main__":
    main()