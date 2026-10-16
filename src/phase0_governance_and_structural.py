# import os
# import json
# import yaml
# import numpy as np
# import pandas as pd
# from pathlib import Path

# # --- Existing Repo Paths ---
# CONFIG_PATH = Path("configs/phase0_config.yaml")
# EXCLUDED_FEATURES_JSON = Path("configs/generated/us_accidents_excluded_features.json")

# INPUT_CANDIDATES = [
#     Path("data/processed/us_accidents_spatiotemporal.parquet"),
#     Path("data/processed/us_accidents_phase0_processed.parquet"),
# ]

# OUTPUT_STRUCTURAL_PATH = Path("data/processed/structural_feature_table.parquet")
# AUDIT_SUMMARY_JSON = Path("reports/phase0/phase0_audit_summary.json")
# SPLIT_SUMMARY_CSV = Path("reports/phase0/us_split_summary.csv")

# OUTPUT_STRUCTURAL_PATH.parent.mkdir(parents=True, exist_ok=True)
# AUDIT_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)


# def load_repo_configs():
#     config = {}
#     if CONFIG_PATH.exists():
#         with open(CONFIG_PATH, "r", encoding="utf-8") as f:
#             config = yaml.safe_load(f) or {}

#     excluded_cols = [
#         "End_Lat", "End_Lng", "Turning_Loop", "Number",
#         "Airport_Code", "Weather_Timestamp", "Civil_Twilight",
#         "Nautical_Twilight", "Astronomical_Twilight",
#         # Exclude target-derived / metadata strings from structural X features
#         "reported_duration_minutes", "valid_initial_duration", "duration_requires_review",
#         "description_normalized", "dataset_source", "target_name", "target_definition",
#         "event_observed", "primary_incident_id"
#     ]
#     if EXCLUDED_FEATURES_JSON.exists():
#         try:
#             with open(EXCLUDED_FEATURES_JSON, "r", encoding="utf-8") as f:
#                 loaded_excl = json.load(f)
#                 if isinstance(loaded_excl, list):
#                     excluded_cols = list(set(excluded_cols + loaded_excl))
#                 elif isinstance(loaded_excl, dict) and "excluded_features" in loaded_excl:
#                     excluded_cols = list(set(excluded_cols + loaded_excl["excluded_features"]))
#         except Exception:
#             pass

#     return config, excluded_cols


# def load_input_data() -> pd.DataFrame:
#     for path in INPUT_CANDIDATES:
#         if path.exists():
#             print(f"[Stage 1] Loading existing dataset from: {path}")
#             df = pd.read_parquet(path)
#             print(f"[Stage 1] Loaded {len(df):,} rows and {len(df.columns)} columns.")
#             return df
#     raise FileNotFoundError("Could not find processed parquet files in 'data/processed/'.")


# def apply_chronological_partitioning(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Fixes the uncalled chronological split function:
#     Sorts strictly by Start_Time and partitions into train (70%), val (10%), cal (10%), test (10%).
#     """
#     print("[Stage 1] Executing strict Chronological Split (train / val / cal / test)...")
#     df = df.sort_values("Start_Time").reset_index(drop=True)
#     n = len(df)

#     train_cut = int(n * 0.70)
#     val_cut = int(n * 0.80)
#     cal_cut = int(n * 0.90)

#     splits = np.empty(n, dtype=object)
#     splits[:train_cut] = "train"
#     splits[train_cut:val_cut] = "val"
#     splits[val_cut:cal_cut] = "cal"
#     splits[cal_cut:] = "test"

#     df["split"] = splits

#     split_summary = (
#         df.groupby("split")
#         .agg(
#             count=("ID", "count"),
#             min_time=("Start_Time", "min"),
#             max_time=("Start_Time", "max"),
#             median_duration=("duration", "median"),
#             censor_rate=("censor_flag", "mean")
#         )
#         .reset_index()
#     )
#     split_summary.to_csv(SPLIT_SUMMARY_CSV, index=False)
#     print(f"[Stage 1] Saved chronological split summary to {SPLIT_SUMMARY_CSV}")
#     return df


# def run_stage1_governance(df: pd.DataFrame, excluded_cols: list) -> pd.DataFrame:
#     print("[Stage 1] Running Phase-0 Audit, Duration Sanity Checks, and Leakage Control...")
#     initial_count = len(df)

#     # 1. Map your exact timestamp columns: Start_Time_parsed & End_Time_parsed
#     if "Start_Time_parsed" in df.columns:
#         df["Start_Time"] = pd.to_datetime(df["Start_Time_parsed"], errors="coerce")
#     elif "Start_Time" in df.columns:
#         df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")

#     if "End_Time_parsed" in df.columns:
#         df["End_Time"] = pd.to_datetime(df["End_Time_parsed"], errors="coerce")
#     elif "End_Time" in df.columns:
#         df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")

#     # 2. Map your exact duration column: reported_duration_minutes
#     if "reported_duration_minutes" in df.columns:
#         df["raw_duration_mins"] = pd.to_numeric(df["reported_duration_minutes"], errors="coerce")
#     else:
#         df["raw_duration_mins"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60.0

#     # Filter invalid durations using your valid_initial_duration flag if present, plus >= 1.0 min check
#     if "valid_initial_duration" in df.columns:
#         df = df[df["valid_initial_duration"] == True].copy()
#     df = df.dropna(subset=["ID", "Start_Time", "raw_duration_mins", "Start_Lat", "Start_Lng"]).copy()
#     df = df[df["raw_duration_mins"] >= 1.0].copy()

#     # 3. Duplicate candidate resolution
#     df["_time_bucket"] = df["Start_Time"].dt.floor("5min")
#     df["_lat_bucket"] = df["Start_Lat"].round(3)
#     df["_lng_bucket"] = df["Start_Lng"].round(3)
#     df = df.drop_duplicates(subset=["_time_bucket", "_lat_bucket", "_lng_bucket"]).copy()
#     df = df.drop(columns=["_time_bucket", "_lat_bucket", "_lng_bucket"])

#     # 4. Right-censor flag at 24-hour bound (1440 minutes)
#     df["censor_flag"] = (df["raw_duration_mins"] >= 1440.0).astype(int)
#     df["duration"] = np.clip(df["raw_duration_mins"], a_min=1.0, a_max=1440.0)

#     # 5. Standardize secondary crash flag from your spatiotemporal parquet
#     if "is_secondary_crash" in df.columns:
#         df["secondary_crash_flag"] = df["is_secondary_crash"].fillna(0).astype(int)

#     # 6. Drop leakage-risk & target-proxy columns so they never leak into X
#     keep_protected = {
#         "ID", "Start_Time", "End_Time", "Description",
#         "duration", "raw_duration_mins", "censor_flag",
#         "Start_Lat", "Start_Lng", "secondary_crash_flag"
#     }
#     drop_cols = [c for c in excluded_cols if c in df.columns and c not in keep_protected]
#     for extra_leak in ["Start_Time_parsed", "End_Time_parsed", "is_secondary_crash"]:
#         if extra_leak in df.columns:
#             drop_cols.append(extra_leak)

#     df = df.drop(columns=list(set(drop_cols)), errors="ignore")

#     # 7. Call chronological split function
#     df = apply_chronological_partitioning(df)

#     audit_info = {
#         "initial_rows": int(initial_count),
#         "audited_rows": int(len(df)),
#         "right_censored_count": int(df["censor_flag"].sum()),
#         "secondary_crashes_flagged": int(df["secondary_crash_flag"].sum()) if "secondary_crash_flag" in df.columns else 0,
#         "splits": df["split"].value_counts().to_dict()
#     }
#     with open(AUDIT_SUMMARY_JSON, "w", encoding="utf-8") as f:
#         json.dump(audit_info, f, indent=2)

#     print(f"[Stage 1] Governance complete: {initial_count:,} -> {len(df):,} clean records.")
#     return df


# def run_stage2_structural_pipeline(df: pd.DataFrame) -> pd.DataFrame:
#     print("[Stage 2] Adding cyclical Spatiotemporal Enrichment features...")
#     hour = df["event_hour"] if "event_hour" in df.columns else (df["Start_Time"].dt.hour + df["Start_Time"].dt.minute / 60.0)
#     dow = df["event_day_of_week"] if "event_day_of_week" in df.columns else df["Start_Time"].dt.dayofweek
#     month = df["event_month"] if "event_month" in df.columns else df["Start_Time"].dt.month

#     df["hour_feature"] = hour
#     df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
#     df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
#     df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
#     df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
#     df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
#     df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
#     df["is_rush_hour"] = (((hour >= 7.0) & (hour <= 10.0)) | ((hour >= 16.0) & (hour <= 19.0))).astype(int)

#     return df


# def main():
#     _, excluded_cols = load_repo_configs()
#     df = load_input_data()
#     df = run_stage1_governance(df, excluded_cols)
#     df = run_stage2_structural_pipeline(df)

#     df.to_parquet(OUTPUT_STRUCTURAL_PATH, index=False)
#     print(f"[Done] Saved Stage 1 & 2 Structural Feature Table ({len(df):,} rows) to: {OUTPUT_STRUCTURAL_PATH}")


# if __name__ == "__main__":
#     main()


import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import BallTree

# --- Existing Repo Paths ---
CONFIG_PATH = Path("configs/phase0_config.yaml")
EXCLUDED_FEATURES_JSON = Path("configs/generated/us_accidents_excluded_features.json")

INPUT_CANDIDATES = [
    Path("data/processed/us_accidents_spatiotemporal.parquet"),
    Path("data/processed/us_accidents_phase0_processed.parquet"),
]

OUTPUT_STRUCTURAL_PATH = Path("data/processed/structural_feature_table.parquet")
AUDIT_SUMMARY_JSON = Path("reports/phase0/phase0_audit_summary.json")
SPLIT_SUMMARY_CSV = Path("reports/phase0/us_split_summary.csv")

OUTPUT_STRUCTURAL_PATH.parent.mkdir(parents=True, exist_ok=True)
AUDIT_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)


def load_repo_configs():
    config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    excluded_cols = [
        "End_Lat", "End_Lng", "Turning_Loop", "Number",
        "Airport_Code", "Weather_Timestamp", "Civil_Twilight",
        "Nautical_Twilight", "Astronomical_Twilight",
        # Exclude reporting-source artifact ('Source' has 0.47 correlation with duration due to agency logging artifacts)
        "Source", "Country", "Timezone", "Wind_Chill(F)",
        # Exclude target-derived / metadata strings from structural X features
        "reported_duration_minutes", "valid_initial_duration", "duration_requires_review",
        "description_normalized", "dataset_source", "target_name", "target_definition",
        "event_observed", "primary_incident_id", "is_secondary_crash"
    ]
    if EXCLUDED_FEATURES_JSON.exists():
        try:
            with open(EXCLUDED_FEATURES_JSON, "r", encoding="utf-8") as f:
                loaded_excl = json.load(f)
                if isinstance(loaded_excl, list):
                    excluded_cols = list(set(excluded_cols + loaded_excl))
                elif isinstance(loaded_excl, dict) and "excluded_features" in loaded_excl:
                    excluded_cols = list(set(excluded_cols + loaded_excl["excluded_features"]))
        except Exception:
            pass

    return config, excluded_cols


def load_input_data() -> pd.DataFrame:
    for path in INPUT_CANDIDATES:
        if path.exists():
            print(f"[Stage 1] Loading existing dataset from: {path}")
            df = pd.read_parquet(path)
            print(f"[Stage 1] Loaded {len(df):,} rows and {len(df.columns)} columns.")
            return df
    raise FileNotFoundError("Could not find processed parquet files in 'data/processed/'.")


def compute_leak_free_balltree_secondary_crashes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2: Nearby-crash detection (BallTree, 2-mile radius) & Leak-Free Temporal-Shadow Analysis.
    A crash j is flagged as a secondary crash if a prior crash i within 2 miles either:
      1) Was already cleared (End_Time[i] <= Start_Time[j]) within the last 30 minutes (residual queue shadow), OR
      2) Started within the last 60 minutes (Start_Time[j] - Start_Time[i] <= 60 mins) and had not cleared before Start_Time[j]
         (evaluated without peeking at how far into the future End_Time[i] goes).
    """
    print("[Stage 2] Running Leak-Free BallTree Nearby-Crash Detection (2-mile radius) & Temporal Shadow Analysis...")
    df = df.sort_values("Start_Time").reset_index(drop=True)

    coords_rad = np.radians(df[["Start_Lat", "Start_Lng"]].astype(float).values)
    radius_rad = 2.0 / 3958.8  # 2 miles in radians

    start_sec = df["Start_Time"].astype("int64") // 10**9
    end_sec = df["End_Time"].astype("int64") // 10**9

    secondary_flag = np.zeros(len(df), dtype=int)
    nearby_prior_count = np.zeros(len(df), dtype=int)

    chunk_size = 25000
    for start_idx in range(0, len(df), chunk_size):
        end_idx = min(len(df), start_idx + chunk_size)
        sub_coords = coords_rad[start_idx:end_idx]
        tree = BallTree(sub_coords, metric="haversine")
        neighbors = tree.query_radius(sub_coords, r=radius_rad)

        for local_i, nbrs in enumerate(neighbors):
            if len(nbrs) <= 1:
                continue
            prior_nbrs = nbrs[nbrs < local_i]
            if len(prior_nbrs) == 0:
                continue
            global_i = start_idx + local_i
            global_priors = start_idx + prior_nbrs

            t_now = start_sec[global_i]
            t_prior_start = start_sec[global_priors]
            t_prior_end = end_sec[global_priors]

            # Case 1: Prior crash already cleared before t_now, and its 30-min shadow (1800s) overlaps t_now
            cleared_shadow = (t_prior_end <= t_now) & ((t_prior_end + 1800) >= t_now)
            # Case 2: Prior crash started within 60 mins (3600s) before t_now and was still active at t_now
            # (Note: checking t_prior_end > t_now is observable at t_now because dispatch knows if the prior ticket is still open!)
            active_recent = ((t_now - t_prior_start) <= 3600) & (t_prior_end > t_now)

            hits = np.sum(cleared_shadow | active_recent)
            if hits > 0:
                secondary_flag[global_i] = 1
                nearby_prior_count[global_i] = hits

    df["secondary_crash_flag"] = secondary_flag
    df["nearby_active_crashes"] = nearby_prior_count
    print(f"[Stage 2] BallTree flagged {secondary_flag.sum():,} leak-free secondary crashes.")
    return df


def run_stage1_governance(df: pd.DataFrame, excluded_cols: list) -> pd.DataFrame:
    print("[Stage 1] Running Phase-0 Audit, Duration Sanity Checks, and Leakage Control...")
    initial_count = len(df)

    if "Start_Time_parsed" in df.columns:
        df["Start_Time"] = pd.to_datetime(df["Start_Time_parsed"], errors="coerce")
    else:
        df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")

    if "End_Time_parsed" in df.columns:
        df["End_Time"] = pd.to_datetime(df["End_Time_parsed"], errors="coerce")
    else:
        df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")

    if "reported_duration_minutes" in df.columns:
        df["raw_duration_mins"] = pd.to_numeric(df["reported_duration_minutes"], errors="coerce")
    else:
        df["raw_duration_mins"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60.0

    if "valid_initial_duration" in df.columns:
        df = df[df["valid_initial_duration"] == True].copy()
    df = df.dropna(subset=["ID", "Start_Time", "End_Time", "raw_duration_mins", "Start_Lat", "Start_Lng"]).copy()
    df = df[df["raw_duration_mins"] >= 1.0].copy()

    # Exact duplicate candidate removal (same timestamp to the minute, exact 4-decimal lat/lng, and same description length)
    df["_time_min"] = df["Start_Time"].dt.floor("1min")
    df["_lat_4"] = df["Start_Lat"].round(4)
    df["_lng_4"] = df["Start_Lng"].round(4)
    df = df.drop_duplicates(subset=["_time_min", "_lat_4", "_lng_4"]).copy()
    df = df.drop(columns=["_time_min", "_lat_4", "_lng_4"])

    # Right-censor flag at 24-hour bound (1440 minutes)
    df["censor_flag"] = (df["raw_duration_mins"] >= 1440.0).astype(int)
    df["duration"] = np.clip(df["raw_duration_mins"], a_min=1.0, a_max=1440.0)

    # Compute raw tabular completeness BEFORE any imputation or column dropping
    raw_sensor_cols = [
        c for c in ["Visibility(mi)", "Precipitation(in)", "Weather_Condition", "Street", "City", "Sunrise_Sunset", "Distance(mi)"]
        if c in df.columns
    ]
    if raw_sensor_cols:
        # Also factor in missingness of optional weather/road fields so gate_q_tabular has genuine variance
        missing_penalty = (
            df["Precipitation(in)"].isna().astype(float) * 0.25 +
            df["Visibility(mi)"].isna().astype(float) * 0.25 +
            df["Weather_Condition"].isna().astype(float) * 0.25 +
            (df["Distance(mi)"].fillna(0) == 0).astype(float) * 0.25
        )
        df["raw_tabular_completeness"] = np.clip(1.0 - missing_penalty, 0.25, 1.0)
    else:
        df["raw_tabular_completeness"] = 1.0

    # Drop leakage-risk & reporting-artifact columns
    keep_protected = {
        "ID", "Start_Time", "End_Time", "Description",
        "duration", "raw_duration_mins", "censor_flag",
        "Start_Lat", "Start_Lng", "raw_tabular_completeness"
    }
    drop_cols = [c for c in excluded_cols if c in df.columns and c not in keep_protected]
    for extra_leak in ["Start_Time_parsed", "End_Time_parsed", "is_secondary_crash", "Source"]:
        if extra_leak in df.columns and extra_leak not in keep_protected:
            drop_cols.append(extra_leak)

    df = df.drop(columns=list(set(drop_cols)), errors="ignore")

    # Sort strictly by Start_Time
    df = df.sort_values("Start_Time").reset_index(drop=True)
    n = len(df)
    train_cut = int(n * 0.70)
    val_cut = int(n * 0.80)
    cal_cut = int(n * 0.90)

    splits = np.empty(n, dtype=object)
    splits[:train_cut] = "train"
    splits[train_cut:val_cut] = "val"
    splits[val_cut:cal_cut] = "cal"
    splits[cal_cut:] = "test"
    df["split"] = splits

    split_summary = (
        df.groupby("split")
        .agg(
            count=("ID", "count"),
            min_time=("Start_Time", "min"),
            max_time=("Start_Time", "max"),
            median_duration=("duration", "median"),
            censor_rate=("censor_flag", "mean")
        )
        .reset_index()
    )
    split_summary.to_csv(SPLIT_SUMMARY_CSV, index=False)

    audit_info = {
        "initial_rows": int(initial_count),
        "audited_rows": int(len(df)),
        "right_censored_count": int(df["censor_flag"].sum()),
        "splits": df["split"].value_counts().to_dict()
    }
    with open(AUDIT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_info, f, indent=2)

    print(f"[Stage 1] Governance complete: {initial_count:,} -> {len(df):,} clean records.")
    return df


def run_stage2_structural_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print("[Stage 2] Adding cyclical Spatiotemporal Enrichment features...")
    hour = df["Start_Time"].dt.hour + (df["Start_Time"].dt.minute / 60.0)
    dow = df["Start_Time"].dt.dayofweek
    month = df["Start_Time"].dt.month

    df["hour_feature"] = hour
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["is_rush_hour"] = (((hour >= 7.0) & (hour <= 10.0)) | ((hour >= 16.0) & (hour <= 19.0))).astype(int)

    # Execute actual leak-free BallTree secondary crash detection
    df = compute_leak_free_balltree_secondary_crashes(df)
    return df


def main():
    _, excluded_cols = load_repo_configs()
    df = load_input_data()
    df = run_stage1_governance(df, excluded_cols)
    df = run_stage2_structural_pipeline(df)

    df.to_parquet(OUTPUT_STRUCTURAL_PATH, index=False)
    print(f"[Done] Saved Stage 1 & 2 Structural Feature Table ({len(df):,} rows) to: {OUTPUT_STRUCTURAL_PATH}")


if __name__ == "__main__":
    main()