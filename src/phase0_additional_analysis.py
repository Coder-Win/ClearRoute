from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/us_accidents_phase0_processed.parquet"
)
OUTPUT_DIR = Path("reports/phase0")


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Processed file not found: {INPUT_PATH}"
        )

    dataframe = pd.read_parquet(INPUT_PATH)

    required = {
        "reported_duration_minutes",
        "valid_initial_duration",
        "event_year",
    }

    missing = required.difference(dataframe.columns)

    if missing:
        raise ValueError(
            f"Required columns missing from processed data: "
            f"{sorted(missing)}"
        )

    return dataframe


def get_valid_records(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.loc[
        dataframe["valid_initial_duration"].fillna(False)
        & dataframe["reported_duration_minutes"].notna()
    ].copy()


def grouped_duration_summary(
    dataframe: pd.DataFrame,
    group_column: str,
    filename: str,
) -> pd.DataFrame:
    if group_column not in dataframe.columns:
        print(f"[SKIP] Column not available: {group_column}")
        return pd.DataFrame()

    grouped = dataframe.groupby(
        group_column,
        dropna=False,
    )["reported_duration_minutes"]

    basic = grouped.agg(
        count="count",
        mean="mean",
        median="median",
        standard_deviation="std",
        minimum="min",
        maximum="max",
    ).reset_index()

    quantiles = (
        grouped.quantile(
            [0.10, 0.25, 0.75, 0.90, 0.95, 0.99]
        )
        .unstack()
        .reset_index()
        .rename(
            columns={
                0.10: "q10",
                0.25: "q25",
                0.75: "q75",
                0.90: "q90",
                0.95: "q95",
                0.99: "q99",
            }
        )
    )

    result = basic.merge(
        quantiles,
        on=group_column,
        how="left",
    )

    result["iqr"] = result["q75"] - result["q25"]

    result["mean_median_ratio"] = (
        result["mean"]
        / result["median"].replace(0, np.nan)
    )

    path = OUTPUT_DIR / filename
    result.to_csv(path, index=False)

    print(f"[OK] Wrote {path}")
    return result


def create_extreme_duration_reports(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    thresholds = {
        "over_24_hours": 24 * 60,
        "over_72_hours": 72 * 60,
        "over_7_days": 7 * 24 * 60,
    }

    counts = {}

    for label, threshold in thresholds.items():
        subset = dataframe.loc[
            dataframe["reported_duration_minutes"] > threshold
        ].copy()

        counts[label] = len(subset)

        columns = [
            column
            for column in [
                "ID",
                "Source",
                "Severity",
                "Start_Time_parsed",
                "End_Time_parsed",
                "reported_duration_minutes",
                "Description",
                "Street",
                "City",
                "County",
                "State",
                "event_year",
                "split",
            ]
            if column in subset.columns
        ]

        output_path = OUTPUT_DIR / f"us_{label}.csv"

        subset[columns].sort_values(
            "reported_duration_minutes",
            ascending=False,
        ).to_csv(output_path, index=False)

        print(f"[OK] Wrote {output_path}")

    return counts


def analyze_round_durations(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    candidate_values = [
        5,
        10,
        15,
        30,
        45,
        60,
        90,
        120,
        180,
        240,
        360,
        720,
        1440,
    ]

    duration = dataframe[
        "reported_duration_minutes"
    ].astype(float)

    rows = []

    for candidate in candidate_values:
        exact_count = int(
            np.isclose(
                duration,
                candidate,
                atol=0.01,
            ).sum()
        )

        rows.append(
            {
                "duration_minutes": candidate,
                "exact_count": exact_count,
                "percentage": (
                    exact_count / len(dataframe) * 100
                    if len(dataframe)
                    else np.nan
                ),
            }
        )

    report = pd.DataFrame(rows)

    output_path = (
        OUTPUT_DIR / "us_round_duration_frequency.csv"
    )
    report.to_csv(output_path, index=False)

    print(f"[OK] Wrote {output_path}")
    return report


def analyze_description_quality(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if "Description" not in dataframe.columns:
        print("[SKIP] Description column not available.")
        return pd.DataFrame()

    normalized = (
        dataframe["Description"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    lengths = normalized.str.len()

    quality_report = pd.DataFrame(
        [
            {
                "metric": "total_valid_duration_rows",
                "value": len(dataframe),
            },
            {
                "metric": "missing_or_empty_descriptions",
                "value": int(normalized.eq("").sum()),
            },
            {
                "metric": "descriptions_under_20_characters",
                "value": int(lengths.lt(20).sum()),
            },
            {
                "metric": "descriptions_under_50_characters",
                "value": int(lengths.lt(50).sum()),
            },
            {
                "metric": "unique_descriptions",
                "value": int(normalized.nunique()),
            },
            {
                "metric": "rows_with_repeated_descriptions",
                "value": int(
                    normalized.duplicated(keep=False).sum()
                ),
            },
            {
                "metric": "median_description_length",
                "value": float(lengths.median()),
            },
            {
                "metric": "mean_description_length",
                "value": float(lengths.mean()),
            },
        ]
    )

    quality_path = (
        OUTPUT_DIR / "us_description_quality.csv"
    )
    quality_report.to_csv(quality_path, index=False)

    repeated = (
        normalized.value_counts()
        .head(200)
        .rename_axis("description_normalized")
        .reset_index(name="count")
    )

    repeated_path = (
        OUTPUT_DIR / "us_top_repeated_descriptions.csv"
    )
    repeated.to_csv(repeated_path, index=False)

    print(f"[OK] Wrote {quality_path}")
    print(f"[OK] Wrote {repeated_path}")

    return quality_report


def analyze_source_by_year(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    required = {"Source", "event_year"}

    if not required.issubset(dataframe.columns):
        print("[SKIP] Source or event_year is unavailable.")
        return pd.DataFrame()

    result = (
        dataframe.groupby(
            ["event_year", "Source"],
            dropna=False,
        )
        .size()
        .reset_index(name="row_count")
    )

    totals = (
        result.groupby("event_year")["row_count"]
        .sum()
        .rename("year_total")
        .reset_index()
    )

    result = result.merge(
        totals,
        on="event_year",
        how="left",
    )

    result["percentage_within_year"] = (
        result["row_count"] / result["year_total"] * 100
    )

    output_path = OUTPUT_DIR / "us_source_by_year.csv"
    result.to_csv(output_path, index=False)

    print(f"[OK] Wrote {output_path}")
    return result


def analyze_duration_bins(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    bins = [
        0,
        30,
        60,
        120,
        360,
        1440,
        np.inf,
    ]

    labels = [
        "00_0_to_30",
        "01_31_to_60",
        "02_61_to_120",
        "03_121_to_360",
        "04_361_to_1440",
        "05_over_1440",
    ]

    dataframe = dataframe.copy()

    dataframe["duration_group"] = pd.cut(
        dataframe["reported_duration_minutes"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    report = (
        dataframe.groupby(
            "duration_group",
            observed=False,
        )
        .size()
        .reset_index(name="row_count")
    )

    report["percentage"] = (
        report["row_count"] / len(dataframe) * 100
    )

    output_path = (
        OUTPUT_DIR / "us_duration_group_distribution.csv"
    )
    report.to_csv(output_path, index=False)

    print(f"[OK] Wrote {output_path}")
    return report


def save_summary(
    dataframe: pd.DataFrame,
    extreme_counts: dict[str, int],
) -> None:
    summary = {
        "valid_duration_rows": int(len(dataframe)),
        "mean_minutes": float(
            dataframe["reported_duration_minutes"].mean()
        ),
        "median_minutes": float(
            dataframe["reported_duration_minutes"].median()
        ),
        "q90_minutes": float(
            dataframe[
                "reported_duration_minutes"
            ].quantile(0.90)
        ),
        "q95_minutes": float(
            dataframe[
                "reported_duration_minutes"
            ].quantile(0.95)
        ),
        "q99_minutes": float(
            dataframe[
                "reported_duration_minutes"
            ].quantile(0.99)
        ),
        "extreme_duration_counts": extreme_counts,
    }

    output_path = (
        OUTPUT_DIR / "additional_analysis_summary.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"[OK] Wrote {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = load_data()
    valid = get_valid_records(dataframe)

    print(f"Processed rows loaded: {len(dataframe):,}")
    print(f"Valid-duration rows: {len(valid):,}")

    grouped_duration_summary(
        valid,
        "Source",
        "us_duration_by_source.csv",
    )

    grouped_duration_summary(
        valid,
        "Severity",
        "us_duration_by_severity.csv",
    )

    grouped_duration_summary(
        valid,
        "event_year",
        "us_duration_by_year.csv",
    )

    grouped_duration_summary(
        valid,
        "State",
        "us_duration_by_state.csv",
    )

    grouped_duration_summary(
        valid,
        "Weather_Condition",
        "us_duration_by_weather.csv",
    )

    extreme_counts = create_extreme_duration_reports(
        valid
    )

    analyze_round_durations(valid)
    analyze_description_quality(valid)
    analyze_source_by_year(valid)
    analyze_duration_bins(valid)
    save_summary(valid, extreme_counts)

    print("\n[COMPLETE] Additional Phase 0 analysis finished.")


if __name__ == "__main__":
    main()