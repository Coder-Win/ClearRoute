from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


LOGGER = logging.getLogger("phase0_audit")


US_REQUIRED_COLUMNS = [
    "ID",
    "Source",
    "Severity",
    "Start_Time",
    "End_Time",
    "Start_Lat",
    "Start_Lng",
    "Distance(mi)",
    "Description",
    "Street",
    "City",
    "County",
    "State",
    "Timezone",
    "Weather_Timestamp",
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Precipitation(in)",
    "Weather_Condition",
    "Junction",
    "Traffic_Signal",
    "Sunrise_Sunset",
]

IOWA_REQUIRED_COLUMNS = [
    "CRASH_KEY",
    "CASENUMBER",
    "CRASH_DATE",
    "CRASH_MONTH",
    "CRASH_DAY",
    "TIMESTR",
    "DISTRICT",
    "COUNTY_NUMBER",
    "CITY_NUMBER",
    "SYSTEMSTR",
    "FRSTHARM",
    "LOCFSTHRM",
    "MAJCSE",
    "DRUGALC",
    "LIGHT",
    "CSRFCND",
    "WEATHER",
    "RDTYP",
    "WZRELATED",
    "CSEV",
    "FATALITIES",
    "INJURIES",
    "PROPDMG",
    "VEHICLES",
    "TOCCUPANTS",
    "CRASH_DATETIME",
    "CRASH_DATETIME_UTC",
    "CITY_NAME",
    "COUNTY_NAME",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_directories(config: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(config["paths"]["output_directory"])
    processed_dir = Path(config["paths"]["processed_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, processed_dir


def inspect_header(csv_path: Path) -> listreturn:
    return pd.read_csv(csv_path, nrows=0).columns.tolist()


def validate_columns(
    available_columns: list[str],
    required_columns: list[str],
    dataset_name: str,
) -> dict[str, list[str]]:
    available_set = set(available_columns)
    missing = [col for col in required_columns if col not in available_set]
    present = [col for col in required_columns if col in available_set]

    LOGGER.info(
        "%s: %d expected columns present; %d missing",
        dataset_name,
        len(present),
        len(missing),
    )

    if missing:
        LOGGER.warning("%s missing columns: %s", dataset_name, missing)

    return {"present": present, "missing": missing}


def dtype_name(series: pd.Series) -> str:
    return str(series.dtype)


def profile_dataframe(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    total = len(dataframe)

    rows = []
    for column in dataframe.columns:
        series = dataframe[column]
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))

        rows.append(
            {
                "dataset": dataset_name,
                "feature_name": column,
                "dtype": dtype_name(series),
                "row_count": total,
                "missing_count": missing_count,
                "missing_percentage": (
                    100.0 * missing_count / total if total else np.nan
                ),
                "unique_count": unique_count,
                "example_value": (
                    str(series.dropna().iloc[0])
                    if not series.dropna().empty
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def read_us_accidents(
    csv_path: Path,
    usecols: list[str],
    chunksize: int,
) -> pd.DataFrame:
    chunks = []

    for index, chunk in enumerate(
        pd.read_csv(
            csv_path,
            usecols=lambda column: column in usecols,
            chunksize=chunksize,
            low_memory=False,
        )
    ):
        LOGGER.info("Reading US-Accidents chunk %d", index + 1)
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def prepare_us_accidents(
    dataframe: pd.DataFrame,
    min_duration: float,
    review_duration: float,
) -> pd.DataFrame:
    df = dataframe.copy()

    df["Start_Time_parsed"] = pd.to_datetime(
        df["Start_Time"],
        errors="coerce",
    )
    df["End_Time_parsed"] = pd.to_datetime(
        df["End_Time"],
        errors="coerce",
    )

    df["reported_duration_minutes"] = (
        df["End_Time_parsed"] - df["Start_Time_parsed"]
    ).dt.total_seconds() / 60.0

    df["dataset_source"] = "US_Accidents"
    df["target_name"] = "reported_event_duration"
    df["target_definition"] = "End_Time minus Start_Time"

    df["duration_missing"] = df["reported_duration_minutes"].isna()
    df["duration_negative"] = df["reported_duration_minutes"] < 0
    df["duration_zero"] = df["reported_duration_minutes"] == 0
    df["duration_below_minimum"] = (
        df["reported_duration_minutes"] < min_duration
    )
    df["duration_requires_review"] = (
        df["reported_duration_minutes"] > review_duration
    )

    df["valid_initial_duration"] = (
        ~df["duration_missing"]
        & ~df["duration_negative"]
        & ~df["duration_zero"]
        & ~df["duration_below_minimum"]
    )

    df["event_year"] = df["Start_Time_parsed"].dt.year
    df["event_month"] = df["Start_Time_parsed"].dt.month
    df["event_day_of_week"] = df["Start_Time_parsed"].dt.dayofweek
    df["event_hour"] = df["Start_Time_parsed"].dt.hour
    df["is_weekend"] = df["event_day_of_week"].isin([5, 6])

    if "Description" in df.columns:
        normalized_description = (
            df["Description"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df["description_normalized"] = normalized_description
        df["description_length"] = normalized_description.str.len()
        df["description_missing"] = (
            normalized_description.eq("")
        )

    return df


def create_us_exact_duplicate_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    duplicate_columns = [
        column
        for column in [
            "Source",
            "Start_Time_parsed",
            "End_Time_parsed",
            "Start_Lat",
            "Start_Lng",
            "Street",
            "description_normalized",
        ]
        if column in dataframe.columns
    ]

    if not duplicate_columns:
        return pd.DataFrame()

    duplicate_mask = dataframe.duplicated(
        subset=duplicate_columns,
        keep=False,
    )

    duplicate_rows = dataframe.loc[duplicate_mask].copy()
    duplicate_rows["duplicate_signature"] = (
        duplicate_rows[duplicate_columns]
        .astype(str)
        .agg("|".join, axis=1)
    )

    return duplicate_rows.sort_values("duplicate_signature")


def duration_summary(
    dataframe: pd.DataFrame,
    duration_column: str,
) -> pd.DataFrame:
    valid = dataframe.loc[
        dataframe["valid_initial_duration"],
        duration_column,
    ].dropna()

    quantiles = valid.quantile(
        [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    )

    rows = [
        {"statistic": "count", "value": float(valid.count())},
        {"statistic": "mean", "value": float(valid.mean())},
        {"statistic": "std", "value": float(valid.std())},
        {"statistic": "minimum", "value": float(valid.min())},
        {"statistic": "maximum", "value": float(valid.max())},
    ]

    for quantile, value in quantiles.items():
        rows.append(
            {
                "statistic": f"quantile_{quantile}",
                "value": float(value),
            }
        )

    return pd.DataFrame(rows)


def prepare_iowa(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()

    for column in [
        "CRASH_DATETIME",
        "CRASH_DATETIME_UTC",
        "REST_UPDATED",
    ]:
        if column in df.columns:
            df[f"{column}_parsed"] = pd.to_datetime(
                df[column],
                errors="coerce",
                utc=("UTC" in column),
            )

    df["dataset_source"] = "Iowa_DOT"
    df["target_name"] = "not_available_in_current_file"
    df["target_definition"] = (
        "No roadway-clearance or end timestamp identified"
    )

    if "CRASH_DATETIME_parsed" in df.columns:
        df["event_year"] = df["CRASH_DATETIME_parsed"].dt.year
        df["event_month"] = df["CRASH_DATETIME_parsed"].dt.month
        df["event_day_of_week"] = (
            df["CRASH_DATETIME_parsed"].dt.dayofweek
        )
        df["event_hour"] = df["CRASH_DATETIME_parsed"].dt.hour
        df["is_weekend"] = df["event_day_of_week"].isin([5, 6])

    return df


def detect_clearance_candidates(columns: list[str]) -> pd.DataFrame:
    keywords = [
        "clear",
        "open",
        "closure",
        "duration",
        "incident_end",
        "roadway_end",
        "lane_open",
        "notification",
        "arrival",
    ]

    rows = []
    for column in columns:
        normalized = column.lower()
        matched = [keyword for keyword in keywords if keyword in normalized]

        if matched:
            rows.append(
                {
                    "column": column,
                    "matched_keywords": ", ".join(matched),
                    "requires_manual_verification": True,
                }
            )

    return pd.DataFrame(rows)


def create_feature_availability_template(
    columns: list[str],
    dataset_name: str,
) -> pd.DataFrame:
    target_derived_keywords = [
        "end_time",
        "clearance",
        "roadway_cleared",
        "duration",
    ]

    post_event_keywords = [
        "rest_updated",
        "rest_update",
    ]

    rows = []
    for column in columns:
        lowered = column.lower()

        if any(keyword in lowered for keyword in target_derived_keywords):
            availability = "TARGET_OR_TARGET_DERIVED"
            leakage_risk = "HIGH"
        elif any(keyword in lowered for keyword in post_event_keywords):
            availability = "METADATA_OR_POST_RECORD_UPDATE"
            leakage_risk = "HIGH"
        else:
            availability = "REQUIRES_MANUAL_CLASSIFICATION"
            leakage_risk = "UNKNOWN"

        rows.append(
            {
                "dataset": dataset_name,
                "feature_name": column,
                "prediction_time_availability": availability,
                "possible_leakage": leakage_risk,
                "manual_notes": "",
            }
        )

    return pd.DataFrame(rows)


def create_chronological_split(
    dataframe: pd.DataFrame,
    year_column: str,
    train_years: list[int],
    validation_years: list[int],
    calibration_years: list[int],
    test_years: list[int],
) -> pd.Series:
    split = pd.Series(
        "unassigned",
        index=dataframe.index,
        dtype="object",
    )

    split.loc[dataframe[year_column].isin(train_years)] = "train"
    split.loc[dataframe[year_column].isin(validation_years)] = "validation"
    split.loc[dataframe[year_column].isin(calibration_years)] = "calibration"
    split.loc[dataframe[year_column].isin(test_years)] = "test"

    return split


def save_json(data: dict[str, Any], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def main(config_path: str) -> None:
    setup_logging()
    config = load_config(config_path)
    output_dir, processed_dir = ensure_directories(config)

    us_path = Path(config["paths"]["us_accidents"])
    iowa_path = Path(config["paths"]["iowa"])

    if not us_path.exists():
        raise FileNotFoundError(f"US-Accidents file not found: {us_path}")

    if not iowa_path.exists():
        raise FileNotFoundError(f"Iowa file not found: {iowa_path}")

    us_columns = inspect_header(us_path)
    iowa_columns = inspect_header(iowa_path)

    schema_report = {
        "US_Accidents": validate_columns(
            us_columns,
            US_REQUIRED_COLUMNS,
            "US_Accidents",
        ),
        "Iowa_DOT": validate_columns(
            iowa_columns,
            IOWA_REQUIRED_COLUMNS,
            "Iowa_DOT",
        ),
    }
    save_json(schema_report, output_dir / "schema_validation.json")

    iowa_clearance_candidates = detect_clearance_candidates(iowa_columns)
    iowa_clearance_candidates.to_csv(
        output_dir / "iowa_clearance_candidate_columns.csv",
        index=False,
    )

    us_usecols = [
        column for column in US_REQUIRED_COLUMNS if column in us_columns
    ]

    us_df = read_us_accidents(
        us_path,
        usecols=us_usecols,
        chunksize=int(config["audit"]["chunksize"]),
    )

    iowa_df = pd.read_csv(iowa_path, low_memory=False)

    us_profile = profile_dataframe(us_df, "US_Accidents")
    iowa_profile = profile_dataframe(iowa_df, "Iowa_DOT")

    pd.concat(
        [us_profile, iowa_profile],
        ignore_index=True,
    ).to_csv(
        output_dir / "column_profile.csv",
        index=False,
    )

    us_prepared = prepare_us_accidents(
        us_df,
        min_duration=float(config["duration"]["minimum_valid_minutes"]),
        review_duration=float(
            config["duration"]["review_threshold_minutes"]
        ),
    )

    iowa_prepared = prepare_iowa(iowa_df)

    us_duration_summary = duration_summary(
        us_prepared,
        "reported_duration_minutes",
    )
    us_duration_summary.to_csv(
        output_dir / "us_duration_summary.csv",
        index=False,
    )

    us_duplicates = create_us_exact_duplicate_report(us_prepared)
    us_duplicates.to_csv(
        output_dir / "us_exact_duplicate_candidates.csv",
        index=False,
    )

    us_availability = create_feature_availability_template(
        us_columns,
        "US_Accidents",
    )
    iowa_availability = create_feature_availability_template(
        iowa_columns,
        "Iowa_DOT",
    )

    pd.concat(
        [us_availability, iowa_availability],
        ignore_index=True,
    ).to_csv(
        output_dir / "feature_availability_template.csv",
        index=False,
    )

    available_us_years = sorted(
        int(year)
        for year in us_prepared["event_year"].dropna().unique()
    )
    LOGGER.info("US-Accidents years: %s", available_us_years)

    if set([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]).intersection(
        available_us_years
    ):
        us_prepared["split"] = create_chronological_split(
            us_prepared,
            year_column="event_year",
            train_years=[2016, 2017, 2018, 2019, 2020],
            validation_years=[2021],
            calibration_years=[2022],
            test_years=[2023],
        )
    else:
        us_prepared["split"] = "define_manually"

    us_output_columns = [
        column
        for column in [
            "ID",
            "Source",
            "Severity",
            "Start_Time_parsed",
            "End_Time_parsed",
            "reported_duration_minutes",
            "valid_initial_duration",
            "duration_requires_review",
            "Start_Lat",
            "Start_Lng",
            "Distance(mi)",
            "Description",
            "description_normalized",
            "description_length",
            "Street",
            "City",
            "County",
            "State",
            "Weather_Condition",
            "Visibility(mi)",
            "Precipitation(in)",
            "Junction",
            "Traffic_Signal",
            "Sunrise_Sunset",
            "event_year",
            "event_month",
            "event_day_of_week",
            "event_hour",
            "is_weekend",
            "dataset_source",
            "target_name",
            "target_definition",
            "split",
        ]
        if column in us_prepared.columns
    ]

    us_processed_path = (
        processed_dir / "us_accidents_phase0_processed.parquet"
    )
    us_prepared[us_output_columns].to_parquet(
        us_processed_path,
        index=False,
    )

    iowa_prepared.to_parquet(
        processed_dir / "iowa_phase0_processed.parquet",
        index=False,
    )

    split_summary = (
        us_prepared.groupby("split", dropna=False)
        .size()
        .reset_index(name="row_count")
    )
    split_summary.to_csv(
        output_dir / "us_split_summary.csv",
        index=False,
    )

    audit_summary = {
        "us_accidents": {
            "rows": len(us_prepared),
            "columns": len(us_prepared.columns),
            "valid_duration_rows": int(
                us_prepared["valid_initial_duration"].sum()
            ),
            "duration_review_rows": int(
                us_prepared["duration_requires_review"].sum()
            ),
            "exact_duplicate_candidate_rows": len(us_duplicates),
            "processed_file": str(us_processed_path),
        },
        "iowa": {
            "rows": len(iowa_prepared),
            "columns": len(iowa_prepared.columns),
            "clearance_target_detected": (
                not iowa_clearance_candidates.empty
            ),
            "clearance_target_status": (
                "Manual verification required"
                if not iowa_clearance_candidates.empty
                else "No clearance-like column detected"
            ),
            "processed_file": str(
                processed_dir / "iowa_phase0_processed.parquet"
            ),
        },
    }

    save_json(
        audit_summary,
        output_dir / "phase0_audit_summary.json",
    )

    LOGGER.info("Phase 0 audit complete.")
    LOGGER.info("Reports saved to %s", output_dir)
    LOGGER.info("Processed files saved to %s", processed_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 0 audit for US-Accidents and Iowa DOT."
    )
    parser.add_argument(
        "--config",
        default="configs/phase0_config.yaml",
        help="Path to the YAML configuration file.",
    )
    arguments = parser.parse_args()
    main(arguments.config)