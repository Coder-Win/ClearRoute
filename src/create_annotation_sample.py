from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/us_accidents_phase0_processed.parquet"
)
OUTPUT_DIR = Path("annotations")
OUTPUT_PATH = OUTPUT_DIR / "semantic_annotation_pilot.csv"

RANDOM_SEED = 42
DESIRED_SAMPLE_SIZE = 300


def load_valid_data() -> pd.DataFrame:
    dataframe = pd.read_parquet(INPUT_PATH)

    required = {
        "ID",
        "Description",
        "reported_duration_minutes",
        "valid_initial_duration",
        "Severity",
        "Source",
        "event_year",
    }

    missing = required.difference(dataframe.columns)

    if missing:
        raise ValueError(
            f"Required columns missing: {sorted(missing)}"
        )

    dataframe = dataframe.loc[
        dataframe["valid_initial_duration"].fillna(False)
        & dataframe["Description"].notna()
    ].copy()

    dataframe["description_clean"] = (
        dataframe["Description"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    dataframe = dataframe.loc[
        dataframe["description_clean"].str.len() >= 15
    ].copy()

    return dataframe


def add_sampling_groups(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    duration_bins = [
        0,
        30,
        60,
        120,
        360,
        1440,
        np.inf,
    ]

    duration_labels = [
        "D1_0_30",
        "D2_31_60",
        "D3_61_120",
        "D4_121_360",
        "D5_361_1440",
        "D6_over_1440",
    ]

    dataframe["duration_group"] = pd.cut(
        dataframe["reported_duration_minutes"],
        bins=duration_bins,
        labels=duration_labels,
        include_lowest=True,
    )

    description_length = dataframe[
        "description_clean"
    ].str.len()

    dataframe["description_group"] = pd.cut(
        description_length,
        bins=[0, 60, 120, 250, np.inf],
        labels=[
            "T1_short",
            "T2_medium",
            "T3_long",
            "T4_very_long",
        ],
        include_lowest=True,
    )

    dataframe["sampling_group"] = (
        dataframe["duration_group"].astype(str)
        + "|S"
        + dataframe["Severity"].astype(str)
        + "|"
        + dataframe["description_group"].astype(str)
    )

    return dataframe


def proportional_stratified_sample(
    dataframe: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    groups = dataframe.groupby(
        "sampling_group",
        dropna=False,
    )

    group_counts = groups.size()

    allocations = (
        group_counts / len(dataframe) * sample_size
    ).round().astype(int)

    allocations = allocations.clip(lower=1)

    sampled_parts = []

    for group_name, group in groups:
        requested = int(
            allocations.get(group_name, 1)
        )

        requested = min(
            requested,
            len(group),
        )

        sampled = group.sample(
            n=requested,
            random_state=RANDOM_SEED,
        )

        sampled_parts.append(sampled)

    sample = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    if len(sample) > sample_size:
        sample = sample.sample(
            n=sample_size,
            random_state=RANDOM_SEED,
        )

    if len(sample) < sample_size:
        remaining = dataframe.loc[
            ~dataframe["ID"].isin(sample["ID"])
        ]

        additional = remaining.sample(
            n=min(
                sample_size - len(sample),
                len(remaining),
            ),
            random_state=RANDOM_SEED,
        )

        sample = pd.concat(
            [sample, additional],
            ignore_index=True,
        )

    return sample.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)


def create_annotation_columns(
    sample: pd.DataFrame,
) -> pd.DataFrame:
    output_columns = [
        column
        for column in [
            "ID",
            "Source",
            "Severity",
            "event_year",
            "reported_duration_minutes",
            "duration_group",
            "description_group",
            "Description",
        ]
        if column in sample.columns
    ]

    result = sample[output_columns].copy()

    annotation_columns = {
        "ann_incident_type": "",
        "ann_heavy_vehicle": "",
        "ann_rollover": "",
        "ann_spill": "",
        "ann_fire": "",
        "ann_debris": "",
        "ann_injury": "",
        "ann_lanes_blocked": "",
        "ann_full_road_blockage": "",
        "ann_ramp_involvement": "",
        "ann_disabled_vehicle": "",
        "ann_hazardous_material": "",
        "ann_specialized_recovery_required": "",
        "ann_emergency_response_mentioned": "",
        "ann_uncertainty_language": "",
        "annotation_notes": "",
        "annotator_id": "",
        "annotation_status": "UNLABELLED",
    }

    for column, default_value in annotation_columns.items():
        result[column] = default_value

    result.insert(
        0,
        "annotation_row_id",
        range(1, len(result) + 1),
    )

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = load_valid_data()
    dataframe = add_sampling_groups(dataframe)

    sample = proportional_stratified_sample(
        dataframe,
        sample_size=DESIRED_SAMPLE_SIZE,
    )

    annotation_file = create_annotation_columns(
        sample
    )

    annotation_file.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    distribution = (
        annotation_file.groupby(
            [
                "duration_group",
                "Severity",
                "description_group",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="row_count")
    )

    distribution.to_csv(
        OUTPUT_DIR / "semantic_annotation_distribution.csv",
        index=False,
    )

    print(
        f"[OK] Annotation sample written to: {OUTPUT_PATH}"
    )
    print(
        f"[OK] Sample size: {len(annotation_file)}"
    )
    print(
        "[OK] Distribution report written to: "
        f"{OUTPUT_DIR / 'semantic_annotation_distribution.csv'}"
    )


if __name__ == "__main__":
    main()