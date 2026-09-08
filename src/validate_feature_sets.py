from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CONFIG_PATH = Path("configs/generated/feature_sets_all.json")
US_DATA_PATH = Path(
    "data/processed/us_accidents_phase0_processed.parquet"
)
IOWA_DATA_PATH = Path(
    "data/processed/iowa_phase0_processed.parquet"
)
REPORT_PATH = Path(
    "reports/phase0/generated_feature_set_validation.csv"
)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_dataset_features(
    dataframe: pd.DataFrame,
    dataset_name: str,
    feature_sets: dict,
) -> list[dict]:
    results = []
    available_columns = set(dataframe.columns)

    for feature_set_name, feature_list in feature_sets.items():
        present = [
            feature
            for feature in feature_list
            if feature in available_columns
        ]

        missing = [
            feature
            for feature in feature_list
            if feature not in available_columns
        ]

        results.append(
            {
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "configured_feature_count": len(feature_list),
                "present_feature_count": len(present),
                "missing_feature_count": len(missing),
                "present_features": " | ".join(present),
                "missing_features": " | ".join(missing),
            }
        )

    return results


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    configurations = load_json(CONFIG_PATH)

    us_data = pd.read_parquet(US_DATA_PATH)
    iowa_data = pd.read_parquet(IOWA_DATA_PATH)

    results = []

    if "US_Accidents" in configurations:
        results.extend(
            validate_dataset_features(
                us_data,
                "US_Accidents",
                configurations["US_Accidents"],
            )
        )

    if "Iowa_DOT" in configurations:
        results.extend(
            validate_dataset_features(
                iowa_data,
                "Iowa_DOT",
                configurations["Iowa_DOT"],
            )
        )

    report = pd.DataFrame(results)
    report.to_csv(REPORT_PATH, index=False)

    print(f"[OK] Validation report written to: {REPORT_PATH}")

    print("\nFeature-set validation summary:")
    print(
        report[
            [
                "dataset",
                "feature_set",
                "configured_feature_count",
                "present_feature_count",
                "missing_feature_count",
            ]
        ].to_string(index=False)
    )

    missing_report = report.loc[
        report["missing_feature_count"] > 0
    ]

    if missing_report.empty:
        print("\n[OK] All configured features exist in processed files.")
    else:
        print(
            "\n[WARNING] Some configured features are absent from "
            "the processed Parquet files."
        )
        print(
            missing_report[
                [
                    "dataset",
                    "feature_set",
                    "missing_features",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()