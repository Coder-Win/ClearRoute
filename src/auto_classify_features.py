from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path(
    "reports/phase0/feature_availability_template.csv"
)
DEFAULT_OUTPUT_DIR = Path("reports/phase0")
DEFAULT_CONFIG_DIR = Path("configs/generated")


# -------------------------------------------------------------------
# Exact field rules are evaluated before general keyword rules.
# -------------------------------------------------------------------

US_EXACT_RULES: dict[str, dict[str, Any]] = {
    "ID": {
        "availability_stage": "IDENTIFIER",
        "recommended_policy": "EXCLUDE",
        "feature_role": "IDENTIFIER",
        "leakage_risk": "LOW",
        "reason": "Unique incident identifier; retain for tracking only.",
    },
    "Source": {
        "availability_stage": "T0_INITIAL",
        "recommended_policy": "RELAXED_T0",
        "feature_role": "DATA_SOURCE",
        "leakage_risk": "MEDIUM",
        "reason": (
            "Known from the reporting system, but may let the model "
            "learn provider-specific duration conventions."
        ),
    },
    "Severity": {
        "availability_stage": "REVIEW_REQUIRED",
        "recommended_policy": "REVIEW",
        "feature_role": "INCIDENT_SEVERITY",
        "leakage_risk": "MEDIUM",
        "reason": (
            "Severity may be assigned initially or updated later. "
            "Run strict models without it and relaxed models with it."
        ),
    },
    "Start_Time": {
        "availability_stage": "T0_INITIAL",
        "recommended_policy": "STRICT_T0",
        "feature_role": "TEMPORAL",
        "leakage_risk": "LOW",
        "reason": "Incident start timestamp.",
    },
    "End_Time": {
        "availability_stage": "TARGET_DERIVED",
        "recommended_policy": "EXCLUDE",
        "feature_role": "TARGET",
        "leakage_risk": "CRITICAL",
        "reason": "Directly used to calculate reported event duration.",
    },
    "Start_Lat": {
        "availability_stage": "T0_INITIAL",
        "recommended_policy": "STRICT_T0",
        "feature_role": "LOCATION",
        "leakage_risk": "LOW",
        "reason": "Initial incident latitude.",
    },
    "Start_Lng": {
        "availability_stage": "T0_INITIAL",
        "recommended_policy": "STRICT_T0",
        "feature_role": "LOCATION",
        "leakage_risk": "LOW",
        "reason": "Initial incident longitude.",
    },
    "End_Lat": {
        "availability_stage": "REVIEW_REQUIRED",
        "recommended_policy": "REVIEW",
        "feature_role": "INCIDENT_EXTENT",
        "leakage_risk": "HIGH",
        "reason": (
            "May describe the final affected extent rather than "
            "information available at initial notification."
        ),
    },
    "End_Lng": {
        "availability_stage": "REVIEW_REQUIRED",
        "recommended_policy": "REVIEW",
        "feature_role": "INCIDENT_EXTENT",
        "leakage_risk": "HIGH",
        "reason": (
            "May describe the final affected extent rather than "
            "information available at initial notification."
        ),
    },
    "Distance(mi)": {
        "availability_stage": "REVIEW_REQUIRED",
        "recommended_policy": "REVIEW",
        "feature_role": "INCIDENT_EXTENT",
        "leakage_risk": "HIGH",
        "reason": (
            "Affected distance may be updated as the event develops. "
            "Exclude from strict T0 and test separately."
        ),
    },
    "Description": {
        "availability_stage": "T0_INITIAL",
        "recommended_policy": "STRICT_T0",
        "feature_role": "INCIDENT_TEXT",
        "leakage_risk": "MEDIUM",
        "reason": (
            "Used as the initial narrative. Verify that descriptions "
            "are not final updated summaries."
        ),
    },
    "Weather_Timestamp": {
        "availability_stage": "T0_INITIAL",
        "recommended_policy": "RELAXED_T0",
        "feature_role": "WEATHER_METADATA",
        "leakage_risk": "LOW",
        "reason": (
            "Use to calculate weather observation age; normally do "
            "not use the raw timestamp directly."
        ),
    },
}


IOWA_EXACT_RULES: dict[str, dict[str, Any]] = {
    "X": {
        "availability_stage": "STATIC_CONTEXT",
        "recommended_policy": "STRICT_T0",
        "feature_role": "LOCATION",
        "leakage_risk": "LOW",
        "reason": "Incident coordinate; validate coordinate reference system.",
    },
    "Y": {
        "availability_stage": "STATIC_CONTEXT",
        "recommended_policy": "STRICT_T0",
        "feature_role": "LOCATION",
        "leakage_risk": "LOW",
        "reason": "Incident coordinate; validate coordinate reference system.",
    },
    "XCOORD": {
        "availability_stage": "STATIC_CONTEXT",
        "recommended_policy": "REVIEW",
        "feature_role": "LOCATION_DUPLICATE",
        "leakage_risk": "LOW",
        "reason": (
            "Possible duplicate coordinate representation. Select either "
            "X/Y or XCOORD/YCOORD after coordinate validation."
        ),
    },
    "YCOORD": {
        "availability_stage": "STATIC_CONTEXT",
        "recommended_policy": "REVIEW",
        "feature_role": "LOCATION_DUPLICATE",
        "leakage_risk": "LOW",
        "reason": (
            "Possible duplicate coordinate representation. Select either "
            "X/Y or XCOORD/YCOORD after coordinate validation."
        ),
    },
    "OBJECTID": {
        "availability_stage": "IDENTIFIER",
        "recommended_policy": "EXCLUDE",
        "feature_role": "IDENTIFIER",
        "leakage_risk": "LOW",
        "reason": "ArcGIS row identifier.",
    },
    "CRASH_KEY": {
        "availability_stage": "IDENTIFIER",
        "recommended_policy": "EXCLUDE",
        "feature_role": "IDENTIFIER",
        "leakage_risk": "LOW",
        "reason": "Crash identifier; retain for grouping only.",
    },
    "CASENUMBER": {
        "availability_stage": "IDENTIFIER",
        "recommended_policy": "EXCLUDE",
        "feature_role": "IDENTIFIER",
        "leakage_risk": "LOW",
        "reason": "Case identifier; retain for grouping only.",
    },
    "LECASENUM": {
        "availability_stage": "IDENTIFIER",
        "recommended_policy": "EXCLUDE",
        "feature_role": "IDENTIFIER",
        "leakage_risk": "LOW",
        "reason": "Law-enforcement case identifier.",
    },
    "GLOBALID": {
        "availability_stage": "IDENTIFIER",
        "recommended_policy": "EXCLUDE",
        "feature_role": "IDENTIFIER",
        "leakage_risk": "LOW",
        "reason": "Global database identifier.",
    },
    "CRASH_DATE": {
        "availability_stage": "T0_INITIAL",
        "recommended_policy": "STRICT_T0",
        "feature_role": "TEMPORAL",
        "leakage_risk": "LOW",
        "reason": "Crash date.",
    },
    "CRASH_MONTH": {
        "availability_stage": "T0_INITIAL",
        "recommended_policy": "EXCLUDE",
        "feature_role": "REDUNDANT_TEMPORAL",
        "leakage_risk": "LOW",
        "reason": "Derive month consistently from CRASH_DATETIME.",
    },
    "CRASH_DAY": {
        "availability_stage": "T0_INITIAL",
        "recommended_policy": "EXCLUDE",
        "feature_role": "REDUNDANT_TEMPORAL",
        "leakage_risk": "LOW",
        "reason": "Derive weekday/day consistently from CRASH_DATETIME.",
    },
    "TIMESTR": {
        "availability_stage": "T0_INITIAL",
        "recommended_policy": "EXCLUDE",
        "feature_role": "REDUNDANT_TEMPORAL",
        "leakage_risk": "LOW",
        "reason": "Use CRASH_DATETIME rather than duplicate time strings.",
    },
    "CRASH_DATETIME": {
        "availability_stage": "T0_INITIAL",
        "recommended_policy": "STRICT_T0",
        "feature_role": "TEMPORAL",
        "leakage_risk": "LOW",
        "reason": "Primary crash timestamp.",
    },
    "CRASH_DATETIME_UTC": {
        "availability_stage": "T0_INITIAL",
        "recommended_policy": "EXCLUDE",
        "feature_role": "REDUNDANT_TEMPORAL",
        "leakage_risk": "LOW",
        "reason": (
            "Use one canonical time representation; avoid both local and "
            "UTC timestamps as predictors."
        ),
    },
    "CRASH_DATETIME_UTC_OFFSET": {
        "availability_stage": "METADATA",
        "recommended_policy": "EXCLUDE",
        "feature_role": "TIME_METADATA",
        "leakage_risk": "LOW",
        "reason": "Timezone conversion metadata.",
    },
    "REST_UPDATED": {
        "availability_stage": "METADATA",
        "recommended_policy": "EXCLUDE",
        "feature_role": "DATABASE_METADATA",
        "leakage_risk": "CRITICAL",
        "reason": "Database update time; not roadway-clearance time.",
    },
    "REST_UPDATE_UTC_OFFSET": {
        "availability_stage": "METADATA",
        "recommended_policy": "EXCLUDE",
        "feature_role": "DATABASE_METADATA",
        "leakage_risk": "HIGH",
        "reason": "REST update metadata.",
    },
    "FATALITIES": {
        "availability_stage": "T2_FINAL",
        "recommended_policy": "FACTOR_ANALYSIS_ONLY",
        "feature_role": "FINAL_OUTCOME",
        "leakage_risk": "HIGH",
        "reason": "Final crash outcome; may not be known initially.",
    },
    "INJURIES": {
        "availability_stage": "T1_DYNAMIC",
        "recommended_policy": "DYNAMIC_ONLY",
        "feature_role": "INCIDENT_OUTCOME",
        "leakage_risk": "MEDIUM",
        "reason": "May be confirmed during response rather than at notification.",
    },
    "MAJINJURY": {
        "availability_stage": "T2_FINAL",
        "recommended_policy": "FACTOR_ANALYSIS_ONLY",
        "feature_role": "FINAL_OUTCOME",
        "leakage_risk": "HIGH",
        "reason": "Final injury classification.",
    },
    "MININJURY": {
        "availability_stage": "T2_FINAL",
        "recommended_policy": "FACTOR_ANALYSIS_ONLY",
        "feature_role": "FINAL_OUTCOME",
        "leakage_risk": "HIGH",
        "reason": "Final injury classification.",
    },
    "POSSINJURY": {
        "availability_stage": "T1_DYNAMIC",
        "recommended_policy": "DYNAMIC_ONLY",
        "feature_role": "INCIDENT_OUTCOME",
        "leakage_risk": "MEDIUM",
        "reason": "Possible injury information may emerge during response.",
    },
    "UNKINJURY": {
        "availability_stage": "T2_FINAL",
        "recommended_policy": "FACTOR_ANALYSIS_ONLY",
        "feature_role": "FINAL_OUTCOME",
        "leakage_risk": "HIGH",
        "reason": "Final report classification.",
    },
    "PROPDMG": {
        "availability_stage": "T2_FINAL",
        "recommended_policy": "FACTOR_ANALYSIS_ONLY",
        "feature_role": "FINAL_OUTCOME",
        "leakage_risk": "HIGH",
        "reason": "Final property-damage estimate.",
    },
    "DRUGALC": {
        "availability_stage": "T2_FINAL",
        "recommended_policy": "FACTOR_ANALYSIS_ONLY",
        "feature_role": "INVESTIGATION_RESULT",
        "leakage_risk": "HIGH",
        "reason": "Often determined through later investigation.",
    },
    "CSEV": {
        "availability_stage": "REVIEW_REQUIRED",
        "recommended_policy": "REVIEW",
        "feature_role": "CRASH_SEVERITY",
        "leakage_risk": "HIGH",
        "reason": (
            "May represent final crash severity. Exclude in strict T0 and "
            "verify before dynamic use."
        ),
    },
    "VEHICLES": {
        "availability_stage": "T1_DYNAMIC",
        "recommended_policy": "DYNAMIC_ONLY",
        "feature_role": "INCIDENT_CHARACTERISTIC",
        "leakage_risk": "MEDIUM",
        "reason": "Exact vehicle count may require scene confirmation.",
    },
    "TOCCUPANTS": {
        "availability_stage": "T1_DYNAMIC",
        "recommended_policy": "DYNAMIC_ONLY",
        "feature_role": "INCIDENT_CHARACTERISTIC",
        "leakage_risk": "MEDIUM",
        "reason": "Occupant count may require scene confirmation.",
    },
    "REPORT": {
        "availability_stage": "REVIEW_REQUIRED",
        "recommended_policy": "REVIEW",
        "feature_role": "REPORT_METADATA",
        "leakage_risk": "MEDIUM",
        "reason": "Meaning requires Iowa data dictionary.",
    },
}


# -------------------------------------------------------------------
# Group membership
# -------------------------------------------------------------------

US_LOCATION_FIELDS = {
    "Street", "City", "County", "State", "Zipcode",
    "Country", "Timezone", "Airport_Code"
}

US_WEATHER_FIELDS = {
    "Temperature(F)", "Wind_Chill(F)", "Humidity(%)",
    "Pressure(in)", "Visibility(mi)", "Wind_Direction",
    "Wind_Speed(mph)", "Precipitation(in)", "Weather_Condition"
}

US_ROAD_CONTEXT_FIELDS = {
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction",
    "No_Exit", "Railway", "Roundabout", "Station", "Stop",
    "Traffic_Calming", "Traffic_Signal", "Turning_Loop"
}

US_LIGHT_FIELDS = {
    "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight",
    "Astronomical_Twilight"
}

IOWA_STATIC_CONTEXT_FIELDS = {
    "DISTRICT", "COUNTY_NUMBER", "CITY_NUMBER", "SYSTEMSTR",
    "LITERAL", "LIGHT", "CSRFCND", "WEATHER", "RDTYP",
    "PAVED", "WZRELATED", "CITY_NAME", "COUNTY_NAME",
    "CARDINAL"
}

IOWA_FINAL_OR_INVESTIGATIVE_FIELDS = {
    "FRSTHARM", "LOCFSTHRM", "CRCOMNNR", "MAJCSE",
    "ECNTCRC", "RCNTCRC"
}


def standard_rule(
    stage: str,
    policy: str,
    role: str,
    risk: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "availability_stage": stage,
        "recommended_policy": policy,
        "feature_role": role,
        "leakage_risk": risk,
        "reason": reason,
    }


def classify_us_feature(feature: str) -> dict[str, Any]:
    if feature in US_EXACT_RULES:
        return US_EXACT_RULES[feature]

    if feature in US_LOCATION_FIELDS:
        return standard_rule(
            "STATIC_CONTEXT",
            "STRICT_T0",
            "GEOGRAPHIC_CONTEXT",
            "LOW",
            "Existing geographic context available from incident location.",
        )

    if feature in US_WEATHER_FIELDS:
        return standard_rule(
            "T0_INITIAL",
            "STRICT_T0",
            "ENVIRONMENT",
            "LOW",
            "Weather observation associated with incident start time.",
        )

    if feature in US_ROAD_CONTEXT_FIELDS:
        return standard_rule(
            "STATIC_CONTEXT",
            "STRICT_T0",
            "ROAD_CONTEXT",
            "LOW",
            "Road or nearby infrastructure context.",
        )

    if feature in US_LIGHT_FIELDS:
        return standard_rule(
            "T0_INITIAL",
            "STRICT_T0",
            "LIGHTING_CONTEXT",
            "LOW",
            "Lighting or twilight condition at incident time.",
        )

    return standard_rule(
        "REVIEW_REQUIRED",
        "REVIEW",
        "UNKNOWN",
        "UNKNOWN",
        "No reliable automatic US rule matched this field.",
    )


def classify_iowa_feature(feature: str) -> dict[str, Any]:
    if feature in IOWA_EXACT_RULES:
        return IOWA_EXACT_RULES[feature]

    if feature in IOWA_STATIC_CONTEXT_FIELDS:
        return standard_rule(
            "STATIC_CONTEXT",
            "STRICT_T0",
            "ROAD_OR_ENVIRONMENT_CONTEXT",
            "LOW",
            "Road, location, lighting, weather, or work-zone context.",
        )

    if feature in IOWA_FINAL_OR_INVESTIGATIVE_FIELDS:
        return standard_rule(
            "REVIEW_REQUIRED",
            "FACTOR_ANALYSIS_ONLY",
            "FINAL_OR_INVESTIGATIVE_CHARACTERISTIC",
            "HIGH",
            (
                "Field likely comes from the completed crash report. "
                "Do not use for strict initial prediction without confirmation."
            ),
        )

    return standard_rule(
        "REVIEW_REQUIRED",
        "REVIEW",
        "UNKNOWN",
        "UNKNOWN",
        "No reliable automatic Iowa rule matched this field.",
    )


def classify_feature(dataset: str, feature: str) -> dict[str, Any]:
    if dataset == "US_Accidents":
        return classify_us_feature(feature)

    if dataset == "Iowa_DOT":
        return classify_iowa_feature(feature)

    return standard_rule(
        "REVIEW_REQUIRED",
        "REVIEW",
        "UNKNOWN",
        "UNKNOWN",
        f"Unknown dataset name: {dataset}",
    )


def sanitize_filename(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def save_json(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def create_feature_sets(
    classified: pd.DataFrame,
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for dataset, group in classified.groupby("dataset"):
        strict = group.loc[
            group["recommended_policy"].eq("STRICT_T0"),
            "feature_name",
        ].tolist()

        relaxed = group.loc[
            group["recommended_policy"].isin(
                ["STRICT_T0", "RELAXED_T0"]
            ),
            "feature_name",
        ].tolist()

        dynamic = group.loc[
            group["recommended_policy"].isin(
                ["STRICT_T0", "RELAXED_T0", "DYNAMIC_ONLY"]
            ),
            "feature_name",
        ].tolist()

        factors = group.loc[
            group["recommended_policy"].isin(
                [
                    "STRICT_T0",
                    "RELAXED_T0",
                    "DYNAMIC_ONLY",
                    "FACTOR_ANALYSIS_ONLY",
                ]
            ),
            "feature_name",
        ].tolist()

        excluded = group.loc[
            group["recommended_policy"].eq("EXCLUDE"),
            "feature_name",
        ].tolist()

        review = group.loc[
            group["recommended_policy"].eq("REVIEW"),
            "feature_name",
        ].tolist()

        output[dataset] = {
            "strict_initial_features": strict,
            "relaxed_initial_features": relaxed,
            "dynamic_features": dynamic,
            "factor_analysis_features": factors,
            "excluded_features": excluded,
            "review_required": review,
        }

    return output


def main(
    input_path: Path,
    output_dir: Path,
    config_dir: Path,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input template not found: {input_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(input_path)

    required_columns = {"dataset", "feature_name"}
    missing = required_columns.difference(source.columns)

    if missing:
        raise ValueError(
            f"Template is missing columns: {sorted(missing)}"
        )

    classified_rows = []

    for row in source.itertuples(index=False):
        dataset = str(row.dataset).strip()
        feature = str(row.feature_name).strip()

        rule = classify_feature(dataset, feature)

        classified_rows.append(
            {
                "dataset": dataset,
                "feature_name": feature,
                "availability_stage": rule["availability_stage"],
                "recommended_policy": rule["recommended_policy"],
                "use_initial_strict": (
                    "YES"
                    if rule["recommended_policy"] == "STRICT_T0"
                    else "NO"
                ),
                "use_initial_relaxed": (
                    "YES"
                    if rule["recommended_policy"]
                    in {"STRICT_T0", "RELAXED_T0"}
                    else "NO"
                ),
                "use_dynamic_model": (
                    "YES"
                    if rule["recommended_policy"]
                    in {
                        "STRICT_T0",
                        "RELAXED_T0",
                        "DYNAMIC_ONLY",
                    }
                    else "NO"
                ),
                "use_factor_analysis": (
                    "YES"
                    if rule["recommended_policy"]
                    in {
                        "STRICT_T0",
                        "RELAXED_T0",
                        "DYNAMIC_ONLY",
                        "FACTOR_ANALYSIS_ONLY",
                    }
                    else "NO"
                ),
                "feature_role": rule["feature_role"],
                "leakage_risk": rule["leakage_risk"],
                "reason": rule["reason"],
                "review_decision": "",
                "review_notes": "",
            }
        )

    classified = pd.DataFrame(classified_rows)

    auto_path = output_dir / "feature_availability_auto.csv"
    classified.to_csv(auto_path, index=False)

    review_required = classified.loc[
        classified["availability_stage"].eq("REVIEW_REQUIRED")
        | classified["recommended_policy"].eq("REVIEW")
    ].copy()

    review_path = output_dir / "feature_review_required.csv"
    review_required.to_csv(review_path, index=False)

    summary = (
        classified.groupby(
            [
                "dataset",
                "availability_stage",
                "recommended_policy",
                "leakage_risk",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="feature_count")
    )

    summary_path = output_dir / "feature_policy_summary.csv"
    summary.to_csv(summary_path, index=False)

    feature_sets = create_feature_sets(classified)

    all_sets_path = config_dir / "feature_sets_all.json"
    save_json(feature_sets, all_sets_path)

    for dataset, values in feature_sets.items():
        name = sanitize_filename(dataset)

        for set_name, features in values.items():
            save_json(
                features,
                config_dir / f"{name}_{set_name}.json",
            )

    print(f"[OK] Classified {len(classified)} features.")
    print(f"[OK] Full classification: {auto_path}")
    print(
        f"[OK] Manual review reduced to "
        f"{len(review_required)} features: {review_path}"
    )
    print(f"[OK] Policy summary: {summary_path}")
    print(f"[OK] Generated feature sets: {all_sets_path}")

    print("\nClassification counts:")
    print(
        classified.groupby(
            ["dataset", "recommended_policy"]
        )
        .size()
        .to_string()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Automatically classify feature availability and leakage risk."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
    )

    args = parser.parse_args()

    main(
        input_path=args.input,
        output_dir=args.output_dir,
        config_dir=args.config_dir,
    )