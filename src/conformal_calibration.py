import pandas as pd
import numpy as np
from pathlib import Path

def run_calibration():
    print("================================================================")
    print("  STEP 4: CONFORMAL CALIBRATION (PICP & MPIW Evaluation)       ")
    print("================================================================")
    
    input_path = Path("data/processed/merged_duration_outputs.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run generate_merged_predictions.py first.")
        
    df = pd.read_csv(input_path)
    target = 'reported_duration_minutes'
    q10_col, q90_col = 'CatBoost_Q10', 'CatBoost_Q90'

    # Raw CatBoost Interval Metrics
    raw_covered = (df[target] >= df[q10_col]) & (df[target] <= df[q90_col])
    raw_picp = raw_covered.mean() * 100
    raw_mpiw = (df[q90_col] - df[q10_col]).mean()

    print(f"[Raw CatBoost Quantile Bounds]")
    print(f"  Coverage (PICP)        : {raw_picp:.2f}% (Target: 80.00%)")
    print(f"  Mean Interval Width    : {raw_mpiw:.2f} mins")

    # Conformal Calibration
    target_coverage = 0.80
    conformal_errors = np.maximum(df[target] - df[q90_col], df[q10_col] - df[target])
    q_shift = np.percentile(conformal_errors, target_coverage * 100)

    # Adjust bounds by conformal shift
    df['Calibrated_Q10'] = np.maximum(1.0, df[q10_col] - q_shift)
    df['Calibrated_Q90'] = np.clip(df[q90_col] + q_shift, 1.0, 1440.0)

    cal_covered = (df[target] >= df['Calibrated_Q10']) & (df[target] <= df['Calibrated_Q90'])
    cal_picp = cal_covered.mean() * 100
    cal_mpiw = (df['Calibrated_Q90'] - df['Calibrated_Q10']).mean()

    print(f"\n[Conformal-Calibrated Bounds]")
    print(f"  Conformal Shift Margin : ±{q_shift:.2f} mins")
    print(f"  Calibrated PICP        : {cal_picp:.2f}%")
    print(f"  Calibrated MPIW        : {cal_mpiw:.2f} mins")

    reports_dir = Path("reports/phase1")
    reports_dir.mkdir(parents=True, exist_ok=True)

    calibration_metrics = pd.DataFrame([{
        'Method': 'Raw CatBoost Quantile (Pre-Calibration)',
        'PICP_Coverage_Pct': raw_picp,
        'MPIW_Width_Mins': raw_mpiw
    }, {
        'Method': 'Conformal-Calibrated (Proposed Framework)',
        'PICP_Coverage_Pct': cal_picp,
        'MPIW_Width_Mins': cal_mpiw
    }])
    calibration_metrics.to_csv(reports_dir / "conformal_calibration_metrics.csv", index=False)

    output_path = Path("data/processed/merged_calibrated_outputs.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Step 4 complete. Calibrated predictions saved to {output_path}")

if __name__ == "__main__":
    run_calibration()