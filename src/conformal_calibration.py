import pandas as pd
import numpy as np
from pathlib import Path

def run_calibration():
    print("================================================================")
    print("  STEP 4: CONFORMAL CALIBRATION (MoE Point Prediction Bounds)   ")
    print("================================================================")
    
    input_path = Path("data/processed/merged_duration_outputs.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run generate_merged_predictions.py first.")
        
    df = pd.read_csv(input_path)
    target = 'reported_duration_minutes'
    pred_col = 'MoE_Expected'

    # 1. Isolate the Calibration Split to strictly prevent Data Leakage
    df_calib = df[df['Data_Split'] == 'Calibration'].copy()
    df_test = df[df['Data_Split'] == 'Test'].copy()

    print(f"[Calibration Phase]")
    print(f"  -> Extracting Absolute Residuals from {len(df_calib)} Calibration samples...")
    
    # Calculate absolute residuals for the MoE point predictions
    residuals = np.abs(df_calib[target] - df_calib[pred_col])
    
    # Conformal Math: Calculate the exact quantile shift needed for 60% coverage (UPDATED for Routing)
    target_coverage = 0.60
    n = len(residuals)
    
    # Finite-sample correction factor for split conformal prediction: (n+1)/n
    q_level = min(1.0, np.ceil((n + 1) * target_coverage) / n)
    q_shift = np.quantile(residuals, q_level)

    print(f"  -> Conformal Shift Margin Calculated: ±{q_shift:.2f} mins")

    # 2. Apply the mathematically guaranteed bounds blindly to the unseen Test Set
    print(f"\n[Test Phase - Independent Evaluation]")
    print(f"  -> Applying ±{q_shift:.2f} min bounds to {len(df_test)} unseen Test samples...")
    
    df_test['Calibrated_Q10'] = np.maximum(1.0, df_test[pred_col] - q_shift)
    df_test['Calibrated_Q90'] = np.clip(df_test[pred_col] + q_shift, 1.0, 1440.0)

    # 3. Evaluate the exact Coverage
    cal_covered = (df_test[target] >= df_test['Calibrated_Q10']) & (df_test[target] <= df_test['Calibrated_Q90'])
    cal_picp = cal_covered.mean() * 100
    cal_mpiw = (df_test['Calibrated_Q90'] - df_test['Calibrated_Q10']).mean()

    print(f"\n[Conformal-Calibrated Bounds (Test Set)]")
    print(f"  Target Coverage (PICP) : {target_coverage * 100:.2f}%")
    print(f"  Actual Test PICP       : {cal_picp:.2f}%")
    print(f"  Calibrated MPIW        : {cal_mpiw:.2f} mins")

    reports_dir = Path("reports/phase1")
    reports_dir.mkdir(parents=True, exist_ok=True)

    calibration_metrics = pd.DataFrame([{
        'Method': 'MoE Conformal-Calibrated Bounds',
        'Target_PICP_Pct': target_coverage * 100,
        'Actual_Test_PICP_Pct': cal_picp,
        'MPIW_Width_Mins': cal_mpiw
    }])
    calibration_metrics.to_csv(reports_dir / "conformal_calibration_metrics.csv", index=False)

    # Save final payload for the Reliability Gate (Box 6)
    output_path = Path("data/processed/merged_calibrated_outputs.csv")
    df_test.to_csv(output_path, index=False)
    print(f"\n✅ Step 4 complete. Calibrated Test predictions saved to {output_path}")

if __name__ == "__main__":
    run_calibration()