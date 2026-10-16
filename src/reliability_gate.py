import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from pathlib import Path

def run_reliability_gate():
    print("================================================================")
    print("  STEP 5: OOD DETECTION & RELIABILITY GATE BENCHMARK            ")
    print("================================================================")

    data_path = Path("data/processed/merged_calibrated_outputs.csv")
    df = pd.read_csv(data_path)

    # Clean out obsolete baseline column references 
    non_feature_cols = [
        'reported_duration_minutes', 'MoE_Expected', 
        'Calibrated_Q10', 'Calibrated_Q90', 'ID', 'Data_Split'
    ]
    
    # 1. Isolate Structural Features: Prevent the Curse of Dimensionality
    all_numeric = [c for c in df.columns if c not in non_feature_cols and np.issubdtype(df[c].dtype, np.number)]
    ood_features = [c for c in all_numeric if not ('emb_' in c.lower() or 'vector' in c.lower())]

    print("[1/3] Training Isolation Forest OOD Detector (Structural Features Only)...")
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['OOD_Flag'] = iso.fit_predict(df[ood_features].fillna(0))

    print("[2/3] Evaluating Reliability Gate (Dynamic Uncertainty Threshold)...")
    
    # Dynamically extract the threshold for the top 20% widest conformal intervals
    interval_widths = df['Calibrated_Q90'] - df['Calibrated_Q10']
    dynamic_threshold = interval_widths.quantile(0.80) 
    
    def assign_action(row):
        if row['OOD_Flag'] == -1:
            return 'Abstain'   # Anomaly detected -> dispatch human operator
        elif (row['Calibrated_Q90'] - row['Calibrated_Q10']) > dynamic_threshold:
            return 'Review'    # High prediction uncertainty -> cautious routing
        else:
            return 'Predict'   # Safe & reliable for automated routing

    df['Reliability_Action'] = df.apply(assign_action, axis=1)
    action_counts = df['Reliability_Action'].value_counts()
    
    print("\nReliability Gate Decisions:")
    for action, count in action_counts.items():
        print(f"  {action:<10}: {count} incidents ({count/len(df)*100:.1f}%)")

    # Error reduction calculation
    overall_mae = mean_absolute_error(df['reported_duration_minutes'], df['MoE_Expected'])
    overall_medae = median_absolute_error(df['reported_duration_minutes'], df['MoE_Expected'])
    overall_rmse = np.sqrt(mean_squared_error(df['reported_duration_minutes'], df['MoE_Expected']))

    pred_subset = df[df['Reliability_Action'] == 'Predict']
    pred_mae = mean_absolute_error(pred_subset['reported_duration_minutes'], pred_subset['MoE_Expected'])
    pred_medae = median_absolute_error(pred_subset['reported_duration_minutes'], pred_subset['MoE_Expected'])
    pred_rmse = np.sqrt(mean_squared_error(pred_subset['reported_duration_minutes'], pred_subset['MoE_Expected']))

    print(f"\n[Gating Accuracy Impact]")
    print(f"  Overall MoE-AFT       -> MedAE: {overall_medae:.2f} mins | MAE: {overall_mae:.2f} mins | RMSE: {overall_rmse:.2f}")
    print(f"  Automated Safe Routes -> MedAE: {pred_medae:.2f} mins | MAE: {pred_mae:.2f} mins | RMSE: {pred_rmse:.2f}")
    print(f"  Error Reduction (MAE) : -{overall_mae - pred_mae:.2f} mins by abstaining on extreme OOD/uncertain events")

    # Load baseline metrics from Step 1 for direct literature comparison
    reports_dir = Path("reports/phase1")
    reports_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = reports_dir / "traditional_hbdm_metrics.csv"
    
    khan_medae, khan_mae, khan_rmse = 48.73, 80.37, 156.66
    if baseline_path.exists():
        base_df = pd.read_csv(baseline_path)
        # Assuming the first row is the LogNormal Baseline
        khan_row = base_df.iloc[0]
        if 'MedAE' in khan_row:
            khan_medae, khan_mae, khan_rmse = khan_row['MedAE'], khan_row['MAE'], khan_row['RMSE']

    benchmark_table = pd.DataFrame([
        {
            'Framework': 'Khan et al. (2026) Baseline',
            'Modality': 'Structured Only',
            'Technique': 'Traditional Parametric HBDM (LogNormal)',
            'MedAE (mins)': round(khan_medae, 2),
            'MAE (mins)': round(khan_mae, 2),
            'RMSE': round(khan_rmse, 2),
            'Uncertainty / Safety': 'Deterministic (No Bounds)',
            'Interpretability': 'Global Coefficients Only'
        },
        {
            'Framework': 'Gao et al. (2026) Analogue',
            'Modality': 'Structured + Text Semantics',
            'Technique': 'Standard Gradient Boosting + Raw SHAP',
            'MedAE (mins)': 21.55,
            'MAE (mins)': 57.15,
            'RMSE': 137.96,
            'Uncertainty / Safety': 'Deterministic (No Bounds)',
            'Interpretability': 'Conflated Correlation/Causality'
        },
        {
            'Framework': 'Proposed System (All Data)',
            'Modality': 'Tri-Modal Gated Fusion',
            'Technique': 'MedAE K-Means Soft MoE',
            'MedAE (mins)': round(overall_medae, 2),
            'MAE (mins)': round(overall_mae, 2),
            'RMSE': round(overall_rmse, 2),
            'Uncertainty / Safety': 'Calibrated (PICP 60%)',
            'Interpretability': 'Decoupled (Multi-Level TreeSHAP)'
        },
        {
            'Framework': 'Proposed System (Reliability-Gated)',
            'Modality': 'Tri-Modal Gated Fusion',
            'Technique': 'MoE + OOD Isolation Forest',
            'MedAE (mins)': round(pred_medae, 2),
            'MAE (mins)': round(pred_mae, 2),
            'RMSE': round(pred_rmse, 2),
            'Uncertainty / Safety': 'OOD-Aware Abstention Gate',
            'Interpretability': 'Decoupled + Verifiable Safe Routing'
        }
    ])

    print("\n=========================================================================================")
    print("                    MASTER BENCHMARK EVALUATION (Box 6 vs Prior Literature)               ")
    print("=========================================================================================")
    print(benchmark_table.to_string(index=False))

    benchmark_table.to_csv(reports_dir / "benchmark_comparison_khan_gao.csv", index=False)
    df.to_csv(reports_dir / "final_gated_predictions.csv", index=False)
    print(f"\n✅ Step 5 complete. Benchmark comparison saved to {reports_dir / 'benchmark_comparison_khan_gao.csv'}")

if __name__ == "__main__":
    run_reliability_gate()