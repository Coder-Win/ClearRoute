import pandas as pd
import numpy as np
from lifelines import LogNormalAFTFitter, WeibullAFTFitter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from pathlib import Path

def run_traditional_hbdm():
    print("================================================================")
    print("  STEP 1: TRADITIONAL HBDM BASELINES (Khan et al., 2026)        ")
    print("================================================================")
    
    data_path = Path("data/processed/us_accidents_struct_only_v2.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run reliability_fusion2.py first.")
        
    df = pd.read_csv(data_path)
    target = 'reported_duration_minutes'
    valid_mask = (df[target] > 0) & (df[target] <= 1440)
    df = df[valid_mask].copy()

    if 'event_observed' not in df.columns:
        df['event_observed'] = 1

    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42)
    df_train, df_test = df.loc[idx_train], df.loc[idx_test]

    # Select numerical features and drop zero-variance / ID columns
    df_num = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in df_num.columns if c not in ['ID', target, 'event_observed'] and df_num[c].nunique() > 1]
    
    train_core = df_train[[target, 'event_observed'] + feature_cols].fillna(0).copy()
    test_core = df_test[feature_cols].fillna(0).copy()

    results = []
    reports_dir = Path("reports/phase1")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Traditional LogNormal AFT (Best model in Khan et al., 2026)
    print("\n[1/2] Fitting Traditional LogNormal AFT Fitter...")
    lnf = LogNormalAFTFitter(penalizer=0.1)
    lnf.fit(train_core, duration_col=target, event_col='event_observed')
    
    preds_lnf_raw = lnf.predict_expectation(test_core).values.flatten()
    preds_lnf = np.clip(np.nan_to_num(preds_lnf_raw, nan=60.0), 1.0, 1440.0)
    
    mae_lnf = mean_absolute_error(df_test[target], preds_lnf)
    medae_lnf = median_absolute_error(df_test[target], preds_lnf)
    rmse_lnf = np.sqrt(mean_squared_error(df_test[target], preds_lnf))
    
    print(f"      LogNormal AFT -> MedAE: {medae_lnf:.2f} mins | MAE: {mae_lnf:.2f} mins | RMSE: {rmse_lnf:.2f}")
    results.append({'Model': 'Traditional LogNormal AFT (Khan 2026)', 'MedAE': medae_lnf, 'MAE': mae_lnf, 'RMSE': rmse_lnf})
    lnf.summary.to_csv(reports_dir / "traditional_lognormal_coefficients.csv")

    # 2. Traditional Weibull AFT
    print("\n[2/2] Fitting Traditional Weibull AFT Fitter...")
    weibull = WeibullAFTFitter(penalizer=0.1)
    weibull.fit(train_core, duration_col=target, event_col='event_observed')
    
    preds_weib_raw = weibull.predict_expectation(test_core).values.flatten()
    preds_weib = np.clip(np.nan_to_num(preds_weib_raw, nan=60.0), 1.0, 1440.0)
    
    mae_weib = mean_absolute_error(df_test[target], preds_weib)
    medae_weib = median_absolute_error(df_test[target], preds_weib)
    rmse_weib = np.sqrt(mean_squared_error(df_test[target], preds_weib))
    
    print(f"      Weibull AFT   -> MedAE: {medae_weib:.2f} mins | MAE: {mae_weib:.2f} mins | RMSE: {rmse_weib:.2f}")
    results.append({'Model': 'Traditional Weibull AFT (Khan 2026)', 'MedAE': medae_weib, 'MAE': mae_weib, 'RMSE': rmse_weib})
    weibull.summary.to_csv(reports_dir / "traditional_weibull_coefficients.csv")

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(reports_dir / "traditional_hbdm_metrics.csv", index=False)
    print(f"\n✅ Step 1 complete. Baseline metrics saved to {reports_dir / 'traditional_hbdm_metrics.csv'}")

if __name__ == "__main__":
    run_traditional_hbdm()