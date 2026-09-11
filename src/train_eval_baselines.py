import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from pathlib import Path

def compute_metrics(y_true, y_pred):
    """Calculates full suite of regression metrics."""
    y_pred = np.maximum(y_pred, 1.0) # Prevent zero/negative div errors
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {'MAE (Mins)': mae, 'MedAE (Mins)': medae, 'RMSE': rmse, 'MAPE (%)': mape}

def evaluate_models():
    print("[1/3] Loading All 4 Ablation Datasets...")
    processed_dir = Path("data/processed")
    
    # Ensure proper ordering for the final printouts
    datasets = {
        '1. Struct Only': pd.read_csv(processed_dir / "us_accidents_struct_only.csv"),
        '2. Struct+Sem': pd.read_csv(processed_dir / "us_accidents_struct_sem.csv"),
        '3. Struct+Emb': pd.read_csv(processed_dir / "us_accidents_struct_emb.csv"),
        '4. Dumb Join': pd.read_csv(processed_dir / "us_accidents_dumb_join.csv"),
        '5. Gated Fusion': pd.read_csv(processed_dir / "us_accidents_fused_multimodal.csv")
    }
    
    target_col = 'reported_duration_minutes'
    
    # ---------------------------------------------------------
    # STRICT 24-HOUR FILTER & ROW ALIGNMENT
    # ---------------------------------------------------------
    master_df = datasets['5. Gated Fusion']
    valid_mask = (master_df[target_col] > 0) & (master_df[target_col] <= 1440)
    master_df = master_df[valid_mask]
    y_master = master_df[target_col]
    
    idx_train, idx_test = train_test_split(master_df.index, test_size=0.2, random_state=42)
    y_train, y_test = y_master.loc[idx_train], y_master.loc[idx_test]

    # ---------------------------------------------------------
    # OPTIMIZED HYPERPARAMETERS
    # ---------------------------------------------------------
    p_xgb_log = {'n_estimators': 200, 'learning_rate': 0.0639, 'max_depth': 8, 'subsample': 0.8388, 'colsample_bytree': 0.9112, 'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1}
    p_xgb_aft = {'learning_rate': 0.0153, 'max_depth': 10, 'subsample': 0.8854, 'colsample_bytree': 0.8474, 'objective': 'survival:aft', 'eval_metric': 'aft-nloglik', 'tree_method': 'hist', 'seed': 42, 'nthread': -1}
    p_cat = {'iterations': 400, 'learning_rate': 0.1333, 'depth': 7, 'l2_leaf_reg': 9.9559, 'loss_function': 'MultiQuantile:alpha=0.1,0.5,0.9', 'verbose': 0, 'random_seed': 42, 'thread_count': -1}

    overall_results = []
    severity_results = []

    print("[2/3] Training and Evaluating Architectures across Modalities...")
    
    for ds_name, df in datasets.items():
        print(f"  -> Processing: {ds_name}")
        
        df_aligned = df.loc[master_df.index]
        X = df_aligned.drop(columns=[target_col, 'ID']).select_dtypes(include=[np.number])
        
        X_train, X_test = X.loc[idx_train], X.loc[idx_test]
        
        # Helper function for Severity Analysis
        def record_severity_metrics(arch, preds):
            if 'Severity' in X_test.columns:
                for sev in sorted(X_test['Severity'].unique()):
                    mask = X_test['Severity'] == sev
                    if mask.sum() > 0:
                        medae = median_absolute_error(y_test[mask], preds[mask])
                        severity_results.append({
                            'Dataset': ds_name, 'Architecture': arch, 
                            'Severity': sev, 'Count': mask.sum(), 'MedAE (Mins)': medae
                        })

        # --- 1. XGBoost Log-Transform ---
        model_log = XGBRegressor(**p_xgb_log)
        model_log.fit(X_train, np.log1p(y_train))
        preds_log = np.expm1(model_log.predict(X_test))
        metrics = compute_metrics(y_test, preds_log)
        overall_results.append({'Architecture': '1. XGB Log', 'Dataset': ds_name, **metrics})
        record_severity_metrics('1. XGB Log', preds_log)

        # --- 2. XGBoost AFT ---
        dtrain = xgb.DMatrix(X_train)
        dtest = xgb.DMatrix(X_test)
        dtrain.set_float_info('label_lower_bound', y_train.values)
        dtrain.set_float_info('label_upper_bound', y_train.values)
        bst_aft = xgb.train(p_xgb_aft, dtrain, num_boost_round=200)
        preds_aft = bst_aft.predict(dtest)
        metrics = compute_metrics(y_test, preds_aft)
        overall_results.append({'Architecture': '2. XGB AFT', 'Dataset': ds_name, **metrics})
        record_severity_metrics('2. XGB AFT', preds_aft)

        # --- 3. CatBoost Quantile ---
        model_cat = CatBoostRegressor(**p_cat)
        model_cat.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=30)
        preds_cat = model_cat.predict(X_test)[:, 1] # Extract Median (50th percentile)
        metrics = compute_metrics(y_test, preds_cat)
        overall_results.append({'Architecture': '3. Cat Quantile', 'Dataset': ds_name, **metrics})
        record_severity_metrics('3. Cat Quantile', preds_cat)

    print("\n[3/3] Generating Final Reports...")
    Path("reports/phase0").mkdir(parents=True, exist_ok=True)
    
    # 1. Overall Ablation Report
    df_overall = pd.DataFrame(overall_results)
    df_overall.to_csv("reports/phase0/overall_modality_ablation.csv", index=False)
    
    # 2. Severity Breakdown Report
    df_sev = pd.DataFrame(severity_results)
    df_sev.to_csv("reports/phase0/severity_error_breakdown.csv", index=False)
    
    # ---------------------------------------------------------
    # CONSOLE PRINTOUTS
    # ---------------------------------------------------------
    print("\n" + "="*85)
    print(" 📊 FULL METRICS: ALL ARCHITECTURES BY MODALITY")
    print("="*85)
    
    # Set multi-index for a beautiful grouped terminal output
    display_overall = df_overall.set_index(['Architecture', 'Dataset'])
    display_overall = display_overall[['MedAE (Mins)', 'MAE (Mins)', 'RMSE', 'MAPE (%)']]
    print(display_overall.round(2).to_string())

    print("\n" + "="*85)
    print(" 🚨 ERROR BY SEVERITY (MedAE Mins) - ALL ARCHITECTURES")
    print("="*85)
    
    # Pivot the severity table to show Architecture & Dataset as rows, and Severity 1-4 as columns
    pivot_sev = df_sev.pivot(index=['Architecture', 'Dataset'], columns='Severity', values='MedAE (Mins)')
    print(pivot_sev.round(2).to_string())

if __name__ == "__main__":
    evaluate_models()