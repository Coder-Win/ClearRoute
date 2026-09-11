import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from pathlib import Path

def compute_metrics(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1.0) # Prevent zero/negative division errors
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'MAE (Mins)': mae, 'MedAE (Mins)': medae, 'RMSE': rmse, 'MAPE (%)': mape}

def evaluate_models():
    print("[1/3] Loading All 5 V2 Ablation Datasets...")
    processed_dir = Path("data/processed")
    
    # Load the V2 datasets containing the new spatiotemporal features
    datasets = {
        '1. Struct Only': pd.read_csv(processed_dir / "us_accidents_struct_only_v2.csv"),
        '2. Struct+Sem': pd.read_csv(processed_dir / "us_accidents_struct_sem_v2.csv"),
        '3. Struct+Emb': pd.read_csv(processed_dir / "us_accidents_struct_emb_v2.csv"),
        '4. Dumb Join': pd.read_csv(processed_dir / "us_accidents_dumb_join_v2.csv"),
        '5. Gated Fusion': pd.read_csv(processed_dir / "us_accidents_fused_multimodal_v2.csv")
    }
    
    target_col = 'reported_duration_minutes'
    
    # Isolate the master logic using the fully fused dataset
    master_df = datasets['5. Gated Fusion']
    valid_mask = (master_df[target_col] > 0) & (master_df[target_col] <= 1440)
    master_df = master_df[valid_mask]
    
    y_master = master_df[target_col]
    
    # Extract the censoring flag (defaults to 1 if missing for safety)
    censor_master = master_df['event_observed'] if 'event_observed' in master_df.columns else pd.Series(np.ones(len(master_df)), index=master_df.index)
    
    # Consistent 80/20 train-test split indices
    idx_train, idx_test = train_test_split(master_df.index, test_size=0.2, random_state=42)
    y_train, y_test = y_master.loc[idx_train], y_master.loc[idx_test]
    censor_train, censor_test = censor_master.loc[idx_train], censor_master.loc[idx_test]

    # FIX 2: Calculate Sample Weights to handle class imbalance for severe crashes
    print("      -> Computing sample weights to balance extreme tail durations...")
    bins = np.digitize(y_train, bins=[30, 60, 120, 240, 1440])
    weights_train = compute_sample_weight(class_weight='balanced', y=bins)

    # Model Hyperparameters
    p_xgb_log = {'n_estimators': 200, 'learning_rate': 0.0639, 'max_depth': 8, 'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1}
    p_xgb_aft = {'learning_rate': 0.0153, 'max_depth': 10, 'objective': 'survival:aft', 'eval_metric': 'aft-nloglik', 'tree_method': 'hist', 'seed': 42, 'nthread': -1}
    p_cat = {'iterations': 400, 'learning_rate': 0.1333, 'depth': 7, 'loss_function': 'MultiQuantile:alpha=0.1,0.5,0.9', 'verbose': 0, 'random_seed': 42, 'thread_count': -1}

    overall_results = []

    print("[2/3] Training and Evaluating Architectures Across All Modalities...")
    
    for ds_name, df in datasets.items():
        print(f"\n  -> Processing Dataset: {ds_name}")
        
        df_aligned = df.loc[master_df.index]
        cols_to_drop = [target_col, 'ID', 'event_observed']
        X = df_aligned.drop(columns=[c for c in cols_to_drop if c in df_aligned.columns]).select_dtypes(include=[np.number])
        
        X_train, X_test = X.loc[idx_train], X.loc[idx_test]

        # --- 1. XGBoost Log-Transform ---
        print("      * Training XGBoost (Standard Regression)...")
        model_log = XGBRegressor(**p_xgb_log)
        model_log.fit(X_train, np.log1p(y_train), sample_weight=weights_train)
        preds_log = np.expm1(model_log.predict(X_test))
        metrics = compute_metrics(y_test, preds_log)
        overall_results.append({'Architecture': '1. XGB Log', 'Dataset': ds_name, **metrics})

        # --- 2. XGBoost AFT (Survival Analysis) ---
        print("      * Training XGBoost-AFT (Survival Analysis)...")
        dtrain = xgb.DMatrix(X_train, weight=weights_train)
        dtest = xgb.DMatrix(X_test)
        
        # FIX 1: Correct Survival Math via Censoring Flag
        y_lower = y_train.values
        y_upper = np.where(censor_train.values == 1, y_train.values, np.inf)
        
        dtrain.set_float_info('label_lower_bound', y_lower)
        dtrain.set_float_info('label_upper_bound', y_upper)
        
        bst_aft = xgb.train(p_xgb_aft, dtrain, num_boost_round=200)
        preds_aft = bst_aft.predict(dtest)
        metrics = compute_metrics(y_test, preds_aft)
        overall_results.append({'Architecture': '2. XGB AFT', 'Dataset': ds_name, **metrics})

        # --- 3. CatBoost Quantile ---
        print("      * Training CatBoost Quantile (Risk Bounds)...")
        model_cat = CatBoostRegressor(**p_cat)
        model_cat.fit(X_train, y_train, sample_weight=weights_train, eval_set=(X_test, y_test), early_stopping_rounds=30)
        preds_cat = model_cat.predict(X_test)[:, 1] # Extract the Median (50th percentile) prediction
        metrics = compute_metrics(y_test, preds_cat)
        overall_results.append({'Architecture': '3. Cat Quantile', 'Dataset': ds_name, **metrics})

    print("\n[3/3] Generating Final Reports...")
    Path("reports/phase0").mkdir(parents=True, exist_ok=True)
    
    df_overall = pd.DataFrame(overall_results)
    df_overall.to_csv("reports/phase0/overall_modality_ablation_v2.csv", index=False)
    
    print("\n" + "="*85)
    print("  📊 FULL METRICS: ALL ARCHITECTURES BY MODALITY")
    print("="*85)
    
    display_overall = df_overall.set_index(['Architecture', 'Dataset'])
    display_overall = display_overall[['MedAE (Mins)', 'MAE (Mins)', 'RMSE', 'MAPE (%)']]
    print(display_overall.round(2).to_string())

if __name__ == "__main__":
    evaluate_models()