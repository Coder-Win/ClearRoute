import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from catboost import CatBoostRegressor
from pathlib import Path

def generate_merged_outputs():
    print("[1/4] Loading the best Multimodal Gated Fusion Dataset...")
    # We use the fused dataset as the official data payload for Box 6
    data_path = Path("data/processed/us_accidents_fused_multimodal_v2.csv")
    df = pd.read_csv(data_path)
    
    target_col = 'reported_duration_minutes'
    
    # Filter valid durations (ensuring no corrupted negative/zero values)
    valid_mask = (df[target_col] > 0) & (df[target_col] <= 1440)
    df = df[valid_mask]
    
    y = df[target_col]
    censor = df['event_observed'] if 'event_observed' in df.columns else pd.Series(np.ones(len(df)), index=df.index)
    
    # Standard Train/Test split (random_state=42 ensures we test on the exact same crashes as before)
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42)
    
    # We only need to save predictions for the unseen TEST set for Box 6 analysis
    df_train = df.loc[idx_train]
    df_test = df.loc[idx_test].copy() 
    
    y_train = y.loc[idx_train]
    censor_train = censor.loc[idx_train]
    
    # Prepare Features (drop targets and identifiers)
    cols_to_drop = [target_col, 'ID', 'event_observed']
    X_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns]).select_dtypes(include=[np.number])
    X_test = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns]).select_dtypes(include=[np.number])
    
    print("[2/4] Calculating Sample Weights...")
    bins = np.digitize(y_train, bins=[30, 60, 120, 240, 1440])
    weights_train = compute_sample_weight(class_weight='balanced', y=bins)
    
    # --- Model 1: CatBoost Quantile ---
    print("[3/4] Training CatBoost Quantile (Extracting Q10, Q50, Q90)...")
    p_cat = {'iterations': 400, 'learning_rate': 0.1333, 'depth': 7, 
             'loss_function': 'MultiQuantile:alpha=0.1,0.5,0.9', 
             'verbose': 0, 'random_seed': 42, 'thread_count': -1}
    model_cat = CatBoostRegressor(**p_cat)
    model_cat.fit(X_train, np.log1p(y_train), sample_weight=weights_train)
    
    # Reverse log-transform
    preds_cat_raw = model_cat.predict(X_test)
    preds_cat_real = np.expm1(preds_cat_raw)
    
    df_test['CatBoost_Q10'] = preds_cat_real[:, 0]
    df_test['CatBoost_Q50'] = preds_cat_real[:, 1]
    df_test['CatBoost_Q90'] = preds_cat_real[:, 2]
    
    # --- Model 2: XGBoost AFT ---
    print("[4/4] Training XGBoost-AFT (Extracting Expected Survival)...")
    p_xgb_aft = {'learning_rate': 0.0153, 'max_depth': 10, 'objective': 'survival:aft', 
                 'eval_metric': 'aft-nloglik', 'tree_method': 'hist', 'seed': 42, 'nthread': -1}
    
    dtrain = xgb.DMatrix(X_train, weight=weights_train)
    dtest = xgb.DMatrix(X_test)
    
    y_lower = y_train.values
    y_upper = np.where(censor_train.values == 1, y_train.values, np.inf)
    dtrain.set_float_info('label_lower_bound', y_lower)
    dtrain.set_float_info('label_upper_bound', y_upper)
    
    bst_aft = xgb.train(p_xgb_aft, dtrain, num_boost_round=200)
    
    preds_aft_raw = bst_aft.predict(dtest)
    # Apply the 1440 reality check clip to prevent exponential explosion
    df_test['XGB_AFT_Expected'] = np.clip(preds_aft_raw, a_min=1.0, a_max=1440.0)
    
    # --- Save the Merged Payload ---
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "merged_duration_outputs.csv"
    
    # Reorder columns so the Target and Predictions are right at the front
    pred_cols = ['reported_duration_minutes', 'CatBoost_Q10', 'CatBoost_Q50', 'CatBoost_Q90', 'XGB_AFT_Expected']
    other_cols = [c for c in df_test.columns if c not in pred_cols]
    df_test = df_test[pred_cols + other_cols]
    
    df_test.to_csv(output_path, index=False)
    print(f"\n✅ Success! Merged outputs saved to: {output_path}")
    print(f"Total test crashes scored: {len(df_test)}")
    print("\nSample of the merged predictions (First 5 crashes):")
    print(df_test[pred_cols].round(2).head())

if __name__ == "__main__":
    generate_merged_outputs()