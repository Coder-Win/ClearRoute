import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from pathlib import Path

# Import the newly perfected MoE architecture
from moe_architecture import TreeCompatibleMoE

def generate_merged_outputs():
    print("[1/4] Loading the best Multimodal Gated Fusion Dataset...")
    data_path = Path("data/processed/us_accidents_fused_multimodal_v2.csv")
    df = pd.read_csv(data_path)
    
    target_col = 'reported_duration_minutes'
    
    valid_mask = (df[target_col] > 0) & (df[target_col] <= 1440)
    df = df[valid_mask]
    
    y = df[target_col]
    censor = df['event_observed'] if 'event_observed' in df.columns else pd.Series(np.ones(len(df)), index=df.index)
    
    # PRESERVING THE EXACT TEST SET FOR FAIR COMPARISON
    # We split the original 80% train set to carve out a dedicated Calibration set
    idx_train_val, idx_test = train_test_split(df.index, test_size=0.2, random_state=42)
    idx_train, idx_calib = train_test_split(idx_train_val, test_size=0.15, random_state=42) 
    
    df_train = df.loc[idx_train]
    df_calib = df.loc[idx_calib].copy()
    df_test = df.loc[idx_test].copy()
    
    # Tag the splits so the downstream conformal calibration script knows which is which
    df_calib['Data_Split'] = 'Calibration'
    df_test['Data_Split'] = 'Test'
    
    y_train = y.loc[idx_train]
    censor_train = censor.loc[idx_train]
    
    # Prepare Features (drop targets and split tags)
    cols_to_drop = [target_col, 'ID', 'event_observed', 'Data_Split']
    X_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns]).select_dtypes(include=[np.number])
    X_calib = df_calib.drop(columns=[c for c in cols_to_drop if c in df_calib.columns]).select_dtypes(include=[np.number])
    X_test = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns]).select_dtypes(include=[np.number])
    
    print("[2/4] Calculating Sample Weights...")
    bins = np.digitize(y_train, bins=[30, 60, 120, 240, 1440])
    weights_train = compute_sample_weight(class_weight='balanced', y=bins)
    
    print("[3/4] Training the TreeCompatibleMoE Architecture...")
    moe = TreeCompatibleMoE(n_experts=3)
    moe.fit(X_train, y_train, censor_train, sample_weight=weights_train)
    
    print("[4/4] Generating Unified Predictions (Calibration & Test Sets)...")
    # Generate point predictions strictly from the MoE
    df_calib['MoE_Expected'] = moe.predict(X_calib)
    df_test['MoE_Expected'] = moe.predict(X_test)
    
    # Merge Calibration and Test sets into a single payload for the Conformal step
    df_export = pd.concat([df_calib, df_test])
    
    # Reorder columns so the Targets and Predictions are right at the front
    pred_cols = ['Data_Split', 'reported_duration_minutes', 'MoE_Expected']
    if 'ID' in df_export.columns:
        pred_cols.insert(0, 'ID')
        
    other_cols = [c for c in df_export.columns if c not in pred_cols]
    df_export = df_export[pred_cols + other_cols]
    
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "merged_duration_outputs.csv"
    
    df_export.to_csv(output_path, index=False)
    
    print(f"\n✅ Success! Unified MoE outputs saved to: {output_path}")
    print(f"Calibration crashes scored: {len(df_calib)}")
    print(f"Test crashes scored: {len(df_test)}")
    print("\nSample of the unified predictions (First 5 test crashes):")
    print(df_export[df_export['Data_Split'] == 'Test'][pred_cols].round(2).head())

if __name__ == "__main__":
    generate_merged_outputs()