import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from pathlib import Path

def run_treeshap():
    print("================================================================")
    print("  STEP 2: TREESHAP FACTOR ANALYSIS (Gao et al., 2026 Upgrade)   ")
    print("================================================================")
    
    data_path = Path("data/processed/us_accidents_fused_multimodal_v2.csv")
    df = pd.read_csv(data_path)
    
    target_col = 'reported_duration_minutes'
    valid_mask = (df[target_col] > 0) & (df[target_col] <= 1440)
    df = df[valid_mask]
    
    y = df[target_col]
    censor = df['event_observed'] if 'event_observed' in df.columns else pd.Series(np.ones(len(df)), index=df.index)
    
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42)
    
    cols_to_drop = [target_col, 'ID', 'event_observed']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).select_dtypes(include=[np.number])
    
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train = y.loc[idx_train]
    censor_train = censor.loc[idx_train]

    print("[1/3] Fitting XGBoost-AFT with Sample Weights...")
    bins = np.digitize(y_train, bins=[30, 60, 120, 240, 1440])
    weights_train = compute_sample_weight(class_weight='balanced', y=bins)
    
    dtrain = xgb.DMatrix(X_train, weight=weights_train)
    y_lower = y_train.values
    y_upper = np.where(censor_train.values == 1, y_train.values, np.inf)
    dtrain.set_float_info('label_lower_bound', y_lower)
    dtrain.set_float_info('label_upper_bound', y_upper)
    
    p_xgb_aft = {
        'learning_rate': 0.0153, 'max_depth': 8, 'objective': 'survival:aft',
        'eval_metric': 'aft-nloglik', 'tree_method': 'hist', 'seed': 42, 'nthread': -1
    }
    bst_aft = xgb.train(p_xgb_aft, dtrain, num_boost_round=150)

    print("[2/3] Extracting TreeSHAP Explanation Values...")
    explainer = shap.TreeExplainer(bst_aft)
    shap_values = explainer.shap_values(X_test)

    print("[3/3] Saving Feature Attribution Visualizations...")
    reports_dir = Path("reports/phase1")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary Beeswarm Plot (shows directional impact: red vs blue)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("TreeSHAP Local Attribution - Multimodal XGBoost-AFT", fontsize=13)
    plt.tight_layout()
    plt.savefig(reports_dir / "treeshap_beeswarm.png", dpi=300)
    plt.close()

    # 2. Bar Plot (Mean Absolute SHAP value ranking)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("TreeSHAP Global Feature Importance Ranking", fontsize=13)
    plt.tight_layout()
    plt.savefig(reports_dir / "treeshap_feature_importance.png", dpi=300)
    plt.close()

    # Save top features table
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_features = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean_Absolute_SHAP_Impact': mean_shap
    }).sort_values(by='Mean_Absolute_SHAP_Impact', ascending=False)
    top_features.to_csv(reports_dir / "treeshap_importance_rankings.csv", index=False)

    print(f"✅ Step 2 complete. SHAP plots saved to {reports_dir}")

if __name__ == "__main__":
    run_treeshap()