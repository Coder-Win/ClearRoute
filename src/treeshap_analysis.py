import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from pathlib import Path

from moe_architecture import TreeCompatibleMoE

def run_treeshap():
    print("================================================================")
    print("  STEP 2: FULL MICRO-ATTRIBUTION (ALL EXPERTS)                  ")
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

    print("[1/3] Fitting Soft MoE Architecture...")
    bins = np.digitize(y_train, bins=[30, 60, 120, 240, 1440])
    weights_train = compute_sample_weight(class_weight='balanced', y=bins)
    
    moe = TreeCompatibleMoE(n_experts=3)
    moe.fit(X_train, y_train, censor_train, sample_weight=weights_train)

    print("[2/3] Extracting SHAP Values for Router and ALL 3 Experts...")
    X_test_sample = shap.sample(X_test, 1000) 
    
    # 1. The Router
    router_explainer = shap.TreeExplainer(moe.gating_network)
    router_shap_values = router_explainer.shap_values(X_test_sample)

    # 2. Expert 0 (Routine)
    expert0_explainer = shap.TreeExplainer(moe.experts[0])
    expert0_shap_values = expert0_explainer.shap_values(X_test_sample)

    # 3. Expert 1 (Moderate)
    expert1_explainer = shap.TreeExplainer(moe.experts[1])
    expert1_shap_values = expert1_explainer.shap_values(X_test_sample)

    # 4. Expert 2 (Severe)
    expert2_explainer = shap.TreeExplainer(moe.experts[2])
    expert2_shap_values = expert2_explainer.shap_values(X_test_sample)

    print("[3/3] Generating Raw Feature Attribution Plots...")
    reports_dir = Path("reports/phase1")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Handle multi-class Router shape
    if isinstance(router_shap_values, list):
        router_shap_severe = router_shap_values[2]
    elif len(router_shap_values.shape) == 3:
        router_shap_severe = router_shap_values[:, :, 2]
    else:
        router_shap_severe = router_shap_values

    # Plot 1: Router
    plt.figure(figsize=(12, 8))
    shap.summary_plot(router_shap_severe, X_test_sample, show=False)
    plt.title("Router Assignment Drivers (Probability of Severe Classification)")
    plt.tight_layout()
    plt.savefig(reports_dir / "treeshap_router_beeswarm.png", dpi=300)
    plt.close()

    # Plot 2: Expert 0
    plt.figure(figsize=(12, 8))
    shap.summary_plot(expert0_shap_values, X_test_sample, show=False)
    plt.title("Expert 0 Drivers (Routine Incidents)")
    plt.tight_layout()
    plt.savefig(reports_dir / "treeshap_expert0_beeswarm.png", dpi=300)
    plt.close()

    # Plot 3: Expert 1
    plt.figure(figsize=(12, 8))
    shap.summary_plot(expert1_shap_values, X_test_sample, show=False)
    plt.title("Expert 1 Drivers (Moderate Incidents)")
    plt.tight_layout()
    plt.savefig(reports_dir / "treeshap_expert1_beeswarm.png", dpi=300)
    plt.close()

    # Plot 4: Expert 2
    plt.figure(figsize=(12, 8))
    shap.summary_plot(expert2_shap_values, X_test_sample, show=False)
    plt.title("Expert 2 Drivers (Severe Tail Incidents)")
    plt.tight_layout()
    plt.savefig(reports_dir / "treeshap_expert2_beeswarm.png", dpi=300)
    plt.close()
    
    # Generate unified CSV comparing feature importance across all experts
    importance_df = pd.DataFrame({
        'Feature': X_test_sample.columns,
        'Expert0_Mean_Impact': np.abs(expert0_shap_values).mean(axis=0),
        'Expert1_Mean_Impact': np.abs(expert1_shap_values).mean(axis=0),
        'Expert2_Mean_Impact': np.abs(expert2_shap_values).mean(axis=0)
    })
    
    # Sort by overall average impact across all regimes
    importance_df['Overall_Mean_Impact'] = importance_df[['Expert0_Mean_Impact', 'Expert1_Mean_Impact', 'Expert2_Mean_Impact']].mean(axis=1)
    importance_df = importance_df.sort_values(by='Overall_Mean_Impact', ascending=False)
    importance_df.to_csv(reports_dir / "treeshap_expert_comparison_rankings.csv", index=False)

    print(f"✅ Step 2 complete. Beeswarm plots for all experts and unified CSV saved to {reports_dir}")

if __name__ == "__main__":
    run_treeshap()