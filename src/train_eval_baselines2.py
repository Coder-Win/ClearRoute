import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.cluster import KMeans
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor
from pathlib import Path
from moe_architecture import TreeCompatibleMoE


class TreeCompatibleMoE:
    def __init__(self, n_experts=3):
        self.n_experts = n_experts
        
        # Gating Network utilizing multi-core CPU ('hist' algorithm is highly optimized for CPU)
        self.gating_network = XGBClassifier(
            n_estimators=150, 
            learning_rate=0.05, 
            max_depth=6, 
            objective='multi:softprob', 
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        )
        self.experts = []
        
        # Expert Hyperparameters utilizing multi-core CPU
        self.expert_params = {
            'learning_rate': 0.015, 
            'max_depth': 8, 
            'objective': 'survival:aft', 
            'eval_metric': 'aft-nloglik', 
            'tree_method': 'hist',
            'seed': 42,
            'nthread': -1
        }

    def fit(self, X_train, y_train, censor_train, sample_weight=None):
        y_log = np.log1p(y_train).values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.n_experts, random_state=42, n_init=10)
        raw_labels = kmeans.fit_predict(y_log)
        
        # Sort clusters so Expert 0 handles the shortest durations and Expert N handles the longest
        sorted_centers = np.argsort(kmeans.cluster_centers_.flatten())
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_centers)}
        expert_labels = np.vectorize(label_mapping.get)(raw_labels)
        
        # Train the router 
        self.gating_network.fit(X_train, expert_labels, sample_weight=sample_weight)

        for k in range(self.n_experts):
            mask = (expert_labels == k)
            X_k = X_train[mask]
            y_k = y_train[mask]
            censor_k = censor_train[mask]
            
            weight_k = sample_weight[mask] if sample_weight is not None else None

            y_lower = y_k.values
            y_upper = np.where(censor_k.values == 1, y_k.values, np.inf)

            dtrain_k = xgb.DMatrix(X_k, weight=weight_k)
            dtrain_k.set_float_info('label_lower_bound', y_lower)
            dtrain_k.set_float_info('label_upper_bound', y_upper)

            expert_model = xgb.train(self.expert_params, dtrain_k, num_boost_round=200)
            self.experts.append(expert_model)

    def predict(self, X_test):
        # 1. Router generates probabilities
        gating_probs = self.gating_network.predict_proba(X_test)
        expert_preds = np.zeros((X_test.shape[0], self.n_experts))
        dtest = xgb.DMatrix(X_test)
        
        for k, expert in enumerate(self.experts):
            # EXTREME TAIL CAP: Prevent the raw AFT from hallucinating infinite values.
            # 43200 minutes = 1 month (physical maximum possible blockage)
            raw_pred = expert.predict(dtest)
            expert_preds[:, k] = np.clip(raw_pred, 1.0, 43200.0)

        # LOG-SPACE BLENDING: Mix the predictions in logarithmic space, then exponentiate.
        # This acts as a Geometric Mean rather than an Arithmetic Mean, stabilizing RMSE.
        log_preds = np.log1p(expert_preds)
        final_log_predictions = np.sum(gating_probs * log_preds, axis=1)
        final_predictions = np.expm1(final_log_predictions)
        
        return final_predictions
    
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
    
    datasets = {
        '1. Struct Only': pd.read_csv(processed_dir / "us_accidents_struct_only_v2.csv"),
        '2. Struct+Sem': pd.read_csv(processed_dir / "us_accidents_struct_sem_v2.csv"),
        '3. Struct+Emb': pd.read_csv(processed_dir / "us_accidents_struct_emb_v2.csv"),
        '4. Dumb Join': pd.read_csv(processed_dir / "us_accidents_dumb_join_v2.csv"),
        '5. Gated Fusion': pd.read_csv(processed_dir / "us_accidents_fused_multimodal_v2.csv")
    }
    
    target_col = 'reported_duration_minutes'
    
    master_df = datasets['5. Gated Fusion']
    valid_mask = (master_df[target_col] > 0) & (master_df[target_col] <= 1440)
    master_df = master_df[valid_mask]
    
    y_master = master_df[target_col]
    
    censor_master = master_df['event_observed'] if 'event_observed' in master_df.columns else pd.Series(np.ones(len(master_df)), index=master_df.index)
    
    idx_train, idx_test = train_test_split(master_df.index, test_size=0.2, random_state=42)
    y_train, y_test = y_master.loc[idx_train], y_master.loc[idx_test]
    censor_train, censor_test = censor_master.loc[idx_train], censor_master.loc[idx_test]

    print("      -> Computing sample weights to balance extreme tail durations...")
    bins = np.digitize(y_train, bins=[30, 60, 120, 240, 1440])
    weights_train = compute_sample_weight(class_weight='balanced', y=bins)

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
        preds_cat = model_cat.predict(X_test)[:, 1] 
        metrics = compute_metrics(y_test, preds_cat)
        overall_results.append({'Architecture': '3. Cat Quantile', 'Dataset': ds_name, **metrics})

        # --- 4. Soft Mixture of Experts (MoE) AFT ---
        print("      * Training Soft MoE AFT (Decoupled Survival Analysis)...")
        moe_model = TreeCompatibleMoE(n_experts=3)
        moe_model.fit(X_train, y_train, censor_train, sample_weight=weights_train)
        preds_moe = moe_model.predict(X_test)
        metrics = compute_metrics(y_test, preds_moe)
        overall_results.append({'Architecture': '4. Soft MoE AFT', 'Dataset': ds_name, **metrics})

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