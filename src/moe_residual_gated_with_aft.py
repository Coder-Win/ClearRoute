import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import median_absolute_error, mean_absolute_error, mean_squared_error
import xgboost as xgb
from catboost import CatBoostRegressor

FUSED_TABLE_PATH = Path("data/processed/fused_multimodal_table.parquet")
ABLATION_CSV_PATH = Path("reports/phase0/overall_modality_ablation_v2.csv")
ABLAT_COMP_CSV_PATH = Path("reports/phase0/ablation_metrics_comprehensive.csv")
FINAL_PREDS_PATH = Path("reports/phase1/final_gated_predictions.csv")

ABLATION_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
FINAL_PREDS_PATH.parent.mkdir(parents=True, exist_ok=True)


def clean_and_flatten_features(df: pd.DataFrame, cols: list) -> np.ndarray:
    """
    Bulletproof flattening: drops metadata columns, unpacks embedding arrays, 
    and forces strict numeric conversion.
    """
    protected_meta = {"ID", "duration", "raw_duration_mins", "censor_flag", "split", "Description", "Start_Time", "End_Time"}
    valid_cols = [c for c in cols if c not in protected_meta]
    
    sub_df = df[valid_cols]
    arrays = []
    flat_cols = []
    
    for c in valid_cols:
        if c not in sub_df.columns:
            continue
        sample_val = sub_df[c].iloc[0] if len(sub_df) > 0 else 0.0
        
        # Check if column is a nested list/ndarray (embedding vector)
        if isinstance(sample_val, (list, np.ndarray)) and len(sample_val) > 1:
            expanded = pd.DataFrame(sub_df[c].tolist(), index=sub_df.index)
            expanded.columns = [f"{c}_sub_{i}" for i in range(expanded.shape[1])]
            arrays.append(expanded)
        else:
            # Ensure column can be converted to float; otherwise ignore string columns
            series_num = pd.to_numeric(sub_df[c], errors="coerce")
            if series_num.notna().sum() > 0:
                flat_cols.append(c)
                
    if arrays:
        flat_part = sub_df[flat_cols] if flat_cols else pd.DataFrame(index=sub_df.index)
        res_df = pd.concat([flat_part] + arrays, axis=1)
    else:
        res_df = sub_df[flat_cols] if flat_cols else sub_df
        
    res_mat = res_df.apply(pd.to_numeric, errors="coerce").values
    return np.nan_to_num(res_mat, nan=0.0, posinf=0.0, neginf=0.0)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, 1.0, 1440.0)
    medae = median_absolute_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100.0
    return {
        "MedAE (Mins)": round(float(medae), 2),
        "MAE (Mins)": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE (%)": round(float(mape), 2)
    }


def make_aft_dmatrix(X: np.ndarray, y: np.ndarray = None, censor: np.ndarray = None) -> xgb.DMatrix:
    dmat = xgb.DMatrix(X)
    if y is not None and censor is not None:
        y_lower = np.maximum(y, 1.0)
        y_upper = np.where(censor == 1, np.inf, y_lower)
        dmat.set_float_info("label_lower_bound", y_lower)
        dmat.set_float_info("label_upper_bound", y_upper)
    return dmat


def train_xgb_log(X_tr, y_tr, X_te):
    model = xgb.XGBRegressor(
        n_estimators=120, max_depth=5, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    model.fit(X_tr, np.log1p(y_tr))
    pred_log = model.predict(X_te)
    sigma2 = np.var(np.log1p(y_tr) - model.predict(X_tr))
    return np.expm1(pred_log + 0.5 * sigma2)


def train_xgb_aft(X_tr, y_tr, c_tr, X_te):
    dtrain = make_aft_dmatrix(X_tr, y_tr, c_tr)
    dtest = make_aft_dmatrix(X_te)
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.15,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "nthread": -1
    }
    bst = xgb.train(params, dtrain, num_boost_round=140, verbose_eval=False)
    return bst.predict(dtest)


def train_catboost_quantile(X_tr, y_tr, X_te):
    cb_q50 = CatBoostRegressor(
        iterations=250, depth=5, learning_rate=0.06,
        loss_function="Quantile:alpha=0.5", random_seed=42, verbose=0
    )
    cb_q10 = CatBoostRegressor(
        iterations=200, depth=5, learning_rate=0.06,
        loss_function="Quantile:alpha=0.1", random_seed=42, verbose=0
    )
    cb_q90 = CatBoostRegressor(
        iterations=200, depth=5, learning_rate=0.06,
        loss_function="Quantile:alpha=0.9", random_seed=42, verbose=0
    )
    cb_q50.fit(X_tr, y_tr)
    cb_q10.fit(X_tr, y_tr)
    cb_q90.fit(X_tr, y_tr)
    
    q50_pred = cb_q50.predict(X_te)
    q10_pred = cb_q10.predict(X_te)
    q90_pred = cb_q90.predict(X_te)
    blended = 0.65 * q50_pred + 0.35 * (0.5 * (q10_pred + q90_pred))
    return blended, q10_pred, q50_pred, q90_pred


def run_5fold_oof_experts(df_tr, df_val, df_te, struct_cols, sem_cols, emb_cols):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_tr = df_tr["duration"].values
    c_tr = df_tr["censor_flag"].values

    oof_tr = np.zeros((len(df_tr), 3))
    oof_val = np.zeros((len(df_val), 3))
    oof_te = np.zeros((len(df_te), 3))

    X_emb_tr = clean_and_flatten_features(df_tr, emb_cols)
    X_emb_val = clean_and_flatten_features(df_val, emb_cols)
    X_emb_te = clean_and_flatten_features(df_te, emb_cols)

    # Expert 0 (Structural) & Expert 2 (Semantic) via XGBoost-AFT
    for exp_idx, cols in [(0, struct_cols), (2, sem_cols)]:
        val_folds = np.zeros((len(df_val), 5))
        te_folds = np.zeros((len(df_te), 5))
        X_exp_tr = clean_and_flatten_features(df_tr, cols)
        X_exp_val = clean_and_flatten_features(df_val, cols)
        X_exp_te = clean_and_flatten_features(df_te, cols)

        for fold, (t_idx, v_idx) in enumerate(kf.split(X_exp_tr)):
            dtr = make_aft_dmatrix(X_exp_tr[t_idx], y_tr[t_idx], c_tr[t_idx])
            dva = make_aft_dmatrix(X_exp_tr[v_idx])
            dval_full = make_aft_dmatrix(X_exp_val)
            dte_full = make_aft_dmatrix(X_exp_te)

            params = {
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                "aft_loss_distribution": "logistic",
                "aft_loss_distribution_scale": 0.90,
                "learning_rate": 0.05,
                "max_depth": 6 if exp_idx == 0 else 4,
                "reg_lambda": 3.0,
                "seed": 42 + fold,
                "nthread": -1
            }
            bst = xgb.train(params, dtr, num_boost_round=160, verbose_eval=False)
            oof_tr[v_idx, exp_idx] = np.log1p(np.clip(bst.predict(dva), 1.0, 1440.0))
            val_folds[:, fold] = np.log1p(np.clip(bst.predict(dval_full), 1.0, 1440.0))
            te_folds[:, fold] = np.log1p(np.clip(bst.predict(dte_full), 1.0, 1440.0))

        oof_val[:, exp_idx] = val_folds.mean(axis=1)
        oof_te[:, exp_idx] = te_folds.mean(axis=1)

    # Expert 1 (Embedding vectors) via L2-Regularized Ridge in log-space
    val_folds_emb = np.zeros((len(df_val), 5))
    te_folds_emb = np.zeros((len(df_te), 5))
    for fold, (t_idx, v_idx) in enumerate(kf.split(X_emb_tr)):
        ridge = Ridge(alpha=50.0, random_state=42 + fold)
        ridge.fit(X_emb_tr[t_idx], np.log1p(y_tr[t_idx]))
        oof_tr[v_idx, 1] = ridge.predict(X_emb_tr[v_idx])
        val_folds_emb[:, fold] = ridge.predict(X_emb_val)
        te_folds_emb[:, fold] = ridge.predict(X_emb_te)

    oof_val[:, 1] = val_folds_emb.mean(axis=1)
    oof_te[:, 1] = te_folds_emb.mean(axis=1)

    return oof_tr, oof_val, oof_te


def train_soft_moe_aft(
    mode: str,
    oof_tr: np.ndarray,
    oof_val: np.ndarray,
    oof_te: np.ndarray,
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    df_te: pd.DataFrame,
    struct_cols: list,
    sem_cols: list,
    gate_cols: list
) -> np.ndarray:
    y_tr = df_tr["duration"].values
    c_tr = df_tr["censor_flag"].values
    y_val = df_val["duration"].values
    c_val = df_val["censor_flag"].values  # <-- ADDED THIS LINE TO FIX THE ERROR

    if mode == "1. Struct Only":
        active_indices = [0]
        use_gating = False
    elif mode == "2. Struct+Sem":
        active_indices = [0, 2]
        use_gating = False
    elif mode == "3. Struct+Emb":
        active_indices = [0, 1]
        use_gating = False
    elif mode == "4. Dumb Join":
        active_indices = [0, 1, 2]
        use_gating = False
    else:
        active_indices = [0, 1, 2]
        use_gating = True

    sub_tr = oof_tr[:, active_indices]
    sub_val = oof_val[:, active_indices]
    sub_te = oof_te[:, active_indices]

    if not use_gating:
        dmeta_tr = make_aft_dmatrix(sub_tr, y_tr, c_tr)
        dmeta_te = make_aft_dmatrix(sub_te)
        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.05,
            "learning_rate": 0.05,
            "max_depth": 3,
            "seed": 42
        }
        bst = xgb.train(params, dmeta_tr, num_boost_round=100, verbose_eval=False)
        return bst.predict(dmeta_te)

    # Gated Fusion
    sem_flat = clean_and_flatten_features(df_tr, sem_cols)
    gate_flat = clean_and_flatten_features(df_tr, gate_cols)
    
    def build_gated_features(oof_mat, df_part):
        q_raw = clean_and_flatten_features(df_part, gate_cols)
        q_exp = np.exp(q_raw * 2.5)
        weights = q_exp / (np.sum(q_exp, axis=1, keepdims=True) + 1e-8)
        gated_blend = np.sum(oof_mat * weights, axis=1, keepdims=True)
        expert_spread = np.std(oof_mat, axis=1, keepdims=True)
        
        sem_f = clean_and_flatten_features(df_part, sem_cols)
        return np.hstack([oof_mat, weights, gated_blend, expert_spread, q_raw, sem_f[:, :5]])

    X_meta_tr = build_gated_features(sub_tr, df_tr)
    X_meta_val = build_gated_features(sub_val, df_val)
    X_meta_te = build_gated_features(sub_te, df_te)

    dmeta_tr = make_aft_dmatrix(X_meta_tr, y_tr, c_tr)
    dmeta_val = make_aft_dmatrix(X_meta_val, y_val, c_val)
    dmeta_te = make_aft_dmatrix(X_meta_te)

    meta_params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "logistic",
        "aft_loss_distribution_scale": 0.78,
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 1.5,
        "reg_lambda": 5.0,
        "seed": 42,
        "nthread": -1
    }
    bst_aft = xgb.train(
        meta_params, dmeta_tr, num_boost_round=350,
        evals=[(dmeta_val, "val")], early_stopping_rounds=25, verbose_eval=False
    )
    pred_aft_val = bst_aft.predict(dmeta_val)
    pred_aft_te = bst_aft.predict(dmeta_te)

    cb_l1 = CatBoostRegressor(
        iterations=400, depth=6, learning_rate=0.04,
        loss_function="MAE", random_seed=42, verbose=0
    )
    cb_l1.fit(X_meta_tr, np.log1p(y_tr), eval_set=(X_meta_val, np.log1p(y_val)), early_stopping_rounds=30)
    pred_l1_val = np.expm1(cb_l1.predict(X_meta_val))
    pred_l1_te = np.expm1(cb_l1.predict(X_meta_te))

    best_w, best_score = 0.5, float("inf")
    for w in np.linspace(0.0, 1.0, 41):
        blend_val = w * pred_aft_val + (1.0 - w) * pred_l1_val
        score = median_absolute_error(y_val, blend_val) + 0.5 * mean_absolute_error(y_val, blend_val)
        if score < best_score:
            best_score = score
            best_w = w

    final_gated_pred = best_w * pred_aft_te + (1.0 - best_w) * pred_l1_te
    return final_gated_pred


def main():
    if not FUSED_TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing {FUSED_TABLE_PATH}. Run Step 2 first!")

    print("[Stage 4] Loading Fused Multimodal Table...")
    df = pd.read_parquet(FUSED_TABLE_PATH)

    struct_cols = [c for c in df.columns if c.startswith("struct_")]
    sem_cols = [c for c in df.columns if c.startswith("sem_")]
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    gate_cols = ["gate_q_tabular", "gate_q_text_len", "gate_q_semantic_density"]

    df_tr = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"].isin(["val", "cal"])].reset_index(drop=True)
    df_te = df[df["split"] == "test"].reset_index(drop=True)

    y_tr, c_tr = df_tr["duration"].values, df_tr["censor_flag"].values
    y_te, c_te = df_te["duration"].values, df_te["censor_flag"].values

    print(f"[Stage 4] Split sizes -> Train: {len(df_tr):,}, Val/Cal: {len(df_val):,}, Test: {len(df_te):,}")
    print("[Stage 4] Training 5-Fold Out-Of-Fold (OOF) Base Experts...")
    oof_tr, oof_val, oof_te = run_5fold_oof_experts(df_tr, df_val, df_te, struct_cols, sem_cols, emb_cols)

    datasets = [
        ("1. Struct Only", struct_cols),
        ("2. Struct+Sem", struct_cols + sem_cols),
        ("3. Struct+Emb", struct_cols + emb_cols),
        ("4. Dumb Join", struct_cols + sem_cols + emb_cols),
        ("5. Gated Fusion", struct_cols + sem_cols + emb_cols)
    ]

    results = []
    cached_dumb_join_baselines = {}
    final_q10, final_q50, final_q90, final_moe_preds = None, None, None, None

    for ds_name, cols in datasets:
        print(f"[Stage 4] Evaluating Dataset tier: {ds_name}...")
        X_tr = clean_and_flatten_features(df_tr, cols)
        X_te = clean_and_flatten_features(df_te, cols)

        if ds_name == "5. Gated Fusion" and cached_dumb_join_baselines:
            pred_log = cached_dumb_join_baselines["1. XGB Log"]
            pred_aft = cached_dumb_join_baselines["2. XGB AFT"]
            pred_cat = cached_dumb_join_baselines["3. Cat Quantile"]
        else:
            pred_log = train_xgb_log(X_tr, y_tr, X_te)
            pred_aft = train_xgb_aft(X_tr, y_tr, c_tr, X_te)
            pred_cat, q10, q50, q90 = train_catboost_quantile(X_tr, y_tr, X_te)
            if ds_name == "4. Dumb Join":
                cached_dumb_join_baselines = {
                    "1. XGB Log": pred_log,
                    "2. XGB AFT": pred_aft,
                    "3. Cat Quantile": pred_cat
                }
                final_q10, final_q50, final_q90 = q10, q50, q90

        pred_moe = train_soft_moe_aft(
            ds_name, oof_tr, oof_val, oof_te,
            df_tr, df_val, df_te,
            struct_cols, sem_cols, gate_cols
        )

        if ds_name == "5. Gated Fusion":
            final_moe_preds = pred_moe

        for arch_name, preds in [
            ("1. XGB Log", pred_log),
            ("2. XGB AFT", pred_aft),
            ("3. Cat Quantile", pred_cat),
            ("4. Soft MoE AFT", pred_moe)
        ]:
            m = calc_metrics(y_te, preds)
            results.append({
                "Architecture": arch_name,
                "Dataset": ds_name,
                **m
            })

    df_res = pd.DataFrame(results)
    df_res.to_csv(ABLATION_CSV_PATH, index=False)
    df_res.to_csv(ABLAT_COMP_CSV_PATH, index=False)

    df_preds = pd.DataFrame({
        "ID": df_te["ID"].values,
        "actual_duration": y_te,
        "censor_flag": c_te,
        "pred_soft_moe_aft": final_moe_preds,
        "pred_q10": final_q10,
        "pred_q50": final_q50,
        "pred_q90": final_q90,
        "gate_q_tabular": df_te["gate_q_tabular"].values,
        "gate_q_text_len": df_te["gate_q_text_len"].values,
        "gate_q_semantic_density": df_te["gate_q_semantic_density"].values
    })
    df_preds.to_csv(FINAL_PREDS_PATH, index=False)

    print("\n" + "=" * 82)
    print(" 📊 FULL METRICS: ALL ARCHITECTURES BY MODALITY (STRICT HIERARCHY ENFORCED)")
    print("=" * 82)
    print(f"{'Architecture':<18} {'Dataset':<17} {'MedAE (Mins)':>13} {'MAE (Mins)':>12} {'RMSE':>10} {'MAPE (%)':>10}")
    print("-" * 82)
    for _, row in df_res.iterrows():
        print(
            f"{row['Architecture']:<18} {row['Dataset']:<17} "
            f"{row['MedAE (Mins)']:>13.2f} {row['MAE (Mins)']:>12.2f} "
            f"{row['RMSE']:>10.2f} {row['MAPE (%)']:>10.2f}"
        )
    print("=" * 82 + "\n")


if __name__ == "__main__":
    main()