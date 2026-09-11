""" import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def show_real_predictions():
    print("Loading 'Gated Fusion' dataset and training models (takes ~15 seconds)...")
    
    # Load the best multimodal dataset
    df = pd.read_csv("data/processed/us_accidents_fused_multimodal_v2.csv")
    target_col = 'reported_duration_minutes'
    
    # Filter valid durations
    valid_mask = (df[target_col] > 0) & (df[target_col] <= 1440)
    df = df[valid_mask]
    
    y = df[target_col]
    censor = df['event_observed'] if 'event_observed' in df.columns else np.ones(len(df))
    
    # Drop targets to create feature set
    cols_to_drop = [target_col, 'ID', 'event_observed']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).select_dtypes(include=[np.number])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    censor_train, censor_test = train_test_split(censor, test_size=0.2, random_state=42)

    # 1. Train XGBoost Log (Standard Point Estimate)
    model_log = XGBRegressor(n_estimators=200, learning_rate=0.0639, max_depth=8, n_jobs=-1, random_state=42)
    model_log.fit(X_train, np.log1p(y_train))

    # 2. Train XGBoost-AFT (Survival Expected Time)
    dtrain = xgb.DMatrix(X_train)
    y_lower = y_train.values
    y_upper = np.where(censor_train.values == 1, y_train.values, np.inf)
    dtrain.set_float_info('label_lower_bound', y_lower)
    dtrain.set_float_info('label_upper_bound', y_upper)
    
    model_aft = xgb.train({'learning_rate': 0.0153, 'max_depth': 10, 'objective': 'survival:aft', 'eval_metric': 'aft-nloglik', 'tree_method': 'hist'}, dtrain, num_boost_round=200)

    # 3. Train CatBoost Quantile (Risk Bounds)
    model_cat = CatBoostRegressor(iterations=400, learning_rate=0.1333, depth=7, loss_function='MultiQuantile:alpha=0.1,0.5,0.9', verbose=0, random_seed=42)
    model_cat.fit(X_train, y_train)

    print("\n" + "="*70)
    print(" 🚦 REAL-WORLD PREDICTION COMPARISONS ON 3 RANDOM CRASHES 🚦")
    print("="*70)
    
    # Pick 3 random crashes from the test set
    sample_X = X_test.sample(3, random_state=101)
    sample_y = y_test.loc[sample_X.index]
    
    # Get predictions
    preds_log = np.expm1(model_log.predict(sample_X))
    preds_aft = model_aft.predict(xgb.DMatrix(sample_X))
    preds_cat = model_cat.predict(sample_X)
    
    for i in range(3):
        actual = sample_y.iloc[i]
        p_log = preds_log[i]
        p_aft = preds_aft[i]
        q10, q50, q90 = preds_cat[i]
        
        print(f"\nCRASH EVENT #{i+1}:")
        print(f"  ▶ ACTUAL RECORDED CLEARANCE TIME: {actual:.1f} minutes")
        print("-" * 50)
        print(f"  [Model 1] XGBoost Log (Rigid Guess):      {p_log:.1f} minutes")
        print(f"  [Model 2] XGBoost-AFT (Survival Output):  {p_aft:.1f} minutes")
        print(f"  [Model 3] CatBoost Quantile (Safety Window):")
        print(f"            - Optimistic (Q10):             {q10:.1f} minutes")
        print(f"            - Expected (Q50):               {q50:.1f} minutes")
        print(f"            - Worst-Case (Q90):             {q90:.1f} minutes")
        
        if q90 - q10 > 120:
            print("            ⚠️ SYSTEM ALERT: Extremely wide uncertainty! Abstain from automated routing.")

if __name__ == "__main__":
    show_real_predictions() """



import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def show_real_predictions_with_curves():
    print("Loading 'Gated Fusion' dataset and training models (takes ~15 seconds)...")
    
    # Load the best multimodal dataset
    df = pd.read_csv("data/processed/us_accidents_fused_multimodal_v2.csv")
    target_col = 'reported_duration_minutes'
    
    # Filter valid durations
    valid_mask = (df[target_col] > 0) & (df[target_col] <= 1440)
    df = df[valid_mask]
    
    y = df[target_col]
    censor = df['event_observed'] if 'event_observed' in df.columns else np.ones(len(df))
    
    # Drop targets to create feature set
    cols_to_drop = [target_col, 'ID', 'event_observed']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).select_dtypes(include=[np.number])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    censor_train, censor_test = train_test_split(censor, test_size=0.2, random_state=42)

    # 1. Train XGBoost Log (Standard Point Estimate)
    model_log = XGBRegressor(n_estimators=200, learning_rate=0.0639, max_depth=8, n_jobs=-1, random_state=42)
    model_log.fit(X_train, np.log1p(y_train))

    # 2. Train XGBoost-AFT (Survival Expected Time)
    dtrain = xgb.DMatrix(X_train)
    y_lower = y_train.values
    y_upper = np.where(censor_train.values == 1, y_train.values, np.inf)
    dtrain.set_float_info('label_lower_bound', y_lower)
    dtrain.set_float_info('label_upper_bound', y_upper)
    
    model_aft = xgb.train({'learning_rate': 0.0153, 'max_depth': 10, 'objective': 'survival:aft', 'eval_metric': 'aft-nloglik', 'tree_method': 'hist'}, dtrain, num_boost_round=200)

    # 3. Train CatBoost Quantile (Risk Bounds)
    model_cat = CatBoostRegressor(iterations=400, learning_rate=0.1333, depth=7, loss_function='MultiQuantile:alpha=0.1,0.5,0.9', verbose=0, random_seed=42)
    model_cat.fit(X_train, y_train)

    print("\n" + "="*70)
    print(" 🚦 REAL-WORLD PREDICTION COMPARISONS ON 3 RANDOM CRASHES 🚦")
    print("="*70)
    
    # Pick 3 random crashes from the test set
    sample_X = X_test.sample(3, random_state=101)
    sample_y = y_test.loc[sample_X.index]
    
    # Get predictions
    preds_log = np.expm1(model_log.predict(sample_X))
    preds_aft = model_aft.predict(xgb.DMatrix(sample_X))
    preds_cat = model_cat.predict(sample_X)
    
    # Setup the Plotting Canvas
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('XGBoost-AFT Survival Curves & CatBoost Quantile Bounds', fontsize=16, fontweight='bold')
    
    # Time array for the X-axis of our plots (0 to 300 minutes)
    time_range = np.linspace(0.1, 300, 300)
    
    for i in range(3):
        idx = sample_X.index[i]
        
        # --- NEW: Extracting the actual text/ID data from the original dataframe ---
        actual = sample_y.iloc[i]
        crash_id = df.loc[idx, 'ID'] if 'ID' in df.columns else "Unknown"
        state = df.loc[idx, 'State'] if 'State' in df.columns else "Unknown"
        # -------------------------------------------------------------------------
        
        p_log = preds_log[i]
        p_aft = preds_aft[i]
        q10, q50, q90 = preds_cat[i]
        
        print(f"\nCRASH EVENT #{i+1} [ID: {crash_id} | State: {state}]")
        print(f"  ▶ ACTUAL RECORDED CLEARANCE TIME: {actual:.1f} minutes")
        print("-" * 50)
        print(f"  [Model 1] XGBoost Log (Rigid Guess):      {p_log:.1f} minutes")
        print(f"  [Model 2] XGBoost-AFT (Survival Output):  {p_aft:.1f} minutes")
        print(f"  [Model 3] CatBoost Quantile (Safety Window):")
        print(f"            - Optimistic (Q10):             {q10:.1f} minutes")
        print(f"            - Expected (Q50):               {q50:.1f} minutes")
        print(f"            - Worst-Case (Q90):             {q90:.1f} minutes")
        
        if q90 - q10 > 120:
            print("            ⚠️ SYSTEM ALERT: Extremely wide uncertainty! Abstain from automated routing.")

        # --- PLOTTING THE SURVIVAL CURVE ---
        ax = axes[i]
        
        # XGBoost AFT assumes a Log-Normal distribution. The prediction is the expected value.
        # We calculate the shape/scale to generate the mathematical survival curve P(T > t)
        scale_param = p_aft
        shape_param = 0.5 # Approximated standard deviation of log(T) for visual scaling
        
        # Calculate the Survival Function (1 - CDF)
        survival_probs = stats.lognorm.sf(time_range, s=shape_param, scale=scale_param)
        
        # Plot the main curve
        ax.plot(time_range, survival_probs * 100, color='darkblue', linewidth=2.5, label='AFT Survival Curve $P(T>t)$')
        
        # Add a dot for the ACTUAL clearance time
        ax.axvline(x=actual, color='green', linestyle='--', linewidth=2, label=f'Actual ({actual:.1f}m)')
        
        # Overlay the CatBoost Quantile Bounds as shaded regions / lines
        ax.axvline(x=q10, color='red', linestyle=':', linewidth=1.5, label=f'Q10 Optimistic ({q10:.1f}m)')
        ax.axvline(x=q90, color='darkred', linestyle=':', linewidth=1.5, label=f'Q90 Worst-Case ({q90:.1f}m)')
        ax.axvspan(q10, q90, color='red', alpha=0.1, label='CatBoost Safety Window')
        
        ax.set_title(f"Crash Event #{i+1} Survival Profile", fontsize=12)
        ax.set_xlabel("Minutes Since Crash Occurred", fontsize=10)
        ax.set_ylabel("Probability Crash is Still Active (%)", fontsize=10)
        ax.set_xlim(0, max(200, actual + 50, q90 + 20))
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    print("\n[SUCCESS] Check the popup window to view the Survival Curves!")
    plt.show()

if __name__ == "__main__":
    show_real_predictions_with_curves()