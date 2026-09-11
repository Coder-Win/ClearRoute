import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

def show_sample_predictions():
    print("Loading data and training CatBoost Quantile model (this takes ~10 seconds)...")
    df_fused = pd.read_csv("data/processed/us_accidents_fused_multimodal.csv")
    
    target_col = 'reported_duration_minutes'
    df_fused = df_fused[df_fused[target_col] > 0]
    
    y = df_fused[target_col]
    X_fused = df_fused.drop(columns=[target_col, 'ID']).select_dtypes(include=[np.number])
    
    X_train, X_test, y_train, y_test = train_test_split(X_fused, y, test_size=0.2, random_state=42)
    
    # Train the Quantile model
    cat_quant = CatBoostRegressor(loss_function='MultiQuantile:alpha=0.1,0.5,0.9', iterations=100, verbose=0)
    cat_quant.fit(X_train, y_train)
    
    print("\n=== MODEL PREDICTIONS FOR 3 RANDOM CRASHES ===")
    
    # Pick 3 random crashes from the test set
    sample_X = X_test.sample(3, random_state=99)
    sample_y = y_test.loc[sample_X.index]
    
    # Generate the 10th, 50th, and 90th percentile predictions
    predictions = cat_quant.predict(sample_X)
    
    for i in range(3):
        actual_time = sample_y.iloc[i]
        optimistic = predictions[i][0]
        expected = predictions[i][1]
        conservative = predictions[i][2]
        
        print(f"\nCrash #{i+1}:")
        print(f"  -> ACTUAL Clearance Time:  {actual_time:.1f} minutes")
        print(f"  -> MODEL PREDICTION WINDOW:")
        print(f"       Optimistic (10%):   {optimistic:.1f} minutes")
        print(f"       Expected (50%):     {expected:.1f} minutes")
        print(f"       Conservative (90%): {conservative:.1f} minutes")

if __name__ == "__main__":
    show_sample_predictions()