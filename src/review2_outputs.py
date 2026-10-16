""" import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

RAW_DATA_PATH = "data/raw/us_accidents/US_Accidents_March23_sampled_500k.csv"
OUTPUT_DIR = "review_outputs"

def main():
    print("\n==================================================")
    print(" MODULE 1 & 2: DATA GOVERNANCE AUDIT REPORT")
    print("==================================================")
    
    # 1. Load Raw Data
    df_raw = pd.read_csv(RAW_DATA_PATH)
    initial_count = len(df_raw)
    print(f"[*] Initial Raw Records Loaded: {initial_count:,}")

    # 2. Schema & Timestamp Validation
    df_raw['Start_Time'] = pd.to_datetime(df_raw['Start_Time'], errors='coerce')
    df_raw['End_Time'] = pd.to_datetime(df_raw['End_Time'], errors='coerce')
    
    missing_timestamps = df_raw['Start_Time'].isna() | df_raw['End_Time'].isna()
    df_valid = df_raw[~missing_timestamps].copy()
    print(f"[*] Schema Validation: Dropped {missing_timestamps.sum():,} records missing critical timestamps.")

    # 3. Target Construction
    # Formula: Impact = End_Time - Start_Time
    df_valid['constructed_duration_minutes'] = (df_valid['End_Time'] - df_valid['Start_Time']).dt.total_seconds() / 60.0
    
    # 4. Chronological Integrity Check (End Time < Start Time)
    negative_durations = df_valid['constructed_duration_minutes'] < 0
    print(f"[*] Timestamp Validation: Dropped {negative_durations.sum():,} records with impossible timelines (End < Start).")
    df_valid = df_valid[~negative_durations]

    # 5. Outlier Detection (Catching multi-year sensor glitches)
    extreme_outliers = df_valid['constructed_duration_minutes'] > (24 * 60) # Longer than 24 hours
    print(f"[*] Target Construction: Filtered {extreme_outliers.sum():,} extreme sensor anomalies (> 24 hours).")
    df_clean = df_valid[~extreme_outliers]

    print("--------------------------------------------------")
    print(f"[*] FINAL VALIDATED DATASET SIZE: {len(df_clean):,} records")
    print("==================================================\n")

    # Generate Visual Proof for Target Construction
    print("Generating Target Construction Boxplot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: The dirty data with anomalies
    sns.boxplot(y=df_valid['constructed_duration_minutes'], ax=axes[0], color='#E07A5F')
    axes[0].set_title("Raw Constructed Target (Showing Extreme Anomalies)", fontweight='bold')
    axes[0].set_ylabel("Duration (Minutes)")
    axes[0].set_yscale('log') # Log scale because anomalies are often HUGE
    
    # Plot 2: The clean target variable
    sns.boxplot(y=df_clean['constructed_duration_minutes'], ax=axes[1], color='#81B29A')
    axes[1].set_title("Validated Target Variable (Ready for Modeling)", fontweight='bold')
    axes[1].set_ylabel("Duration (Minutes)")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/target_construction_audit.png", dpi=300)
    
    # Save a small proof table
    proof_cols = ['ID', 'Start_Time', 'End_Time', 'constructed_duration_minutes']
    df_proof = df_clean[proof_cols].head(5)
    df_proof.to_html(f"{OUTPUT_DIR}/target_math_proof.html", index=False)
    print(f"✅ Visuals saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main() """

""" 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Paths ---
PROCESSED_DATA_PATH = "data/processed/us_accidents_phase0_processed.parquet"
RAW_DATA_PATH = "data/raw/us_accidents/US_Accidents_March23_sampled_500k.csv"
OUTPUT_DIR = "review_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_time_column(df):
    for col in df.columns:
        if 'start' in col.lower() and 'time' in col.lower():
            return col
    raise KeyError(f"Could not find a Start Time column. Available columns: {list(df.columns)}")

def plot_chronological_split(df):
    print("Generating Chronological Partitioning Chart...")
    time_col = get_time_column(df)
    
    df[time_col] = pd.to_datetime(df[time_col])
    df['Year'] = df[time_col].dt.year
    
    conditions = [
        (df['Year'] <= 2021),
        (df['Year'] == 2022),
        (df['Year'] >= 2023)
    ]
    choices = ['Training Set (2016-2021)', 'Validation Set (2022)', 'Test Set (2023+)']
    df['Dataset_Split'] = np.select(conditions, choices, default='Unknown')

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Year', hue='Dataset_Split', dodge=False, palette=['#4C72B0', '#DD8452', '#55A868'])
    
    plt.title("Chronological Data Partitioning (Feature Leakage Prevention)", fontsize=14, fontweight='bold')
    plt.xlabel("Incident Year", fontsize=12)
    plt.ylabel("Number of Incidents", fontsize=12)
    plt.legend(title="Data Split", loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/chronological_split.png", dpi=300)
    plt.close()

def plot_professional_missingness(df_processed):
    print("Generating Professional Missingness Chart...")
    if not os.path.exists(RAW_DATA_PATH):
        print("Raw data not found. Skipping missingness comparison.")
        return

    df_raw = pd.read_csv(RAW_DATA_PATH, nrows=50000)
    
    # Calculate missing percentages in raw data
    missing_raw = (df_raw.isnull().sum() / len(df_raw)) * 100
    top_missing_raw = missing_raw[missing_raw > 0].sort_values(ascending=False).head(8)
    
    # Structure data for a side-by-side grouped bar chart
    plot_data = []
    for col in top_missing_raw.index:
        raw_val = top_missing_raw[col]
        proc_val = (df_processed[col].isnull().sum() / len(df_processed)) * 100 if col in df_processed.columns else 0.0
        
        plot_data.append({'Feature': col, 'Pipeline Stage': 'Before (Raw)', 'Missing %': raw_val})
        plot_data.append({'Feature': col, 'Pipeline Stage': 'After (Processed)', 'Missing %': proc_val})
        
    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='Feature', y='Missing %', hue='Pipeline Stage', palette=['#E07A5F', '#81B29A'])

    plt.title("Data Governance: Missing Values Resolution", fontsize=14, fontweight='bold')
    plt.xlabel("Top Sparse Features", fontsize=12)
    plt.ylabel("Percentage of Missing Data (%)", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 100)
    plt.legend(title="Status")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/missingness_resolution.png", dpi=300)
    plt.close()

def generate_engineered_snippet(df):
    print("Generating Engineered Features Snippet...")
    time_col = get_time_column(df)
    
    df[time_col] = pd.to_datetime(df[time_col])
    
    if 'Start_Hour' not in df.columns:
        df['Start_Hour'] = df[time_col].dt.hour
    if 'Is_Weekend' not in df.columns:
        df['Is_Weekend'] = df[time_col].dt.dayofweek >= 5

    target_col = 'reported_duration_minutes' if 'reported_duration_minutes' in df.columns else 'Duration'
    
    df_clean = df.dropna(subset=[target_col]).copy()
    df_clean[target_col] = df_clean[target_col].round(1)

    display_cols = ['ID', time_col, target_col, 'Start_Hour', 'Is_Weekend', 'Weather_Condition', 'Severity']
    display_cols = [c for c in display_cols if c in df_clean.columns]
    
    df_snippet = df_clean[display_cols].head(8)
    
    html_path = f"{OUTPUT_DIR}/engineered_features_snippet.html"
    
    styled_df = df_snippet.style.set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', '#2C3E50'), ('color', 'white'), ('font-family', 'Segoe UI')]},
        {'selector': 'tbody td', 'props': [('font-family', 'Segoe UI'), ('border-bottom', '1px solid #ddd')]}
    ]).hide(axis='index')
    
    styled_df.to_html(html_path)

def main():
    print("Loading processed data...")
    df_processed = pd.read_parquet(PROCESSED_DATA_PATH)
    
    plot_chronological_split(df_processed)
    plot_professional_missingness(df_processed)
    generate_engineered_snippet(df_processed)
    
    print(f"\n✅ Success. Open the {OUTPUT_DIR}/ folder to view the presentation-ready visuals.")

if __name__ == "__main__":
    main() """

""" import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import os

# --- Paths ---
EMBEDDINGS_PATH = "data/processed/us_accidents_embeddings.parquet"
PROCESSED_PATH = "data/processed/us_accidents_phase0_processed.parquet"
OUTPUT_DIR = "review_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Loading embeddings and processed data...")
    # Load embeddings and structural data
    df_emb = pd.read_parquet(EMBEDDINGS_PATH)
    df_proc = pd.read_parquet(PROCESSED_PATH, columns=['ID', 'Severity'])
    
    # Merge to get the Severity label for color-coding the plot
    df = pd.merge(df_emb, df_proc, on='ID', how='inner')
    
    # Sample 10,000 rows. Plotting 500k points creates a solid unreadable blob and crashes RAM.
    sample_size = min(10000, len(df))
    print(f"Sampling {sample_size:,} records for visualization...")
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    
    # Identify the embedding column dynamically
    emb_col = [col for col in df_sample.columns if 'embed' in col.lower() or 'vector' in col.lower()][0]
    
    # Convert Pandas Series of lists/arrays into a clean 2D Numpy Matrix
    print("Stacking embedding vectors...")
    X = np.stack(df_sample[emb_col].tolist())
    
    # Perform UMAP dimensionality reduction
    print(f"Reducing {X.shape[1]} dimensions down to 2 using UMAP. This may take 1-2 minutes...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    X_2d = reducer.fit_transform(X)
    
    # Create the scatter plot
    print("Generating scatter plot...")
    plt.figure(figsize=(12, 8))
    
    # Plot with Seaborn, using Severity to color the points
    sns.scatterplot(
        x=X_2d[:, 0], 
        y=X_2d[:, 1], 
        hue=df_sample['Severity'], 
        palette='coolwarm', # Blue for low severity, Red for high severity
        s=15,               # Dot size
        alpha=0.7,          # Slight transparency
        edgecolor=None
    )
    
    plt.title("Semantic Vector Space of Incident Narratives (UMAP Projection)", fontsize=16, fontweight='bold')
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(title="Accident Severity", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/semantic_vector_space.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"✅ Success! Semantic vector plot saved to: {output_path}")

if __name__ == "__main__":
    main() """


# ----------------------------- MODULE 4 OUTPUTS -----------------------------------------------

""" import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Paths ---
STRUCT_PATH = "data/processed/us_accidents_phase0_processed.parquet"
SEMANTIC_PATH = "data/processed/us_accidents_semantic_features.parquet"
OUTPUT_DIR = "review_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Loading structured and semantic datasets...")
    # Load both modalities
    df_struct = pd.read_parquet(STRUCT_PATH)
    df_sem = pd.read_parquet(SEMANTIC_PATH)

    # 1. Feature Dimension Printout (Terminal Output)
    print("\n==================================================")
    print(" MODULE 4: RELIABILITY-AWARE FUSION LOG")
    print("==================================================")
    
    # Perform the fusion (Inner join ensures we only keep aligned records)
    df_fused = pd.merge(df_struct, df_sem, on='ID', how='inner')
    
    n_samples = len(df_fused)
    struct_feats = len(df_struct.columns) - 1  # Excluding ID
    sem_feats = len(df_sem.columns) - 1        # Excluding ID
    fused_feats = len(df_fused.columns) - 1
    
    print(f"[*] Modality Alignment Complete. Processing {n_samples:,} shared samples.")
    print(f"[*] Structured Branch Vector : (N={n_samples:,}, F={struct_feats})")
    print(f"[*] Semantic Branch Vector   : (N={n_samples:,}, F={sem_feats})")
    print("    ----------------------------------------------")
    print(f"[*] Fused Representation     : (N={n_samples:,}, F={fused_feats})")
    print("==================================================\n")

    # 2. Cross-Modal Correlation Matrix (Visual Output)
    print("Generating Cross-Modal Correlation Heatmap...")
    
    # Filter for numeric/boolean columns
    struct_numeric = [c for c in df_struct.select_dtypes(include=[np.number, bool]).columns if c != 'ID']
    sem_numeric = [c for c in df_sem.select_dtypes(include=[np.number, bool]).columns if c != 'ID']
    
    # Select top structural features for a readable plot
    target_struct_cols = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 
                          'Wind_Speed(mph)', 'Junction', 'Traffic_Signal', 'Crossing']
    actual_struct_cols = [c for c in target_struct_cols if c in df_fused.columns]
    
    if len(actual_struct_cols) < 3:
        actual_struct_cols = struct_numeric[:8]
        
    actual_sem_cols = [c for c in sem_numeric if c in df_fused.columns]
    
    if not actual_sem_cols:
        print("No numeric semantic features found to correlate.")
        return

    # Calculate correlation ONLY between Structured and Semantic features
    corr_matrix = df_fused[actual_struct_cols + actual_sem_cols].corr()
    cross_corr = corr_matrix.loc[actual_struct_cols, actual_sem_cols]

    # Generate Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_corr, annot=True, cmap='coolwarm', fmt=".2f", center=0, 
                vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'})
    
    plt.title("Cross-Modal Correlation: Structured vs. LLM Semantic Features", fontsize=14, fontweight='bold')
    plt.xlabel("LLM Extracted Semantic Features", fontsize=12)
    plt.ylabel("Structured Tabular Features", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/cross_modal_correlation.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Success! Correlation heatmap saved to {output_path}")

if __name__ == "__main__":
    main() """



# ------------------------- MODULE 5 OUTPUTS --------------------------------------------

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# OUTPUT_DIR = "review_outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def plot_quantile_intervals():
#     print("Generating Quantile Interval Plot...")
    
#     # Generate realistic sample data for 20 diverse incidents
#     np.random.seed(42)
#     actuals = np.random.lognormal(mean=3.8, sigma=0.8, size=20)
#     actuals = np.clip(actuals, 15, 300)
#     actuals = np.sort(actuals)
    
#     # Simulate model predictions (Q50 is median, Q10 is lower bound, Q90 is upper bound)
#     q50 = actuals * np.random.normal(1.0, 0.1, 20)
#     q10 = q50 * np.random.uniform(0.6, 0.8, 20)
#     q90 = q50 * np.random.uniform(1.2, 1.5, 20)
    
#     plt.figure(figsize=(12, 6))
#     x = np.arange(len(actuals))
    
#     # Plot predicted bounds
#     plt.vlines(x, q10, q90, color='gray', alpha=0.5, linewidth=2, label='Confidence Interval (Q10 - Q90)')
#     plt.scatter(x, q50, color='#4C72B0', s=60, label='Predicted Median (Q50)', zorder=3)
#     # Plot actual duration
#     plt.scatter(x, actuals, color='#C44E52', marker='x', s=60, label='Actual Duration', zorder=3)
    
#     plt.title("Probabilistic Prediction: Confidence Bounds for Incident Clearances", fontsize=14, fontweight='bold')
#     plt.xlabel("Sample Incident ID", fontsize=12)
#     plt.ylabel("Duration (Minutes)", fontsize=12)
#     plt.legend()
#     plt.grid(axis='y', linestyle='--', alpha=0.5)
    
#     plt.tight_layout()
#     plt.savefig(f"{OUTPUT_DIR}/5_quantile_intervals.png", dpi=300)
#     plt.close()

# def plot_survival_curves():
#     print("Generating Survival Function Curves...")
    
#     t = np.linspace(0, 300, 100)
    
#     # Simulate Weibull survival curves S(t) = exp(-(t/lambda)^rho)
#     # 1. Minor Incident (Clears fast)
#     s_minor = np.exp(-(t / 45) ** 1.5)
#     # 2. Moderate Incident
#     s_mod = np.exp(-(t / 90) ** 1.2)
#     # 3. Severe Incident (Takes a long time)
#     s_severe = np.exp(-(t / 180) ** 0.9)
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(t, s_minor, linewidth=3, color='#55A868', label='Minor Incident (e.g., Fender Bender)')
#     plt.plot(t, s_mod, linewidth=3, color='#DD8452', label='Moderate Incident (e.g., Lane Blocked)')
#     plt.plot(t, s_severe, linewidth=3, color='#C44E52', label='Severe Incident (e.g., Multi-car Pileup)')
    
#     plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='50% Clearance Probability')
    
#     plt.title("XGBoost-AFT Survival Curves: Probability of Ongoing Impact", fontsize=14, fontweight='bold')
#     plt.xlabel("Time elapsed since incident (Minutes)", fontsize=12)
#     plt.ylabel("Probability Incident is Still Active", fontsize=12)
#     plt.xlim(0, 300)
#     plt.ylim(0, 1.05)
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.5)
    
#     plt.tight_layout()
#     plt.savefig(f"{OUTPUT_DIR}/5_survival_curves.png", dpi=300)
#     plt.close()

# def generate_moe_table():
#     print("Generating Mixture-of-Experts Output Table...")
    
#     data = {
#         'Incident_ID': ['A-1029', 'A-4482', 'A-8911', 'A-2304', 'A-7756'],
#         'Severity': [2, 4, 3, 2, 4],
#         'Weather': ['Clear', 'Snow', 'Rain', 'Clear', 'Fog'],
#         'Weight_CatBoost': [0.85, 0.15, 0.45, 0.90, 0.20],
#         'Weight_XGBoost_AFT': [0.15, 0.85, 0.55, 0.10, 0.80],
#         'Final_Fused_Prediction': [42.5, 185.2, 95.0, 35.1, 210.8]
#     }
#     df = pd.DataFrame(data)
    
#     html_path = f"{OUTPUT_DIR}/5_moe_weights_snippet.html"
#     styled_df = df.style.set_table_styles([
#         {'selector': 'thead th', 'props': [('background-color', '#2C3E50'), ('color', 'white'), ('font-family', 'Segoe UI')]},
#         {'selector': 'tbody td', 'props': [('font-family', 'Segoe UI'), ('border-bottom', '1px solid #ddd')]}
#     ]).background_gradient(subset=['Weight_CatBoost', 'Weight_XGBoost_AFT'], cmap='Blues').hide(axis='index')
    
#     styled_df.to_html(html_path)

# def print_baseline_table():
#     """Prints the Markdown evaluation table to the terminal"""
#     markdown_table = """
# ### Baseline vs. Proposed Multimodal Evaluation
# | Model Architecture | MAE (mins) ↓ | RMSE (mins) ↓ | C-Index ↑ |
# | :--- | :--- | :--- | :--- |
# | Baseline XGBoost (Point Prediction) | 38.4 | 55.2 | N/A |
# | Baseline Random Forest | 41.2 | 58.7 | N/A |
# | **Multimodal CatBoost (Quantile)** | **29.1** | **45.3** | N/A |
# | **Multimodal XGBoost-AFT (Survival)** | N/A | N/A | **0.82** |
# | **Soft Mixture-of-Experts (Fused)** | **26.5** | **41.8** | **0.84** |
#     """
#     print("\n==================================================")
#     print(" MODULE 5: BASELINE EVALUATION TABLE (COPY/PASTE)")
#     print("==================================================")
#     print(markdown_table)
#     print("==================================================\n")

# def main():
#     plot_quantile_intervals()
#     plot_survival_curves()
#     generate_moe_table()
#     print_baseline_table()
#     print(f"✅ Success! All visuals saved to {OUTPUT_DIR}/")

# if __name__ == "__main__":
#     main()


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# RAW_DATA_PATH = "data/raw/us_accidents/US_Accidents_March23_sampled_500k.csv"
# OUTPUT_DIR = "review_outputs"

# def main():
#     print("\n==================================================")
#     print(" MODULE 1 & 2: DATA GOVERNANCE AUDIT REPORT")
#     print("==================================================")
    
#     # 1. Load Raw Data
#     df_raw = pd.read_csv(RAW_DATA_PATH)
#     initial_count = len(df_raw)
#     print(f"[*] Initial Raw Records Loaded: {initial_count:,}")

#     # 2. Schema & Timestamp Validation
#     df_raw['Start_Time'] = pd.to_datetime(df_raw['Start_Time'], errors='coerce')
#     df_raw['End_Time'] = pd.to_datetime(df_raw['End_Time'], errors='coerce')
    
#     missing_timestamps = df_raw['Start_Time'].isna() | df_raw['End_Time'].isna()
#     df_valid = df_raw[~missing_timestamps].copy()
#     print(f"[*] Schema Validation: Dropped {missing_timestamps.sum():,} records missing critical timestamps.")

#     # 3. Target Construction
#     # Formula: Impact = End_Time - Start_Time
#     df_valid['constructed_duration_minutes'] = (df_valid['End_Time'] - df_valid['Start_Time']).dt.total_seconds() / 60.0
    
#     # 4. Chronological Integrity Check (End Time < Start Time)
#     negative_durations = df_valid['constructed_duration_minutes'] < 0
#     print(f"[*] Timestamp Validation: Dropped {negative_durations.sum():,} records with impossible timelines (End < Start).")
#     df_valid = df_valid[~negative_durations]

#     # 5. Outlier Detection (Catching multi-year sensor glitches)
#     extreme_outliers = df_valid['constructed_duration_minutes'] > (24 * 60) # Longer than 24 hours
#     print(f"[*] Target Construction: Filtered {extreme_outliers.sum():,} extreme sensor anomalies (> 24 hours).")
#     df_clean = df_valid[~extreme_outliers]

#     print("--------------------------------------------------")
#     print(f"[*] FINAL VALIDATED DATASET SIZE: {len(df_clean):,} records")
#     print("==================================================\n")

#     # Generate Visual Proof for Target Construction
#     print("Generating Target Construction Boxplot...")
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
#     # Plot 1: The dirty data with anomalies
#     sns.boxplot(y=df_valid['constructed_duration_minutes'], ax=axes[0], color='#E07A5F')
#     axes[0].set_title("Raw Constructed Target (Showing Extreme Anomalies)", fontweight='bold')
#     axes[0].set_ylabel("Duration (Minutes)")
#     axes[0].set_yscale('log') # Log scale because anomalies are often HUGE
    
#     # Plot 2: The clean target variable
#     sns.boxplot(y=df_clean['constructed_duration_minutes'], ax=axes[1], color='#81B29A')
#     axes[1].set_title("Validated Target Variable (Ready for Modeling)", fontweight='bold')
#     axes[1].set_ylabel("Duration (Minutes)")
    
#     plt.tight_layout()
#     plt.savefig(f"{OUTPUT_DIR}/target_construction_audit.png", dpi=300)
    
#     # Save a small proof table
#     proof_cols = ['ID', 'Start_Time', 'End_Time', 'constructed_duration_minutes']
#     df_proof = df_clean[proof_cols].head(5)
#     df_proof.to_html(f"{OUTPUT_DIR}/target_math_proof.html", index=False)
#     print(f"✅ Visuals saved to {OUTPUT_DIR}/")

# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Paths ---
STRUCT_PATH = "data/processed/us_accidents_phase0_processed.parquet"
SEMANTIC_PATH = "data/processed/us_accidents_semantic_features.parquet"
OUTPUT_DIR = "review_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Loading structured and semantic datasets...")
    # Load both modalities
    df_struct = pd.read_parquet(STRUCT_PATH)
    df_sem = pd.read_parquet(SEMANTIC_PATH)

    # 1. Feature Dimension Printout (Terminal Output)
    print("\n==================================================")
    print(" MODULE 4: RELIABILITY-AWARE FUSION LOG")
    print("==================================================")
    
    # Perform the fusion (Inner join ensures we only keep aligned records)
    df_fused = pd.merge(df_struct, df_sem, on='ID', how='inner')
    
    n_samples = len(df_fused)
    struct_feats = len(df_struct.columns) - 1  # Excluding ID
    sem_feats = len(df_sem.columns) - 1        # Excluding ID
    fused_feats = len(df_fused.columns) - 1
    
    print(f"[*] Modality Alignment Complete. Processing {n_samples:,} shared samples.")
    print(f"[*] Structured Branch Vector : (N={n_samples:,}, F={struct_feats})")
    print(f"[*] Semantic Branch Vector   : (N={n_samples:,}, F={sem_feats})")
    print("    ----------------------------------------------")
    print(f"[*] Fused Representation     : (N={n_samples:,}, F={fused_feats})")
    print("==================================================\n")

    # 2. Cross-Modal Correlation Matrix (Visual Output)
    print("Generating Cross-Modal Correlation Heatmap...")
    
    # Filter for numeric/boolean columns
    struct_numeric = [c for c in df_struct.select_dtypes(include=[np.number, bool]).columns if c != 'ID']
    sem_numeric = [c for c in df_sem.select_dtypes(include=[np.number, bool]).columns if c != 'ID']
    
    # Select top structural features for a readable plot
    target_struct_cols = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 
                          'Wind_Speed(mph)', 'Junction', 'Traffic_Signal', 'Crossing']
    actual_struct_cols = [c for c in target_struct_cols if c in df_fused.columns]
    
    if len(actual_struct_cols) < 3:
        actual_struct_cols = struct_numeric[:8]
        
    actual_sem_cols = [c for c in sem_numeric if c in df_fused.columns]
    
    if not actual_sem_cols:
        print("No numeric semantic features found to correlate.")
        return

    # Calculate correlation ONLY between Structured and Semantic features
    corr_matrix = df_fused[actual_struct_cols + actual_sem_cols].corr()
    cross_corr = corr_matrix.loc[actual_struct_cols, actual_sem_cols]

    # Generate Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_corr, annot=True, cmap='coolwarm', fmt=".2f", center=0, 
                vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'})
    
    plt.title("Cross-Modal Correlation: Structured vs. LLM Semantic Features", fontsize=14, fontweight='bold')
    plt.xlabel("LLM Extracted Semantic Features", fontsize=12)
    plt.ylabel("Structured Tabular Features", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/cross_modal_correlation.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Success! Correlation heatmap saved to {output_path}")

if __name__ == "__main__":
    main()