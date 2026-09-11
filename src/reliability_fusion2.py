import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

def safe_parse_embedding(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            clean = val.replace('[', '').replace(']', '').split()
            return [float(i) for i in clean]
    return val

def main():
    print("Step 1: Loading Data Streams...")
    processed_dir = Path("data/processed")

    # READ FROM THE NEW SPATIOTEMPORAL PARQUET FILE
    df_struct = pd.read_parquet(processed_dir / "us_accidents_spatiotemporal.parquet")
    df_sem = pd.read_csv(processed_dir / "us_accidents_semantic_features.csv")
    df_emb = pd.read_csv(processed_dir / "us_accidents_embeddings.csv")

    print("Step 2: Merging Data Streams for Perfect Alignment...")
    df_merged = df_struct.merge(df_sem, on="ID", how="inner").merge(df_emb, on="ID", how="inner")

    # 🚨 EXACT SCHEMA LEAK & METADATA PREVENTION 🚨
    explicit_exclude = {
        # Identifiers & Text
        'ID', 'Description', 'description_normalized', 'description_length',
        # Time Leaks (Model would just do End - Start to cheat)
        'Start_Time_parsed', 'End_Time_parsed', 
        # Target Variables & Derivative Leaks (ADDED event_observed)
        'reported_duration_minutes', 'valid_initial_duration', 'duration_requires_review', 'event_observed',
        # Pipeline Metadata (No predictive value)
        'dataset_source', 'target_name', 'target_definition', 'split',
        # Exclude the string ID from the spatial step
        'primary_incident_id'
    }

    # Only safely filter columns that actually belong to the structured feature set.
    # NOTE: 'is_secondary_crash' will automatically be included here as a valid feature!
    struct_cols = [c for c in df_struct.columns if c not in explicit_exclude]
    sem_cols = [c for c in df_sem.columns if c != 'ID']

    print("Step 3: Calculating Advanced Gating Meta-Features...")
    # Calculate tabular quality purely based on the safe, predictive columns
    df_merged['gate_q_tabular'] = df_merged[struct_cols].notna().mean(axis=1)
    
    desc_lengths = df_merged['Description'].astype(str).apply(len)
    log_lengths = np.log1p(desc_lengths).values.reshape(-1, 1) 
    scaler = MinMaxScaler()
    df_merged['gate_q_text_len'] = scaler.fit_transform(log_lengths)
    df_merged['gate_q_semantic_density'] = df_merged[sem_cols].sum(axis=1) / len(sem_cols)

    gate_cols = ['gate_q_tabular', 'gate_q_text_len', 'gate_q_semantic_density']

    print("Step 4: Parsing Embeddings...")
    parsed_embeddings = df_merged['text_embedding'].apply(safe_parse_embedding)
    emb_matrix = np.vstack(parsed_embeddings.values)
    emb_df_raw = pd.DataFrame(emb_matrix, columns=[f'emb_{i}' for i in range(emb_matrix.shape[1])])

    print("Step 5: Generating Cleaned Ablation Datasets...")
    
    # ADDED event_observed to the base targets to pass it to the models
    base_targets = ['ID', 'reported_duration_minutes', 'event_observed']
    
    # Dataset 1: Structured Only (Target + Safe Tabular features, no semantics, no embeddings, no gates)
    df_struct_only = df_merged[base_targets + struct_cols]
    
    # Dataset 2: Structured + Semantic (Target + Tabular + LLM Flags, no embeddings, no gates)
    df_struct_sem = df_merged[base_targets + struct_cols + sem_cols]
    
    # Dataset 3: Structured + Embedding (Target + Tabular + Raw Embeddings, NO semantics, NO gates)
    df_struct_emb = pd.concat([df_struct_only, emb_df_raw], axis=1)
    
    # Dataset 4: Dumb Join (Target + Tabular + LLM Flags + Embeddings, NO gates)
    df_dumb = pd.concat([df_struct_sem, emb_df_raw], axis=1)
    
    # Dataset 5: Gated Fusion (Target + Tabular + LLM Flags + Embeddings + Gates)
    df_fused = pd.concat([df_struct_sem, df_merged[gate_cols], emb_df_raw], axis=1)

    # Save all five datasets with "_v2" to ensure original files are not overwritten
    df_struct_only.to_csv(processed_dir / "us_accidents_struct_only_v2.csv", index=False)
    df_struct_sem.to_csv(processed_dir / "us_accidents_struct_sem_v2.csv", index=False)
    df_struct_emb.to_csv(processed_dir / "us_accidents_struct_emb_v2.csv", index=False)
    df_dumb.to_csv(processed_dir / "us_accidents_dumb_join_v2.csv", index=False)
    df_fused.to_csv(processed_dir / "us_accidents_fused_multimodal_v2.csv", index=False)
    
    print("\n[SUCCESS] All 5 V2 Ablation Datasets safely sanitized and saved.")

if __name__ == "__main__":
    main()