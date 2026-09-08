"""
Generates Sentence Transformer embeddings for the incident descriptions.
"""

from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import warnings

warnings.filterwarnings("ignore")

INPUT_PATH = Path("data/processed/us_accidents_phase0_processed.parquet")
OUTPUT_PATH = Path("data/processed/us_accidents_embeddings.parquet")

def main():
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    
    # Filter to valid rows with descriptions
    df_valid = df[df["valid_initial_duration"].fillna(False) & df["Description"].notna()].copy()
    
    # WARNING: To save time and memory during testing, we sample 10,000 rows.
    # Remove the .sample() call when you are ready to process the entire dataset.
    if len(df_valid) > 10000:
        print("Sampling 10,000 rows for processing...")
        df_valid = df_valid.sample(n=10000, random_state=42).copy()

    descriptions = df_valid["Description"].astype(str).tolist()

    print("Loading Sentence Transformer model (all-MiniLM-L6-v2)...")
    # This is a lightweight, highly efficient embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating dense text embeddings (this may take a few minutes)...")
    embeddings = model.encode(descriptions, show_progress_bar=True)

    # Convert the dense matrix into a list of arrays to store in Parquet
    df_valid["text_embedding"] = list(embeddings)
    
    # Keep only the ID and the embedding to merge later
    df_embeddings = df_valid[["ID", "text_embedding"]]

    df_embeddings.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n[COMPLETE] Embeddings generated and saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()