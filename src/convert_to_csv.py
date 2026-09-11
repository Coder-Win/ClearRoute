""" import pandas as pd
from pathlib import Path

# Define paths
processed_pq = Path("data/processed/us_accidents_phase0_processed.parquet")
processed_csv = Path("data/processed/us_accidents_phase0_processed.csv")

embeddings_pq = Path("data/processed/us_accidents_embeddings.parquet")
embeddings_csv = Path("data/processed/us_accidents_embeddings.csv")

# Convert Processed Data
if processed_pq.exists():
    print("Converting processed data to CSV...")
    df = pd.read_parquet(processed_pq)
    df.to_csv(processed_csv, index=False)
    print(f"Saved: {processed_csv}")

# Convert Embeddings
if embeddings_pq.exists():
    print("Converting embeddings to CSV...")
    df_emb = pd.read_parquet(embeddings_pq)
    df_emb.to_csv(embeddings_csv, index=False)
    print(f"Saved: {embeddings_csv}")

if embeddings_pq.exists():
    print("Converting embeddings to CSV...")
    df_emb = pd.read_parquet(embeddings_pq)
    df_emb.to_csv(embeddings_csv, index=False)
    print(f"Saved: {embeddings_csv}")


print("\n[COMPLETE] You can now delete the .parquet files from your folder.") """


from pathlib import Path
import pandas as pd

parquet_path = Path(
    r"C:\Programming\clear-route\data\processed\us_accidents_semantic_features.parquet"
)
csv_path = parquet_path.with_suffix(".csv")

if parquet_path.exists():
    print("Converting semantic features to CSV...")
    df = pd.read_parquet(parquet_path)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print("\n[COMPLETE] You can now delete the .parquet file if desired.")
else:
    print(f"File not found: {parquet_path}")
