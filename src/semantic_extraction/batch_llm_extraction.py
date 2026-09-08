"""
Batch LLM Semantic Extraction (Local Version)
Processes 10,000 sampled incidents using a local LLM via Ollama.
"""

import pandas as pd
import json
import ollama
from pathlib import Path
from tqdm import tqdm

# Paths
EMBEDDINGS_PATH = Path("data/processed/us_accidents_embeddings.parquet")
PROCESSED_PATH = Path("data/processed/us_accidents_phase0_processed.parquet")
OUTPUT_PATH = Path("data/processed/us_accidents_semantic_features.parquet")
SCHEMA_PATH = Path("schemas/semantic_schema.json")

def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

def query_local_llm(description: str, schema: dict) -> dict:
    prompt = f"""
    Extract the operational features according to the JSON schema below.
    Use ONLY "true", "false", or "unknown" for boolean fields.
    Schema: {json.dumps(schema['fields'])}
    Description: "{description}"
    """
    try:
        # Calls the local Llama 3 model running on your machine
        response = ollama.chat(
            model='llama3',
            messages=[
                {"role": "system", "content": "You are a data extractor. You must output strict, valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            format='json' # Forces Ollama to return a JSON object
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        return {}

def main():
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError("Run generate_embeddings.py first to establish the 10k sample.")

    print("Loading datasets...")
    sampled_ids = pd.read_parquet(EMBEDDINGS_PATH)["ID"]
    
    df_full = pd.read_parquet(PROCESSED_PATH)
    df_sample = df_full[df_full["ID"].isin(sampled_ids)].copy()

    schema = load_schema()
    extracted_data = []

    print(f"Starting Local LLM extraction for {len(df_sample)} rows...")
    
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        incident_id = row['ID']
        description = row['Description']
        
        llm_output = query_local_llm(str(description), schema)
        llm_output['ID'] = incident_id
        extracted_data.append(llm_output)

    print("\nSaving semantic features...")
    df_semantic = pd.DataFrame(extracted_data)
    
    for col in df_semantic.columns:
        if col != 'ID':
            df_semantic[col] = df_semantic[col].astype(str).str.lower().map({'true': 1, 'false': 0, 'unknown': 0}).fillna(0)

    df_semantic.to_parquet(OUTPUT_PATH, index=False)
    print(f"[COMPLETE] Semantic features saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()