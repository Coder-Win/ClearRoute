import numpy as np
import pandas as pd
from pathlib import Path

STRUCT_TABLE_PATH = Path("data/processed/structural_feature_table.parquet")
SEMANTIC_FULL_PATH = Path("data/processed/us_accidents_semantic_features.parquet")
EMBEDDING_TABLE_PATH = Path("data/processed/us_accidents_embeddings.parquet")
FUSED_TABLE_OUT_PATH = Path("data/processed/fused_multimodal_table.parquet")


def load_fallback_semantic_modality(df_struct: pd.DataFrame) -> pd.DataFrame:
    """Fallback zero-shot rule extractor if semantic parquet file is missing."""
    print("[Stage 3] Generating fallback semantic feature table from descriptions...")
    desc = df_struct["Description"].fillna("").astype(str).str.lower()
    df_sem = pd.DataFrame({"ID": df_struct["ID"].astype(str)})
    
    df_sem["incident_type"] = desc.str.contains("accident|crash|collision").astype(int)
    df_sem["heavy_vehicle"] = desc.str.contains("truck|semi|tractor|trailer|commercial|bus").astype(int)
    df_sem["rollover"] = desc.str.contains("overturned|rollover|flipped").astype(int)
    df_sem["spill"] = desc.str.contains("spill|leak|fuel|diesel").astype(int)
    df_sem["fire"] = desc.str.contains("fire|burning|smoke|flames").astype(int)
    df_sem["debris"] = desc.str.contains("debris|tree|boulder|object").astype(int)
    df_sem["injury"] = desc.str.contains("injury|injured|ambulance|medical|fatality").astype(int)
    df_sem["lanes_blocked"] = desc.str.contains("lane blocked|lanes blocked|blocking|blocked").astype(int)
    df_sem["full_road_blockage"] = desc.str.contains("closed|all lanes|road blocked|both ways").astype(int)
    df_sem["ramp_involvement"] = desc.str.contains("ramp|exit|entry").astype(int)
    df_sem["disabled_vehicle"] = desc.str.contains("disabled|stalled|stopped vehicle|abandoned").astype(int)
    df_sem["hazardous_material"] = desc.str.contains("hazmat|hazardous|chemical|toxic").astype(int)
    df_sem["specialized_recovery_required"] = desc.str.contains("heavy tow|crane|recovery|hazmat").astype(int)
    df_sem["emergency_response_mentioned"] = desc.str.contains("police|patrol|responder|crew|fire dept").astype(int)
    df_sem["uncertainty_language"] = desc.str.contains("possible|unconfirmed|reported|unknown").astype(int)
    return df_sem


def load_fallback_embedding_modality(df_struct: pd.DataFrame) -> pd.DataFrame:
    """Fallback SVD vectorizer if embedding parquet file is missing."""
    print("[Stage 3] Generating fallback 64-d TruncatedSVD narrative vector representations...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    
    texts = df_struct["Description"].fillna("traffic incident").astype(str).tolist()
    vec = TfidfVectorizer(max_features=2000, stop_words="english")
    svd = TruncatedSVD(n_components=64, random_state=42)
    embeddings = svd.fit_transform(vec.fit_transform(texts)).astype(np.float32)
    
    emb_dict = {f"d{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
    return pd.DataFrame({"ID": df_struct["ID"].astype(str), **emb_dict})


def main():
    if not STRUCT_TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing {STRUCT_TABLE_PATH}. Run Phase 0 script first!")

    print(f"[Stage 3] Loading structural feature table from {STRUCT_TABLE_PATH}")
    df_struct = pd.read_parquet(STRUCT_TABLE_PATH)
    df_struct["ID"] = df_struct["ID"].astype(str)

    # Load Semantic Modality
    if SEMANTIC_FULL_PATH.exists():
        print(f"[Stage 3] Loading Semantic features from {SEMANTIC_FULL_PATH}")
        df_sem = pd.read_parquet(SEMANTIC_FULL_PATH)
    else:
        df_sem = load_fallback_semantic_modality(df_struct)
    df_sem["ID"] = df_sem["ID"].astype(str)

    # Load Embedding Modality
    if EMBEDDING_TABLE_PATH.exists():
        print(f"[Stage 3] Loading Embeddings from {EMBEDDING_TABLE_PATH}")
        df_emb = pd.read_parquet(EMBEDDING_TABLE_PATH)
    else:
        df_emb = load_fallback_embedding_modality(df_struct)
    df_emb["ID"] = df_emb["ID"].astype(str)

    # 1. Incident-ID Alignment (Inner Join across all 3 modalities)
    print("[Stage 3] Executing Incident-ID Alignment (Inner Join across all modalities)...")
    common_ids = set(df_struct["ID"]) & set(df_sem["ID"]) & set(df_emb["ID"])
    
    df_struct = df_struct[df_struct["ID"].isin(common_ids)].drop_duplicates("ID").sort_values("ID").reset_index(drop=True)
    df_sem = df_sem[df_sem["ID"].isin(common_ids)].drop_duplicates("ID").sort_values("ID").reset_index(drop=True)
    df_emb = df_emb[df_emb["ID"].isin(common_ids)].drop_duplicates("ID").sort_values("ID").reset_index(drop=True)

    # Keep crucial metadata columns explicitly attached to df_struct before prefixing
    meta_cols = ["ID", "duration", "raw_duration_mins", "censor_flag", "split", "Description", "Start_Time", "End_Time"]
    meta_df = df_struct[[c for c in meta_cols if c in df_struct.columns]].copy()

    raw_struct_feature_cols = [c for c in df_struct.columns if c not in meta_cols]
    
    # Encode categorical/boolean structural columns to numeric
    for col in raw_struct_feature_cols:
        if df_struct[col].dtype == "bool":
            df_struct[col] = df_struct[col].astype(int)
        elif df_struct[col].dtype == "object" or str(df_struct[col].dtype) == "category":
            df_struct[col] = pd.factorize(df_struct[col])[0]
        df_struct[col] = df_struct[col].fillna(df_struct[col].median())

    # 2. Modality Prefix Assignment
    print("[Stage 3] Assigning Modality Prefixes (struct_, sem_, emb_)...")
    struct_prefix_map = {c: f"struct_{c}" for c in raw_struct_feature_cols}
    sem_prefix_map = {c: f"sem_{c}" for c in df_sem.columns if c != "ID"}
    emb_prefix_map = {c: f"emb_{c}" for c in df_emb.columns if c != "ID"}

    df_struct_prefixed = df_struct[raw_struct_feature_cols].rename(columns=struct_prefix_map)
    df_sem = df_sem.rename(columns=sem_prefix_map)
    df_emb = df_emb.rename(columns=emb_prefix_map)

    # Merge metadata + prefixed structural + semantic + embedding tables on ID
    df_fused = pd.concat([
        meta_df, 
        df_struct_prefixed, 
        df_sem.drop(columns=["ID"], errors="ignore"), 
        df_emb.drop(columns=["ID"], errors="ignore")
    ], axis=1)

    # 3. Quality Scores (gate_q_tabular, gate_q_text_len, gate_q_semantic_density)
    print("[Stage 3] Computing Quality Signals / Scores...")
    prefixed_struct_cols = list(struct_prefix_map.values())
    prefixed_sem_cols = list(sem_prefix_map.values())

    df_fused["gate_q_tabular"] = df_fused[prefixed_struct_cols].notna().mean(axis=1)
    
    char_len = df_fused["Description"].fillna("").astype(str).str.len()
    log_len = np.log1p(char_len)
    df_fused["gate_q_text_len"] = (log_len - log_len.min()) / (log_len.max() - log_len.min() + 1e-8)
    
    df_fused["gate_q_semantic_density"] = df_fused[prefixed_sem_cols].fillna(0).mean(axis=1)

    # Final save
    df_fused.to_parquet(FUSED_TABLE_OUT_PATH, index=False)
    print(f"[Success] Saved Aligned & Fused Multimodal Table ({len(df_fused):,} rows, {df_fused.shape[1]} columns) to {FUSED_TABLE_OUT_PATH}")


if __name__ == "__main__":
    main()