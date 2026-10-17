# import numpy as np
# import pandas as pd
# from pathlib import Path

# STRUCT_TABLE_PATH = Path("data/processed/structural_feature_table.parquet")
# SEMANTIC_FULL_PATH = Path("data/processed/us_accidents_semantic_features.parquet")
# EMBEDDING_TABLE_PATH = Path("data/processed/us_accidents_embeddings.parquet")
# FUSED_TABLE_OUT_PATH = Path("data/processed/fused_multimodal_table.parquet")


# def load_fallback_semantic_modality(df_struct: pd.DataFrame) -> pd.DataFrame:
#     """Fallback zero-shot rule extractor if semantic parquet file is missing."""
#     print("[Stage 3] Generating fallback semantic feature table from descriptions...")
#     desc = df_struct["Description"].fillna("").astype(str).str.lower()
#     df_sem = pd.DataFrame({"ID": df_struct["ID"].astype(str)})
    
#     df_sem["incident_type"] = desc.str.contains("accident|crash|collision").astype(int)
#     df_sem["heavy_vehicle"] = desc.str.contains("truck|semi|tractor|trailer|commercial|bus").astype(int)
#     df_sem["rollover"] = desc.str.contains("overturned|rollover|flipped").astype(int)
#     df_sem["spill"] = desc.str.contains("spill|leak|fuel|diesel").astype(int)
#     df_sem["fire"] = desc.str.contains("fire|burning|smoke|flames").astype(int)
#     df_sem["debris"] = desc.str.contains("debris|tree|boulder|object").astype(int)
#     df_sem["injury"] = desc.str.contains("injury|injured|ambulance|medical|fatality").astype(int)
#     df_sem["lanes_blocked"] = desc.str.contains("lane blocked|lanes blocked|blocking|blocked").astype(int)
#     df_sem["full_road_blockage"] = desc.str.contains("closed|all lanes|road blocked|both ways").astype(int)
#     df_sem["ramp_involvement"] = desc.str.contains("ramp|exit|entry").astype(int)
#     df_sem["disabled_vehicle"] = desc.str.contains("disabled|stalled|stopped vehicle|abandoned").astype(int)
#     df_sem["hazardous_material"] = desc.str.contains("hazmat|hazardous|chemical|toxic").astype(int)
#     df_sem["specialized_recovery_required"] = desc.str.contains("heavy tow|crane|recovery|hazmat").astype(int)
#     df_sem["emergency_response_mentioned"] = desc.str.contains("police|patrol|responder|crew|fire dept").astype(int)
#     df_sem["uncertainty_language"] = desc.str.contains("possible|unconfirmed|reported|unknown").astype(int)
#     return df_sem


# def load_fallback_embedding_modality(df_struct: pd.DataFrame) -> pd.DataFrame:
#     """Fallback SVD vectorizer if embedding parquet file is missing."""
#     print("[Stage 3] Generating fallback 64-d TruncatedSVD narrative vector representations...")
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.decomposition import TruncatedSVD
    
#     texts = df_struct["Description"].fillna("traffic incident").astype(str).tolist()
#     vec = TfidfVectorizer(max_features=2000, stop_words="english")
#     svd = TruncatedSVD(n_components=64, random_state=42)
#     embeddings = svd.fit_transform(vec.fit_transform(texts)).astype(np.float32)
    
#     emb_dict = {f"d{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
#     return pd.DataFrame({"ID": df_struct["ID"].astype(str), **emb_dict})


# def main():
#     if not STRUCT_TABLE_PATH.exists():
#         raise FileNotFoundError(f"Missing {STRUCT_TABLE_PATH}. Run Phase 0 script first!")

#     print(f"[Stage 3] Loading structural feature table from {STRUCT_TABLE_PATH}")
#     df_struct = pd.read_parquet(STRUCT_TABLE_PATH)
#     df_struct["ID"] = df_struct["ID"].astype(str)

#     # Load Semantic Modality
#     if SEMANTIC_FULL_PATH.exists():
#         print(f"[Stage 3] Loading Semantic features from {SEMANTIC_FULL_PATH}")
#         df_sem = pd.read_parquet(SEMANTIC_FULL_PATH)
#     else:
#         df_sem = load_fallback_semantic_modality(df_struct)
#     df_sem["ID"] = df_sem["ID"].astype(str)

#     # Load Embedding Modality
#     if EMBEDDING_TABLE_PATH.exists():
#         print(f"[Stage 3] Loading Embeddings from {EMBEDDING_TABLE_PATH}")
#         df_emb = pd.read_parquet(EMBEDDING_TABLE_PATH)
#     else:
#         df_emb = load_fallback_embedding_modality(df_struct)
#     df_emb["ID"] = df_emb["ID"].astype(str)

#     # 1. Incident-ID Alignment (Inner Join across all 3 modalities)
#     print("[Stage 3] Executing Incident-ID Alignment (Inner Join across all modalities)...")
#     common_ids = set(df_struct["ID"]) & set(df_sem["ID"]) & set(df_emb["ID"])
    
#     df_struct = df_struct[df_struct["ID"].isin(common_ids)].drop_duplicates("ID").sort_values("ID").reset_index(drop=True)
#     df_sem = df_sem[df_sem["ID"].isin(common_ids)].drop_duplicates("ID").sort_values("ID").reset_index(drop=True)
#     df_emb = df_emb[df_emb["ID"].isin(common_ids)].drop_duplicates("ID").sort_values("ID").reset_index(drop=True)

#     # Keep crucial metadata columns explicitly attached to df_struct before prefixing
#     meta_cols = ["ID", "duration", "raw_duration_mins", "censor_flag", "split", "Description", "Start_Time", "End_Time"]
#     meta_df = df_struct[[c for c in meta_cols if c in df_struct.columns]].copy()

#     raw_struct_feature_cols = [c for c in df_struct.columns if c not in meta_cols]
    
#     # Encode categorical/boolean structural columns to numeric
#     for col in raw_struct_feature_cols:
#         if df_struct[col].dtype == "bool":
#             df_struct[col] = df_struct[col].astype(int)
#         elif df_struct[col].dtype == "object" or str(df_struct[col].dtype) == "category":
#             df_struct[col] = pd.factorize(df_struct[col])[0]
#         df_struct[col] = df_struct[col].fillna(df_struct[col].median())

#     # 2. Modality Prefix Assignment
#     print("[Stage 3] Assigning Modality Prefixes (struct_, sem_, emb_)...")
#     struct_prefix_map = {c: f"struct_{c}" for c in raw_struct_feature_cols}
#     sem_prefix_map = {c: f"sem_{c}" for c in df_sem.columns if c != "ID"}
#     emb_prefix_map = {c: f"emb_{c}" for c in df_emb.columns if c != "ID"}

#     df_struct_prefixed = df_struct[raw_struct_feature_cols].rename(columns=struct_prefix_map)
#     df_sem = df_sem.rename(columns=sem_prefix_map)
#     df_emb = df_emb.rename(columns=emb_prefix_map)

#     # Merge metadata + prefixed structural + semantic + embedding tables on ID
#     df_fused = pd.concat([
#         meta_df, 
#         df_struct_prefixed, 
#         df_sem.drop(columns=["ID"], errors="ignore"), 
#         df_emb.drop(columns=["ID"], errors="ignore")
#     ], axis=1)

#     # 3. Quality Scores (gate_q_tabular, gate_q_text_len, gate_q_semantic_density)
#     print("[Stage 3] Computing Quality Signals / Scores...")
#     prefixed_struct_cols = list(struct_prefix_map.values())
#     prefixed_sem_cols = list(sem_prefix_map.values())

#     df_fused["gate_q_tabular"] = df_fused[prefixed_struct_cols].notna().mean(axis=1)
    
#     char_len = df_fused["Description"].fillna("").astype(str).str.len()
#     log_len = np.log1p(char_len)
#     df_fused["gate_q_text_len"] = (log_len - log_len.min()) / (log_len.max() - log_len.min() + 1e-8)
    
#     df_fused["gate_q_semantic_density"] = df_fused[prefixed_sem_cols].fillna(0).mean(axis=1)

#     # Final save
#     df_fused.to_parquet(FUSED_TABLE_OUT_PATH, index=False)
#     print(f"[Success] Saved Aligned & Fused Multimodal Table ({len(df_fused):,} rows, {df_fused.shape[1]} columns) to {FUSED_TABLE_OUT_PATH}")


# if __name__ == "__main__":
#     main()

## --------------------FINAL CODE FOR 10K ROWS --------------------------

# import numpy as np
# import pandas as pd
# from pathlib import Path

# STRUCT_TABLE_PATH = Path("data/processed/structural_feature_table.parquet")
# SEMANTIC_FULL_PATH = Path("data/processed/us_accidents_semantic_features.parquet")
# EMBEDDING_TABLE_PATH = Path("data/processed/us_accidents_embeddings.parquet")
# FUSED_TABLE_OUT_PATH = Path("data/processed/fused_multimodal_table.parquet")


# def enrich_and_fix_constant_semantic_tags(df_sem: pd.DataFrame, descriptions: pd.Series) -> pd.DataFrame:
#     """
#     Ensures zero-variance semantic columns in the 10k sample are populated with operational
#     lexical signals from the dispatcher text and constructs Stage 2 Semantic Feature Groups.
#     """
#     desc = descriptions.fillna("").astype(str).str.lower()
#     df_sem = df_sem.copy()

#     # Fix the 3 near-constant columns identified in audit so they carry genuine signal
#     df_sem["hazardous_material"] = (
#         df_sem.get("hazardous_material", pd.Series(0, index=df_sem.index)).fillna(0).astype(int) |
#         desc.str.contains("hazmat|fuel|chemical|spill|leak|tanker|flammable|gas|toxic").astype(int)
#     )
#     df_sem["specialized_recovery_required"] = (
#         df_sem.get("specialized_recovery_required", pd.Series(0, index=df_sem.index)).fillna(0).astype(int) |
#         desc.str.contains("overturned|rollover|jackknife|semi|truck|tractor|trailer|ditched|embankment|heavy|crane|tow").astype(int)
#     )
#     df_sem["uncertainty_language"] = (
#         df_sem.get("uncertainty_language", pd.Series(0, index=df_sem.index)).fillna(0).astype(int) |
#         desc.str.contains("possible|reported|unconfirmed|unknown|expect|approx|delayed|caution|alert").astype(int)
#     )

#     # Operational blockage & highway corridor cues from text
#     df_sem["explicit_delay_warning"] = desc.str.contains("delay|delays|queued|backed up|congestion|slow|stop and go").astype(int)
#     df_sem["multi_lane_impact"] = desc.str.contains("two lanes|three lanes|all lanes|multiple lanes|both directions|closed").astype(int)
#     df_sem["interstate_corridor"] = desc.str.contains(r"\bi-\d+|\binterstate\b|\bfreeway\b|\bpkwy\b|\bturnpike\b", regex=True).astype(int)

#     # Encode string incident_type if object/categorical
#     if "incident_type" in df_sem.columns and (df_sem["incident_type"].dtype == "object" or str(df_sem["incident_type"].dtype) == "category"):
#         df_sem["incident_type"] = pd.factorize(df_sem["incident_type"])[0]

#     # Stage 2 Architecture Box: "Semantic feature groups (Incident-type tags | Severity-related tags)"
#     type_cols = [c for c in ["heavy_vehicle", "rollover", "debris", "disabled_vehicle", "spill", "fire"] if c in df_sem.columns]
#     sev_cols = [c for c in [
#         "lanes_blocked", "full_road_blockage", "ramp_involvement", "injury",
#         "hazardous_material", "specialized_recovery_required", "explicit_delay_warning", "multi_lane_impact"
#     ] if c in df_sem.columns]

#     for c in type_cols + sev_cols:
#         df_sem[c] = pd.to_numeric(df_sem[c], errors="coerce").fillna(0).astype(int)

#     df_sem["incident_type_group_score"] = df_sem[type_cols].sum(axis=1)
#     df_sem["severity_group_score"] = df_sem[sev_cols].sum(axis=1)

#     return df_sem


# def main():
#     if not STRUCT_TABLE_PATH.exists():
#         raise FileNotFoundError(f"Missing {STRUCT_TABLE_PATH}. Run Phase 0 script first!")

#     print(f"[Stage 3] Loading structural feature table from {STRUCT_TABLE_PATH}")
#     df_struct = pd.read_parquet(STRUCT_TABLE_PATH)
#     df_struct["ID"] = df_struct["ID"].astype(str)

#     print(f"[Stage 3] Loading Semantic features from {SEMANTIC_FULL_PATH}")
#     df_sem = pd.read_parquet(SEMANTIC_FULL_PATH)
#     df_sem["ID"] = df_sem["ID"].astype(str)

#     print(f"[Stage 3] Loading Embeddings from {EMBEDDING_TABLE_PATH}")
#     df_emb = pd.read_parquet(EMBEDDING_TABLE_PATH)
#     df_emb["ID"] = df_emb["ID"].astype(str)

#     # 1. Incident-ID Alignment (Strict Inner Join across all 3 modalities)
#     print("[Stage 3] Executing Incident-ID Alignment (Inner Join across all modalities)...")
#     common_ids = set(df_struct["ID"]) & set(df_sem["ID"]) & set(df_emb["ID"])

#     df_struct = df_struct[df_struct["ID"].isin(common_ids)].drop_duplicates("ID").sort_values("ID").reset_index(drop=True)
#     df_sem = df_sem[df_sem["ID"].isin(common_ids)].drop_duplicates("ID").sort_values("ID").reset_index(drop=True)
#     df_emb = df_emb[df_emb["ID"].isin(common_ids)].drop_duplicates("ID").sort_values("ID").reset_index(drop=True)

#     # Enrich semantic tags and build Stage 2 Semantic Feature Groups
#     df_sem = enrich_and_fix_constant_semantic_tags(df_sem, df_struct["Description"])

#     # Unpack embedding column if stored as a 1D array/list column (e.g. 'embedding')
#     emb_non_id = [c for c in df_emb.columns if c != "ID"]
#     if len(emb_non_id) == 1 and isinstance(df_emb[emb_non_id[0]].iloc[0], (list, np.ndarray)):
#         raw_vecs = np.vstack(df_emb[emb_non_id[0]].values).astype(np.float32)
#         emb_expanded = pd.DataFrame(raw_vecs, columns=[f"d{i}" for i in range(raw_vecs.shape[1])])
#         df_emb = pd.concat([df_emb[["ID"]], emb_expanded], axis=1)

#     # Preserve pre-imputation tabular completeness score
#     raw_tab_q = df_struct["raw_tabular_completeness"].values if "raw_tabular_completeness" in df_struct.columns else np.ones(len(df_struct))

#     meta_cols = ["ID", "duration", "raw_duration_mins", "censor_flag", "split", "Description", "Start_Time", "End_Time", "raw_tabular_completeness"]
#     meta_df = df_struct[[c for c in meta_cols if c in df_struct.columns]].copy()

#     raw_struct_feature_cols = [c for c in df_struct.columns if c not in meta_cols and c != "Source"]

#     # Target-free frequency encoding for high-cardinality location strings (Street, City, County, State)
#     # Frequency encoding captures corridor event density without arbitrary alphabetical factorize() noise
#     for col in raw_struct_feature_cols:
#         if df_struct[col].dtype == "bool":
#             df_struct[col] = df_struct[col].astype(int)
#         elif df_struct[col].dtype == "object" or str(df_struct[col].dtype) == "category":
#             if col in ["Street", "City", "County", "State", "Weather_Condition"]:
#                 freq = df_struct[col].value_counts(normalize=True)
#                 df_struct[col] = df_struct[col].map(freq).fillna(0.0).astype(float)
#             else:
#                 df_struct[col] = pd.factorize(df_struct[col])[0]
#         df_struct[col] = pd.to_numeric(df_struct[col], errors="coerce")
#         df_struct[col] = df_struct[col].fillna(df_struct[col].median())

#     # 2. Modality Prefix Assignment (struct_, sem_, emb_)
#     print("[Stage 3] Assigning Modality Prefixes (struct_, sem_, emb_)...")
#     struct_prefix_map = {c: f"struct_{c}" for c in raw_struct_feature_cols}
#     sem_prefix_map = {c: f"sem_{c}" for c in df_sem.columns if c != "ID"}
#     emb_prefix_map = {c: f"emb_{c}" for c in df_emb.columns if c != "ID"}

#     df_struct_prefixed = df_struct[raw_struct_feature_cols].rename(columns=struct_prefix_map)
#     df_sem_prefixed = df_sem.drop(columns=["ID"], errors="ignore").rename(columns=sem_prefix_map)
#     df_emb_prefixed = df_emb.drop(columns=["ID"], errors="ignore").rename(columns=emb_prefix_map)

#     df_fused = pd.concat([meta_df, df_struct_prefixed, df_sem_prefixed, df_emb_prefixed], axis=1)

#     # 3. Quality Scores (Non-constant, computed from raw pre-imputed completeness, log text length, and active tag fraction)
#     print("[Stage 3] Computing Non-Constant Quality Signals (gate_q_tabular, gate_q_text_len, gate_q_semantic_density)...")
#     df_fused["gate_q_tabular"] = raw_tab_q

#     char_len = df_fused["Description"].fillna("").astype(str).str.len()
#     log_len = np.log1p(char_len)
#     df_fused["gate_q_text_len"] = (log_len - log_len.min()) / (log_len.max() - log_len.min() + 1e-8)

#     binary_sem_cols = [c for c in df_sem_prefixed.columns if "score" not in c and "incident_type" not in c]
#     df_fused["gate_q_semantic_density"] = df_fused[binary_sem_cols].fillna(0).mean(axis=1)

#     # Sort strictly by Start_Time and enforce exact 70/10/10/10 chronological split on the aligned multimodal table
#     df_fused = df_fused.sort_values("Start_Time").reset_index(drop=True)
#     n = len(df_fused)
#     t_cut, v_cut, c_cut = int(n * 0.70), int(n * 0.80), int(n * 0.90)
#     splits = np.empty(n, dtype=object)
#     splits[:t_cut] = "train"
#     splits[t_cut:v_cut] = "val"
#     splits[v_cut:c_cut] = "cal"
#     splits[c_cut:] = "test"
#     df_fused["split"] = splits

#     df_fused.to_parquet(FUSED_TABLE_OUT_PATH, index=False)
#     print(f"[Success] Saved Aligned & Fused Multimodal Table ({len(df_fused):,} rows, {df_fused.shape[1]} columns) to {FUSED_TABLE_OUT_PATH}")


# if __name__ == "__main__":
#     main()


#  ---------------------CODE FOR 85K ROWS ----------------------------------


import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

STRUCT_TABLE_PATH = Path("data/processed/structural_feature_table.parquet")
SEMANTIC_85K_PARQUET = Path("data/processed/us_accidents_semantic_features_85k.parquet")
SEMANTIC_85K_CSV = Path("data/processed/us_accidents_semantic_features_85k.csv")
EMBEDDING_85K_PATH = Path("data/processed/us_accidents_embeddings_85k.parquet")
EMBEDDING_OLD_PATH = Path("data/processed/us_accidents_embeddings.parquet")
FUSED_TABLE_OUT_PATH = Path("data/processed/fused_multimodal_table.parquet")


def build_85k_semantic_groups(df_sem: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the 45-column 85k semantic schema and builds Stage 2 Semantic Feature Groups
    (Incident-type tags & Severity-related tags).
    """
    df_sem = df_sem.copy()
    non_id_cols = [c for c in df_sem.columns if c != "ID"]

    for c in non_id_cols:
        if df_sem[c].dtype == "bool":
            df_sem[c] = df_sem[c].astype(int)
        elif df_sem[c].dtype == "object" or str(df_sem[c].dtype) == "category":
            s_lower = df_sem[c].astype(str).str.lower().str.strip()
            mapped = s_lower.map({"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0, "1.0": 1, "0.0": 0})
            if mapped.notna().mean() > 0.5:
                df_sem[c] = mapped.fillna(0).astype(int)
            else:
                df_sem[c] = pd.factorize(df_sem[c])[0]
        else:
            df_sem[c] = pd.to_numeric(df_sem[c], errors="coerce").fillna(0)

    type_keywords = ["incident_type__", "heavy_vehicle", "rollover", "debris", "fire", "disabled_vehicle", "spill"]
    type_cols = [c for c in non_id_cols if any(k in c for k in type_keywords) and "unknown" not in c]

    sev_keywords = [
        "injury", "lanes_blocked", "full_road_blockage", "ramp_involvement",
        "hazardous_material", "specialized_recovery", "emergency_response"
    ]
    sev_cols = [c for c in non_id_cols if any(k in c for k in sev_keywords) and "unknown" not in c and c not in type_cols]

    df_sem["incident_type_group_score"] = df_sem[type_cols].sum(axis=1) if type_cols else 0
    df_sem["severity_group_score"] = df_sem[sev_cols].sum(axis=1) if sev_cols else 0

    unknown_cols = [c for c in non_id_cols if "unknown" in c or "uncertainty" in c]
    df_sem["uncertainty_group_score"] = df_sem[unknown_cols].sum(axis=1) if unknown_cols else 0

    return df_sem


def load_or_build_85k_embeddings(df_struct_aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures embeddings cover all 85k aligned IDs instead of dropping 83k rows
    due to intersecting with the old 10k embedding file.
    """
    target_ids = set(df_struct_aligned["ID"])

    for cand_path in [EMBEDDING_85K_PATH, EMBEDDING_OLD_PATH]:
        if cand_path.exists():
            df_emb = pd.read_parquet(cand_path)
            df_emb["ID"] = df_emb["ID"].astype(str)
            overlap = len(target_ids & set(df_emb["ID"]))
            if overlap >= int(0.85 * len(target_ids)):
                print(f"[Stage 3] Loaded matching 85k embeddings from {cand_path} ({overlap:,} overlapping IDs).")
                emb_non_id = [c for c in df_emb.columns if c != "ID"]
                if len(emb_non_id) == 1 and isinstance(df_emb[emb_non_id[0]].iloc[0], (list, np.ndarray)):
                    raw_vecs = np.vstack(df_emb[emb_non_id[0]].values).astype(np.float32)
                    emb_expanded = pd.DataFrame(raw_vecs, columns=[f"d{i}" for i in range(raw_vecs.shape[1])])
                    df_emb = pd.concat([df_emb[["ID"]], emb_expanded], axis=1)
                return df_emb

    print(f"[Stage 3] Existing embedding file only matched a small subset. Generating 64-d Narrative Embeddings for all {len(df_struct_aligned):,} rows...")
    texts = df_struct_aligned["Description"].fillna("traffic incident").astype(str).tolist()
    vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    X_tfidf = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=64, random_state=42)
    emb_mat = svd.fit_transform(X_tfidf).astype(np.float32)

    df_emb = pd.DataFrame(emb_mat, columns=[f"d{i}" for i in range(emb_mat.shape[1])])
    df_emb.insert(0, "ID", df_struct_aligned["ID"].values)
    df_emb.to_parquet(EMBEDDING_85K_PATH, index=False)
    print(f"[Stage 3] Saved 85k embeddings ({len(df_emb):,} rows) to {EMBEDDING_85K_PATH}")
    return df_emb


def main():
    if not STRUCT_TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing {STRUCT_TABLE_PATH}. Run Phase 0 script first!")

    print(f"[Stage 3] Loading structural feature table from {STRUCT_TABLE_PATH}...")
    df_struct = pd.read_parquet(STRUCT_TABLE_PATH)
    df_struct["ID"] = df_struct["ID"].astype(str)

    if SEMANTIC_85K_PARQUET.exists():
        print(f"[Stage 3] Loading 85k Semantic features from {SEMANTIC_85K_PARQUET}...")
        df_sem = pd.read_parquet(SEMANTIC_85K_PARQUET)
    elif SEMANTIC_85K_CSV.exists():
        print(f"[Stage 3] Loading 85k Semantic features from {SEMANTIC_85K_CSV}...")
        df_sem = pd.read_csv(SEMANTIC_85K_CSV)
    else:
        raise FileNotFoundError("Could not find us_accidents_semantic_features_85k.parquet or .csv in data/processed/!")

    df_sem["ID"] = df_sem["ID"].astype(str)

    # First align Structural and 85k Semantic tables on ID
    common_struct_sem = set(df_struct["ID"]) & set(df_sem["ID"])
    print(f"[Stage 3] Matched {len(common_struct_sem):,} incidents between Structural and 85k Semantic tables.")

    df_struct = df_struct[df_struct["ID"].isin(common_struct_sem)].drop_duplicates("ID").sort_values("ID").reset_index(drop=True)
    df_sem = df_sem[df_sem["ID"].isin(common_struct_sem)].drop_duplicates("ID").sort_values("ID").reset_index(drop=True)

    # Load or generate 85k embeddings covering all matched IDs
    df_emb = load_or_build_85k_embeddings(df_struct)

    common_ids = set(df_struct["ID"]) & set(df_sem["ID"]) & set(df_emb["ID"])
    df_struct = df_struct[df_struct["ID"].isin(common_ids)].drop_duplicates("ID").sort_values("ID").reset_index(drop=True)
    df_sem = df_sem[df_sem["ID"].isin(common_ids)].drop_duplicates("ID").sort_values("ID").reset_index(drop=True)
    df_emb = df_emb[df_emb["ID"].isin(common_ids)].drop_duplicates("ID").sort_values("ID").reset_index(drop=True)

    df_sem = build_85k_semantic_groups(df_sem)

    raw_tab_q = df_struct["raw_tabular_completeness"].values if "raw_tabular_completeness" in df_struct.columns else np.ones(len(df_struct))
    meta_cols = ["ID", "duration", "raw_duration_mins", "censor_flag", "split", "Description", "Start_Time", "End_Time", "raw_tabular_completeness"]
    meta_df = df_struct[[c for c in meta_cols if c in df_struct.columns]].copy()
    raw_struct_feature_cols = [c for c in df_struct.columns if c not in meta_cols and c != "Source"]

    for col in raw_struct_feature_cols:
        if df_struct[col].dtype == "bool":
            df_struct[col] = df_struct[col].astype(int)
        elif df_struct[col].dtype == "object" or str(df_struct[col].dtype) == "category":
            if col in ["Street", "City", "County", "State", "Weather_Condition"]:
                freq = df_struct[col].value_counts(normalize=True)
                df_struct[col] = df_struct[col].map(freq).fillna(0.0).astype(float)
            else:
                df_struct[col] = pd.factorize(df_struct[col])[0]
        df_struct[col] = pd.to_numeric(df_struct[col], errors="coerce").fillna(df_struct[col].median())

    print("[Stage 3] Assigning Modality Prefixes (struct_, sem_, emb_)...")
    df_struct_prefixed = df_struct[raw_struct_feature_cols].rename(columns={c: f"struct_{c}" for c in raw_struct_feature_cols})
    df_sem_prefixed = df_sem.drop(columns=["ID"], errors="ignore").rename(columns={c: f"sem_{c}" for c in df_sem.columns if c != "ID"})
    df_emb_prefixed = df_emb.drop(columns=["ID"], errors="ignore").rename(columns={c: f"emb_{c}" for c in df_emb.columns if c != "ID"})

    df_fused = pd.concat([meta_df, df_struct_prefixed, df_sem_prefixed, df_emb_prefixed], axis=1)

    print("[Stage 3] Computing Quality Signals (gate_q_tabular, gate_q_text_len, gate_q_semantic_density)...")
    df_fused["gate_q_tabular"] = raw_tab_q
    log_len = np.log1p(df_fused["Description"].fillna("").astype(str).str.len())
    df_fused["gate_q_text_len"] = (log_len - log_len.min()) / (log_len.max() - log_len.min() + 1e-8)

    sem_num_df = df_sem_prefixed.apply(pd.to_numeric, errors="coerce").fillna(0)
    binary_sem_cols = [c for c in sem_num_df.columns if "unknown" not in c and "score" not in c]
    df_fused["gate_q_semantic_density"] = sem_num_df[binary_sem_cols].mean(axis=1) if binary_sem_cols else 0.5

    # Strict 70/10/10/10 chronological split on the aligned 85k dataset
    df_fused = df_fused.sort_values("Start_Time").reset_index(drop=True)
    n = len(df_fused)
    t_cut, v_cut, c_cut = int(n * 0.70), int(n * 0.80), int(n * 0.90)
    splits = np.empty(n, dtype=object)
    splits[:t_cut] = "train"
    splits[t_cut:v_cut] = "val"
    splits[v_cut:c_cut] = "cal"
    splits[c_cut:] = "test"
    df_fused["split"] = splits

    df_fused.to_parquet(FUSED_TABLE_OUT_PATH, index=False)
    print(f"[Success] Saved 85k Fused Multimodal Table ({len(df_fused):,} rows, {df_fused.shape[1]} columns) to {FUSED_TABLE_OUT_PATH}")


if __name__ == "__main__":
    main()