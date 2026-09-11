# src/spatiotemporal_features.py
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def engineer_survival_and_causality(input_path, output_path, radius_miles=2.0, time_buffer_mins=30):
    logging.info(f"Loading Phase 0 processed data from {input_path}...")
    df = pd.read_parquet(input_path)

    # Ensure timestamps are datetime objects
    df['Start_Time_parsed'] = pd.to_datetime(df['Start_Time_parsed'])
    df['End_Time_parsed'] = pd.to_datetime(df['End_Time_parsed'])
    
    # Sort strictly chronologically (Crucial for causality)
    df = df.sort_values('Start_Time_parsed').reset_index(drop=True)

    # =====================================================================
    # 1. CENSORING FLAG (For XGBoost-AFT Survival Analysis)
    # =====================================================================
    logging.info("Engineering Censoring Flag for Survival Analysis...")
    
    MAX_DURATION_MINS = 24 * 60
    
    # Initialize all as observed (1)
    df['event_observed'] = 1
    
    # Flag extreme outliers as censored (0) using the correct duration column
    df.loc[df['reported_duration_minutes'] > MAX_DURATION_MINS, 'event_observed'] = 0
    
    # Cap the duration for the censored events
    df.loc[df['reported_duration_minutes'] > MAX_DURATION_MINS, 'reported_duration_minutes'] = MAX_DURATION_MINS

    # =====================================================================
    # 2. SECONDARY ACCIDENT CAUSALITY (Spatio-Temporal Shadow)
    # =====================================================================
    logging.info("Building Spatial Tree for Secondary Accident Detection...")
    
    # Convert Lat/Lng to Radians for the Haversine formula
    earth_radius_miles = 3958.8
    radius_radians = radius_miles / earth_radius_miles
    
    coords_radians = np.radians(df[['Start_Lat', 'Start_Lng']].values)
    
    # Build a BallTree for ultra-fast spatial querying
    tree = BallTree(coords_radians, metric='haversine')
    
    logging.info(f"Querying accidents within a {radius_miles}-mile radius...")
    # Find all accidents within the radius for every accident
    indices_within_radius = tree.query_radius(coords_radians, r=radius_radians)
    
    logging.info("Evaluating Temporal Shadows (Optimized NumPy processing)...")
    
    # OPTIMIZATION: Extract to raw numpy arrays (Unix epoch seconds) to bypass Pandas loop overhead
    # This reduces processing time from hours to seconds and prevents the DeprecationWarning.
    start_times = df['Start_Time_parsed'].astype('int64').values // 10**9
    end_times = df['End_Time_parsed'].astype('int64').values // 10**9
    buffer_seconds = time_buffer_mins * 60
    
    # Extract IDs to a fast numpy array
    ids = df['ID'].values
    
    # Initialize fast output arrays
    is_secondary = np.zeros(len(df), dtype=int)
    primary_incident_id = np.full(len(df), None, dtype=object)
    
    # Extremely fast pure Python/NumPy loop
    for i, neighbors in enumerate(indices_within_radius):
        primary_start = start_times[i]
        primary_end = end_times[i] + buffer_seconds
        
        for neighbor_idx in neighbors:
            if neighbor_idx == i:
                continue # Skip itself
                
            neighbor_start = start_times[neighbor_idx]
            
            # CONDITION: If the neighbor started AFTER the primary crash, 
            # but BEFORE the primary crash cleared (plus buffer)
            if primary_start < neighbor_start <= primary_end:
                is_secondary[neighbor_idx] = 1
                primary_incident_id[neighbor_idx] = ids[i]

    # Map the results back to the Pandas DataFrame
    df['is_secondary_crash'] = is_secondary
    df['primary_incident_id'] = primary_incident_id

    logging.info(f"Identified {df['is_secondary_crash'].sum()} secondary crashes.")

    # =====================================================================
    # 3. SAVE ENRICHED DATA
    # =====================================================================
    logging.info(f"Saving enriched dataset to {output_path}...")
    df.to_parquet(output_path, index=False)
    logging.info("Preprocessing step complete. Ready for Reliability Fusion.")

if __name__ == "__main__":
    INPUT_FILE = "data/processed/us_accidents_phase0_processed.parquet"
    OUTPUT_FILE = "data/processed/us_accidents_spatiotemporal.parquet"
    
    engineer_survival_and_causality(
        input_path=INPUT_FILE, 
        output_path=OUTPUT_FILE, 
        radius_miles=2.0, 
        time_buffer_mins=30
    )