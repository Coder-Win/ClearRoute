import pandas as pd
import json
import os
import time
from openai import OpenAI
from pathlib import Path

# Setup paths
INPUT_CSV = Path("annotations/semantic_annotation_pilot.csv")
OUTPUT_CSV = Path("annotations/semantic_annotation_pilot_labelled.csv")
SCHEMA_PATH = Path("schemas/semantic_schema.json")

# Initialize LLM Client pointing to Google's OpenAI-compatible endpoint
client = OpenAI(
    api_key=os.environ.get("AIzaSyBLO-QLufm4gTNJTjUZ14aMzT7sjiy_31c"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

def query_llm_for_labels(description: str, schema: dict) -> dict:
    prompt = f"""
    You are a highly accurate traffic incident data extraction AI.
    Read the following incident description and extract the operational features according to the JSON schema below.
    
    Rules:
    1. Output strictly valid JSON.
    2. Use ONLY "true", "false", or "unknown" for boolean fields.
    3. Do not infer or guess. If it is not explicitly stated or heavily implied, output "unknown".
    
    Schema:
    {json.dumps(schema['fields'], indent=2)}
    
    Incident Description:
    "{description}"
    """
    
    try:
        response = client.chat.completions.create(
            model="gemini-1.5-flash", # Google's fast, free-tier eligible model
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You output strict, validated JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return {}

def main():
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    schema = load_schema()
    
    # Process each row
    for index, row in df.iterrows():
        description = row['Description']
        if pd.isna(description) or str(description).strip() == "":
            continue
            
        print(f"Processing row {index + 1}/{len(df)}...")
        llm_output = query_llm_for_labels(str(description), schema)
        
        # Map JSON output back to the annotation columns
        df.at[index, 'ann_incident_type'] = llm_output.get('incident_type', 'unknown')
        df.at[index, 'ann_heavy_vehicle'] = llm_output.get('heavy_vehicle', 'unknown')
        df.at[index, 'ann_rollover'] = llm_output.get('rollover', 'unknown')
        df.at[index, 'ann_spill'] = llm_output.get('spill', 'unknown')
        df.at[index, 'ann_fire'] = llm_output.get('fire', 'unknown')
        df.at[index, 'ann_debris'] = llm_output.get('debris', 'unknown')
        df.at[index, 'ann_injury'] = llm_output.get('injury', 'unknown')
        df.at[index, 'ann_lanes_blocked'] = llm_output.get('lanes_blocked', 'unknown')
        df.at[index, 'ann_full_road_blockage'] = llm_output.get('full_road_blockage', 'unknown')
        df.at[index, 'ann_ramp_involvement'] = llm_output.get('ramp_involvement', 'unknown')
        df.at[index, 'ann_disabled_vehicle'] = llm_output.get('disabled_vehicle', 'unknown')
        df.at[index, 'ann_hazardous_material'] = llm_output.get('hazardous_material', 'unknown')
        df.at[index, 'ann_specialized_recovery_required'] = llm_output.get('specialized_recovery_required', 'unknown')
        df.at[index, 'ann_emergency_response_mentioned'] = llm_output.get('emergency_response_mentioned', 'unknown')
        df.at[index, 'ann_uncertainty_language'] = llm_output.get('uncertainty_language', 'unknown')
        df.at[index, 'annotation_status'] = 'PRE-LABELLED_SILVER'
        
        # 4-second delay to comply with the 15 requests per minute free tier limit
        time.sleep(4)
        
    # Save the silver standard dataset
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[COMPLETE] Silver standard labels saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()