# CLEAR-ROUTE

Traffic-incident duration prediction (Phase 1: stages 1-5, Phase 2: calibration, reliability gate, routing).
Datasets are not in the repo (`data/` is gitignored). Put them in `data/raw/us_accidents/` and `data/raw/iowa/` (see `configs/phase0_config.yaml`).

## Setup

    pip install -r requirements.txt

The LLM extraction step needs a local Ollama server. `auto_prelabel_pilot.py` needs an OpenAI key.

## Run order (run from the repo root)

Phase 1

1. `python src/phase0_audit.py` - audit, writes `data/processed/*_phase0_processed.parquet` and `reports/phase0/`
2. `python src/auto_classify_features.py` then `python src/validate_feature_sets.py` - feature sets into `configs/generated/`
3. `python src/spatiotemporal_features.py` - `us_accidents_spatiotemporal.parquet`
4. `python src/semantic_extraction/generate_embeddings.py` - MiniLM embeddings (10k sample)
5. `python src/semantic_extraction/batch_llm_extraction.py` - LLM semantic fields for the embedding sample
6. `python src/convert_to_csv.py` - converts `us_accidents_semantic_features.parquet` to CSV (the fusion script reads CSV)
7. `python src/reliability_fusion2.py` - quality signals and fusion, writes the `*_v2.csv` datasets
8. `python src/train_eval_baselines2.py` - modality ablation
9. `python src/moe_residual_gated_with_aft.py` - stacking MoE experiment (standalone, saves no model)
10. `python src/generate_merged_predictions.py` - exporter, writes `data/processed/merged_duration_outputs.csv`

Phase 2 (reads the exporter output)

11. `python src/conformal_calibration.py`
12. `python src/traditional_hbdm_baseline.py` (optional baseline, read by the gate)
13. `python src/reliability_gate.py` - writes `reports/phase1/final_gated_predictions.csv`
14. `python src/treeshap_analysis.py`, `python src/iowa_transfer_eval.py`

Optional: `src/create_annotation_sample.py`, `src/semantic_extraction/{keyword_baseline,tfidf_baseline,compare_baselines}.py`, `src/show_predictions_v2.py`.

## Notes

- Generated outputs (`reports/phase0/`, `configs/generated/`, `final_gated_predictions.csv`) are gitignored; regenerate them with the steps above.
- Known issue: nothing in the repo writes `us_accidents_embeddings.csv`, which `reliability_fusion2.py` reads. Convert the embeddings parquet to CSV before step 7.
