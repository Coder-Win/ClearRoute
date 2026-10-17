import pandas as pd
from pathlib import Path

# Read the CSV your main script already saved
csv_path = Path("reports/phase0/ablation_metrics_comprehensive_85k.csv")
df = pd.read_csv(csv_path)

# Filter for just the Soft MoE AFT architectures in Tiers 4 and 5
compare_df = df[
    (df["Architecture"] == "4. Soft MoE AFT") & 
    (df["Dataset"].isin(["4. Dumb Join", "5. Gated Fusion"]))
]

print("\n" + "=" * 82)
print(" 📊 PERFORMANCE GAP: DUMB JOIN VS. GATED FUSION")
print("=" * 82)
print(f"{'Architecture':<18} {'Dataset':<17} {'MedAE (Mins)':>13} {'MAE (Mins)':>12} {'RMSE':>10} {'MAPE (%)':>10}")
print("-" * 82)
for _, row in compare_df.iterrows():
    print(
        f"{row['Architecture']:<18} {row['Dataset']:<17} "
        f"{row['MedAE (Mins)']:>13.2f} {row['MAE (Mins)']:>12.2f} "
        f"{row['RMSE']:>10.2f} {row['MAPE (%)']:>10.2f}"
    )
print("=" * 82 + "\n")