import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

#  Load data 
df = pd.read_csv("data/processed/tem1_dms.csv")  # DMS scores
struct_df = pd.read_csv("data/processed/mutant_vs_wt_rmsd_plddt.csv")

#  Extract fitness scores from mutant name
def extract_score(mutant_name):
    try:
        parts = mutant_name.split("_")
        for part in parts:
            if part.startswith("score"):
                return float(part.replace("score", "").replace(".result", "")) / 1000
    except:
        return None

struct_df["score"] = struct_df["mutant"].apply(extract_score)

#  Correlation
print("Spearman: RMSD vs Score:", spearmanr(struct_df["rmsd_CA"], struct_df["score"]))
print("Spearman: pLDDT vs Score:", spearmanr(struct_df["plddt_mut"], struct_df["score"]))

#  Plot 1: Scatter fitness vs RMSD
plt.figure()
plt.scatter(struct_df["score"], struct_df["rmsd_CA"])
plt.xlabel("Mutation Fitness Score")
plt.ylabel("RMSD (Cα)")
plt.title("Fitness vs RMSD")
plt.tight_layout()
plt.savefig("plots/fitness_vs_rmsd.png")

#  Plot 2: Scatter fitness vs pLDDT
plt.figure()
plt.scatter(struct_df["score"], struct_df["plddt_mut"])
plt.xlabel("Mutation Fitness Score")
plt.ylabel("pLDDT (mut site)")
plt.title("Fitness vs pLDDT")
plt.tight_layout()
plt.savefig("plots/fitness_vs_plddt.png")

#  Plot 3: Bar plot of RMSD and pLDDT
plt.figure(figsize=(10, 4))
sorted_df = struct_df.sort_values("score")
plt.bar(sorted_df["mutant"], sorted_df["rmsd_CA"], label="RMSD")
plt.xticks(rotation=45, ha='right')
plt.ylabel("RMSD (Cα)")
plt.title("RMSD Across Mutants")
plt.tight_layout()
plt.savefig("plots/rmsd_across_mutants.png")
