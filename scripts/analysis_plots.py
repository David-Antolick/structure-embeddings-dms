import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from Bio.Align import substitution_matrices
import os

# Load data
df = pd.read_csv("data/processed/tem1_dms_embeddings.csv")

# Extract embedding matrix
embedding_cols = [col for col in df.columns if col.isdigit()]
X = df[embedding_cols].values
y = df["score"].values

# (A) Correlation: L2 norm vs score
df["embedding_norm"] = np.linalg.norm(X, axis=1)
corr, pval = spearmanr(df["embedding_norm"], df["score"])
print(f"[A] Spearman correlation (embedding norm vs score): {corr:.3f}, p = {pval:.3e}")

# (B) ML Model with embeddings
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
r2_emb = r2_score(y_test, y_pred)
print(f"[B] Random Forest R² (embeddings): {r2_emb:.3f}")

# (C) Add BLOSUM62 substitution score
blosum62 = substitution_matrices.load("BLOSUM62")

def get_blosum_score(wt, mut):
    try:
        return blosum62[(wt, mut)]
    except KeyError:
        try:
            return blosum62[(mut, wt)]
        except KeyError:
            return 0

df["blosum62"] = df.apply(lambda row: get_blosum_score(row["wt_residue"], row["mutant_residue"]), axis=1)

# Correlation of BLOSUM62
bl_corr, bl_pval = spearmanr(df["blosum62"], df["score"])
print(f"[C] Spearman correlation (BLOSUM62 vs score): {bl_corr:.3f}, p = {bl_pval:.3e}")

# R² with both embeddings + BLOSUM62
X_combo = np.hstack([X, df[["blosum62"]].values])
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_combo, y, test_size=0.2, random_state=42)
rf_combo = RandomForestRegressor(n_estimators=100, random_state=42)
rf_combo.fit(X_train_c, y_train_c)
y_pred_c = rf_combo.predict(X_test_c)
r2_combo = r2_score(y_test_c, y_pred_c)
print(f"[C] Random Forest R² (embeddings + BLOSUM62): {r2_combo:.3f}")

# Create plots directory
os.makedirs("plots", exist_ok=True)

# Plot 1: Embedding norm vs score (with regression line)
plt.figure(figsize=(6, 4))
sns.regplot(x="embedding_norm", y="score", data=df, scatter_kws={'alpha': 0.1})
plt.xlabel("Embedding L2 Norm")
plt.ylabel("Mutation Fitness Score")
plt.title("Embedding Norm vs Mutation Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/embedding_norm_vs_score.png", dpi=300)
plt.close()

# Bonus: Binned average score vs embedding norm
df["norm_bin"] = pd.cut(df["embedding_norm"], bins=20)
binned = df.groupby("norm_bin")["score"].mean().reset_index()
plt.figure(figsize=(6, 4))
sns.barplot(x="norm_bin", y="score", data=binned, color="steelblue")
plt.xticks(rotation=90)
plt.xlabel("Binned Embedding Norm")
plt.ylabel("Avg Mutation Score")
plt.title("Avg Mutation Score by Embedding Norm Bin")
plt.tight_layout()
plt.savefig("plots/embedding_norm_binned_avg.png", dpi=300)
plt.close()

# Plot 2: BLOSUM62 score vs score (with jitter)
plt.figure(figsize=(6, 4))
sns.stripplot(x="blosum62", y="score", data=df, alpha=0.2, jitter=0.25, color="orange")
plt.xlabel("BLOSUM62 Substitution Score")
plt.ylabel("Mutation Fitness Score")
plt.title("BLOSUM62 Score vs Mutation Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/blosum62_vs_score_jittered.png", dpi=300)
plt.close()

# Plot 3: R² score comparison
plt.figure(figsize=(6, 4))
bars = ["ESM Embeddings", "BLOSUM62 Only", "Embeddings + BLOSUM62"]
r2_scores = [r2_emb, 0, r2_combo]
plt.bar(bars, r2_scores, color=["blue", "gray", "green"])
plt.ylabel("R² Score")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("plots/model_r2_comparison.png", dpi=300)
plt.close()
