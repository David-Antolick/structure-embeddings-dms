import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from Bio.Align import substitution_matrices

#  Load merged DMS + embeddings data 
df = pd.read_csv("data/processed/tem1_dms_embeddings.csv")

#  A. Correlation: L2 norm of WT embedding vs mutation score 
embedding_cols = df.columns[4:]
embedding_matrix = df[embedding_cols].values
embedding_norms = np.linalg.norm(embedding_matrix, axis=1)
df["embedding_norm"] = embedding_norms

corr, pval = spearmanr(df["embedding_norm"], df["score"])
print(f"[A] Spearman correlation (embedding norm vs score): {corr:.3f}, p = {pval:.3e}")

#  B. ML Model: Predict mutation score from embedding 
X = embedding_matrix
y = df["score"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"[B] Random Forest R² on test set: {r2:.3f}")

#  C. Add BLOSUM62 score as a baseline feature 
blosum62 = substitution_matrices.load("BLOSUM62")

def get_blosum_score(wt, mut):
    try:
        return blosum62[(wt, mut)]
    except KeyError:
        try:
            return blosum62[(mut, wt)]
        except KeyError:
            return 0  # substitution not found

df["blosum62"] = df.apply(lambda row: get_blosum_score(row["wt_residue"], row["mutant_residue"]), axis=1)

# Compare correlation
bl_corr, bl_pval = spearmanr(df["blosum62"], df["score"])
print(f"[C] Spearman correlation (BLOSUM62 vs score): {bl_corr:.3f}, p = {bl_pval:.3e}")

# Optional: ML model with both embedding + BLOSUM62
X_combo = np.hstack([embedding_matrix, df[["blosum62"]].values])
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_combo, y, test_size=0.2, random_state=42)

rf_combo = RandomForestRegressor(n_estimators=100, random_state=42)
rf_combo.fit(X_train_c, y_train_c)
y_pred_c = rf_combo.predict(X_test_c)

r2_combo = r2_score(y_test_c, y_pred_c)
print(f"[C] Random Forest R² with embedding + BLOSUM62: {r2_combo:.3f}")