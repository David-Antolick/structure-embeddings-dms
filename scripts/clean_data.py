import pandas as pd
import re

df = pd.read_csv("data/raw/tem1_dms.csv")
print(df.head())

df = df.dropna(subset=["score"])

# Keep only single substitutions in 'hgvs_pro', like 'p.Ala237Thr'
pattern = r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})$"

df = df[df["hgvs_pro"].str.match(pattern, na=False)]

# Extract relevent info
df["wt_residue"] = df["hgvs_pro"].str.extract(pattern)[0]
df["position"] = df["hgvs_pro"].str.extract(pattern)[1].astype(int)
df["mutant_residue"] = df["hgvs_pro"].str.extract(pattern)[2]

# Drop stop codon mutations
df = df[df["wt_residue"] != "Ter"]

#Convert to single letter AA
aa3_to_1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': '*'
}

# Apply the conversion
df["wt_residue"] = df["wt_residue"].map(aa3_to_1)
df["mutant_residue"] = df["mutant_residue"].map(aa3_to_1)


# Format correctly
parsed_df = df[["position", "wt_residue", "mutant_residue", "score"]]
parsed_df = parsed_df.reset_index(drop=True)

# Show and save
print(parsed_df.head())
parsed_df.to_csv('data/processed/tem1_dms.csv', index=False)