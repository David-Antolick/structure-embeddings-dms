import pandas as pd

dms_path = "data/processed/tem1_dms.csv"
output_fasta = "data/processed/mutant_sequences.fasta"

full_wt_sequence = (
    "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
)

# Truncate to remove residues 1–22 (positions 1–22), keep 23 onward (position 23 is index 22)
wt_sequence = full_wt_sequence[22:]

# ==== LOAD CLEANED DMS DATA ====
df = pd.read_csv(dms_path)

# === Filter: keep only mutations at position >= 26 (after truncation, 23+3) ===
df = df[df["position"] >= 26]

# Step 1: Pick mutations
bad_mutations = df.sort_values("score").head(2)
good_mutations = df.sort_values("score", ascending=False).head(2)
mid_mutation = df.iloc[(df["score"] - 1.0).abs().argsort()[:1]]
selected = pd.concat([bad_mutations, good_mutations, mid_mutation])
selected = selected.reset_index(drop=True)

# Step 2: Generate full mutant sequences
mutant_records = []

for i, row in selected.iterrows():
    pos = int(row["position"])
    wt = row["wt_residue"]
    mut = row["mutant_residue"]
    score = row["score"]

    # Adjust index relative to truncated sequence (starts at position 23)
    adjusted_pos = pos - 22
    assert wt_sequence[adjusted_pos - 1] == wt, f"WT residue mismatch at position {pos}: expected {wt}, found {wt_sequence[adjusted_pos-1]}"

    # Mutate sequence
    mutant_seq = list(wt_sequence)
    mutant_seq[adjusted_pos - 1] = mut
    mutant_seq = "".join(mutant_seq)

    # Create FASTA entry
    label = f"mut_{wt}{pos}{mut}_score{score:.3f}"
    mutant_records.append((label, mutant_seq))

# Step 3: Write to FASTA
with open(output_fasta, "w") as f:
    for name, seq in mutant_records:
        f.write(f">{name}\n")
        f.write(f"{seq}\n")

print(f"Wrote {len(mutant_records)} mutant sequences to {output_fasta}")