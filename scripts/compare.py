import os
import re
import json
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer, is_aa

# === Configuration ===
root_dir = "af_outs"
wt_folder = "tem1_wt_785a3.result"
parser = PDBParser(QUIET=True)

def get_ca_atoms(structure, chain_id="A", residue_range=(23,)):
    ca_atoms = []
    for res in structure[0][chain_id]:
        if is_aa(res) and 'CA' in res:
            pos = res.get_id()[1]
            if pos >= residue_range[0]:
                ca_atoms.append(res['CA'])
    return ca_atoms

# === Load WT structure
wt_inner = os.path.join(root_dir, wt_folder)
wt_subdir = [d for d in os.listdir(wt_inner) if os.path.isdir(os.path.join(wt_inner, d))][0]
wt_pdb_files = [f for f in os.listdir(os.path.join(wt_inner, wt_subdir)) if f.endswith(".pdb") and "unrelaxed" in f]
wt_pdb_path = os.path.join(wt_inner, wt_subdir, wt_pdb_files[0])

wt_structure = parser.get_structure("WT", wt_pdb_path)
wt_chain = next(wt_structure[0].get_chains())
wt_atoms = get_ca_atoms(wt_structure, chain_id=wt_chain.id)

# === Process mutants
results = []

for folder in os.listdir(root_dir):
    if not folder.endswith(".result") or folder == wt_folder:
        continue

    match = re.search(r"_([A-Z])(\d+)([A-Z])_", folder)
    if not match:
        print(f"❌ Could not extract mutation position from {folder}")
        continue
    mut_pos = int(match.group(2))

    folder_path = os.path.join(root_dir, folder)
    inner_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    if not inner_dirs:
        continue

    subdir = inner_dirs[0]
    pdb_dir = os.path.join(folder_path, subdir)

    # === Get PDB
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb") and "unrelaxed" in f]
    if not pdb_files:
        continue
    pdb_path = os.path.join(pdb_dir, pdb_files[0])

    # === Get JSON
    json_files = [f for f in os.listdir(pdb_dir) if f.endswith(".json") and "_scores_" in f]
    if not json_files:
        continue
    json_path = os.path.join(pdb_dir, json_files[0])

    # === Compute RMSD
    try:
        mut_structure = parser.get_structure(folder, pdb_path)
        mut_chain = next(mut_structure[0].get_chains())
        mut_atoms = get_ca_atoms(mut_structure, chain_id=mut_chain.id)

        sup = Superimposer()
        sup.set_atoms(wt_atoms, mut_atoms)
        rmsd = sup.rms
    except Exception as e:
        print(f"❌ Error processing {folder}: {e}")
        rmsd = np.nan

    # === Extract pLDDT
    try:
        with open(json_path) as jf:
            data = json.load(jf)
        plddt = data["plddt"]
        adj_index = mut_pos - 23
        if 0 <= adj_index < len(plddt):
            mut_plddt = plddt[adj_index]
        else:
            mut_plddt = np.nan
            print(f"⚠️ Mutation position {mut_pos} out of bounds for pLDDT array in {folder}")
    except Exception as e:
        mut_plddt = np.nan
        print(f"❌ Error reading JSON for {folder}: {e}")

    results.append({
        "mutant": folder,
        "mut_pos": mut_pos,
        "rmsd_CA": rmsd,
        "plddt_mut": mut_plddt
    })

# === Save output
df = pd.DataFrame(results)
out_path = os.path.join(root_dir, "mutant_vs_wt_rmsd_plddt.csv")
df.to_csv(out_path, index=False)

print("✅ RMSD + pLDDT analysis complete. Results saved to:")
print(out_path)
print(df)
