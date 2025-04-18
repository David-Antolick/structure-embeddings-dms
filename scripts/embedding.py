import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from Bio.Seq import Seq
import pandas as pd

# Path to the locally downloaded model
model_path = "/large_models/models/facebook-esm2_t36_3B_UR50D/"

# Load tokenizer and model from local path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Set to eval mode + no gradient calc
model.eval()
torch.set_grad_enabled(False)

# Example: TEM-1 β-lactamase wild-type sequence (replace with full sequence)
dna_seq = (
    "ATGAGTATTCAACATTTCCGTGTCGCCCTTATTCCCTTTTTTGCGGCATTTTGCCTTCCTGTTTTTGCTCACCCAGAAACGCTGGTGAAAGTAAAAGATGCTGAAGATCAGTTGGGTGCACGAGTGGGTTACATCGAACTGGATCTCAACAGCGGTAAGATCCTTGAGAGTTTTCGCCCCGAAGAACGTTTTCCAATGATGAGCACTTTTAAAGTTCTGCTATGTGGCGCGGTATTATCCCGTGTTGACGCCGGGCAAGAGCAACTCGGTCGCCGCATACACTATTCTCAGAATGACTTGGTTGAGTACTCACCAGTCACAGAAAAGCATCTTACGGATGGCATGACAGTAAGAGAATTATGCAGTGCTGCCATAACCATGAGTGATAACACTGCGGCCAACTTACTTCTGACAACGATCGGAGGACCGAAGGAGCTAACCGCTTTTTTGCACAACATGGGGGATCATGTAACTCGCCTTGATCGTTGGGAACCGGAGCTGAATGAAGCCATACCAAACGACGAGCGTGACACCACGATGCCTGCAGCAATGGCAACAACGTTGCGCAAACTATTAACTGGCGAACTACTTACTCTAGCTTCCCGGCAACAATTAATAGACTGGATGGAGGCGGATAAAGTTGCAGGACCACTTCTGCGCTCGGCCCTTCCGGCTGGCTGGTTTATTGCTGATAAATCTGGAGCCGGTGAGCGTGGGTCTCGCGGTATCATTGCAGCACTGGGGCCAGATGGTAAGCCCTCCCGTATCGTAGTTATCTACACGACGGGGAGTCAGGCAACTATGGATGAACGAAATAGACAGATCGCTGAGATAGGTGCCTCACTGATTAAGCATTGGTAA"
)
seq_obj = Seq(dna_seq)

# Translate to protein (use standard genetic code, first reading frame)
aa_seq = str(seq_obj.translate(to_stop=True))
print(aa_seq)

# Tokenize and batch
inputs = tokenizer(aa_seq, return_tensors="pt", add_special_tokens=True)

# Forward pass to get last hidden states
with torch.no_grad():
    outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state  # Shape: (1, 286+2, D)

# Remove special tokens ([CLS], [EOS])
embedding_tensor = token_embeddings[0, 1:-1]  # Shape: (286, D)

# Convert to numpy array
residue_embeddings = embedding_tensor.numpy()  # Shape: (286, 2560)

# Confirm shape
print("Per-residue embedding shape:", residue_embeddings.shape)

# Save if needed
np.save("data/processed/tem1_wt_esm2_embeddings.npy", residue_embeddings)

dms_df = pd.read_csv("data/processed/tem1_dms.csv")

assert dms_df["position"].max() <= residue_embeddings.shape[0], "DMS position exceeds sequence length!"

embedding_df = pd.DataFrame(residue_embeddings)
embedding_df["position"] = np.arange(1, len(residue_embeddings) + 1)  # positions are 1-based in DMS

# Merge DMS data with embeddings on position
merged_df = pd.merge(dms_df, embedding_df, on="position")

# Save final merged dataframe
merged_df.to_csv("data/processed/tem1_dms_embeddings.csv", index=False)

# Preview
print(merged_df.head())
print("Final shape:", merged_df.shape)