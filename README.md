# Evaluating the Functional Information Content of Protein Structure Prediction Model Embeddings using Deep Mutational Scanning Data

This project investigates whether per-residue embeddings from structure prediction models such as ESMFold contain functional information about amino acid mutations. We test whether these embeddings can predict experimental fitness scores from Deep Mutational Scanning (DMS) datasets.

---

## Research Question

Do the representations learned by protein structure prediction models (like AlphaFold2 or ESMFold) contain information that correlates with the functional consequences of amino acid mutations?

---

## Dataset and Tools

- **Protein**: TEM-1 β-lactamase  
- **Mutation data**: Retrieved from [MaveDB](https://www.mavedb.org/)  
  Accession: `urn:mavedb:00000070-a-1`
- **Model**: ESM-2 (3B), via HuggingFace Transformers
- **Embeddings**: Extracted from wild-type sequence using ESM-2
- **Language**: Python  
- **Libraries**: `transformers`, `torch`, `pandas`, `scikit-learn`, `biopython`, `scipy`, `matplotlib`

---

## Pipeline

1. **`clean_data.py`**  
   Parses and filters the raw DMS data to retain only valid single amino acid substitutions.

2. **`embedding.py`**  
   Loads ESM-2 model, extracts per-residue embeddings from the wild-type protein sequence, and merges them with DMS scores.

3. **`analysis.py`**  
   Performs:
   - Correlation analysis (embedding norm vs mutation score)
   - Random Forest regression to predict mutation scores
   - BLOSUM62-based baseline comparisons

4. **`analysis_plots.py`**  
   Repeats analyses with plots:
   - Embedding norm vs mutation score
   - BLOSUM62 score vs mutation score
   - Model R² performance comparison

---

## Key Results

| Task | Description | Value |
|------|-------------|-------|
| A | Spearman correlation (embedding norm vs DMS score) | ~0.27 |
| B | Random Forest R² (embeddings only) | ~0.45 |
| C | Spearman correlation (BLOSUM62 vs DMS score) | ~0.46 |
| C | Random Forest R² (embeddings + BLOSUM62) | ~0.73 |

---

## Outputs

- 📁 `data/processed/`  
  Contains cleaned DMS and merged embedding datasets

- 📁 `plots/`  
  Contains all figures:
  - `embedding_norm_vs_score.png`
  - `blosum62_vs_score.png`
  - `model_r2_comparison.png`

---

## References

1. Jumper, J., et al. (2021). *Highly accurate protein structure prediction with AlphaFold*. **Nature**, 596(7873), 583–589.  
2. Lin, Z., et al. (2022). *Language models of protein sequences at the scale of evolution enable accurate structure prediction*. **bioRxiv**.  
3. Esposito, D., et al. (2019). *MaveDB: an open-source platform for massive assay data*. **Genome Biology**, 20(1), 269.

---

## Author

**David Antolick**
