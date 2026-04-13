# GemVAE — Graph-Enhanced Multi-modal Variational Autoencoder

> A deep learning framework for integrating high-dimensional spatial transcriptomics and proteomics data within a single, spatially-aware generative model.

**Author:** Omprakash Pugazhendhi  
**Affiliation:** Dept. of Computer Science and Engineering, Vellore Institute of Technology, Chennai  
**Contact:** omprakash.2021@vitstudent.ac.in

---

## Read the Paper

The full research paper is available right here in the repository:

**[GemVAE_Pugazhendhi_2023.pdf](paper/GemVAE_Pugazhendhi_2023.pdf)**

It covers the complete architecture, the motivation behind every design decision, the experiments across four biological datasets, and a breakdown of what each component actually contributes. If you want to understand what this project does and why, that's the place to start.

---

## The Problem This Solves

Modern spatial omics platforms like Stereo-seq and Spatial CITE-seq can measure gene expression (20,000–30,000 genes) and protein levels simultaneously, at precise locations within a tissue slice. That sounds powerful — and it is — but it creates a hard computational problem:

- **Too many dimensions.** With 30,000+ features per cell, standard algorithms like k-means or PCA start breaking down. Distances become meaningless, and linear projections miss the nonlinear structure in the data.
- **Two completely different data types.** RNA counts and protein (ADT) counts behave differently statistically. Throwing them into the same model naively doesn't work.
- **Location matters.** A T cell in the centre of a tumour behaves differently from one at the edge — but most methods treat every cell as if it exists in isolation, ignoring where it sits in the tissue entirely.

GemVAE was built to address all three of these at once.

---

## How It Works

GemVAE learns a compact 30-dimensional representation of each cell that captures both what molecules it contains and where it lives in the tissue.

It does this through four interlocking pieces:

**Dual-stream encoding** — Gene expression and protein data are compressed separately first, through independent stacks of BatchNorm → Dropout → Linear layers with ELU activations. Only after each modality has been reduced to a manageable size are they merged. This prevents one modality from drowning out the other.

**Spatial graph attention** — From the tissue coordinates, GemVAE builds a K-nearest-neighbour graph connecting each cell to its six closest spatial neighbours. A bidirectional graph attention mechanism then learns which neighbours are most informative for each cell, encoding the tissue topology directly into the representation.

**Variational bottleneck** — The attended features pass through a VAE bottleneck (reparameterization trick), producing a smooth, regularized latent space that generalises better to downstream tasks.

**Contrastive spatial loss** — Cells that sit next to each other in tissue are pulled close together in the latent space using an NT-Xent contrastive objective (temperature τ = 0.5). This is what makes the embeddings spatially coherent rather than just molecularly accurate.

Everything is trained together end-to-end with a combined loss:

```
L = λ_recon · L_recon + λ_kl · L_KL + λ_contrast · L_contrast + λ_wd · L_wd
```

Default weights: reconstruction = 1, KL = 0 (deterministic mode), contrastive = 10, weight decay = 1.

---

## Results

GemVAE was benchmarked against six methods on four datasets (mouse thymus via Stereo-seq, and human breast cancer + spleen × 2 via Spatial CITE-seq). Clustering quality is measured by ARI (Adjusted Rand Index) and NMI (Normalized Mutual Information) — higher is better on both.

| Method | Mouse Thymus | Breast Cancer | Spleen R1 | Spleen R2 |
|--------|:------------:|:-------------:|:---------:|:---------:|
| PCA | 0.38 / 0.52 | 0.32 / 0.48 | 0.35 / 0.50 | 0.33 / 0.49 |
| SpaGCN | 0.44 / 0.58 | 0.38 / 0.53 | 0.42 / 0.56 | 0.40 / 0.55 |
| BayesSpace | 0.47 / 0.61 | 0.41 / 0.57 | 0.44 / 0.59 | 0.43 / 0.58 |
| totalVI | 0.41 / 0.55 | 0.45 / 0.60 | 0.43 / 0.58 | 0.42 / 0.57 |
| STAGATE | 0.56 / 0.67 | 0.49 / 0.63 | 0.53 / 0.64 | 0.51 / 0.63 |
| GraphST | 0.59 / 0.70 | 0.52 / 0.65 | 0.57 / 0.68 | 0.55 / 0.67 |
| **GemVAE** | **0.62 / 0.73** | **0.67 / 0.76** | **0.64 / 0.74** | **0.63 / 0.73** |

The biggest gains are on the CITE-seq datasets (Breast Cancer, Spleen) where all RNA-only baselines are working with one hand tied behind their back — they simply cannot use the protein channel. GemVAE's dual-stream design makes that information accessible.

An ablation study in the paper shows that each of the three main components (graph attention, protein modality, contrastive loss) contributes independently, with the spatial graph being the single most important factor (ΔAR = +0.19 when removed on Breast Cancer).

---

## Getting Started

**Install:**
```bash
git clone https://github.com/<your-username>/GemVAE.git
cd GemVAE
pip install -r requirements.txt
```

For mclust-based clustering (optional, needs R):
```r
install.packages("mclust")
```

**Run on your data:**
```python
import scanpy as sc
from GEMVAE import GEMVAE, Cal_Spatial_Net

# Your AnnData needs adata.obsm['spatial'] with (x, y) tissue coordinates
adata = sc.read_h5ad("your_data.h5ad")

# Preprocess
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:, adata.var.highly_variable]

X_gene = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
X_prot = adata.obsm["protein"]  # protein ADT matrix [N × d_protein]

# Build the spatial K-NN graph
Cal_Spatial_Net(adata, k_cutoff=6, model="KNN")

# Train
model = GEMVAE(hidden_dims1=[512, 256], hidden_dims2=[128, 64], z_dim=30, n_epochs=500)
model.train(adata, X_gene, X_prot)

# Get embeddings and cluster
adata.obsm["GemVAE"] = model.get_embeddings(X_gene, X_prot)

from GEMVAE import mclust_R
mclust_R(adata, num_cluster=7, use_rep="GemVAE")

# Visualize
sc.pp.neighbors(adata, use_rep="GemVAE")
sc.tl.umap(adata)
sc.pl.umap(adata, color="mclust")
```

---

## Repository Layout

```
GemVAE/
├── GEMVAE/
│   ├── __init__.py      — public API
│   ├── GEMVAE.py        — training loop, losses, inference
│   ├── model.py         — GATE: dual encoders, graph attention, VAE bottleneck
│   ├── utils.py         — spatial graph construction (2-D and 3-D multi-section)
│   └── clustering.py    — mclust_R, Leiden, Louvain wrappers
├── paper/
│   ├── GemVAE_Pugazhendhi_2023.pdf  — research paper
│   ├── GemVAE_paper.tex             — LaTeX source
│   └── references.bib               — all 16 citations in BibTeX
├── requirements.txt
└── README.md
```

---

## Citation

If you build on this work, please cite:

```bibtex
@misc{pugazhendhi2023gemvae,
  title  = {GemVAE: A Graph-Enhanced Multi-modal Variational Autoencoder
            for Spatially Resolved Multi-Omic Data Integration},
  author = {Pugazhendhi, Omprakash},
  year   = {2023},
  note   = {Vellore Institute of Technology, Chennai}
}
```

---

## License

MIT
