# GemVAE

**Graph-Enhanced Multi-modal Variational Autoencoder for Spatially Resolved Multi-Omic Data Integration**

---

## Author

**Omprakash Pugazhendhi**  
Dept. of Computer Science and Engineering, Vellore Institute of Technology, Chennai  
omprakash.2021@vitstudent.ac.in

### Also Published

> **WarLens – Transfer Learning for Event Classification in Conflict Zones**  
> Gautham R., Riyaas M., **Omprakash P.**, Alphonse S.  
> *6th International Conference on Innovative Trends in Information Technology (ICITIIT 2025), IEEE*  
> DOI: [10.1109/ICITIIT64777.2025.11040802](https://doi.org/10.1109/ICITIIT64777.2025.11040802)

---

## What Is GemVAE?

Spatial multi-omic platforms (e.g., Stereo-seq, Spatial CITE-seq) can simultaneously measure **gene expression** (20,000–30,000 features) and **surface protein abundance** at tissue-location resolution. Standard methods fail here because:

- High dimensionality breaks distance-based algorithms
- RNA counts and ADT protein counts have completely different statistical profiles
- Ignoring spatial coordinates discards tissue architecture

**GemVAE** solves all three by combining:

| Component | What it does |
|-----------|-------------|
| Dual-stream encoder | Compresses genes and proteins independently via BatchNorm-Dropout-Linear blocks before merging |
| Graph attention | Builds a K-NN tissue graph from spatial coordinates; attention weights encode cell-neighbourhood similarity |
| Variational bottleneck | Projects to a 30-dimensional latent space via the reparameterization trick |
| Contrastive loss | Pulls spatially adjacent cells together in latent space (NT-Xent, τ = 0.5) |

The result is a compact embedding that preserves both **modality information** and **tissue spatial structure**, ready for downstream clustering and visualization.

---

## Repository Structure

```
GemVAE/
├── GEMVAE/
│   ├── __init__.py        # Public API
│   ├── GEMVAE.py          # Training loop, loss functions, inference
│   ├── model.py           # GATE architecture: dual encoders, graph attention, VAE
│   ├── utils.py           # Spatial graph builders (2-D KNN/Radius, 3-D multi-section)
│   └── clustering.py      # mclust_R, Leiden, Louvain wrappers
├── paper/
│   ├── GemVAE_paper.tex   # Full IEEE conference paper (LaTeX source)
│   └── references.bib     # BibTeX for all 16 citations
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/GemVAE.git
cd GemVAE
pip install -r requirements.txt
```

For R-based mclust clustering (optional but recommended):
```r
install.packages("mclust")
```

---

## Quick Start

```python
import scanpy as sc
import numpy as np
from GEMVAE import GEMVAE, Cal_Spatial_Net

# Load AnnData — must have adata.obsm['spatial'] with tissue coordinates
adata = sc.read_h5ad("your_dataset.h5ad")

# Standard single-cell preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:, adata.var.highly_variable]

X_gene = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
X_prot = adata.obsm["protein"]  # [N, d_protein] ADT protein array

# Build spatial K-NN graph (k=6) from tissue coordinates
Cal_Spatial_Net(adata, k_cutoff=6, model="KNN")

# Train GemVAE
model = GEMVAE(
    hidden_dims1=[512, 256],   # gene encoder layers
    hidden_dims2=[128, 64],    # protein encoder layers
    z_dim=30,
    n_epochs=500,
    lambda_contrast=10.0,
)
model.train(adata, X_gene, X_prot)

# Extract 30-dimensional latent embeddings
adata.obsm["GemVAE"] = model.get_embeddings(X_gene, X_prot)

# Cluster
from GEMVAE import mclust_R
mclust_R(adata, num_cluster=7, use_rep="GemVAE")

# UMAP visualization
sc.pp.neighbors(adata, use_rep="GemVAE")
sc.tl.umap(adata)
sc.pl.umap(adata, color="mclust", title="GemVAE spatial domains")
```

---

## Evaluated Datasets

| Dataset | Technology | Tissue | Modalities |
|---------|-----------|--------|------------|
| SSC-MT | Stereo-seq SiteSeq | Mouse Thymus | Transcriptomics |
| Breast Cancer | Spatial CITE-seq | Human Breast Tumor | RNA + Protein |
| Spleen R1 | Spatial CITE-seq | Human Spleen | RNA + Protein |
| Spleen R2 | Spatial CITE-seq | Human Spleen | RNA + Protein |

---

## Results (ARI / NMI)

| Method | SSC-MT | Breast Cancer | Spleen R1 | Spleen R2 |
|--------|--------|--------------|-----------|-----------|
| PCA | 0.38 / 0.52 | 0.32 / 0.48 | 0.35 / 0.50 | 0.33 / 0.49 |
| SpaGCN | 0.44 / 0.58 | 0.38 / 0.53 | 0.42 / 0.56 | 0.40 / 0.55 |
| BayesSpace | 0.47 / 0.61 | 0.41 / 0.57 | 0.44 / 0.59 | 0.43 / 0.58 |
| totalVI | 0.41 / 0.55 | 0.45 / 0.60 | 0.43 / 0.58 | 0.42 / 0.57 |
| STAGATE | 0.56 / 0.67 | 0.49 / 0.63 | 0.53 / 0.64 | 0.51 / 0.63 |
| GraphST | 0.59 / 0.70 | 0.52 / 0.65 | 0.57 / 0.68 | 0.55 / 0.67 |
| **GemVAE** | **0.62 / 0.73** | **0.67 / 0.76** | **0.64 / 0.74** | **0.63 / 0.73** |

GemVAE's largest gains are on CITE-seq datasets (Breast Cancer, Spleen) where single-modality baselines cannot use the protein channel.

---

## Paper

The full methodology, experiments, and ablation study are written up in the IEEE conference paper format:

```
paper/GemVAE_paper.tex   — LaTeX source
paper/references.bib     — BibTeX references
```

To compile locally:
```bash
cd paper
pdflatex GemVAE_paper.tex && pdflatex GemVAE_paper.tex
```

Or paste `GemVAE_paper.tex` into [Overleaf](https://overleaf.com) to compile online.

---

## Citation

```bibtex
@misc{pugazhendhi2023gemvae,
  title   = {GemVAE: A Graph-Enhanced Multi-modal Variational Autoencoder
             for Spatially Resolved Multi-Omic Data Integration},
  author  = {Pugazhendhi, Omprakash},
  year    = {2023},
  note    = {Vellore Institute of Technology, Chennai}
}
```

---

## License

MIT License
