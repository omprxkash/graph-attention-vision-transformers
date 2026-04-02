# GemVAE: Graph-Enhanced Multi-modal Variational Autoencoder for Spatially Resolved Multi-Omic Data Integration

**Author:** Omprakash Pugazhendhi  
**Affiliation:** Dept. of Computer Science and Engineering, Vellore Institute of Technology, Chennai  
**Contact:** omprakash.2021@vitstudent.ac.in

---

## Overview

GemVAE is a deep learning framework for integrating high-dimensional, spatially resolved multi-omic data — transcriptomics and proteomics — within a unified variational generative model. It addresses three core challenges of spatial omics analysis:

- **High dimensionality**: 20,000–30,000 gene features plus hundreds of protein targets
- **Modality heterogeneity**: RNA counts vs. antibody-derived tag (ADT) protein counts have distinct statistical distributions
- **Spatial context**: tissue organization must be preserved in the latent embedding

### Key Components

| Component | Description |
|-----------|-------------|
| Dual-stream encoder | Independent BatchNorm-Dropout-Linear blocks for gene and protein streams |
| Graph attention | Bidirectional dual-path attention over a spatial K-NN tissue graph |
| Variational bottleneck | 30-dimensional latent space with reparameterization trick |
| Contrastive loss | NT-Xent objective (τ = 0.5) pulling spatially adjacent cells together |
| Flexible decoder | MSE or ZINB reconstruction for both modalities |

---

## Installation

```bash
git clone https://github.com/<your-username>/GemVAE.git
cd GemVAE
pip install -r requirements.txt
```

For R-based mclust clustering (optional):
```r
install.packages("mclust")
```

---

## Quick Start

```python
import scanpy as sc
import numpy as np
from GEMVAE import GEMVAE, Cal_Spatial_Net

# Load AnnData with spatial coordinates in adata.obsm['spatial']
adata = sc.read_h5ad("data/breast_cancer.h5ad")

# Preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:, adata.var.highly_variable]

# Prepare inputs
X_gene = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
X_prot = adata.obsm["protein"]   # [N, d_protein] array of ADT counts

# Build spatial graph (KNN, k=6)
Cal_Spatial_Net(adata, k_cutoff=6, model="KNN")

# Train GemVAE
model = GEMVAE(
    hidden_dims1=[512, 256],
    hidden_dims2=[128, 64],
    z_dim=30,
    n_epochs=500,
    lambda_contrast=10.0,
)
model.train(adata, X_gene, X_prot)

# Extract latent embeddings
Z = model.get_embeddings(X_gene, X_prot)
adata.obsm["GemVAE"] = Z

# Downstream clustering
from GEMVAE import mclust_R
mclust_R(adata, num_cluster=7, use_rep="GemVAE")

# UMAP visualization
sc.pp.neighbors(adata, use_rep="GemVAE")
sc.tl.umap(adata)
sc.pl.umap(adata, color="mclust")
```

---

## Datasets

Place extracted dataset files in the `data/` directory. Supported datasets:

| Dataset | Technology | Tissue | Modalities |
|---------|-----------|--------|------------|
| SSC-MT | Stereo-seq SiteSeq | Mouse Thymus | Transcriptomics |
| Breast Cancer | Spatial CITE-seq | Human Breast Tumor | RNA + Protein |
| Spleen R1 | Spatial CITE-seq | Human Spleen | RNA + Protein |
| Spleen R2 | Spatial CITE-seq | Human Spleen | RNA + Protein |

---

## Repository Structure

```
GemVAE/
├── GEMVAE/
│   ├── __init__.py        # Public API
│   ├── GEMVAE.py          # Main model class: training, inference
│   ├── model.py           # GATE architecture: encoders, attention, VAE
│   ├── utils.py           # Spatial graph construction utilities
│   └── clustering.py      # mclust_R, Leiden, Louvain wrappers
├── data/                  # Dataset files (add .h5ad files here)
├── tutorials/             # Jupyter notebook tutorials per dataset
├── paper/
│   ├── GemVAE_paper.tex   # IEEE LaTeX source
│   ├── references.bib     # BibTeX entries
│   └── figures/           # Architecture diagrams and result figures
├── requirements.txt
└── README.md
```

---

## Citation

If you use GemVAE in your research, please cite:

```
Omprakash Pugazhendhi. GemVAE: A Graph-Enhanced Multi-modal Variational 
Autoencoder for Spatially Resolved Multi-Omic Data Integration. 
Vellore Institute of Technology, 2023.
```

---

## License

MIT License
