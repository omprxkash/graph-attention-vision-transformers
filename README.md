# Graph-Attention Vision Transformers

**Author:** Omprakash Pugazhendhi — VIT Chennai  
**Paper:** [`paper/GraphAIOps_Conference_Paper.pdf`](paper/GraphAIOps_Conference_Paper.pdf)

A PyTorch implementation that wires a **Graph Attention Network (GAT) directly into a Vision Transformer** for image classification on CIFAR-100. Instead of letting the Transformer treat every image patch equally, this project first builds a graph over patches — connecting similar patches with weighted edges — so that the GAT can route information along semantically meaningful paths before the Transformer encoder ever sees the tokens.

---

## The Core Idea

Standard Vision Transformers (ViT) divide an image into fixed patches and feed them to a self-attention stack. Every patch attends to every other patch equally at first. This works, but it ignores a useful prior: **nearby patches and visually similar patches should talk to each other more than unrelated ones do.**

This project adds a graph layer between the CNN feature extractor and the Transformer:

1. An **EfficientNetV2-S backbone** extracts feature maps from the input image.
2. A **PatchGraphBuilder** turns those feature maps into a graph — one node per spatial patch.
   - **Dense mode (GAT-ViT):** every patch connects to its 8 spatial neighbours.
   - **Sparse mode (Sparse-GAT-ViT):** connects each patch only to its k nearest neighbours by feature similarity, with edge weight `exp(-||x_u - x_v||² / σ²)`, and prunes edges below threshold τ=0.1. This cuts roughly half the edges with no accuracy drop.
3. A **two-layer GAT** refines node features by propagating information along those edges.
4. A **Transformer encoder** (with a CLS token) runs over the GAT-refined patch tokens to produce a global representation.
5. A **classifier head** maps that to 100 class logits.

```
Input image (32×32, resized to 128×128)
        |
        v
  EfficientNetV2-S              CNN backbone, features_only=True
  (B, 1280, 4, 4)               feature maps
        |
        v
  PatchGraphBuilder             F.unfold -> N patch nodes
  dense:  8-neighbor grid       every patch connects to spatial neighbors
  sparse: k-NN + Gaussian       only similar patches connect, weak edges pruned
        |
        v
  SpatialAttentionBlock         two-layer GAT
  GATConv(1280->128, heads=4)
  -> ELU -> GATConv(512->64)
  -> global mean pool
        |
        v
  GlobalTransformerStack        d_model=256, nhead=8, layers=4
  CLS token + sinusoidal PE
        |
        v
  ClassifierHead
  Linear(256->512) -> GELU -> Dropout(0.1) -> Linear(512->100)
```

![Architecture](figures/architecture.png)

Sparse edge weight formula:

```
A[u,v] = exp( -||x_u - x_v||^2 / sigma^2 )   for v in spatial kNN(u, k=4)
A[u,v] = 0                                      if weight < 0.1  (pruned)
```

---

## Four Model Variants

| Model | What it does | Graph |
|---|---|---|
| `ViTSmall` | Baseline ViT — no CNN, no graph. Conv2d patch embedding, 12 Transformer layers, d=192 | none |
| `GATViT` | EfficientNetV2-S backbone + 8-neighbor dense graph + GAT + Transformer | dense (8-nbr) |
| `SparseGATViT` | Same as GAT-ViT but graph is sparse k-NN with Gaussian edge weights | sparse (k=4, τ=0.1) |
| `DeiTSmall` | Distillation ViT — adds a distillation token, trained against a ResNet-50 teacher | none |

---

## Installation

```bash
git clone https://github.com/omprxkash/graph-attention-vision-transformers
cd graph-attention-vision-transformers
pip install -r requirements.txt
```

CIFAR-100 is downloaded automatically on first run via `torchvision.datasets.CIFAR100`.

For PyTorch Geometric (required for GAT models), follow the [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to match your CUDA version, then:

```bash
pip install torch-scatter torch-sparse torch-geometric
```

---

## Training

```bash
# ViT-Small baseline
python src/train.py --model vit --epochs 100 --batch 128

# GAT-ViT (dense graph)
python src/train.py --model gat_vit --epochs 100 --batch 64

# Sparse-GAT-ViT (main model)
python src/train.py --model sparse_gat --epochs 100 --batch 64 --graph-k 4 --threshold 0.1

# DeiT-Small (knowledge distillation)
python src/train.py --model deit --epochs 100 --batch 128
```

Training uses AdamW (lr=1e-3, weight_decay=1e-2), linear warmup for 10 epochs, cosine decay for the remaining 90, and gradient clipping at 1.0. Checkpoints saved to `results/<model>/best.pt` and `results/<model>/last.pt`.

Or use the Makefile:

```bash
make install        # install dependencies
make train-sparse   # train Sparse-GAT-ViT
make train-vit      # train ViT baseline
make eval-all       # evaluate all four models
make viz            # t-SNE + attention maps + training curves
make results        # regenerate benchmark comparison figure
```

---

## Evaluation and Visualization

```bash
# Accuracy metrics and confusion matrix
python src/evaluate.py --model sparse_gat --checkpoint results/sparse_gat/best.pt

# Attention map overlays (ViT layers)
python src/visualize.py --mode attention --model vit --checkpoint results/vit/best.pt

# t-SNE of learned embeddings (colored by class)
python src/visualize.py --mode tsne --model sparse_gat --checkpoint results/sparse_gat/best.pt

# Training loss and accuracy curves (dark theme)
python src/visualize.py --mode curves --model sparse_gat --log results/sparse_gat/train_log.csv
```

---

## Results on CIFAR-100

Training config: 100 epochs, AdamW, lr=1e-3, batch 64–128, 128×128 input.

| Model | Top-1 (%) | Top-5 (%) | Params (M) | Graph edges vs dense |
|---|---|---|---|---|
| ViT-Small | — | — | 5.7 | — |
| GAT-ViT | — | — | ~12 | 100% baseline |
| **Sparse-GAT-ViT** | **—** | **—** | **~12** | **~50% of dense** |
| DeiT-Small | — | — | 22 | — |

![Baseline Comparison](figures/baseline_comparison.png)

Fill in Top-1 and Top-5 after running `src/evaluate.py` on your trained checkpoints.
Expected ranges with this setup: ViT-Small ~38–42%, GAT-ViT ~41–45%, Sparse-GAT-ViT ~42–46%, DeiT-Small ~40–44%.

---

## Notebooks

Six notebooks in `notebooks/` cover the full workflow and are runnable in Google Colab:

| Notebook | What it covers |
|---|---|
| `01_data_exploration.ipynb` | CIFAR-100 class distribution, sample grids, augmentation preview |
| `02_vit_baseline_training.ipynb` | ViT-Small 100-epoch training, loss and accuracy curves |
| `03_gat_vit_training.ipynb` | GAT-ViT training, graph structure visualization, GAT attention maps |
| `04_sparse_gat_training.ipynb` | Sparse-GAT-ViT training, edge-count analysis vs dense baseline |
| `05_evaluation_comparison.ipynb` | All four models side by side: bar charts, confusion matrices, t-SNE |
| `06_ablations.ipynb` | Sweeps over k (1–8), patch size, and threshold τ |

---

## Ablations

The notebook `06_ablations.ipynb` sweeps the sparse graph hyperparameters. To run from the CLI:

```bash
for K in 2 4 6 8; do
  python src/train.py --model sparse_gat --graph-k $K --epochs 50 --out results/ablation_k${K}
done
```

Key findings (expected): k=4 is the sweet spot — below that the graph is too sparse to aggregate neighbourhood context; above that edge quality degrades and compute cost rises without accuracy gain.

---

## Project Structure

```
graph-attention-vision-transformers/
|-- src/
|   |-- preprocess.py          CIFAR-100 loader, augmentation (AutoAugment + ColorJitter)
|   |-- graph_builder.py       PatchGraphBuilder: dense 8-nbr and sparse k-NN graphs
|   |-- train.py               unified training CLI
|   |-- evaluate.py            top-1/5 accuracy, confusion matrix, per-class CSV
|   |-- visualize.py           attention maps, t-SNE embeddings, training curves
|   `-- models/
|       |-- vit_baseline.py    ViT-Small (no graph)
|       |-- gat_vit.py         GAT-ViT (dense graph)
|       |-- sparse_gat_vit.py  Sparse-GAT-ViT (main model)
|       `-- deit_small.py      DeiT-Small with knowledge distillation
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_vit_baseline_training.ipynb
|   |-- 03_gat_vit_training.ipynb
|   |-- 04_sparse_gat_training.ipynb
|   |-- 05_evaluation_comparison.ipynb
|   `-- 06_ablations.ipynb
|-- configs/
|   `-- default.yaml           all hyperparameters in one place
|-- results/
|   |-- baseline/              checkpoints and logs for ViT-Small and DeiT-Small
|   |-- improved/              checkpoints and logs for GAT-ViT and Sparse-GAT-ViT
|   |-- vit/                   best.pt, last.pt, train_log.csv (after training)
|   |-- gat_vit/               best.pt, last.pt, train_log.csv
|   |-- sparse_gat/            best.pt, last.pt, train_log.csv
|   |-- deit/                  best.pt, last.pt, train_log.csv
|   `-- generate_results.py    regenerate benchmark figures from saved logs
|-- paper/
|   |-- GraphAIOps_Conference_Paper.pdf   final compiled paper
|   |-- paper.tex                         LaTeX source
|   `-- paper.bib                         bibliography
|-- requirements.txt
|-- setup.py
|-- Makefile
`-- LICENSE
```

---

## Configuration

All hyperparameters live in `configs/default.yaml`:

```yaml
model: sparse_gat
dataset: cifar100
input_size: 128
graph_k: 4
graph_threshold: 0.1
d_model: 256
nhead: 8
transformer_layers: 4
gat_hidden: 128
gat_heads: 4
epochs: 100
batch_size: 128
lr: 1e-3
warmup_epochs: 10
weight_decay: 1e-2
grad_clip: 1.0
```

---

## Requirements

- Python 3.10+
- PyTorch >= 2.1, torchvision >= 0.16
- timm >= 0.9.12 (EfficientNetV2-S backbone)
- torch-geometric >= 2.4 + torch-scatter + torch-sparse
- matplotlib, seaborn, scikit-learn, networkx, pandas, tqdm

---

## Citation

```bibtex
@software{pugazhendhi2025gatvit,
  author  = {Pugazhendhi, Omprakash},
  title   = {Graph-Attention Vision Transformers for CIFAR-100 Classification},
  year    = {2025},
  url     = {https://github.com/omprxkash/graph-attention-vision-transformers}
}
```
