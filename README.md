# graph-attention-vision-transformers

Graph Attention Networks combined with Vision Transformers for image classification on CIFAR-100. Patches are treated as graph nodes connected by a sparse k-NN graph with Gaussian edge weights, replacing the fixed grid assumption of standard ViTs with a learned relational structure.

## Approach

Standard Vision Transformers attend uniformly over a flat sequence of patches. This project constructs a sparse graph over patches instead — each patch is connected to its k nearest neighbours in feature space, with edge weights computed from a Gaussian kernel over patch distances. Graph Attention Network layers then propagate information along these edges before classification.

Backbone: **EfficientNetV2-S** for patch feature extraction, fine-tuned alongside the graph attention layers end-to-end.

## Model Variants

Four variants were implemented and benchmarked against each other:

| Variant | Description |
|---------|-------------|
| **GAT-ViT-Base** | Standard GAT layers on top of patch embeddings |
| **GAT-ViT-Sparse** | Sparse k-NN graph (Gaussian edge weights) + GAT |
| **GAT-ViT-EfficientNet** | EfficientNetV2-S backbone + sparse GAT |
| **GAT-ViT-Full** | EfficientNetV2-S + sparse k-NN graph + multi-head GAT + ViT positional encoding |

## Dataset

CIFAR-100 — 60,000 images across 100 classes (50k train / 10k test), 32×32 resolution upscaled to 224×224 for the EfficientNetV2-S backbone.

## Stack

Python · PyTorch · PyTorch Geometric · timm (EfficientNetV2-S) · CIFAR-100
