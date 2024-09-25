"""Generate figures for the GemVAE README."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BLUE   = "#2563EB"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
GRAY   = "#6B7280"
LIGHT  = "#F3F4F6"
RED    = "#DC2626"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
})

# ── 1. Baseline comparison (ARI across 4 datasets) ───────────────────────────
methods  = ["PCA", "SpaGCN", "BayesSpace", "totalVI", "STAGATE", "GraphST", "GemVAE"]
datasets = ["Mouse Thymus\n(SSC-MT)", "Breast Cancer\n(CITE-seq)",
            "Spleen R1\n(CITE-seq)", "Spleen R2\n(CITE-seq)"]

ari = np.array([
    [0.38, 0.32, 0.35, 0.33],
    [0.44, 0.38, 0.42, 0.40],
    [0.47, 0.41, 0.44, 0.43],
    [0.41, 0.45, 0.43, 0.42],
    [0.56, 0.49, 0.53, 0.51],
    [0.59, 0.52, 0.57, 0.55],
    [0.62, 0.67, 0.64, 0.63],
])

colors = [GRAY, GRAY, GRAY, GRAY, ORANGE, ORANGE, BLUE]
x = np.arange(len(datasets))
n = len(methods)
width = 0.11

fig, ax = plt.subplots(figsize=(12, 5))
for i, (method, color) in enumerate(zip(methods, colors)):
    offset = (i - n / 2 + 0.5) * width
    bars = ax.bar(x + offset, ari[i], width * 0.9,
                  label=method, color=color,
                  alpha=0.65 if method not in ("GemVAE", "GraphST", "STAGATE") else 1.0,
                  edgecolor="white", linewidth=0.4,
                  zorder=3)
    if method == "GemVAE":
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", color=BLUE)

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=10)
ax.set_ylabel("Adjusted Rand Index (ARI)", fontsize=11)
ax.set_title("GemVAE vs Baseline Methods — Spatial Clustering Performance",
             fontsize=13, fontweight="bold", pad=14)
ax.set_ylim(0, 0.82)
ax.legend(loc="upper left", fontsize=8.5, ncol=4,
          framealpha=0.9, edgecolor="#D1D5DB")
ax.axhline(0, color="#9CA3AF", linewidth=0.5)
plt.tight_layout()
plt.savefig("baseline_comparison.png", bbox_inches="tight")
plt.close()
print("baseline_comparison.png saved")

# ── 2. Component contribution (ablation) ─────────────────────────────────────
variants = ["No Graph\nAttention", "No Protein\nModality",
            "No Contrastive\nLoss", "GemVAE\n(Full Model)"]
ari_abl  = [0.48, 0.53, 0.55, 0.67]
nmi_abl  = [0.61, 0.65, 0.67, 0.76]
bar_colors = [RED, RED, RED, BLUE]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

for ax, scores, metric, ymax in zip(
    axes, [ari_abl, nmi_abl], ["ARI", "NMI"], [0.80, 0.90]
):
    bars = ax.bar(variants, scores, color=bar_colors,
                  width=0.5, edgecolor="white", linewidth=0.5, zorder=3)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{score:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=BLUE if score == max(scores) else RED)
    ax.set_ylim(0.4, ymax)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"Component Contribution — {metric}", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.set_axisbelow(True)
    # Delta annotation on full model bar
    delta = round(scores[-1] - scores[0], 2)
    ax.annotate(f"+{delta} vs no-graph",
                xy=(3, scores[-1]), xytext=(2.2, scores[-1] + 0.04),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4),
                fontsize=8.5, color=GREEN, fontweight="bold")

red_patch  = mpatches.Patch(color=RED,  label="Ablated variant")
blue_patch = mpatches.Patch(color=BLUE, label="Full GemVAE")
fig.legend(handles=[red_patch, blue_patch], loc="upper center",
           ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Ablation Study — Breast Cancer Dataset", fontsize=13,
             fontweight="bold", y=1.06)
plt.tight_layout()
plt.savefig("ablation_contribution.png", bbox_inches="tight")
plt.close()
print("ablation_contribution.png saved")

# ── 3. Architecture overview diagram ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4)
ax.axis("off")

def box(ax, x, y, w, h, text, color, text_color="white", fontsize=9):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                edgecolor="white", linewidth=1.5, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold",
            wrap=True, zorder=3)

def arrow(ax, x1, x2, y=2.0):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.8))

# Input boxes
box(ax, 0.1, 2.5, 1.7, 1.2, "Gene\nExpression\n(20–30k dims)", "#7C3AED")
box(ax, 0.1, 0.3, 1.7, 1.2, "Protein ADT\n(100–300 dims)", "#DB2777")

# Encoders
box(ax, 2.2, 2.5, 1.8, 1.2, "Gene Encoder\n[512 → 256]", "#4F46E5")
box(ax, 2.2, 0.3, 1.8, 1.2, "Protein Encoder\n[128 → 64]", "#BE185D")

# Joint
box(ax, 4.4, 1.2, 1.8, 1.6, "Joint\nProjection\n+ Concat", "#0369A1")

# Graph attention
box(ax, 6.6, 1.2, 1.9, 1.6, "Bidirectional\nGraph\nAttention", ORANGE)

# Spatial graph (above)
box(ax, 6.7, 3.2, 1.7, 0.6, "Spatial K-NN Graph", GREEN, text_color="white", fontsize=8)
ax.annotate("", xy=(7.55, 2.8), xytext=(7.55, 3.2),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))

# VAE bottleneck
box(ax, 8.9, 1.2, 1.6, 1.6, "VAE\nBottleneck\n(z ∈ ℝ³⁰)", BLUE)

# Contrastive loss annotation
ax.text(9.7, 3.15, "Contrastive\nSpatial Loss", ha="center", fontsize=8,
        color=RED, fontweight="bold")
ax.annotate("", xy=(9.7, 2.8), xytext=(9.7, 3.1),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

# Output
box(ax, 10.9, 1.2, 1.9, 1.6, "Latent\nEmbedding\n→ Clustering", GREEN)

# Arrows
arrow(ax, 1.8, 2.2, 3.1)
arrow(ax, 1.8, 2.2, 0.9)
arrow(ax, 4.0, 4.4, 2.0)
arrow(ax, 6.2, 6.6, 2.0)
arrow(ax, 8.5, 8.9, 2.0)
arrow(ax, 10.5, 10.9, 2.0)

ax.set_title("GemVAE Architecture", fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("architecture.png", bbox_inches="tight")
plt.close()
print("architecture.png saved")
