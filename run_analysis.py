"""
Standalone PCA and Clustering Analysis Script
Run this on the server where your model checkpoints are saved.

Usage:
    python run_analysis.py --checkpoint ./checkpoints_egd_103/egd_student_model.pth

Or to analyze embeddings from scratch (no checkpoint needed):
    python run_analysis.py --from_scratch
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import os
import argparse
import math
import re
from collections import Counter
from tqdm import tqdm

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


# ============================================================================
# Model Definition (same as egd_dlm.py)
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, t):
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class DiscreteDiffusionTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers,
                 dim_feedforward, dropout, max_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_len)
        self.time_embedding = TimeEmbedding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_final = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, t, attention_mask=None):
        h = self.token_embedding(x)
        h = self.pos_encoding(h)
        t_emb = self.time_embedding(t)
        h = h + t_emb.unsqueeze(1)
        h = self.transformer(h)
        h = self.ln_final(h)
        logits = self.output_proj(h)
        return logits

    def get_embeddings(self):
        return self.token_embedding.weight.detach().cpu().numpy()


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_pca(embeddings, save_dir, model_name="model"):
    """PCA analysis on embeddings"""
    print(f"\n{'='*60}")
    print(f"PCA Analysis: {model_name}")
    print(f"{'='*60}")
    print(f"Embedding shape: {embeddings.shape}")

    # Apply PCA
    n_components = min(50, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print(f"Top 10 PC explained variance: {explained_var[:10].sum():.2%}")
    print(f"Top {n_components} PC cumulative variance: {cumulative_var[-1]:.2%}")

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Explained variance
    axes[0].bar(range(1, min(21, n_components+1)), explained_var[:20], color='steelblue')
    axes[0].set_xlabel('Principal Component', fontsize=11)
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=11)
    axes[0].set_title(f'{model_name}: PCA Explained Variance', fontsize=12)

    # Plot 2: Cumulative variance
    axes[1].plot(range(1, n_components+1), cumulative_var, 'b-o', markersize=3)
    axes[1].axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
    axes[1].set_xlabel('Number of Components', fontsize=11)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=11)
    axes[1].set_title(f'{model_name}: Cumulative Variance', fontsize=12)
    axes[1].legend()

    # Plot 3: 2D projection
    pca_2d = PCA(n_components=2)
    embeddings_2d = pca_2d.fit_transform(embeddings)

    # Sample for visualization if too many points
    n_points = min(5000, len(embeddings_2d))
    indices = np.random.choice(len(embeddings_2d), n_points, replace=False)

    scatter = axes[2].scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                              c=indices, cmap='viridis', alpha=0.6, s=3)
    axes[2].set_xlabel('PC1', fontsize=11)
    axes[2].set_ylabel('PC2', fontsize=11)
    axes[2].set_title(f'{model_name}: 2D PCA Projection', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'pca_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")
    return explained_var, cumulative_var


def analyze_clustering(embeddings, save_dir, model_name="model"):
    """Clustering analysis using K-Means and GMM"""
    print(f"\n{'='*60}")
    print(f"Clustering Analysis: {model_name}")
    print(f"{'='*60}")

    # Reduce dimensionality first
    pca = PCA(n_components=min(50, embeddings.shape[1]))
    embeddings_reduced = pca.fit_transform(embeddings)

    # Evaluate different K values
    k_range = range(2, 12)
    kmeans_scores = []
    gmm_scores = []

    print("Evaluating cluster numbers...")
    for k in tqdm(k_range):
        # K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings_reduced)
        kmeans_scores.append(silhouette_score(embeddings_reduced, kmeans_labels))

        # GMM
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
        gmm_labels = gmm.fit_predict(embeddings_reduced)
        gmm_scores.append(silhouette_score(embeddings_reduced, gmm_labels))

    best_k_kmeans = list(k_range)[np.argmax(kmeans_scores)]
    best_k_gmm = list(k_range)[np.argmax(gmm_scores)]

    print(f"Best K (K-Means): {best_k_kmeans}, Silhouette: {max(kmeans_scores):.4f}")
    print(f"Best K (GMM): {best_k_gmm}, Silhouette: {max(gmm_scores):.4f}")

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Silhouette scores
    axes[0].plot(list(k_range), kmeans_scores, 'b-o', label='K-Means', linewidth=2)
    axes[0].plot(list(k_range), gmm_scores, 'r-s', label='GMM', linewidth=2)
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[0].set_ylabel('Silhouette Score', fontsize=11)
    axes[0].set_title(f'{model_name}: Cluster Evaluation', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Best clustering visualization
    best_k = best_k_kmeans
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_reduced)

    pca_2d = PCA(n_components=2)
    emb_2d = pca_2d.fit_transform(embeddings_reduced)

    # Sample for visualization
    n_points = min(5000, len(emb_2d))
    indices = np.random.choice(len(emb_2d), n_points, replace=False)

    scatter = axes[1].scatter(emb_2d[indices, 0], emb_2d[indices, 1],
                              c=labels[indices], cmap='tab10', alpha=0.6, s=5)
    axes[1].set_xlabel('PC1', fontsize=11)
    axes[1].set_ylabel('PC2', fontsize=11)
    axes[1].set_title(f'{model_name}: K-Means Clustering (K={best_k})', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'clustering_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")
    return best_k_kmeans, max(kmeans_scores)


def analyze_tsne(embeddings, save_dir, model_name="model", n_samples=3000):
    """t-SNE visualization"""
    print(f"\n{'='*60}")
    print(f"t-SNE Visualization: {model_name}")
    print(f"{'='*60}")

    # Sample for efficiency
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_sample = embeddings[indices]
    else:
        embeddings_sample = embeddings
        indices = np.arange(len(embeddings))

    print(f"Running t-SNE on {len(embeddings_sample)} tokens...")

    # Apply t-SNE (use max_iter for newer sklearn versions)
    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings_sample)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                         c=indices, cmap='viridis', alpha=0.6, s=5)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(f'{model_name}: t-SNE Visualization of Token Embeddings', fontsize=14)
    plt.colorbar(scatter, label='Token Index (by frequency)')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'tsne_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


def create_summary_figure(results, save_dir):
    """Create a summary figure combining key results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # This will be filled with actual data when run
    # For now, create placeholder with instructions

    fig.suptitle('EGD-DLM: Embedding Analysis Summary', fontsize=14, fontweight='bold')

    for ax in axes.flat:
        ax.text(0.5, 0.5, 'Run analysis to generate',
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'analysis_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PCA and Clustering Analysis for EGD-DLM")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pth file)")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Create random model for demo (no checkpoint needed)")
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="./analysis_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*60)
    print("EGD-DLM: PCA and Clustering Analysis")
    print("="*60)
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")

    # Create or load model
    model = DiscreteDiffusionTransformer(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=0.1
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nLoading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model_name = "EGD_DLM"
    elif args.from_scratch:
        print("\nUsing randomly initialized model (demo mode)")
        model_name = "Random_Init"
    else:
        print("\nError: Please provide --checkpoint or use --from_scratch")
        print("Example: python run_analysis.py --checkpoint ./checkpoints_egd_103/egd_student_model.pth")
        return

    model.to(device)
    model.eval()

    # Get embeddings
    embeddings = model.get_embeddings()
    print(f"\nEmbedding matrix shape: {embeddings.shape}")

    # Run analyses
    results = {}

    # 1. PCA Analysis
    explained_var, cumulative_var = analyze_pca(embeddings, args.output_dir, model_name)
    results['pca_var_10'] = explained_var[:10].sum()
    results['pca_var_50'] = cumulative_var[-1] if len(cumulative_var) >= 50 else cumulative_var[-1]

    # 2. Clustering Analysis
    best_k, silhouette = analyze_clustering(embeddings, args.output_dir, model_name)
    results['best_k'] = best_k
    results['silhouette'] = silhouette

    # 3. t-SNE Visualization
    analyze_tsne(embeddings, args.output_dir, model_name)

    # Summary
    print("\n" + "="*60)
    print("Analysis Summary")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Vocabulary Size: {args.vocab_size}")
    print(f"Embedding Dimension: {args.embed_dim}")
    print(f"\nPCA Results:")
    print(f"  Top 10 PC variance: {results['pca_var_10']:.2%}")
    print(f"\nClustering Results:")
    print(f"  Optimal K (K-Means): {results['best_k']}")
    print(f"  Silhouette Score: {results['silhouette']:.4f}")
    print(f"\nAll figures saved to: {args.output_dir}/")
    print("  - pca_{model}.png")
    print("  - clustering_{model}.png")
    print("  - tsne_{model}.png")


if __name__ == "__main__":
    main()
