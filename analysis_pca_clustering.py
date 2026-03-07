"""
PCA and Clustering Analysis for EGD-DLM

This script performs:
1. PCA analysis on token embeddings
2. Clustering analysis on generated text embeddings
3. Comparison between Base and EGD-DLM models

These are unsupervised analysis methods covered in the course.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import os
import argparse
import re
from collections import Counter
from tqdm import tqdm

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')


# ============================================================================
# Load Model Components (copied from egd_dlm.py for compatibility)
# ============================================================================

def simple_tokenizer(text):
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    return tokens


class SimpleVocab:
    def __init__(self, counter, min_freq=2, specials=None, max_size=None):
        self.stoi = {}
        self.itos = []
        if specials:
            for token in specials:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
        sorted_tokens = sorted(counter.items(), key=lambda x: -x[1])
        for token, count in sorted_tokens:
            if max_size and len(self.itos) >= max_size:
                break
            if count >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
        self.default_index = self.stoi.get("<unk>", 0)

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)

    def __len__(self):
        return len(self.itos)

    def get_itos(self):
        return self.itos


import math

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
        """Return token embeddings for analysis"""
        return self.token_embedding.weight.detach().cpu().numpy()


# ============================================================================
# PCA Analysis
# ============================================================================

def analyze_embeddings_pca(model, vocab, save_dir, model_name="model"):
    """
    Perform PCA analysis on token embeddings.

    This visualizes the learned representation space.
    """
    print(f"\n{'='*60}")
    print(f"PCA Analysis: {model_name}")
    print(f"{'='*60}")

    # Get embeddings
    embeddings = model.get_embeddings()  # (vocab_size, embed_dim)
    print(f"Embedding shape: {embeddings.shape}")

    # Apply PCA
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)

    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print(f"Top 10 PC explained variance: {explained_var[:10].sum():.2%}")
    print(f"Top 50 PC explained variance: {cumulative_var[-1]:.2%}")

    # Plot 1: Explained variance
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(range(1, 21), explained_var[:20])
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title(f'{model_name}: PCA Explained Variance')

    # Plot 2: Cumulative variance
    axes[1].plot(range(1, 51), cumulative_var, 'b-o', markersize=3)
    axes[1].axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title(f'{model_name}: Cumulative Variance')
    axes[1].legend()

    # Plot 3: 2D projection
    pca_2d = PCA(n_components=2)
    embeddings_2d = pca_2d.fit_transform(embeddings)

    # Color by frequency (first tokens are special, then high freq)
    colors = np.arange(len(embeddings))
    scatter = axes[2].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                              c=colors, cmap='viridis', alpha=0.5, s=1)
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    axes[2].set_title(f'{model_name}: 2D PCA Projection')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'pca_analysis_{model_name}.png'), dpi=150)
    plt.close()

    print(f"Saved PCA plot to {save_dir}/pca_analysis_{model_name}.png")

    return {
        'explained_variance': explained_var,
        'cumulative_variance': cumulative_var,
        'embeddings_2d': embeddings_2d
    }


def compare_embeddings_pca(model1, model2, vocab, save_dir, name1="Base", name2="EGD"):
    """Compare embeddings between two models using PCA"""
    print(f"\n{'='*60}")
    print(f"Comparing Embeddings: {name1} vs {name2}")
    print(f"{'='*60}")

    emb1 = model1.get_embeddings()
    emb2 = model2.get_embeddings()

    # Joint PCA
    combined = np.vstack([emb1, emb2])
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined)

    emb1_pca = combined_pca[:len(emb1)]
    emb2_pca = combined_pca[len(emb1):]

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(emb1_pca[:, 0], emb1_pca[:, 1], alpha=0.3, s=1, c='blue', label=name1)
    axes[0].scatter(emb2_pca[:, 0], emb2_pca[:, 1], alpha=0.3, s=1, c='red', label=name2)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('Joint PCA: Embedding Comparison')
    axes[0].legend()

    # Embedding difference analysis
    diff = emb1 - emb2
    diff_norms = np.linalg.norm(diff, axis=1)

    axes[1].hist(diff_norms, bins=50, edgecolor='black')
    axes[1].set_xlabel('L2 Distance')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Embedding Difference Distribution\nMean: {diff_norms.mean():.4f}')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'embedding_comparison.png'), dpi=150)
    plt.close()

    print(f"Mean embedding difference: {diff_norms.mean():.4f}")
    print(f"Saved comparison plot to {save_dir}/embedding_comparison.png")


# ============================================================================
# Clustering Analysis
# ============================================================================

def analyze_clustering(embeddings, save_dir, model_name="model", max_clusters=10):
    """
    Perform clustering analysis on embeddings.

    Uses K-Means and GMM (Gaussian Mixture Model) - both covered in class.
    """
    print(f"\n{'='*60}")
    print(f"Clustering Analysis: {model_name}")
    print(f"{'='*60}")

    # Reduce dimensionality first for efficiency
    pca = PCA(n_components=50)
    embeddings_reduced = pca.fit_transform(embeddings)

    # Try different numbers of clusters
    k_range = range(2, max_clusters + 1)
    kmeans_scores = []
    gmm_scores = []

    print("Evaluating cluster numbers...")
    for k in tqdm(k_range):
        # K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings_reduced)
        kmeans_scores.append(silhouette_score(embeddings_reduced, kmeans_labels))

        # GMM
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm_labels = gmm.fit_predict(embeddings_reduced)
        gmm_scores.append(silhouette_score(embeddings_reduced, gmm_labels))

    # Find optimal k
    best_k_kmeans = k_range[np.argmax(kmeans_scores)]
    best_k_gmm = k_range[np.argmax(gmm_scores)]

    print(f"Best K (K-Means): {best_k_kmeans}, Silhouette: {max(kmeans_scores):.4f}")
    print(f"Best K (GMM): {best_k_gmm}, Silhouette: {max(gmm_scores):.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(list(k_range), kmeans_scores, 'b-o', label='K-Means')
    axes[0].plot(list(k_range), gmm_scores, 'r-s', label='GMM')
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title(f'{model_name}: Cluster Evaluation')
    axes[0].legend()

    # Visualize best clustering
    best_k = best_k_kmeans
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_reduced)

    # 2D projection for visualization
    pca_2d = PCA(n_components=2)
    emb_2d = pca_2d.fit_transform(embeddings_reduced)

    scatter = axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.5, s=1)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title(f'{model_name}: K-Means Clustering (K={best_k})')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'clustering_{model_name}.png'), dpi=150)
    plt.close()

    print(f"Saved clustering plot to {save_dir}/clustering_{model_name}.png")

    return {
        'kmeans_scores': kmeans_scores,
        'gmm_scores': gmm_scores,
        'best_k_kmeans': best_k_kmeans,
        'best_k_gmm': best_k_gmm,
        'best_silhouette': max(kmeans_scores)
    }


# ============================================================================
# t-SNE Visualization
# ============================================================================

def visualize_tsne(model, vocab, save_dir, model_name="model", n_samples=5000):
    """
    t-SNE visualization of token embeddings.

    t-SNE is a manifold learning technique covered in class.
    """
    print(f"\n{'='*60}")
    print(f"t-SNE Visualization: {model_name}")
    print(f"{'='*60}")

    embeddings = model.get_embeddings()

    # Sample if too many tokens
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_sample = embeddings[indices]
    else:
        embeddings_sample = embeddings
        indices = np.arange(len(embeddings))

    print(f"Running t-SNE on {len(embeddings_sample)} tokens...")

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings_sample)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                c=indices, cmap='viridis', alpha=0.5, s=5)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f'{model_name}: t-SNE Visualization of Token Embeddings')
    plt.colorbar(label='Token Index (by frequency)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'tsne_{model_name}.png'), dpi=150)
    plt.close()

    print(f"Saved t-SNE plot to {save_dir}/tsne_{model_name}.png")


# ============================================================================
# Main
# ============================================================================

def load_model(checkpoint_path, vocab_size, embed_dim, n_heads, n_layers,
               dim_feedforward, dropout, device):
    """Load a trained model from checkpoint"""
    model = DiscreteDiffusionTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="PCA and Clustering Analysis")
    parser.add_argument("--base_checkpoint", type=str,
                        default="./checkpoints_base/egd_teacher_model.pth")
    parser.add_argument("--egd_checkpoint", type=str,
                        default="./checkpoints_egd_103/egd_student_model.pth")
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
    print("PCA and Clustering Analysis for EGD-DLM")
    print("="*60)
    print(f"Device: {device}")

    # Load models
    print("\nLoading models...")

    if os.path.exists(args.base_checkpoint):
        base_model = load_model(
            args.base_checkpoint, args.vocab_size, args.embed_dim,
            args.n_heads, args.n_layers, args.dim_feedforward, 0.1, device
        )
        print(f"Loaded Base model from {args.base_checkpoint}")

        # PCA analysis for Base
        base_pca = analyze_embeddings_pca(base_model, None, args.output_dir, "Base_Teacher")

        # Clustering for Base
        base_emb = base_model.get_embeddings()
        base_cluster = analyze_clustering(base_emb, args.output_dir, "Base_Teacher")

        # t-SNE for Base
        visualize_tsne(base_model, None, args.output_dir, "Base_Teacher")
    else:
        print(f"Warning: Base checkpoint not found at {args.base_checkpoint}")
        base_model = None

    if os.path.exists(args.egd_checkpoint):
        egd_model = load_model(
            args.egd_checkpoint, args.vocab_size, args.embed_dim,
            args.n_heads, args.n_layers, args.dim_feedforward, 0.1, device
        )
        print(f"Loaded EGD model from {args.egd_checkpoint}")

        # PCA analysis for EGD
        egd_pca = analyze_embeddings_pca(egd_model, None, args.output_dir, "EGD_Student")

        # Clustering for EGD
        egd_emb = egd_model.get_embeddings()
        egd_cluster = analyze_clustering(egd_emb, args.output_dir, "EGD_Student")

        # t-SNE for EGD
        visualize_tsne(egd_model, None, args.output_dir, "EGD_Student")
    else:
        print(f"Warning: EGD checkpoint not found at {args.egd_checkpoint}")
        egd_model = None

    # Compare if both models available
    if base_model is not None and egd_model is not None:
        compare_embeddings_pca(base_model, egd_model, None, args.output_dir,
                              "Base_Teacher", "EGD_Student")

    # Summary
    print("\n" + "="*60)
    print("Analysis Summary")
    print("="*60)

    if base_model is not None:
        print(f"\nBase Teacher:")
        print(f"  Top 10 PC variance: {base_pca['explained_variance'][:10].sum():.2%}")
        print(f"  Best K-Means K: {base_cluster['best_k_kmeans']}")
        print(f"  Silhouette Score: {base_cluster['best_silhouette']:.4f}")

    if egd_model is not None:
        print(f"\nEGD Student:")
        print(f"  Top 10 PC variance: {egd_pca['explained_variance'][:10].sum():.2%}")
        print(f"  Best K-Means K: {egd_cluster['best_k_kmeans']}")
        print(f"  Silhouette Score: {egd_cluster['best_silhouette']:.4f}")

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
