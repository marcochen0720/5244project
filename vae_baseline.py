"""
VAE Baseline for Language Modeling

This implements a Variational Autoencoder for text generation as a baseline
to compare with our EGD-DLM approach.

VAE is a classic unsupervised generative model covered in the course.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import argparse
import math
import os
import re
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset


# ============================================================================
# Simple Tokenizer and Vocabulary (same as EGD-DLM)
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


# ============================================================================
# Configuration
# ============================================================================

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    DATASET = "wikitext-103"
    SEQ_LEN = 128
    MIN_FREQ = 2
    MAX_VOCAB_SIZE = 30000

    # Model
    VOCAB_SIZE = None
    EMBED_DIM = 512
    HIDDEN_DIM = 512
    LATENT_DIM = 256
    N_LAYERS = 2
    DROPOUT = 0.1

    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    KL_WEIGHT = 0.1  # Weight for KL divergence term

    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    OUTPUT_DIR = "./checkpoints_vae"

config = Config()


# ============================================================================
# Data Loading
# ============================================================================

def get_data_and_vocab(dataset_name):
    if dataset_name == "wikitext-2":
        dataset_id = "wikitext-2-raw-v1"
    else:
        dataset_id = "wikitext-103-raw-v1"

    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset("wikitext", dataset_id)

    print("Building vocabulary...")
    counter = Counter()
    for split in ["train", "validation", "test"]:
        for example in tqdm(dataset[split], desc=f"Tokenizing {split}"):
            text = example["text"]
            if text.strip():
                tokens = simple_tokenizer(text)
                counter.update(tokens)

    vocabulary = SimpleVocab(
        counter,
        min_freq=config.MIN_FREQ,
        specials=[config.PAD_TOKEN, config.UNK_TOKEN, config.BOS_TOKEN, config.EOS_TOKEN],
        max_size=config.MAX_VOCAB_SIZE
    )

    config.VOCAB_SIZE = len(vocabulary)
    config.PAD_IDX = vocabulary[config.PAD_TOKEN]
    config.BOS_IDX = vocabulary[config.BOS_TOKEN]
    config.EOS_IDX = vocabulary[config.EOS_TOKEN]

    print(f"Vocabulary size: {config.VOCAB_SIZE}")

    def tokenize_and_numericalize(data_split):
        result = []
        for example in data_split:
            text = example["text"]
            if text.strip():
                tokens = simple_tokenizer(text)
                if len(tokens) >= 5:
                    ids = [vocabulary[t] for t in tokens]
                    result.append(ids)
        return result

    train_data = tokenize_and_numericalize(dataset["train"])
    val_data = tokenize_and_numericalize(dataset["validation"])

    return train_data, val_data, vocabulary


def create_sequences(data, seq_len):
    sequences = []
    for ids in data:
        for i in range(0, len(ids) - seq_len + 1, seq_len // 2):
            seq = ids[i:i + seq_len]
            if len(seq) == seq_len:
                sequences.append(torch.tensor(seq, dtype=torch.long))
    return sequences


def collate_fn(batch):
    return torch.stack(batch)


def get_dataloaders(train_data, val_data):
    train_seqs = create_sequences(train_data, config.SEQ_LEN)
    val_seqs = create_sequences(val_data, config.SEQ_LEN)

    print(f"Training sequences: {len(train_seqs)}")
    print(f"Validation sequences: {len(val_seqs)}")

    train_loader = DataLoader(train_seqs, batch_size=config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_seqs, batch_size=config.BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, drop_last=True)

    return train_loader, val_loader


# ============================================================================
# VAE Model
# ============================================================================

class VAEEncoder(nn.Module):
    """LSTM-based VAE Encoder"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)

        # Bidirectional doubles the hidden size
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        _, (hidden, _) = self.lstm(embedded)  # hidden: (n_layers*2, batch, hidden_dim)

        # Concatenate forward and backward final hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # (batch, hidden_dim*2)

        mu = self.fc_mu(hidden)  # (batch, latent_dim)
        logvar = self.fc_logvar(hidden)  # (batch, latent_dim)

        return mu, logvar


class VAEDecoder(nn.Module):
    """LSTM-based VAE Decoder"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * n_layers)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, x, z):
        # x: (batch, seq_len) - input tokens
        # z: (batch, latent_dim) - latent vector

        batch_size = x.size(0)

        # Initialize hidden state from latent
        hidden = self.latent_to_hidden(z)  # (batch, hidden_dim * n_layers)
        hidden = hidden.view(batch_size, self.n_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()  # (n_layers, batch, hidden_dim)
        cell = torch.zeros_like(hidden)

        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, _ = self.lstm(embedded, (hidden, cell))  # (batch, seq_len, hidden_dim)
        logits = self.output_proj(output)  # (batch, seq_len, vocab_size)

        return logits


class TextVAE(nn.Module):
    """Complete VAE for Text Generation"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, n_layers, dropout):
        super().__init__()
        self.encoder = VAEEncoder(vocab_size, embed_dim, hidden_dim, latent_dim, n_layers, dropout)
        self.decoder = VAEDecoder(vocab_size, embed_dim, hidden_dim, latent_dim, n_layers, dropout)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode (teacher forcing: use input as decoder input)
        logits = self.decoder(x, z)

        return logits, mu, logvar

    def generate(self, batch_size, seq_len, device, temperature=1.0):
        """Generate sequences from prior"""
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(batch_size, self.latent_dim, device=device)

            # Start with BOS token
            generated = torch.full((batch_size, 1), config.BOS_IDX,
                                   dtype=torch.long, device=device)

            for _ in range(seq_len - 1):
                logits = self.decoder(generated, z)
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

            return generated


# ============================================================================
# Training
# ============================================================================

def vae_loss(logits, targets, mu, logvar, kl_weight):
    """
    VAE Loss = Reconstruction Loss + KL Divergence

    This is the classic VAE objective (ELBO).
    """
    # Reconstruction loss (cross-entropy)
    batch_size, seq_len, vocab_size = logits.shape
    recon_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        reduction='mean'
    )

    # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


def train_vae(model, train_loader, val_loader, config):
    print("\n" + "="*60)
    print("Training VAE Baseline")
    print("="*60)

    model.to(config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        for batch in pbar:
            x = batch.to(config.DEVICE)

            optimizer.zero_grad()

            logits, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(logits, x, mu, logvar, config.KL_WEIGHT)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'kl': f"{kl_loss.item():.4f}"
            })

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_recon = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(config.DEVICE)
                logits, mu, logvar = model(x)
                loss, recon_loss, _ = vae_loss(logits, x, mu, logvar, config.KL_WEIGHT)
                val_loss += loss.item()
                val_recon += recon_loss.item()
                val_batches += 1

        avg_train_loss = total_loss / num_batches
        avg_val_loss = val_loss / val_batches
        avg_val_recon = val_recon / val_batches

        # Compute perplexity from reconstruction loss
        val_ppl = math.exp(avg_val_recon)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val PPL={val_ppl:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config.OUTPUT_DIR, "vae_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

    return model


def evaluate_vae(model, val_loader, config):
    """Evaluate VAE and return perplexity"""
    model.eval()
    total_recon = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating VAE"):
            x = batch.to(config.DEVICE)
            logits, mu, logvar = model(x)
            _, recon_loss, _ = vae_loss(logits, x, mu, logvar, config.KL_WEIGHT)
            total_recon += recon_loss.item()
            num_batches += 1

    avg_recon = total_recon / num_batches
    ppl = math.exp(avg_recon)

    print(f"VAE - Validation Perplexity: {ppl:.2f}")
    return ppl


def decode_tokens(token_ids, vocab, pad_idx):
    itos = vocab.get_itos()
    texts = []
    for seq in token_ids:
        tokens = []
        for idx in seq:
            idx = idx.item()
            if idx != pad_idx and idx != config.BOS_IDX and idx != config.EOS_IDX:
                tokens.append(itos[idx])
        texts.append(" ".join(tokens))
    return texts


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="VAE Baseline for Language Modeling")
    parser.add_argument("--dataset", type=str, default="wikitext-103",
                        choices=["wikitext-2", "wikitext-103"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_vae")
    return parser.parse_args()


def main():
    args = parse_args()

    config.DATASET = args.dataset
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.KL_WEIGHT = args.kl_weight
    config.OUTPUT_DIR = args.output_dir

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("="*60)
    print("VAE Baseline for Language Modeling")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"Dataset: {config.DATASET}")
    print(f"KL Weight: {config.KL_WEIGHT}")

    # Load data
    train_data, val_data, vocab = get_data_and_vocab(config.DATASET)
    train_loader, val_loader = get_dataloaders(train_data, val_data)

    # Create model
    model = TextVAE(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        n_layers=config.N_LAYERS,
        dropout=config.DROPOUT
    )

    print(f"\nVAE Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    model = train_vae(model, train_loader, val_loader, config)

    # Evaluate
    vae_ppl = evaluate_vae(model, val_loader, config)

    # Generate samples
    print("\n--- VAE Generated Samples ---")
    samples = model.generate(4, config.SEQ_LEN, config.DEVICE, temperature=0.8)
    texts = decode_tokens(samples, vocab, config.PAD_IDX)
    for i, text in enumerate(texts[:4]):
        print(f"  [{i+1}] {text[:100]}...")

    # Save results
    results_path = os.path.join(config.OUTPUT_DIR, "vae_results.txt")
    with open(results_path, "w") as f:
        f.write("VAE Baseline Results\n")
        f.write("="*40 + "\n")
        f.write(f"Dataset: {config.DATASET}\n")
        f.write(f"Validation Perplexity: {vae_ppl:.2f}\n")
        f.write(f"KL Weight: {config.KL_WEIGHT}\n")
        f.write(f"Latent Dim: {config.LATENT_DIM}\n")

    print(f"\nResults saved to {results_path}")

    return vae_ppl


if __name__ == "__main__":
    main()
