"""
Entropy-Guided Distillation for Discrete Diffusion Language Models (EGD-DLM)

核心创新：
1. LLaDA风格离散扩散：mask-based diffusion in token space
2. 熵调度训练 (Entropy-Scheduled Training)：用token熵加权训练损失
3. 熵引导蒸馏 (Entropy-Guided Distillation)：用熵识别难学token，蒸馏时重点关注

关键洞察：
- 高熵token = 模型不确定 = 更难学习 = 需要更多蒸馏注意力
- 将熵调度从训练阶段延伸到蒸馏阶段，形成统一框架

Course: STAT GR 5244 Unsupervised Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import math
import copy
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab as build_vocab

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    SEQ_LEN = 64
    MIN_FREQ = 2

    # Model Architecture
    VOCAB_SIZE = None  # Set after loading data
    EMBED_DIM = 256
    N_HEADS = 8
    N_LAYERS = 6
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    # Diffusion
    NUM_DIFFUSION_STEPS = 1000  # T for training
    BETA_SCHEDULE = "cosine"  # "linear" or "cosine"

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-5
    NUM_EPOCHS_TEACHER = 10
    NUM_EPOCHS_STUDENT = 5
    GRAD_CLIP = 1.0

    # Entropy Scheduling
    LAMBDA_ENTROPY = 0.5  # Weight for entropy-based loss scaling
    ENTROPY_TEMPERATURE = 1.0  # Temperature for entropy calculation

    # Distillation
    NUM_TEACHER_STEPS = 50  # Steps for teacher sampling
    NUM_STUDENT_STEPS = 8   # Steps for student sampling (accelerated)
    LAMBDA_DISTILL_ENTROPY = 1.0  # Weight for entropy-guided distillation
    ALPHA_SOFT = 0.7  # Weight for soft targets
    ALPHA_HARD = 0.3  # Weight for hard targets
    DISTILL_TEMPERATURE = 2.0  # Temperature for distillation

    # Special Tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    MASK_TOKEN = "<mask>"

    # Sampling
    NUM_SAMPLING_STEPS = 50
    CFG_SCALE = 1.5  # Classifier-free guidance scale

config = Config()

# ============================================================================
# Data Loading
# ============================================================================

def get_data_and_vocab():
    """Load WikiText-2 dataset and build vocabulary."""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenizer = get_tokenizer("basic_english")

    # Build vocabulary
    counter = Counter()
    for split in ["train", "validation", "test"]:
        for example in dataset[split]:
            text = example["text"]
            if text.strip():
                tokens = tokenizer(text.lower())
                counter.update(tokens)

    # Create vocab with special tokens
    sorted_tokens = sorted(counter.items(), key=lambda x: -x[1])
    filtered_tokens = [(t, c) for t, c in sorted_tokens if c >= config.MIN_FREQ]

    vocabulary = build_vocab(
        Counter(dict(filtered_tokens)),
        specials=[config.PAD_TOKEN, config.UNK_TOKEN, config.MASK_TOKEN]
    )
    vocabulary.set_default_index(vocabulary[config.UNK_TOKEN])

    config.VOCAB_SIZE = len(vocabulary)
    config.PAD_IDX = vocabulary[config.PAD_TOKEN]
    config.UNK_IDX = vocabulary[config.UNK_TOKEN]
    config.MASK_IDX = vocabulary[config.MASK_TOKEN]

    print(f"Vocabulary size: {config.VOCAB_SIZE}")
    print(f"Special tokens - PAD: {config.PAD_IDX}, UNK: {config.UNK_IDX}, MASK: {config.MASK_IDX}")

    # Tokenize dataset
    def tokenize_and_numericalize(examples):
        result = []
        for text in examples["text"]:
            if text.strip():
                tokens = tokenizer(text.lower())
                if len(tokens) >= 5:  # Filter very short sequences
                    ids = [vocabulary[t] for t in tokens]
                    result.append(ids)
        return result

    train_data = tokenize_and_numericalize(dataset["train"])
    val_data = tokenize_and_numericalize(dataset["validation"])

    return train_data, val_data, vocabulary, tokenizer


def create_sequences(data, seq_len):
    """Create fixed-length sequences from tokenized data."""
    sequences = []
    for ids in data:
        for i in range(0, len(ids) - seq_len + 1, seq_len // 2):  # Overlapping
            seq = ids[i:i + seq_len]
            if len(seq) == seq_len:
                sequences.append(torch.tensor(seq, dtype=torch.long))
    return sequences


def collate_fn(batch):
    """Collate batch of sequences."""
    return torch.stack(batch)


def get_dataloaders(train_data, val_data):
    """Create DataLoaders."""
    train_seqs = create_sequences(train_data, config.SEQ_LEN)
    val_seqs = create_sequences(val_data, config.SEQ_LEN)

    print(f"Training sequences: {len(train_seqs)}")
    print(f"Validation sequences: {len(val_seqs)}")

    train_loader = DataLoader(
        train_seqs,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_seqs,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True
    )

    return train_loader, val_loader

# ============================================================================
# Model Architecture
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""

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
    """Sinusoidal time embedding for diffusion timestep."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, t):
        """
        Args:
            t: (batch_size,) tensor of timesteps in [0, 1]
        Returns:
            (batch_size, d_model) time embeddings
        """
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class DiscreteDiffusionTransformer(nn.Module):
    """
    Transformer-based discrete diffusion model (LLaDA style).
    Predicts masked tokens conditioned on timestep.
    """

    def __init__(self, vocab_size, embed_dim, n_heads, n_layers,
                 dim_feedforward, dropout, max_len=512):
        super().__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_len)

        # Time embedding
        self.time_embedding = TimeEmbedding(embed_dim)

        # Transformer encoder (bidirectional, like BERT/LLaDA)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm
        self.ln_final = nn.LayerNorm(embed_dim)

        # Output projection to vocabulary
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, t, attention_mask=None):
        """
        Forward pass.

        Args:
            x: (batch_size, seq_len) token indices (may contain MASK tokens)
            t: (batch_size,) timesteps in [0, 1]
            attention_mask: optional mask for padding

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape

        # Token embeddings
        h = self.token_embedding(x)  # (B, L, D)

        # Add positional encoding
        h = self.pos_encoding(h)

        # Add time embedding (broadcast to all positions)
        t_emb = self.time_embedding(t)  # (B, D)
        h = h + t_emb.unsqueeze(1)  # (B, L, D)

        # Create attention mask if needed
        if attention_mask is not None:
            # Convert to transformer format (True = ignore)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        # Transformer
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)

        # Final layer norm and projection
        h = self.ln_final(h)
        logits = self.output_proj(h)

        return logits

# ============================================================================
# Discrete Diffusion Process (LLaDA Style)
# ============================================================================

class DiscreteDiffusion:
    """
    LLaDA-style discrete diffusion with mask-based corruption.

    Forward process: progressively mask tokens
    Reverse process: predict and unmask tokens
    """

    def __init__(self, num_steps, beta_schedule="cosine"):
        self.num_steps = num_steps
        self.beta_schedule = beta_schedule

        # Compute masking schedule (probability of masking at each timestep)
        if beta_schedule == "linear":
            self.mask_probs = torch.linspace(0, 1, num_steps + 1)
        elif beta_schedule == "cosine":
            # Cosine schedule (smoother)
            steps = torch.linspace(0, 1, num_steps + 1)
            self.mask_probs = 1 - torch.cos(steps * math.pi / 2)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

    def q_sample(self, x0, t, mask_idx):
        """
        Forward process: sample x_t given x_0.
        Randomly mask tokens with probability determined by t.

        Args:
            x0: (batch_size, seq_len) original tokens
            t: (batch_size,) timesteps in [0, 1]
            mask_idx: index of MASK token

        Returns:
            x_t: (batch_size, seq_len) masked tokens
            mask: (batch_size, seq_len) boolean mask (True = masked)
        """
        batch_size, seq_len = x0.shape
        device = x0.device

        # Masking probability for each sample
        mask_prob = t.view(-1, 1).expand(-1, seq_len)  # (B, L)

        # Sample mask
        rand = torch.rand(batch_size, seq_len, device=device)
        mask = rand < mask_prob  # True = will be masked

        # Apply mask
        x_t = x0.clone()
        x_t[mask] = mask_idx

        return x_t, mask

    def get_timestep(self, step, num_steps):
        """Convert discrete step to continuous timestep t in [0, 1]."""
        return step / num_steps

# ============================================================================
# Entropy Calculation Utilities
# ============================================================================

def compute_token_entropy(logits, temperature=1.0):
    """
    Compute per-token entropy from logits.

    Args:
        logits: (batch_size, seq_len, vocab_size)
        temperature: temperature for softmax

    Returns:
        entropy: (batch_size, seq_len) entropy values
    """
    probs = F.softmax(logits / temperature, dim=-1)
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (B, L)
    return entropy


def compute_normalized_entropy(entropy, vocab_size):
    """Normalize entropy to [0, 1] range."""
    max_entropy = math.log(vocab_size)
    return entropy / max_entropy

# ============================================================================
# Training Losses
# ============================================================================

def compute_entropy_scheduled_loss(model, x0, diffusion, mask_idx, config):
    """
    Compute entropy-scheduled training loss.

    Key Innovation: Use token entropy to weight the loss.
    High entropy tokens (uncertain) get higher weights.

    Args:
        model: DiscreteDiffusionTransformer
        x0: (batch_size, seq_len) original tokens
        diffusion: DiscreteDiffusion object
        mask_idx: MASK token index
        config: configuration object

    Returns:
        loss: scalar loss
        metrics: dict of metrics for logging
    """
    batch_size, seq_len = x0.shape
    device = x0.device

    # Sample random timesteps
    t = torch.rand(batch_size, device=device)  # [0, 1]

    # Forward diffusion: mask tokens
    x_t, mask = diffusion.q_sample(x0, t, mask_idx)

    # Predict logits
    logits = model(x_t, t)  # (B, L, V)

    # Compute base cross-entropy loss (only on masked positions)
    loss_per_token = F.cross_entropy(
        logits.view(-1, config.VOCAB_SIZE),
        x0.view(-1),
        reduction='none'
    ).view(batch_size, seq_len)

    # Compute entropy for weighting
    with torch.no_grad():
        entropy = compute_token_entropy(logits, config.ENTROPY_TEMPERATURE)
        normalized_entropy = compute_normalized_entropy(entropy, config.VOCAB_SIZE)

        # Entropy-based weights: higher entropy = higher weight
        # w_i = 1 + lambda * entropy_i
        entropy_weights = 1.0 + config.LAMBDA_ENTROPY * normalized_entropy

    # Apply weights (only on masked tokens)
    mask_float = mask.float()
    weighted_loss = loss_per_token * entropy_weights * mask_float

    # Normalize by number of masked tokens
    num_masked = mask_float.sum()
    if num_masked > 0:
        loss = weighted_loss.sum() / num_masked
    else:
        loss = weighted_loss.sum()

    # Metrics
    metrics = {
        'loss': loss.item(),
        'mean_entropy': entropy.mean().item(),
        'mask_ratio': mask_float.mean().item(),
        'mean_weight': entropy_weights[mask].mean().item() if mask.any() else 1.0
    }

    return loss, metrics

# ============================================================================
# Entropy-Guided Distillation
# ============================================================================

def compute_entropy_guided_distillation_loss(
    teacher_model,
    student_model,
    x0,
    diffusion,
    mask_idx,
    config,
    teacher_steps,
    student_steps
):
    """
    Compute entropy-guided distillation loss.

    Key Innovation: Use entropy to identify hard-to-learn tokens
    and give them more weight during distillation.

    The intuition:
    - Tokens where teacher is uncertain (high entropy) are harder
    - Student should pay more attention to learning these tokens
    - This creates a curriculum-like effect in distillation

    Args:
        teacher_model: trained teacher model
        student_model: student model to train
        x0: (batch_size, seq_len) original tokens
        diffusion: DiscreteDiffusion object
        mask_idx: MASK token index
        config: configuration object
        teacher_steps: number of steps for teacher
        student_steps: number of steps for student

    Returns:
        loss: scalar loss
        metrics: dict of metrics
    """
    batch_size, seq_len = x0.shape
    device = x0.device

    # Sample random timesteps (for training, we sample from full range)
    t = torch.rand(batch_size, device=device)

    # Forward diffusion
    x_t, mask = diffusion.q_sample(x0, t, mask_idx)

    # Teacher prediction (no gradient)
    with torch.no_grad():
        teacher_logits = teacher_model(x_t, t)  # (B, L, V)
        teacher_probs = F.softmax(teacher_logits / config.DISTILL_TEMPERATURE, dim=-1)

        # Compute teacher entropy (key for entropy-guided distillation)
        teacher_entropy = compute_token_entropy(teacher_logits, config.ENTROPY_TEMPERATURE)
        normalized_teacher_entropy = compute_normalized_entropy(teacher_entropy, config.VOCAB_SIZE)

        # Entropy-guided weights for distillation
        # Higher teacher entropy = more attention during distillation
        distill_weights = 1.0 + config.LAMBDA_DISTILL_ENTROPY * normalized_teacher_entropy

    # Student prediction
    student_logits = student_model(x_t, t)
    student_log_probs = F.log_softmax(student_logits / config.DISTILL_TEMPERATURE, dim=-1)

    # === Soft Target Loss (KL Divergence) ===
    # KL(teacher || student) weighted by entropy
    kl_loss_per_token = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='none'
    ).sum(dim=-1)  # (B, L)

    # Apply entropy-guided weights (only on masked positions)
    mask_float = mask.float()
    weighted_kl_loss = kl_loss_per_token * distill_weights * mask_float

    # === Hard Target Loss (Cross-Entropy with ground truth) ===
    hard_loss_per_token = F.cross_entropy(
        student_logits.view(-1, config.VOCAB_SIZE),
        x0.view(-1),
        reduction='none'
    ).view(batch_size, seq_len)

    weighted_hard_loss = hard_loss_per_token * distill_weights * mask_float

    # Normalize
    num_masked = mask_float.sum()
    if num_masked > 0:
        soft_loss = weighted_kl_loss.sum() / num_masked
        hard_loss = weighted_hard_loss.sum() / num_masked
    else:
        soft_loss = weighted_kl_loss.sum()
        hard_loss = weighted_hard_loss.sum()

    # Combined loss
    # Scale soft loss by temperature^2 (standard practice in distillation)
    loss = (config.ALPHA_SOFT * soft_loss * (config.DISTILL_TEMPERATURE ** 2) +
            config.ALPHA_HARD * hard_loss)

    # Compute student entropy for monitoring
    with torch.no_grad():
        student_entropy = compute_token_entropy(student_logits, config.ENTROPY_TEMPERATURE)

    metrics = {
        'loss': loss.item(),
        'soft_loss': soft_loss.item(),
        'hard_loss': hard_loss.item(),
        'teacher_entropy': teacher_entropy.mean().item(),
        'student_entropy': student_entropy.mean().item(),
        'mean_distill_weight': distill_weights[mask].mean().item() if mask.any() else 1.0,
    }

    return loss, metrics

# ============================================================================
# Sampling
# ============================================================================

@torch.no_grad()
def sample(model, batch_size, seq_len, vocab_size, mask_idx, pad_idx,
           num_steps, device, cfg_scale=1.0, use_cfg=False):
    """
    Sample from the discrete diffusion model.

    Uses ancestral sampling: start from all masks, progressively unmask.

    Args:
        model: trained model
        batch_size: number of sequences to generate
        seq_len: sequence length
        vocab_size: vocabulary size
        mask_idx: MASK token index
        pad_idx: PAD token index (for CFG unconditioned)
        num_steps: number of sampling steps
        device: torch device
        cfg_scale: classifier-free guidance scale
        use_cfg: whether to use CFG

    Returns:
        samples: (batch_size, seq_len) generated token indices
    """
    model.eval()

    # Start with all MASK tokens
    x = torch.full((batch_size, seq_len), mask_idx, dtype=torch.long, device=device)

    # Sampling schedule: unmask progressively
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)

    for i in tqdm(range(num_steps), desc="Sampling"):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]

        # Current timestep for model
        t = torch.full((batch_size,), t_now.item(), device=device)

        # Get predictions
        logits = model(x, t)  # (B, L, V)

        # Optional: Classifier-Free Guidance
        if use_cfg and cfg_scale > 1.0:
            # Unconditioned prediction (use PAD as "null" condition)
            x_uncond = torch.full_like(x, pad_idx)
            logits_uncond = model(x_uncond, t)
            logits = logits_uncond + cfg_scale * (logits - logits_uncond)

        # Sample from predictions
        probs = F.softmax(logits, dim=-1)  # (B, L, V)

        # Determine which tokens to unmask at this step
        # Unmask tokens with probability proportional to confidence
        mask_positions = (x == mask_idx)

        if mask_positions.any():
            # Sample new tokens for masked positions
            sampled_tokens = torch.multinomial(
                probs.view(-1, vocab_size),
                num_samples=1
            ).view(batch_size, seq_len)

            # Determine unmask ratio for this step
            unmask_ratio = (t_now - t_next).item()

            # For each masked position, decide whether to unmask
            # Use confidence-based selection
            max_probs, _ = probs.max(dim=-1)  # (B, L)

            # Only unmask high-confidence predictions
            confidence_threshold = torch.quantile(
                max_probs[mask_positions],
                1 - unmask_ratio
            )

            should_unmask = mask_positions & (max_probs >= confidence_threshold)

            # Update tokens
            x = torch.where(should_unmask, sampled_tokens, x)

    # Final pass: unmask any remaining MASK tokens
    final_mask = (x == mask_idx)
    if final_mask.any():
        t = torch.zeros(batch_size, device=device)
        logits = model(x, t)
        probs = F.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(
            probs.view(-1, vocab_size),
            num_samples=1
        ).view(batch_size, seq_len)
        x = torch.where(final_mask, sampled_tokens, x)

    return x


def decode_tokens(token_ids, vocab, pad_idx):
    """Convert token indices to text."""
    itos = vocab.get_itos()
    texts = []
    for seq in token_ids:
        tokens = []
        for idx in seq:
            idx = idx.item()
            if idx != pad_idx:
                tokens.append(itos[idx])
        texts.append(" ".join(tokens))
    return texts

# ============================================================================
# Training Loop
# ============================================================================

def train_teacher(model, train_loader, val_loader, diffusion, config):
    """Train the teacher model with entropy-scheduled loss."""
    print("\n" + "="*60)
    print("Training Teacher Model with Entropy-Scheduled Loss")
    print("="*60)

    model.to(config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS_TEACHER)

    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS_TEACHER):
        model.train()
        total_loss = 0
        total_entropy = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS_TEACHER}")
        for batch in pbar:
            x0 = batch.to(config.DEVICE)

            optimizer.zero_grad()

            loss, metrics = compute_entropy_scheduled_loss(
                model, x0, diffusion, config.MASK_IDX, config
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

            total_loss += metrics['loss']
            total_entropy += metrics['mean_entropy']
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'entropy': f"{metrics['mean_entropy']:.4f}",
                'weight': f"{metrics['mean_weight']:.2f}"
            })

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x0 = batch.to(config.DEVICE)
                loss, _ = compute_entropy_scheduled_loss(
                    model, x0, diffusion, config.MASK_IDX, config
                )
                val_loss += loss.item()
                val_batches += 1

        avg_train_loss = total_loss / num_batches
        avg_val_loss = val_loss / val_batches
        avg_entropy = total_entropy / num_batches

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Avg Entropy={avg_entropy:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "egd_teacher_model.pth")
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

    return model


def train_student_with_distillation(teacher_model, student_model, train_loader,
                                     val_loader, diffusion, config):
    """Train student model using entropy-guided distillation."""
    print("\n" + "="*60)
    print("Training Student with Entropy-Guided Distillation")
    print("="*60)

    teacher_model.to(config.DEVICE)
    student_model.to(config.DEVICE)
    teacher_model.eval()  # Teacher is frozen

    optimizer = AdamW(student_model.parameters(), lr=config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS_STUDENT)

    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS_STUDENT):
        student_model.train()
        total_loss = 0
        total_teacher_entropy = 0
        total_student_entropy = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Distill Epoch {epoch+1}/{config.NUM_EPOCHS_STUDENT}")
        for batch in pbar:
            x0 = batch.to(config.DEVICE)

            optimizer.zero_grad()

            loss, metrics = compute_entropy_guided_distillation_loss(
                teacher_model, student_model, x0, diffusion,
                config.MASK_IDX, config,
                config.NUM_TEACHER_STEPS, config.NUM_STUDENT_STEPS
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.GRAD_CLIP)
            optimizer.step()

            total_loss += metrics['loss']
            total_teacher_entropy += metrics['teacher_entropy']
            total_student_entropy += metrics['student_entropy']
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'soft': f"{metrics['soft_loss']:.4f}",
                'hard': f"{metrics['hard_loss']:.4f}",
                'T_ent': f"{metrics['teacher_entropy']:.2f}",
                'S_ent': f"{metrics['student_entropy']:.2f}"
            })

        scheduler.step()

        # Validation
        student_model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x0 = batch.to(config.DEVICE)
                loss, _ = compute_entropy_guided_distillation_loss(
                    teacher_model, student_model, x0, diffusion,
                    config.MASK_IDX, config,
                    config.NUM_TEACHER_STEPS, config.NUM_STUDENT_STEPS
                )
                val_loss += loss.item()
                val_batches += 1

        avg_train_loss = total_loss / num_batches
        avg_val_loss = val_loss / val_batches

        print(f"Distill Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}")
        print(f"  Teacher Entropy={total_teacher_entropy/num_batches:.4f}, "
              f"Student Entropy={total_student_entropy/num_batches:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student_model.state_dict(), "egd_student_model.pth")
            print(f"  -> Saved best student model (val_loss={best_val_loss:.4f})")

    return student_model

# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_model(model, val_loader, diffusion, config, model_name="Model"):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for batch in tqdm(val_loader, desc=f"Evaluating {model_name}"):
        x0 = batch.to(config.DEVICE)
        batch_size, seq_len = x0.shape

        # Use fixed t=0.5 for evaluation
        t = torch.full((batch_size,), 0.5, device=config.DEVICE)
        x_t, mask = diffusion.q_sample(x0, t, config.MASK_IDX)

        logits = model(x_t, t)

        loss = F.cross_entropy(
            logits[mask],
            x0[mask],
            reduction='sum'
        )

        total_loss += loss.item()
        total_tokens += mask.sum().item()

    perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

    print(f"{model_name} - Validation Perplexity: {perplexity:.2f}")
    return perplexity


def compare_sampling_speeds(teacher_model, student_model, vocab, config):
    """Compare sampling speeds between teacher and student."""
    import time

    print("\n" + "="*60)
    print("Comparing Sampling Speeds")
    print("="*60)

    batch_size = 4

    # Teacher sampling (more steps)
    teacher_model.eval()
    start_time = time.time()
    teacher_samples = sample(
        teacher_model, batch_size, config.SEQ_LEN, config.VOCAB_SIZE,
        config.MASK_IDX, config.PAD_IDX, config.NUM_TEACHER_STEPS,
        config.DEVICE
    )
    teacher_time = time.time() - start_time

    # Student sampling (fewer steps)
    student_model.eval()
    start_time = time.time()
    student_samples = sample(
        student_model, batch_size, config.SEQ_LEN, config.VOCAB_SIZE,
        config.MASK_IDX, config.PAD_IDX, config.NUM_STUDENT_STEPS,
        config.DEVICE
    )
    student_time = time.time() - start_time

    print(f"Teacher ({config.NUM_TEACHER_STEPS} steps): {teacher_time:.2f}s")
    print(f"Student ({config.NUM_STUDENT_STEPS} steps): {student_time:.2f}s")
    print(f"Speedup: {teacher_time/student_time:.2f}x")

    # Decode and show samples
    teacher_texts = decode_tokens(teacher_samples, vocab, config.PAD_IDX)
    student_texts = decode_tokens(student_samples, vocab, config.PAD_IDX)

    print("\n--- Teacher Samples ---")
    for i, text in enumerate(teacher_texts[:2]):
        print(f"  [{i+1}] {text[:100]}...")

    print("\n--- Student Samples ---")
    for i, text in enumerate(student_texts[:2]):
        print(f"  [{i+1}] {text[:100]}...")

    return teacher_samples, student_samples

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*60)
    print("EGD-DLM: Entropy-Guided Distillation for")
    print("Discrete Diffusion Language Models")
    print("="*60)
    print(f"Device: {config.DEVICE}")

    # Load data
    train_data, val_data, vocab, tokenizer = get_data_and_vocab()
    train_loader, val_loader = get_dataloaders(train_data, val_data)

    # Initialize diffusion
    diffusion = DiscreteDiffusion(config.NUM_DIFFUSION_STEPS, config.BETA_SCHEDULE)

    # ===== Stage 1: Train Teacher with Entropy-Scheduled Loss =====
    teacher_model = DiscreteDiffusionTransformer(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    )

    print(f"\nTeacher Model Parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")

    teacher_model = train_teacher(teacher_model, train_loader, val_loader, diffusion, config)

    # Evaluate teacher
    teacher_ppl = evaluate_model(teacher_model, val_loader, diffusion, config, "Teacher")

    # ===== Stage 2: Train Student with Entropy-Guided Distillation =====
    # Student has same architecture but will learn to work with fewer steps
    student_model = DiscreteDiffusionTransformer(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    )

    # Initialize student with teacher weights (warm start)
    student_model.load_state_dict(teacher_model.state_dict())

    student_model = train_student_with_distillation(
        teacher_model, student_model, train_loader, val_loader, diffusion, config
    )

    # Evaluate student
    student_ppl = evaluate_model(student_model, val_loader, diffusion, config, "Student")

    # ===== Stage 3: Compare Sampling =====
    teacher_samples, student_samples = compare_sampling_speeds(
        teacher_model, student_model, vocab, config
    )

    # ===== Summary =====
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Teacher Perplexity: {teacher_ppl:.2f}")
    print(f"Student Perplexity: {student_ppl:.2f}")
    print(f"Teacher Steps: {config.NUM_TEACHER_STEPS}")
    print(f"Student Steps: {config.NUM_STUDENT_STEPS}")
    print(f"Speedup: {config.NUM_TEACHER_STEPS / config.NUM_STUDENT_STEPS:.1f}x")

    # Save final results
    with open("egd_results.txt", "w") as f:
        f.write("EGD-DLM Results\n")
        f.write("="*40 + "\n")
        f.write(f"Teacher Perplexity: {teacher_ppl:.2f}\n")
        f.write(f"Student Perplexity: {student_ppl:.2f}\n")
        f.write(f"Teacher Steps: {config.NUM_TEACHER_STEPS}\n")
        f.write(f"Student Steps: {config.NUM_STUDENT_STEPS}\n")
        f.write(f"Speedup: {config.NUM_TEACHER_STEPS / config.NUM_STUDENT_STEPS:.1f}x\n")
        f.write("\nKey Hyperparameters:\n")
        f.write(f"  Lambda Entropy: {config.LAMBDA_ENTROPY}\n")
        f.write(f"  Lambda Distill Entropy: {config.LAMBDA_DISTILL_ENTROPY}\n")
        f.write(f"  Distill Temperature: {config.DISTILL_TEMPERATURE}\n")
        f.write(f"  Alpha Soft/Hard: {config.ALPHA_SOFT}/{config.ALPHA_HARD}\n")

    print("\nResults saved to egd_results.txt")
    print("Models saved: egd_teacher_model.pth, egd_student_model.pth")


if __name__ == "__main__":
    main()
