#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
离散扩散语言模型 - GPU 优化版本
===============================

针对 GPU 训练的优化版本，包含：
1. 混合精度训练 (AMP) - 加速 2x
2. 梯度累积 - 支持更大的有效 batch size
3. 梯度裁剪 - 稳定训练
4. 检查点保存/恢复 - 支持断点续训
5. TensorBoard 日志 - 可视化训练过程
6. 更多训练 epochs 和更大模型
7. 学习率 warmup + cosine decay

使用方法:
    python discrete_diffusion_gpu.py --epochs 50 --batch_size 64

依赖:
    pip install torch datasets torchtext tqdm tensorboard
"""

import os
import argparse
import time
import math
import random
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("TensorBoard not found. Install with: pip install tensorboard")

try:
    from datasets import load_dataset
except ImportError:
    print("Please install: pip install datasets")
    exit()

import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

# 禁用警告
try:
    torchtext.disable_torchtext_deprecation_warning()
except:
    pass


# ============================================================================
# 命令行参数
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Discrete Diffusion LM - GPU Training")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup 步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    
    # 模型参数
    parser.add_argument("--embed_dim", type=int, default=512, help="嵌入维度")
    parser.add_argument("--n_layers", type=int, default=8, help="Transformer 层数")
    parser.add_argument("--n_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="FFN 维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--seq_len", type=int, default=128, help="序列长度")
    
    # 扩散参数
    parser.add_argument("--diffusion_steps", type=int, default=1000, help="扩散步数 T")
    parser.add_argument("--sampling_steps", type=int, default=100, help="采样步数")
    parser.add_argument("--beta_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    
    # CFG 参数
    parser.add_argument("--cfg_dropout", type=float, default=0.1, help="CFG 训练 dropout")
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="CFG 推理 scale")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--fp16", action="store_true", help="使用混合精度训练")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="保存目录")
    parser.add_argument("--log_interval", type=int, default=100, help="日志间隔")
    parser.add_argument("--sample_interval", type=int, default=5, help="采样间隔 (epochs)")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    
    return parser.parse_args()


# ============================================================================
# 设备设置
# ============================================================================

def setup_device():
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (training will be slow)")
    return device


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 数据处理
# ============================================================================

def get_data_and_vocab():
    """加载数据和词汇表"""
    print("\n📚 Loading WikiText-2 dataset...")
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_data = [text for text in dataset['text'] if text.strip()]
    
    tokenizer = get_tokenizer('basic_english')
    
    print("📖 Building vocabulary...")
    counter = Counter()
    for text in tqdm(train_data, desc="Tokenizing"):
        counter.update(tokenizer(text))
    
    # 只保留出现次数 >= 2 的词 (减少词汇表大小)
    min_freq = 2
    filtered_counter = Counter({k: v for k, v in counter.items() if v >= min_freq})
    
    vocab = Vocab(filtered_counter, specials=["<unk>", "<pad>", "<mask>"])
    
    print(f"✅ Vocab size: {len(vocab)} (min_freq={min_freq})")
    print(f"   <unk>={vocab.stoi['<unk>']}, <pad>={vocab.stoi['<pad>']}, <mask>={vocab.stoi['<mask>']}")
    
    return train_data, vocab, tokenizer


class TextDataset(torch.utils.data.Dataset):
    """文本数据集"""
    def __init__(self, data, tokenizer, vocab, seq_len):
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.seq_len = seq_len
        self.pad_id = vocab.stoi["<pad>"]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer(text)
        ids = [self.vocab.stoi.get(t, self.vocab.stoi["<unk>"]) for t in tokens]
        
        # 截断或填充
        if len(ids) > self.seq_len:
            ids = ids[:self.seq_len]
        else:
            ids = ids + [self.pad_id] * (self.seq_len - len(ids))
        
        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch):
    """批次整理函数"""
    return torch.stack(batch)


# ============================================================================
# 扩散调度
# ============================================================================

def get_beta_schedule(schedule_type, num_steps, device):
    """获取 beta 调度"""
    if schedule_type == "linear":
        betas = torch.linspace(0.0001, 0.02, num_steps, device=device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
    elif schedule_type == "cosine":
        steps = torch.arange(num_steps + 1, dtype=torch.float32, device=device)
        f_t = torch.cos((steps / num_steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bars = f_t / f_t[0]
        alpha_bars = alpha_bars[1:]
        betas = 1 - (alpha_bars / torch.cat([torch.ones(1, device=device), alpha_bars[:-1]]))
        betas = torch.clamp(betas, min=0.0001, max=0.9999)
        alphas = 1 - betas
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")
    
    return betas, alphas, alpha_bars


# ============================================================================
# 模型定义
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t):
        half_dim = self.embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embed_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding)


class DiscreteDiffusionTransformer(nn.Module):
    """离散扩散 Transformer"""
    def __init__(self, vocab_size, embed_dim, n_layers, n_heads, dim_feedforward, 
                 dropout=0.1, max_seq_len=512):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)
        self.time_embed = SinusoidalTimeEmbedding(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True  # Pre-LN (更稳定)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.ln_final = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # 权重绑定 (tie weights)
        self.output_proj.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, t, padding_mask=None):
        B, L = x.shape
        
        h = self.token_embedding(x)
        h = self.pos_encoder(h)
        h = h + self.time_embed(t).unsqueeze(1)
        
        if padding_mask is not None:
            h = self.transformer(h, src_key_padding_mask=padding_mask)
        else:
            h = self.transformer(h)
        
        h = self.ln_final(h)
        logits = self.output_proj(h)
        
        return logits


# ============================================================================
# 离散扩散过程
# ============================================================================

class DiscreteDiffusion:
    def __init__(self, model, vocab_size, num_steps, mask_token_id, pad_token_id,
                 beta_schedule="cosine", device="cuda"):
        self.model = model
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.device = device
        
        self.betas, self.alphas, self.alpha_bars = get_beta_schedule(
            beta_schedule, num_steps, device
        )
    
    def q_sample(self, x_0, t):
        """前向过程: 加噪"""
        B, L = x_0.shape
        
        alpha_bar_t = self.alpha_bars[t].view(B, 1)
        mask_probs = 1 - alpha_bar_t
        mask = torch.bernoulli(mask_probs.expand(B, L)).bool()
        
        # 不 mask padding
        is_pad = (x_0 == self.pad_token_id)
        mask = mask & ~is_pad
        
        x_t = x_0.clone()
        x_t[mask] = self.mask_token_id
        
        return x_t, mask
    
    def compute_loss(self, x_0, use_amp=False):
        """计算训练损失"""
        B, L = x_0.shape
        
        t = torch.randint(0, self.num_steps, (B,), device=self.device)
        x_t, mask = self.q_sample(x_0, t)
        
        padding_mask = (x_0 == self.pad_token_id)
        
        with autocast(enabled=use_amp):
            logits = self.model(x_t, t, padding_mask)
            
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                x_0.view(-1),
                reduction='none'
            ).view(B, L)
            
            # 只计算 mask 位置
            loss = loss * mask.float()
            loss = loss.sum() / (mask.sum() + 1e-8)
        
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == x_0) & mask
            accuracy = correct.sum().float() / (mask.sum() + 1e-8)
        
        return loss, {"accuracy": accuracy.item(), "mask_ratio": mask.float().mean().item()}
    
    @torch.no_grad()
    def sample(self, batch_size, seq_len, num_steps=None, cfg_scale=1.0,
               temperature=0.8, top_k=50, top_p=0.9):
        """采样生成"""
        if num_steps is None:
            num_steps = self.num_steps
        
        self.model.eval()
        
        x = torch.full((batch_size, seq_len), self.mask_token_id, 
                       dtype=torch.long, device=self.device)
        
        step_size = max(1, self.num_steps // num_steps)
        timesteps = list(range(self.num_steps - 1, -1, -step_size))[:num_steps]
        
        for i, t in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            
            logits = self.model(x, t_tensor, padding_mask=None)
            
            # CFG
            if cfg_scale != 1.0:
                x_uncond = torch.full_like(x, self.mask_token_id)
                logits_uncond = self.model(x_uncond, t_tensor, padding_mask=None)
                logits = logits_uncond + cfg_scale * (logits - logits_uncond)
            
            logits = logits / temperature
            
            # Top-K
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[..., [-1]]] = -float('Inf')
            
            # Top-P
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')
            
            # 安全的 softmax：处理全是 -inf 的情况
            probs = F.softmax(logits, dim=-1)
            
            # 检查并修复 nan/inf
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 如果某行全为 0，使用均匀分布
            row_sums = probs.view(-1, self.vocab_size).sum(dim=-1, keepdim=True)
            zero_rows = (row_sums == 0).expand_as(probs.view(-1, self.vocab_size))
            uniform_prob = 1.0 / self.vocab_size
            probs_flat = probs.view(-1, self.vocab_size)
            probs_flat[zero_rows] = uniform_prob
            probs = probs_flat.view(batch_size, seq_len, -1)
            
            # 重新归一化
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
            
            sampled_tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, seq_len)
            
            if i < len(timesteps) - 1:
                next_t = timesteps[i + 1] if i + 1 < len(timesteps) else 0
                alpha_bar_t = self.alpha_bars[t].item()
                alpha_bar_next = self.alpha_bars[next_t].item() if next_t > 0 else 1.0
                
                current_mask = (x == self.mask_token_id)
                unmask_prob = max(0.0, min(1.0, (alpha_bar_next - alpha_bar_t) / (1 - alpha_bar_t + 1e-8)))
                unmask = torch.bernoulli(torch.full_like(x, unmask_prob, dtype=torch.float)).bool()
                unmask = unmask & current_mask
                
                x = torch.where(unmask, sampled_tokens, x)
            else:
                current_mask = (x == self.mask_token_id)
                x = torch.where(current_mask, sampled_tokens, x)
        
        return x


# ============================================================================
# 学习率调度
# ============================================================================

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Warmup + Cosine decay"""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 训练循环
# ============================================================================

def train_epoch(diffusion, dataloader, optimizer, scheduler, scaler, args, epoch, writer=None):
    """训练一个 epoch"""
    diffusion.model.train()
    
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(pbar):
        batch = batch.to(args.device)
        
        loss, metrics = diffusion.compute_loss(batch, use_amp=args.fp16)
        loss = loss / args.grad_accum_steps
        
        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (step + 1) % args.grad_accum_steps == 0:
            if args.fp16:
                scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), args.max_grad_norm)
            
            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * args.grad_accum_steps
        total_accuracy += metrics["accuracy"]
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            "loss": f"{total_loss / num_batches:.4f}",
            "acc": f"{total_accuracy / num_batches:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        # TensorBoard 日志
        if writer and step % args.log_interval == 0:
            global_step = epoch * len(dataloader) + step
            writer.add_scalar("train/loss", loss.item() * args.grad_accum_steps, global_step)
            writer.add_scalar("train/accuracy", metrics["accuracy"], global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
    
    return total_loss / num_batches, total_accuracy / num_batches


def decode_tokens(token_ids, vocab):
    """解码 tokens"""
    texts = []
    for ids in token_ids:
        tokens = []
        for idx in ids.cpu().numpy():
            token = vocab.itos[idx]
            if token not in ["<pad>", "<unk>", "<mask>"]:
                tokens.append(token)
        texts.append(" ".join(tokens))
    return texts


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, args, path):
    """保存检查点"""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if args.fp16 else None,
        "loss": loss,
        "args": vars(args),
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, args):
    """加载检查点"""
    checkpoint = torch.load(path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if args.fp16 and checkpoint["scaler_state_dict"]:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()
    
    # 设置
    args.device = setup_device()
    set_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # TensorBoard
    writer = None
    if HAS_TENSORBOARD:
        log_dir = os.path.join(args.save_dir, f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        writer = SummaryWriter(log_dir)
        print(f"📊 TensorBoard logs: {log_dir}")
    
    print("\n" + "=" * 60)
    print("🚀 Discrete Diffusion Language Model - GPU Training")
    print("=" * 60)
    
    # 打印配置
    print(f"\n📋 Configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size} x {args.grad_accum_steps} (grad accum)")
    print(f"   Learning rate: {args.lr}")
    print(f"   Model: {args.n_layers} layers, {args.embed_dim} dim, {args.n_heads} heads")
    print(f"   Diffusion steps: {args.diffusion_steps}")
    print(f"   FP16: {args.fp16}")
    
    # 加载数据
    train_data, vocab, tokenizer = get_data_and_vocab()
    
    VOCAB_SIZE = len(vocab)
    MASK_TOKEN_ID = vocab.stoi["<mask>"]
    PAD_TOKEN_ID = vocab.stoi["<pad>"]
    
    # 创建数据集和 DataLoader
    dataset = TextDataset(train_data, tokenizer, vocab, args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    print(f"\n📦 Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")
    
    # 创建模型
    model = DiscreteDiffusionTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.seq_len
    ).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # 创建扩散过程
    diffusion = DiscreteDiffusion(
        model=model,
        vocab_size=VOCAB_SIZE,
        num_steps=args.diffusion_steps,
        mask_token_id=MASK_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        beta_schedule=args.beta_schedule,
        device=args.device
    )
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98)
    )
    
    # 学习率调度器
    total_steps = len(dataloader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)
    
    # 混合精度
    scaler = GradScaler() if args.fp16 else None
    
    # 恢复训练
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        print(f"\n📂 Resuming from {args.resume}")
        start_epoch, best_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, args
        )
        start_epoch += 1
        print(f"   Resumed at epoch {start_epoch}, best loss: {best_loss:.4f}")
    
    # 训练循环
    print("\n" + "=" * 60)
    print("🏋️ Starting training...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        avg_loss, avg_accuracy = train_epoch(
            diffusion, dataloader, optimizer, scheduler, scaler, args, epoch, writer
        )
        
        epoch_time = time.time() - start_time
        
        print(f"\n📊 Epoch {epoch + 1}/{args.epochs} Summary:")
        print(f"   Loss: {avg_loss:.4f} | Accuracy: {avg_accuracy:.4f}")
        print(f"   Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, avg_loss, args,
                os.path.join(args.save_dir, "best_model.pth")
            )
            print(f"   ✅ Best model saved!")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, avg_loss, args,
                os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            )
        
        # 生成样本
        if (epoch + 1) % args.sample_interval == 0 or epoch == args.epochs - 1:
            print("\n🎨 Generating samples...")
            
            samples = diffusion.sample(
                batch_size=4,
                seq_len=args.seq_len,
                num_steps=args.sampling_steps,
                cfg_scale=args.cfg_scale,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            
            texts = decode_tokens(samples, vocab)
            
            print("\n--- Generated Text ---")
            for i, text in enumerate(texts):
                print(f"  [{i+1}] {text[:150]}...")
            
            if writer:
                for i, text in enumerate(texts):
                    writer.add_text(f"samples/sample_{i}", text, epoch)
    
    # 最终生成
    print("\n" + "=" * 60)
    print("🎉 Training complete! Final generation with different CFG scales...")
    print("=" * 60)
    
    results = []
    
    for cfg_scale in [1.0, 2.0, 3.0]:
        print(f"\n📝 CFG scale = {cfg_scale}")
        
        samples = diffusion.sample(
            batch_size=4,
            seq_len=args.seq_len,
            num_steps=args.sampling_steps,
            cfg_scale=cfg_scale,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        
        texts = decode_tokens(samples, vocab)
        
        results.append(f"\n--- CFG scale = {cfg_scale} ---")
        for i, text in enumerate(texts):
            results.append(f"Sample {i+1}: {text}")
            print(f"  [{i+1}] {text[:100]}...")
    
    # 保存结果
    results_path = os.path.join(args.save_dir, "generation_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("Discrete Diffusion LM - Generation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.n_layers} layers, {args.embed_dim} dim\n")
        f.write(f"Trained for: {args.epochs} epochs\n")
        f.write(f"Vocab size: {VOCAB_SIZE}\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(results))
    
    print(f"\n💾 Results saved to {results_path}")
    print(f"💾 Best model saved to {os.path.join(args.save_dir, 'best_model.pth')}")
    
    if writer:
        writer.close()
    
    print("\n✅ All done!")


if __name__ == "__main__":
    main()

