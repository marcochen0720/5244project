#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
离散扩散语言模型 V2 - 增强版
============================

改进：
1. 使用 WikiText-103 (比 WikiText-2 大 100 倍)
2. 使用 GPT-2 Tokenizer (BPE 分词，效果更好)
3. 更大的模型配置
4. 更好的数据预处理 

使用方法:
    pip install transformers
    python discrete_diffusion_v2.py --epochs 30 --batch_size 64 --fp16
"""

import os
import argparse
import time
import math
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

try:
    from transformers import GPT2Tokenizer
except ImportError:
    print("请安装: pip install transformers")
    exit()

try:
    from datasets import load_dataset
except ImportError:
    print("请安装: pip install datasets")
    exit()

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# ============================================================================
# 命令行参数
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Discrete Diffusion LM V2")
    
    # 数据参数
    parser.add_argument("--dataset", type=str, default="wikitext-103", 
                        choices=["wikitext-2", "wikitext-103"],
                        help="数据集选择")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数 (用于快速测试)")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # 模型参数
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--dim_feedforward", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seq_len", type=int, default=128)
    
    # 扩散参数
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--sampling_steps", type=int, default=100)
    parser.add_argument("--beta_schedule", type=str, default="cosine")
    
    # CFG 参数
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_dir", type=str, default="checkpoints_v2")
    parser.add_argument("--sample_interval", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点恢复训练，例如: --resume checkpoints_v2/checkpoint_epoch_10.pth")
    
    return parser.parse_args()


# ============================================================================
# 设备设置
# ============================================================================

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU")
    return device


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 数据处理 (使用 GPT-2 Tokenizer)
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len, mask_token_id):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mask_token_id = mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        
        # 预处理：过滤空文本，并进行分词
        print("预处理数据...")
        self.examples = []
        
        for text in tqdm(texts, desc="Tokenizing"):
            if not text.strip():
                continue
            
            # 分词
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # 如果太短，跳过
            if len(tokens) < 10:
                continue
            
            # 如果太长，切分成多个样本
            for i in range(0, len(tokens) - seq_len + 1, seq_len // 2):
                chunk = tokens[i:i + seq_len]
                if len(chunk) == seq_len:
                    self.examples.append(chunk)
        
        print(f"✅ 创建了 {len(self.examples)} 个训练样本")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def load_data(args, tokenizer):
    """加载数据集"""
    print(f"\n📚 Loading {args.dataset}...")
    
    if args.dataset == "wikitext-103":
        dataset_name = "wikitext-103-raw-v1"
    else:
        dataset_name = "wikitext-2-raw-v1"
    
    dataset = load_dataset("wikitext", dataset_name, split="train")
    texts = dataset['text']
    
    if args.max_samples:
        texts = texts[:args.max_samples]
    
    print(f"   原始文本数: {len(texts)}")
    
    # 添加 [MASK] token 到 tokenizer
    mask_token = "<|mask|>"
    tokenizer.add_special_tokens({'additional_special_tokens': [mask_token]})
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    
    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集
    train_dataset = TextDataset(texts, tokenizer, args.seq_len, mask_token_id)
    
    return train_dataset, mask_token_id


# ============================================================================
# 扩散调度
# ============================================================================

def get_beta_schedule(schedule_type, num_steps, device):
    if schedule_type == "cosine":
        steps = torch.arange(num_steps + 1, dtype=torch.float32, device=device)
        f_t = torch.cos((steps / num_steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bars = f_t / f_t[0]
        alpha_bars = alpha_bars[1:]
        betas = 1 - (alpha_bars / torch.cat([torch.ones(1, device=device), alpha_bars[:-1]]))
        betas = torch.clamp(betas, min=0.0001, max=0.9999)
    else:
        betas = torch.linspace(0.0001, 0.02, num_steps, device=device)
        alpha_bars = torch.cumprod(1 - betas, dim=0)
    
    return betas, alpha_bars


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
    def __init__(self, vocab_size, embed_dim, n_layers, n_heads, dim_feedforward, 
                 dropout=0.1, max_seq_len=512):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)
        self.time_embed = SinusoidalTimeEmbedding(embed_dim)
        
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.ln_final = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # 权重绑定
        self.output_proj.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, t, padding_mask=None):
        h = self.token_embedding(x)
        h = self.pos_encoder(h)
        h = self.input_proj(h)
        h = h + self.time_embed(t).unsqueeze(1)
        
        h = self.transformer(h, src_key_padding_mask=padding_mask)
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
        
        self.betas, self.alpha_bars = get_beta_schedule(beta_schedule, num_steps, device)
    
    def q_sample(self, x_0, t):
        """前向过程"""
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
        B, L = x_0.shape
        t = torch.randint(0, self.num_steps, (B,), device=self.device)
        x_t, mask = self.q_sample(x_0, t)
        
        padding_mask = (x_0 == self.pad_token_id)
        
        with autocast(enabled=use_amp):
            logits = self.model(x_t, t, padding_mask)
            
            # 只在 mask 位置计算损失
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                x_0.view(-1),
                reduction='none',
                ignore_index=self.pad_token_id
            ).view(B, L)
            
            loss = loss * mask.float()
            loss = loss.sum() / (mask.sum() + 1e-8)
        
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == x_0) & mask
            accuracy = correct.sum().float() / (mask.sum() + 1e-8)
        
        return loss, {"accuracy": accuracy.item()}
    
    @torch.no_grad()
    def sample(self, batch_size, seq_len, num_steps=None, cfg_scale=1.0,
               temperature=0.8, top_k=50, top_p=0.9):
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
            probs_flat = probs.view(-1, self.vocab_size)
            row_sums = probs_flat.sum(dim=-1, keepdim=True)
            zero_rows = (row_sums < 1e-8).expand_as(probs_flat)
            uniform_prob = 1.0 / self.vocab_size
            probs_flat = torch.where(zero_rows, torch.full_like(probs_flat, uniform_prob), probs_flat)
            
            # 重新归一化
            probs_flat = probs_flat / (probs_flat.sum(dim=-1, keepdim=True) + 1e-8)
            
            sampled_tokens = torch.multinomial(probs_flat, 1).view(batch_size, seq_len)
            
            # 渐进式 unmask
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
# 训练和工具函数
# ============================================================================

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(diffusion, dataloader, optimizer, scheduler, scaler, args, epoch):
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
        
        pbar.set_postfix({
            "loss": f"{total_loss / num_batches:.4f}",
            "acc": f"{total_accuracy / num_batches:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    return total_loss / num_batches, total_accuracy / num_batches


def decode_and_print(samples, tokenizer, num_show=4):
    """解码并打印样本"""
    texts = []
    for i, ids in enumerate(samples[:num_show]):
        text = tokenizer.decode(ids, skip_special_tokens=True)
        text = text.replace('\n', ' ').strip()
        texts.append(text)
        print(f"  [{i+1}] {text[:150]}...")
    return texts


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()
    args.device = setup_device()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🚀 Discrete Diffusion Language Model V2 - Enhanced")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"   Model: {args.n_layers}L, {args.embed_dim}D, {args.n_heads}H")
    print(f"   Seq length: {args.seq_len}")
    print(f"   FP16: {args.fp16}")
    
    # 加载 GPT-2 Tokenizer
    print("\n📝 Loading GPT-2 Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # 加载数据
    train_dataset, mask_token_id = load_data(args, tokenizer)
    
    VOCAB_SIZE = len(tokenizer)
    PAD_TOKEN_ID = tokenizer.pad_token_id
    
    print(f"\n📊 Vocab size: {VOCAB_SIZE}")
    print(f"   Mask token ID: {mask_token_id}")
    print(f"   Pad token ID: {PAD_TOKEN_ID}")
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"   Batches per epoch: {len(dataloader)}")
    
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
        mask_token_id=mask_token_id,
        pad_token_id=PAD_TOKEN_ID,
        beta_schedule=args.beta_schedule,
        device=args.device
    )
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    scaler = GradScaler() if args.fp16 else None
    
    print(f"\n   Total steps: {total_steps}, Warmup: {warmup_steps}")
    
    # 从检查点恢复
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\n📂 Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('loss', float('inf'))
            print(f"   Resumed from epoch {start_epoch}, loss: {best_loss:.4f}")
        else:
            print(f"⚠️  Checkpoint not found: {args.resume}")
    
    # 训练
    print("\n" + "=" * 70)
    print("🏋️ Starting training...")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        avg_loss, avg_accuracy = train_epoch(
            diffusion, dataloader, optimizer, scheduler, scaler, args, epoch + 1
        )
        
        epoch_time = time.time() - start_time
        
        print(f"\n📊 Epoch {epoch + 1}/{args.epochs}: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}, Time={epoch_time:.1f}s")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'tokenizer_name': 'gpt2',
                'mask_token_id': mask_token_id,
                'args': vars(args),
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"   ✅ Best model saved!")
        
        # 每 5 个 epoch 保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'tokenizer_name': 'gpt2',
                'mask_token_id': mask_token_id,
                'args': vars(args),
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path}")
        
        # 生成样本
        if (epoch + 1) % args.sample_interval == 0 or epoch == args.epochs - 1:
            print("\n🎨 Generating samples...")
            samples = diffusion.sample(4, args.seq_len, args.sampling_steps, args.cfg_scale)
            decode_and_print(samples, tokenizer)
    
    # 最终生成
    print("\n" + "=" * 70)
    print("🎉 Training complete! Final generation...")
    print("=" * 70)
    
    all_results = []
    
    for cfg_scale in [1.0, 2.0, 3.0]:
        print(f"\n📝 CFG scale = {cfg_scale}")
        samples = diffusion.sample(4, args.seq_len, args.sampling_steps, cfg_scale)
        texts = decode_and_print(samples, tokenizer)
        all_results.append((cfg_scale, texts))
    
    # 保存结果
    results_path = os.path.join(args.save_dir, "generation_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("Discrete Diffusion LM V2 - Generation Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.n_layers}L, {args.embed_dim}D, {num_params/1e6:.1f}M params\n")
        f.write(f"Best loss: {best_loss:.4f}\n")
        f.write("=" * 70 + "\n\n")
        
        for cfg_scale, texts in all_results:
            f.write(f"\n--- CFG scale = {cfg_scale} ---\n")
            for i, text in enumerate(texts):
                f.write(f"[{i+1}] {text}\n\n")
    
    print(f"\n💾 Results saved to {results_path}")
    print(f"💾 Best model saved to {os.path.join(args.save_dir, 'best_model.pth')}")
    print("\n✅ All done!")


if __name__ == "__main__":
    main()

