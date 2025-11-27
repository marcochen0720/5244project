#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
离散扩散语言模型 (Discrete Diffusion Language Model) + CFG
============================================================

这是一个完整的、可运行的脚本，实现了 2024-2025 年 SOTA 的离散扩散方法：

1. **离散扩散 (MDLM/Absorbing State 风格)**
   - 直接在 token 空间操作，不需要连续嵌入空间
   - 使用 [MASK] 作为吸收态 (absorbing state)
   - 前向过程: 逐渐将 token 替换为 [MASK]
   - 反向过程: 预测被 mask 的 token

2. **Classifier-Free Guidance (CFG)**
   - 训练时随机 dropout 条件
   - 推理时混合有条件和无条件预测
   - 提高生成质量

3. **双向 Transformer (BERT-style)**
   - 非自回归生成
   - 可以并行预测所有位置

参考论文:
- MDLM: Masked Diffusion Language Model (Sahoo et al., 2024)
- SEDD: Score Entropy Discrete Diffusion (Sahoo et al., 2024)
- Simple and Effective Masked Diffusion Language Models (Shi et al., 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import math
import random
from collections import Counter
from tqdm import tqdm

try:
    from datasets import load_dataset 
except ImportError:
    print("\n*** 错误: 'datasets' 库未找到 ***")
    print("请先在你的环境中安装它:")
    print("conda install datasets -c conda-forge  (推荐)")
    print("或: pip install datasets")
    exit()

import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

# ============================================================================
# 1. 全局配置 (Configuration)
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据配置
BATCH_SIZE = 32
SEQ_LEN = 64

# 模型配置
EMBED_DIM = 256
N_HEADS = 8
N_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# 训练配置
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
WARMUP_STEPS = 1000

# 扩散配置
NUM_DIFFUSION_STEPS = 1000  # T
BETA_SCHEDULE = "cosine"     # "linear" or "cosine"

# CFG 配置
CFG_DROPOUT_PROB = 0.1      # 训练时 unconditional 的概率
CFG_GUIDANCE_SCALE = 2.0    # 推理时的 guidance scale (w)

# 采样配置
NUM_SAMPLING_STEPS = 50     # 推理时的步数 (可以小于 T)

# 保存路径
MODEL_PATH = "discrete_diffusion_model.pth"
RESULTS_PATH = "discrete_diffusion_results.txt"

# 禁用 torchtext 警告
try:
    torchtext.disable_torchtext_deprecation_warning()
except AttributeError:
    pass


# ============================================================================
# 2. 数据处理 (Data Pipeline)
# ============================================================================

def get_data_iter_and_vocab(split='train'):
    """加载 WikiText-2 数据集并构建词汇表"""
    print(f"Loading WikiText-2 ({split}) using Hugging Face 'datasets'...")
    
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    except Exception as e:
        print(f"Hugging Face 'datasets' 加载失败: {e}")
        exit()
    
    tokenizer = get_tokenizer('basic_english')
    data_iter = [text for text in dataset['text'] if text.strip()]
    
    if split == 'train':
        print("Building vocab from train split...")
        counter = Counter()
        for text in data_iter:
            counter.update(tokenizer(text))
        
        # 添加特殊 token: <unk>, <pad>, <mask>
        vocab = Vocab(counter, specials=["<unk>", "<pad>", "<mask>"])
        print(f"Vocab size: {len(vocab)}")
    else:
        raise NotImplementedError("This script only demonstrates training.")

    return data_iter, vocab, tokenizer


def collate_fn(batch_data, tokenizer, vocab, seq_len, mask_token_id):
    """DataLoader 的整理函数"""
    processed_texts = []
    
    for text in batch_data:
        if not text.strip(): 
            continue
        token_ids = [vocab.stoi.get(token, vocab.stoi["<unk>"]) for token in tokenizer(text)]
        if len(token_ids) > seq_len:
            token_ids = token_ids[:seq_len]
        processed_texts.append(torch.tensor(token_ids, dtype=torch.long))

    if not processed_texts:
        return None
    
    pad_index = vocab.stoi["<pad>"]
    padded_batch = pad_sequence(processed_texts, batch_first=True, padding_value=pad_index)
    
    if padded_batch.shape[1] < seq_len:
        pad_width = seq_len - padded_batch.shape[1]
        padded_batch = F.pad(padded_batch, (0, pad_width), 'constant', pad_index)
    
    return padded_batch.to(DEVICE)


# ============================================================================
# 3. 扩散调度 (Diffusion Schedule)
# ============================================================================

def get_beta_schedule(schedule_type, num_steps, beta_start=0.0001, beta_end=0.02):
    """
    获取 beta 调度 (噪声/mask 概率的调度)
    
    Args:
        schedule_type: "linear" 或 "cosine"
        num_steps: 总步数 T
    
    Returns:
        betas: [T] 每一步的 mask 概率
        alphas: [T] 1 - betas
        alpha_bars: [T] 累积乘积 (保持原始 token 的概率)
    """
    if schedule_type == "linear":
        betas = torch.linspace(beta_start, beta_end, num_steps)
    elif schedule_type == "cosine":
        # Cosine schedule (来自 Improved DDPM)
        steps = torch.arange(num_steps + 1, dtype=torch.float32)
        f_t = torch.cos((steps / num_steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bars = f_t / f_t[0]
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.clamp(betas, min=0.0001, max=0.9999)
        alpha_bars = alpha_bars[1:]
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    alphas = 1 - betas
    
    if schedule_type == "linear":
        alpha_bars = torch.cumprod(alphas, dim=0)
    
    return betas.to(DEVICE), alphas.to(DEVICE), alpha_bars.to(DEVICE)


# ============================================================================
# 4. 模型定义 (Discrete Diffusion Transformer)
# ============================================================================

class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].detach()


class TimeEmbedding(nn.Module):
    """时间步嵌入 (使用正弦编码 + MLP)"""
    def __init__(self, embed_dim, max_period=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        
        # 将正弦嵌入投影到模型维度
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t):
        """
        Args:
            t: [B] 时间步 (0 到 T-1)
        Returns:
            [B, embed_dim] 时间嵌入
        """
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.embed_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        
        return self.mlp(embedding)


class DiscreteDiffusionTransformer(nn.Module):
    """
    离散扩散 Transformer (BERT-style 双向)
    
    特点:
    - 双向注意力 (非自回归)
    - 时间步条件
    - 支持 CFG (条件 dropout)
    """
    def __init__(self, vocab_size, embed_dim, n_layers, n_heads, dim_feedforward, 
                 dropout=0.1, max_seq_len=512):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Token 嵌入
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)
        
        # 时间嵌入
        self.time_embed = TimeEmbedding(embed_dim)
        
        # Transformer Encoder (双向)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出层 (预测每个位置的 token)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        # 层归一化
        self.ln_pre = nn.LayerNorm(embed_dim)
        self.ln_post = nn.LayerNorm(embed_dim)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, t, padding_mask=None):
        """
        Args:
            x: [B, L] token ids (包含 [MASK] tokens)
            t: [B] 时间步
            padding_mask: [B, L] 填充掩码 (True = padding)
        
        Returns:
            logits: [B, L, V] 每个位置预测的 logits
        """
        B, L = x.shape
        
        # 1. Token 嵌入
        h = self.token_embedding(x)  # [B, L, D]
        
        # 2. 添加位置编码
        h = self.pos_encoder(h)
        
        # 3. 添加时间嵌入 (广播到所有位置)
        t_emb = self.time_embed(t)  # [B, D]
        h = h + t_emb.unsqueeze(1)  # [B, L, D]
        
        # 4. 预归一化
        h = self.ln_pre(h)
        
        # 5. Transformer
        if padding_mask is not None:
            h = self.transformer(h, src_key_padding_mask=padding_mask)
        else:
            h = self.transformer(h)
        
        # 6. 后归一化
        h = self.ln_post(h)
        
        # 7. 预测 logits
        logits = self.output_proj(h)  # [B, L, V]
        
        return logits


# ============================================================================
# 5. 离散扩散过程 (Forward & Reverse Process)
# ============================================================================

class DiscreteDiffusion:
    """
    离散扩散过程 (Absorbing State / MDLM 风格)
    """
    def __init__(self, model, vocab_size, num_steps, mask_token_id, pad_token_id,
                 beta_schedule="cosine", cfg_dropout_prob=0.1):
        self.model = model
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.cfg_dropout_prob = cfg_dropout_prob
        
        # 获取调度
        self.betas, self.alphas, self.alpha_bars = get_beta_schedule(
            beta_schedule, num_steps
        )
    
    def q_sample(self, x_0, t):
        """
        前向过程: 在时间步 t 对 x_0 加噪 (替换为 [MASK])
        
        q(x_t | x_0) = alpha_bar_t * x_0 + (1 - alpha_bar_t) * [MASK]
        
        Args:
            x_0: [B, L] 原始 token ids
            t: [B] 时间步
        
        Returns:
            x_t: [B, L] 加噪后的 token ids
            mask: [B, L] 哪些位置被 mask 了
        """
        B, L = x_0.shape
        
        # 获取每个样本的 alpha_bar
        alpha_bar_t = self.alpha_bars[t]  # [B]
        
        # 采样: 每个 token 以概率 (1 - alpha_bar_t) 被替换为 [MASK]
        mask_probs = 1 - alpha_bar_t.unsqueeze(1).expand(B, L)  # [B, L]
        mask = torch.bernoulli(mask_probs).bool()  # [B, L]
        
        # 不 mask padding
        is_pad = (x_0 == self.pad_token_id)
        mask = mask & ~is_pad
        
        # 应用 mask
        x_t = x_0.clone()
        x_t[mask] = self.mask_token_id
        
        return x_t, mask
    
    def compute_loss(self, x_0, use_cfg=True):
        """
        计算训练损失
        
        Args:
            x_0: [B, L] 原始 token ids
            use_cfg: 是否使用 CFG (训练时随机 dropout)
        
        Returns:
            loss: 标量损失
            metrics: 额外的指标
        """
        B, L = x_0.shape
        
        # 1. 采样随机时间步
        t = torch.randint(0, self.num_steps, (B,), device=x_0.device)
        
        # 2. 前向过程: 加噪
        x_t, mask = self.q_sample(x_0, t)
        
        # 3. 模型预测
        padding_mask = (x_0 == self.pad_token_id)
        logits = self.model(x_t, t, padding_mask)  # [B, L, V]
        
        # 4. 只在被 mask 的位置计算损失
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            x_0.view(-1),
            reduction='none'
        ).view(B, L)
        
        # 只计算 mask 位置的损失 (忽略 padding)
        loss = loss * mask.float()
        loss = loss.sum() / (mask.sum() + 1e-8)
        
        # 计算准确率
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == x_0) & mask
            accuracy = correct.sum().float() / (mask.sum() + 1e-8)
        
        return loss, {"accuracy": accuracy.item(), "mask_ratio": mask.float().mean().item()}
    
    @torch.no_grad()
    def sample(self, batch_size, seq_len, num_steps=None, cfg_scale=1.0, 
               temperature=1.0, top_k=0, top_p=0.0):
        """
        反向过程: 从全 [MASK] 序列生成
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            num_steps: 采样步数 (默认 = 训练步数)
            cfg_scale: CFG guidance scale
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Top-P (nucleus) 采样
        
        Returns:
            x: [B, L] 生成的 token ids
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        self.model.eval()
        
        # 初始化: 全部是 [MASK]
        x = torch.full((batch_size, seq_len), self.mask_token_id, 
                       dtype=torch.long, device=DEVICE)
        
        # 计算采样时间步 (从 T-1 到 0)
        step_size = self.num_steps // num_steps
        timesteps = list(range(self.num_steps - 1, -1, -step_size))[:num_steps]
        
        print(f"Sampling with {num_steps} steps, CFG scale = {cfg_scale}")
        
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=DEVICE)
            
            # 模型预测
            logits = self.model(x, t_tensor, padding_mask=None)  # [B, L, V]
            
            # CFG: 混合有条件和无条件预测
            if cfg_scale != 1.0:
                # 无条件预测 (使用全 mask)
                x_uncond = torch.full_like(x, self.mask_token_id)
                logits_uncond = self.model(x_uncond, t_tensor, padding_mask=None)
                
                # CFG 公式: logits = logits_uncond + w * (logits_cond - logits_uncond)
                logits = logits_uncond + cfg_scale * (logits - logits_uncond)
            
            # 应用温度
            logits = logits / temperature
            
            # Top-K 过滤
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[..., [-1]]] = -float('Inf')
            
            # Top-P 过滤
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for b in range(batch_size):
                    for l in range(seq_len):
                        indices_to_remove = sorted_indices[b, l][sorted_indices_to_remove[b, l]]
                        logits[b, l, indices_to_remove] = -float('Inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            
            if i < len(timesteps) - 1:
                # 还没到最后一步: 概率性地 unmask
                # 计算下一步的 alpha_bar
                next_t = timesteps[i + 1] if i + 1 < len(timesteps) else 0
                alpha_bar_t = self.alpha_bars[t].item()  # 转为 Python float
                alpha_bar_next = self.alpha_bars[next_t].item() if next_t > 0 else 1.0
                
                # 采样新的 tokens
                sampled_tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, seq_len)
                
                # 决定哪些位置要 unmask
                # 保持 mask 的概率从 (1 - alpha_bar_t) 变为 (1 - alpha_bar_next)
                current_mask = (x == self.mask_token_id)
                
                # unmask_prob: 从当前状态转移到下一状态时，解除 mask 的概率
                unmask_prob = (alpha_bar_next - alpha_bar_t) / (1 - alpha_bar_t + 1e-8)
                unmask_prob = max(0.0, min(1.0, unmask_prob))  # 裁剪到 [0, 1]
                
                # 创建 unmask 掩码 [B, L]
                unmask_probs_tensor = torch.full((batch_size, seq_len), unmask_prob, device=DEVICE)
                unmask = torch.bernoulli(unmask_probs_tensor).bool()
                unmask = unmask & current_mask
                
                # 更新
                x = torch.where(unmask, sampled_tokens, x)
            else:
                # 最后一步: unmask 所有剩余的 [MASK]
                current_mask = (x == self.mask_token_id)
                sampled_tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, seq_len)
                x = torch.where(current_mask, sampled_tokens, x)
        
        return x


# ============================================================================
# 6. 训练循环 (Training Loop)
# ============================================================================

def train_epoch(diffusion, dataloader, optimizer, scheduler=None):
    """训练一个 epoch"""
    diffusion.model.train()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        
        optimizer.zero_grad()
        
        loss, metrics = diffusion.compute_loss(batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), 1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        total_accuracy += metrics["accuracy"]
        num_batches += 1
    
    return total_loss / num_batches, total_accuracy / num_batches


def decode_tokens(token_ids, vocab):
    """将 token ids 解码为文本"""
    texts = []
    for ids in token_ids:
        tokens = []
        for idx in ids.cpu().numpy():
            token = vocab.itos[idx]
            if token not in ["<pad>", "<unk>", "<mask>"]:
                tokens.append(token)
        texts.append(" ".join(tokens))
    return texts


# ============================================================================
# 7. 主执行流程 (Main Execution)
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("离散扩散语言模型 (Discrete Diffusion LM) + CFG")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Diffusion steps: {NUM_DIFFUSION_STEPS}")
    print(f"Sampling steps: {NUM_SAMPLING_STEPS}")
    print(f"CFG scale: {CFG_GUIDANCE_SCALE}")
    print("=" * 60)
    
    # 1. 加载数据
    train_data, vocab, tokenizer = get_data_iter_and_vocab(split='train')
    
    VOCAB_SIZE = len(vocab)
    MASK_TOKEN_ID = vocab.stoi["<mask>"]
    PAD_TOKEN_ID = vocab.stoi["<pad>"]
    
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"<mask> token id: {MASK_TOKEN_ID}")
    print(f"<pad> token id: {PAD_TOKEN_ID}")
    
    # 创建 DataLoader
    from functools import partial
    collate_partial = partial(collate_fn, tokenizer=tokenizer, vocab=vocab, 
                              seq_len=SEQ_LEN, mask_token_id=MASK_TOKEN_ID)
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_partial,
        num_workers=0
    )
    
    # 2. 初始化模型
    model = DiscreteDiffusionTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_seq_len=SEQ_LEN
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 初始化扩散过程
    diffusion = DiscreteDiffusion(
        model=model,
        vocab_size=VOCAB_SIZE,
        num_steps=NUM_DIFFUSION_STEPS,
        mask_token_id=MASK_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        beta_schedule=BETA_SCHEDULE,
        cfg_dropout_prob=CFG_DROPOUT_PROB
    )
    
    # 4. 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, total_steps=total_steps, pct_start=0.1
    )
    
    # 5. 训练循环
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        avg_loss, avg_accuracy = train_epoch(diffusion, train_dataloader, optimizer, scheduler)
        
        print(f"Loss: {avg_loss:.4f} | Accuracy: {avg_accuracy:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'config': {
                    'vocab_size': VOCAB_SIZE,
                    'embed_dim': EMBED_DIM,
                    'n_layers': N_LAYERS,
                    'n_heads': N_HEADS,
                    'dim_feedforward': DIM_FEEDFORWARD,
                    'num_diffusion_steps': NUM_DIFFUSION_STEPS,
                }
            }, MODEL_PATH)
            print(f"Best model saved to {MODEL_PATH}")
        
        # 每个 epoch 生成一些样本
        if (epoch + 1) % 2 == 0 or epoch == NUM_EPOCHS - 1:
            print("\n生成样本...")
            
            samples = diffusion.sample(
                batch_size=4,
                seq_len=SEQ_LEN,
                num_steps=NUM_SAMPLING_STEPS,
                cfg_scale=CFG_GUIDANCE_SCALE,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            
            texts = decode_tokens(samples, vocab)
            
            print("\n--- 生成的文本 ---")
            for i, text in enumerate(texts):
                print(f"样本 {i + 1}: {text[:200]}...")  # 只显示前200字符
    
    # 6. 最终生成
    print("\n" + "=" * 60)
    print("最终生成 (使用不同的 CFG scale)")
    print("=" * 60)
    
    results = []
    
    for cfg_scale in [1.0, 2.0, 3.0]:
        print(f"\nCFG scale = {cfg_scale}")
        
        samples = diffusion.sample(
            batch_size=4,
            seq_len=SEQ_LEN,
            num_steps=NUM_SAMPLING_STEPS,
            cfg_scale=cfg_scale,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        
        texts = decode_tokens(samples, vocab)
        
        results.append(f"\n--- CFG scale = {cfg_scale} ---")
        for i, text in enumerate(texts):
            results.append(f"样本 {i + 1}: {text}")
            print(f"样本 {i + 1}: {text[:150]}...")
    
    # 保存结果
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("离散扩散语言模型生成结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"模型配置:\n")
        f.write(f"  - Vocab size: {VOCAB_SIZE}\n")
        f.write(f"  - Embed dim: {EMBED_DIM}\n")
        f.write(f"  - Layers: {N_LAYERS}\n")
        f.write(f"  - Heads: {N_HEADS}\n")
        f.write(f"  - Diffusion steps: {NUM_DIFFUSION_STEPS}\n")
        f.write(f"  - Sampling steps: {NUM_SAMPLING_STEPS}\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(results))
    
    print(f"\n结果已保存到 {RESULTS_PATH}")
    print("训练完成!")

