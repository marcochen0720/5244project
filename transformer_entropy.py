#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
项目第二部分 (升级版)：实现带 Transformer 的 ES-RF
(基于版本 7 的工作代码)

这是一个可运行的脚本，演示了如何：
1. (同上) 加载数据和构建词汇表。
2. (升级) 定义一个基于 Transformer 的 LanguageRFTransformer 模型。
   - 包含了 PositionalEncoding (位置编码)
   - 包含了 TimeEmbedding (时间嵌入)
3. (同上) 实现一个 `calculate_es_rf_loss`，它使用熵来加权损失。
4. (同上) 从噪声中采样生成文本。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical

import math # (Transformer 需要)
from collections import Counter
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

# --- 1. 全局配置 (Configuration) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16       
SEQ_LEN = 32          
LEARNING_RATE = 1e-4  
NUM_EPOCHS = 5        
N_STEPS = 50          
LAMBDA_ENTROPY = 0.1 

# --- *** Transformer 超参数 *** ---
# (为了 Transformer，EMBED_DIM 必须能被 N_HEADS 整除)
EMBED_DIM = 128       # (从 64 提升)
N_HEADS = 4           # (EMBED_DIM % N_HEADS == 0)
N_LAYERS = 4          # Transformer 层的数量
DIM_FEEDFORWARD = 512 # (Transformer 内部 MLP 的维度)

# 禁用 torchtext 的弃用警告
try:
    torchtext.disable_torchtext_deprecation_warning()
except AttributeError:
    pass

# --- 2. 数据处理 (Data Pipeline) ---

def get_data_iter_and_vocab(split='train'):
    """
    (*** 此函数无需更改 ***)
    """
    print(f"Loading WikiText-2 ({split}) using Hugging Face 'datasets'...")
    
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    except Exception as e:
        print(f"Hugging Face 'datasets' 加载失败: {e}")
        print("请检查你的网络连接。")
        exit()
    
    tokenizer = get_tokenizer('basic_english')
    data_iter = (text for text in dataset['text'] if text.strip())
    
    if split == 'train':
        print("Building vocab from train split...")
        vocab_iter = (text for text in dataset['text'] if text.strip())
        counter = Counter(token for tokens in map(tokenizer, vocab_iter) for token in tokens)
        vocab = Vocab(counter, specials=["<unk>", "<pad>"])
        print(f"Vocab size: {len(vocab)}")
    else:
        raise NotImplementedError("This script only demonstrates training. Vocab should be built once.")

    return data_iter, vocab, tokenizer

def collate_fn(batch_data, tokenizer, vocab, seq_len):
    """
    (*** 此函数无需更改 ***)
    """
    texts, processed_texts = [], []
    for text in batch_data:
        if not text.strip(): 
            continue
        token_ids = [vocab.stoi[token] for token in tokenizer(text)]
        if len(token_ids) > seq_len:
            token_ids = token_ids[:seq_len]
        processed_texts.append(torch.tensor(token_ids, dtype=torch.long))

    if not processed_texts:
        return torch.tensor([], dtype=torch.long).to(DEVICE)
    
    pad_index = vocab["<pad>"]
    padded_batch = pad_sequence(processed_texts, batch_first=True, padding_value=pad_index)
    
    if padded_batch.shape[1] < seq_len:
        pad_width = seq_len - padded_batch.shape[1]
        padded_batch = F.pad(padded_batch, (0, pad_width), 'constant', pad_index)
        
    return padded_batch.to(DEVICE)

# --- 3. 模型定义 (Model Definition) ---

# --- *** 新增：位置编码 *** ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # d_model = EMBED_DIM
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 形状: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 形状: [BATCH_SIZE, SEQ_LEN, EMBED_DIM]
        # self.pe[:, :x.size(1), :] 形状: [1, SEQ_LEN, EMBED_DIM]
        return x + self.pe[:, :x.size(1), :].detach()

# --- *** 新增：时间嵌入 *** ---
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(TimeEmbedding, self).__init__()
        # 将 t (1维) 映射到 embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, t):
        # t 形状: [BATCH_SIZE, 1, 1]
        # 输出形状: [BATCH_SIZE, 1, EMBED_DIM]
        return self.mlp(t)

# --- *** 升级：RF Transformer 模型 *** ---
class LanguageRFTransformer(nn.Module):
    def __init__(self, embed_dim, vocab_size, n_layers, n_heads, dim_feedforward):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # 1. 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # 2. 时间嵌入
        self.time_embed = TimeEmbedding(embed_dim)
        
        # 3. Transformer 主体
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True, # (重要: 我们的数据是 [B, L, D])
            device=DEVICE
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # 4. 输出层 (预测速度 v)
        self.output_layer = nn.Linear(embed_dim, embed_dim) 
        
        # 5. Logits 层 (用于熵)
        self.to_logits = nn.Linear(embed_dim, vocab_size)

    def forward(self, x_t, t):
        # x_t 形状: [BATCH_SIZE, SEQ_LEN, EMBED_DIM]
        # t 形状: [BATCH_SIZE, 1, 1]
        
        # 1. 添加位置信息
        x_with_pos = self.pos_encoder(x_t)
        
        # 2. 计算时间嵌入
        t_embed = self.time_embed(t) # 形状: [B, 1, D]
        
        # 3. 将时间加到所有词元上 (广播)
        model_input = x_with_pos + t_embed
        
        # 4. 通过 Transformer
        transformer_output = self.transformer(model_input)
        
        # 5. 预测速度 v
        predicted_velocity = self.output_layer(transformer_output)
        
        # 6. (同上) 推断 x1，用于计算 logits
        predicted_x1 = x_t + (1.0 - t) * predicted_velocity
        logits = self.to_logits(predicted_x1)
        
        return predicted_velocity, logits

# --- 4. RF 训练逻辑 (RF Training Logic) ---

def get_rf_train_sample(x0, x1):
    """
    (*** 此函数无需更改 ***)
    """
    t = torch.rand(x0.shape[0], 1, 1, device=x0.device)
    x_t = (1.0 - t) * x0 + t * x1
    target_v = x1 - x0
    return x_t, t, target_v

def calculate_proxy_entropy(logits):
    """
    (*** 此函数无需更改 ***)
    """
    dist = Categorical(logits=logits)
    entropy = dist.entropy() # 形状: [BATCH_SIZE, SEQ_LEN]
    return entropy

def calculate_es_rf_loss(model, x0_embeds, x1_embeds, lambda_entropy):
    """
    (*** 此函数无需更改 ***)
    """
    x_t, t, target_v = get_rf_train_sample(x0_embeds, x1_embeds)
    predicted_v, logits = model(x_t, t)
    rf_loss_per_position = (predicted_v - target_v).pow(2).mean(dim=-1)
    entropy_per_position = calculate_proxy_entropy(logits)
    loss_weights = 1.0 + lambda_entropy * entropy_per_position.detach()
    weighted_loss = rf_loss_per_position * loss_weights
    final_loss = weighted_loss.mean()
    return final_loss, rf_loss_per_position.mean(), entropy_per_position.mean()


# --- 5. 采样/生成 (Sampling / Generation) ---

def sample_rf(model, x0_noise, n_steps=100):
    """
    (*** 此函数无需更改 ***)
    """
    print(f"RF 采样 (NFE={n_steps})...")
    x_t = x0_noise.to(DEVICE)
    dt = 1.0 / n_steps
    model.eval() 
    
    for i in range(n_steps):
        t_val = (i / n_steps)
        t = torch.full((x_t.shape[0], 1, 1), t_val, device=DEVICE)
        with torch.no_grad():
            v_pred, _ = model(x_t, t)
        x_t = x_t + v_pred * dt
    
    print("RF 采样完成。")
    return x_t 

def decode_embeddings_to_text(model, x1_pred_embeds, vocab):
    """
    (*** 此函数无需更改 ***)
    """
    print("解码生成的词嵌入...")
    model.eval()
    
    with torch.no_grad():
         final_logits = model.to_logits(x1_pred_embeds)
        
    token_ids = torch.argmax(final_logits, dim=-1) 
    
    all_texts = []
    for i in range(token_ids.shape[0]):
        tokens = [vocab.itos[idx] for idx in token_ids[i].cpu().numpy()]
        filtered_tokens = [t for t in tokens if t not in ["<pad>", "<unk>"]]
        all_texts.append(" ". join(filtered_tokens))
        
    return all_texts


# --- 6. 主执行流程 (Main Execution) ---

if __name__ == "__main__":
    
    print(f"Using device: {DEVICE}")
    print(f"--- ES-RF w/ Transformer ---")
    print(f"EMBED_DIM: {EMBED_DIM}, N_HEADS: {N_HEADS}, N_LAYERS: {N_LAYERS}")
    
    # 1. 加载数据
    train_iter, vocab, tokenizer = get_data_iter_and_vocab(split='train')

    VOCAB_SIZE = len(vocab)
    print(f"Actual VOCAB_SIZE set to {VOCAB_SIZE}")
    print(f"Vocab '<unk>' index: {vocab.stoi['<unk>']}")
    print(f"Vocab '<pad>' index: {vocab.stoi['<pad>']}")

    # 创建 DataLoader
    from functools import partial
    collate_partial = partial(collate_fn, tokenizer=tokenizer, vocab=vocab, seq_len=SEQ_LEN)
    train_data_list = list(train_iter)
    train_dataloader = DataLoader(
        train_data_list,
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_partial
    )
    
    # 2. 初始化模型
    # (*** 已更新为使用新的 Transformer ***)
    embedding_layer = nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(DEVICE)
    
    model = LanguageRFTransformer(
        embed_dim=EMBED_DIM,
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dim_feedforward=DIM_FEEDFORWARD
    ).to(DEVICE)
    
    params = list(model.parameters()) + list(embedding_layer.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    
    print("--- 开始训练 ES-RF Transformer 模型 ---")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train() 
        embedding_layer.train()
        
        for i, batch_token_ids in enumerate(train_dataloader):
            if batch_token_ids.shape[0] == 0: 
                continue
                
            optimizer.zero_grad()
            
            x1_embeds = embedding_layer(batch_token_ids)
            x0_noise = torch.randn_like(x1_embeds)
            
            # (*** 训练循环无需更改 ***)
            loss, base_rf_loss, entropy = calculate_es_rf_loss(
                model, x0_noise, x1_embeds, LAMBDA_ENTROPY
            )
            
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 50 == 0:
                print(f"  [Epoch {epoch+1}, Step {i+1}] 加权损失: {loss.item():.6f} | (基础RF损失: {base_rf_loss.item():.6f}, 熵: {entropy.item():.4f})")
                
    print("\n--- 训练完成 ---")

    # --- 7. 生成文本 (Generation) ---
    print("\n--- 开始从噪声生成文本 ---")
    
    z_noise = torch.randn(2, SEQ_LEN, EMBED_DIM).to(DEVICE) # 生成 2 个样本
    generated_embeds = sample_rf(model, z_noise, n_steps=N_STEPS)
    generated_texts = decode_embeddings_to_text(model, generated_embeds, vocab)
    
    print("\n--- 生成的文本 (免责声明：可能还是乱码，但会更有结构) ---")
    for idx, text in enumerate(generated_texts):
        print(f"样本 {idx+1}: {text}")