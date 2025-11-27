#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
项目第二部分：实现 Entropy-Scheduled (ES) Rectified Flow
(基于版本 7 的工作代码)

这是一个可运行的脚本，演示了如何：
1. (同上) 加载数据和构建词汇表。
2. (同上) 定义 RF 模型。
3. (新增) 实现一个 `calculate_proxy_entropy` 函数。
4. (新增) 实现一个 `calculate_es_rf_loss`，它使用熵来加权损失，
   从而将学习预算分配给高不确定性的位置。
5. (同上) 从噪声中采样生成文本。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical

# --- 新的导入 ---
from collections import Counter
try:
    from datasets import load_dataset 
except ImportError:
    print("\n*** 错误: 'datasets' 库未找到 ***")
    print("请先在你的环境中安装它:")
    print("conda install datasets -c conda-forge  (推荐)")
    print("或: pip install datasets")
    exit()

# --- 仍然需要的 torchtext 导入 ---
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

# --- 1. 全局配置 (Configuration) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16       
SEQ_LEN = 32          
EMBED_DIM = 64        
LEARNING_RATE = 1e-4  
NUM_EPOCHS = 5        
N_STEPS = 50          

# --- *** 新的超参数 *** ---
# 这是你提案中的核心贡献：熵调度的权重。
# 值为 0.0 时，此脚本等同于第一部分（标准 RF）。
# 值 > 0 时，模型会更关注高熵（不确定）的词元。
LAMBDA_ENTROPY = 0.1 

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

class LanguageRFModel(nn.Module):
    """
    (*** 此函数无需更改 ***)
    """
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        self.model_body = nn.Sequential(
            nn.Linear(embed_dim + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim) 
        )
        self.to_logits = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, t):
        t_broadcast = t.expand(-1, x.shape[1], 1)
        net_input = torch.cat([x, t_broadcast], dim=-1)
        predicted_velocity = self.model_body(net_input)
        predicted_x1 = x + (1.0 - t) * predicted_velocity
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

# --- *** 新函数：计算熵 *** ---
def calculate_proxy_entropy(logits):
    """
    计算代理熵 (proxy entropy)
    logits: [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]
    """
    # 1. 在最后一个维度 (VOCAB_SIZE) 上计算 softmax
    #    我们使用 Categorical 分布，它在内部为我们处理 softmax
    #    这在数值上比 logits.softmax(-1).log() 更稳定
    dist = Categorical(logits=logits)
    
    # 2. 计算每个词元位置的熵
    entropy = dist.entropy() # 形状: [BATCH_SIZE, SEQ_LEN]
    
    return entropy

# --- *** 新函数：ES-RF 损失 *** ---
def calculate_es_rf_loss(model, x0_embeds, x1_embeds, lambda_entropy):
    """
    计算 Entropy-Scheduled (ES) RF 损失
    """
    # 1. (同上) 采样 x_t, t 和目标速度
    x_t, t, target_v = get_rf_train_sample(x0_embeds, x1_embeds)
    
    # 2. (同上) 使用模型预测速度和 logits
    predicted_v, logits = model(x_t, t)
    
    # 3. 计算基础 RF 损失 (注意：在词元级别，而不是批次级别)
    #    (predicted_v - target_v) 形状: [B, L, D]
    #    .pow(2).mean(dim=-1) -> 计算嵌入维度 D 上的 MSE
    #    结果形状: [BATCH_SIZE, SEQ_LEN]
    rf_loss_per_position = (predicted_v - target_v).pow(2).mean(dim=-1)
    
    # 4. 计算熵 (不确定性)
    #    结果形状: [BATCH_SIZE, SEQ_LEN]
    entropy_per_position = calculate_proxy_entropy(logits)
    
    # 5. 计算损失权重
    #    我们希望熵 *高* 的时候，损失权重也 *高*
    #    权重 = 1.0 (基础) + lambda * entropy
    #    .detach() 很重要：我们不希望通过熵来反向传播梯度
    #    我们只使用熵的值作为调度的“信号”
    loss_weights = 1.0 + lambda_entropy * entropy_per_position.detach()
    
    # 6. 计算加权损失
    #    (形状 [B, L]) * (形状 [B, L]) -> 形状 [B, L]
    weighted_loss = rf_loss_per_position * loss_weights
    
    # 7. (同上) 取所有位置的平均值，得到最终的标量损失
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
    print(f"--- Entropy-Scheduling (ES-RF) ---")
    print(f"熵权重 (LAMBDA_ENTROPY): {LAMBDA_ENTROPY}")
    
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
    embedding_layer = nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(DEVICE)
    model = LanguageRFModel(EMBED_DIM, VOCAB_SIZE).to(DEVICE)
    
    params = list(model.parameters()) + list(embedding_layer.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    
    print("--- 开始训练 ES-RF 模型 ---")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train() 
        embedding_layer.train()
        
        for i, batch_token_ids in enumerate(train_dataloader):
            if batch_token_ids.shape[0] == 0: 
                continue
                
            optimizer.zero_grad()
            
            # 1. 准备 x1 (真实数据)
            x1_embeds = embedding_layer(batch_token_ids)
            
            # 2. 准备 x0 (噪声)
            x0_noise = torch.randn_like(x1_embeds)
            
            # --- *** 训练步骤已更新 *** ---
            # 3. 计算 ES-RF 损失
            #    (旧: loss = calculate_rf_loss(model, x0_noise, x1_embeds))
            loss, base_rf_loss, entropy = calculate_es_rf_loss(
                model, x0_noise, x1_embeds, LAMBDA_ENTROPY
            )
            
            # 4. 反向传播
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
    
    print("\n--- 生成的文本 (免责声明：预计是乱码) ---")
    for idx, text in enumerate(generated_texts):
        print(f"样本 {idx+1}: {text}")