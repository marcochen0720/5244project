#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
项目第一部分：Rectified Flow (RF) 语言模型基线
(版本 7: 修复了 'Vocab' object is not callable 的问题)

这是一个可运行的脚本，演示了如何：
1. 使用 Hugging Face 'datasets' 加载 WikiText-2 数据集。
2. (手动) 处理文本和构建词汇表以绕过旧的 API。
3. 实现一个在词嵌入空间 (embedding space) 运行的 RF 模型。
4. 训练模型学习从高斯噪声 (x0) 到真实词嵌入 (x1) 的速度场 (velocity field)。
5. 从噪声 (x0) 开始采样，生成新的文本 (x1)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- 新的导入 ---
# (我们现在需要 collections.Counter)
from collections import Counter

# 安装: pip install datasets (或 conda install datasets -c conda-forge)
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
# (我们不再使用 build_vocab_from_iterator, 而是直接使用 Vocab 类)
from torchtext.vocab import Vocab

# --- 1. 全局配置 (Configuration) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16       # 批次大小
SEQ_LEN = 32          # 句子（序列）的最大长度
EMBED_DIM = 64        # 词嵌入维度 (设置得很小，用于快速演示)
# VOCAB_SIZE = 10000  # (这现在只是一个占位符，真实大小将从数据中构建)
LEARNING_RATE = 1e-4  # 学习率
NUM_EPOCHS = 5        # 训练轮数 (设置得很小，用于快速演示)
N_STEPS = 50          # 采样（生成）时的步数

# 禁用 torchtext 的弃用警告
try:
    torchtext.disable_torchtext_deprecation_warning()
except AttributeError:
    pass

# --- 2. 数据处理 (Data Pipeline) ---

def get_data_iter_and_vocab(split='train'):
    """
    加载 WikiText-2 数据集，构建词汇表，并返回数据迭代器
    (*** 已更新为使用 Hugging Face 'datasets' ***)
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
        
        # --- *** API 修复 v5 (最终) *** ---
        # 1. 创建一个新的迭代器用于构建词汇表
        vocab_iter = (text for text in dataset['text'] if text.strip())
        
        # 2. 手动对所有文本进行分词
        tokenized_iter = map(tokenizer, vocab_iter)
        
        # 3. 手动构建词频
        counter = Counter(token for tokens in tokenized_iter for token in tokens)
        
        # 4. 使用 Vocab 构造函数，它在旧版本中也支持 'specials'
        #    这将自动使 '<unk>' (索引 0) 成为默认值
        vocab = Vocab(counter, specials=["<unk>", "<pad>"])
        
        # 5. *** (已删除) ***: 删除 vocab.set_default_index(vocab["<unk>"])
        #    因为它在旧版本中不存在，而且在第 4 步中已经是隐式设置了。
        # --- *** 修复结束 *** ---
        
        print(f"Vocab size: {len(vocab)}")
    else:
        raise NotImplementedError("This script only demonstrates training. Vocab should be built once.")

    return data_iter, vocab, tokenizer

def collate_fn(batch_data, tokenizer, vocab, seq_len):
    """
    DataLoader 的整理函数
    (*** 此函数已更新以修复 'not callable' 错误 ***)
    """
    texts, processed_texts = [], []
    for text in batch_data:
        if not text.strip(): 
            continue
        
        # --- *** API 修复 v5 *** ---
        # 错误: token_ids = vocab(tokenizer(text))
        # 修复: 手动使用 vocab.stoi (string-to-index) 字典
        token_ids = [vocab.stoi[token] for token in tokenizer(text)]
        # --- *** 修复结束 *** ---
        
        if len(token_ids) > seq_len:
            token_ids = token_ids[:seq_len]
        
        processed_texts.append(torch.tensor(token_ids, dtype=torch.long))

    if not processed_texts:
        return torch.tensor([], dtype=torch.long).to(DEVICE)
    
    # *** 确保这里的 padding_value 与上面 'specials' 的顺序一致 ***
    # Vocab(..., specials=["<unk>", "<pad>"])
    # 意味着: vocab["<unk>"] == 0
    #        vocab["<pad>"] == 1
    # 我们需要用 1 来填充
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

def calculate_rf_loss(model, x0_embeds, x1_embeds):
    """
    (*** 此函数无需更改 ***)
    """
    x_t, t, target_v = get_rf_train_sample(x0_embeds, x1_embeds)
    predicted_v, _ = model(x_t, t)
    loss = F.mse_loss(predicted_v, target_v)
    return loss

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
    (*** 此函数已更新以修复 'lookup_tokens' 问题 ***)
    """
    print("解码生成的词嵌入...")
    model.eval()
    
    with torch.no_grad():
         final_logits = model.to_logits(x1_pred_embeds)
        
    token_ids = torch.argmax(final_logits, dim=-1) 
    
    all_texts = []
    for i in range(token_ids.shape[0]):
        # --- *** API 修复 v5 *** ---
        # 错误: tokens = vocab.lookup_tokens(token_ids[i].cpu().numpy())
        # 修复: 手动使用 vocab.itos (index-to-string) 列表
        tokens = [vocab.itos[idx] for idx in token_ids[i].cpu().numpy()]
        # --- *** 修复结束 *** ---
        
        filtered_tokens = [t for t in tokens if t not in ["<pad>", "<unk>"]]
        all_texts.append(" ". join(filtered_tokens))
        
    return all_texts


# --- 6. 主执行流程 (Main Execution) ---

if __name__ == "__main__":
    
    print(f"Using device: {DEVICE}")
    
    # 1. 加载数据
    train_iter, vocab, tokenizer = get_data_iter_and_vocab(split='train')

    # 更新 VOCAB_SIZE 为真实大小
    VOCAB_SIZE = len(vocab)
    print(f"Actual VOCAB_SIZE set to {VOCAB_SIZE}")
    
    # 打印 <pad> token 的索引，确保它是 1
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
    
    print("--- 开始训练 RF 基线模型 ---")
    
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
            
            # 3. 计算 RF 损失
            loss = calculate_rf_loss(model, x0_noise, x1_embeds)
            
            # 4. 反向传播
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 50 == 0:
                print(f"  [Epoch {epoch+1}, Step {i+1}] 损失: {loss.item():.6f}")

    print("\n--- 训练完成 ---")

    # --- 7. 生成文本 (Generation) ---
    print("\n--- 开始从噪声生成文本 ---")
    
    z_noise = torch.randn(2, SEQ_LEN, EMBED_DIM).to(DEVICE) # 生成 2 个样本
    generated_embeds = sample_rf(model, z_noise, n_steps=N_STEPS)
    generated_texts = decode_embeddings_to_text(model, generated_embeds, vocab)
    
    print("\n--- 生成的文本 (免责声明：预计是乱码) ---")
    for idx, text in enumerate(generated_texts):
        print(f"样本 {idx+1}: {text}")