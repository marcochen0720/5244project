#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
项目第三部分：实现一致性蒸馏 (CD)
(基于版本 8 的工作代码)

这是一个完整的、可运行的脚本，实现了你提案中的所有三个部分：
1. (阶段1) 训练一个 ES-RF Transformer "教师" 模型 (来自第二部分)。
2. (阶段2) 冻结教师模型。
3. (阶段2) 训练一个 "学生" 模型，使用 "双重一致性蒸馏" 损失函数 [cite: 58-59]，
   使其学会在嵌入空间和离SYN空间中保持一致。
4. (采样) 对比 "教师" (慢速, 50步) 和 "学生" (快速, 4步) [cite: 54-57] 的生成结果。

(*** 新增 ***)
5. (保存) 将训练好的 "教师" 和 "学生" 模型权重保存到 .pth 文件。
6. (保存) 将最终生成的样本保存到 generation_results.txt 文件。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical

import math 
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
LEARNING_RATE_TEACHER = 1e-4
LEARNING_RATE_STUDENT = 1e-4

# --- *** 训练周期配置 *** ---
NUM_EPOCHS_TEACHER = 5   
NUM_EPOCHS_STUDENT = 3   

# --- *** 采样步数配置 *** ---
N_STEPS_TEACHER = 50     
N_FEW_STEPS_STUDENT = 4  

# --- *** 损失函数超参数 *** ---
LAMBDA_ENTROPY = 0.1     
N_DISTILL_STEPS = 50     
LAMBDA_DISCRETE = 0.1    

# --- Transformer 超参数 ---
EMBED_DIM = 128       
N_HEADS = 4           
N_LAYERS = 4          
DIM_FEEDFORWARD = 512 

# --- *** 新增：保存路径 *** ---
TEACHER_MODEL_PATH = "teacher_model.pth"
STUDENT_MODEL_PATH = "student_model.pth"
RESULTS_TXT_PATH = "generation_results.txt"


# 禁用 torchtext 的弃用警告
try:
    torchtext.disable_torchtext_deprecation_warning()
except AttributeError:
    pass

# --- 2. 数据处理 (Data Pipeline) ---
# (这部分无需更改)

def get_data_iter_and_vocab(split='train'):
    print(f"Loading WikiText-2 ({split}) using Hugging Face 'datasets'...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    except Exception as e:
        print(f"Hugging Face 'datasets' 加载失败: {e}")
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
# (这部分无需更改)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
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
    def __init__(self, embed_dim):
        super(TimeEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
    def forward(self, t):
        return self.mlp(t)

class LanguageRFTransformer(nn.Module):
    def __init__(self, embed_dim, vocab_size, n_layers, n_heads, dim_feedforward):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.time_embed = TimeEmbedding(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            device=DEVICE
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(embed_dim, embed_dim) 
        self.to_logits = nn.Linear(embed_dim, vocab_size)

    def forward(self, x_t, t):
        x_with_pos = self.pos_encoder(x_t)
        t_embed = self.time_embed(t) 
        model_input = x_with_pos + t_embed
        transformer_output = self.transformer(model_input)
        predicted_velocity = self.output_layer(transformer_output)
        predicted_x1 = x_t + (1.0 - t) * predicted_velocity
        logits = self.to_logits(predicted_x1)
        return predicted_velocity, logits

# --- 4. 训练逻辑 (Training Logic) ---
# (这部分无需更改，除了上一个 bug 修复)

def get_rf_train_sample(x0, x1, t_val=None):
    """ (已更新：允许传入 t) """
    if t_val is None:
        t = torch.rand(x0.shape[0], 1, 1, device=x0.device)
    else:
        # 修复: t_val 已经是我们想要的 [B, 1, 1] 张量 t。
        t = t_val
    x_t = (1.0 - t) * x0 + t * x1
    target_v = x1 - x0
    return x_t, t, target_v

def calculate_proxy_entropy(logits):
    dist = Categorical(logits=logits)
    return dist.entropy() 

def calculate_es_rf_loss(model, x0_embeds, x1_embeds, lambda_entropy):
    """ (教师模型的损失函数) """
    x_t, t, target_v = get_rf_train_sample(x0_embeds, x1_embeds)
    predicted_v, logits = model(x_t, t)
    rf_loss_per_position = (predicted_v - target_v).pow(2).mean(dim=-1)
    entropy_per_position = calculate_proxy_entropy(logits)
    loss_weights = 1.0 + lambda_entropy * entropy_per_position.detach()
    weighted_loss = rf_loss_per_position * loss_weights
    final_loss = weighted_loss.mean()
    return final_loss, rf_loss_per_position.mean(), entropy_per_position.mean()

def calculate_consistency_loss(teacher_model, student_model, x0_embeds, x1_embeds, n_steps, lambda_discrete):
    """ (学生模型的损失函数，已修复) """
    n = torch.randint(1, n_steps, (x0_embeds.shape[0],), device=x0_embeds.device)
    t_n_plus_1 = n.float() / n_steps
    t_n = (n - 1).float() / n_steps
    dt = 1.0 / n_steps
    
    x_t_n_plus_1, _, _ = get_rf_train_sample(
        x0_embeds, x1_embeds, 
        t_val=t_n_plus_1.reshape(-1, 1, 1)
    )

    with torch.no_grad():
        v_teacher_for_step, _ = teacher_model(x_t_n_plus_1, t_n_plus_1.reshape(-1, 1, 1))
        x_t_n = x_t_n_plus_1 - v_teacher_for_step * dt
        teacher_v_target, teacher_logits_target = teacher_model(
            x_t_n.detach(),
            t_n.reshape(-1, 1, 1).detach()
        )

    student_v_pred, student_logits_pred = student_model(
        x_t_n_plus_1, 
        t_n_plus_1.reshape(-1, 1, 1)
    )
    
    loss_embed = F.mse_loss(student_v_pred, teacher_v_target.detach())
    
    loss_discrete = F.kl_div(
        F.log_softmax(student_logits_pred, dim=-1),
        F.softmax(teacher_logits_target.detach(), dim=-1),
        reduction='batchmean'
    )
    
    final_loss = loss_embed + lambda_discrete * loss_discrete
    return final_loss, loss_embed, loss_discrete


# --- 5. 采样/生成 (Sampling / Generation) ---
# (这部分无需更改)

def sample_rf(model, x0_noise, n_steps=100):
    """ (慢速采样器，用于教师) """
    x_t = x0_noise.to(DEVICE)
    dt = 1.0 / n_steps
    model.eval() 
    
    for i in range(n_steps):
        t_val = (i / n_steps)
        t = torch.full((x_t.shape[0], 1, 1), t_val, device=DEVICE)
        with torch.no_grad():
            v_pred, _ = model(x_t, t)
        x_t = x_t + v_pred * dt # Euler step
    
    return x_t 

def sample_cd(model, x0_noise, n_steps=4):
    """ (快速采样器) [cite: 54-57] """
    print(f"一致性采样 (NFE={n_steps})...")
    x_t = sample_rf(model, x0_noise, n_steps=n_steps)
    print("一致性采样完成。")
    return x_t

def decode_embeddings_to_text(model, x1_pred_embeds, vocab):
    """ (无需更改) """
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
    print(f"--- ES-RF w/ Transformer & CD (Full Project) ---")
    
    # 1. 加载数据
    train_iter, vocab, tokenizer = get_data_iter_and_vocab(split='train')
    VOCAB_SIZE = len(vocab)
    print(f"Actual VOCAB_SIZE set to {VOCAB_SIZE}")
    print(f"Vocab '<unk>' index: {vocab.stoi['<unk>']}")
    print(f"Vocab '<pad>' index: {vocab.stoi['<pad>']}")

    from functools import partial
    collate_partial = partial(collate_fn, tokenizer=tokenizer, vocab=vocab, seq_len=SEQ_LEN)
    train_data_list = list(train_iter)
    train_dataloader = DataLoader(
        train_data_list, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_partial
    )
    
    # 2. 初始化模型
    embedding_layer = nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(DEVICE)
    
    teacher_model = LanguageRFTransformer(
        embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, n_layers=N_LAYERS,
        n_heads=N_HEADS, dim_feedforward=DIM_FEEDFORWARD
    ).to(DEVICE)
    
    params_teacher = list(teacher_model.parameters()) + list(embedding_layer.parameters())
    optimizer_teacher = optim.Adam(params_teacher, lr=LEARNING_RATE_TEACHER)
    
    # --- *** 阶段 1: 训练教师模型 *** ---
    print("\n--- 阶段 1: 开始训练 教师 (Teacher) 模型 ---")
    
    for epoch in range(NUM_EPOCHS_TEACHER):
        print(f"\n--- [教师] Epoch {epoch+1}/{NUM_EPOCHS_TEACHER} ---")
        teacher_model.train() 
        embedding_layer.train()
        
        for i, batch_token_ids in enumerate(train_dataloader):
            if batch_token_ids.shape[0] == 0: continue
            
            optimizer_teacher.zero_grad()
            x1_embeds = embedding_layer(batch_token_ids)
            x0_noise = torch.randn_like(x1_embeds)
            
            loss, base_rf_loss, entropy = calculate_es_rf_loss(
                teacher_model, x0_noise, x1_embeds, LAMBDA_ENTROPY
            )
            loss.backward()
            optimizer_teacher.step()
            
            if (i + 1) % 100 == 0: 
                print(f"  [教师 Epoch {epoch+1}, Step {i+1}] 加权损失: {loss.item():.6f} | (基础RF: {base_rf_loss.item():.6f}, 熵: {entropy.item():.4f})")
                
    print("\n--- 教师模型 训练完成 ---")
    
    # --- *** 新增：保存教师模型 *** ---
    print(f"\n--- 正在保存教师模型到 {TEACHER_MODEL_PATH} ---")
    torch.save({
        'teacher_model_state_dict': teacher_model.state_dict(),
        'embedding_layer_state_dict': embedding_layer.state_dict(),
        'vocab': vocab, # 保存词汇表，以便将来加载
    }, TEACHER_MODEL_PATH)
    print(f"教师模型已保存。")

    # --- 冻结教师模型和嵌入层 ---
    teacher_model.eval()
    embedding_layer.eval()

    # --- *** 初始化学生模型 *** ---
    student_model = LanguageRFTransformer(
        embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, n_layers=N_LAYERS,
        n_heads=N_HEADS, dim_feedforward=DIM_FEEDFORWARD
    ).to(DEVICE)
    student_model.load_state_dict(teacher_model.state_dict())
    
    optimizer_student = optim.Adam(student_model.parameters(), lr=LEARNING_RATE_STUDENT)
    
    # --- *** 阶段 2: 训练学生模型 *** ---
    print("\n--- 阶段 2: 开始蒸馏 学生 (Student) 模型 ---")

    for epoch in range(NUM_EPOCHS_STUDENT):
        print(f"\n--- [学生] Epoch {epoch+1}/{NUM_EPOCHS_STUDENT} ---")
        student_model.train() 
        
        for i, batch_token_ids in enumerate(train_dataloader):
            if batch_token_ids.shape[0] == 0: continue
            
            optimizer_student.zero_grad()
            
            with torch.no_grad():
                x1_embeds = embedding_layer(batch_token_ids)
                x0_noise = torch.randn_like(x1_embeds)
            
            loss, loss_e, loss_d = calculate_consistency_loss(
                teacher_model, student_model, x0_noise, x1_embeds,
                n_steps=N_DISTILL_STEPS, lambda_discrete=LAMBDA_DISCRETE
            )
            
            loss.backward()
            optimizer_student.step()
            
            if (i + 1) % 100 == 0: 
                print(f"  [学生 Epoch {epoch+1}, Step {i+1}] 一致性损失: {loss.item():.6f} | (嵌入: {loss_e.item():.6f}, 离散: {loss_d.item():.6f})")

    print("\n--- 学生模型 蒸馏完成 ---")
    
    # --- *** 新增：保存学生模型 *** ---
    print(f"\n--- 正在保存学生模型到 {STUDENT_MODEL_PATH} ---")
    torch.save({
        'student_model_state_dict': student_model.state_dict(),
        'embedding_layer_state_dict': embedding_layer.state_dict(),
        'vocab': vocab,
    }, STUDENT_MODEL_PATH)
    print(f"学生模型已保存。")
    
    # --- 7. 生成文本 (最终对比) ---
    print("\n--- 开始从噪声生成文本 (最终对比) ---")
    
    z_noise = torch.randn(2, SEQ_LEN, EMBED_DIM).to(DEVICE) # 生成 2 个样本
    
    # --- 教师 (慢速) ---
    print(f"\n--- 教师 (慢速, NFE={N_STEPS_TEACHER}) ---")
    generated_embeds_teacher = sample_rf(teacher_model, z_noise, n_steps=N_STEPS_TEACHER)
    generated_texts_teacher = decode_embeddings_to_text(teacher_model, generated_embeds_teacher, vocab)
        
    # --- 学生 (快速) ---
    print(f"\n--- 学生 (快速, NFE={N_FEW_STEPS_STUDENT}) ---")
    generated_embeds_student = sample_cd(student_model, z_noise, n_steps=N_FEW_STEPS_STUDENT)
    generated_texts_student = decode_embeddings_to_text(student_model, generated_embeds_student, vocab)

    # --- *** 新增：保存生成结果到文件 *** ---
    print(f"\n--- 正在保存生成结果到 {RESULTS_TXT_PATH} ---")
    with open(RESULTS_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(f"--- 教师 (慢速, NFE={N_STEPS_TEACHER}) ---\n")
        for idx, text in enumerate(generated_texts_teacher):
            f.write(f"  教师样本 {idx+1}: {text}\n")
            print(f"  教师样本 {idx+1}: {text}") # (同时在控制台打印)
            
        f.write(f"\n--- 学生 (快速, NFE={N_FEW_STEPS_STUDENT}) ---\n")
        for idx, text in enumerate(generated_texts_student):
            f.write(f"  学生样本 {idx+1}: {text}\n")
            print(f"  学生样本 {idx+1}: {text}") # (同时在控制台打印)
    
    print(f"生成结果已保存到 {RESULTS_TXT_PATH}。")