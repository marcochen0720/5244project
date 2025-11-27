#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
项目最终版：实现带精炼器 (Refiner) 的 ES-RF-CD
(基于版本 9 的工作代码)

这是一个完整的、可运行的脚本，实现了你提案中的所有三个部分 + 2025 SOTA 改进：
1. (阶段1) 训练一个 ES-RF Transformer "教师" 模型。
2. (阶段2) 蒸馏一个 "学生" 草稿模型 (Draft Model)。
3. (新增 阶段3) 训练一个 "精炼器" (Refiner) 模型，它学会"修正"学生模型的草稿。
4. (新增 阶段4) 对比 教师(慢)、学生(快-草稿)、精炼器(快-最终) 的结果。
5. (保存) 保存所有三个模型和最终的文本结果。
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
LEARNING_RATE_REFINER = 1e-4 # (新增)

# --- *** 训练周期配置 *** ---
NUM_EPOCHS_TEACHER = 5   
NUM_EPOCHS_STUDENT = 3   
NUM_EPOCHS_REFINER = 3   # (新增) 精炼器也需要训练

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
REFINER_MODEL_PATH = "refiner_model.pth" # (新增)
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
    
    pad_index = vocab.stoi["<pad>"] # (确保我们使用的是 <pad> 的索引)
    
    padded_batch = pad_sequence(processed_texts, batch_first=True, padding_value=pad_index)
    
    if padded_batch.shape[1] < seq_len:
        pad_width = seq_len - padded_batch.shape[1]
        padded_batch = F.pad(padded_batch, (0, pad_width), 'constant', pad_index)
    return padded_batch.to(DEVICE)

# --- 3. 模型定义 (Model Definition) ---

class PositionalEncoding(nn.Module):
    # (无需更改)
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
    # (无需更改)
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
    # (教师和学生模型) (无需更改)
    def __init__(self, embed_dim, vocab_size, n_layers, n_heads, dim_feedforward):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.time_embed = TimeEmbedding(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=dim_feedforward, batch_first=True, device=DEVICE
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

# --- *** 新增：精炼器 (Refiner) 模型 *** ---
class LanguageRefinerModel(nn.Module):
    def __init__(self, embed_dim, vocab_size, n_layers, n_heads, dim_feedforward):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # 精炼器也需要位置编码
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # 核心：Transformer *Decoder*
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=dim_feedforward, batch_first=True, device=DEVICE
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # 输出层 (预测词汇)
        self.to_logits = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt_embeds, memory_embeds, tgt_mask):
        # tgt_embeds: 目标 (正确) 句子的嵌入 [B, L, D]
        # memory_embeds: "草稿" 句子的嵌入 (来自学生模型) [B, L, D]
        # tgt_mask: 自回归掩码 (causal mask) [L, L]
        
        # 1. 为目标添加位置编码
        tgt_with_pos = self.pos_encoder(tgt_embeds)
        
        # 2. (注意: memory 也需要位置编码，以使其与 tgt 对齐)
        memory_with_pos = self.pos_encoder(memory_embeds)
        
        # 3. 通过 Decoder
        #    tgt -> 自回归输入 (正确的句子)
        #    memory -> 交叉注意力输入 (学生的草稿)
        decoder_output = self.transformer_decoder(
            tgt=tgt_with_pos,
            memory=memory_with_pos,
            tgt_mask=tgt_mask
        )
        
        # 4. 预测 logits
        return self.to_logits(decoder_output)


# --- 4. 训练逻辑 (Training Logic) ---
# (calculate_es_rf_loss 和 calculate_consistency_loss 无需更改)

def get_rf_train_sample(x0, x1, t_val=None):
    if t_val is None:
        t = torch.rand(x0.shape[0], 1, 1, device=x0.device)
    else:
        t = t_val
    x_t = (1.0 - t) * x0 + t * x1
    target_v = x1 - x0
    return x_t, t, target_v

def calculate_proxy_entropy(logits):
    dist = Categorical(logits=logits)
    return dist.entropy() 

def calculate_es_rf_loss(model, x0_embeds, x1_embeds, lambda_entropy):
    x_t, t, target_v = get_rf_train_sample(x0_embeds, x1_embeds)
    predicted_v, logits = model(x_t, t)
    rf_loss_per_position = (predicted_v - target_v).pow(2).mean(dim=-1)
    entropy_per_position = calculate_proxy_entropy(logits)
    loss_weights = 1.0 + lambda_entropy * entropy_per_position.detach()
    weighted_loss = rf_loss_per_position * loss_weights
    final_loss = weighted_loss.mean()
    return final_loss, rf_loss_per_position.mean(), entropy_per_position.mean()

def calculate_consistency_loss(teacher_model, student_model, x0_embeds, x1_embeds, n_steps, lambda_discrete):
    n = torch.randint(1, n_steps, (x0_embeds.shape[0],), device=x0_embeds.device)
    t_n_plus_1 = n.float() / n_steps
    t_n = (n - 1).float() / n_steps
    dt = 1.0 / n_steps
    
    x_t_n_plus_1, _, _ = get_rf_train_sample(
        x0_embeds, x1_embeds, t_val=t_n_plus_1.reshape(-1, 1, 1)
    )

    with torch.no_grad():
        v_teacher_for_step, _ = teacher_model(x_t_n_plus_1, t_n_plus_1.reshape(-1, 1, 1))
        x_t_n = x_t_n_plus_1 - v_teacher_for_step * dt
        teacher_v_target, teacher_logits_target = teacher_model(
            x_t_n.detach(), t_n.reshape(-1, 1, 1).detach()
        )

    student_v_pred, student_logits_pred = student_model(
        x_t_n_plus_1, t_n_plus_1.reshape(-1, 1, 1)
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
        x_t = x_t + v_pred * dt
    return x_t 

def sample_cd(model, x0_noise, n_steps=4):
    """ (快速采样器，用于学生/草稿) """
    print(f"一致性采样 (NFE={n_steps})...")
    x_t = sample_rf(model, x0_noise, n_steps=n_steps)
    print("一致性采样完成。")
    return x_t

# --- *** 已修复：精炼器采样 (自回归 + 多样性采样) *** ---
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    对 logits 进行 Top-k 和 Top-p (nucleus) 过滤。
    logits: [batch_size, vocab_size]
    """
    top_k = min(top_k, logits.size(-1))  # 安全检查
    
    if top_k > 0:
        # 移除概率不在 top k 的所有 token
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过阈值的 token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将第一个 token 保留（即使它超过阈值）
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # 将过滤后的 logits 散射回原来的位置
        for b in range(logits.shape[0]):
            indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
            logits[b, indices_to_remove] = filter_value
    
    return logits

def sample_refiner(refiner_model, memory_embeds, embedding_layer, vocab, max_len=SEQ_LEN,
                   temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2):
    """
    使用精炼器进行自回归解码 (已修复：使用多样性采样策略)。
    
    Args:
        refiner_model: 精炼器模型
        memory_embeds: 学生的 "草稿" [B, L, D]
        embedding_layer: 嵌入层
        vocab: 词汇表
        max_len: 最大生成长度
        temperature: 温度参数 (越低越确定，越高越随机)
        top_k: Top-K 采样 (0 表示禁用)
        top_p: Top-P (nucleus) 采样 (0 表示禁用)
        repetition_penalty: 重复惩罚系数 (>1 惩罚重复)
    """
    print(f"精炼器自回归解码 (NFE={max_len}, temp={temperature}, top_k={top_k}, top_p={top_p})...")
    refiner_model.eval()
    
    batch_size = memory_embeds.shape[0]
    start_token_id = vocab.stoi["<pad>"]
    
    # tgt_tokens 是我们逐步构建的句子 [B, 1] -> [B, 2] -> ...
    tgt_tokens = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        for step in range(max_len - 1):
            # 1. 获取当前序列的嵌入
            tgt_embeds = embedding_layer(tgt_tokens)
            
            # 2. 创建因果掩码
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.shape[1]).to(DEVICE)
            
            # 3. 运行精炼器
            logits = refiner_model(tgt_embeds, memory_embeds, tgt_mask)
            
            # 4. 只看最后一个词的 logits
            next_token_logits = logits[:, -1, :].clone()  # [B, V]
            
            # 5. 应用重复惩罚 (Repetition Penalty)
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    for prev_token in set(tgt_tokens[b].tolist()):
                        # 如果 logit > 0，除以 penalty；否则乘以 penalty
                        if next_token_logits[b, prev_token] > 0:
                            next_token_logits[b, prev_token] /= repetition_penalty
                        else:
                            next_token_logits[b, prev_token] *= repetition_penalty
            
            # 6. 应用温度
            next_token_logits = next_token_logits / temperature
            
            # 7. 应用 Top-K 和 Top-P 过滤
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            
            # 8. 从过滤后的分布中采样
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            
            # 9. 将新词元拼接到序列中
            tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)

    print("精炼器解码完成。")
    
    # 将 ID 解码回文本
    all_texts = []
    for i in range(tgt_tokens.shape[0]):
        tokens = [vocab.itos[idx] for idx in tgt_tokens[i].cpu().numpy()]
        filtered_tokens = [t for t in tokens if t not in ["<pad>", "<unk>"]]
        all_texts.append(" ".join(filtered_tokens))
        
    return all_texts

def decode_embeddings_to_text(model, x1_pred_embeds, vocab):
    """ (用于教师/学生模型的解码器) """
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
    print(f"--- ES-RF w/ Transformer & CD & Refiner (Full Project) ---")
    
    # 1. 加载数据
    train_iter, vocab, tokenizer = get_data_iter_and_vocab(split='train')
    VOCAB_SIZE = len(vocab)
    PAD_INDEX = vocab.stoi["<pad>"]
    print(f"Actual VOCAB_SIZE set to {VOCAB_SIZE}")
    print(f"Vocab '<unk>' index: {vocab.stoi['<unk>']}")
    print(f"Vocab '<pad>' index: {PAD_INDEX}")

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
        teacher_model.train(); embedding_layer.train()
        for i, batch_token_ids in enumerate(train_dataloader):
            if batch_token_ids.shape[0] == 0: continue
            optimizer_teacher.zero_grad()
            x1_embeds = embedding_layer(batch_token_ids)
            x0_noise = torch.randn_like(x1_embeds)
            loss, base_rf_loss, entropy = calculate_es_rf_loss(
                teacher_model, x0_noise, x1_embeds, LAMBDA_ENTROPY
            )
            loss.backward(); optimizer_teacher.step()
            if (i + 1) % 100 == 0: 
                print(f"  [教师 Epoch {epoch+1}, Step {i+1}] 加权损失: {loss.item():.6f} | (基础RF: {base_rf_loss.item():.6f}, 熵: {entropy.item():.4f})")
    print("\n--- 教师模型 训练完成 ---")
    torch.save({
        'teacher_model_state_dict': teacher_model.state_dict(),
        'embedding_layer_state_dict': embedding_layer.state_dict(),
        'vocab': vocab,
    }, TEACHER_MODEL_PATH); print(f"教师模型已保存到 {TEACHER_MODEL_PATH}")
    teacher_model.eval(); embedding_layer.eval()

    # --- *** 阶段 2: 训练学生模型 *** ---
    student_model = LanguageRFTransformer(
        embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, n_layers=N_LAYERS,
        n_heads=N_HEADS, dim_feedforward=DIM_FEEDFORWARD
    ).to(DEVICE)
    student_model.load_state_dict(teacher_model.state_dict())
    optimizer_student = optim.Adam(student_model.parameters(), lr=LEARNING_RATE_STUDENT)
    
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
            loss.backward(); optimizer_student.step()
            if (i + 1) % 100 == 0: 
                print(f"  [学生 Epoch {epoch+1}, Step {i+1}] 一致性损失: {loss.item():.6f} | (嵌入: {loss_e.item():.6f}, 离散: {loss_d.item():.6f})")
    print("\n--- 学生模型 蒸馏完成 ---")
    torch.save({
        'student_model_state_dict': student_model.state_dict(),
        'embedding_layer_state_dict': embedding_layer.state_dict(),
        'vocab': vocab,
    }, STUDENT_MODEL_PATH); print(f"学生模型已保存到 {STUDENT_MODEL_PATH}")
    student_model.eval()

    # --- *** 阶段 3: 训练精炼器模型 (已修复: 使用真实数据) *** ---
    refiner_model = LanguageRefinerModel(
        embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, n_layers=N_LAYERS,
        n_heads=N_HEADS, dim_feedforward=DIM_FEEDFORWARD
    ).to(DEVICE)
    optimizer_refiner = optim.Adam(refiner_model.parameters(), lr=LEARNING_RATE_REFINER)
    criterion_refiner = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
    
    print("\n--- 阶段 3: 开始训练 精炼器 (Refiner) 模型 (使用真实数据) ---")
    
    for epoch in range(NUM_EPOCHS_REFINER):
        print(f"\n--- [精炼器] Epoch {epoch+1}/{NUM_EPOCHS_REFINER} ---")
        refiner_model.train()
        
        # *** 修复: 使用真实数据训练，而不是噪声到噪声 ***
        for i, batch_token_ids in enumerate(train_dataloader):
            if batch_token_ids.shape[0] == 0: 
                continue
            
            optimizer_refiner.zero_grad()
            
            # 1. 真实目标 (来自数据集) - 这是我们希望精炼器生成的!
            target_ids = batch_token_ids  # [B, L]
            
            # 2. 模拟推理时的情况: 从噪声生成 "草稿" 作为 memory
            #    这里我们用真实数据的嵌入加噪声来模拟学生的输出
            with torch.no_grad():
                x1_embeds = embedding_layer(batch_token_ids)
                # 方法1: 从纯噪声生成草稿 (更真实但更慢)
                # z_noise = torch.randn_like(x1_embeds)
                # memory_embeds = sample_cd(student_model, z_noise, n_steps=N_FEW_STEPS_STUDENT)
                
                # 方法2: 添加噪声到真实嵌入 (更快，训练更稳定)
                noise_level = 0.3  # 噪声强度
                memory_embeds = x1_embeds + noise_level * torch.randn_like(x1_embeds)
            
            # 3. 准备自回归 (AR) 输入 (Teacher Forcing)
            #    输入: <pad> A B C D ...
            #    目标: A B C D E ...
            pad_tensor = torch.full((target_ids.shape[0], 1), PAD_INDEX, dtype=torch.long, device=DEVICE)
            input_ids = torch.cat([pad_tensor, target_ids[:, :-1]], dim=1)  # [B, L]
            
            tgt_embeds = embedding_layer(input_ids)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(SEQ_LEN).to(DEVICE)
            
            # 4. 运行精炼器
            logits = refiner_model(tgt_embeds, memory_embeds.detach(), tgt_mask)
            
            # 5. 计算损失 (预测真实的 token)
            loss = criterion_refiner(logits.transpose(1, 2), target_ids)
            
            loss.backward()
            optimizer_refiner.step()
            
            if (i + 1) % 100 == 0: 
                print(f"  [精炼器 Epoch {epoch+1}, Step {i+1}] 交叉熵损失: {loss.item():.6f}")

    print("\n--- 精炼器模型 训练完成 ---")
    
    # 保存精炼器模型
    torch.save({
        'refiner_model_state_dict': refiner_model.state_dict(),
        'embedding_layer_state_dict': embedding_layer.state_dict(),
        'vocab': vocab,
    }, REFINER_MODEL_PATH)
    print(f"精炼器模型已保存到 {REFINER_MODEL_PATH}")

    # --- 7. 生成文本 (最终对比) ---
    print("\n--- 开始从噪声生成文本 (最终对比) ---")
    
    z_noise = torch.randn(2, SEQ_LEN, EMBED_DIM).to(DEVICE) # 生成 2 个样本
    
    # --- 教师 (慢速) ---
    print(f"\n--- 教师 (慢速, NFE={N_STEPS_TEACHER}) ---")
    generated_embeds_teacher = sample_rf(teacher_model, z_noise, n_steps=N_STEPS_TEACHER)
    generated_texts_teacher = decode_embeddings_to_text(teacher_model, generated_embeds_teacher, vocab)
        
    # --- 学生 (快速草稿) ---
    print(f"\n--- 学生 (快速草稿, NFE={N_FEW_STEPS_STUDENT}) ---")
    # (我们需要为精炼器保存这份草稿)
    generated_embeds_student = sample_cd(student_model, z_noise, n_steps=N_FEW_STEPS_STUDENT)
    generated_texts_student = decode_embeddings_to_text(student_model, generated_embeds_student, vocab)

    # --- *** 新增：精炼器 (快速最终) *** ---
    print(f"\n--- 精炼器 (快速最终, NFE={SEQ_LEN}) ---")
    # (使用学生生成的 "草稿" embed 作为 memory)
    generated_texts_refiner = sample_refiner(
        refiner_model, generated_embeds_student, embedding_layer, vocab, max_len=SEQ_LEN
    )

    # --- 8. 保存所有结果到文件 ---
    print(f"\n--- 正在保存所有生成结果到 {RESULTS_TXT_PATH} ---")
    with open(RESULTS_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(f"--- 教师 (慢速, NFE={N_STEPS_TEACHER}) ---\n")
        for idx, text in enumerate(generated_texts_teacher):
            f.write(f"  教师样本 {idx+1}: {text}\n")
            print(f"  教师样本 {idx+1}: {text}") 
            
        f.write(f"\n--- 学生 (快速草稿, NFE={N_FEW_STEPS_STUDENT}) ---\n")
        for idx, text in enumerate(generated_texts_student):
            f.write(f"  学生样本 {idx+1}: {text}\n")
            print(f"  学生样本 {idx+1}: {text}")
            
        f.write(f"\n--- 精炼器 (快速最终, NFE={SEQ_LEN}) ---\n")
        for idx, text in enumerate(generated_texts_refiner):
            f.write(f"  精炼器样本 {idx+1}: {text}\n")
            print(f"  精炼器样本 {idx+1}: {text}")
    
    print(f"所有生成结果已保存到 {RESULTS_TXT_PATH}。")