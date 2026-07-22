# %%
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# 读取同文件夹下的对话 csv（列：问题、回答）
df = pd.read_csv("样本收集.csv", encoding="utf-8")
df = df.dropna(subset=["问题", "回答"]).reset_index(drop=True)
# 清洗空白字符：去首尾空白、合并连续空白为单个空格、替换中文全角空格
df["问题"] = df["问题"].astype(str).str.replace("\u3000", " ").str.replace(r"\s+", " ", regex=True).str.strip()
df["回答"] = df["回答"].astype(str).str.replace("\u3000", " ").str.replace(r"\s+", " ", regex=True).str.strip()
print(f"共 {len(df)} 条对话样本")
print(df.head())

# %%
# 加载 Qwen2.5-1.5B-Instruct，bf16 全量微调（单卡 16GB 够用）
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.gradient_checkpointing_enable()
model.config.use_cache = False
model.to(device)

# 冻结前两层 decoder，节省显存
for i in range(2):
    for p in model.model.layers[i].parameters():
        p.requires_grad = False
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable}/{total} ({100*trainable/total:.2f}%)")
print(model)

# %%
# 对话数据集：把 (问题, 回答) 拼成提示文本，做因果语言建模
PROMPT_TEMPLATE = (
    "你是一名材料科学助手，请根据问题给出准确、专业的回答。\n"
    "问题：{q}\n"
    "回答：{a}{eos}"
)


class ChatDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        q = str(self.df.loc[idx, "问题"])
        a = str(self.df.loc[idx, "回答"])
        text = PROMPT_TEMPLATE.format(q=q, a=a, eos=self.tokenizer.eos_token)
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        # pad 位置用 -100 屏蔽损失
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return input_ids, attention_mask, labels


def collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    attention_mask = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return input_ids, attention_mask, labels


dataset = ChatDataset(df, tokenizer, max_length=512)
train_dataloader = DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_fn
)
# 抽样查看一条
sample = dataset[0]
print("sample input_ids shape:", sample[0].shape)
print("decoded:", tokenizer.decode(sample[0], skip_special_tokens=True))

# %%
# 训练配置
epochs = 30
learning_rate = 2e-5
warmup_ratio = 0.05

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=learning_rate,
    weight_decay=0.01,
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(warmup_ratio * epochs * len(train_dataloader)),
    num_training_steps=epochs * len(train_dataloader),
)

best_loss = float("inf")
save_dir = "./output"
os.makedirs(save_dir, exist_ok=True)

# %%
# 训练循环
for epoch in range(epochs):
    model.train()
    training_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    lens = 0
    for input_ids, attention_mask, labels in progress_bar:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        training_loss += loss.item()
        lens += 1
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{training_loss / lens:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    scheduler.step()
    avg_train_loss = training_loss / lens
    print(f"Epoch {epoch+1} train_loss: {avg_train_loss:.4f}")

    # 仅记录最优 loss，不每个 epoch 保存（1.5B 整存太费时费空间）
    if avg_train_loss < best_loss:
        best_loss = avg_train_loss
        print(f"新最优 loss: {best_loss:.4f}")

# 训练结束保存整个最终模型
final_dir = os.path.join(save_dir, "final_model")
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"训练完成，最佳 loss: {best_loss:.4f}")
print(f"最终模型已保存到: {final_dir}")
