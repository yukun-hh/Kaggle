# %%
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from tqdm import tqdm
# %%
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
# %%
train_df.head()
# %%
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
# %%
embedded_text=tokenizer("你好啊")
# %%
tokenizer.decode(embedded_text['input_ids'],skip_special_tokens=True)
# %%
embedded_text=tokenizer(["你好啊","我是灰太狼"],["我不好","我是红太狼"])
embedded_text=tokenizer("你好啊",return_tensors='pt')
# %%
class NliDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        premise = str(row['premise'])
        hypothesis = str(row['hypothesis'])
        # 编码文本对，返回 input_ids, attention_mask
        encoding = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'  # 返回 PyTorch Tensor
        )
        # 去掉 batch 维度（因为只处理单条）
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        if 'label' in row:
            item['labels'] = torch.tensor(row['label'], dtype=torch.long)
        return item
# %%

# %%
batch_size = 16  # 根据显存调整，推荐使用 16 或 32
max_length = 128

train_dataset = NliDataset(train_df, tokenizer, max_length)
val_dataset = NliDataset(val_df, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# %%

# %%
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 计算总训练步数（用于 warmup）
total_steps = len(train_loader) * 3  # 假设训练 3 个 epoch
warmup_steps = int(0.1 * total_steps)  # warmup 比例为 10%

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# %%
def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc
# %%
num_epochs = 3
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for batch in progress_bar:
        # 将数据移至设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # 反向传播
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()
        scheduler.step()   # 更新学习率
        optimizer.zero_grad()

        # 记录损失
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Average Train Loss: {avg_train_loss:.4f}")

    # 在每个 epoch 结束后评估验证集
    val_acc = evaluate(model, val_loader, device)
    print(f"Epoch {epoch+1} - Validation Accuracy: {val_acc:.4f}")

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')
        print("Best model saved!")
# %%
# 加载最佳模型权重
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# 构建测试集 Dataset 和 DataLoader（注意测试集没有 label）
class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        premise = str(row['premise'])
        hypothesis = str(row['hypothesis'])
        encoding = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

test_dataset = TestDataset(test_df, tokenizer, max_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 预测
all_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())

# 生成提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'label': all_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission saved!")