# %%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
# %%
# %cd /kaggle/working
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_df=pd.read_csv("/kaggle/input/competitions/digit-recognizer/train.csv")
test_df=pd.read_csv("/kaggle/input/competitions/digit-recognizer/test.csv")
# %%
fig, ax = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
ax = ax.flatten()
for i in range(4):
    img = train_df.iloc[i][1:].to_numpy().reshape(28,28)
    # ax[i].imshow(img,cmap='Greys')
    ax[i].imshow(img)
    ax[i].set_title(f'{train_df.iloc[i][0]}')
# %%
class DatasetMnist(Dataset):
    def __init__(self,df):
        self.df=df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        item= {"Data": torch.tensor(self.df.iloc[idx][1:].to_numpy().reshape(28, 28),dtype=torch.float), "label": torch.tensor(self.df.iloc[idx][0],dtype=torch.long)}
        return item
batch_size =64
train_dataset = DatasetMnist(train_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# %%
class MnistModule(nn.Module):
    def __init__(self):
        super(MnistModule, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128,10)
    def forward(self,X):
        return self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(X.view(-1,28*28))))))))
# %%
model = MnistModule()
model=model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# %%
epochs = 30
for epoch in range(epochs):
    model.train()
    training_loss=0
    for batch in train_loader:
        optimizer.zero_grad()
        X = batch['Data']
        labels=batch['label']
        X=X.to(device)
        labels=labels.to(device)
        #print(model(X),'\n',labels)
        loss = loss_func(model(X),labels)
        loss.backward()
        optimizer.step()
        training_loss+=loss.item()
    scheduler.step()
    print(f"train_loss: {training_loss/len(train_loader)}")
# %%
class DatasetMnistTest(Dataset):
    def __init__(self,df):
        self.df=df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        item= {"Data": torch.tensor(self.df.iloc[idx].to_numpy().reshape(28, 28),dtype=torch.float)}
        return item
test_dataset = DatasetMnistTest(test_df)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['Data'].to(device)
        outputs = model(input_ids)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
idd = range(1,len(all_preds)+1)
submission = pd.DataFrame({
    'ImageId':idd,
    'Label': all_preds
})
print(submission)
submission.to_csv('submission.csv', index=False)
print("Submission saved!")
# %%
