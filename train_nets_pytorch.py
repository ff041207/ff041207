import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt

# 假设仓库的 speck.py 在同目录或已安装，接口与原来相同
import speck as sp

# 全局超参（保持和你原脚本一致）
bs = 5000
wdir = './freshly_trained_nets_pytorch/'
os.makedirs(wdir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cyclic_lr_value(epochs, num_epochs, high_lr, low_lr):
    # 原 Keras lambda: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    # 返回该 epoch 的绝对 lr 值
    return low_lr + ((num_epochs - 1) - (epochs % num_epochs)) / (num_epochs - 1) * (high_lr - low_lr)

def make_lr_scheduler(optimizer, num_epochs, high_lr=0.002, low_lr=0.0001):
    # 我们把 optimizer 的初始 lr 设为 high_lr，然后 LambdaLR 根据比例调整
    def lr_lambda(epochs):
        val = cyclic_lr_value(epochs, num_epochs, high_lr, low_lr)
        # lr_lambda(epoch) 不直接返回学习率，而是返回一个乘法因子
        # 当前epoch的lr = high_lr(optimizer初始化时的学习率) × lr_lambda(epoch)
        # 所以需要当前epoch的lr/high_lr
        return val / high_lr
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

# 把 NumPy 数据（X,Y）封装成 PyTorch 能直接使用的 Dataset
class SpeckDataset(Dataset):
    def __init__(self, X_np, Y_np):
        # X_np.shape: (N, num_blocks*word_size*2)
        # Y_np.shape: (N, 1)
        self.X = torch.from_numpy(X_np.astype(np.float32))
        self.Y = torch.from_numpy(Y_np.astype(np.float32)).view(-1,1)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class MyResNet(nn.Module):
    def __init__(self, num_blocks=2, num_filters=32, num_outputs=1, d1=64, d2=64,
                 word_size=16, ks=3, depth=5, reg_param=1e-4, final_activation='sigmoid'):
        super().__init__()
        self.word_size = word_size
        self.num_blocks = num_blocks
        in_channels = 2 * num_blocks
        self.num_filters = num_filters
        # torch > 1.9 之后支持 padding='same'
        pad = (ks - 1) // 2

        # conv0: expand channels -> num_filters
        self.conv0 = nn.Conv1d(in_channels=in_channels, out_channels=num_filters, kernel_size=1, padding=0)
        self.bn0 = nn.BatchNorm1d(num_filters)

        # residual blocks, each block has conv->bn->relu->conv->bn->relu and add skip
        # 在 PyTorch 里，L2 正则化不需要写在层里面，而是通过优化器的 weight_decay 参数实现
        self.res_blocks = nn.ModuleList()
        for _ in range(depth):
            block = nn.Sequential(
                nn.Conv1d(num_filters, num_filters, kernel_size=ks, padding=pad),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(inplace=True),
                nn.Conv1d(num_filters, num_filters, kernel_size=ks, padding=pad),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.res_blocks.append(block)

        # prediction head
        # flatten size = num_filters * word_size (Conv1d output shape: (batch, num_filters, word_size))
        # PyTorch 里通常在 forward() 里用 x.view() 来展平，而不是像 Keras 那样用一个专门的层
        self.fc1 = nn.Linear(num_filters * word_size, d1)
        self.bn_fc1 = nn.BatchNorm1d(d1)
        self.fc2 = nn.Linear(d1, d2)
        self.bn_fc2 = nn.BatchNorm1d(d2)
        self.out = nn.Linear(d2, num_outputs)
        self.final_activation = final_activation

    def forward(self, x):
        # x shape: (batch, num_blocks*word_size*2)
        batch = x.shape[0]
        # reshape to (batch, channels, length) = (batch, 2*num_blocks, word_size)
        x = x.view(batch, 2 * self.num_blocks, self.word_size)
        # conv0
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)
        # residual blocks (add skip)
        for block in self.res_blocks:
            shortcut = x
            out = block(x)
            x = shortcut + out
        # flatten
        x = x.view(batch, -1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.out(x)
        if self.final_activation == 'sigmoid':
            x = torch.sigmoid(x)
        return x

def train_speck_distinguisher(epochs, num_rounds=7, depth=1, bs_local=bs, reg_param=1e-5):
    # 模型与优化器
    model = MyResNet(depth=depth, reg_param=reg_param).to(device)
    # 我们把 optimizer 的 lr 初始设置为 high_lr，scheduler 按比例调整
    high_lr = 0.002
    low_lr = 0.0001
    optimizer = Adam(model.parameters(), lr=high_lr, weight_decay=reg_param)
    scheduler = make_lr_scheduler(optimizer, num_epochs=10, high_lr=high_lr, low_lr=low_lr)
    criterion = nn.MSELoss()

    # # 生成训练与验证数据（同原始 Keras 脚本）
    # print("Generating training data (this may be large)...")
    # X_train_np, Y_train_np = sp.make_train_data(10**7, num_rounds)   # 注意：数据量非常大，实际运行时请按需缩小
    # X_val_np, Y_val_np     = sp.make_train_data(10**6, num_rounds)
    # 从本地数据集读取数据
    print("Loading dataset from local files...")
    X_train_np = np.load(f"train_X_{10 ** 7}_{num_rounds}r.npy")
    Y_train_np = np.load(f"train_Y_{10 ** 7}_{num_rounds}r.npy")
    X_val_np = np.load(f"val_X_{10 ** 6}_{num_rounds}r.npy")
    Y_val_np = np.load(f"val_Y_{10 ** 6}_{num_rounds}r.npy")

    train_dataset = SpeckDataset(X_train_np, Y_train_np)
    val_dataset   = SpeckDataset(X_val_np, Y_val_np)

    train_loader = DataLoader(train_dataset, batch_size=bs_local, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=bs_local, shuffle=False, num_workers=4, pin_memory=True)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        batches = 0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1
        train_loss = running_loss / max(1, batches)

        # 验证
        model.eval()
        val_losses = 0.0
        val_batches = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                val_losses += loss.item()
                val_batches += 1
                # 计算准确率（阈值 0.5）
                preds = (outputs > 0.5).float()
                correct += (preds == Y_batch).sum().item()
                total += Y_batch.numel()
        val_loss = val_losses / max(1, val_batches)
        val_acc = correct / total if total > 0 else 0.0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # ModelCheckpoint: 根据 val_loss 保存最佳模型（与 Keras 中 monitor='val_loss' 对齐）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(wdir, f'best{num_rounds}r_depth{depth}.pt'))
            print(f"Epoch {epoch+1}: val_loss improved -> saved model")

        # step lr scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} - time: {epoch_time:.1f}s - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f} - val_acc: {val_acc:.6f}")

    # 保存训练历史（分别存 val_acc 与 val_loss，修正了原代码覆盖问题）
    np.save(os.path.join(wdir, f'h_val_acc_{num_rounds}r_depth{depth}.npy'), np.array(history['val_acc']))
    np.save(os.path.join(wdir, f'h_val_loss_{num_rounds}r_depth{depth}.npy'), np.array(history['val_loss']))
    with open(os.path.join(wdir, f'hist{num_rounds}r_depth{depth}.p'), 'wb') as f:
        pickle.dump(history, f)

    print("Best validation accuracy: ", np.max(history['val_acc']))
    return model, history

if __name__ == "__main__":
    # 示例：训练 3 个 epoch（真实训练请把 num_epochs 调大）
    model, hist = train_speck_distinguisher(epochs=200, num_rounds=5, depth=10)

    # ---------- 绘制 Loss 曲线 ----------
    epochs = range(1, len(hist['train_loss']) + 1)
    plt.figure(figsize=(10, 4))

    # 图1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['val_loss'], 'o-', label="Validation loss")
    plt.plot(epochs, hist['train_loss'], '^-', label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 图2: Validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist['val_acc'], 'o-')
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=300)  # 保存图片
    plt.show()
