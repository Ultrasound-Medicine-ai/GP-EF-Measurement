# train_bottleneck.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_and_config import EchonetLatentDataset, LATENT_ROOT, CSV_PATH, SPLIT_VALUES
import os
import numpy as np
import pandas as pd 

class BottleneckEFRegressor(nn.Module):
    """
    Learnable GP bottleneck + 简单 EF 回归头
    z_t (T, D) -> u_t (T, K) -> mean over time -> EF
    """
    def __init__(self, in_dim: int, bottleneck_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        # 线性瓶颈，建议无 bias，方便当作投影矩阵 W
        self.bottleneck = nn.Linear(in_dim, bottleneck_dim, bias=False)

        # 简单的 MLP EF 头
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z_seq: torch.Tensor):
        """
        z_seq: (T, D)
        返回 EF 预测值: scalar
        """
        # (T, K)
        u_seq = self.bottleneck(z_seq)
        # 简单时间平均 (可以后面改成更复杂的 pooling)
        u_mean = u_seq.mean(dim=0)  # (K,)

        ef_pred = self.mlp(u_mean)  # (1,)
        return ef_pred.squeeze(-1)  # scalar

    def get_bottleneck_matrix(self):
        """
        返回 numpy 数组形式的 W (D, K)
        PyTorch: weight shape = (K, D)
        """
        W = self.bottleneck.weight.detach().cpu().numpy().T  # (D, K)
        return W


def orthogonality_regularizer(linear_layer: nn.Linear):
    """
    对 W^T W 施加正交约束，鼓励瓶颈通道彼此正交。
    """
    W = linear_layer.weight  # (K, D)
    # (D, K) for W^T; (K, D) @ (D, K) = (K, K)
    WWT = torch.matmul(W, W.t())  # (K, K)
    I = torch.eye(WWT.shape[0], device=WWT.device)
    return torch.norm(WWT - I, p="fro")


def infer_latent_dim(latent_root: str, csv_path: str):
    """
    简单地从一个样本的 .npy 推断 latent 维度 D。
    """

    df = pd.read_csv(csv_path)
    # 找第一个存在的 latent 文件
    for fname in df["FileName"].tolist():
        path = os.path.join(latent_root, f"{fname}.npy")
        if os.path.exists(path):
            arr = np.load(path)
            return arr.shape[1]  # (T, D)
    raise RuntimeError("Cannot infer latent dimension, no .npy found.")


def main_train_bottleneck():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 推断 latent 维数 D
    D = infer_latent_dim(str(LATENT_ROOT), str(CSV_PATH))
    print(f"Inferred latent dimension D = {D}")

    bottleneck_dim = 16  # 你可以改成 8/16 等
    hidden_dim = 64

    # 2. 构造数据集 (只用 TRAIN split 预训练瓶颈)
    train_dataset = EchonetLatentDataset(
        csv_path=CSV_PATH,
        latent_root=LATENT_ROOT,
        split_names=(SPLIT_VALUES["train"],),
        max_frames=None  # 如需截断帧数可以设置
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,   # 每个样本长度不一，batch_size=1 最简单
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 3. 初始化模型
    model = BottleneckEFRegressor(in_dim=D, bottleneck_dim=bottleneck_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    # 正交正则系数
    lambda_ortho = 1e-3
    num_epochs = 10  # 你可以根据情况调大一些

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for latents, ef, _fname in train_loader:
            latents = latents.to(device)  # (1, T, D)
            ef = ef.to(device)            # (1,)

            # 由于 batch_size=1，去掉 batch 维
            latents = latents.squeeze(0)  # (T, D)
            ef = ef.squeeze(0)            # scalar

            optimizer.zero_grad()
            ef_pred = model(latents)      # scalar

            loss_mse = mse_loss(ef_pred, ef)
            loss_ortho = orthogonality_regularizer(model.bottleneck)
            loss = loss_mse + lambda_ortho * loss_ortho

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Epoch {epoch+1}/{num_epochs}] Loss = {running_loss/len(train_loader):.4f}")

    # 4. 保存整个模型 & 单独导出瓶颈矩阵 W
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/bottleneck_efreg.pt")
    W = model.get_bottleneck_matrix()
    np.save("checkpoints/bottleneck_W.npy", W)
    print("Saved bottleneck model and W to checkpoints/")

if __name__ == "__main__":
    main_train_bottleneck()
