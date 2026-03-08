# ef_features_hetero_conformal.py
"""
1) φ_E 构造：从 GP μ(t), σ_cal(t) 生成视频级特征
2) Heteroscedastic MLP：输出 μ_EF, σ_EF
3) σ-aware conformal：基于 |y-μ|/σ_EF 标定 q，区间为 μ ± q·σ_EF
"""

import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================================================
# 路径 & 基本配置（按你机器路径改）
# =========================================================

CSV_PATH = Path("/mnt/EF_measurement/data/echonet/FileList_aligned.csv")
GP_ROOT = Path("/mnt/EF_measurement/data/echonet/latents_gp")  # 存 μ/σ_cal 的目录

SPLIT_VALUES = {
    "train": "TRAIN",
    "val": "VAL",
    "test": "TEST",
}

# 方便统一管理保存
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)


# =========================================================
# φ_E 构造：从 μ(t), σ(t) 生成 video-level 特征
# =========================================================

def compute_phi_E(mu: np.ndarray,
                  sigma: np.ndarray,
                  eps: float = 1e-6) -> np.ndarray:
    """
    构造一个简单但实用的 φ_E：
      - 对每个 bottleneck 维度 k：
        • 不确定性加权 mean： m_k
        • 振幅 amplitude： a_k
        • σ 的时间均值： sigma_mean_k
        • σ 的时间最大值： sigma_max_k
      - 拼在一起得到长度 4K 的 feature 向量

    参数
    ----
    mu:    (T, K)  GP posterior mean
    sigma: (T, K)  calibrated GP sigma (已经乘过 s*)
    eps:   防止除零

    返回
    ----
    phi_E: (4K,) 的 np.float32 向量
    """
    assert mu.shape == sigma.shape, f"mu shape {mu.shape} != sigma shape {sigma.shape}"
    T, K = mu.shape

    # 不确定性权重：σ 越大权重越低
    w = 1.0 / (sigma ** 2 + eps)      # (T, K)
    w_sum = w.sum(axis=0) + eps       # (K,)

    # 不确定性加权 mean
    m = (w * mu).sum(axis=0) / w_sum  # (K,)

    # 简单振幅：max-min 的一半
    amp = 0.5 * (mu.max(axis=0) - mu.min(axis=0))  # (K,)

    # σ 的 summary
    sigma_mean = sigma.mean(axis=0)   # (K,)
    sigma_max = sigma.max(axis=0)     # (K,)

    phi = np.concatenate([m, amp, sigma_mean, sigma_max], axis=0).astype(np.float32)  # (4K,)
    return phi


# =========================================================
# 数据集：从 μ/σ_cal 直接生成 φ_E + EF
# =========================================================

class EchonetFeatureDataset(Dataset):
    """
    读取：
      - CSV: FileName, EF, Split
      - GP_ROOT: {FileName}_mu.npy, {FileName}_sigma_cal.npy
    生成：
      - φ_E: (F,) 的 feature
      - EF: scalar
    """
    def __init__(self,
                 csv_path: Path,
                 gp_root: Path,
                 split_names=("TRAIN",),
                 use_calibrated_sigma: bool = True,
                 dtype=torch.float32):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.gp_root = Path(gp_root)
        self.split_names = set(split_names)
        self.use_calibrated_sigma = use_calibrated_sigma
        self.dtype = dtype

        df = pd.read_csv(self.csv_path)
        df = df[df["Split"].isin(self.split_names)].copy()
        self.records = df[["FileName", "EF"]].reset_index(drop=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        fname = str(row["FileName"])
        ef = float(row["EF"])

        mu_path = self.gp_root / f"{fname}_mu.npy"
        if self.use_calibrated_sigma:
            sigma_path = self.gp_root / f"{fname}_sigma_cal.npy"
        else:
            sigma_path = self.gp_root / f"{fname}_sigma_raw.npy"

        if not mu_path.exists() or not sigma_path.exists():
            raise FileNotFoundError(f"Missing GP files for {fname}: {mu_path}, {sigma_path}")

        mu = np.load(mu_path)      # (T, K)
        sigma = np.load(sigma_path)  # (T, K)

        phi = compute_phi_E(mu, sigma)  # (F,)
        phi_tensor = torch.from_numpy(phi).to(self.dtype)
        ef_tensor = torch.tensor(ef, dtype=self.dtype)

        return phi_tensor, ef_tensor, fname


# =========================================================
# 计算特征归一化：mean / std（在 train 上）
# =========================================================

def compute_feature_norm_stats(dataset: Dataset,
                               batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在 train 上计算 φ_E 的 mean / std，用于标准化。
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_feats = []

    for feats, ef, _ in loader:
        all_feats.append(feats)

    all_feats = torch.cat(all_feats, dim=0)  # (N, F)
    mean = all_feats.mean(dim=0)
    std = all_feats.std(dim=0) + 1e-6
    return mean, std


# =========================================================
# Heteroscedastic MLP：输出 μ_EF & σ_EF
# =========================================================

class HeteroscedasticEFMLP(nn.Module):
    """
    输入 φ_E → backbone → μ_EF, log σ_EF^2
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Tuple[int, ...] = (128, 64)):
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        self.mu_head = nn.Linear(prev_dim, 1)
        self.log_sigma2_head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, F)
        返回：
          mu:    (B,)
          sigma: (B,)  (已保证 >0)
        """
        h = self.backbone(x)
        mu = self.mu_head(h).squeeze(-1)             # (B,)
        log_sigma2 = self.log_sigma2_head(h).squeeze(-1)  # (B,)

        sigma2 = torch.exp(log_sigma2) + 1e-6        # 防止数值问题
        sigma = torch.sqrt(sigma2)
        return mu, sigma


# =========================================================
# 训练 Heteroscedastic MLP
# =========================================================

def heteroscedastic_gaussian_nll(y_true: torch.Tensor,
                                 mu: torch.Tensor,
                                 sigma: torch.Tensor) -> torch.Tensor:
    """
    异方差高斯的 NLL（省略常数项）：
      0.5 * [(y-mu)^2 / σ^2 + log σ^2]
    """
    sigma2 = sigma ** 2
    return 0.5 * (((y_true - mu) ** 2) / sigma2 + torch.log(sigma2))


def train_heteroscedastic_mlp(num_epochs: int = 50,
                              batch_size: int = 64,
                              lr: float = 1e-3,
                              weight_decay: float = 1e-4,
                              hidden_dims: Tuple[int, ...] = (128, 64)):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 构造 train / val feature dataset
    train_dataset = EchonetFeatureDataset(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES["train"],),
        use_calibrated_sigma=True,
    )
    val_dataset = EchonetFeatureDataset(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES["val"],),
        use_calibrated_sigma=True,
    )

    # 2. 计算 feature 标准化参数
    feat_mean, feat_std = compute_feature_norm_stats(train_dataset)
    feat_mean = feat_mean.to(device)
    feat_std = feat_std.to(device)

    # 3. DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 4. 初始化模型
    input_dim = train_dataset[0][0].shape[0]  # φ_E 维度
    model = HeteroscedasticEFMLP(input_dim=input_dim, hidden_dims=hidden_dims).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(num_epochs):
        # ------- train -------
        model.train()
        train_losses = []
        for feats, ef, _ in train_loader:
            feats = feats.to(device)
            ef = ef.to(device)

            # 标准化特征
            feats_norm = (feats - feat_mean) / feat_std

            optimizer.zero_grad()
            mu, sigma = model(feats_norm)
            nll = heteroscedastic_gaussian_nll(ef, mu, sigma).mean()
            nll.backward()
            optimizer.step()

            train_losses.append(nll.item())

        # ------- val -------
        model.eval()
        val_losses = []
        val_sq_errors = []
        with torch.no_grad():
            for feats, ef, _ in val_loader:
                feats = feats.to(device)
                ef = ef.to(device)
                feats_norm = (feats - feat_mean) / feat_std

                mu, sigma = model(feats_norm)
                nll = heteroscedastic_gaussian_nll(ef, mu, sigma).mean()
                val_losses.append(nll.item())
                val_sq_errors.append((ef - mu) ** 2)

        val_loss = float(np.mean(val_losses))
        val_rmse = torch.sqrt(torch.cat(val_sq_errors).mean()).item()

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train NLL: {np.mean(train_losses):.4f} | "
              f"Val NLL: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}")

        # 保存最优模型（按 RMSE）
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {
                "model": model.state_dict(),
                "feat_mean": feat_mean.cpu(),
                "feat_std": feat_std.cpu(),
                "epoch": epoch + 1,
                "val_rmse": val_rmse,
            }
            torch.save(best_state, CHECKPOINT_DIR / "hetero_mlp_best.pt")

    print(f"Training finished. Best Val RMSE = {best_val_rmse:.4f}")


# =========================================================
# σ-aware conformal：|y-μ|/σ_EF 的分位数 q
# =========================================================

def compute_conformal_q(alpha: float = 0.1,
                        batch_size: int = 64,
                        checkpoint_path: Path = CHECKPOINT_DIR / "hetero_mlp_best.pt") -> float:
    """
    在 VAL split 上：
      - 用最优 heteroscedastic MLP 得到 μ_EF, σ_EF
      - 计算 z_i = |y_i - μ_i| / σ_i
      - 取 (1 - alpha)-分位数 q
      - 保存 q
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载最优模型 & 归一化参数
    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)

    # 构造一个临时 dataset 拿到 input_dim
    tmp_dataset = EchonetFeatureDataset(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES["val"],),
        use_calibrated_sigma=True,
    )
    input_dim = tmp_dataset[0][0].shape[0]
    model = HeteroscedasticEFMLP(input_dim=input_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_loader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_z = []

    with torch.no_grad():
        for feats, ef, _ in val_loader:
            feats = feats.to(device)
            ef = ef.to(device)
            feats_norm = (feats - feat_mean) / feat_std

            mu, sigma = model(feats_norm)
            residual = torch.abs(ef - mu)   # |y - μ|
            z = residual / (sigma + 1e-6)   # 标准化残差
            all_z.append(z.cpu().numpy())

    all_z = np.concatenate(all_z, axis=0)  # (N_val,)

    q = float(np.quantile(all_z, 1.0 - alpha))
    print(f"Conformal q (1-alpha={1-alpha:.2f}) = {q:.4f}")

    np.save(CHECKPOINT_DIR / "conformal_q.npy", np.array(q, dtype=np.float32))
    return q


# =========================================================
# 用 σ-aware conformal 在 test 上打区间 + 简单评估
# =========================================================

def evaluate_on_split(split_key: str = "test",
                      alpha: float = 0.1,
                      batch_size: int = 64,
                      checkpoint_path: Path = CHECKPOINT_DIR / "hetero_mlp_best.pt"):
    """
    在给定 split (train/val/test) 上：
      - 加载 hetero MLP + 归一化参数 + q
      - 输出 μ, σ_EF, 区间 [μ - qσ, μ + qσ]
      - 打印 coverage 和平均宽度
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)

    q_path = CHECKPOINT_DIR / "conformal_q.npy"
    if not q_path.exists():
        raise FileNotFoundError("No conformal_q.npy found. Please run compute_conformal_q() first.")
    q = float(np.load(q_path))

    dataset = EchonetFeatureDataset(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES[split_key],),
        use_calibrated_sigma=True,
    )
    input_dim = dataset[0][0].shape[0]

    model = HeteroscedasticEFMLP(input_dim=input_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_mu = []
    all_sigma = []
    all_y = []
    all_L = []
    all_U = []

    with torch.no_grad():
        for feats, ef, _ in loader:
            feats = feats.to(device)
            ef = ef.to(device)
            feats_norm = (feats - feat_mean) / feat_std

            mu, sigma = model(feats_norm)

            L = mu - q * sigma
            U = mu + q * sigma

            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_y.append(ef.cpu().numpy())
            all_L.append(L.cpu().numpy())
            all_U.append(U.cpu().numpy())

    mu = np.concatenate(all_mu, axis=0)
    sigma = np.concatenate(all_sigma, axis=0)
    y = np.concatenate(all_y, axis=0)
    L = np.concatenate(all_L, axis=0)
    U = np.concatenate(all_U, axis=0)

    # 覆盖率 & 区间宽度
    covered = (y >= L) & (y <= U)
    coverage = covered.mean()
    mean_width = (U - L).mean()

    rmse = np.sqrt(((y - mu) ** 2).mean())
    mae = np.abs(y - mu).mean()

    print(f"[Split={split_key}] α={alpha:.2f}, q={q:.4f}")
    print(f"Point RMSE={rmse:.4f}, MAE={mae:.4f}")
    print(f"Coverage={coverage:.4f}, Mean width={mean_width:.4f}")

def analyze_uncertainty_on_split(split_key: str = "test",
                                 batch_size: int = 64,
                                 num_bins: int = 5,
                                 checkpoint_path: Path = CHECKPOINT_DIR / "hetero_mlp_best.pt"):
    """
    在指定 split (train/val/test) 上分析不确定性质量：
      - Pearson corr(σ_EF, |y - μ|)
      - 按 σ_EF 分成 num_bins 组，统计每组：
          • 样本数
          • 平均 |error|
          • 平均区间宽度 2 * q * σ_EF
    要求：
      - 已经训练好 hetero MLP (hetero_mlp_best.pt)
      - 已经计算好 conformal_q.npy
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型 & 归一化参数 & q
    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)

    q_path = CHECKPOINT_DIR / "conformal_q.npy"
    if not q_path.exists():
        raise FileNotFoundError("No conformal_q.npy found. Run compute_conformal_q() first.")
    q = float(np.load(q_path))

    # 2. 构造 dataset 和 model
    dataset = EchonetFeatureDataset(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES[split_key],),
        use_calibrated_sigma=True,
    )
    input_dim = dataset[0][0].shape[0]

    model = HeteroscedasticEFMLP(input_dim=input_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_mu = []
    all_sigma = []
    all_y = []

    with torch.no_grad():
        for feats, ef, _ in loader:
            feats = feats.to(device)
            ef = ef.to(device)
            feats_norm = (feats - feat_mean) / feat_std

            mu, sigma = model(feats_norm)

            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_y.append(ef.cpu().numpy())

    mu = np.concatenate(all_mu, axis=0)      # (N,)
    sigma = np.concatenate(all_sigma, axis=0)  # (N,)
    y = np.concatenate(all_y, axis=0)        # (N,)

    err = np.abs(y - mu)                    # |error|
    width = 2.0 * q * sigma                 # 区间宽度

    # 3. Pearson 相关系数
    corr = np.corrcoef(sigma, err)[0, 1]
    print(f"[Uncertainty analysis] Split={split_key}")
    print(f"  Pearson corr(sigma_EF, |error|) = {corr:.4f}")

    # 4. 按 σ 分成 num_bins 组
    #    使用分位数做切分
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.quantile(sigma, quantiles)

    print(f"\n  Binned statistics by sigma_EF (num_bins={num_bins}):")
    print("  Bin\tRange\t\tCount\tMean|err|\tMean width")

    for b in range(num_bins):
        left = bin_edges[b]
        right = bin_edges[b + 1]
        if b == num_bins - 1:
            # 右端点包含在最后一组
            mask = (sigma >= left) & (sigma <= right)
        else:
            mask = (sigma >= left) & (sigma < right)

        count = mask.sum()
        if count == 0:
            mean_err = float("nan")
            mean_width = float("nan")
        else:
            mean_err = err[mask].mean()
            mean_width = width[mask].mean()

        print(f"  {b+1}\t[{left:.3f}, {right:.3f}]"
              f"\t{int(count)}\t{mean_err:.3f}\t\t{mean_width:.3f}")


# =========================================================
# 脚本入口示例
# =========================================================

if __name__ == "__main__":
    # 1. 先训练 heteroscedastic MLP
    train_heteroscedastic_mlp(
        num_epochs=50,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dims=(128, 64),
    )

    # 2. 在 VAL 上做 σ-aware conformal，得到 q
    compute_conformal_q(alpha=0.1)  # 目标 nominal coverage ≈ 1-α = 0.9

    # 3. 在 TEST 上评估区间
    evaluate_on_split(split_key="test", alpha=0.1)
