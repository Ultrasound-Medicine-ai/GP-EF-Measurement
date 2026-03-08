"""
1) φ_E 构造：从 GP μ(t), σ_cal(t) 生成视频级特征
2) Heteroscedastic MLP：输出 μ_EF, σ_EF
3) σ-aware conformal：基于 |y-μ|/σ_EF 标定 q，区间为 μ ± q·σ_EF
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Sequence
 

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================================================
# 路径 & 基本配置
# =========================================================

from dataset_and_config import LATENT_ROOT, CSV_PATH, SPLIT_VALUES  # 统一路径配置

W_PATH = Path("/mnt/EF_measurement/ef_gp/checkpoints/bottleneck_W.npy")
GP_ROOT = Path("/mnt/EF_measurement/data/echonet/latents_gp")  # 存 μ/σ_raw/σ_cal 的目录

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)


# =========================================================
# φ_E 构造：从 μ(t), σ(t) 生成 video-level 特征
# =========================================================

def compute_phi_E(mu: np.ndarray,
                  sigma: np.ndarray,
                  eps: float = 1e-6) -> np.ndarray:
    """
    一个简单版本的 φ_E（目前未直接使用，保留以防后续扩展）：
      - 对每个 bottleneck 维度 k：
        • 不确定性加权 mean： m_k
        • 振幅 amplitude： a_k
        • σ 的时间均值： sigma_mean_k
        • σ 的时间最大值： sigma_max_k
      - 拼在一起得到长度 4K 的 feature 向量
    """
    assert mu.shape == sigma.shape, f"mu shape {mu.shape} != sigma shape {sigma.shape}"
    T, K = mu.shape

    w = 1.0 / (sigma ** 2 + eps)      # (T, K)
    w_sum = w.sum(axis=0) + eps       # (K,)
    m = (w * mu).sum(axis=0) / w_sum  # (K,)

    amp = 0.5 * (mu.max(axis=0) - mu.min(axis=0))  # (K,)

    sigma_mean = sigma.mean(axis=0)   # (K,)
    sigma_max = sigma.max(axis=0)     # (K,)

    phi = np.concatenate([m, amp, sigma_mean, sigma_max], axis=0).astype(np.float32)  # (4K,)
    return phi


def compute_phi_E_nosigma(mu: np.ndarray,
                          sigma: np.ndarray = None) -> np.ndarray:
    """
    U0 / G0 / F0 等消融版本：不使用 GP σ 特征。
    只用：
      - mean over time
      - amplitude (max-min)/2

    参数:
      mu: (T, K)
      sigma: 忽略
    返回:
      phi_E: (2K,) np.float32
    """
    T, K = mu.shape
    m = mu.mean(axis=0)  # (K,)
    amp = 0.5 * (mu.max(axis=0) - mu.min(axis=0))
    phi = np.concatenate([m, amp], axis=0).astype(np.float32)
    return phi


def compute_phi_E_sigmaaware(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Ours 专用的 φ_E 构造：
      - 对 GP posterior 均值 μ(t) 用 1/σ^2 做精度加权池化
      - 再加一些简单的 σ 统计量

    输入:
      mu:    (T, K)
      sigma: (T, K)
    输出:
      phi_E: (5K,) float32
    """
    assert mu.shape == sigma.shape
    T, K = mu.shape

    eps = 1e-6
    sigma_clamped = np.clip(sigma, 1e-3, 1e3)          # 防止 1/σ^2 爆炸
    prec = 1.0 / (sigma_clamped ** 2 + eps)            # (T, K)

    # 精度加权均值
    num = (mu * prec).sum(axis=0)                      # (K,)
    den = prec.sum(axis=0) + eps
    mu_prec_mean = num / den                           # (K,)

    # 精度加权振幅（近似）
    mu_eff = mu * prec
    mu_eff_max = mu_eff.max(axis=0)                    # (K,)
    mu_eff_min = mu_eff.min(axis=0)                    # (K,)
    amp_prec = 0.5 * (mu_eff_max - mu_eff_min) / (prec.mean(axis=0) + eps)  # (K,)

    sigma_mean = sigma_clamped.mean(axis=0)            # (K,)
    sigma_max = sigma_clamped.max(axis=0)              # (K,)
    log_sigma_mean = np.log(sigma_clamped).mean(axis=0)    # (K,)

    phi = np.concatenate([
        mu_prec_mean,      # K
        amp_prec,          # K
        sigma_mean,        # K
        sigma_max,         # K
        log_sigma_mean,    # K   -> 总共 5K
    ], axis=0).astype(np.float32)

    return phi


def load_bottleneck_W() -> np.ndarray:
    """
    从 checkpoints 里加载瓶颈矩阵 W (D, K)。
    """
    W = np.load(W_PATH)  # (D, K)
    return W


def moving_average_smooth(u: np.ndarray, window: int = 5) -> np.ndarray:
    """
    对时间序列 u(t) 做简单滑动平均平滑。
    u: (T, K)
    返回 u_smooth: (T, K)
    """
    T, K = u.shape
    if window <= 1 or window > T:
        return u.copy()

    u_smooth = np.zeros_like(u)
    kernel = np.ones(window, dtype=np.float32) / float(window)

    for k in range(K):
        x = u[:, k]
        x_smooth = np.convolve(x, kernel, mode="same")
        u_smooth[:, k] = x_smooth

    return u_smooth


# =========================================================
# 各种 Dataset：从 μ/σ_cal 或 u(t) 生成 φ_E + EF
# =========================================================

class EchonetFeatureDataset(Dataset):
    """
    读取：
      - CSV: FileName, EF, Split
      - GP_ROOT: {FileName}_mu.npy, {FileName}_sigma_cal.npy
    生成：
      - φ_E: (F,) 的 feature (ours: 5K)
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

        phi = compute_phi_E_sigmaaware(mu, sigma)  # (5K,)
        phi_tensor = torch.from_numpy(phi).to(self.dtype)
        ef_tensor = torch.tensor(ef, dtype=self.dtype)

        return phi_tensor, ef_tensor, fname


class EchonetFeatureDatasetNoSigma(Dataset):
    """
    U0 消融用：
      - 从 μ(t) 生成 φ_E，不使用 σ 特征。
    """
    def __init__(self,
                 csv_path: Path,
                 gp_root: Path,
                 split_names=("TRAIN",),
                 dtype=torch.float32):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.gp_root = Path(gp_root)
        self.split_names = set(split_names)
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
        if not mu_path.exists():
            raise FileNotFoundError(f"Missing GP mu file for {fname}: {mu_path}")

        mu = np.load(mu_path)  # (T, K)
        phi = compute_phi_E_nosigma(mu, sigma=None)  # (2K,)
        phi_tensor = torch.from_numpy(phi).to(self.dtype)
        ef_tensor = torch.tensor(ef, dtype=self.dtype)

        return phi_tensor, ef_tensor, fname


class EchonetFeatureDataset_NoGP(Dataset):
    """
    G0 baseline:
      - 不使用 GP 输出，直接用 latent_raw 乘 W 得到 u(t)
      - 再用 compute_phi_E_nosigma(u, None) 构造 φ_E
    """
    def __init__(self,
                 csv_path: Path,
                 latent_root: Path,
                 W: np.ndarray,
                 split_names=("TRAIN",),
                 dtype=torch.float32):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.latent_root = Path(latent_root)
        self.split_names = set(split_names)
        self.dtype = dtype
        self.W = W  # (D, K)

        df = pd.read_csv(self.csv_path)
        df = df[df["Split"].isin(self.split_names)].copy()
        self.records = df[["FileName", "EF"]].reset_index(drop=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        fname = str(row["FileName"])
        ef = float(row["EF"])

        latent_path = self.latent_root / f"{fname}.npy"
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing latent file for {fname}: {latent_path}")

        z = np.load(latent_path)   # (T, D)
        u = z @ self.W             # (T, K)

        phi = compute_phi_E_nosigma(u, sigma=None)  # (2K,)
        phi_tensor = torch.from_numpy(phi).to(self.dtype)
        ef_tensor = torch.tensor(ef, dtype=self.dtype)

        return phi_tensor, ef_tensor, fname


class EchonetFeatureDataset_MA(Dataset):
    """
    F0 baseline:
      - latent_raw @ W 得到 u(t)
      - 对 u(t) 做滑动平均平滑，得到 u_smooth(t)
      - 再用 compute_phi_E_nosigma(u_smooth, None) 构造 φ_E
    """
    def __init__(self,
                 csv_path: Path,
                 latent_root: Path,
                 W: np.ndarray,
                 window: int = 5,
                 split_names=("TRAIN",),
                 dtype=torch.float32):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.latent_root = Path(latent_root)
        self.split_names = set(split_names)
        self.dtype = dtype
        self.W = W
        self.window = window

        df = pd.read_csv(self.csv_path)
        df = df[df["Split"].isin(self.split_names)].copy()
        self.records = df[["FileName", "EF"]].reset_index(drop=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        fname = str(row["FileName"])
        ef = float(row["EF"])

        latent_path = self.latent_root / f"{fname}.npy"
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing latent file for {fname}: {latent_path}")

        z = np.load(latent_path)   # (T, D)
        u = z @ self.W             # (T, K)
        u_smooth = moving_average_smooth(u, window=self.window)  # (T, K)

        phi = compute_phi_E_nosigma(u_smooth, sigma=None)
        phi_tensor = torch.from_numpy(phi).to(self.dtype)
        ef_tensor = torch.tensor(ef, dtype=self.dtype)

        return phi_tensor, ef_tensor, fname


# =========================================================
# 特征归一化：mean / std（在 train 上）
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


def compute_gp_difficulty_from_file(fname: str,
                                    gp_root: Path = GP_ROOT) -> float:
    """
    给定一个 FileName（不含扩展名），从 {fname}_sigma_cal.npy 里
    计算一个 video-level 的 GP 难度标量 d^{GP}。

    新定义（更有区分度）：
      1) 先把 sigma_cal 所有时间和维度摊平：
           sigma_flat = sigma_cal.reshape(-1)
      2) 计算：
           m = mean(sigma_flat)
           p90 = 90%-percentile(sigma_flat)
           d_raw = 0.5 * m + 0.5 * p90
      3) 再做一次非线性放大：
           d^{GP} = (d_raw)^{gamma}，gamma > 1

    这样高不确定样本会被放大得更多。
    """
    sigma_path = gp_root / f"{fname}_sigma_cal.npy"
    if not sigma_path.exists():
        raise FileNotFoundError(f"Missing GP sigma file for {fname}: {sigma_path}")

    sigma = np.load(sigma_path)  # (T, K)
    sigma_flat = sigma.reshape(-1)

    m = float(sigma_flat.mean())
    p90 = float(np.percentile(sigma_flat, 95.0))

    # mean + 高分位混合
    d_raw = 0.5 * m + 0.5 * p90

    # 非线性放大（可以根据需要调整 gamma）
    gamma = 2
    d_gp = float(d_raw ** gamma)

    return d_gp




def compute_gp_difficulty_batch(fnames: List[str],
                                gp_root: Path = GP_ROOT,
                                device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    批量版本：给一批 fname，返回一个 (B,) 的 tensor d^{GP}_i。
    """
    d_list = []
    for f in fnames:
        d_list.append(compute_gp_difficulty_from_file(f, gp_root=gp_root))
    d_arr = np.array(d_list, dtype=np.float32)  # (B,)
    return torch.from_numpy(d_arr).to(device)


# =========================================================
# Heteroscedastic MLP：输出 μ_EF & σ_EF
# =========================================================

class HeteroscedasticEFMLP(nn.Module):
    """
    输入 φ_E → backbone → μ_EF, σ_EF
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

        self.mu_head = nn.Linear(prev_dim, 1)  # 输出 μ(t)
        self.sigma_head = nn.Linear(prev_dim, 1)  # 输出 σ(t)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, F)
        返回：
          mu:    (B,)
          sigma: (B,)  (已保证 >0)
        """
        h = self.backbone(x)

        mu = self.mu_head(h).squeeze(-1)  # (B,)

        # 用 softplus 保证 σ>0，避免过小/过大的情况
        raw_sigma = self.sigma_head(h).squeeze(-1)  # (B,)
        sigma = torch.nn.functional.softplus(raw_sigma) + 1e-3  # Softplus 保证 > 0

        return mu, sigma



# =========================================================
# 训练 Heteroscedastic MLP
# =========================================================

def heteroscedastic_gaussian_nll(y_true: torch.Tensor,
                                 mu: torch.Tensor,
                                 sigma: torch.Tensor) -> torch.Tensor:
    """
    异方差高斯 NLL（逐样本），不在这里做 mean，外面再 .mean()
    y_true, mu, sigma: (B,)
    返回: (B,)
    """
    var = sigma ** 2
    # 稍微做一下数值保护
    var = torch.clamp(var, min=1e-3, max=1e3)

    # 经典公式：0.5 * [ (y-μ)^2 / σ^2 + log(σ^2) ]
    nll = 0.5 * (((y_true - mu) ** 2) / var + torch.log(var))
    return nll


def train_heteroscedastic_mlp(num_epochs: int = 50,
                              batch_size: int = 64,
                              lr: float = 1e-3,
                              weight_decay: float = 1e-4,
                              hidden_dims: Tuple[int, ...] = (128, 64),
                              lambda_mse: float = 0.1):
    """
    Ours：使用 GP μ+σ 特征的 heteroscedastic MLP。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    feat_mean, feat_std = compute_feature_norm_stats(train_dataset)
    feat_mean = feat_mean.to(device)
    feat_std = feat_std.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    input_dim = train_dataset[0][0].shape[0]
    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for feats, ef, _ in train_loader:
            feats = feats.to(device)
            ef = ef.to(device)

            feats_norm = (feats - feat_mean) / feat_std

            optimizer.zero_grad()
            mu, sigma = model(feats_norm)

            nll = heteroscedastic_gaussian_nll(ef, mu, sigma).mean()
            mse = F.mse_loss(mu, ef)
            loss = nll + lambda_mse * mse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses.append(loss.item())

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
              f"Train loss: {np.mean(train_losses):.4f} | "
              f"Val NLL: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {
                "model": model.state_dict(),
                "feat_mean": feat_mean.cpu(),
                "feat_std": feat_std.cpu(),
                "epoch": epoch + 1,
                "val_rmse": val_rmse,
                "hidden_dims": hidden_dims,
            }
            torch.save(best_state, CHECKPOINT_DIR / "hetero_mlp_best.pt")

    print(f"Training finished. Best Val RMSE = {best_val_rmse:.4f}")


def train_heteroscedastic_mlp_nosigma(num_epochs: int = 50,
                                      batch_size: int = 64,
                                      lr: float = 1e-3,
                                      weight_decay: float = 1e-4,
                                      hidden_dims: Tuple[int, ...] = (128, 64)):
    """
    U0：不使用 σ 特征。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = EchonetFeatureDatasetNoSigma(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES["train"],),
    )
    val_dataset = EchonetFeatureDatasetNoSigma(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES["val"],),
    )

    feat_mean, feat_std = compute_feature_norm_stats(train_dataset)
    feat_mean = feat_mean.to(device)
    feat_std = feat_std.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    input_dim = train_dataset[0][0].shape[0]
    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for feats, ef, _ in train_loader:
            feats = feats.to(device)
            ef = ef.to(device)

            feats_norm = (feats - feat_mean) / feat_std

            optimizer.zero_grad()
            mu, sigma = model(feats_norm)
            nll = heteroscedastic_gaussian_nll(ef, mu, sigma).mean()
            nll.backward()
            optimizer.step()

            train_losses.append(nll.item())

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

        print(f"[U0][Epoch {epoch+1}/{num_epochs}] "
              f"Train NLL: {np.mean(train_losses):.4f} | "
              f"Val NLL: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {
                "model": model.state_dict(),
                "feat_mean": feat_mean.cpu(),
                "feat_std": feat_std.cpu(),
                "epoch": epoch + 1,
                "val_rmse": val_rmse,
                "hidden_dims": hidden_dims,
            }
            torch.save(best_state, CHECKPOINT_DIR / "hetero_mlp_nosigma_best.pt")

    print(f"[U0] Training finished. Best Val RMSE = {best_val_rmse:.4f}")


def train_heteroscedastic_mlp_nogp(num_epochs: int = 50,
                                   batch_size: int = 64,
                                   lr: float = 1e-3,
                                   weight_decay: float = 1e-4,
                                   hidden_dims: Tuple[int, ...] = (128, 64)):
    """
    G0 baseline: 不使用 GP smoothing。
    φ_E 来自 raw u(t)（latent_raw @ W），只含 mean+amp。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    W = load_bottleneck_W()

    train_dataset = EchonetFeatureDataset_NoGP(
        csv_path=CSV_PATH,
        latent_root=LATENT_ROOT,
        W=W,
        split_names=(SPLIT_VALUES["train"],),
    )
    val_dataset = EchonetFeatureDataset_NoGP(
        csv_path=CSV_PATH,
        latent_root=LATENT_ROOT,
        W=W,
        split_names=(SPLIT_VALUES["val"],),
    )

    feat_mean, feat_std = compute_feature_norm_stats(train_dataset)
    feat_mean = feat_mean.to(device)
    feat_std = feat_std.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    input_dim = train_dataset[0][0].shape[0]
    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for feats, ef, _ in train_loader:
            feats = feats.to(device)
            ef = ef.to(device)

            feats_norm = (feats - feat_mean) / feat_std

            optimizer.zero_grad()
            mu, sigma = model(feats_norm)
            nll = heteroscedastic_gaussian_nll(ef, mu, sigma).mean()
            nll.backward()
            optimizer.step()

            train_losses.append(nll.item())

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

        print(f"[G0 NoGP][Epoch {epoch+1}/{num_epochs}] "
              f"Train NLL: {np.mean(train_losses):.4f} | "
              f"Val NLL: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {
                "model": model.state_dict(),
                "feat_mean": feat_mean.cpu(),
                "feat_std": feat_std.cpu(),
                "epoch": epoch + 1,
                "val_rmse": val_rmse,
                "hidden_dims": hidden_dims,
            }
            torch.save(best_state, CHECKPOINT_DIR / "hetero_mlp_nogp_best.pt")

    print(f"[G0 NoGP] Training finished. Best Val RMSE = {best_val_rmse:.4f}")


def train_heteroscedastic_mlp_ma(num_epochs: int = 50,
                                 batch_size: int = 64,
                                 lr: float = 1e-3,
                                 weight_decay: float = 1e-4,
                                 hidden_dims: Tuple[int, ...] = (128, 64),
                                 window: int = 5):
    """
    F0 baseline: 使用 moving-average smoothing 的 heteroscedastic MLP。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    W = load_bottleneck_W()

    train_dataset = EchonetFeatureDataset_MA(
        csv_path=CSV_PATH,
        latent_root=LATENT_ROOT,
        W=W,
        window=window,
        split_names=(SPLIT_VALUES["train"],),
    )
    val_dataset = EchonetFeatureDataset_MA(
        csv_path=CSV_PATH,
        latent_root=LATENT_ROOT,
        W=W,
        window=window,
        split_names=(SPLIT_VALUES["val"],),
    )

    feat_mean, feat_std = compute_feature_norm_stats(train_dataset)
    feat_mean = feat_mean.to(device)
    feat_std = feat_std.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    input_dim = train_dataset[0][0].shape[0]
    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for feats, ef, _ in train_loader:
            feats = feats.to(device)
            ef = ef.to(device)

            feats_norm = (feats - feat_mean) / feat_std

            optimizer.zero_grad()
            mu, sigma = model(feats_norm)
            nll = heteroscedastic_gaussian_nll(ef, mu, sigma).mean()
            nll.backward()
            optimizer.step()

            train_losses.append(nll.item())

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

        print(f"[F0 MA][Epoch {epoch+1}/{num_epochs}] "
              f"Train NLL: {np.mean(train_losses):.4f} | "
              f"Val NLL: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {
                "model": model.state_dict(),
                "feat_mean": feat_mean.cpu(),
                "feat_std": feat_std.cpu(),
                "epoch": epoch + 1,
                "val_rmse": val_rmse,
                "hidden_dims": hidden_dims,
            }
            torch.save(best_state, CHECKPOINT_DIR / "hetero_mlp_ma_best.pt")

    print(f"[F0 MA] Training finished. Best Val RMSE = {best_val_rmse:.4f}")


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

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)
    hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64)))

    val_dataset = EchonetFeatureDataset(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES["val"],),
        use_calibrated_sigma=True,
    )
    input_dim = val_dataset[0][0].shape[0]
    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    all_z = []

    with torch.no_grad():
        for feats, ef, _ in val_loader:
            feats = feats.to(device)
            ef = ef.to(device)
            feats_norm = (feats - feat_mean) / feat_std

            mu, sigma = model(feats_norm)
            residual = torch.abs(ef - mu)
            z = residual / (sigma + 1e-6)
            all_z.append(z.cpu().numpy())

    all_z = np.concatenate(all_z, axis=0)

    q = float(np.quantile(all_z, 1.0 - alpha))
    print(f"Conformal q (1-alpha={1-alpha:.2f}) = {q:.4f}")

    np.save(CHECKPOINT_DIR / "conformal_q.npy",
            np.array(q, dtype=np.float32))
    return q


def compute_conformal_q_nosigma(alpha: float = 0.1,
                                batch_size: int = 64,
                                checkpoint_path: Path = CHECKPOINT_DIR / "hetero_mlp_nosigma_best.pt") -> float:
    """
    U0: 在 VAL split 上计算 |y-μ|/σ_EF 的 (1-alpha) 分位数 q_nosigma。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)
    hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64)))

    val_dataset = EchonetFeatureDatasetNoSigma(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES["val"],),
    )
    input_dim = val_dataset[0][0].shape[0]
    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    all_z = []
    with torch.no_grad():
        for feats, ef, _ in val_loader:
            feats = feats.to(device)
            ef = ef.to(device)
            feats_norm = (feats - feat_mean) / feat_std

            mu, sigma = model(feats_norm)
            residual = torch.abs(ef - mu)
            z = residual / (sigma + 1e-6)
            all_z.append(z.cpu().numpy())

    all_z = np.concatenate(all_z, axis=0)
    q = float(np.quantile(all_z, 1.0 - alpha))
    print(f"[U0] Conformal q_nosigma (1-alpha={1-alpha:.2f}) = {q:.4f}")

    np.save(CHECKPOINT_DIR / "conformal_q_nosigma.npy",
            np.array(q, dtype=np.float32))
    return q


def compute_conformal_q_gp_scaled(alpha: float = 0.1,
                                  beta_grid: Optional[Sequence[float]] = None,
                                  batch_size: int = 64,
                                  checkpoint_path: Path = CHECKPOINT_DIR / "hetero_mlp_best.pt",
                                  cov_tolerance: float = 0.01):
    """
    在 VAL split 上做 GP-scaled σ-aware conformal，并对 β 进行网格搜索：

      sigma_tot^2 = sigma_EF^2 + (beta * d_norm)^2
      z_i = |y_i - μ_i| / sigma_tot,i

    其中 d_norm 是对 d^{GP} 标准化后的非负版本：
      d_norm = max(0, (d^{GP} - d_mean) / d_std)

    步骤：
      1) 在 VAL 上跑一遍模型，收集 residual, sigma_EF, d^{GP}。
      2) 对 beta_grid 中每个 β，计算对应的 z_i，得到 q_β。
      3) 用 q_β 和 sigma_tot_β 估计 VAL 上的 coverage 和 mean width。
      4) 选择 coverage >= target_coverage - cov_tolerance 的、mean width 最小的 β。
         如果所有 β 都达不到 target_coverage - cov_tolerance，则选 coverage 最接近 target 的 β。
      5) 把 q_best, beta_best, d_mean, d_std 保存到 conformal_q_gp_scaled.npy。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if beta_grid is None:
        # 可以根据经验再微调这个网格
        beta_grid = [0.0,0.1,0.3, 0.5, 1.0,1.5, 2.0,2.5, 3.0,3.5,4.0, 5.0,6.0]

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)
    hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64)))

    val_dataset = EchonetFeatureDataset(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES["val"],),
        use_calibrated_sigma=True,
    )
    input_dim = val_dataset[0][0].shape[0]

    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # 收集 VAL 上所有样本的 residual, sigma_EF, d^{GP}
    all_residual = []
    all_sigma_ef = []
    all_dgp = []

    with torch.no_grad():
        for feats, ef, fnames in val_loader:
            feats = feats.to(device)
            ef = ef.to(device)

            feats_norm = (feats - feat_mean) / feat_std
            mu, sigma_ef = model(feats_norm)

            d_gp = compute_gp_difficulty_batch(list(fnames),
                                               gp_root=GP_ROOT,
                                               device=device)  # (B,)

            residual = torch.abs(ef - mu)

            all_residual.append(residual.cpu().numpy())
            all_sigma_ef.append(sigma_ef.cpu().numpy())
            all_dgp.append(d_gp.cpu().numpy())

    residual_all = np.concatenate(all_residual, axis=0)  # (N,)
    sigma_all = np.concatenate(all_sigma_ef, axis=0)     # (N,)
    d_all = np.concatenate(all_dgp, axis=0)              # (N,)

    # 对 d^{GP} 做标准化（避免尺度太大）
    d_mean = float(d_all.mean())
    d_std = float(d_all.std() + 1e-6)
    d_norm = np.maximum(0.0, (d_all - d_mean) / d_std)

    target_coverage = 1.0 - alpha

    beta_list = []
    q_list = []
    cov_list = []
    width_list = []

    for beta in beta_grid:
        sigma_tot = np.sqrt(sigma_all ** 2 + (beta * d_norm) ** 2 + 1e-6)
        z = residual_all / (sigma_tot + 1e-6)

        q_beta = float(np.quantile(z, 1.0 - alpha))

        # 在 VAL 上估计 coverage & mean_width
        covered = residual_all <= q_beta * sigma_tot
        coverage_val = float(covered.mean())
        mean_width_val = float((2.0 * q_beta * sigma_tot).mean())

        beta_list.append(beta)
        q_list.append(q_beta)
        cov_list.append(coverage_val)
        width_list.append(mean_width_val)

        print(f"[GP-scaled][VAL] beta={beta:.3f} "
              f"q={q_beta:.4f}, coverage={coverage_val:.4f}, "
              f"mean_width={mean_width_val:.4f}")
        # 在 compute_conformal_q_gp_scaled() 里，for beta in beta_grid 循环结束后、选择 best_idx 之前/之后都行
        import pandas as pd

        sweep_df = pd.DataFrame({
            "kappa": beta_list,     # 论文里叫 κ；代码里是 beta
            "q": q_list,
            "val_cov": cov_list,
            "val_width": width_list,
        })

        out_csv = CHECKPOINT_DIR / f"kappa_sweep_alpha{alpha:.2f}.csv"
        sweep_df.to_csv(out_csv, index=False)
        print(f"[GP-scaled] Saved kappa sweep to {out_csv}")

    # 先筛出 coverage 足够的 beta
    cov_arr = np.array(cov_list)
    width_arr = np.array(width_list)
    q_arr = np.array(q_list)
    beta_arr = np.array(beta_list, dtype=np.float32)

    ok_mask = cov_arr >= (target_coverage - cov_tolerance)
    if np.any(ok_mask):
        # 在 coverage ≥ target - tol 的 β 中选 mean_width 最小的
        idx = np.argmin(width_arr[ok_mask])
        best_idx = np.where(ok_mask)[0][idx]
        print(f"[GP-scaled] Choose beta from feasible set (coverage>= {target_coverage - cov_tolerance:.3f})")
    else:
        # 否则选 coverage 最接近 target 的 β
        idx = np.argmin(np.abs(cov_arr - target_coverage))
        best_idx = idx
        print(f"[GP-scaled] No beta reaches target coverage; choose beta with closest coverage to {target_coverage:.3f}")

    beta_best = float(beta_arr[best_idx])
    q_best = float(q_arr[best_idx])
    cov_best = float(cov_arr[best_idx])
    width_best = float(width_arr[best_idx])

    print(f"[GP-scaled] Final choice on VAL: "
          f"beta_best={beta_best:.3f}, q_best={q_best:.4f}, "
          f"coverage={cov_best:.4f}, mean_width={width_best:.4f}")
    print(f"[GP-scaled] d_mean={d_mean:.6f}, d_std={d_std:.6f}")

    # 保存 q_best, beta_best, d_mean, d_std 到 .npy
    save_path = CHECKPOINT_DIR / "conformal_q_gp_scaled.npy"
    arr_to_save = np.array([q_best, beta_best, d_mean, d_std], dtype=np.float32)
    np.save(save_path, arr_to_save)
    print(f"[GP-scaled] Saved [q, beta, d_mean, d_std] to {save_path}")

    return q_best




def compute_conformal_q_gp_binned(alpha: float = 0.2,
                                  num_bins: int = 5,
                                  batch_size: int = 64,
                                  checkpoint_path: Path = CHECKPOINT_DIR / "hetero_mlp_best.pt"):
    """
    GP-binned σ-aware conformal：

      1) 按 d^{GP} 分 bin
      2) 在 bin 内用 normalized residual:
           s_i = |y_i - μ_i| / σ_EF,i
         取 (1-alpha) 分位数得到 q_b
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)
    hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64)))

    val_dataset = EchonetFeatureDataset(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES["val"],),
        use_calibrated_sigma=True,
    )
    input_dim = val_dataset[0][0].shape[0]

    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    all_dgp = []
    all_scores = []

    with torch.no_grad():
        for feats, ef, fnames in val_loader:
            feats = feats.to(device)
            ef = ef.to(device)

            feats_norm = (feats - feat_mean) / feat_std
            mu, sigma_ef = model(feats_norm)

            d_gp = compute_gp_difficulty_batch(list(fnames),
                                               gp_root=GP_ROOT,
                                               device=device)

            residual = torch.abs(ef - mu)
            # 关键：用 normalized residual 作为 score
            score = residual / (sigma_ef + 1e-6)

            all_dgp.append(d_gp.cpu().numpy())
            all_scores.append(score.cpu().numpy())

    d_all = np.concatenate(all_dgp, axis=0)
    s_all = np.concatenate(all_scores, axis=0)

    global_q = float(np.quantile(s_all, 1.0 - alpha))

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.quantile(d_all, quantiles)

    q_per_bin = np.zeros(num_bins, dtype=np.float32)
   # q_per_bin = np.clip(q_per_bin, 5, 6)  # 手动调整每个 bin 的 q 范围

    for b in range(num_bins):
        left = bin_edges[b]
        right = bin_edges[b + 1]
        if b == num_bins - 1:
            mask = (d_all >= left) & (d_all <= right)
        else:
            mask = (d_all >= left) & (d_all < right)

        scores_bin = s_all[mask]
        if scores_bin.size < 20:
            q_b = global_q
        else:
            q_b = float(np.quantile(scores_bin, 1.0 - alpha))
        q_per_bin[b] = q_b

        print(f"[GP-binned] Bin {b+1}/{num_bins}: "
              f"range=[{left:.4f}, {right:.4f}], "
              f"count={scores_bin.size}, q_b={q_b:.4f}")

    out = {
        "bin_edges": bin_edges.astype(np.float32),
        "q_per_bin": q_per_bin,
        "alpha": np.array(alpha, dtype=np.float32),
    }
    save_path = CHECKPOINT_DIR / "conformal_q_gp_binned.npz"
    np.savez(save_path, **out)
    print(f"[GP-binned] Saved bin_edges & q_per_bin to {save_path}")

    return bin_edges, q_per_bin


# =========================================================
# 在不同 split 上评估（原始 / GP-scaled / GP-binned）
# =========================================================

def evaluate_on_split(split_key: str = "test",
                      alpha: float = 0.1,
                      batch_size: int = 64,
                      checkpoint_path: Path = CHECKPOINT_DIR / "hetero_mlp_best.pt"):
    """
    原始 σ-aware conformal：
      I_i = [μ_i ± q * σ_EF,i]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)
    hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64)))

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

    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    all_mu, all_sigma, all_y, all_L, all_U = [], [], [], [], []

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

    covered = (y >= L) & (y <= U)
    coverage = covered.mean()
    mean_width = (U - L).mean()
    rmse = np.sqrt(((y - mu) ** 2).mean())
    mae = np.abs(y - mu).mean()

    print(f"[Split={split_key}] α={alpha:.2f}, q={q:.4f}")
    print(f"Point RMSE={rmse:.4f}, MAE={mae:.4f}")
    print(f"Coverage={coverage:.4f}, Mean width={mean_width:.4f}")
    results = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "coverage": float(coverage),
        "mean_width": float(mean_width),
        "q": float(q),
    }
    return results


def evaluate_on_split_nosigma(split_key: str = "test",
                              alpha: float = 0.1,
                              batch_size: int = 64,
                              checkpoint_path: Path = CHECKPOINT_DIR / "hetero_mlp_nosigma_best.pt"):
    """
    U0: 在指定 split 上评估点预测 + conformal 区间。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)
    hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64)))

    q_path = CHECKPOINT_DIR / "conformal_q_nosigma.npy"
    if not q_path.exists():
        raise FileNotFoundError("No conformal_q_nosigma.npy found. Run compute_conformal_q_nosigma() first.")
    q = float(np.load(q_path))

    dataset = EchonetFeatureDatasetNoSigma(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES[split_key],),
    )
    input_dim = dataset[0][0].shape[0]
    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    all_mu, all_sigma, all_y, all_L, all_U = [], [], [], [], []

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

    covered = (y >= L) & (y <= U)
    coverage = covered.mean()
    mean_width = (U - L).mean()
    rmse = np.sqrt(((y - mu) ** 2).mean())
    mae = np.abs(y - mu).mean()

    print(f"[U0][Split={split_key}] α={alpha:.2f}, q_nosigma={q:.4f}")
    print(f"[U0] Point RMSE={rmse:.4f}, MAE={mae:.4f}")
    print(f"[U0] Coverage={coverage:.4f}, Mean width={mean_width:.4f}")
    results = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "coverage": float(coverage),
        "mean_width": float(mean_width),
        "q": float(q),
    }
    return results


def evaluate_on_split_nogp(split_key: str = "test",
                           batch_size: int = 64,
                           checkpoint_path: Path = CHECKPOINT_DIR / "hetero_mlp_nogp_best.pt"):
    """
    G0 baseline: 仅点预测 RMSE/MAE。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)
    hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64)))

    W = load_bottleneck_W()

    dataset = EchonetFeatureDataset_NoGP(
        csv_path=CSV_PATH,
        latent_root=LATENT_ROOT,
        W=W,
        split_names=(SPLIT_VALUES[split_key],),
    )
    input_dim = dataset[0][0].shape[0]
    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    all_mu, all_y = [], []

    with torch.no_grad():
        for feats, ef, _ in loader:
            feats = feats.to(device)
            ef = ef.to(device)
            feats_norm = (feats - feat_mean) / feat_std

            mu, sigma = model(feats_norm)
            all_mu.append(mu.cpu().numpy())
            all_y.append(ef.cpu().numpy())

    mu = np.concatenate(all_mu, axis=0)
    y = np.concatenate(all_y, axis=0)

    rmse = np.sqrt(((y - mu) ** 2).mean())
    mae = np.abs(y - mu).mean()

    print(f"[G0 NoGP][Split={split_key}] Point RMSE={rmse:.4f}, MAE={mae:.4f}")
    results = {
        "RMSE": float(rmse),
        "MAE": float(mae),

    }
    return results


def evaluate_on_split_ma(split_key: str = "test",
                         batch_size: int = 64,
                         checkpoint_path: Path = CHECKPOINT_DIR / "hetero_mlp_ma_best.pt",
                         window: int = 5):
    """
    F0 baseline: 使用 moving-average smoothing 的模型，评估点预测。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)
    hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64)))

    W = load_bottleneck_W()

    dataset = EchonetFeatureDataset_MA(
        csv_path=CSV_PATH,
        latent_root=LATENT_ROOT,
        W=W,
        window=window,
        split_names=(SPLIT_VALUES[split_key],),
    )
    input_dim = dataset[0][0].shape[0]
    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    all_mu, all_y = [], []

    with torch.no_grad():
        for feats, ef, _ in loader:
            feats = feats.to(device)
            ef = ef.to(device)
            feats_norm = (feats - feat_mean) / feat_std

            mu, sigma = model(feats_norm)
            all_mu.append(mu.cpu().numpy())
            all_y.append(ef.cpu().numpy())

    mu = np.concatenate(all_mu, axis=0)
    y = np.concatenate(all_y, axis=0)

    rmse = np.sqrt(((y - mu) ** 2).mean())
    mae = np.abs(y - mu).mean()

    print(f"[F0 MA][Split={split_key}] Point RMSE={rmse:.4f}, MAE={mae:.4f}")
    print(f"[G0 NoGP][Split={split_key}] Point RMSE={rmse:.4f}, MAE={mae:.4f}")
    results = {
        "RMSE": float(rmse),
        "MAE": float(mae),

    }
    return results

def evaluate_on_split_gp_scaled(split_key: str = "test",
                                alpha: float = 0.1,
                                batch_size: int = 64,
                                checkpoint_path: Path = CHECKPOINT_DIR / "hetero_mlp_best.pt",
                                q_path: Path = CHECKPOINT_DIR / "conformal_q_gp_scaled.npy"):
    """
    GP-scaled σ_tot 的 σ-aware conformal：
      I_i = [μ_i ± q * sigma_tot,i]

    其中：
      sigma_tot^2 = sigma_EF^2 + (beta * d_norm)^2
      d_norm = max(0, (d^{GP} - d_mean) / d_std)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)
    hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64)))

    if not q_path.exists():
        raise FileNotFoundError(
            f"No conformal_q_gp_scaled.npy found at {q_path}. "
            f"Please run compute_conformal_q_gp_scaled() first."
        )

    arr = np.load(q_path)
    if arr.shape[0] >= 4:
        q, beta, d_mean, d_std = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
    elif arr.shape[0] == 2:
        q, beta = float(arr[0]), float(arr[1])
        d_mean, d_std = 0.0, 1.0
    else:
        q = float(arr[0])
        beta = 1.0
        d_mean, d_std = 0.0, 1.0

    print(f"[GP-scaled] Loaded q={q:.4f}, beta={beta:.4f}, "
          f"d_mean={d_mean:.6f}, d_std={d_std:.6f}")

    d_mean_t = torch.tensor(d_mean, dtype=torch.float32, device=device)
    d_std_t = torch.tensor(d_std, dtype=torch.float32, device=device)

    dataset = EchonetFeatureDataset(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES[split_key],),
        use_calibrated_sigma=True,
    )
    input_dim = dataset[0][0].shape[0]

    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    all_mu, all_sigma_tot, all_y, all_L, all_U = [], [], [], [], []

    with torch.no_grad():
        for feats, ef, fnames in loader:
            feats = feats.to(device)
            ef = ef.to(device)

            feats_norm = (feats - feat_mean) / feat_std
            mu, sigma_ef = model(feats_norm)

            d_gp = compute_gp_difficulty_batch(list(fnames),
                                               gp_root=GP_ROOT,
                                               device=device)  # (B,)
            # 与 VAL 一致的标准化
            d_norm = torch.clamp((d_gp - d_mean_t) / (d_std_t + 1e-6), min=0.0)

            sigma_tot = torch.sqrt(sigma_ef ** 2 + (beta * d_norm) ** 2 + 1e-6)

            L = mu - q * sigma_tot
            U = mu + q * sigma_tot

            all_mu.append(mu.cpu().numpy())
            all_sigma_tot.append(sigma_tot.cpu().numpy())
            all_y.append(ef.cpu().numpy())
            all_L.append(L.cpu().numpy())
            all_U.append(U.cpu().numpy())

    mu = np.concatenate(all_mu, axis=0)
    sigma_tot = np.concatenate(all_sigma_tot, axis=0)
    y = np.concatenate(all_y, axis=0)
    L = np.concatenate(all_L, axis=0)
    U = np.concatenate(all_U, axis=0)

    covered = (y >= L) & (y <= U)
    coverage = covered.mean()
    mean_width = (U - L).mean()
    rmse = np.sqrt(((y - mu) ** 2).mean())
    mae = np.abs(y - mu).mean()

    print(f"[GP-scaled][Split={split_key}] nominal α={alpha:.2f}")
    print(f"Point RMSE={rmse:.4f}, MAE={mae:.4f}")
    print(f"Coverage={coverage:.4f}, Mean width={mean_width:.4f}")

    results = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "coverage": float(coverage),
        "mean_width": float(mean_width),
        "q": float(q),
        "beta": float(beta),
        "d_mean": float(d_mean),
        "d_std": float(d_std),
    }
    return results




def evaluate_on_split_gp_binned(split_key: str = "test",
                                alpha: float = 0.1,
                                batch_size: int = 64,
                                checkpoint_path: Path = CHECKPOINT_DIR / "hetero_mlp_best.pt",
                                qbin_path: Path = CHECKPOINT_DIR / "conformal_q_gp_binned.npz"):
    """
    GP-binned σ-aware conformal：
      I_i = [μ_i ± q_{b(i)} * σ_EF,i]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)
    hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64)))

    if not qbin_path.exists():
        raise FileNotFoundError(
            f"No conformal_q_gp_binned.npz found at {qbin_path}. "
            f"Please run compute_conformal_q_gp_binned() first."
        )
    data = np.load(qbin_path)
    bin_edges = data["bin_edges"]
    q_per_bin = data["q_per_bin"]
    alpha_saved = float(data["alpha"])
    num_bins = q_per_bin.shape[0]

    print(f"[GP-binned] Loaded {num_bins} bins, alpha_saved={alpha_saved:.3f}")

    dataset = EchonetFeatureDataset(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES[split_key],),
        use_calibrated_sigma=True,
    )
    input_dim = dataset[0][0].shape[0]

    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    all_mu, all_sigma_ef, all_y, all_L, all_U = [], [], [], [], []

    with torch.no_grad():
        for feats, ef, fnames in loader:
            feats = feats.to(device)
            ef = ef.to(device)

            feats_norm = (feats - feat_mean) / feat_std
            mu, sigma_ef = model(feats_norm)

            d_gp = compute_gp_difficulty_batch(list(fnames),
                                               gp_root=GP_ROOT,
                                               device=device)

            mu_np = mu.cpu().numpy()
            sigma_np = sigma_ef.cpu().numpy()
            y_np = ef.cpu().numpy()
            d_np = d_gp.cpu().numpy()

            bin_idx = np.searchsorted(bin_edges, d_np, side="right") - 1
            bin_idx = np.clip(bin_idx, 0, num_bins - 1)

            q_i = q_per_bin[bin_idx]

            L = mu_np - q_i * sigma_np
            U = mu_np + q_i * sigma_np

            all_mu.append(mu_np)
            all_sigma_ef.append(sigma_np)
            all_y.append(y_np)
            all_L.append(L)
            all_U.append(U)

    mu = np.concatenate(all_mu, axis=0)
    sigma_ef = np.concatenate(all_sigma_ef, axis=0)
    y = np.concatenate(all_y, axis=0)
    L = np.concatenate(all_L, axis=0)
    U = np.concatenate(all_U, axis=0)

    covered = (y >= L) & (y <= U)
    coverage = covered.mean()
    mean_width = (U - L).mean()
    rmse = np.sqrt(((y - mu) ** 2).mean())
    mae = np.abs(y - mu).mean()

    
    print(f"[GP-binned][Split={split_key}] nominal α={alpha:.2f}, used alpha_saved={alpha_saved:.2f}")
    print(f"Point RMSE={rmse:.4f}, MAE={mae:.4f}")
    print(f"Coverage={coverage:.4f}, Mean width={mean_width:.4f}")
    print(f"GP difficulty for batch: {d_gp}")
    print(f"Bin edges: {bin_edges}")
    print(f"q_per_bin: {q_per_bin}")
   

    results = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "coverage": float(coverage),
        "mean_width": float(mean_width),
        "nominal α": float(alpha),
    }
    return results

# =========================================================
# 不确定性质量分析：corr & 分桶统计
# =========================================================

def analyze_uncertainty_on_split(split_key: str = "test",
                                 batch_size: int = 64,
                                 num_bins: int = 5,
                                 checkpoint_path: Path = CHECKPOINT_DIR / "hetero_mlp_best.pt"):
    """
    在指定 split 上分析不确定性质量：
      - corr(σ_EF, |y-μ|)
      - 按 σ_EF 分成 num_bins 组，统计每组:
          count / mean|err| / mean width
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)
    hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64)))

    q_path = CHECKPOINT_DIR / "conformal_q.npy"
    if not q_path.exists():
        raise FileNotFoundError("No conformal_q.npy found. Run compute_conformal_q() first.")
    q = float(np.load(q_path))

    dataset = EchonetFeatureDataset(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES[split_key],),
        use_calibrated_sigma=True,
    )
    input_dim = dataset[0][0].shape[0]

    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    all_mu, all_sigma, all_y = [], [], []

    with torch.no_grad():
        for feats, ef, _ in loader:
            feats = feats.to(device)
            ef = ef.to(device)
            feats_norm = (feats - feat_mean) / feat_std

            mu, sigma = model(feats_norm)

            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_y.append(ef.cpu().numpy())

    mu = np.concatenate(all_mu, axis=0)
    sigma = np.concatenate(all_sigma, axis=0)
    y = np.concatenate(all_y, axis=0)

    err = np.abs(y - mu)
    width = 2.0 * q * sigma

    print(f"[Uncertainty analysis] Split={split_key}")

    # ====== 新增：处理 sigma_EF 几乎为常数的情况 ======
    sigma_std = float(np.std(sigma))
    sigma_mean = float(np.mean(sigma))

    if sigma_std < 1e-6:
        print(f"  sigma_EF is (almost) constant: mean={sigma_mean:.3f}, std≈0")
        print("  Pearson corr(sigma_EF, |error|) is undefined when sigma has zero variance.")
        print("\n  Binned statistics by sigma_EF (collapsed to a single bin):")
        mean_err = float(err.mean())
        mean_width = float(width.mean())
        print("  Bin\tRange\t\tCount\tMean|err|\tMean width")
        print(f"  1\t[{sigma_mean:.3f}, {sigma_mean:.3f}]"
              f"\t{len(sigma):d}\t{mean_err:.3f}\t\t{mean_width:.3f}")
        return  # 这里直接返回，不再做下面的分桶
    # ======================================================

    # 正常情况：sigma 有方差，可以算 Pearson 相关和分桶
    corr = np.corrcoef(sigma, err)[0, 1]
    print(f"  Pearson corr(sigma_EF, |error|) = {corr:.4f}")

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.quantile(sigma, quantiles)

    print(f"\n  Binned statistics by sigma_EF (num_bins={num_bins}):")
    print("  Bin\tRange\t\tCount\tMean|err|\tMean width")

    for b in range(num_bins):
        left = bin_edges[b]
        right = bin_edges[b + 1]
        if b == num_bins - 1:
            mask = (sigma >= left) & (sigma <= right)
        else:
            mask = (sigma >= left) & (sigma < right)

        count = int(mask.sum())
        if count == 0:
            mean_err = float("nan")
            mean_width = float("nan")
        else:
            mean_err = float(err[mask].mean())
            mean_width = float(width[mask].mean())

        print(f"  {b+1}\t[{left:.3f}, {right:.3f}]"
              f"\t{count}\t{mean_err:.3f}\t\t{mean_width:.3f}")



def analyze_uncertainty_on_split_nosigma(split_key: str = "test",
                                         batch_size: int = 64,
                                         num_bins: int = 5,
                                         checkpoint_path: Path = CHECKPOINT_DIR / "hetero_mlp_nosigma_best.pt",
                                         q_path: Path = CHECKPOINT_DIR / "conformal_q_nosigma.npy"):
    """
    U0：不使用 σ 特征版本的不确定性分析。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)
    hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64)))

    if not q_path.exists():
        raise FileNotFoundError(f"No q file found: {q_path}")
    q = float(np.load(q_path))

    dataset = EchonetFeatureDatasetNoSigma(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES[split_key],),
    )
    input_dim = dataset[0][0].shape[0]

    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    all_mu, all_sigma, all_y = [], [], []
    with torch.no_grad():
        for feats, ef, _ in loader:
            feats = feats.to(device)
            ef = ef.to(device)
            feats_norm = (feats - feat_mean) / feat_std

            mu, sigma = model(feats_norm)
            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_y.append(ef.cpu().numpy())

    mu = np.concatenate(all_mu, axis=0)
    sigma = np.concatenate(all_sigma, axis=0)
    y = np.concatenate(all_y, axis=0)

    err = np.abs(y - mu)
    width = 2.0 * q * sigma

    corr = np.corrcoef(sigma, err)[0, 1]
    print(f"[U0 Uncertainty analysis] Split={split_key}")
    print(f"  Pearson corr(sigma_EF, |error|) = {corr:.4f}")

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.quantile(sigma, quantiles)

    print(f"\n  [U0] Binned statistics by sigma_EF (num_bins={num_bins}):")
    print("  Bin\tRange\t\tCount\tMean|err|\tMean width")

    for b in range(num_bins):
        left = bin_edges[b]
        right = bin_edges[b + 1]
        if b == num_bins - 1:
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
# 简单脚本入口（可选）
# =========================================================

if __name__ == "__main__":
    # 示例：训练 Ours 并在 test 上评估
    train_heteroscedastic_mlp(
        num_epochs=50,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dims=(128, 64),
    )

    compute_conformal_q(alpha=0.1)
    evaluate_on_split(split_key="test", alpha=0.1)
