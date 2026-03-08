# fit_gp_and_calibrate.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

from dataset_and_config import (
    EchonetLatentDataset,
    LATENT_ROOT,
    CSV_PATH,
    SPLIT_VALUES,
)

# 瓶颈矩阵保存位置（上一节输出）
W_PATH = Path("checkpoints/bottleneck_W.npy")

# GP 输出保存目录
GP_OUTPUT_ROOT = Path("/mnt/EF_measurement/data/echonet/latents_gp")
GP_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_bottleneck_matrix():
    W = np.load(W_PATH)  # (D, K)
    return W


def make_gp_model():
    """
    定义 1D GP 模型。
    可以根据需要调 kernel 超参数。
    """
    # 常数核 * RBF + white noise
    kernel = C(1.0, constant_value_bounds="fixed") * \
             RBF(length_scale=10.0, length_scale_bounds="fixed") + \
             WhiteKernel(noise_level=1e-2, noise_level_bounds="fixed")

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        optimizer=None,      # 🔴 关闭超参数优化
        normalize_y=True,
    )
    return gp


def fit_gp_for_dataset(split_name: str, W: np.ndarray):
    """
    对指定 split（TRAIN/VAL/TEST）所有样本：
    - 应用瓶颈 W: z (T, D) -> u (T, K)
    - 对每个维度做 1D GP 拟合，保存 μ, σ_raw 到磁盘
    """
    # 读取 csv，只是为了遍历 file list
    df = pd.read_csv(CSV_PATH)
    df_split = df[df["Split"] == split_name].copy().reset_index(drop=True)

    D, K = W.shape

    for idx, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Fitting GP for split={split_name}"):
        fname = str(row["FileName"])
        latent_path = LATENT_ROOT / f"{fname}.npy"
        if not latent_path.exists():
            print(f"[WARN] latent not found: {latent_path}, skip.")
            continue

        # 1. 读 latent
        z = np.load(latent_path)  # (T, D)
        T, D_check = z.shape
        if D_check != D:
            raise ValueError(f"Latent dim mismatch for {fname}: got {D_check}, expected {D}")

        # 2. 应用瓶颈 W: (T, D) @ (D, K) -> (T, K)
        u = z @ W  # (T, K)

        # 3. 为该样本建立 GP，并求 μ, σ_raw
        t_grid = np.arange(T).reshape(-1, 1)  # (T, 1)

        mu = np.zeros_like(u)       # (T, K)
        sigma_raw = np.zeros_like(u)  # (T, K)

        for k in range(K):
            y = u[:, k:k+1]  # (T, 1)
            gp = make_gp_model()
            gp.fit(t_grid, y)
            mu_k, std_k = gp.predict(t_grid, return_std=True)
            mu[:, k] = np.asarray(mu_k).reshape(-1)       # 保证是一维 (T,)
            sigma_raw[:, k] = np.asarray(std_k).reshape(-1)


        # 4. 保存
        out_mu = GP_OUTPUT_ROOT / f"{fname}_mu.npy"
        out_sigma = GP_OUTPUT_ROOT / f"{fname}_sigma_raw.npy"
        np.save(out_mu, mu)
        np.save(out_sigma, sigma_raw)


def calibrate_sigma_global(splits_for_calib=("TRAIN", "VAL"),
                           target_coverage=0.9,
                           eps=1e-8,
                           return_r_values: bool = False):
    """
    ...
    如果 return_r_values=True，则返回 (s_star, all_r_flat)；
    否则只返回 s_star。
    """
    W = load_bottleneck_matrix()
    D, K = W.shape

    all_r = []

    df = pd.read_csv(CSV_PATH)
    for split_name in splits_for_calib:
        df_split = df[df["Split"] == split_name].copy().reset_index(drop=True)
        for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Calib r for split={split_name}"):
            fname = str(row["FileName"])
            latent_path = LATENT_ROOT / f"{fname}.npy"
            mu_path = GP_OUTPUT_ROOT / f"{fname}_mu.npy"
            sigma_path = GP_OUTPUT_ROOT / f"{fname}_sigma_raw.npy"

            if not (latent_path.exists() and mu_path.exists() and sigma_path.exists()):
                print(f"[WARN] missing file(s) for {fname}, skip.")
                continue

            z = np.load(latent_path)        # (T, D)
            u = z @ W                       # (T, K)
            mu = np.load(mu_path)           # (T, K)
            sigma_raw = np.load(sigma_path) # (T, K)

            if u.shape != mu.shape or u.shape != sigma_raw.shape:
                raise ValueError(f"Shape mismatch for {fname}: u={u.shape}, mu={mu.shape}, sigma={sigma_raw.shape}")

            residual = np.abs(u - mu)     # (T, K)
            r = residual / (sigma_raw + eps)
            all_r.append(r.reshape(-1))   # 展平成一维

    if not all_r:
        raise RuntimeError("No r values collected for calibration.")

    all_r = np.concatenate(all_r, axis=0)  # (N_total,)

    s_star = np.quantile(all_r, target_coverage)
    print(f"Global sigma scale s* = {s_star:.4f} for target coverage {target_coverage}")

    scale_path = GP_OUTPUT_ROOT / "sigma_scale_s_star.npy"
    np.save(scale_path, np.array(s_star, dtype=np.float32))
    print(f"Saved s* to {scale_path}")

    if return_r_values:
        return s_star, all_r
    else:
        return s_star



def apply_sigma_scale_to_all_splits(s_star: float, splits=("TRAIN", "VAL", "TEST")):
    """
    把 σ_raw 统一乘上 s*，得到 σ_calibrated，并保存。
    """
    df = pd.read_csv(CSV_PATH)

    for split_name in splits:
        df_split = df[df["Split"] == split_name].copy().reset_index(drop=True)
        for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Apply s* to split={split_name}"):
            fname = str(row["FileName"])
            sigma_path = GP_OUTPUT_ROOT / f"{fname}_sigma_raw.npy"
            if not sigma_path.exists():
                print(f"[WARN] sigma_raw not found for {fname}, skip.")
                continue

            sigma_raw = np.load(sigma_path)
            sigma_cal = s_star * sigma_raw
            out_path = GP_OUTPUT_ROOT / f"{fname}_sigma_cal.npy"
            np.save(out_path, sigma_cal)


def main_fit_gp_and_calibrate():
    # 1. 加载瓶颈矩阵 W
    W = load_bottleneck_matrix()
    print(f"Loaded bottleneck W with shape {W.shape}")

    # 2. 为 TRAIN/VAL/TEST 分别拟合 GP（你也可以只先做 TRAIN+VAL）
    for split in [SPLIT_VALUES["train"], SPLIT_VALUES["val"], SPLIT_VALUES["test"]]:
        fit_gp_for_dataset(split_name=split, W=W)

    # 3. 用 TRAIN+VAL 做 σ 标定，求 s*
    s_star = calibrate_sigma_global(
        splits_for_calib=(SPLIT_VALUES["train"], SPLIT_VALUES["val"]),
        target_coverage=0.9  # 你可以根据需要改 0.68/0.95 等
    )

    # 4. 把 σ_raw 统一乘以 s*，得到 σ_cal
    apply_sigma_scale_to_all_splits(s_star, splits=(SPLIT_VALUES["train"], SPLIT_VALUES["val"], SPLIT_VALUES["test"]))


if __name__ == "__main__":
    main_fit_gp_and_calibrate()
