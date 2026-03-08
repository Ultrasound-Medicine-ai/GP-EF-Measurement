import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_and_config import CSV_PATH, LATENT_ROOT, SPLIT_VALUES  # 数据与路径配置
from fit_gp_and_calibrate import load_bottleneck_matrix, make_gp_model  # 瓶颈矩阵 + GP 模型
from ef_features_hetero_conformal import (
    EchonetFeatureDataset,
    HeteroscedasticEFMLP,
    GP_ROOT,
    CHECKPOINT_DIR,
    train_heteroscedastic_mlp,
)


# =========================================================
# 1. 单个视频 1D GP 拟合时间
# =========================================================

def fit_gp_for_single_video(u: np.ndarray):
    """
    对单个视频的瓶颈轨迹 u(t) 做 GP 拟合，和 fit_gp_for_dataset 保持一致风格：
      - u: (T, K)
      - 对每个维度 k 拟合 GP，返回 μ(t), σ(t)
    """
    T, K = u.shape
    t_grid = np.arange(T).reshape(-1, 1)  # (T, 1)

    mu = np.zeros_like(u)         # (T, K)
    sigma_raw = np.zeros_like(u)  # (T, K)

    for k in range(K):
        y = u[:, k:k+1]  # (T, 1)
        gp = make_gp_model()
        gp.fit(t_grid, y)
        mu_k, std_k = gp.predict(t_grid, return_std=True)
        mu[:, k] = np.asarray(mu_k).reshape(-1)
        sigma_raw[:, k] = np.asarray(std_k).reshape(-1)

    return mu, sigma_raw


def benchmark_gp_per_video(split_key: str = "test",
                           max_samples: int = 50):
    """
    在指定 split 上统计：
      - 单个视频做 1D GP 拟合的平均时间
    不修改磁盘，只在内存中拟合。
    """
    split_name = SPLIT_VALUES[split_key]
    df = pd.read_csv(CSV_PATH)
    df = df[df["Split"] == split_name].reset_index(drop=True)

    # 瓶颈矩阵 W: (D, K)
    W = load_bottleneck_matrix()

    total_time = 0.0
    n_runs = 0

    for _, row in df.iterrows():
        fname = str(row["FileName"])
        latent_path = LATENT_ROOT / f"{fname}.npy"
        if not latent_path.exists():
            print(f"[WARN] Latent not found: {latent_path}, skip.")
            continue

        z = np.load(latent_path)  # (T, D)
        u = z @ W                 # (T, K)

        t0 = time.perf_counter()
        _ = fit_gp_for_single_video(u)
        t1 = time.perf_counter()

        total_time += (t1 - t0)
        n_runs += 1

        if n_runs >= max_samples:
            break

    if n_runs == 0:
        print("No valid samples for GP benchmark.")
        return

    avg_time = total_time / n_runs
    print(f"[GP benchmark] split={split_key}, N={n_runs}")
    print(f"Average GP fitting time per video: {avg_time:.4f} s")


# =========================================================
# 2. EF 头训练时间（Ours: GP μ+σ 特征）
# =========================================================

def benchmark_training_time(num_epochs: int = 12):
    """
    粗略统计 EF 头训练耗时（Ours GP μ+σ，和 exp2_train_and_eval_all 一致的设置）。
    注意：这里会真的训练一次 heteroscedastic MLP。
    """
    t0 = time.perf_counter()
    train_heteroscedastic_mlp(
        num_epochs=num_epochs,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dims=(128, 64),
        lambda_mse=0.1,
    )
    t1 = time.perf_counter()

    elapsed = t1 - t0
    print(f"[Training benchmark] Ours EF head, epochs={num_epochs}")
    print(f"Total training time: {elapsed:.1f} s ({elapsed/60:.1f} min)")


# =========================================================
# 3. 推理阶段：φ_E → EF 预测的平均时间
# =========================================================

def benchmark_inference_time(split_key: str = "test",
                             max_samples: int = 200):
    """
    统计推理阶段的平均单视频时间：
      - 假设 GP μ/σ_cal 已离线预计算并保存在 GP_ROOT
      - 使用 EchonetFeatureDataset 读取 φ_E
      - 使用 hetero MLP 做一次 forward，计时
    I/O 和数据加载不计入（只计 forward）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 载入最优 EF 头 checkpoint
    ckpt_path = CHECKPOINT_DIR / "hetero_mlp_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)
    hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64)))

    split_name = SPLIT_VALUES[split_key]
    dataset = EchonetFeatureDataset(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(split_name,),
        use_calibrated_sigma=True,
    )
    input_dim = dataset[0][0].shape[0]

    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=0, pin_memory=True)

    times = []
    n_runs = 0

    for feats, ef, fname in loader:
        feats = feats.to(device)  # (1, F)
        feats_norm = (feats - feat_mean) / feat_std

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _mu, _sigma = model(feats_norm)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append(t1 - t0)
        n_runs += 1
        if n_runs >= max_samples:
            break

    if n_runs == 0:
        print("No valid samples for inference benchmark.")
        return

    times = np.array(times, dtype=np.float64)
    avg_ms = times.mean() * 1000.0
    std_ms = times.std() * 1000.0

    print(f"[Inference benchmark] split={split_key}, N={n_runs}")
    print(f"Average EF inference time per video: {avg_ms:.2f} ± {std_ms:.2f} ms")


# =========================================================
# 主入口：按需调用
# =========================================================

if __name__ == "__main__":
    # 1) 单个视频 GP 拟合时间（离线步骤）
    benchmark_gp_per_video(split_key="test", max_samples=50)

    # 2) EF 头训练时间（可以只在需要时打开）
    benchmark_training_time(num_epochs=12)

    # 3) 推理阶段平均延迟（在线）
    benchmark_inference_time(split_key="test", max_samples=200)
