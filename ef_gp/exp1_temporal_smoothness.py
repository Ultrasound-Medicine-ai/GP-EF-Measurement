# exp1_temporal_smoothness.py
import numpy as np
import pandas as pd
from pathlib import Path

from dataset_and_config import LATENT_ROOT, CSV_PATH, SPLIT_VALUES
from ef_features_hetero_conformal import load_bottleneck_W, moving_average_smooth

GP_ROOT = Path("/mnt/EF_measurement/data/echonet/latents_gp")  # 按你实际的 GP 输出目录改


def tv1_tv2_hf(x_1d: np.ndarray, hf_ratio: float = 0.25):
    """
    x_1d: (T,)
    返回：TV1, TV2, HF%
    """
    x = x_1d.astype(np.float32)
    T = x.shape[0]

    # 一阶 TV
    tv1 = np.abs(x[1:] - x[:-1]).mean()

    # 二阶 TV
    if T > 2:
        tv2 = np.abs(x[2:] - 2 * x[1:-1] + x[:-2]).mean()
    else:
        tv2 = 0.0

    # 高频能量占比
    x_centered = x - x.mean()
    if T <= 1:
        return float(tv1), float(tv2), 0.0

    X = np.fft.rfft(x_centered)
    power = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(T, d=1.0)  # 采样间隔按 1 frame

    if freqs.size <= 1:
        hf = 0.0
    else:
        f_cut = freqs.max() * hf_ratio
        mask_hf = freqs >= f_cut
        E_total = power.sum() + 1e-8
        E_hf = power[mask_hf].sum()
        hf = float(E_hf / E_total)

    return float(tv1), float(tv2), float(hf)


def stats_for_seq(seq_2d: np.ndarray):
    """
    seq_2d: (T, K) 时间 × 瓶颈维度
    返回每个样本的 (TV1_mean, TV2_mean, HF%_mean)
    """
    T, K = seq_2d.shape
    tv1_list, tv2_list, hf_list = [], [], []
    for k in range(K):
        tv1, tv2, hf = tv1_tv2_hf(seq_2d[:, k])
        tv1_list.append(tv1)
        tv2_list.append(tv2)
        hf_list.append(hf)
    return np.mean(tv1_list), np.mean(tv2_list), np.mean(hf_list)


def summarize_temporal_metrics(split_key="train"):
    """
    在指定 split 上统计：
      raw u(t), MA u(t), GP μ(t) 的 TV1 / TV2 / HF%
    """
    split_name = SPLIT_VALUES[split_key]
    df = pd.read_csv(CSV_PATH)
    df = df[df["Split"] == split_name].copy().reset_index(drop=True)

    W = load_bottleneck_W()  # (D, K)
    D, K = W.shape

    stats_raw, stats_ma, stats_gp = [], [], []

    for _, row in df.iterrows():
        fname = str(row["FileName"])

        latent_path = LATENT_ROOT / f"{fname}.npy"
        mu_path = GP_ROOT / f"{fname}_mu.npy"
        if not latent_path.exists() or not mu_path.exists():
            continue

        z = np.load(latent_path)   # (T, D)
        u = z @ W                  # (T, K)
        u_ma = moving_average_smooth(u, window=5)  # (T, K)
        mu = np.load(mu_path)      # (T, K)

        stats_raw.append(stats_for_seq(u))
        stats_ma.append(stats_for_seq(u_ma))
        stats_gp.append(stats_for_seq(mu))

    def mean_std(arr_list):
        arr = np.array(arr_list)  # (N, 3)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        return mean, std

    m_raw, s_raw = mean_std(stats_raw)
    m_ma, s_ma = mean_std(stats_ma)
    m_gp, s_gp = mean_std(stats_gp)

    print(f"=== Temporal smoothness on split={split_key} (N={len(stats_raw)}) ===")
    print("Method\tTV1\t\tTV2\t\tHF%")
    print(f"raw u(t)\t{m_raw[0]:.4f}±{s_raw[0]:.4f}\t{m_raw[1]:.4f}±{s_raw[1]:.4f}\t{m_raw[2]:.3f}±{s_raw[2]:.3f}")
    print(f"MA u(t)\t{m_ma[0]:.4f}±{s_ma[0]:.4f}\t{m_ma[1]:.4f}±{s_ma[1]:.4f}\t{m_ma[2]:.3f}±{s_ma[2]:.3f}")
    print(f"GP μ(t)\t{m_gp[0]:.4f}±{s_gp[0]:.4f}\t{m_gp[1]:.4f}±{s_gp[1]:.4f}\t{m_gp[2]:.3f}±{s_gp[2]:.3f}")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        summarize_temporal_metrics(split_key=split)

