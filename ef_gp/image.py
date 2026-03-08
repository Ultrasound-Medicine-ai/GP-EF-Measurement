"""
绘图脚本：基于现有训练结果画 I2MTC 论文中的分析图

包含：
1) latent 轨迹对比图：B0/G0_NoGP, B1/F0_MA, Ours(GP μ±2σ)
2) σ_EF vs |error| & 区间宽度 图
3) 归一化残差 z=|y-μ|/σ 的 CDF（conformal 前后）
4) EF 散点 + 误差条、Bland–Altman 图

使用前提：
- 已经运行过 fit_gp_and_calibrate.py（生成 μ / σ_cal 和 W）。
- 已经训练过 Ours / G0 / F0，并在 VAL 上算好 conformal q，
  即 CHECKPOINT_DIR 下存在 hetero_mlp_best.pt, conformal_q.npy 等。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from pathlib import Path

# ===== 全局样式配置（按要求设置）=====
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# 方法对应颜色（低饱和度：柔和蓝 / 绿 / 橙）
method_colors = {
    'raw u(t)': '#6baed6',    # 柔和蓝（B0）
    'MA u(t)': '#74c476',     # 柔和绿（B1）
    'GP μ(t)': '#fd8d3c'      # 柔和橙（Ours）
}
fill_color = '#fd8d3c80'     # GP 区间填充色（橙色半透明）
sigma_error_color = '#2c7bb6'# σ_EF vs error 颜色（深蓝）
width_color = '#d73027'      # 区间宽度颜色（深红）
scatter_color = '#3182bd'    # 散点图颜色（蓝）
bland_altman_color = '#6baed6'# Bland-Altman 散点色（灰）
reference_line_color = '#636363'# 参考线颜色（深灰）

# ===== 导入你工程里的模块 =====
from dataset_and_config import LATENT_ROOT, CSV_PATH, SPLIT_VALUES
from ef_features_hetero_conformal import (
    W_PATH,
    GP_ROOT,
    CHECKPOINT_DIR,
    HeteroscedasticEFMLP,
    EchonetFeatureDataset,
    EchonetFeatureDataset_NoGP,
    EchonetFeatureDataset_MA,
    moving_average_smooth,
)

# ---------------------------------------------------------
# 一些共用的小工具
# ---------------------------------------------------------

def load_best_model(checkpoint_name: str = "hetero_mlp_best.pt"):
    """加载 Ours 的最优 heteroscedastic MLP 和特征归一化统计。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CHECKPOINT_DIR / checkpoint_name
    ckpt = torch.load(ckpt_path, map_location=device)

    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)
    hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64)))

    # 用 EchonetFeatureDataset 推断输入维度
    tmp_dataset = EchonetFeatureDataset(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES["test"],),
        use_calibrated_sigma=True,
    )
    input_dim = tmp_dataset[0][0].shape[0]

    model = HeteroscedasticEFMLP(input_dim=input_dim,
                                 hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, feat_mean, feat_std, device


def collect_predictions_on_split(split_key="test"):
    """
    收集某个 split 上 Ours 的预测：
      返回 dict: { 'y':..., 'mu':..., 'sigma':..., 'L':..., 'U':..., 'fname':... }
    """
    # 1) 加载模型 & q
    model, feat_mean, feat_std, device = load_best_model()
    q_path = CHECKPOINT_DIR / "conformal_q.npy"
    q = float(np.load(q_path))

    # 2) 构造数据集和 dataloader
    dataset = EchonetFeatureDataset(
        csv_path=CSV_PATH,
        gp_root=GP_ROOT,
        split_names=(SPLIT_VALUES[split_key],),
        use_calibrated_sigma=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    all_mu, all_sigma, all_y, all_L, all_U, all_fname = [], [], [], [], [], []

    with torch.no_grad():
        for feats, ef, fnames in loader:
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
            all_fname.extend(list(fnames))

    mu = np.concatenate(all_mu, axis=0)
    sigma = np.concatenate(all_sigma, axis=0)
    y = np.concatenate(all_y, axis=0)
    L = np.concatenate(all_L, axis=0)
    U = np.concatenate(all_U, axis=0)
    fnames = np.array(all_fname)

    return {
        "mu": mu,
        "sigma": sigma,
        "y": y,
        "L": L,
        "U": U,
        "fname": fnames,
        "q": q,
    }


# =========================================================
# 1) latent 轨迹对比图：B0 / B1 / Ours
# =========================================================

def get_latent_trajectories(fname: str, dim: int = 0, ma_window: int = 5):
    """
    对给定样本 FileName（不带扩展名）：
      - 读取 raw latent z, 计算 u_raw = z @ W
      - 计算 moving-average 平滑 u_ma
      - 读取 GP μ(t), σ_cal(t)
    返回 dict，包含该维度上的 (T,) 序列。
    """
    # load W
    W = np.load(W_PATH)          # (D, K)
    latent_path = LATENT_ROOT / f"{fname}.npy"
    z = np.load(latent_path)     # (T, D)
    u = z @ W                    # (T, K)
    T = u.shape[0]

    u_raw_1d = u[:, dim]         # (T,)

    # moving average
    u_ma = moving_average_smooth(u, window=ma_window)
    u_ma_1d = u_ma[:, dim]

    # GP μ & σ_cal
    mu_path = GP_ROOT / f"{fname}_mu.npy"
    sigma_cal_path = GP_ROOT / f"{fname}_sigma_cal.npy"
    mu = np.load(mu_path)            # (T, K)
    sigma_cal = np.load(sigma_cal_path)
    mu_1d = mu[:, dim]
    sigma_1d = sigma_cal[:, dim]

    t = np.arange(T)

    return {
        "t": t,
        "u_raw": u_raw_1d,
        "u_ma": u_ma_1d,
        "mu_gp": mu_1d,
        "sigma_gp": sigma_1d,
    }


def select_easy_and_hard_cases(split_key="test"):
    """
    从 test split 里挑选一个“容易”（误差最小）和一个“困难”（误差最大）的样本。
    返回 two file names: (easy_fname, hard_fname)
    """
    preds = collect_predictions_on_split(split_key)
    y = preds["y"]
    mu = preds["mu"]
    fnames = preds["fname"]

    err = np.abs(y - mu)
    easy_idx = np.argmin(err)
    hard_idx = np.argmax(err)

    easy_fname = fnames[easy_idx]
    hard_fname = fnames[hard_idx]

    print(f"Easy case: {easy_fname}, |err|={err[easy_idx]:.2f}")
    print(f"Hard case: {hard_fname}, |err|={err[hard_idx]:.2f}")
    return easy_fname, hard_fname


def plot_latent_trajectory_comparison(dim: int = 0, ma_window: int = 5,
                                      split_key="test",
                                      save_path: Path = Path("fig_latent_traj.png")):
    """
    Fig: latent 轨迹对比 (B0 raw, B1 moving average, Ours GP μ±2σ)
    - 两行：上面 easy case，下面 hard case
    - 每行在同一子图展示三种曲线
    """
    easy_fname, hard_fname = select_easy_and_hard_cases(split_key)

    cases = [easy_fname, hard_fname]
    titles = ["Easy case", "Hard case"]

    fig, axes = plt.subplots(len(cases), 1, figsize=(6, 4 * len(cases)), sharex=True)

    if len(cases) == 1:
        axes = [axes]

    for ax, fname, title in zip(axes, cases, titles):
        traj = get_latent_trajectories(fname, dim=dim, ma_window=ma_window)
        t = traj["t"]

        # 绘制三条曲线（使用指定颜色）
        ax.plot(t, traj["u_raw"], label="B0: raw u(t)", 
                color=method_colors['raw u(t)'], linewidth=1.5)
        ax.plot(t, traj["u_ma"], label=f"B1: MA (w={ma_window})", 
                color=method_colors['MA u(t)'], linewidth=1.5)
        ax.plot(t, traj["mu_gp"], label="Ours: GP μ(t)", 
                color=method_colors['GP μ(t)'], linewidth=2.0)  # 加粗突出Ours
        # 填充GP区间（半透明橙色）
        ax.fill_between(
            t,
            traj["mu_gp"] - 2 * traj["sigma_gp"],
            traj["mu_gp"] + 2 * traj["sigma_gp"],
            alpha=0.3,
            color=fill_color,
            label="Ours: μ ± 2σ",
        )

        ax.set_ylabel("Latent value")
        ax.set_title(f"{title} (File={fname}, dim={dim})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)

    axes[-1].set_xlabel("Frame index t")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved latent trajectory figure to {save_path}")


# =========================================================
# 2) σ_EF vs |error| & 区间宽度
# =========================================================

def plot_sigma_vs_error(num_bins: int = 5,
                        split_key: str = "test",
                        save_path: Path = Path("fig_sigma_vs_error.png")):
    """
    按 σ_EF 分成 num_bins 组，画每组的平均 |error| 和 平均区间宽度。
    使用双 y 轴：左 y 轴是 mean |error|，右 y 轴是 mean width。
    """
    preds = collect_predictions_on_split(split_key)
    y = preds["y"]
    mu = preds["mu"]
    sigma = preds["sigma"]
    q = preds["q"]

    err = np.abs(y - mu)
    width = 2.0 * q * sigma

    # 分桶
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.quantile(sigma, quantiles)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mean_err = []
    mean_width = []
    counts = []

    for b in range(num_bins):
        left, right = bin_edges[b], bin_edges[b + 1]
        if b == num_bins - 1:
            mask = (sigma >= left) & (sigma <= right)
        else:
            mask = (sigma >= left) & (sigma < right)

        counts.append(mask.sum())
        if mask.sum() == 0:
            mean_err.append(np.nan)
            mean_width.append(np.nan)
        else:
            mean_err.append(err[mask].mean())
            mean_width.append(width[mask].mean())

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    # 绘制两条曲线（使用指定颜色）
    line1 = ax1.plot(bin_centers, mean_err, marker="o", markersize=6, 
                     color=sigma_error_color, linewidth=2.0, label="Mean |error|")
    line2 = ax2.plot(bin_centers, mean_width, marker="s", markersize=6, 
                     color=width_color, linewidth=2.0, label="Mean interval width")

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", framealpha=0.9)

    ax1.set_xlabel(r"$\sigma_{\mathrm{EF}}$ bin center")
    ax1.set_ylabel("Mean |error| (%)", color=sigma_error_color)
    ax2.set_ylabel("Mean interval width (%)", color=width_color)
    ax1.tick_params(axis='y', labelcolor=sigma_error_color)
    ax2.tick_params(axis='y', labelcolor=width_color)
    ax1.grid(True, alpha=0.3)

    plt.title(f"$\sigma_{{EF}}$ vs Error / Interval Width ({split_key} split)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved sigma-vs-error figure to {save_path}")


# =========================================================
# 3) 归一化残差 z 的 CDF（conformal 前 / 后）
# =========================================================

def compute_z_values(split_key="val"):
    """
    在某个 split 上计算：
      z_raw = |y-μ| / σ
      z_conf = |y-μ| / (q * σ)
    用于画 CDF。
    """
    preds = collect_predictions_on_split(split_key)
    y = preds["y"]
    mu = preds["mu"]
    sigma = preds["sigma"]
    q = preds["q"]

    z_raw = np.abs(y - mu) / (sigma + 1e-6)
    z_conf = z_raw / (q + 1e-6)
    return z_raw, z_conf, q


def plot_z_cdf(split_key="val",
               save_path: Path = Path("fig_z_cdf.png")):
    """
    绘制经验 CDF：
      - z_raw 的 CDF
      - z_conf 的 CDF
    并在 x=1 处画一条竖线，对应 “理想 90% 覆盖”。
    """
    z_raw, z_conf, q = compute_z_values(split_key)

    def ecdf(x):
        xs = np.sort(x)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        return xs, ys

    xs_raw, ys_raw = ecdf(z_raw)
    xs_conf, ys_conf = ecdf(z_conf)

    fig, ax = plt.subplots(figsize=(6, 4))
    # 绘制两条CDF曲线（使用指定颜色）
    ax.plot(xs_raw, ys_raw, linewidth=2.0, 
            color=method_colors['raw u(t)'], label=r"Raw: $|y-\mu|/\sigma$")
    ax.plot(xs_conf, ys_conf, linewidth=2.0, 
            color=method_colors['GP μ(t)'], label=r"Post-conformal: $|y-\mu|/(q\sigma)$")

    # 参考线（x=1，理想90%覆盖）
    ax.axvline(1.0, linestyle="--", linewidth=1.5, 
               color=reference_line_color, alpha=0.8, label=r"$z=1$ (Ideal 90% coverage)")

    ax.set_xlabel("Normalized residual $z$")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"ECDF of Normalized Residuals ({split_key} split)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved z-CDF figure to {save_path}")
    print(f"Conformal q on this split = {q:.3f}")


# =========================================================
# 4) EF 散点图 + 误差条；Bland–Altman 图
# =========================================================

def plot_ef_scatter_with_errorbars(split_key="test",
                                   max_points: int = 300,
                                   save_path: Path = Path("fig_scatter_errorbar.png")):
    """
    画 GT EF vs 预测 EF 的散点图，并画 ±qσ 的竖向误差条。
    为避免太挤，可以随机采样一部分点。
    """
    preds = collect_predictions_on_split(split_key)
    y = preds["y"]
    mu = preds["mu"]
    sigma = preds["sigma"]
    q = preds["q"]

    N = len(y)
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        y = y[idx]
        mu = mu[idx]
        sigma = sigma[idx]

    fig, ax = plt.subplots(figsize=(5, 5))

    # 绘制散点+误差条（使用指定颜色）
    ax.errorbar(y, mu, yerr=q * sigma, fmt="o", alpha=0.6, capsize=2, 
                color=scatter_color, markersize=4, ecolor=scatter_color, elinewidth=1)
    # y=x参考线
    min_val = min(y.min(), mu.min()) - 5
    max_val = max(y.max(), mu.max()) + 5
    ax.plot([min_val, max_val], [min_val, max_val], linewidth=1.5, 
            color=reference_line_color, linestyle="-", alpha=0.8)

    ax.set_xlabel("Ground truth EF (%)")
    ax.set_ylabel("Predicted EF (%)")
    ax.set_title(f"EF Prediction with ±$q\sigma$ Error Bars ({split_key})")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved EF scatter figure to {save_path}")


def plot_bland_altman(split_key="test",
                      max_points: int = 300,
                      save_path: Path = Path("fig_bland_altman.png")):
    """
    经典 Bland–Altman 图：
      x = (y + μ)/2,  y = μ - y
    画出 diff 的均值和 ±1.96 std 参考线。
    """
    preds = collect_predictions_on_split(split_key)
    y = preds["y"]
    mu = preds["mu"]

    N = len(y)
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        y = y[idx]
        mu = mu[idx]

    mean_val = (y + mu) / 2.0
    diff = mu - y

    bias = diff.mean()
    sd = diff.std()

    fig, ax = plt.subplots(figsize=(5, 4))
    # 绘制散点（使用指定颜色）
    ax.scatter(mean_val, diff, alpha=0.6, color=bland_altman_color, s=30)

    # 参考线（均值和±1.96SD）
    ax.axhline(bias, linestyle="-", linewidth=1.5, 
               color=reference_line_color, alpha=0.8, label=f"Bias = {bias:.2f}")
    ax.axhline(bias + 1.96 * sd, linestyle="--", linewidth=1.2, 
               color=reference_line_color, alpha=0.6, label=f"+1.96SD = {bias+1.96*sd:.2f}")
    ax.axhline(bias - 1.96 * sd, linestyle="--", linewidth=1.2, 
               color=reference_line_color, alpha=0.6, label=f"-1.96SD = {bias-1.96*sd:.2f}")

    ax.set_xlabel("Mean of (GT EF, Pred EF) (%)")
    ax.set_ylabel("Pred EF − GT EF (%)")
    ax.set_title(f"Bland–Altman Plot ({split_key} split)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Bland–Altman figure to {save_path}")


# =========================================================
# main：示例调用
# =========================================================

if __name__ == "__main__":
    # 确保输出目录存在
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # 1) latent 轨迹对比图（默认取 bottleneck 的第 0 维）
    plot_latent_trajectory_comparison(dim=0, ma_window=5, split_key="test")

    # 2) σ_EF vs |error| & 宽度
    plot_sigma_vs_error(num_bins=5, split_key="test")

    # 3) z 的 CDF（用 val split 比较更接近标定过程）
    plot_z_cdf(split_key="val")

    # 4) EF 散点 + 误差条；Bland–Altman
    plot_ef_scatter_with_errorbars(split_key="test")
    plot_bland_altman(split_key="test")