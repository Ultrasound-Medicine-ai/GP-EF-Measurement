import pandas as pd
import matplotlib.pyplot as plt

csv_path = "./checkpoints/kappa_sweep_alpha0.10.csv"
alpha = 0.1
k_star = 0.0  # 你想标注的 κ*，不想标注就设为 None

df = pd.read_csv(csv_path)
kappa = df["kappa"].values
cov = df["test_cov"].values
width = df["test_width"].values

# IEEE-like typography
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
})

# low-saturation colors
c_cov = "#4C78A8"   # muted blue
c_w   = "#9AA0A6"   # muted gray

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(3.5, 2.6), dpi=300, sharex=True,
    gridspec_kw={"hspace": 0.12}
)

# --- Top: Coverage ---
ax1.plot(kappa, cov, marker="o", markersize=3.0, color=c_cov, label="Coverage (test)")
ax1.axhline(1 - alpha, linestyle=":", linewidth=1.0, color="black", alpha=0.8, label="Nominal (0.90)")
ax1.set_ylabel("Coverage")
ax1.set_ylim(0.89, 0.92)  # 刻意不放大波动
ax1.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
ax1.legend(loc="lower left", frameon=False)

# Optional: mark kappa*
if k_star is not None and k_star in list(kappa):
    idx = list(kappa).index(k_star)
    ax1.scatter([k_star], [cov[idx]], marker="*", s=45, color="black", zorder=5)
    ax1.text(k_star, cov[idx] + 0.0005, r"$\kappa^{*}$", fontsize=7, ha="center")

# --- Bottom: Width ---
ax2.plot(kappa, width, marker="s", markersize=2.8, linestyle="--", color=c_w, label="Mean width (test)")
ax2.set_ylabel("Width (%EF)")
ax2.set_xlabel(r"$\kappa$")
ax2.set_ylim(width.min() - 0.3, width.max() + 0.3)
ax2.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
ax2.legend(loc="upper left", frameon=False)

fig.tight_layout()
fig.savefig("appendix_kappa_sensitivity_paper.pdf", bbox_inches="tight")
fig.savefig("appendix_kappa_sensitivity_paper.png", bbox_inches="tight")
print("Saved: appendix_kappa_sensitivity_paper.pdf/.png")
