# exp4_gp_calibration_analysis.py
import numpy as np

from fit_gp_and_calibrate import calibrate_sigma_global


def main():
    # 只用 TRAIN + VAL 做分析（和实际标定一致）
    target = 0.9
    s_star, all_r = calibrate_sigma_global(
        splits_for_calib=("TRAIN", "VAL"),
        target_coverage=target,
        return_r_values=True,
    )

    all_r = np.asarray(all_r, dtype=np.float32)

    # 标定后 r_cal = r / s_star
    r_cal = all_r / s_star

    # 覆盖率：|r| <= 1
    cov_before = np.mean(all_r <= 1.0)
    cov_after = np.mean(r_cal <= 1.0)

    # 一些分位数看看形状
    qs = [0.5, 0.68, 0.9, 0.95, 0.99]
    quant_before = np.quantile(all_r, qs)
    quant_after = np.quantile(r_cal, qs)

    print(f"=== GP sigma global calibration analysis ===")
    print(f"target_coverage = {target}, s* = {s_star:.4f}")
    print(f"Coverage before calib  (|r|<=1): {cov_before:.4f}")
    print(f"Coverage after calib   (|r_cal|<=1): {cov_after:.4f}\n")

    print("Quantiles of r BEFORE calib:")
    for q, v in zip(qs, quant_before):
        print(f"  q={q:.2f}: {v:.4f}")

    print("\nQuantiles of r AFTER calib (r/s*):")
    for q, v in zip(qs, quant_after):
        print(f"  q={q:.2f}: {v:.4f}")

    print("\n(理想情况下，after calib 时 q≈0.90 对应的分位数应该接近 1.0)")

if __name__ == "__main__":
    main()
