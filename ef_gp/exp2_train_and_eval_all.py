from ef_features_hetero_conformal import (
    train_heteroscedastic_mlp,
    train_heteroscedastic_mlp_nosigma,
    train_heteroscedastic_mlp_nogp,
    train_heteroscedastic_mlp_ma,
    compute_conformal_q,
    compute_conformal_q_nosigma,
    compute_conformal_q_gp_scaled,   # ★ 新增导入
    evaluate_on_split,
    evaluate_on_split_nosigma,
    evaluate_on_split_nogp,
    evaluate_on_split_ma,
    evaluate_on_split_gp_scaled,     # ★ 新增导入
)

import json
from pathlib import Path

RESULTS_PATH = Path("results_exp2_metrics70.json")

all_results = {}

def main():
    # 统一一个 hidden_dims，便于对比
    hidden_dims = (128, 64)

    # 1) Ours: GP μ+σ 特征
    print("===== Training Ours (GP μ+σ) =====")
    train_heteroscedastic_mlp(
        num_epochs=70,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dims=hidden_dims,
        lambda_mse=0.1,
    )
    compute_conformal_q(alpha=0.1)
    res_ours = evaluate_on_split(split_key="test", alpha=0.1)
    all_results["Ours_GP_mu_sigma"] = res_ours

    # 2) U0: GP μ，无 σ 特征
    print("\n===== Training U0 (GP μ, no σ) =====")
    train_heteroscedastic_mlp_nosigma(
        num_epochs=70,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dims=hidden_dims,
    )
    compute_conformal_q_nosigma(alpha=0.1)
    res_u0 = evaluate_on_split_nosigma(split_key="test", alpha=0.1)
    all_results["U0_GP_mu"] = res_u0

    # 3) G0: NoGP
    print("\n===== Training G0 (NoGP, raw u(t)) =====")
    train_heteroscedastic_mlp_nogp(
        num_epochs=70,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dims=hidden_dims,
    )
    res_g0 = evaluate_on_split_nogp(split_key="test")
    all_results["G0_NoGP"] = res_g0

    # 4) F0: MA smoothing
    print("\n===== Training F0 (MA smoothing) =====")
    train_heteroscedastic_mlp_ma(
        num_epochs=70,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dims=hidden_dims,
        window=5,
    )
    res_f0 = evaluate_on_split_ma(split_key="test")
    all_results["F0_MA"] = res_f0

    # 5) GP-scaled: GP-scaled σ-aware Conformal Prediction
    print("\n===== Training GP-scaled (σ-aware conformal with GP difficulty scaling) =====")
    compute_conformal_q_gp_scaled(alpha=0.1)
    res_gp_scaled = evaluate_on_split_gp_scaled(split_key="test", alpha=0.1)  # GP-scaled 评估
    all_results["GP-scaled"] = res_gp_scaled

    # # 6) GP-binned: GP-binned σ-aware Conformal Prediction
    # print("\n===== Training GP-binned (σ-aware conformal with GP difficulty binning) =====")
    # res_gp_binned = evaluate_on_split_gp_binned(split_key="test", alpha=0.1)  # GP-binned 评估
    # all_results["GP-binned"] = res_gp_binned

    # 保存所有结果到 JSON 文件
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            all_results,
            f,
            indent=2,
            default=lambda o: float(o) if hasattr(o, "item") else o,  # 转换为普通 Python float
        )

if __name__ == "__main__":
    main()

