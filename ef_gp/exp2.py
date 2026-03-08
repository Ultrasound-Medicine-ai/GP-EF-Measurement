from ef_features_hetero_conformal import (
    # 训练函数（如果只想评估，下面这几个可以不调用，但保留 import 也没问题）
    train_heteroscedastic_mlp,
    train_heteroscedastic_mlp_nosigma,
    train_heteroscedastic_mlp_nogp,
    train_heteroscedastic_mlp_ma,

    # 基本 conformal（U1 / U0）
    compute_conformal_q,
    compute_conformal_q_nosigma,

    # ★★★ 新增：GP-scaled / GP-binned 的 q 计算函数
    compute_conformal_q_gp_scaled,
    compute_conformal_q_gp_binned,

    # 各种评估函数
    evaluate_on_split,
    evaluate_on_split_nosigma,
    evaluate_on_split_nogp,
    evaluate_on_split_ma,
    evaluate_on_split_gp_scaled,   # GP-scaled
    evaluate_on_split_gp_binned,   # GP-binned
)

import json
from pathlib import Path

RESULTS_PATH = Path("results_exp2_metrics4.json")

all_results = {}

def main():
    # 这里 hidden_dims 目前没用到，如果你想在同一个脚本里重新训练模型，可以传进去
    hidden_dims = (128, 64)
    # # 1) Ours: GP μ+σ 特征
    # print("===== Training Ours (GP μ+σ) =====")
    # train_heteroscedastic_mlp(
    #     num_epochs=70,
    #     batch_size=64,
    #     lr=1e-3,
    #     weight_decay=1e-4,
    #     hidden_dims=hidden_dims,
    #     lambda_mse=0.1,
    # )
    # compute_conformal_q(alpha=0.1)
    # res_ours = evaluate_on_split(split_key="test", alpha=0.1)
    # all_results["Ours_GP_mu_sigma"] = res_ours
    
    # ===== 5) GP-scaled: GP-scaled σ-aware Conformal Prediction =====
    print("\n===== GP-scaled: σ-aware conformal with GP difficulty scaling =====")
    # 1) 先在 VAL split 上标定 q、beta（会写入 CHECKPOINT_DIR / conformal_q_gp_scaled.npy）
    compute_conformal_q_gp_scaled(alpha=0.1)
    # 2) 再在 TEST split 上评估
    res_gp_scaled = evaluate_on_split_gp_scaled(split_key="test", alpha=0.1)
    all_results["GP-scaled"] = res_gp_scaled

    # ===== 6) GP-binned: GP-binned σ-aware Conformal Prediction =====
    print("\n===== GP-binned: σ-aware conformal with GP difficulty binning =====")
    # 1) 同样先在 VAL 上算 bin-wise 的 q_b
    compute_conformal_q_gp_binned(alpha=0.1)   # 注意 alpha=0.1 或你想要的值
    # 2) 然后在 TEST 上评估
    res_gp_binned = evaluate_on_split_gp_binned(split_key="test", alpha=0.1)
    all_results["GP-binned"] = res_gp_binned

    # ===== 保存所有结果到 JSON =====
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            all_results,
            f,
            indent=2,
            default=lambda o: float(o) if hasattr(o, "item") else o,
        )
    print(f"\nSaved metrics to {RESULTS_PATH}")

if __name__ == "__main__":
    main()

