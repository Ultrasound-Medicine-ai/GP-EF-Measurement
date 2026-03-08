# exp3_uncertainty_quality.py
from ef_features_hetero_conformal import (
    compute_conformal_q,
    compute_conformal_q_nosigma,
    evaluate_on_split,
    evaluate_on_split_nosigma,
    analyze_uncertainty_on_split,
    analyze_uncertainty_on_split_nosigma,
) 


def main():
    # 假设你已经按照 Exp2 训练好了模型
    # 如果不确定，可以在这里再跑一遍 compute_conformal_q 系列

    # print("===== Ours: coverage & width & RMSE/MAE on TEST =====")
    # compute_conformal_q(alpha=0.1)          # 如果已经有 conformal_q.npy，会覆盖
    # evaluate_on_split(split_key="test", alpha=0.1)

    # print("\n===== U0: coverage & width & RMSE/MAE on TEST =====")
    # compute_conformal_q_nosigma(alpha=0.1)  # 同理
    # evaluate_on_split_nosigma(split_key="test", alpha=0.1)

    print("\n===== Ours: uncertainty analysis (σ_EF vs |error|) =====")
    analyze_uncertainty_on_split(split_key="test", num_bins=5)

    print("\n===== U0: uncertainty analysis (σ_EF vs |error|) =====")
    analyze_uncertainty_on_split_nosigma(split_key="test", num_bins=5)


if __name__ == "__main__":
    main()
