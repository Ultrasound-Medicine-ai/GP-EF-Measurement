import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

def transform_sigma(sigma, clip_q=0.99, power=1.0, eps=1e-6):
    import numpy as np
    s = np.asarray(sigma, dtype=np.float64)
    s = np.maximum(s, eps)

    if clip_q is not None and clip_q < 1.0:
        hi = np.quantile(s, clip_q)
        s = np.minimum(s, hi)

    if power is not None and abs(power - 1.0) > 1e-12:
        # log-space power transform keeps monotonicity
        med = np.median(s)
        s = np.exp(np.log(s) * power)
        # keep median roughly stable (数值更稳；理论上全局scale不影响覆盖率)
        s = s * (med / np.median(s))

    return s.astype(np.float64)

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(max(level, 0.0), 1.0)
    return float(np.quantile(scores, level))


def resolve_gp_paths(gp_root: Path, sid: str):
    mu = gp_root / f"{sid}_mu.npy"
    sig_cal = gp_root / f"{sid}_sigma_cal.npy"
    sig_raw = gp_root / f"{sid}_sigma_raw.npy"
    if not mu.exists():
        return None
    if sig_cal.exists():
        sig = sig_cal
    elif sig_raw.exists():
        sig = sig_raw
    else:
        return None
    return mu, sig


def gp_features(mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    # mu, sig: (T, K)
    T, K = mu.shape

    sig_mean = float(sig.mean())
    sig_std = float(sig.std())
    sig_p90 = float(np.quantile(sig, 0.90))
    sig_max = float(sig.max())
    sig_t = sig.mean(axis=1)               # (T,)
    sig_t_std = float(sig_t.std())
    sig_t_p90 = float(np.quantile(sig_t, 0.90))

    mu_std = float(mu.std())
    mu_t = mu.mean(axis=1)                 # (T,)
    mu_t_std = float(mu_t.std())
    mu_diff = float(np.abs(np.diff(mu, axis=0)).mean())
    mu_range = float((mu.max(axis=0) - mu.min(axis=0)).mean())

    # 一点点结构信息（避免全是均值）
    return np.array([
        sig_mean, sig_std, sig_p90, sig_max, sig_t_std, sig_t_p90,
        mu_std, mu_t_std, mu_diff, mu_range,
        float(T), float(K)
    ], dtype=np.float32)


def load_preds(pred_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    # 兼容列名
    cols = {c.lower(): c for c in df.columns}
    if "filename" not in cols:
        raise ValueError(f"{pred_csv} must contain FileName column")
    if "pred" not in cols and "yhat" not in cols:
        raise ValueError(f"{pred_csv} must contain pred (or yhat) column")
    if "ef" not in cols and "y" not in cols:
        raise ValueError(f"{pred_csv} must contain EF (or y) column")

    fn = cols["filename"]
    pred = cols.get("pred", cols.get("yhat"))
    ef = cols.get("ef", cols.get("y"))

    out = df[[fn, pred, ef]].copy()
    out.columns = ["FileName", "pred", "EF"]
    out["FileName"] = out["FileName"].astype(str)
    out["pred"] = out["pred"].astype(float)
    out["EF"] = out["EF"].astype(float)
    return out


def build_matrix(pred_df: pd.DataFrame, gp_root: Path):
    X_list, y_list, pred_list, sid_list = [], [], [], []
    missing = 0
    for r in pred_df.itertuples(index=False):
        sid = str(r.FileName)
        paths = resolve_gp_paths(gp_root, sid)
        if paths is None:
            missing += 1
            continue
        mu = np.load(paths[0]).astype(np.float32)
        sig = np.load(paths[1]).astype(np.float32)
        if mu.ndim != 2 or sig.ndim != 2 or mu.shape != sig.shape:
            missing += 1
            continue
        feat = gp_features(mu, sig)
        X_list.append(feat)
        y_list.append(float(r.EF))
        pred_list.append(float(r.pred))
        sid_list.append(sid)

    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, 12), dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    pred = np.asarray(pred_list, dtype=np.float32)
    if missing:
        print(f"[Warn] missing GP files for {missing}/{len(pred_df)} samples. Dropping those.")
    return X, y, pred, sid_list


def threshold_metrics(y, pred, L, U, t: float):
    y = np.asarray(y)
    pred = np.asarray(pred)
    L = np.asarray(L)
    U = np.asarray(U)

    y_cls = (y < t).astype(int)
    pred_cls = (pred < t).astype(int)

    actionable = (U < t) | (L > t)      # 区间完全在阈值一侧
    abstain = ~actionable

    # actionable 子集分类表现
    idx = np.where(actionable)[0]
    if idx.size == 0:
        return {"actionable": 0.0, "abstain": 1.0}

    y_a = y_cls[idx]
    p_a = pred_cls[idx]
    acc = float((y_a == p_a).mean())

    # Sens/Spec for "positive = EF < t"
    tp = float(((y_a == 1) & (p_a == 1)).sum())
    fn = float(((y_a == 1) & (p_a == 0)).sum())
    tn = float(((y_a == 0) & (p_a == 0)).sum())
    fp = float(((y_a == 0) & (p_a == 1)).sum())
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)

    # “点预测犯错”被区间拦截（abstain）的比例
    mistakes = (pred_cls != y_cls)
    flag_rate = float((abstain & mistakes).sum() / (mistakes.sum() + 1e-12))

    near = float((np.abs(y - t) <= 5).mean())

    return {
        "actionable": float(actionable.mean()),
        "abstain": float(abstain.mean()),
        "acc": acc,
        "sens": float(sens),
        "spec": float(spec),
        "flag_rate": flag_rate,
        "near": near,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gp_root", required=True, type=str)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--sigma_clip_q", type=float, default=0.99)
    ap.add_argument("--sigma_power", type=float, default=1.0)

    ap.add_argument("--pred_train", required=True, type=str)
    ap.add_argument("--pred_val", required=True, type=str)
    ap.add_argument("--pred_test", required=True, type=str)

    ap.add_argument("--out_prefix", type=str, default="echonet_gp_sigma_error")
    args = ap.parse_args()

    gp_root = Path(args.gp_root)

    df_tr = load_preds(Path(args.pred_train))
    df_va = load_preds(Path(args.pred_val))
    df_te = load_preds(Path(args.pred_test))

    Xtr, ytr, ptr, _ = build_matrix(df_tr, gp_root)
    Xva, yva, pva, _ = build_matrix(df_va, gp_root)
    Xte, yte, pte, sid_te = build_matrix(df_te, gp_root)

    # target: log1p(|err|)
    err_tr = np.abs(ytr - ptr)
    err_va = np.abs(yva - pva)
    err_te = np.abs(yte - pte)

    ytr_t = np.log1p(err_tr)
    yva_t = np.log1p(err_va)

    model = XGBRegressor(
        n_estimators=4000,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=8,
    )

    model.fit(
        Xtr, ytr_t,
        eval_set=[(Xva, yva_t)],
        verbose=False,
    )

    # sigma_hat = exp(pred)-1, >= eps
    def sigma_hat(X):
        s = np.expm1(model.predict(X))
        return np.maximum(s, 1e-3)

    s_va = sigma_hat(Xva)
    s_te = sigma_hat(Xte)
    s_va = transform_sigma(s_va, clip_q=args.sigma_clip_q, power=args.sigma_power)
    s_te = transform_sigma(s_te, clip_q=args.sigma_clip_q, power=args.sigma_power)

    # Baseline residual conformal (no sigma)
    q_abs = conformal_quantile(err_va, alpha=args.alpha)
    L0 = pte - q_abs
    U0 = pte + q_abs
    cov0 = float(((yte >= L0) & (yte <= U0)).mean())
    w0 = float((U0 - L0).mean())

    # GP sigma-aware conformal
    scores = err_va / (s_va + 1e-12)
    q_norm = conformal_quantile(scores, alpha=args.alpha)
    L = pte - q_norm * s_te
    U = pte + q_norm * s_te
    cov = float(((yte >= L) & (yte <= U)).mean())
    w = float((U - L).mean())

    rmse = float(np.sqrt(np.mean((pte - yte) ** 2)))
    mae = float(np.mean(np.abs(pte - yte)))

    print("\n=== EchoNet point ===")
    print(f"TEST RMSE={rmse:.4f}  MAE={mae:.4f}")

    print("\n=== Baseline: Residual Conformal (no GP) ===")
    print(f"alpha={args.alpha}  q_abs={q_abs:.4f}")
    print(f"TEST coverage={cov0:.4f}  mean_width={w0:.4f}")

    print("\n=== GP-informed σ-aware Conformal (GP->error scale) ===")
    print(f"alpha={args.alpha}  q_norm={q_norm:.4f}")
    print(f"TEST coverage={cov:.4f}  mean_width={w:.4f}")

    # conditional widths
    width = (U - L)
    q25, q75 = np.quantile(s_te, [0.25, 0.75])
    w_low = float(width[s_te <= q25].mean())
    w_high = float(width[s_te >= q75].mean())
    print(f"Width by σ_hat quartiles: low25%={w_low:.3f}  high25%={w_high:.3f}")

    for t in [40.0, 50.0]:
        m0 = threshold_metrics(yte, pte, L0, U0, t)
        m1 = threshold_metrics(yte, pte, L, U, t)
        print(f"\n[Threshold t={t:.0f}]  (Residual conformal)")
        print(f"Actionable={m0['actionable']:.3f} Abstain={m0['abstain']:.3f} Near(|EF-{t:.0f}|<=5)={m0['near']:.3f}")
        print(f"Actionable subset: Acc={m0['acc']:.3f} Sens={m0['sens']:.3f} Spec={m0['spec']:.3f}")
        print(f"Point mistakes flagged by abstain: {m0['flag_rate']:.3f}")

        print(f"[Threshold t={t:.0f}]  (GP σ-aware)")
        print(f"Actionable={m1['actionable']:.3f} Abstain={m1['abstain']:.3f} Near(|EF-{t:.0f}|<=5)={m1['near']:.3f}")
        print(f"Actionable subset: Acc={m1['acc']:.3f} Sens={m1['sens']:.3f} Spec={m1['spec']:.3f}")
        print(f"Point mistakes flagged by abstain: {m1['flag_rate']:.3f}")

    out = pd.DataFrame({
        "FileName": sid_te,
        "EF": yte,
        "pred": pte,
        "L_residual": L0,
        "U_residual": U0,
        "sigma_hat": s_te,
        "L_gp": L,
        "U_gp": U,
    })
    out_path = f"{args.out_prefix}_test_intervals.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
