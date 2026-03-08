import os, math, argparse
from pathlib import Path
import numpy as np
import pandas as pd

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(max(level, 0.0), 1.0)
    return float(np.quantile(scores, level))

def find_id_col(df: pd.DataFrame):
    for c in ["FileName", "Filename", "ID", "id"]:
        if c in df.columns:
            return c
    # fallback: first column
    return df.columns[0]

def load_sigma_proxy(gp_root: Path, sid: str, prefer="cal"):
    # prefer calibrated sigma if exists
    p_cal = gp_root / f"{sid}_sigma_cal.npy"
    p_raw = gp_root / f"{sid}_sigma_raw.npy"
    if prefer == "cal" and p_cal.exists():
        sig = np.load(str(p_cal))
    elif p_raw.exists():
        sig = np.load(str(p_raw))
    elif p_cal.exists():
        sig = np.load(str(p_cal))
    else:
        return None

    sig = np.asarray(sig, dtype=np.float64)
    # sig shape (T,K) — aggregate to 1 scalar proxy per video
    # robust choice: mean over time and dims
    return float(np.mean(sig))

def fit_monotone_bins(x: np.ndarray, y: np.ndarray, n_bins: int = 60):
    """
    Lightweight monotone calibration without sklearn:
    - bin x into quantile bins
    - compute mean y per bin
    - enforce monotone nondecreasing via cumulative max
    - return piecewise-linear interpolator over bin centers
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x_s = x[order]; y_s = y[order]

    # quantile bin edges
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x_s, qs)
    # make edges strictly increasing
    edges = np.unique(edges)
    if edges.size < 3:
        # fallback: constant
        c = float(np.mean(y_s))
        def f(z): 
            return np.full_like(np.asarray(z, dtype=np.float64), c)
        return f

    # assign bins
    # bins: [edges[i], edges[i+1])
    bin_idx = np.clip(np.searchsorted(edges, x_s, side="right") - 1, 0, edges.size - 2)

    bin_means = []
    bin_centers = []
    for b in range(edges.size - 1):
        m = y_s[bin_idx == b]
        if m.size == 0:
            continue
        bin_means.append(float(np.mean(m)))
        bin_centers.append(float(0.5 * (edges[b] + edges[b+1])))

    bin_centers = np.array(bin_centers, dtype=np.float64)
    bin_means = np.array(bin_means, dtype=np.float64)

    # enforce monotone increasing
    bin_means = np.maximum.accumulate(bin_means)

    def f(z):
        z = np.asarray(z, dtype=np.float64)
        # extrapolate flat outside range
        out = np.interp(z, bin_centers, bin_means, left=bin_means[0], right=bin_means[-1])
        return out

    return f

def interval_and_metrics(mu, y, L, U):
    mu = np.asarray(mu, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)

    rmse = float(np.sqrt(np.mean((mu-y)**2)))
    mae = float(np.mean(np.abs(mu-y)))
    cov = float(np.mean((y>=L) & (y<=U)))
    width = float(np.mean(U-L))
    return rmse, mae, cov, width

def threshold_report(y, mu, L, U, t):
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)

    yb = (y >= t).astype(int)
    pb = (mu >= t).astype(int)

    action = (U < t) | (L >= t)
    pred_action = np.where(L >= t, 1, 0)

    ar = float(action.mean())
    abst = 1.0 - ar
    idx = np.where(action)[0]
    if idx.size > 0:
        yt = yb[idx]; pt = pred_action[idx]
        acc = float((yt==pt).mean())
        tp = int(((yt==1)&(pt==1)).sum()); tn = int(((yt==0)&(pt==0)).sum())
        fp = int(((yt==0)&(pt==1)).sum()); fn = int(((yt==1)&(pt==0)).sum())
        sens = tp/(tp+fn) if (tp+fn)>0 else float("nan")
        spec = tn/(tn+fp) if (tn+fp)>0 else float("nan")
    else:
        acc=sens=spec=float("nan")

    point_wrong = (pb != yb)
    caught = point_wrong & (~action)
    catch_rate = float(caught.sum() / max(point_wrong.sum(), 1))

    near = float(np.mean(np.abs(y - t) <= 5))
    return ar, abst, acc, sens, spec, catch_rate, near

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist_csv", required=True)
    ap.add_argument("--gp_root", required=True)
    ap.add_argument("--pred_train", required=True)
    ap.add_argument("--pred_val", required=True)
    ap.add_argument("--pred_test", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--prefer_sigma", type=str, default="cal", choices=["cal","raw"])
    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()

    gp_root = Path(args.gp_root)

    tr = pd.read_csv(args.pred_train)
    va = pd.read_csv(args.pred_val)
    te = pd.read_csv(args.pred_test)

    idc_tr = find_id_col(tr); idc_va = find_id_col(va); idc_te = find_id_col(te)

    # load sigma proxies
    def attach_sigma(df, idc):
        sig = []
        keep = []
        for sid in df[idc].astype(str).tolist():
            s = load_sigma_proxy(gp_root, sid, prefer=args.prefer_sigma)
            sig.append(s)
            keep.append(s is not None)
        df = df.copy()
        df["sigma_proxy"] = sig
        return df

    tr = attach_sigma(tr, idc_tr)
    va = attach_sigma(va, idc_va)
    te = attach_sigma(te, idc_te)

    # drop missing sigma samples (report)
    def drop_missing(df, name):
        m = df["sigma_proxy"].isna().sum()
        if m > 0:
            print(f"[Warn] {name}: missing sigma for {m}/{len(df)} samples. Dropping those.")
        return df.dropna(subset=["sigma_proxy"]).copy()

    tr = drop_missing(tr, "TRAIN")
    va = drop_missing(va, "VAL")
    te = drop_missing(te, "TEST")

    # baseline residual conformal (constant width), calibrated on VAL
    y_val = va["EF"].astype(float).values
    mu_val = va["pred"].astype(float).values
    res_val = np.abs(y_val - mu_val)
    q_res = conformal_quantile(res_val, args.alpha)

    y_test = te["EF"].astype(float).values
    mu_test = te["pred"].astype(float).values
    L_res = mu_test - q_res
    U_res = mu_test + q_res
    rmse, mae, cov, width = interval_and_metrics(mu_test, y_test, L_res, U_res)
    print("\n=== Baseline: Residual Conformal (no GP) ===")
    print(f"alpha={args.alpha}  q={q_res:.4f}")
    print(f"TEST RMSE={rmse:.4f}  MAE={mae:.4f}  coverage={cov:.4f}  mean_width={width:.4f}")

    # σ-aware: learn monotone mapping sigma_proxy -> sigma_hat using TRAIN residuals
    y_tr = tr["EF"].astype(float).values
    mu_tr = tr["pred"].astype(float).values
    abs_res_tr = np.abs(y_tr - mu_tr)
    sig_tr = tr["sigma_proxy"].astype(float).values

    f = fit_monotone_bins(sig_tr, abs_res_tr, n_bins=args.bins)

    # apply mapping
    va["sigma_hat"] = f(va["sigma_proxy"].astype(float).values) + 1e-6
    te["sigma_hat"] = f(te["sigma_proxy"].astype(float).values) + 1e-6

    # normalized conformal calibration on VAL
    sig_val = va["sigma_hat"].astype(float).values
    score_val = np.abs(y_val - mu_val) / sig_val
    q_norm = conformal_quantile(score_val, args.alpha)

    sig_test = te["sigma_hat"].astype(float).values
    L_gp = mu_test - q_norm * sig_test
    U_gp = mu_test + q_norm * sig_test
    rmse2, mae2, cov2, width2 = interval_and_metrics(mu_test, y_test, L_gp, U_gp)

    print("\n=== GP σ-aware Conformal (uses GP sigma) ===")
    print(f"alpha={args.alpha}  q_norm={q_norm:.4f}")
    print(f"TEST RMSE={rmse2:.4f}  MAE={mae2:.4f}  coverage={cov2:.4f}  mean_width={width2:.4f}")

    # conditional width by sigma quartiles (to show adaptivity)
    q25, q75 = np.quantile(sig_test, [0.25, 0.75])
    low = sig_test <= q25
    high = sig_test >= q75
    if low.any() and high.any():
        w_low = float(np.mean((U_gp - L_gp)[low]))
        w_high = float(np.mean((U_gp - L_gp)[high]))
        print(f"Width by σ quartiles: low25%={w_low:.3f}  high25%={w_high:.3f}")

    # threshold decision reports
    for t in [40, 50]:
        ar, abst, acc, sens, spec, catch, near = threshold_report(y_test, mu_test, L_gp, U_gp, t)
        print(f"\n[GP σ-aware Threshold t={t}]")
        print(f"Actionable={ar:.3f}  Abstain={abst:.3f}  Near(|EF-{t}|<=5)={near:.3f}")
        print(f"Actionable subset: Acc={acc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}")
        print(f"Point mistakes flagged by abstain: {catch:.3f}")

    # save intervals
    out = te.copy()
    out["L_residual"] = L_res
    out["U_residual"] = U_res
    out["L_gp_sigma"] = L_gp
    out["U_gp_sigma"] = U_gp
    out.to_csv("echonet_test_intervals_residual_vs_gp_sigma.csv", index=False)
    print("\n[Saved] echonet_test_intervals_residual_vs_gp_sigma.csv")

if __name__ == "__main__":
    main()