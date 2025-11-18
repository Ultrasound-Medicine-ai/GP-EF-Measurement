#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, glob, json
from pathlib import Path
import numpy as np

# ---------- metrics ----------
def tv1(L: np.ndarray) -> float:  # 一阶总变分（越小越平滑）
    D1 = np.diff(L, axis=0)
    return float(np.mean(np.abs(D1)))

def tv2(L: np.ndarray) -> float:  # 二阶总变分（越小越平滑/抖动少）
    D1 = np.diff(L, axis=0)
    D2 = np.diff(D1, axis=0)
    return float(np.mean(np.abs(D2)))

def hf_ratio(L: np.ndarray, cutoff: float = 0.25) -> float:
    """
    高频能量占比（越小越平滑）。
    逐维 rFFT 后统计 > cutoff*Nyquist 的频带能量占比。
    cutoff=0.25 表示统计最高 75% 的频带能量比例。
    """
    T, D = L.shape
    X = L - L.mean(axis=0, keepdims=True)
    F = np.fft.rfft(X, axis=0)
    P = np.abs(F)**2
    n = P.shape[0]
    k0 = int(np.floor(cutoff * (n-1)))
    hi = P[k0:, :].sum()
    tot = P.sum() + 1e-12
    return float(hi / tot)

# ---------- IO helpers ----------
def load_npz_latent_mu(npz_path: Path) -> np.ndarray:
    d = np.load(str(npz_path), allow_pickle=True)
    for key in ("latent_smooth","latents_smooth","L_smooth","Z_smooth","Z_hat","mu"):
        if key in d.files:
            return d[key]
    raise KeyError(f"No latent mean key in {npz_path}, has {d.files}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents_dir_raw", required=True, type=Path,
                    help="原始潜变量目录（每视频 T×D 的 .npy）")
    ap.add_argument("--latents_dir_gp", required=True, type=Path,
                    help="GP 结果目录（每视频 .npz，含 latent_smooth；可为未校准或校准后目录）")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="输出目录，保存 summary_temporal.json")
    ap.add_argument("--list_file", type=Path, default=None,
                    help="可选：仅评估该清单中的视频（每行一个 ID，无扩展名）")
    ap.add_argument("--hf_cutoff", type=float, default=0.25,
                    help="HF 频带阈值，统计 >cutoff*Nyquist 的能量占比，默认 0.25")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 目标评估集合
    if args.list_file is not None and args.list_file.exists():
        vids = [v.strip() for v in open(args.list_file) if v.strip()]
        npz_files = [args.latents_dir_gp / f"{v}.npz" for v in vids]
    else:
        npz_files = [Path(p) for p in sorted(glob.glob(str(args.latents_dir_gp / "*.npz")))]

    rows = []
    red_tv1, red_tv2, red_hf = [], [], []

    for p in npz_files:
        if not Path(p).exists():
            continue
        vid = Path(p).stem
        raw_path = args.latents_dir_raw / f"{vid}.npy"
        if not raw_path.exists():
            print(f"[WARN] raw missing: {raw_path}")
            continue

        L_raw = np.load(str(raw_path))
        L_gp  = load_npz_latent_mu(Path(p))
        T = min(L_raw.shape[0], L_gp.shape[0])
        if T <= 2:
            print(f"[WARN] too short: {vid}")
            continue
        L_raw, L_gp = L_raw[:T], L_gp[:T]

        tv1_raw, tv1_gp = tv1(L_raw), tv1(L_gp)
        tv2_raw, tv2_gp = tv2(L_raw), tv2(L_gp)
        hf_raw,  hf_gp  = hf_ratio(L_raw, args.hf_cutoff), hf_ratio(L_gp, args.hf_cutoff)

        def pct_drop(a, b):
            return float(100.0 * (a - b) / (a + 1e-12))

        m = {
            "video": vid, "T": int(T), "D": int(L_raw.shape[1]),
            "tv1_raw": tv1_raw, "tv1_gp": tv1_gp, "tv1_drop_%": pct_drop(tv1_raw, tv1_gp),
            "tv2_raw": tv2_raw, "tv2_gp": tv2_gp, "tv2_drop_%": pct_drop(tv2_raw, tv2_gp),
            "hf_raw":  hf_raw,  "hf_gp":  hf_gp,  "hf_drop_%":  pct_drop(hf_raw,  hf_gp),
        }
        rows.append(m)
        red_tv1.append(m["tv1_drop_%"])
        red_tv2.append(m["tv2_drop_%"])
        red_hf.append(m["hf_drop_%"])

        print({k: m[k] for k in ["video","tv1_raw","tv1_gp","tv1_drop_%","tv2_raw","tv2_gp","tv2_drop_%","hf_raw","hf_gp","hf_drop_%"]})

    summary = {
        "N_videos": len(rows),
        "overall_mean_drop_%": {
            "tv1": float(np.mean(red_tv1)) if red_tv1 else None,
            "tv2": float(np.mean(red_tv2)) if red_tv2 else None,
            "hf":  float(np.mean(red_hf))  if red_hf  else None,
        },
        "per_video": rows
    }

    with (args.out_dir / "summary_temporal.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", args.out_dir / "summary_temporal.json")


if __name__ == "__main__":
    main()
