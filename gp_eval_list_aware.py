#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation of GP uncertainty quality with optional --list_file filtering.
- Supports CRPS, coverage @ {0.68, 0.90, 0.95}, reliability curve.
- If --list_file is provided, only evaluates those videos (IDs, no extension).
- Robust meta parsing and silent handling of missing videos.
"""

import os, json, argparse, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try importing scipy, but be tolerant to binary warnings
try:
    from scipy.stats import norm
except Exception:
    norm = None

# ---- Gaussian helpers ----
if norm is not None:
    def phi(z):
        return norm.pdf(z)
    def Phi(z):
        return norm.cdf(z)
else:
    # Fallback approximations if SciPy is unavailable
    inv_sqrt_2pi = 1.0/np.sqrt(2*np.pi)
    def phi(z):
        return inv_sqrt_2pi * np.exp(-0.5*z*z)
    # Abramowitz-Stegun erf-based CDF approx
    def Phi(z):
        return 0.5*(1.0+np.erf(z/np.sqrt(2.0)))


def crps_gaussian(y, mu, sigma, eps=1e-12):
    """
    y, mu, sigma: same shape (T, D)
    CRPS = σ * [ z*(2Φ(z)-1) + 2φ(z) - 1/√π ],  z=(y-μ)/σ
    """
    sigma = np.maximum(sigma, eps)
    z = (y - mu) / sigma
    return sigma * ( z * (2.0 * Phi(z) - 1.0) + 2.0 * phi(z) - 1.0 / np.sqrt(np.pi) )


def coverage_rate(y, mu, sigma, alpha):
    """Empirical coverage: |y-μ| <= z*σ. Uses common two-sided z."""
    z_map = {0.68: 1.0, 0.80: 1.282, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(alpha, 1.96)
    ok = np.abs(y - mu) <= z * np.maximum(sigma, 1e-12)
    return ok.mean()


def parse_meta(arr) -> dict:
    if isinstance(arr, dict):
        return arr
    try:
        x = arr.item() if hasattr(arr, 'item') else arr
        if isinstance(x, (bytes, bytearray)):
            return json.loads(x.decode('utf-8', errors='ignore'))
        if isinstance(x, str):
            return json.loads(x)
        if isinstance(x, dict):
            return x
    except Exception:
        pass
    # Last resort
    try:
        return dict(arr.tolist())
    except Exception:
        return {}


def load_gp_npz(npz_path: Path):
    d = np.load(str(npz_path), allow_pickle=True)
    keys = set(d.files)
    def pick(*cands):
        for k in cands:
            if k in keys:
                return d[k]
        raise KeyError(f"No candidate keys {cands} in {npz_path}, has {keys}")
    L_smooth = pick('latent_smooth','latents_smooth','L_smooth','Z_smooth','Z_hat')
    L_std    = pick('latent_std','std','latent_sigma','Z_std')
    meta = parse_meta(d['meta']) if 'meta' in keys else {}
    return L_smooth, L_std, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latents_dir_raw', required=True, type=Path,
                    help='Raw latent directory (T×D .npy per video)')
    ap.add_argument('--latents_dir_gp', required=True, type=Path,
                    help='GP outputs directory (.npz per video)')
    ap.add_argument('--out_dir', required=True, type=Path,
                    help='Output directory for figures & summary.json')
    ap.add_argument('--example_video', type=str, default=None,
                    help='ID (no extension) to visualize in detail')
    ap.add_argument('--list_file', type=Path, default=None,
                    help='Optional list of video IDs (one per line) to evaluate')
    ap.add_argument('--alphas', type=float, nargs='*', default=[0.68, 0.90, 0.95],
                    help='Coverage levels to report')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.list_file is not None and args.list_file.exists():
        vids = [v.strip() for v in open(args.list_file) if v.strip()]
        gp_files = [args.latents_dir_gp / f"{v}.npz" for v in vids]
    else:
        gp_files = [Path(p) for p in sorted(glob.glob(str(args.latents_dir_gp / "*.npz")))]

    print(f"Found GP npz files: {len(gp_files)}")

    alphas = list(args.alphas)
    all_rows = []
    cov_points = {a: [] for a in alphas}
    crps_all = []

    for npz_path in gp_files:
        if not Path(npz_path).exists():
            continue
        vid = Path(npz_path).stem
        raw_path = args.latents_dir_raw / f"{vid}.npy"
        if not raw_path.exists():
            print(f"[WARN] raw latent not found for {vid}: {raw_path}")
            continue

        y = np.load(raw_path)                 # (T,D)
        mu, sigma, meta = load_gp_npz(npz_path)  # (T,D), (T,D)
        T = min(len(y), len(mu), len(sigma))
        if T <= 0:
            continue
        y, mu, sigma = y[:T], mu[:T], sigma[:T]

        mean_std = float(np.mean(sigma))
        p95_std  = float(np.percentile(sigma, 95))
        has_nan  = bool(np.isnan(mu).any() or np.isnan(sigma).any())
        K        = (meta.get('n_components') if isinstance(meta, dict) else None)

        cov_row = {}
        for a in alphas:
            cov = coverage_rate(y, mu, sigma, a)
            cov_row[a] = float(cov)
            cov_points[a].append(cov)

        crps = float(np.mean(crps_gaussian(y, mu, sigma)))
        crps_all.append(crps)

        row = {
            'video': vid, 'T': int(T), 'D': int(y.shape[1]),
            'K': int(K) if K is not None else None,
            'mean_std': mean_std, 'p95_std': p95_std,
            'coverage': {str(a): cov_row[a] for a in alphas},
            'crps_mean': crps,
            'has_nan': has_nan
        }
        all_rows.append(row)

    summary = {
        'N_videos': len(all_rows),
        'overall': {
            'mean_std': float(np.mean([r['mean_std'] for r in all_rows])) if all_rows else None,
            'p95_std':  float(np.mean([r['p95_std']  for r in all_rows])) if all_rows else None,
            'crps_mean': float(np.mean(crps_all)) if crps_all else None,
            'coverage_mean': {str(a): float(np.mean(cov_points[a])) if len(cov_points[a])>0 else None
                              for a in alphas}
        },
        'per_video': all_rows
    }

    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {args.out_dir / 'summary.json'}")

    # Reliability curve
    xs = np.array(alphas, dtype=float)
    ys = np.array([summary['overall']['coverage_mean'][str(a)] for a in alphas], dtype=float)
    plt.figure(figsize=(4,4))
    plt.plot(xs, xs, '--', label='ideal')
    plt.plot(xs, ys, 'o-', label='empirical')
    plt.xlabel('Nominal coverage')
    plt.ylabel('Empirical coverage')
    plt.title('Reliability curve (latent Gaussian)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "reliability_curve.png", dpi=180)

    # Example visualization
    example = args.example_video or (all_rows[0]['video'] if all_rows else None)
    if example is not None:
        npz_path = args.latents_dir_gp / f"{example}.npz"
        raw_path = args.latents_dir_raw / f"{example}.npy"
        if npz_path.exists() and raw_path.exists():
            y = np.load(raw_path)
            mu, sigma, meta = load_gp_npz(npz_path)
            T = min(len(y), len(mu), len(sigma))
            y, mu, sigma = y[:T], mu[:T], sigma[:T]
            # heatmap
            plt.figure(figsize=(8,3))
            im = plt.imshow(sigma, aspect='auto', origin='lower')
            plt.colorbar(im, fraction=0.046, pad=0.04, label='std')
            plt.xlabel('latent dim'); plt.ylabel('time')
            plt.title(f'Uncertainty heatmap (std), {example}')
            plt.tight_layout(); plt.savefig(args.out_dir / f"{example}_std_heatmap.png", dpi=180)
            # bands
            dims = [0,1,2] if y.shape[1] >= 3 else list(range(y.shape[1]))
            t = np.arange(T)
            for d in dims:
                plt.figure(figsize=(8,3))
                plt.plot(t, y[:, d], lw=1, label='raw')
                plt.plot(t, mu[:, d], lw=1.5, label='GP-mean')
                up = mu[:, d] + 2*sigma[:, d]
                lo = mu[:, d] - 2*sigma[:, d]
                plt.fill_between(t, lo, up, alpha=0.2, label='±2σ')
                plt.xlabel('time'); plt.ylabel(f'latent[{d}]')
                plt.title(f'{example} latent[{d}] GP smoothing ±2σ')
                plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
                plt.savefig(args.out_dir / f"{example}_dim{d}_gp_band.png", dpi=180)


if __name__ == "__main__":
    main()
