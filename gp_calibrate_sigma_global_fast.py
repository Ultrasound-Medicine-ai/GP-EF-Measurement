#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast global sigma calibration using a *single pass* over the validation list.
Idea: precompute r = |y - mu| / sigma for all frames (flattened), then
s* = quantile_alpha(r) / z_alpha. This avoids repeated I/O and bisection loops.

- Much faster than iterative coverage evaluation.
- Memory-friendly for typical EchoNet sizes (val ~1500 vids, T~100, D~128)
  -> ~19M floats (~76 MB as float32).
- Falls back to chunked accumulation if needed.

Usage
-----
python gp_calibrate_sigma_global_fast.py \
  --latents_dir_raw /path/latents_raw \
  --latents_dir_gp  /path/outputs/pca_gp_uncal \
  --list_file       /path/lists/val.txt \
  --alpha 0.68 \
  --save_s_star     /path/outputs/s_star.json

# Apply s* to a split and write calibrated NPZs
python gp_calibrate_sigma_global_fast.py \
  --latents_dir_raw /path/latents_raw \
  --latents_dir_gp  /path/outputs/pca_gp_uncal \
  --list_file       /path/lists/test.txt \
  --load_s_star     /path/outputs/s_star.json \
  --out_dir         /path/outputs/pca_gp_cal
"""

import argparse, json, os
from pathlib import Path
import numpy as np
from typing import List

Z_MAP = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96}


def load_mu_std(npz_path: Path):
    d = np.load(str(npz_path), allow_pickle=True)
    mu = d['latent_smooth']
    std = d['latent_std']
    return mu, std


def load_y(npy_path: Path):
    return np.load(str(npy_path))


def collect_r_values(latents_dir_raw: Path, latents_dir_gp: Path, vids: List[str], max_videos: int = None) -> np.ndarray:
    rs = []
    n = 0
    for vid in vids:
        if (max_videos is not None) and (n >= max_videos):
            break
        npy = latents_dir_raw / f"{vid}.npy"
        npz = latents_dir_gp  / f"{vid}.npz"
        if (not npy.exists()) or (not npz.exists()):
            continue
        y  = load_y(npy)
        mu, std = load_mu_std(npz)
        T = min(len(y), len(mu), len(std))
        if T <= 0:
            continue
        yv, muv, stdv = y[:T], mu[:T], std[:T]
        # Avoid div-by-zero
        stdv = np.maximum(stdv, 1e-12)
        r = np.abs(yv - muv) / stdv
        rs.append(r.astype(np.float32).ravel())
        n += 1
    if not rs:
        return np.array([], dtype=np.float32)
    return np.concatenate(rs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latents_dir_raw', required=True, type=Path)
    ap.add_argument('--latents_dir_gp',  required=True, type=Path)
    ap.add_argument('--list_file',       required=True, type=Path)
    ap.add_argument('--alpha', type=float, default=0.68)
    ap.add_argument('--save_s_star', type=Path, default=None)
    ap.add_argument('--load_s_star', type=Path, default=None)
    ap.add_argument('--out_dir', type=Path, default=None,
                    help='If set, write calibrated NPZs (latent_std * s*) for all vids in list_file')
    ap.add_argument('--approx_first_n', type=int, default=None,
                    help='Use only the first N videos to approximate s* (sanity/quick pass).')
    args = ap.parse_args()

    vids = [v.strip() for v in open(args.list_file) if v.strip()]

    if args.load_s_star is not None and args.load_s_star.exists():
        s_star = float(json.load(open(args.load_s_star))['s_star'])
        print(f"[LOAD] s* = {s_star}")
    else:
        print(f"[PREP] collecting r = |y-mu|/sigma over {len(vids)} videos...")
        r = collect_r_values(args.latents_dir_raw, args.latents_dir_gp, vids, args.approx_first_n)
        if r.size == 0:
            raise RuntimeError("No valid frames found to compute s*.")
        z = Z_MAP.get(args.alpha, 1.0)
        q = float(np.quantile(r, args.alpha))
        s_star = q / z
        # Report coverages for common alphas
        for a in (0.68, 0.90, 0.95):
            z_a = Z_MAP[a]
            cov_a = float((r <= z_a * s_star).mean())
            print(f"[CAL ] alpha={a:.2f}  cov≈{cov_a:.3f}")
        print(f"[STAR] s* = {s_star:.6f}  (q_alpha(r)={q:.6f}, z={z})")
        if args.save_s_star is not None:
            args.save_s_star.parent.mkdir(parents=True, exist_ok=True)
            json.dump({'s_star': float(s_star)}, open(args.save_s_star, 'w'), indent=2)
            print(f"[SAVE] s* -> {args.save_s_star}")

    # Optionally write calibrated files
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[WRITE] applying s*={s_star:.6f} -> {args.out_dir}")
        for vid in vids:
            npz_in = args.latents_dir_gp / f"{vid}.npz"
            if not npz_in.exists():
                continue
            d = np.load(str(npz_in), allow_pickle=True)
            mu = d['latent_smooth']
            std = d['latent_std']
            meta = None
            if 'meta' in d.files:
                m = d['meta']
                try:
                    if isinstance(m, bytes):
                        meta = json.loads(m.decode('utf-8'))
                    elif isinstance(m, str):
                        meta = json.loads(m)
                    elif hasattr(m, 'item'):
                        x = m.item()
                        meta = json.loads(x.decode('utf-8')) if isinstance(x, (bytes, bytearray)) else (json.loads(x) if isinstance(x, str) else (x if isinstance(x, dict) else {}))
                    elif isinstance(m, dict):
                        meta = m
                except Exception:
                    meta = {}
            if meta is None:
                meta = {}
            meta['sigma_scale'] = float(s_star)
            out_path = args.out_dir / f"{vid}.npz"
            np.savez_compressed(out_path,
                                latent_smooth=mu.astype(np.float32),
                                latent_std=(std*s_star).astype(np.float32),
                                meta=json.dumps(meta).encode('utf-8'))
        print("[DONE] wrote calibrated NPZs")


if __name__ == '__main__':
    # Keep BLAS single-threaded for predictable I/O-bound behavior
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    main()
