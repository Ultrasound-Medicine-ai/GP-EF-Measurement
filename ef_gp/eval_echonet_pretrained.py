import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

def build_r2plus1d_18_regressor():
    import torchvision
    try:
        model = torchvision.models.video.r2plus1d_18(weights=None)
    except TypeError:
        model = torchvision.models.video.r2plus1d_18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def _strip_module_prefix(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def load_checkpoint(weights_path: Path, model: nn.Module):
    ckpt = torch.load(str(weights_path), map_location="cpu")

    state_dict = None
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        else:
            tensor_items = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
            if len(tensor_items) > 0:
                state_dict = tensor_items
    else:
        state_dict = ckpt

    if state_dict is None:
        raise RuntimeError("Cannot find state_dict in checkpoint.")

    state_dict = _strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[Warn] missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[Warn] unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

def load_video_opencv(video_path: Path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"Empty video: {video_path}")
    return np.stack(frames, axis=0)  # (T,H,W,3)

def pick_video_path(videos_dir: Path, file_id: str):
    p = videos_dir / file_id
    if p.exists():
        return p
    if not file_id.endswith(".avi"):
        p2 = videos_dir / (file_id + ".avi")
        if p2.exists():
            return p2
    for ext in [".avi", ".mp4", ".mkv"]:
        p3 = videos_dir / (file_id + ext)
        if p3.exists():
            return p3
    raise FileNotFoundError(f"Video not found: {file_id} under {videos_dir}")

def extract_clips(frames_uint8, length=32, period=2, nclips=10):
    T = frames_uint8.shape[0]
    span = period * (length - 1) + 1
    max_start = max(0, T - span)
    starts = [max_start // 2] if nclips <= 1 else np.linspace(0, max_start, num=nclips).astype(int).tolist()
    base = np.arange(length) * period

    clips = []
    for s in starts:
        idx = np.clip(s + base, 0, T - 1)
        clip = frames_uint8[idx].transpose(3, 0, 1, 2)  # (3,L,H,W)
        clips.append(clip)
    return np.stack(clips, axis=0)  # (N,3,L,H,W)

@torch.no_grad()
def infer_video(model, device, frames_uint8, length=32, period=2, nclips=10, amp=True):
    clips = extract_clips(frames_uint8, length=length, period=period, nclips=nclips)
    x = torch.from_numpy(clips).float() / 255.0
    x = x.to(device, non_blocking=True)
    if amp and device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            pred = model(x).squeeze(-1)
    else:
        pred = model(x).squeeze(-1)
    pred = pred.float().cpu().numpy()
    return float(pred.mean())

def autodetect_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of columns {candidates} found. Available: {list(df.columns)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str)
    ap.add_argument("--videos", required=True, type=str)
    ap.add_argument("--weights", required=True, type=str)
    ap.add_argument("--split", default="TEST", type=str)
    ap.add_argument("--length", default=32, type=int)
    ap.add_argument("--period", default=2, type=int)
    ap.add_argument("--nclips", default=10, type=int)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save_preds", default="", type=str)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    id_col = autodetect_col(df, ["FileName", "Filename", "file", "id", "ID"])
    ef_col = autodetect_col(df, ["EF", "ef"])
    split_col = autodetect_col(df, ["Split", "split", "SPLIT"])

    df[split_col] = df[split_col].astype(str).str.upper()
    split = args.split.upper()
    df_s = df[df[split_col] == split].copy()
    if len(df_s) == 0:
        raise RuntimeError(f"No samples for split={split}. Check split labels in CSV.")

    videos_dir = Path(args.videos)
    weights_path = Path(args.weights)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = build_r2plus1d_18_regressor()
    load_checkpoint(weights_path, model)
    model.eval().to(device)

    preds, ys, ids = [], [], []
    for k, row in enumerate(df_s.itertuples(index=False), start=1):
        fid = str(getattr(row, id_col))
        y = float(getattr(row, ef_col))
        vp = pick_video_path(videos_dir, fid)
        frames = load_video_opencv(vp)
        mu = infer_video(model, device, frames, length=args.length, period=args.period, nclips=args.nclips, amp=args.amp)
        preds.append(mu); ys.append(y); ids.append(fid)
        if (k % 50) == 0:
            print(f"[{split}] {k}/{len(df_s)} done...")

    preds = np.array(preds, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    rmse = float(np.sqrt(np.mean((preds - ys) ** 2)))
    mae = float(np.mean(np.abs(preds - ys)))

    print(f"[EchoNet r2plus1d_18_32_2 pretrained] Split={split} N={len(df_s)}")
    print(f"Point RMSE={rmse:.4f}, MAE={mae:.4f}")

    if args.save_preds:
        out = pd.DataFrame({id_col: ids, "EF": ys, "pred": preds})
        out.to_csv(args.save_preds, index=False)
        print("[Saved]", args.save_preds)

if __name__ == "__main__":
    main()