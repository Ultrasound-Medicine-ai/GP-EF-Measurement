# dataset_and_config.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# === 路径配置 ===
LATENT_ROOT = Path("/mnt/EF_measurement/data/echonet/latents_raw")
CSV_PATH = Path("/mnt/EF_measurement/data/echonet/FileList_aligned.csv")

# 你可以根据需要修改
SPLIT_VALUES = {
    "train": "TRAIN",
    "val": "VAL",
    "test": "TEST",
}


class EchonetLatentDataset(Dataset):
    """
    读取 EchoNet latent 序列 (T, D) 和 EF 真值。
    默认按照 CSV 里的 Split 字段筛选样本。
    """
    def __init__(self,
                 csv_path: Path,
                 latent_root: Path,
                 split_names=("TRAIN",),
                 max_frames=None,
                 dtype=torch.float32):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.latent_root = Path(latent_root)
        self.split_names = set(split_names)
        self.max_frames = max_frames
        self.dtype = dtype

        df = pd.read_csv(self.csv_path)

        # 只保留指定 split
        df = df[df["Split"].isin(self.split_names)].copy()

        # 只保留必须列
        self.records = df[["FileName", "EF"]].reset_index(drop=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        fname = str(row["FileName"])

        # 假设 latent 文件名是 FileName + ".npy"
        latent_path = self.latent_root / f"{fname}.npy"
        if not latent_path.exists():
            # 如果你的 .csv 里本来就带了 .npy，可改成：
            # latent_path = self.latent_root / fname
            raise FileNotFoundError(f"Latent file not found: {latent_path}")

        latents = np.load(latent_path)  # (T, D)
        # 可选：截断帧数
        if self.max_frames is not None and latents.shape[0] > self.max_frames:
            latents = latents[: self.max_frames]

        latents = torch.as_tensor(latents, dtype=self.dtype)  # (T, D)
        ef = torch.as_tensor(float(row["EF"]), dtype=self.dtype)

        return latents, ef, fname
