import os, json
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

latents_dir = "/mnt/4DHeartModel/experiments/EchoNet/latents_raw"
list_file = "/mnt/4DHeartModel/experiments/EchoNet/lists/train.txt"
save_path = "/mnt/4DHeartModel/experiments/EchoNet/pca_meta.json"

video_ids = [x.strip() for x in open(list_file)]
print(f"Found {len(video_ids)} training videos")

all_latents = []
for vid in tqdm(video_ids):
    f = os.path.join(latents_dir, f"{vid}.npy")
    if not os.path.exists(f):
        continue
    L = np.load(f)
    Lc = L - L.mean(0, keepdims=True)
    all_latents.append(Lc)

X = np.concatenate(all_latents, axis=0)
print("Concatenated shape:", X.shape)

pca = PCA(n_components=0.98, svd_solver='full')
Z = pca.fit_transform(X)

meta = {
    "mean": pca.mean_,
    "components": pca.components_,
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "n_components": pca.n_components_
}

def json_safe(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    elif isinstance(o, np.ndarray):
        return o.tolist()
    else:
        return o

with open(save_path, "w") as f:
    json.dump(meta, f, indent=2, default=json_safe)

print(f"[SAVE] PCA meta saved to {save_path}")
print(f"[INFO] explained variance ratio sum = {pca.explained_variance_ratio_.sum():.3f}")

