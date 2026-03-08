import numpy as np, pandas as pd, math

alpha=0.1  # 90% interval
val = pd.read_csv("echonet_r2plus1d_val_preds_20c.csv")
test = pd.read_csv("echonet_r2plus1d_test_preds_20c.csv")

# finite-sample corrected conformal quantile
scores = np.abs(val["EF"].values - val["pred"].values).astype(float)
n = scores.size
level = math.ceil((n+1)*(1-alpha))/n
level = min(max(level,0.0),1.0)
q = float(np.quantile(scores, level))

y = test["EF"].values.astype(float)
mu = test["pred"].values.astype(float)

L = mu - q
U = mu + q

cov = float(np.mean((y>=L)&(y<=U)))
width = float(np.mean(U-L))
rmse = float(np.sqrt(np.mean((mu-y)**2)))
mae = float(np.mean(np.abs(mu-y)))

print(f"[Residual Conformal] alpha={alpha}")
print(f"VAL q={q:.4f} (|y-mu| quantile)")
print(f"TEST point RMSE={rmse:.4f}, MAE={mae:.4f}")
print(f"TEST coverage={cov:.4f}, mean_width={width:.4f}")

out = test.copy()
out["L"] = L
out["U"] = U
out.to_csv("echonet_r2plus1d_test_intervals_residual.csv", index=False)
print("[Saved] echonet_r2plus1d_test_intervals_residual.csv")