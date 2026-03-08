import numpy as np, pandas as pd
#临床阈值分析
df = pd.read_csv("echonet_r2plus1d_test_intervals_residual.csv")
y = df["EF"].values.astype(float)
mu = df["pred"].values.astype(float)
L = df["L"].values.astype(float)
U = df["U"].values.astype(float)

def report_threshold(t):
    # True label: 1 if EF >= t
    yb = (y >= t).astype(int)
    # Point prediction class
    pb = (mu >= t).astype(int)

    # Interval-based "actionable" decision:
    # if interval entirely below t => class 0
    # if interval entirely above/at t => class 1
    # else abstain (crosses threshold)
    action = (U < t) | (L >= t)
    pred_action = np.where(L >= t, 1, 0)  # only meaningful where action==True

    # Metrics on actionable subset
    idx = np.where(action)[0]
    ar = float(action.mean())
    abst = 1.0 - ar
    if len(idx) > 0:
        yt = yb[idx]; pt = pred_action[idx]
        acc = float((yt==pt).mean())
        tp = int(((yt==1)&(pt==1)).sum()); tn = int(((yt==0)&(pt==0)).sum())
        fp = int(((yt==0)&(pt==1)).sum()); fn = int(((yt==1)&(pt==0)).sum())
        sens = tp / (tp+fn) if (tp+fn)>0 else float("nan")
        spec = tn / (tn+fp) if (tn+fp)>0 else float("nan")
    else:
        acc=sens=spec=float("nan")

    # "Does the interval catch point mistakes?"
    point_wrong = (pb != yb)
    caught = point_wrong & (~action)  # wrong + abstain => flagged for review
    catch_rate = float(caught.sum() / max(point_wrong.sum(), 1))

    print(f"\n[Threshold t={t}]")
    print(f"Actionable rate={ar:.3f} | Abstain rate={abst:.3f}")
    print(f"Actionable subset: Acc={acc:.3f} Sens={sens:.3f} Spec={spec:.3f}")
    print(f"Point mistakes flagged by interval (abstain): {catch_rate:.3f}  (higher is better)")

for t in [40, 50]:
    report_threshold(t)

# Optional: also show how many cases are near-threshold (hard clinical cases)
near40 = float(np.mean(np.abs(y-40) <= 5))
near50 = float(np.mean(np.abs(y-50) <= 5))
print(f"\n[Hard cases] |EF-40|<=5: {near40:.3f}  |EF-50|<=5: {near50:.3f}")