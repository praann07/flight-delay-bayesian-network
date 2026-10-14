import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

BASE = r"C:\Users\Som Praneeth\Desktop\PR PROJECT\PR PROJECT"
OUT = r"C:\Users\Som Praneeth\Desktop\PR PROJECT\FINAL_RESULTS\robustness"
os.makedirs(OUT, exist_ok=True)
FEATURES = ["AIRLINE", "SCHED_HOUR", "SEASON", "ROUTE_DELAY_HISTORY", "ORIGIN_CONGESTION"]
TARGET = "DELAY_BINARY"


def season(m):
    return "Winter" if m in (12, 1, 2) else "Spring" if m in (3, 4, 5) else "Summer" if m in (6, 7, 8) else "Fall"


def hour_bucket(t):
    try:
        h = int(t) // 100
    except (ValueError, TypeError):
        return None
    if h < 6: return "EarlyMorning"
    if h < 9: return "AMPeak"
    if h < 12: return "Midmorning"
    if h < 15: return "EarlyAfternoon"
    if h < 19: return "PMPeak"
    return "Night"


raw = pd.read_csv(BASE + r"\archive(1)\flights.csv", low_memory=False)
raw = raw[raw["CANCELLED"] == 0].copy()
raw["SEASON"] = raw["MONTH"].apply(season)
raw["SCHED_HOUR"] = raw["SCHEDULED_DEPARTURE"].apply(hour_bucket)
raw["ROUTE_RAW"] = raw["ORIGIN_AIRPORT"] + "->" + raw["DESTINATION_AIRPORT"]
raw[TARGET] = pd.Series(np.where(raw["DEPARTURE_DELAY"] > 15, "Delayed", "OnTime"), index=raw.index).where(raw["DEPARTURE_DELAY"].notna())
raw = raw.dropna(subset=["SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL", "DISTANCE", "DEPARTURE_DELAY", "ORIGIN_AIRPORT",
                         "DESTINATION_AIRPORT", "AIRLINE", "MONTH", "DAY", "SCHED_HOUR", "ROUTE_RAW", TARGET]).copy()
print("Clean flights:", len(raw), flush=True)


def build(tr_mask, te_mask):
    tr = raw[tr_mask].copy()
    te = raw[te_mask].copy()
    top_routes = tr["ROUTE_RAW"].value_counts().head(100).index
    for d in (tr, te):
        d["ROUTE"] = d["ROUTE_RAW"].where(d["ROUTE_RAW"].isin(top_routes), "Other")
    gdr = (tr[TARGET] == "Delayed").mean()
    rs = tr.groupby("ROUTE")[TARGET].agg(route_count="size", route_delayed=lambda s: (s == "Delayed").sum())
    tc = tr["ROUTE"].map(rs["route_count"])
    td = tr["ROUTE"].map(rs["route_delayed"])
    isd = (tr[TARGET] == "Delayed").astype(int)
    tr_rate = ((td - isd) / (tc - 1)).where(tc > 1, gdr).fillna(gdr)
    te_rate = te["ROUTE"].map(rs["route_delayed"] / rs["route_count"]).fillna(gdr)
    bins, labels = [-0.01, 0.25, 0.40, 0.55, 1.01], ["VeryLow", "Low", "High", "VeryHigh"]
    tr["ROUTE_DELAY_HISTORY"] = pd.cut(tr_rate, bins=bins, labels=labels).astype(str)
    te["ROUTE_DELAY_HISTORY"] = pd.cut(te_rate, bins=bins, labels=labels).astype(str)
    n_months = tr["MONTH"].nunique()
    airport_vol = tr["ORIGIN_AIRPORT"].value_counts() / n_months
    edges = sorted(set(airport_vol.quantile([0, 1 / 3, 2 / 3, 1]).values.tolist()))
    edges[-1] = float("inf")
    for d in (tr, te):
        c = d["ORIGIN_AIRPORT"].map(airport_vol).fillna(0)
        d["ORIGIN_CONGESTION"] = pd.cut(c, bins=edges, labels=False, include_lowest=True).fillna(0).astype(int).astype(str)
    return tr[FEATURES + [TARGET]].reset_index(drop=True), te[FEATURES + [TARGET]].reset_index(drop=True)


def prob(model, df):
    inf = VariableElimination(model)
    combos = df[FEATURES].astype(str).drop_duplicates().reset_index(drop=True)
    vals = []
    for row in combos.itertuples(index=False):
        try:
            r = inf.query([TARGET], evidence=dict(zip(FEATURES, row)), show_progress=False)
            vals.append(float(r.values[list(r.state_names[TARGET]).index("Delayed")]))
        except Exception:
            vals.append(np.nan)
    combos["P"] = vals
    m = df[FEATURES].astype(str).merge(combos, on=FEATURES, how="left")
    return m["P"].to_numpy()


def choose_threshold(p, y):
    best = None
    for t in np.arange(0.10, 0.501, 0.05):
        t = round(float(t), 2)
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if best is None or (f1, t) > best:
            best = (f1, t)
    return best[1]


def fit_and_calibrate(tr, ess):
    cal_df = tr.sample(n=100000, random_state=123)
    rest = tr.drop(index=cal_df.index)
    d = rest[rest[TARGET] == "Delayed"].sample(n=100000, random_state=42)
    o = rest[rest[TARGET] == "OnTime"].sample(n=100000, random_state=42)
    fit_df = pd.concat([d, o]).sample(frac=1, random_state=42)
    model = DiscreteBayesianNetwork([(f, TARGET) for f in FEATURES])
    model.fit(fit_df[FEATURES + [TARGET]], estimator=DiscreteBayesianEstimator(prior_type="BDeu", equivalent_sample_size=ess))
    raw_cal = prob(model, cal_df)
    ok = ~np.isnan(raw_cal)
    y_cal = (cal_df[TARGET] == "Delayed").astype(int).to_numpy()[ok]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw_cal[ok], y_cal)
    thr = choose_threshold(iso.predict(raw_cal[ok]), y_cal)
    return model, iso, thr


def metrics(p, y, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(n=int(len(y)), base_rate=float(y.mean()), threshold=float(thr), auc=float(roc_auc_score(y, p)),
                pr_auc=float(average_precision_score(y, p)), accuracy=(tp + tn) / len(y), precision=prec,
                recall=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def run_split(name, tr_mask, te_mask, ess):
    print(f"\n[{name}] ESS={ess}: building features", flush=True)
    tr, te = build(tr_mask, te_mask)
    print(f"[{name}] train={len(tr):,} test={len(te):,}; fitting", flush=True)
    print(f"[{name}] congestion train:", tr["ORIGIN_CONGESTION"].value_counts().sort_index().to_dict(),
          "test:", te["ORIGIN_CONGESTION"].value_counts().sort_index().to_dict(), flush=True)
    model, iso, thr = fit_and_calibrate(tr, ess)
    p_raw = prob(model, te)
    ok = ~np.isnan(p_raw)
    p_cal = iso.predict(p_raw[ok])
    y = (te[TARGET] == "Delayed").astype(int).to_numpy()[ok]
    m = metrics(p_cal, y, thr)
    m["dropped_rows"] = int((~ok).sum())
    print(f"[{name}] ESS={ess}: AUC={m['auc']:.4f} PR-AUC={m['pr_auc']:.4f} thr={thr:.2f} "
          f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} acc={m['accuracy']:.4f} dropped={m['dropped_rows']}", flush=True)
    return m, p_cal, y, thr


def bootstrap(p, y, thr, reps=150, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs, f1s = [], []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        pp, yy = p[idx], y[idx]
        aucs.append(roc_auc_score(yy, pp))
        pred = pp >= thr
        tp = (pred & (yy == 1)).sum(); fp = (pred & (yy == 0)).sum(); fn = (~pred & (yy == 1)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"auc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
            "f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))], "reps": reps}


results = {"day_split": {}, "month_split": {}}
curves = {}

day_tr = raw["DAY"] <= 20
day_te = raw["DAY"] > 20
for ess in [10]:
    m, p, y, thr = run_split("day", day_tr, day_te, ess)
    results["day_split"][ess] = m
    if ess == 10:
        curves["day"] = (p, y, thr)
        results["day_split"]["bootstrap_ess10"] = bootstrap(p, y, thr)
        print("[day] bootstrap:", results["day_split"]["bootstrap_ess10"], flush=True)

month_tr = raw["MONTH"] <= 10
month_te = raw["MONTH"] > 10
m, p, y, thr = run_split("month", month_tr, month_te, 10)
results["month_split"][10] = m
curves["month"] = (p, y, thr)
results["month_split"]["bootstrap_ess10"] = bootstrap(p, y, thr)
print("[month] bootstrap:", results["month_split"]["bootstrap_ess10"], flush=True)

p, y, _ = curves["month"]
N = len(y)
pos = int(y.sum())
neg = N - pos
grid = np.round(np.arange(0.02, 0.81, 0.01), 2)
cost_table = {}
for r in [1, 2, 5]:
    costs = []
    for t in grid:
        pred = p >= t
        fn = int((~pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        costs.append((r * fn + fp) / N)
    costs = np.array(costs)
    i = int(np.argmin(costs))
    at20 = float(costs[list(grid).index(0.2)])
    cost_table[r] = {"optimal_threshold": float(grid[i]), "optimal_cost": float(costs[i]),
                     "cost_at_0.20": at20,
                     "always_ontime_cost": r * pos / N, "always_delayed_cost": neg / N}
    print(f"[cost] FN:FP={r}:1 optimal thr={grid[i]:.2f} cost={costs[i]:.5f} | cost@0.20={at20:.5f} | "
          f"always-OnTime={r*pos/N:.5f} always-Delayed={neg/N:.5f}", flush=True)
results["cost"] = cost_table

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for key, label in [("day", "Day split (days 1-20 / 21-31)"), ("month", "Month split (Jan-Oct / Nov-Dec)")]:
    p, y, _ = curves[key]
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, label=f"{label}, AUC={roc_auc_score(y, p):.3f}")
    pr, rc, _ = precision_recall_curve(y, p)
    axes[1].plot(rc, pr, label=f"{label}, PR-AUC={average_precision_score(y, p):.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="grey", label="Chance")
axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (calibrated base BN)")
axes[1].axhline(curves["month"][1].mean(), ls="--", color="grey", label="Base rate (Nov-Dec)")
axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (calibrated base BN)")
for ax in axes:
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "roc_pr_curves.png"), dpi=150)
plt.close(fig)

p, y, _ = curves["month"]
fig, ax = plt.subplots(figsize=(7, 4.8))
for r in [1, 2, 5]:
    costs = []
    for t in grid:
        pred = p >= t
        fn = int((~pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        costs.append((r * fn + fp) / N)
    ax.plot(grid, costs, label=f"FN:FP = {r}:1")
ax.axvline(0.2, ls="--", color="black", label="Chosen threshold 0.20")
ax.set(xlabel="Decision threshold", ylabel="Expected cost per flight (normalized)", title="Cost curve, month split")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "cost_curve.png"), dpi=150)
plt.close(fig)

with open(os.path.join(OUT, "robustness_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved:", OUT, flush=True)
