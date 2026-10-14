import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from pgmpy.inference import VariableElimination
from sklearn.metrics import roc_auc_score

BASE = r"C:\Users\Som Praneeth\Desktop\PR PROJECT\PR PROJECT"
with open(BASE + r"\flight_delay_model_final.pkl", "rb") as fh:
    bundle = pickle.load(fh)
model, calibrator = bundle["model"], bundle["calibrator"]
features, target = bundle["features"], bundle["target"]
threshold = float(bundle["threshold"])
print("Bundle: features =", features)
print("Bundle: threshold =", threshold, "| target =", target)
print("Nodes:", list(model.nodes()), "| edges:", list(model.edges()))
for cpd in model.get_cpds():
    print("  state_names", cpd.variable, list(cpd.state_names[cpd.variable]))

f = pd.read_csv(BASE + r"\archive(1)\flights.csv", low_memory=False)
f = f[f["CANCELLED"] == 0].copy()


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


f["SEASON"] = f["MONTH"].apply(season)
f["SCHED_HOUR"] = f["SCHEDULED_DEPARTURE"].apply(hour_bucket)
f["ROUTE_RAW"] = f["ORIGIN_AIRPORT"] + "->" + f["DESTINATION_AIRPORT"]
f["DELAY_BINARY"] = pd.Series(np.where(f["DEPARTURE_DELAY"] > 15, "Delayed", "OnTime"), index=f.index).where(f["DEPARTURE_DELAY"].notna())
f = f.dropna(subset=["MONTH", "DAY", "AIRLINE", "SEASON", "SCHED_HOUR", "SCHEDULED_ARRIVAL", "DISTANCE",
                     "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ROUTE_RAW", "DELAY_BINARY"]).copy()

# Same split as train_model.py default (SPLIT_MODE=day20)
train = f[f["DAY"] <= 20].copy()
test = f[f["DAY"] > 20].copy()
print(f"\nTrain rows: {len(train):,} | Test rows: {len(test):,}")

top_routes = train["ROUTE_RAW"].value_counts().head(100).index
train["ROUTE"] = train["ROUTE_RAW"].where(train["ROUTE_RAW"].isin(top_routes), "Other")
test["ROUTE"] = test["ROUTE_RAW"].where(test["ROUTE_RAW"].isin(top_routes), "Other")
gdr = (train["DELAY_BINARY"] == "Delayed").mean()
rs = train.groupby("ROUTE")["DELAY_BINARY"].agg(route_count="size", route_delayed=lambda s: (s == "Delayed").sum())
tc = train["ROUTE"].map(rs["route_count"])
td = train["ROUTE"].map(rs["route_delayed"])
isd = (train["DELAY_BINARY"] == "Delayed").astype(int)
tr_rate = ((td - isd) / (tc - 1)).where(tc > 1, gdr).fillna(gdr)
te_rate = test["ROUTE"].map(rs["route_delayed"] / rs["route_count"]).fillna(gdr)
bins, labels = [-0.01, 0.25, 0.40, 0.55, 1.01], ["VeryLow", "Low", "High", "VeryHigh"]
train["ROUTE_DELAY_HISTORY"] = pd.cut(tr_rate, bins=bins, labels=labels).astype(str)
test["ROUTE_DELAY_HISTORY"] = pd.cut(te_rate, bins=bins, labels=labels).astype(str)

train["ORIGIN_MONTH"] = train["ORIGIN_AIRPORT"] + "_" + train["MONTH"].astype(str)
test["ORIGIN_MONTH"] = test["ORIGIN_AIRPORT"] + "_" + test["MONTH"].astype(str)
cc = train["ORIGIN_MONTH"].value_counts()
edges = sorted(set(cc.quantile([0, 0.33, 0.66, 1]).values.tolist()))
if len(edges) < 4:
    edges = [0, 1, 2, float("inf")]
else:
    edges[-1] = float("inf")
for d in (train, test):
    c = d["ORIGIN_MONTH"].map(cc).fillna(0)
    d["ORIGIN_CONGESTION"] = pd.cut(c, bins=edges, labels=False, include_lowest=True).fillna(0).astype(int).astype(str)

test_clean = test.dropna(subset=features + [target]).copy()
for c in features:
    test_clean[c] = test_clean[c].astype(str)
combos = test_clean[features].drop_duplicates().reset_index(drop=True)
print(f"Unique feature combinations in test: {len(combos):,}")

inf = VariableElimination(model)
raw_list = []
unseen = 0
for row in combos.itertuples(index=False):
    ev = dict(zip(features, row))
    try:
        r = inf.query([target], evidence=ev, show_progress=False)
        raw_list.append(float(r.values[list(r.state_names[target]).index("Delayed")]))
    except Exception:
        raw_list.append(np.nan)
        unseen += 1
combos["RAW"] = raw_list
m = test_clean[features].merge(combos, on=features, how="left")
y_all = test_clean[target].to_numpy()
ok = m["RAW"].notna().to_numpy()
raw = m["RAW"].to_numpy()[ok]
y = (y_all[ok] == "Delayed").astype(int)
cal = calibrator.predict(raw)
print(f"Rows scored: {ok.sum():,} | rows skipped (unseen state): {(~ok).sum():,} | combos failed: {unseen}")


def metrics(pred, y, score=None):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = dict(acc=acc, prec=prec, rec=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)
    if score is not None and len(set(y)) > 1:
        out["auc"] = roc_auc_score(y, score)
    return out


def show(name, d):
    auc = f" AUC={d['auc']:.4f}" if "auc" in d else ""
    print(f"{name:38s} acc={d['acc']*100:6.2f}% prec={d['prec']*100:6.2f}% rec={d['rec']*100:6.2f}% "
          f"F1={d['f1']:.4f}{auc} | TP={d['tp']:,} FP={d['fp']:,} FN={d['fn']:,} TN={d['tn']:,}")


print(f"\nTest Delayed share: {y.mean()*100:.2f}%")
show("Always On-time", metrics(np.zeros(len(y), int), y))
show("Always Delayed", metrics(np.ones(len(y), int), y))
show(f"Base BN calibrated, thr={threshold}", metrics((cal >= threshold).astype(int), y, cal))
show("Base BN raw (uncalibrated), thr=0.5", metrics((raw >= 0.5).astype(int), y, raw))

print("\nCalibration by bin (calibrated probability):")
print(f"{'bin':12s}{'rows':>10s}{'mean pred':>12s}{'observed':>12s}")
for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, 1.01)]:
    mk = (cal >= lo) & (cal < hi)
    if mk.sum():
        print(f"{lo:.1f}-{min(hi,1):.1f}      {mk.sum():>10,d}{cal[mk].mean()*100:11.2f}%{y[mk].mean()*100:11.2f}%")
