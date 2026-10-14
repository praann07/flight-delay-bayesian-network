import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from pgmpy.inference import VariableElimination

BASE = r"C:\Users\Som Praneeth\Desktop\PR PROJECT\PR PROJECT"
with open(BASE + r"\cascade_model.pkl", "rb") as fh:
    model = pickle.load(fh)

edge = list(model.edges())[0]
parent, child = edge
child_states = list(model.get_cpds(child).state_names[child])
parent_states = list(model.get_cpds(parent).state_names[parent])
print("Edge:", parent, "->", child, "| states:", child_states)

inf = VariableElimination(model)
p_next_delayed = {}
for s in parent_states:
    r = inf.query([child], evidence={parent: s}, show_progress=False)
    p_next_delayed[s] = float(r.values[child_states.index("Delayed")])
print("P(next Delayed | current):", {k: round(v, 4) for k, v in p_next_delayed.items()})

f = pd.read_csv(BASE + r"\archive(1)\flights.csv", low_memory=False,
                usecols=["TAIL_NUMBER", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "DEPARTURE_DELAY", "CANCELLED"])
f = f[f["CANCELLED"] == 0][["TAIL_NUMBER", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "DEPARTURE_DELAY"]].dropna()
f = f.sort_values(["TAIL_NUMBER", "MONTH", "DAY", "SCHEDULED_DEPARTURE"])
f["CUR"] = np.where(f["DEPARTURE_DELAY"] > 15, "Delayed", "OnTime")
f["NEXT"] = f.groupby("TAIL_NUMBER")["CUR"].shift(-1)
pairs = f.dropna(subset=["NEXT"])[["MONTH", "CUR", "NEXT"]]


def metrics(df, pred_delayed):
    true = (df["NEXT"] == "Delayed").to_numpy()
    pred = np.asarray(pred_delayed, dtype=bool)
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum()); tn = int((~pred & ~true).sum())
    n = len(true)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(n=n, acc=acc, prec=prec, rec=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def show(name, m):
    print(f"{name:34s} n={m['n']:>9,d} acc={m['acc']*100:6.2f}% prec={m['prec']*100:6.2f}% "
          f"rec={m['rec']*100:6.2f}% F1={m['f1']:.4f} | TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")


for label, df in [("TEST Nov-Dec (current flight)", pairs[pairs["MONTH"] > 10]),
                  ("ALL pairs (full year)", pairs)]:
    print("\n" + "=" * 110)
    print(label, "| natural Delayed share of next flight: %.2f%%" % ((df["NEXT"] == "Delayed").mean() * 100))
    print("=" * 110)
    cur_prob = df["CUR"].map(p_next_delayed).to_numpy()
    show("Always On-time (baseline)", metrics(df, np.zeros(len(df), dtype=bool)))
    show("Always Delayed", metrics(df, np.ones(len(df), dtype=bool)))
    for t in [0.05, 0.10, 0.12, 0.20, 0.30, 0.40, 0.48, 0.50]:
        show(f"Cascade, threshold {t:.2f}", metrics(df, cur_prob >= t))
    print("\nCurrent-delay share in this set: Delayed %.2f%%" % ((df["CUR"] == "Delayed").mean() * 100))
