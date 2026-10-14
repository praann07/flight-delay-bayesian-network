import pickle, os, json, warnings
warnings.filterwarnings("ignore")
BASE = r"C:\Users\Som Praneeth\Desktop\PR PROJECT\PR PROJECT"
files = [
    "flight_delay_model_final.pkl",
    "flight_delay_model_final.bak.pkl",
    "flight_delay_model_v2_1.pkl",
    "cascade_model.pkl",
    os.path.join("model_v2", "flight_delay_model_v2.pkl"),
]
for f in files:
    p = os.path.join(BASE, f)
    print("=" * 70)
    print(f, "| size:", os.path.getsize(p))
    with open(p, "rb") as fh:
        obj = pickle.load(fh)
    print("type:", type(obj).__module__, type(obj).__name__)
    if isinstance(obj, dict):
        for k, v in obj.items():
            tn = type(v).__name__
            if k in ("model", "gbm", "calibrator") or hasattr(v, "fit"):
                extra = ""
                if hasattr(v, "nodes") and hasattr(v, "edges"):
                    extra = f"nodes={len(v.nodes())} edges={len(v.edges())}"
                print(f"  {k}: {tn} {extra}")
            elif k == "metrics" and isinstance(v, dict):
                print("  metrics:", json.dumps({a: b for a, b in v.items() if a != "calibration_table"}, default=str))
                if "calibration_table" in v:
                    print("  calibration_table:", json.dumps(v["calibration_table"], default=str))
            elif k == "gbm_metrics":
                print("  gbm_metrics:", json.dumps(v, default=str))
            elif k in ("support_table", "cpt"):
                print(f"  {k}: {len(v)} entries")
            else:
                print(f"  {k}: {repr(v)[:300]}")
    else:
        print("  object:", repr(obj)[:300])
        if hasattr(obj, "nodes"):
            print("  nodes:", len(obj.nodes()), "edges:", len(obj.edges()))
