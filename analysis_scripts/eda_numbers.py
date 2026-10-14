import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
BASE = r"C:\Users\Som Praneeth\Desktop\PR PROJECT\PR PROJECT\archive(1)"
airlines = pd.read_csv(f"{BASE}\\airlines.csv").rename(columns={"AIRLINE": "AIRLINE_NAME"})
airports = pd.read_csv(f"{BASE}\\airports.csv")
f = pd.read_csv(f"{BASE}\\flights.csv", low_memory=False)
print("FLIGHTS rows:", len(f), "cols:", f.shape[1])
print("Airlines:", f["AIRLINE"].nunique(), "| Tail numbers:", f["TAIL_NUMBER"].nunique())
print("Origins:", f["ORIGIN_AIRPORT"].nunique(), "| Destinations:", f["DESTINATION_AIRPORT"].nunique())
print("Cancelled:", int((f["CANCELLED"] == 1).sum()), "| Diverted:", int((f["DIVERTED"] == 1).sum()))
d = f["DEPARTURE_DELAY"]
print("Dep delay mean %.2f median %.2f p75 %.2f" % (d.mean(), d.median(), d.quantile(0.75)))
rate = (d > 15).mean()
print("Delayed >15: %.2f%% | On-time: %.2f%%" % (rate * 100, (1 - rate) * 100))
print("Missing DEPARTURE_DELAY:", int(d.isna().sum()))
cause = ["AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"]
print("\nDELAY CAUSE COVERAGE")
for c in cause:
    print(f"{c}: rows={int(f[c].notna().sum())} share={f[c].notna().mean()*100:.2f}%")

print("\nAIRLINE DELAY RATE")
al = (f.groupby("AIRLINE").agg(n=("FLIGHT_NUMBER", "count"), rate=("DEPARTURE_DELAY", lambda s: (s > 15).mean()))
      .reset_index().merge(airlines, left_on="AIRLINE", right_on="IATA_CODE", how="left")
      .sort_values("rate", ascending=False))
for _, r in al.iterrows():
    print(f"{r['AIRLINE']:4s} {str(r['AIRLINE_NAME'])[:28]:28s} {int(r['n']):>9d} {r['rate']*100:6.2f}%")

print("\nMONTH")
m = f.groupby("MONTH")["DEPARTURE_DELAY"].agg(mean="mean", rate=lambda s: (s > 15).mean())
for mo, r in m.iterrows():
    print(f"{mo:2d} mean={r['mean']:6.2f} rate={r['rate']*100:6.2f}%")

print("\nDAY OF WEEK")
dw = f.groupby("DAY_OF_WEEK")["DEPARTURE_DELAY"].agg(mean="mean", rate=lambda s: (s > 15).mean())
for k, r in dw.iterrows():
    print(f"{k} mean={r['mean']:6.2f} rate={r['rate']*100:6.2f}%")
wk = f["DAY_OF_WEEK"].isin([6, 7])
print("Weekday rate %.2f%% | Weekend rate %.2f%%" % (((d[~wk]) > 15).mean() * 100, ((d[wk]) > 15).mean() * 100))

print("\nTIME OF DAY")
hour = f["SCHEDULED_DEPARTURE"] // 100
period = np.select([(hour >= 5) & (hour < 12), (hour >= 12) & (hour < 17), (hour >= 17) & (hour < 21)],
                   ["Morning", "Afternoon", "Evening"], default="Night")
t = f.assign(P=period).groupby("P")["DEPARTURE_DELAY"].agg(n="count", mean="mean", rate=lambda s: (s > 15).mean())
for k, r in t.iterrows():
    print(f"{k:10s} n={int(r['n']):>9d} mean={r['mean']:6.2f} rate={r['rate']*100:6.2f}%")

print("\nTOP 15 ORIGINS")
o = (f.groupby("ORIGIN_AIRPORT")["DEPARTURE_DELAY"].agg(n="count", rate=lambda s: (s > 15).mean())
     .reset_index().sort_values("n", ascending=False).head(15)
     .merge(airports, left_on="ORIGIN_AIRPORT", right_on="IATA_CODE", how="left"))
for _, r in o.iterrows():
    print(f"{r['ORIGIN_AIRPORT']:4s} {str(r['CITY'])[:18]:18s} {int(r['n']):>8d} {r['rate']*100:6.2f}%")

print("\nCASCADE")
s = f[f["CANCELLED"] == 0][["TAIL_NUMBER", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "DEPARTURE_DELAY"]].dropna()
s = s.sort_values(["TAIL_NUMBER", "MONTH", "DAY", "SCHEDULED_DEPARTURE"])
s["CAT"] = np.where(s["DEPARTURE_DELAY"] > 15, "Delayed", "OnTime")
s["NEXT"] = s.groupby("TAIL_NUMBER")["CAT"].shift(-1)
c = s.dropna(subset=["NEXT"])
print("Pairs:", len(c), "| Aircraft:", c["TAIL_NUMBER"].nunique())
ct = pd.crosstab(c["CAT"], c["NEXT"], normalize="index") * 100
print(ct.round(2).to_string())
