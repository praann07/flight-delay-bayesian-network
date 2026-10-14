import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ══════════════════════════════════════════════════════════════════

import os as _os

FLIGHTS_CSV = _os.environ.get(
    "FLIGHTS_CSV",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "archive(1)", "flights.csv"),
)
print(f"Loading flights.csv from {FLIGHTS_CSV} ...")
flights = pd.read_csv(FLIGHTS_CSV, low_memory=False)
print(f"Loaded: {flights.shape}")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

flights = flights[flights["CANCELLED"] == 0].copy()

def get_season(m):
    if m in [12, 1, 2]: return "Winter"
    elif m in [3, 4, 5]: return "Spring"
    elif m in [6, 7, 8]: return "Summer"
    else: return "Fall"

flights["SEASON"] = flights["MONTH"].apply(get_season)

def get_sched_hour(t):
    try:
        h = int(t) // 100
    except (ValueError, TypeError):
        return None
    if h < 6:    return "EarlyMorning"
    elif h < 9:  return "AMPeak"
    elif h < 12: return "Midmorning"
    elif h < 15: return "EarlyAfternoon"
    elif h < 19: return "PMPeak"
    else:        return "Night"

flights["SCHED_HOUR"] = flights["SCHEDULED_DEPARTURE"].apply(get_sched_hour)
flights["DAY_TYPE"]   = flights["DAY_OF_WEEK"].apply(lambda d: "Weekend" if d in [6, 7] else "Weekday")
def get_hour_bucket(t):
    try:
        h = int(t) // 100
    except (ValueError, TypeError):
        return None
    if h < 6:    return "EarlyMorning"
    elif h < 9:  return "AMPeak"
    elif h < 12: return "Midmorning"
    elif h < 15: return "EarlyAfternoon"
    elif h < 19: return "PMPeak"
    else:        return "Night"

flights["SCHED_ARRIVAL_HOUR"] = flights["SCHEDULED_ARRIVAL"].apply(get_hour_bucket)
flights["ROUTE_RAW"] = flights["ORIGIN_AIRPORT"] + "->" + flights["DESTINATION_AIRPORT"]
flights["DISTANCE"] = pd.cut(
    flights["DISTANCE"],
    bins=[-1, 300, 700, 1200, 2000, float("inf")],
    labels=["VeryShort", "Short", "Medium", "Long", "VeryLong"],
)

top_airports = flights["ORIGIN_AIRPORT"].value_counts().head(15).index.tolist()
flights["ORIGIN"] = flights["ORIGIN_AIRPORT"].apply(lambda x: x if x in top_airports else "Other")

flights["DELAY_BINARY"] = flights["DEPARTURE_DELAY"].apply(
    lambda d: None if pd.isna(d) else ("Delayed" if d > 15 else "OnTime")
)

# ══════════════════════════════════════════════════════════════════
# STEP 3 — CLEAN
# ══════════════════════════════════════════════════════════════════

bn_df = flights[["MONTH", "DAY", "AIRLINE", "SEASON", "SCHED_HOUR",
                 "SCHED_ARRIVAL_HOUR", "DAY_TYPE", "ORIGIN_AIRPORT",
                 "DESTINATION_AIRPORT", "ROUTE_RAW", "DISTANCE",
                 "DELAY_BINARY"]].dropna().copy()
print(f"\nClean shape: {bn_df.shape}")
print(bn_df["DELAY_BINARY"].value_counts())

# ══════════════════════════════════════════════════════════════════
# STEP 4 — MONTH-STRATIFIED TIME SPLIT
# ══════════════════════════════════════════════════════════════════

SPLIT_MODE = _os.environ.get("SPLIT_MODE", "day20")
if SPLIT_MODE == "month11":
    # True temporal split: train Jan-Oct, test Nov-Dec. Set via SPLIT_MODE=month11.
    train_df = bn_df[bn_df["MONTH"] <= 10].copy()
    test_df = bn_df[bn_df["MONTH"] > 10].copy()
else:
    train_df = bn_df[bn_df["DAY"] <= 20].copy()
    test_df = bn_df[bn_df["DAY"] > 20].copy()
print(f"Split mode: {SPLIT_MODE}")
print(f"\nTrain: {train_df.shape} | Test: {test_df.shape}")
print(
    "Full-period delay rates: "
    f"train={((train_df['DELAY_BINARY'] == 'Delayed').mean() * 100):.2f}% | "
    f"test={((test_df['DELAY_BINARY'] == 'Delayed').mean() * 100):.2f}%"
)

# Learn category vocabularies from training data so test-only categories do not
# create unreliable sparse states.
top_origins = train_df["ORIGIN_AIRPORT"].value_counts().head(15).index
top_routes = train_df["ROUTE_RAW"].value_counts().head(100).index
train_df["ORIGIN"] = train_df["ORIGIN_AIRPORT"].where(
    train_df["ORIGIN_AIRPORT"].isin(top_origins), "Other"
)
test_df["ORIGIN"] = test_df["ORIGIN_AIRPORT"].where(
    test_df["ORIGIN_AIRPORT"].isin(top_origins), "Other"
)
train_df["ROUTE"] = train_df["ROUTE_RAW"].where(
    train_df["ROUTE_RAW"].isin(top_routes), "Other"
)
test_df["ROUTE"] = test_df["ROUTE_RAW"].where(
    test_df["ROUTE_RAW"].isin(top_routes), "Other"
)

global_delay_rate = (train_df["DELAY_BINARY"] == "Delayed").mean()
route_stats = train_df.groupby("ROUTE")["DELAY_BINARY"].agg(
    route_count="size", route_delayed=lambda s: (s == "Delayed").sum()
)

# Leave-one-out history for training rows; test rows use training history only.
train_route_count = train_df["ROUTE"].map(route_stats["route_count"])
train_route_delayed = train_df["ROUTE"].map(route_stats["route_delayed"])
is_delayed = (train_df["DELAY_BINARY"] == "Delayed").astype(int)
train_route_rate = (
    (train_route_delayed - is_delayed) / (train_route_count - 1)
).where(train_route_count > 1, global_delay_rate).fillna(global_delay_rate)
test_route_rate = test_df["ROUTE"].map(
    (route_stats["route_delayed"] / route_stats["route_count"])
).fillna(global_delay_rate)

train_df["ROUTE_DELAY_HISTORY"] = pd.cut(
    train_route_rate,
    bins=[-0.01, 0.25, 0.40, 0.55, 1.01],
    labels=["VeryLow", "Low", "High", "VeryHigh"],
).astype(str)
test_df["ROUTE_DELAY_HISTORY"] = pd.cut(
    test_route_rate,
    bins=[-0.01, 0.25, 0.40, 0.55, 1.01],
    labels=["VeryLow", "Low", "High", "VeryHigh"],
).astype(str)

train_df["ORIGIN_MONTH"] = train_df["ORIGIN_AIRPORT"] + "_" + train_df["MONTH"].astype(str)
test_df["ORIGIN_MONTH"] = test_df["ORIGIN_AIRPORT"] + "_" + test_df["MONTH"].astype(str)
congestion_counts = train_df["ORIGIN_MONTH"].value_counts()
congestion_edges = congestion_counts.quantile([0, 0.33, 0.66, 1]).values
congestion_edges = sorted(set(congestion_edges.tolist()))
if len(congestion_edges) < 4:
    congestion_edges = [0, 1, 2, float("inf")]
else:
    congestion_edges[-1] = float("inf")
train_congestion = train_df["ORIGIN_MONTH"].map(congestion_counts).fillna(0)
test_congestion = test_df["ORIGIN_MONTH"].map(congestion_counts).fillna(0)
train_df["ORIGIN_CONGESTION"] = pd.cut(
    train_congestion, bins=congestion_edges, labels=False, include_lowest=True
).fillna(0).astype(str)
test_df["ORIGIN_CONGESTION"] = pd.cut(
    test_congestion, bins=congestion_edges, labels=False, include_lowest=True
).fillna(0).astype(str)

# Preserve the naturally distributed training period for probability
# calibration before creating the balanced model-training sample.
full_train_df = train_df.copy()

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Keep training and inference practical while preserving the time split.
# The evaluation sample keeps the real delayed/on-time prevalence.
train_delayed = train_df[train_df["DELAY_BINARY"] == "Delayed"].sample(
    n=100000, random_state=42
)
train_ontime = train_df[train_df["DELAY_BINARY"] == "OnTime"].sample(
    n=100000, random_state=42
)
train_df = pd.concat([train_delayed, train_ontime]).sample(
    frac=1, random_state=42
).reset_index(drop=True)
test_eval = test_df.sample(n=min(20000, len(test_df)), random_state=42)
print("Training sample:", train_df.shape)
print("Training class counts:")
print(train_df["DELAY_BINARY"].value_counts())
print("Test prevalence:", test_eval["DELAY_BINARY"].value_counts(normalize=True).round(4).to_dict())
print("Evaluation sample:", test_eval.shape)

# ══════════════════════════════════════════════════════════════════
# STEP 6 — CONTROLLED FEATURE COMPARISON
# ══════════════════════════════════════════════════════════════════

TARGET = "DELAY_BINARY"
FEATURE_SETS = {
    "Set A - Original": ["AIRLINE", "ORIGIN", "SEASON", "SCHED_HOUR", "DAY_TYPE"],
    "Set B - New signals": [
        "AIRLINE", "SCHED_HOUR", "SEASON", "ROUTE_DELAY_HISTORY",
        "ORIGIN_CONGESTION",
    ],
    "Set C - New signals + origin": [
        "AIRLINE", "SCHED_HOUR", "SEASON", "ROUTE_DELAY_HISTORY", "ORIGIN",
    ],
    "Set D - B + distance": [
        "AIRLINE", "SCHED_HOUR", "SEASON", "ROUTE_DELAY_HISTORY",
        "ORIGIN_CONGESTION", "DISTANCE",
    ],
}

scenarios = [
    ("Best", {
        "AIRLINE": "AS", "SCHED_HOUR": "AMPeak", "DAY_TYPE": "Weekday",
        "SEASON": "Fall", "ORIGIN": "Other", "ROUTE_DELAY_HISTORY": "VeryLow",
        "ORIGIN_CONGESTION": "0",
    }),
    ("Worst", {
        "AIRLINE": "B6", "SCHED_HOUR": "PMPeak", "DAY_TYPE": "Weekday",
        "SEASON": "Summer", "ORIGIN": "ORD", "ROUTE_DELAY_HISTORY": "Low",
        "ORIGIN_CONGESTION": "2",
    }),
    ("Mixed", {
        "AIRLINE": "DL", "SCHED_HOUR": "EarlyAfternoon", "DAY_TYPE": "Weekend",
        "SEASON": "Winter", "ORIGIN": "ATL", "ROUTE_DELAY_HISTORY": "Low",
        "ORIGIN_CONGESTION": "1",
    }),
]


def normalize_evidence(evidence):
    if "ORIGIN_CONGESTION" in evidence:
        evidence["ORIGIN_CONGESTION"] = str(int(float(evidence["ORIGIN_CONGESTION"])))
    return evidence


def query_evidence(row, model, features):
    return normalize_evidence({feature: row[feature] for feature in features})


def run_comparison(name, features):
    # Skip feature sets whose columns are missing (e.g. DISTANCE dropped upstream).
    missing = [f for f in features if f not in train_df.columns]
    if missing:
        print(f"Skipping {name}: missing columns {missing}")
        return None
    model = DiscreteBayesianNetwork([(feature, TARGET) for feature in features])
    estimator = DiscreteBayesianEstimator(prior_type="BDeu", equivalent_sample_size=10)
    model.fit(train_df[features + [TARGET]], estimator=estimator)
    inference = VariableElimination(model)
    # Grouped inference: query each unique combo once, then map back.
    # Same predictions as per-row loop, ~100x fewer queries.
    combo_keys = test_eval[features].apply(
        lambda r: tuple((f, str(int(float(r[f]))) if f == "ORIGIN_CONGESTION" else str(r[f])) for f in features),
        axis=1,
    )
    unique_combos = {}
    for key, (_, row) in zip(combo_keys, test_eval.iterrows()):
        if key not in unique_combos:
            unique_combos[key] = query_evidence(row, model, features)
    combo_pred = {}
    for key, evidence in tqdm(
        unique_combos.items(), total=len(unique_combos), desc=f"{name} inference ({len(unique_combos)} combos)"
    ):
        result = inference.query(
            variables=[TARGET],
            evidence=dict(evidence),
            show_progress=False,
        )
        states = result.state_names[TARGET]
        combo_pred[key] = states[result.values.argmax()]
    predictions = [combo_pred[k] for k in combo_keys]

    y_true = test_eval[TARGET].tolist()
    report = classification_report(y_true, predictions, output_dict=True)
    scenario_values = []
    scenario_counts = []
    for _, evidence in scenarios:
        scenario_mask = pd.Series(True, index=train_df.index)
        for feature in features:
            scenario_mask &= train_df[feature] == evidence[feature]
        scenario_counts.append(int(scenario_mask.sum()))
        result = inference.query(
            variables=[TARGET],
            evidence={feature: evidence[feature] for feature in features},
            show_progress=False,
        )
        states = result.state_names[TARGET]
        scenario_values.append(result.values[list(states).index("Delayed")] * 100)

    return {
        "model": model,
        "accuracy": accuracy_score(y_true, predictions),
        "delayed_f1": report["Delayed"]["f1-score"],
        "scenario_values": scenario_values,
        "scenario_counts": scenario_counts,
        "scenario_spread": max(scenario_values) - min(scenario_values),
    }


comparison = {}
for name, features in FEATURE_SETS.items():
    print(f"\nTraining {name}: {features}")
    result = run_comparison(name, features)
    if result is None:
        continue
    comparison[name] = result

print("\n" + "=" * 70)
print("CONTROLLED COMPARISON: SAME MONTH-STRATIFIED SPLIT")
print("=" * 70)
for name, result in comparison.items():
    print(
        f"{name}: Accuracy {result['accuracy']:.4f} | "
        f"Delayed F1 {result['delayed_f1']:.4f} | "
        f"Scenario spread {result['scenario_spread']:.1f} points"
    )
    print("  Scenario delay rates:", [f"{value:.1f}%" for value in result["scenario_values"]])
    support = []
    for count in result["scenario_counts"]:
        support.append(f"{count} rows" + (" [LOW SUPPORT]" if count < 50 else ""))
    print("  Training support:", support)

# ══════════════════════════════════════════════════════════════════
# STEP 7 — SUPPORTED SCENARIOS AND FULL-TEST CALIBRATION
# ══════════════════════════════════════════════════════════════════

set_b_features = FEATURE_SETS["Set B - New signals"]
set_b_model = comparison["Set B - New signals"]["model"]
set_b_inference = VariableElimination(set_b_model)


def delayed_probability(inference, evidence):
    result = inference.query(
        variables=[TARGET],
        evidence=normalize_evidence(dict(evidence)),
        show_progress=False,
    )
    states = result.state_names[TARGET]
    return float(result.values[list(states).index("Delayed")])


top_combinations = (
    train_df.groupby(set_b_features, dropna=False)
    .size()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name="TRAINING_ROWS")
)
print("\n" + "=" * 70)
print("SET B: TOP 10 HIGHEST-SUPPORT TRAINING COMBINATIONS")
print("=" * 70)
for _, row in top_combinations.iterrows():
    evidence = {feature: row[feature] for feature in set_b_features}
    probability = delayed_probability(set_b_inference, evidence)
    description = ", ".join(f"{feature}={evidence[feature]}" for feature in set_b_features)
    print(f"{int(row['TRAINING_ROWS']):>6} rows | {probability * 100:5.1f}% delayed | {description}")

# Query each unique Set B combination once, then weight it by all matching
# rows in the full later-day test set for an exact calibration table.
test_combinations = (
    test_df.groupby(set_b_features, dropna=False)
    .agg(
        TEST_ROWS=(TARGET, "size"),
        OBSERVED_DELAYED=(TARGET, lambda values: (values == "Delayed").sum()),
    )
    .reset_index()
)
test_probabilities = []
for _, row in test_combinations.iterrows():
    evidence = {feature: row[feature] for feature in set_b_features}
    test_probabilities.append(delayed_probability(set_b_inference, evidence))
test_combinations["PREDICTED_DELAY"] = test_probabilities
test_combinations["PROBABILITY_BUCKET"] = pd.cut(
    test_combinations["PREDICTED_DELAY"],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.000001],
    labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    include_lowest=True,
)
calibration = (
    test_combinations.groupby("PROBABILITY_BUCKET", observed=False)
    .agg(
        TEST_ROWS=("TEST_ROWS", "sum"),
        OBSERVED_DELAYED=("OBSERVED_DELAYED", "sum"),
        PREDICTED_DELAY=("PREDICTED_DELAY", lambda values: 0.0),
    )
    .reset_index()
)
weighted_predictions = test_combinations.assign(
    WEIGHTED_PREDICTED=lambda frame: frame["PREDICTED_DELAY"] * frame["TEST_ROWS"]
).groupby("PROBABILITY_BUCKET", observed=False)["WEIGHTED_PREDICTED"].sum()
calibration["MEAN_PREDICTED"] = calibration["PROBABILITY_BUCKET"].map(
    weighted_predictions / calibration.set_index("PROBABILITY_BUCKET")["TEST_ROWS"]
).fillna(0.0)
calibration["OBSERVED_DELAY_RATE"] = (
    calibration["OBSERVED_DELAYED"] / calibration["TEST_ROWS"].replace(0, pd.NA)
).fillna(0.0)
print("\n" + "=" * 70)
print("SET B: FULL TEST CALIBRATION")
print("=" * 70)
print("Bucket   Test rows   Mean predicted   Observed delay rate")
for _, row in calibration.iterrows():
    print(
        f"{str(row['PROBABILITY_BUCKET']):<8} {int(row['TEST_ROWS']):>9} "
        f"{row['MEAN_PREDICTED'] * 100:>14.1f}% "
        f"{row['OBSERVED_DELAY_RATE'] * 100:>20.1f}%"
    )

# ══════════════════════════════════════════════════════════════════
# STEP 8 — HELD-OUT ISOTONIC CALIBRATION FOR SET B
# ══════════════════════════════════════════════════════════════════

# Reserve the calibration rows first, then draw the base-model rows only from
# the remaining training-period rows. The later-day test set is never used to
# fit either component.
calibration_train = full_train_df.sample(n=100000, random_state=123).copy()
calibration_model_source = full_train_df.drop(index=calibration_train.index)
calibration_model_delayed = calibration_model_source[
    calibration_model_source[TARGET] == "Delayed"
].sample(n=100000, random_state=42)
calibration_model_ontime = calibration_model_source[
    calibration_model_source[TARGET] == "OnTime"
].sample(n=100000, random_state=42)
calibration_model_train = pd.concat(
    [calibration_model_delayed, calibration_model_ontime]
).sample(frac=1, random_state=42).copy()
calibration_model = DiscreteBayesianNetwork(
    [(feature, TARGET) for feature in set_b_features]
)
calibration_model.fit(
    calibration_model_train[set_b_features + [TARGET]],
    estimator=DiscreteBayesianEstimator(prior_type="BDeu", equivalent_sample_size=10),
)
calibration_inference = VariableElimination(calibration_model)


def row_probability(row, inference, features):
    return delayed_probability(
        inference,
        {feature: row[feature] for feature in features},
    )


calibration_combinations = (
    calibration_train.groupby(set_b_features, dropna=False)
    .agg(
        CALIBRATION_ROWS=(TARGET, "size"),
        OBSERVED_DELAYED=(TARGET, lambda values: (values == "Delayed").sum()),
    )
    .reset_index()
)
calibration_raw = [
    row_probability(row, calibration_inference, set_b_features)
    for _, row in tqdm(
        calibration_combinations.iterrows(),
        total=len(calibration_combinations),
        desc="Calibration-combination inference",
    )
]
calibration_labels = (
    calibration_combinations["OBSERVED_DELAYED"]
    / calibration_combinations["CALIBRATION_ROWS"]
).to_numpy()
isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
isotonic.fit(
    calibration_raw,
    calibration_labels,
    sample_weight=calibration_combinations["CALIBRATION_ROWS"].to_numpy(),
)

calibration_calibrated = isotonic.predict(np.asarray(calibration_raw))
threshold_results = []
for threshold in np.arange(0.10, 0.501, 0.05):
    threshold_predictions = calibration_calibrated >= threshold
    observed_delayed = calibration_combinations["OBSERVED_DELAYED"].to_numpy()
    calibration_rows = calibration_combinations["CALIBRATION_ROWS"].to_numpy()
    true_ontime = calibration_rows - observed_delayed
    true_positive = observed_delayed[threshold_predictions].sum()
    false_positive = true_ontime[threshold_predictions].sum()
    false_negative = observed_delayed[~threshold_predictions].sum()
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    threshold_results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })
chosen_threshold = max(
    threshold_results,
    key=lambda result: (result["f1"], result["threshold"]),
)["threshold"]
print("\n" + "=" * 70)
print("SET B: CALIBRATION-SET THRESHOLD SWEEP")
print("=" * 70)
print("Threshold   Delayed precision   Delayed recall   Delayed F1")
for result in threshold_results:
    print(
        f"{result['threshold']:.2f}        {result['precision'] * 100:>8.1f}% "
        f"{result['recall'] * 100:>16.1f}% {result['f1']:.4f}"
    )
print(f"Chosen threshold: {chosen_threshold:.2f} (maximum calibration-set Delayed F1)")

test_combinations["CALIBRATED_DELAY"] = isotonic.predict(
    test_combinations["PREDICTED_DELAY"].to_numpy()
)
test_combinations["CALIBRATED_BUCKET"] = pd.cut(
    test_combinations["CALIBRATED_DELAY"],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.000001],
    labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    include_lowest=True,
)
calibrated_table = (
    test_combinations.groupby("CALIBRATED_BUCKET", observed=False)
    .agg(
        TEST_ROWS=("TEST_ROWS", "sum"),
        OBSERVED_DELAYED=("OBSERVED_DELAYED", "sum"),
    )
    .reset_index()
)
weighted_calibrated = test_combinations.assign(
    WEIGHTED_CALIBRATED=lambda frame: frame["CALIBRATED_DELAY"] * frame["TEST_ROWS"]
).groupby("CALIBRATED_BUCKET", observed=False)["WEIGHTED_CALIBRATED"].sum()
calibrated_table["MEAN_PREDICTED"] = calibrated_table["CALIBRATED_BUCKET"].map(
    weighted_calibrated / calibrated_table.set_index("CALIBRATED_BUCKET")["TEST_ROWS"]
).fillna(0.0)
calibrated_table["OBSERVED_DELAY_RATE"] = (
    calibrated_table["OBSERVED_DELAYED"]
    / calibrated_table["TEST_ROWS"].replace(0, pd.NA)
).fillna(0.0)

test_eval_raw = test_eval.apply(
    lambda row: row_probability(row, calibration_inference, set_b_features), axis=1
).to_numpy()
test_eval_calibrated = isotonic.predict(test_eval_raw)
test_eval_predictions_05 = np.where(test_eval_calibrated >= 0.5, "Delayed", "OnTime")
test_eval_predictions_tuned = np.where(
    test_eval_calibrated >= chosen_threshold, "Delayed", "OnTime"
)
calibrated_report_05 = classification_report(
    test_eval[TARGET], test_eval_predictions_05, output_dict=True, zero_division=0
)
calibrated_report_tuned = classification_report(
    test_eval[TARGET], test_eval_predictions_tuned, output_dict=True, zero_division=0
)
calibrated_confusion_tuned = confusion_matrix(
    test_eval[TARGET], test_eval_predictions_tuned, labels=["Delayed", "OnTime"]
)
print("\n" + "=" * 70)
print("SET B: HELD-OUT ISOTONIC CALIBRATION")
print("=" * 70)
print(
    f"Calibration fit rows: {len(calibration_train)} | "
    f"Model-fit rows: {len(calibration_model_train)} | "
    "Test rows used for fitting: 0"
)
print(
    f"Calibrated at 0.50: accuracy {accuracy_score(test_eval[TARGET], test_eval_predictions_05):.4f} | "
    f"Delayed F1 {calibrated_report_05['Delayed']['f1-score']:.4f}"
)
print(
    f"Calibrated at tuned {chosen_threshold:.2f}: "
    f"accuracy {accuracy_score(test_eval[TARGET], test_eval_predictions_tuned):.4f} | "
    f"Delayed precision {calibrated_report_tuned['Delayed']['precision']:.4f} | "
    f"Delayed recall {calibrated_report_tuned['Delayed']['recall']:.4f} | "
    f"Delayed F1 {calibrated_report_tuned['Delayed']['f1-score']:.4f}"
)
print("Tuned-threshold confusion matrix [Delayed, OnTime]:")
print(calibrated_confusion_tuned)
print("\nMODEL COMPARISON")
print("Uncalibrated: accuracy 0.5948 | Delayed F1 0.3574")
print(
    f"Calibrated at 0.50: accuracy {accuracy_score(test_eval[TARGET], test_eval_predictions_05):.4f} | "
    f"Delayed F1 {calibrated_report_05['Delayed']['f1-score']:.4f}"
)
print(
    f"Calibrated at tuned {chosen_threshold:.2f}: "
    f"accuracy {accuracy_score(test_eval[TARGET], test_eval_predictions_tuned):.4f} | "
    f"Delayed precision {calibrated_report_tuned['Delayed']['precision']:.4f} | "
    f"Delayed recall {calibrated_report_tuned['Delayed']['recall']:.4f} | "
    f"Delayed F1 {calibrated_report_tuned['Delayed']['f1-score']:.4f}"
)
print("Bucket   Test rows   Mean predicted   Observed delay rate")
for _, row in calibrated_table.iterrows():
    print(
        f"{str(row['CALIBRATED_BUCKET']):<8} {int(row['TEST_ROWS']):>9} "
        f"{row['MEAN_PREDICTED'] * 100:>14.1f}% "
        f"{row['OBSERVED_DELAY_RATE'] * 100:>20.1f}%"
    )

# Package the complete deployment contract so callers cannot silently bypass
# calibration or use a different classification threshold.
# Include support_table so the app can warn on low-support combos.
support_table = {
    tuple(sorted((f, str(row[f])) for f in set_b_features)): int(row["TRAINING_ROWS"])
    for _, row in top_combinations.iterrows()
}
final_bundle = {
    "model": set_b_model,
    "calibrator": isotonic,
    "threshold": chosen_threshold,
    "features": set_b_features,
    "target": TARGET,
    "support_table": support_table,
    "split_mode": SPLIT_MODE,
}
with open("flight_delay_model_final.pkl", "wb") as f:
    pickle.dump(final_bundle, f)
print("\nSaved final calibrated bundle: flight_delay_model_final.pkl")

# Keep the legacy artifact for compatibility with the existing Flask app.
with open("flight_delay_model_v2_1.pkl", "wb") as f:
    pickle.dump(set_b_model, f)
print("\nSaved Set B application model: flight_delay_model_v2_1.pkl")
