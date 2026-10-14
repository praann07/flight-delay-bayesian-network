"""
Run existing Bayesian Network model and generate comprehensive outputs
"""
import pickle
import json
import pandas as pd
from pgmpy.inference import VariableElimination
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Load trained model
MODEL_PATH = "flight_delay_model_final.pkl"
logger.info(f"Loading model from {MODEL_PATH}...")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
calibrator = bundle["calibrator"]
target = bundle["target"]
features = bundle["features"]

logger.info(f"✓ Model loaded successfully")
logger.info(f"  - Target: {target}")
logger.info(f"  - Features: {features}")
logger.info(f"  - Model nodes: {list(model.nodes())}")
logger.info(f"  - Model edges: {len(model.edges())} connections")

# Print model structure
print("\n" + "="*70)
print("BAYESIAN NETWORK STRUCTURE")
print("="*70)
print(f"Nodes ({len(model.nodes())}): {list(model.nodes())}")
print(f"\nEdges ({len(model.edges())}):")
for edge in sorted(model.edges()):
    print(f"  {edge[0]} → {edge[1]}")

# Print CPDs (Conditional Probability Distributions) summary
print("\n" + "="*70)
print("CONDITIONAL PROBABILITY DISTRIBUTIONS (CPDs)")
print("="*70)
for cpd in model.get_cpds():
    print(f"\n{cpd.variable}:")
    print(f"  Cardinality: {cpd.cardinality}")
    print(f"  Evidence: {cpd.variables[1:] if len(cpd.variables) > 1 else 'None'}")

# Load and evaluate on test data
logger.info("\nLoading test data...")
df_all = pd.read_csv("archive(1)/flights.csv", low_memory=False)
df_all = df_all[df_all["CANCELLED"] == 0].copy()
logger.info(f"Total flights: {len(df_all)}")

# Check if we have metrics saved
if "metrics" in bundle:
    print("\n" + "="*70)
    print("MODEL PERFORMANCE METRICS (from training)")
    print("="*70)
    metrics = bundle["metrics"]
    for key, value in metrics.items():
        if key != "calibration_table":
            print(f"{key:20s}: {value:.4f}")

    # Calibration table
    if "calibration_table" in metrics:
        print("\nCalibration Table:")
        print(f"{'Bin':12s} {'Count':>8s} {'Predicted':>12s} {'Observed':>12s}")
        print("-" * 45)
        for row in metrics["calibration_table"]:
            print(f"{row['bin']:12s} {row['count']:>8d} {row['mean_predicted']:>12.4f} {row['observed_rate']:>12.4f}")

# Test inference on sample cases
print("\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

inference = VariableElimination(model)

# Define test cases - using only features in the model
test_cases = [
    {
        "name": "Peak hours, high congestion, high route delay history (AA)",
        "evidence": {
            "AIRLINE": "AA",
            "SCHED_HOUR": "PMPeak",
            "SEASON": "Summer",
            "ROUTE_DELAY_HISTORY": "VeryHigh",
            "ORIGIN_CONGESTION": "3"
        }
    },
    {
        "name": "Off-peak, low congestion, low route delay history (WN)",
        "evidence": {
            "AIRLINE": "WN",
            "SCHED_HOUR": "EarlyMorning",
            "SEASON": "Spring",
            "ROUTE_DELAY_HISTORY": "VeryLow",
            "ORIGIN_CONGESTION": "0"
        }
    },
    {
        "name": "Midday, moderate congestion, moderate route delay (UA)",
        "evidence": {
            "AIRLINE": "UA",
            "SCHED_HOUR": "Midmorning",
            "SEASON": "Winter",
            "ROUTE_DELAY_HISTORY": "High",
            "ORIGIN_CONGESTION": "2"
        }
    },
    {
        "name": "Morning peak, early morning, fall, low delay history (DL)",
        "evidence": {
            "AIRLINE": "DL",
            "SCHED_HOUR": "AMPeak",
            "SEASON": "Fall",
            "ROUTE_DELAY_HISTORY": "Low",
            "ORIGIN_CONGESTION": "1"
        }
    }
]

results = []
for i, test_case in enumerate(test_cases, 1):
    print(f"\nCase {i}: {test_case['name']}")
    print("-" * 70)

    try:
        result = inference.query(
            variables=[target],
            evidence=test_case["evidence"],
            show_progress=False
        )
        states = result.state_names[target]
        raw_prob = float(result.values[list(states).index("Delayed")])
        calibrated_prob = float(calibrator.predict([raw_prob])[0])
        decision = "DELAYED" if calibrated_prob >= 0.5 else "ON-TIME"

        print(f"  Raw Probability:        {raw_prob:.4f}")
        print(f"  Calibrated Probability: {calibrated_prob:.4f}")
        print(f"  Decision:               {decision}")

        results.append({
            "case": test_case["name"],
            "raw_prob": raw_prob,
            "calibrated_prob": calibrated_prob,
            "decision": decision
        })
    except Exception as e:
        print(f"  ERROR: {e}")

# Save results
print("\n" + "="*70)
print("SAVING OUTPUTS")
print("="*70)

# Save model info to JSON
model_info = {
    "model_path": MODEL_PATH,
    "nodes": list(model.nodes()),
    "edges": list(model.edges()),
    "num_edges": len(model.edges()),
    "features": features,
    "target": target,
    "predictions": results
}

with open("bn_model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)
logger.info("✓ Saved model info to bn_model_info.json")

# Save predictions
with open("bn_predictions.json", "w") as f:
    json.dump(results, f, indent=2)
logger.info("✓ Saved predictions to bn_predictions.json")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ Model Structure: {len(model.nodes())} nodes, {len(model.edges())} edges")
print(f"✓ Features: {len(features)}")
print(f"✓ Test Cases Evaluated: {len(results)}")
print(f"✓ Outputs saved to: bn_model_info.json, bn_predictions.json")
