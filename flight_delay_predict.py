import os
import pickle

from pgmpy.inference import VariableElimination


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "FLIGHT_MODEL_PATH", os.path.join(BASE_DIR, "flight_delay_model_final.pkl")
)
with open(MODEL_PATH, "rb") as file:
    BUNDLE = pickle.load(file)

_INFERENCE = VariableElimination(BUNDLE["model"])
_CACHE = {}
_SUPPORT = BUNDLE.get("support_table", {})


def _key(evidence):
    return tuple(sorted((k, str(v)) for k, v in evidence.items()))


def predict(evidence_dict):
    """Return calibrated delay probability and the thresholded decision."""
    evidence = dict(evidence_dict)
    if "ORIGIN_CONGESTION" in evidence:
        evidence["ORIGIN_CONGESTION"] = str(int(float(evidence["ORIGIN_CONGESTION"])))

    key = _key(evidence)
    if key in _CACHE:
        return dict(_CACHE[key])

    result = _INFERENCE.query(
        variables=[BUNDLE["target"]],
        evidence=evidence,
        show_progress=False,
    )
    states = result.state_names[BUNDLE["target"]]
    raw_probability = float(result.values[list(states).index("Delayed")])
    calibrated_probability = float(BUNDLE["calibrator"].predict([raw_probability])[0])
    decision = "Delayed" if calibrated_probability >= BUNDLE["threshold"] else "OnTime"
    payload = {
        "raw_probability": raw_probability,
        "calibrated_probability": calibrated_probability,
        "decision": decision,
        "support_rows": int(_SUPPORT.get(key, 0)) if _SUPPORT else None,
    }
    _CACHE[key] = dict(payload)
    return payload