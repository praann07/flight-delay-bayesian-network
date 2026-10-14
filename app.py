import logging
import warnings

warnings.filterwarnings("ignore")

from flask import Flask, render_template, request

import flight_delay_predict as bn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES = bn.BUNDLE["features"]
THRESHOLD = float(bn.BUNDLE["threshold"])

AIRLINES = {
    "AA": "American", "AS": "Alaska", "B6": "JetBlue", "DL": "Delta",
    "EV": "Atlantic Southeast", "F9": "Frontier", "HA": "Hawaiian",
    "MQ": "American Eagle", "NK": "Spirit", "OO": "Skywest",
    "UA": "United", "US": "US Airways", "VX": "Virgin America", "WN": "Southwest",
}
AIRLINE_CODES = sorted(AIRLINES)
SEASONS = ["Winter", "Spring", "Summer", "Fall"]
SCHED_HOURS = ["EarlyMorning", "AMPeak", "Midmorning", "EarlyAfternoon", "PMPeak", "Night"]
ROUTE_DELAY_HISTORY = ["VeryLow", "Low", "High", "VeryHigh"]
ORIGIN_CONGESTION_LEVELS = ["0", "1", "2", "3", "4"]

app = Flask(__name__)


def build_evidence(form_data):
    return {key: form_data[key] for key in FEATURES if form_data.get(key)}


def query_risk(evidence):
    if set(evidence) != set(FEATURES):
        missing = [f for f in FEATURES if f not in evidence]
        return {"error": f"Missing features: {missing}"}
    try:
        payload = bn.predict(evidence)
    except Exception as exc:
        logger.exception("BN inference failed for %s", evidence)
        return {"error": f"Inference failed: {exc}"}
    p = payload["calibrated_probability"]
    return {
        "raw_probability": payload["raw_probability"],
        "calibrated_probability": p,
        "decision": payload["decision"],
        "delayed_percent": round(p * 100, 1),
        "ontime_percent": round((1.0 - p) * 100, 1),
        "model": "Bayesian Network",
    }


@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "model": "Bayesian Network",
        "features": FEATURES,
        "target": bn.BUNDLE["target"],
        "threshold": THRESHOLD,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = {}
    if request.method == "POST":
        form_data = {f: request.form.get(f) or request.form.get(f.lower()) for f in FEATURES}
        raw_result = query_risk(build_evidence(form_data))
        if "error" in raw_result:
            form_data["form_error"] = raw_result["error"]
        else:
            result = raw_result

    return render_template(
        "index.html",
        airlines=AIRLINES,
        airline_codes=AIRLINE_CODES,
        seasons=SEASONS,
        sched_hours=SCHED_HOURS,
        route_delay_history_options=ROUTE_DELAY_HISTORY,
        origin_congestion_levels=ORIGIN_CONGESTION_LEVELS,
        result=result,
        evidence=form_data,
        threshold=THRESHOLD,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
