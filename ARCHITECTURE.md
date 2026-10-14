# Flight Delay Prediction — Architecture & Results

## Summary

Pure probabilistic reasoning: every prediction, calibration, and diagnostic query in this project comes from a discrete Bayesian network with exact inference. No black-box machine learning model is used or served.

| Metric | Base BN (time split, primary) | Base BN (same-month split) |
|---|---|---|
| Split | Jan–Oct train / Nov–Dec test (true temporal) | Day ≤20 train / >20 test (same-month) |
| Training | Balanced 100k/100k, calibrated on held-out natural-prevalence data | Balanced 100k/100k |
| Features | 5 | 5 |
| Model | Star-structured discrete BN | Star-structured discrete BN |
| AUC-ROC | 0.638 (0.636–0.639) | 0.658 (0.657–0.659) |
| Accuracy | 67.3% | 65.0% |
| Delayed F1 | 0.322 | 0.353 |
| Serving | VariableElimination per query | VariableElimination per query |

## System

### Data & Split
- **True temporal split (primary):** Jan–Oct 2015 train (4.79M rows), Nov–Dec 2015 test (934k rows)
- **Secondary same-month split:** days 1–20 train, days 21–31 test — more optimistic, used for sensitivity checks
- Route and congestion statistics are computed from training data only

### Features (5 total, all available at prediction time)
1. `AIRLINE` — carrier code (14 states)
2. `SCHED_HOUR` — departure time bucket (6 states)
3. `SEASON` — Winter/Spring/Summer/Fall
4. `ROUTE_DELAY_HISTORY` — historical route delay rate (4 bins, train-only)
5. `ORIGIN_CONGESTION` — historical airport congestion (3 bands, train-only)

### Model
- **Structure:** star-shaped DAG — each feature has a single edge into `DELAY_BINARY`, specified from domain knowledge
- **Parameter learning:** CPTs estimated with a BDeu prior (equivalent sample size 10) on a balanced 100k/100k learning sample
- **Inference:** exact inference by variable elimination (pgmpy) — every query is answered exactly, no approximation
- **Calibration:** isotonic regression on held-out natural-prevalence training data; decision threshold 0.20 (maximizes Delayed F1)

### Serving
- `flight_delay_predict.py` loads the saved bundle (`flight_delay_model_final.pkl`) and runs `VariableElimination` per query
- Result cache for repeated evidence combinations
- `app.py` — Flask web interface exposing the calibrated probability and threshold decision

### Cascade Network
- Separate two-node BN: `CURRENT_DELAY → NEXT_DELAY`, estimated from 5,724,299 consecutive same-aircraft flight pairs
- P(next Delayed | current Delayed) = 48.2–48.3% vs P(next Delayed | current OnTime) = 11.2% — a 4.3× risk multiplier
- Used for forward reasoning and backward diagnosis (Bayes' rule), not yet wired into the web interface

## Files
- `train_model.py` — base network training pipeline
- `run_bn_outputs.py` — cascade/forward/backward query runner
- `app.py` — Flask serving (Bayesian network only)
- `templates/index.html` — UI with 5-feature form
- `flight_delay_model_final.pkl` — production bundle (base network, calibrator, threshold)
- `cascade_model.pkl` — cascade network bundle
- `robustness/robustness_results.json` — evaluation metrics, bootstrap CIs, cost curves

## Known Limitations
- 2015-only data — model will degrade on newer schedules
- No weather data (not in the DOT dataset)
- No network-level effects (hub congestion, crew connections, airport centrality)
- Congestion bands are coarse (defined by airport count, not flight volume)
- Sparse feature combinations fall back to a flat 0.5 (not learned from data)
- Accuracy is modest by design: the network trades predictive power for full transparency — every output traces to an inspectable conditional probability table

## Next Steps
1. Add weather data (NOAA METAR) as a node
2. Weight congestion bands by flight volume rather than airport count
3. Connect the base and cascade networks into a single network
4. Extend to multi-year data (2016–2019)
5. Add cascade and backward-diagnosis queries to the web interface
