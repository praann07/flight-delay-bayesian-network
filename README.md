# Flight Delay Risk Assessment and Cascade Diagnosis Using Bayesian Networks

Discrete Bayesian networks (exact inference, no black-box ML) for flight delay risk prediction and delay-cascade diagnosis, trained on the 2015 US DOT flight delays dataset.

- Full results, methodology, and backup slides: [`PPT_CONTENT.md`](PPT_CONTENT.md)
- System design: [`ARCHITECTURE.md`](ARCHITECTURE.md)

## Highlights
- Base network (5 features: airline, scheduled hour, season, route delay history, origin congestion) predicting delay risk, calibrated with isotonic regression. Time-split AUC 0.638.
- Cascade network modeling same-aircraft delay propagation: a delayed flight raises the next flight's delay probability by ~4.3x (48.2% vs 11.2%).
- Backward (Bayesian) diagnosis and value-of-information ranking over candidate features.
- Flask web app (`app.py`) serving the calibrated base network.

## Dataset
This repo does not include the raw dataset (it exceeds GitHub's file size limits). To reproduce:
1. Download "2015 Flight Delays and Cancellations" from Kaggle: `usdot/flight-delays`.
2. Place `flights.csv`, `airlines.csv`, `airports.csv` into `archive(1)/`.

## Running the app
```
pip install flask pgmpy
python app.py
```
Then open `http://127.0.0.1:5000`.

## Key files
- `train_model.py` — base network training pipeline
- `flight_delay_predict.py` — inference/serving layer
- `run_bn_outputs.py` — cascade, forward/backward, value-of-information queries
- `analysis_scripts/` — evaluation, robustness, and calibration analysis
- `flight_delay_model_final.pkl`, `cascade_model.pkl` — trained model bundles
