# Flight Delay Risk Assessment and Cascade Diagnosis Using Bayesian Networks
## Presentation Text (Review 2)

Slides follow in presentation order. No separate introduction slide; Slide 1 opens with the Review 1 recap.

---

## Slide 1: From Review 1 to Final Results

**Review 1 proposed** four goals, all built on discrete Bayesian networks (BN) trained on the 2015 US DOT flight dataset:
- Predict delay risk from airline, season, time of day, and origin
- Model cascade propagation along an aircraft's sequence of flights
- Diagnose delays backward: inherited from an earlier flight, or newly caused
- Rank unknown factors by value of information

**Review 1 reported:** 68% accuracy for the base network, a cascade effect of 48.2% vs 11.2%, 81.2% of delays inherited, and time of day as the most informative factor.

**Since Review 1 we:**
- Re-evaluated every model against baselines, using precision, recall, F1, and AUC instead of accuracy alone
- Showed that the 68% figure is close to the majority-class rate on its test set (66.7%), so it is replaced
- Corrected the backward-reasoning figure for the delay prior (81.2% → about 47–48%)
- Revised the base network's features: route delay history and origin congestion replace airport identity
- Calibrated the base network's probabilities and selected its decision threshold on held-out training data
- Validated the base network on a time-based split (train January–October, test November–December): AUC 0.64
- Built a local web interface for the base network

**How:** Every model is a discrete Bayesian network with exact inference (variable elimination). No black-box learners are used, so every probability traces to a conditional probability table that can be inspected.

---

## Slide 2: Data and Preprocessing

- **Source:** US Department of Transportation, 2015 Flight Delays and Cancellations (Kaggle: usdot/flight-delays)
- **Scale:** 5,819,079 flights, 14 airlines, 4,897 aircraft, 628 origin airports
- **Cleaning:** 89,884 cancelled flights removed; rows with missing scheduling fields dropped, leaving 5,729,195 flights
- **Target:** a flight is *Delayed* if departure delay exceeds 15 minutes, otherwise *OnTime*
- **Base rate:** roughly 1 in 6 departures is delayed (17.3–17.7% depending on subset)
- **Leakage control:** route and congestion statistics are computed from training data only; test flights are never used to estimate any parameter or threshold

---

## Slide 3: Base Bayesian Network: Structure

**Variables** (one target, five explanatory variables):

| Variable | States | Meaning |
|---|---|---|
| AIRLINE | 14 | Carrier code |
| SCHED_HOUR | 6 | Scheduled departure window (EarlyMorning, AMPeak, Midmorning, EarlyAfternoon, PMPeak, Night) |
| SEASON | 4 | Winter, Spring, Summer, Fall |
| ROUTE_DELAY_HISTORY | 4 | Historical delay rate of the origin–destination route: VeryLow (≤25%), Low (25–40%), High (40–55%), VeryHigh (>55%) |
| ORIGIN_CONGESTION | 3 | Average monthly departures at the origin airport, grouped into three bands from training data (0 = lowest, 2 = highest) |
| DELAY_BINARY | 2 | Delayed / OnTime (target) |

**Structure:** a star-shaped directed acyclic graph. Each of the five explanatory variables has a single directed edge into DELAY_BINARY, and there are no edges between explanatory variables. The structure is specified from domain knowledge (carrier, season, time of day, route history, and origin congestion all affect delay). Each cell of the target's conditional probability table therefore corresponds to one of 4,032 possible combinations (14 × 6 × 4 × 4 × 3).

---

## Slide 4: Base Bayesian Network: Parameter Learning and Inference

**Parameter learning**
- Conditional probability tables (CPTs) are estimated with a Bayesian Dirichlet equivalent uniform (BDeu) prior, equivalent sample size 10
- Bayesian estimation avoids extreme probabilities for rare feature combinations, where raw frequency counts are unreliable
- Estimation used 100,000 delayed and 100,000 on-time flights from the training period (days 1–20 in the same-month split; January–October in the time split), sampled to give a balanced learning set

**Inference**
- Exact inference by variable elimination (pgmpy)
- For any combination of observed features, the network returns P(DELAY_BINARY = Delayed | evidence)
- Every combination of observed features is answered exactly; no approximation is used

---

## Slide 5: Calibration and Decision Threshold

- The raw network was trained on balanced data, so its outputs are not true probabilities. Calibration corrects this.
- **Calibration:** isotonic regression, fitted on 100,000 held-out training flights that were excluded from parameter estimation. No test data was used.
- **Decision threshold:** 0.20, selected to maximize Delayed F1 on the calibration split (search range 0.10–0.50)
- **Calibration on the test period** (saved model, days 21–31; predicted vs observed delay rate by bin):

| Predicted bin | Flights | Mean predicted | Observed |
|---|---|---|---|
| 0.0–0.1 | 472,897 | 7.7% | 7.8% |
| 0.1–0.2 | 726,818 | 15.3% | 15.0% |
| 0.2–0.3 | 541,844 | 24.6% | 23.6% |
| 0.3–0.4 | 158,143 | 32.7% | 29.9% |
| 0.4–0.5 | 47,891 | 43.4% | 36.1% |

- Predictions are closely calibrated below 30%. Above 30%, the network over-predicts modestly.
- This table comes from the saved model, which still contains borrowed states (see Slide 16). It is not yet recomputed for the corrected model.

---

## Slide 6: Cascade Bayesian Network: Concept

- Delays propagate along an aircraft's rotation: a late arrival can delay the aircraft's next departure
- The cascade network has two binary nodes and one directed edge: **CURRENT_DELAY → NEXT_DELAY**
- Flights are linked into chains by tail number and ordered by date and scheduled departure
- The same edge applies repeatedly along an aircraft's day, so a delay on one leg can affect later legs

---

## Slide 7: Cascade Bayesian Network: Learning from Real Flight Pairs

- **Data:** 5,724,299 consecutive same-aircraft flight pairs across 4,888 aircraft, all from real records (no synthetic rows)
- **Estimated probabilities** (full-year empirical):
  - P(next delayed | current delayed) = 48.3%
  - P(next delayed | current on time) = 11.2%
- The saved CPT gives 48.2% and 11.2%, matching the empirical values
- Conditional probabilities are not affected by how the sample is balanced on the current flight; the marginal (prior) is, and that matters for backward reasoning (Slide 9)

---

## Slide 8: Forward Reasoning

**Question:** given the status of the previous flight on an aircraft, what is the risk for its next flight?

| Current flight | P(next flight delayed) | P(next flight on time) |
|---|---|---|
| Delayed | 48.2% | 51.8% |
| On time | 11.2% | 88.8% |

- A delayed flight raises the probability that the next flight on the same aircraft is delayed by a factor of 4.3
- This is the central empirical finding of the project

---

## Slide 9: Backward Diagnosis

**Question:** given that the next flight was delayed, how likely was the previous flight to have been delayed?

By Bayes' rule:

P(current delayed | next delayed) = P(next delayed | current delayed) × P(current delayed) / P(next delayed)

- Review 1 computed this under a balanced prior, P(current delayed) = 0.5, giving **81.2%**. That prior does not match real operations.
- Using the natural delay rate (about 17–18%), the same calculation gives **about 47–48%**.
- **Interpretation:** about half of delayed next-flights follow a delayed previous flight, compared with about 17% of all flights. This is consistent with inherited delay, but the analysis is associative and does not establish causation.

---

## Slide 10: Value of Information

**Method:** the expected reduction in uncertainty about delay status from learning a factor, measured in bits of entropy:

Information gain = H(Delay) − H(Delay | Factor)

- Baseline entropy: H(Delay) = 0.9415 bits

| Factor | Information gain (bits) |
|---|---|
| Time of day | 0.0209 |
| Airline | 0.0061 |
| Season | 0.0055 |
| Origin airport | 0.0011 |
| Day type | 0.0003 |

- Time of day is about 3.4 times more informative than the next factor, airline. It is the most useful single piece of information to obtain about an unknown flight.
- These values were computed on the exploratory network from the notebook, not the final base network (see Slide 16).

---

## Slide 11: Evaluation Protocol

- **Base network, time split (primary):** trained on January–October (4,794,655 flights); tested on November–December (934,540 flights, 17.26% delayed). Route and congestion statistics come from January–October only.
- **Base network, same-month split (secondary):** trained on days 1–20; tested on days 21–31 (1,947,593 flights, 17.36% delayed). Training and test cover the same calendar months, so this split is more optimistic.
- **Cascade network:** evaluated on pairs whose current flight falls in November–December (930,055 pairs) and on the full year (5,724,299 pairs)
- **Metrics:** precision, recall, and F1 for the Delayed class; AUC for the base network with 95% bootstrap intervals; accuracy reported for completeness
- **Baselines:** always predict On-time; always predict Delayed
- **Lift:** precision divided by the base delay rate of the evaluation set

---

## Slide 12: Results: Base Bayesian Network

| Model | Accuracy | Precision | Recall | F1 | AUC (95% CI) |
|---|---|---|---|---|---|
| **Base BN, time split** (train Jan–Oct, test Nov–Dec), threshold 0.20 | 67.3% | 25.1% | 45.1% | 0.322 | 0.638 (0.636–0.639) |
| Base BN, same-month split (train days 1–20, test days 21–31), threshold 0.20 | 65.0% | 26.0% | 55.0% | 0.353 | 0.658 (0.657–0.659) |
| Baseline: always On-time (Nov–Dec) | 82.7% | – | 0% | 0.000 | 0.500 |
| Baseline: always Delayed (Nov–Dec) | 17.3% | 17.3% | 100% | 0.294 | 0.500 |

**Confusion matrix, time split:** true positives 72,706; false positives 217,106; false negatives 88,590; true negatives 556,138.

- **AUC 0.638 on the time split:** the network ranks delayed flights above on-time flights more often than chance (0.5). The same-month figure (0.658) is slightly higher, as expected when the test months are seen in training.
- **Precision 25.1%:** about 1.5 times the base delay rate of 17.3%
- **Recall 45.1%:** at the chosen threshold, the network flags about 45% of delays
- **F1 0.322 vs 0.294:** the network's gain over always predicting Delayed is modest
- **Accuracy is not the headline metric:** always predicting On-time scores higher, which is why precision, recall, F1, and AUC are reported instead

---

## Slide 13: Results: Cascade Bayesian Network

**Rule evaluated:** predict the next flight is delayed if and only if the current flight is delayed. With two states, every threshold between 0.12 and 0.48 produces this same rule, so the threshold was not tuned on test data.

| Evaluation set | Accuracy | Precision | Recall | F1 | Base rate |
|---|---|---|---|---|---|
| Nov–Dec (930,055 pairs) | 82.5% | 49.2% | 49.1% | 0.492 | 17.3% |
| Full year (5,724,299 pairs) | 81.7% | 48.3% | 48.3% | 0.483 | 17.7% |
| Baseline: always On-time (Nov–Dec) | 82.7% | 0% | 0% | 0.000 | – |

- Precision is about 2.8 times the base rate (Nov–Dec) and 2.7 times (full year)
- Recall of 49% means the rule identifies about half of all next-flight delays
- Accuracy is slightly below the always-On-time baseline, which reflects the class imbalance; F1 and precision show the gain
- This rule is a diagnostic finding rather than a standalone predictor, because it uses only the previous flight's status

---

## Slide 14: Summary: Review 1 Claims vs Final Results

| Claim | Review 1 | Final | Status |
|---|---|---|---|
| Base network performance | 68% accuracy | Time split: accuracy 67.3%, AUC 0.638, F1 0.322 | Replaced: 68% was near the 66.7% majority rate |
| Cascade effect | 48.2% vs 11.2% | 48.2% vs 11.2% (4.3×) | Confirmed |
| Inherited delays (backward) | 81.2% | About 47–48% at natural delay rate | Corrected: 81.2% reflects a balanced prior |
| Value of information | Time of day, 0.0209 bits | Time of day, 0.0209 bits (exploratory network) | Confirmed; to re-run on final network |
| Cascade classification | Not reported | Precision 49.2%, recall 49.1%, F1 0.49 | Added with baseline |

---

## Slide 15: Application

- A local Flask web interface serves the base network
- The interface returns the calibrated delay probability and the decision at threshold 0.20
- Automated tests confirm valid predictions (for example, American Airlines, PM peak, summer, very high route history, congestion level 2: 33.6% delay probability, classified as high risk) and correct handling of unknown or missing inputs
- **Current limitation:** the form asks users for origin congestion and route delay history as abstract levels with no definitions. The saved model behind the form still contains borrowed states (see Slide 16). The planned replacement is to ask for airport, month, and route, and to look up these features from training data.
- The cascade network is not yet in the interface (see Slide 17)

---

## Slide 16: Limitations

- **Scope:** one year of data (2015); no weather; no network-level effects such as hub congestion or crew connections
- **Evaluation:** the time-based split (AUC 0.638) is the primary result. The same-month split (AUC 0.658) is more optimistic because training and test share calendar months.
- **Saved model behind the web form:** route states High and VeryHigh were copied from Low, congestion levels 3 and 4 were copied from level 2, and levels 0 and 1 gave a flat 0.5 in the contexts tested. The results in this deck come from a refit with airport-based congestion and do not use these states. The web model has not yet been rebuilt.
- **Congestion bands are coarse:** bands are defined by airport count, so 97% of test flights fall in the highest band
- **Sparse configurations:** where a combination of features has no training data, the prior returns 0.5. Those values are not learned from data.
- **Cascade sample:** the composition of the sample used to estimate the cascade CPT has not been verified to exclude the test months (the full-year result is nearly identical)
- **Backward and value-of-information analyses:** based on the exploratory network and the training prior; to be re-run on the final models
- **Literature comparison:** published Bayesian network results often report mean absolute error in minutes, which is not directly comparable to the classification metrics used here

---

## Slide 17: Future Work

- Rebuild the web model with airport-based congestion and route-based inputs, and report support counts for every combination
- Weight congestion bands by flight volume rather than airport count
- Add weather (for example, NOAA METAR) and inbound-flight delay as nodes
- Connect the base and cascade models in a single network
- Extend to multi-year data (2016–2019)
- Add cascade and backward queries to the web interface

---

## Slide 18: Conclusion

- Bayesian networks give transparent, queryable reasoning about flight delays, with every probability traceable to a conditional probability table
- Cascade propagation is strongly supported by the data: a delayed flight raises the next flight's delay probability by about 4.3 times
- The base network shows modest but real discrimination (AUC 0.64 on a time-based split) and is best used as an interpretable baseline, not a stand-alone operational predictor
- This review corrects several Review 1 figures and reports every model against explicit baselines

---

## Slide 19: References

1. Flight Delay and Delay Propagation Analysis Based on Bayesian Network. ResearchGate.
2. Characterization of delay propagation in the air traffic network. ScienceDirect, 2021.
3. Wu, C., & Law, K. (2019). Modelling the delay propagation effects of multiple resource connections in an airline network using a Bayesian network model. *Transportation Research Part E*, 122, 62–77.
4. A Review of Research on Flight Delay Propagation: Current Situation and Prospect. *Journal of Advanced Transportation*, Wiley, 2025.
5. US Department of Transportation. 2015 Flight Delays and Cancellations dataset. Kaggle: usdot/flight-delays.
6. Heckerman, D., Geiger, D., & Chickering, D. M. (1995). Learning Bayesian networks: The combination of knowledge and statistical data. *Machine Learning*, 20, 197–243.
7. Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *Proceedings of KDD*.
8. Ankan, A., & Panda, A. (2015). pgmpy: Probabilistic graphical models using Python. *Proceedings of SciPy*.

*Items 1–4 are taken from the Review 1 slides; verify full bibliographic details before submission. Items 6–8 are standard methodological references added for this review; check formatting against your department's style.*

---

# Backup Slides

## Backup A: Sensitivity to the BDeu Prior

**Question:** how sensitive is the model to the equivalent sample size (ESS) of the BDeu prior?

| ESS | AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 5 | 0.657 | 0.268 | 26.0% | 54.9% | 0.353 |
| 10 (used) | 0.657 | 0.268 | 26.0% | 54.9% | 0.353 |
| 50 | 0.657 | 0.268 | 26.0% | 54.9% | 0.353 |

*Same-month split. The sweep ran before the congestion fix; the ESS 10 result is unchanged after the fix (AUC 0.6575).*

- Results are identical to three decimals across ESS 5, 10, and 50
- **Why:** BDeu spreads the equivalent sample size across every parent configuration and state. With 4,032 configurations, ESS 10 gives each cell a pseudo-count of about 0.001, against roughly 50 flights per configuration in the learning sample. The prior barely changes the estimates.
- **Implication:** the prior does not drive the results. The earlier explanation that balancing is needed to stop the prior from being overwhelmed is not the reason (see Backup C).
- **Caveat:** where a configuration has no training data, the estimate falls back to a flat 0.5. Those values reflect absence of data, not learned probabilities.

## Backup B: Threshold Selection and Cost

**Question:** why 0.20, and did we consider cost-sensitive decisions?

- **Why 0.20:** it maximizes Delayed F1 on the calibration split (searched 0.10–0.50). F1 treats false positives and false negatives as equally costly and is not a cost measure.
- **Cost-sensitive view:** expected cost per flight = (r × missed delays + false alarms) ÷ flights, where r is the cost of a missed delay relative to a false alarm. Results on the time split (test set, for illustration):

| Cost ratio (missed : false alarm) | Best threshold | Cost at best | Cost at 0.20 | Always On-time | Always Delayed |
|---|---|---|---|---|---|
| 1 : 1 | 0.45 | 0.173 | 0.327 | 0.173 | 0.827 |
| 2 : 1 | 0.45 | 0.345 | 0.422 | 0.345 | 0.827 |
| 5 : 1 | 0.14 | 0.677 | 0.706 | 0.863 | 0.827 |

- At equal costs, always predicting On-time is as good as the model, and 0.20 is worse than both. The model helps only when a missed delay costs about five times a false alarm.
- **Recommendation:** choose the threshold from the stakeholder's cost ratio, using the calibration split rather than the test set. Present 0.20 as an F1 choice, not a cost-optimal one.
- Cost curves: `FINAL_RESULTS/robustness/cost_curve.png`

## Backup C: Why Balanced Sampling for Parameter Learning

**Question:** why 100,000 delayed and 100,000 on-time flights instead of all training flights?

- **What was done:** the learning sample is balanced across outcomes. The calibration flights are excluded from it.
- **Why:** balancing keeps the learning sample manageable and gives each outcome comparable support in the CPT. Isotonic calibration on natural-prevalence held-out data then corrects the shift in the outcome prior that balancing introduces.
- **What it is not:** it is not needed to stop the BDeu prior from being overwhelmed. Backup A shows the prior's influence is negligible.
- **Open question:** fitting on all training flights has not yet been compared. Because the prior has little effect, we expect only small changes, but this should be checked before claiming it.

## Backup D: The Weather Question

**Question:** flight delays depend heavily on weather. Why is weather not in the model?

- **Answer:** the 2015 DOT dataset contains no weather observations. Weather is acknowledged as a limitation and as future work (NOAA METAR, Slide 17).
- **Why the dataset's weather-delay field cannot be used:** it records minutes attributed to weather after a flight is delayed, so it is not known when a prediction is made.
- **What the model uses instead, as proxies:**
  - **Season:** monthly delay rates range from 11.9% (October) to 22.4% (June), consistent with seasonal weather patterns (summer storms, winter conditions). This is correlational.
  - **Departure window:** the delay rate is 10.8% in the morning and 25.2% in the evening, consistent with disruptions building up through the day.
  - **Route history:** captures routes with persistently high delay rates, which includes weather-exposed routes but also operational causes.
- **Honest statement:** these proxies capture part of the seasonal and daily pattern that weather produces. They do not measure weather, and the model cannot answer weather-specific questions.

## Backup E: Why a Bayesian Network and Not a Black-Box Classifier?

**Question:** a gradient-boosted or other black-box classifier would likely score higher on accuracy. Why use a Bayesian network?

- **Choice:** interpretability and queryability. A Bayesian network exposes its conditional probability tables and supports forward queries, backward diagnosis, and value-of-information ranking. A classifier does not expose any of this.
- **Answer:** this project commits to pure probabilistic reasoning throughout. We accept the network's modest discrimination (AUC 0.638 on the time split) in exchange for every probability being traceable to an inspectable conditional probability table, exact (not approximate) inference, and the ability to answer forward, backward, and information-value queries that a classifier cannot.

## Backup F: Confidence Intervals and ROC / Precision-Recall Curves

| Split | AUC (95% CI) | PR-AUC | F1 (95% CI) |
|---|---|---|---|
| Time split (Jan–Oct / Nov–Dec) | 0.638 (0.636–0.639) | 0.245 | 0.322 (0.321–0.324) |
| Same-month split (days 1–20 / 21–31) | 0.658 (0.657–0.659) | 0.268 | 0.353 (0.352–0.354) |

- Intervals come from 150 bootstrap resamples of the test flights.
- The intervals are narrow because the test sets are large. They capture sampling noise only, not the shift between training and test periods, which is why the time split is the primary result.
- PR-AUC of 0.245 compares with a chance level equal to the delay rate of 0.173.
- Curves: `FINAL_RESULTS/robustness/roc_pr_curves.png`

## Backup G: Time-Based Validation

- **Procedure:** train on January–October (4,794,655 flights), test on November–December (934,540 flights). Route and congestion statistics come from January–October only. The calibration split is drawn from the training months.

| Measure | Same-month split | Time split |
|---|---|---|
| AUC | 0.658 | 0.638 |
| F1 (Delayed) | 0.353 | 0.322 |
| Precision | 26.0% | 25.1% |
| Recall | 55.0% | 45.1% |

- **Interpretation:** AUC falls by about 0.02 when the test months are unseen, which is the expected cost of seasonal shift. The model still discriminates above chance.
- **Correction:** an earlier time-split run gave AUC 0.527. That run defined congestion by origin airport and month, so every November–December flight fell in congestion level 0. The result is superseded and should not be cited.

---

## Appendix: Source of Each Number

| Number | Source file |
|---|---|
| 5,819,079 flights; delay rates; airline, month, day, time, origin tables | `archive(1)/flights.csv` (computed; `analysis_scripts/eda_numbers.py`) |
| 5,724,299 cascade pairs; 48.3% / 11.2% | `archive(1)/flights.csv` (computed) |
| Time-split and same-month results (Slides 11–12, Backups A, F, G) | `robustness/robustness_results.json`, produced by `analysis_scripts/robustness.py` |
| Bootstrap confidence intervals (Backup F) | `robustness/robustness_results.json` (`bootstrap_ess10`) |
| Cost curve (Backup B) | `robustness/robustness_results.json` (`cost`), `robustness/cost_curve.png` |
| ROC and PR curves (Backup F) | `robustness/roc_pr_curves.png` |
| Saved-model metrics and calibration table (Slide 5) | `analysis_scripts/run_base_bn_as_is.py` on `flight_delay_model_final.pkl` |
| Decision threshold 0.20 | `robustness/robustness_results.json` (`threshold`) and `flight_delay_model_final.pkl` |
| Cascade CPT 0.482 / 0.112 | `cascade_model.pkl` |
| Cascade rule metrics (Slide 13) | `analysis_scripts/cascade_threshold.py` |
| Backward 81.2% and natural-prior 47–48% | Derived from the cascade CPT with prior 0.5 and with the observed delay rate |
| Value of information (Slide 10) | Review 1 notebook `notebookacef6915e7.ipynb` (exploratory network) |
| Review 1 claims (Slides 1, 14) | `Flight_Delay_BN_Review1_2.pptx` |
| Application test results (Slide 15) | `app.py` with `flight_delay_predict.py`, tested via the Flask test client |
