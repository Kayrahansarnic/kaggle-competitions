# Predicting Airline Satisfaction · Playground S6E10

[Competition page](https://www.kaggle.com/competitions/playground-series-s6e10) · October 2026 · binary classification · ROC AUC

**Result:** private leaderboard **0.96107**, rank **1729 / 3747**. 1st place scored 0.96318.

## Data

- 700K synthetic training rows, 21 features: 13 service ratings (0–5), age, flight distance, departure/arrival delays, customer type, travel type, class, gender.
- Target `satisfaction`, 44% positive.
- The synthetic data was generated from the [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) dataset (130K rows), which I also tried as extra training data.

## Validation

5-fold `StratifiedKFold` with a fixed seed, the same folds for every experiment. Target encodings are fitted only on the training part of each fold (with smoothing, `m=20`), so the validation score never sees its own labels.

## Experiment log

All runs are LightGBM unless noted. "Kept" means the change carried forward into the next experiments.

| # | Experiment | CV AUC | Public LB | Kept |
|---|---|---|---|---|
| 1 | Drop `Gender` | 0.95846 | | ✓ |
| 2 | Add original dataset as extra training rows | 0.95821 | | ✗ |
| 3 | Original dataset + `is_orig` flag | 0.95849 | | ✓ |
| 4 | Rating aggregates (sum, mean, zero count, low count) | 0.95830 | | ✗ |
| 5 | A: `num_leaves=127` | 0.95872 | | ✓ |
| 6 | B: ratings duplicated as categoricals | 0.95871 | | ✗ |
| 7 | C: route popularity (`Flight Distance` frequency) | 0.95947 | 0.95876 | ✓ |
| 8 | Drop original dataset, keep route popularity | 0.95941 | | ✓ (simpler, same score) |
| 9 | D: target-encode `Flight Distance` | 0.96014 | 0.95970 | ✓ |
| 10 | E: + target-encode route × class | **0.96022** | | ✓ best single model |
| 11 | F: + target-encode route × travel type | 0.96024 | | ✗ (improved only 2 of 5 folds) |

### Final submission

Same features as experiment E, trained with three models, then blended by averaging ranks:

| Model | CV AUC |
|---|---|
| LightGBM | 0.96022 |
| XGBoost | 0.96035 |
| CatBoost | 0.96019 |
| **Equal-weight rank blend** | **0.96063** |

I also tried 2-model and weighted blends. The equal 3-way blend scored best on out-of-fold predictions.

## What I learned

- **`Flight Distance` behaves like a route ID.** The same distances repeat thousands of times (the value 594 appears 12,062 times), so each distance is effectively a specific route. Counting and target-encoding it (experiments 7–10) gave about +0.0015 AUC, the biggest jump in the log and about four times what the 3-model blend added (+0.0004).
- **Not every gain is real.** Experiment F was +0.00002 on average but only helped 2 of 5 folds, so I treated it as noise.
- **Extra data isn't automatically useful.** The original dataset made no measurable difference (+0.00003 with a flag, worse without one). Once route features were in, dropping it cost only 0.00006, so I dropped it to keep the pipeline simpler.

## Files

- `airline-satisfaction.ipynb`: the notebook as submitted. Experiments that weren't used in the final run are commented out, with their CV score next to them.

XGBoost and CatBoost were trained on a Kaggle GPU (`device="cuda"`, `task_type="GPU"`). Without a GPU, change those to CPU.
