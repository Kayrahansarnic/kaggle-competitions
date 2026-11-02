# Kaggle Competitions

My Kaggle competition work. Each folder has the notebook I submitted from and a write-up of what I tried, what it scored, and what actually moved the score.

Kaggle profile: [kaggle.com/kayrahansarn](https://www.kaggle.com/kayrahansarn)

| Competition | Date | Metric | Private LB | Rank | Approach |
|---|---|---|---|---|---|
| [Predicting Airline Satisfaction](playground-s6e10-airline-satisfaction/) · Playground S6E10 | Oct 2026 | ROC AUC | **0.96107** | 1729 / 3747 | Route-level target encoding, LightGBM + XGBoost + CatBoost rank blend |
| [Predicting Electric Vehicle Purchases](playground-s6e09-ev-purchases/) · Playground S6E9 | Sep 2026 | ROC AUC | **0.94023** | 3053 / 3575 | First competition: logistic regression → LightGBM, measuring each change |

## What carried over between competitions

- **Trust local validation, not the leaderboard.** In both competitions my CV score tracked the public leaderboard within about 0.001, so I could test ideas locally and keep daily submissions for final candidates.
- **Data work beats model swaps.** In S6E9, scaling and encoding were worth about 20× more than switching to LightGBM. In S6E10, treating `Flight Distance` as a route ID was worth more than adding two extra models to the blend.
- **Log every experiment, including the ones that failed.** The S6E10 notebook keeps each experiment's CV score next to the code that produced it.

## Running the notebooks

The notebooks were written on Kaggle and read data from `/kaggle/input/...`. To run them elsewhere, download the data from each competition's Data page and change the paths at the top.

```bash
pip install -r requirements.txt
```

Competition data is not included in this repo.
