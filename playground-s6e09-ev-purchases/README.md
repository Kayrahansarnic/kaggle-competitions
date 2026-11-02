# Predicting Electric Vehicle Purchases · Playground S6E9

[Competition page](https://www.kaggle.com/competitions/playground-series-s6e9) · September 2026 · binary classification · ROC AUC · my first Kaggle competition

**Result:** private leaderboard **0.94023**, rank 3053 / 3575. 1st place scored 0.94602.

## Data

- 669K synthetic training rows, 13 features: 7 numeric (age, income, commute, cars owned, charging stations near home/work, environmental concern) and 6 categorical (gender, city type, current car type, home charging, subsidy, range anxiety).
- Target `Will_Buy_EV`, 17.4% positive. That imbalance is why the metric is ROC AUC and not accuracy.

## What I did

I went step by step and measured each change on a stratified 80/20 holdout before submitting anything.

| Step | Validation AUC | Public LB |
|---|---|---|
| Constant prediction (pipeline check) | | 0.5 |
| Logistic regression, 7 numeric features, scaled | 0.8757 | 0.87451 |
| + 6 categorical features (one-hot for unordered, 0/1/2 for `Range_Anxiety_Level`) | 0.9380 | |
| LightGBM, same 17 features | 0.9411 | 0.94082 |

## What I learned

- **Data preparation mattered about 20× more than the model.** Scaling gave +0.184 AUC and the categorical columns +0.062. Switching from logistic regression to LightGBM gave +0.003.
- **Local validation was reliable.** The holdout score matched the leaderboard within 0.0012 and 0.0003. That depended on a stratified split, a fixed seed, fitting the scaler on training data only, and keeping test columns in the same order as training.
- **Encoding depends on whether categories have an order, not on how many there are.** `Range_Anxiety_Level` (Low < Medium < High) gets numbers, while `City_Type` gets one-hot columns.

## Files

- `ev-purchase-prediction.ipynb`: the notebook, with my notes on each step. (On Kaggle it's published as [House Prediction](https://www.kaggle.com/code/kayrahansarn/house-prediction).)
