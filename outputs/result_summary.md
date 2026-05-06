# Result Summary

> Note: results are generated from pseudo-observed synthetic data for a controlled residual-learning demonstration. They should be described as simulation results, not industrial field-test results.

## Main comparison

| Model | RMSE / Pa | R² |
|---|---:|---:|
| Physical baseline | 684162.24 | 0.9857 |
| RF residual correction | 324816.42 | 0.9968 |

## One-sentence description for CV / README

Random-Forest residual learning reduced RMSE by **52.52%** compared with the mechanistic baseline, with R² improving by **0.0111** to **0.9968** on the held-out test set.

## Recommended figures

1. `true_vs_pred.png`: observed vs predicted pressure drop.
2. `residual_histogram.png`: residual distribution before/after correction.
3. `metrics_bar_R2.png` and `metrics_bar_RMSE.png`: quantitative comparison.
4. `feature_importance.png`: interpretable feature importance of the RF residual model.
