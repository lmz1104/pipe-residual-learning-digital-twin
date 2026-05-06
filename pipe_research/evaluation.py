from pathlib import Path
import pandas as pd


def rmse_reduction_percent(metrics_df: pd.DataFrame, baseline: str = "Physical baseline", target: str = "RandomForest correction") -> float:
    b = float(metrics_df.loc[metrics_df["Model"] == baseline, "RMSE"].iloc[0])
    t = float(metrics_df.loc[metrics_df["Model"] == target, "RMSE"].iloc[0])
    return (b - t) / b * 100.0


def r2_delta(metrics_df: pd.DataFrame, baseline: str = "Physical baseline", target: str = "RandomForest correction") -> float:
    b = float(metrics_df.loc[metrics_df["Model"] == baseline, "R2"].iloc[0])
    t = float(metrics_df.loc[metrics_df["Model"] == target, "R2"].iloc[0])
    return t - b


def write_result_summary(metrics_df: pd.DataFrame, output_path: Path) -> None:
    reduction = rmse_reduction_percent(metrics_df)
    delta = r2_delta(metrics_df)
    rf = metrics_df[metrics_df["Model"] == "RandomForest correction"].iloc[0]
    baseline = metrics_df[metrics_df["Model"] == "Physical baseline"].iloc[0]

    text = f"""# Result Summary

> Note: results are generated from pseudo-observed synthetic data for a controlled residual-learning demonstration. They should be described as simulation results, not industrial field-test results.

## Main comparison

| Model | RMSE / Pa | R² |
|---|---:|---:|
| Physical baseline | {baseline['RMSE']:.2f} | {baseline['R2']:.4f} |
| RF residual correction | {rf['RMSE']:.2f} | {rf['R2']:.4f} |

## One-sentence description for CV / README

Random-Forest residual learning reduced RMSE by **{reduction:.2f}%** compared with the mechanistic baseline, with R² improving by **{delta:.4f}** to **{rf['R2']:.4f}** on the held-out test set.

## Recommended figures

1. `true_vs_pred.png`: observed vs predicted pressure drop.
2. `residual_histogram.png`: residual distribution before/after correction.
3. `metrics_bar_R2.png` and `metrics_bar_RMSE.png`: quantitative comparison.
4. `feature_importance.png`: interpretable feature importance of the RF residual model.
"""
    output_path.write_text(text, encoding="utf-8")
