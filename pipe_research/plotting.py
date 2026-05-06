from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import pandas as pd


def plot_pressure_drop_curves(results: List[Dict[str, float]], save_path: Optional[str] = None) -> None:
    Q = [r["flowrate_m3_s"] for r in results]
    dp_f = [r["dp_friction_kPa"] for r in results]
    dp_l = [r["dp_local_kPa"] for r in results]
    dp_t = [r["dp_total_kPa"] for r in results]
    plt.figure(figsize=(8, 5))
    plt.plot(Q, dp_f, label="Friction loss")
    plt.plot(Q, dp_l, label="Local loss")
    plt.plot(Q, dp_t, label="Total loss")
    plt.xlabel("Flow rate Q (m^3/s)")
    plt.ylabel("Pressure drop (kPa)")
    plt.title("Pressure drop components vs flow rate")
    plt.legend(); plt.grid(True); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pseudo_dataset(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(df["dp_phy_Pa"], df["dp_obs_Pa"], alpha=0.6)
    mn = min(df["dp_phy_Pa"].min(), df["dp_obs_Pa"].min())
    mx = max(df["dp_phy_Pa"].max(), df["dp_obs_Pa"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", label="y=x")
    plt.xlabel("Physical model pressure drop (Pa)")
    plt.ylabel("Pseudo observed pressure drop (Pa)")
    plt.title("Physical vs Pseudo-observed Pressure Drop")
    plt.legend(); plt.grid(True); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()


def plot_true_vs_pred(pred_df: pd.DataFrame, save_path: Optional[str] = None):
    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df["dp_obs_true"], pred_df["dp_phy_baseline"], alpha=0.5, label="Physical baseline")
    plt.scatter(pred_df["dp_obs_true"], pred_df["dp_corr_rf"], alpha=0.5, label="RF corrected")
    mn = min(pred_df["dp_obs_true"].min(), pred_df["dp_phy_baseline"].min(), pred_df["dp_corr_rf"].min())
    mx = max(pred_df["dp_obs_true"].max(), pred_df["dp_phy_baseline"].max(), pred_df["dp_corr_rf"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", label="y=x")
    plt.xlabel("Observed pressure drop (Pa)")
    plt.ylabel("Predicted pressure drop (Pa)")
    plt.title("Observed vs Predicted Pressure Drop")
    plt.legend(); plt.grid(True); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residual_histogram(pred_df: pd.DataFrame, save_path: Optional[str] = None):
    baseline_res = pred_df["dp_phy_baseline"] - pred_df["dp_obs_true"]
    rf_res = pred_df["dp_corr_rf"] - pred_df["dp_obs_true"]
    plt.figure(figsize=(7, 5))
    plt.hist(baseline_res, bins=30, alpha=0.6, label="Physical baseline residual")
    plt.hist(rf_res, bins=30, alpha=0.6, label="RF corrected residual")
    plt.xlabel("Residual (Pa)")
    plt.ylabel("Count")
    plt.title("Residual Distribution Comparison")
    plt.legend(); plt.grid(True); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()


def plot_model_metric_bar(metrics_df: pd.DataFrame, save_prefix: Optional[str] = None):
    for metric in ["MAE", "RMSE", "R2"]:
        plt.figure(figsize=(7, 4))
        plt.bar(metrics_df["Model"], metrics_df[metric])
        plt.ylabel(metric)
        plt.title(f"Model Comparison: {metric}")
        plt.xticks(rotation=15)
        plt.tight_layout()
        if save_prefix: plt.savefig(f"{save_prefix}_{metric}.png", dpi=300)
        plt.close()


def plot_feature_importance(feature_importance_df: pd.DataFrame, save_path: Optional[str] = None):
    plt.figure(figsize=(8, 5))
    plt.bar(feature_importance_df["feature"], feature_importance_df["importance"])
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()


def plot_vibration_vs_flow(results, save_path=None):
    Q = [r["Q_m3_s"] for r in results]
    Y = [r["vibration_amplitude"] for r in results]
    plt.figure(figsize=(8, 5))
    plt.plot(Q, Y)
    plt.xlabel("Flow rate Q (m^3/s)")
    plt.ylabel("Vibration amplitude")
    plt.title("Vibration amplitude vs flow rate")
    plt.grid(True); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()


def plot_vibration_vs_pressure(results, save_path=None):
    dp = [r["dp_total_Pa"] for r in results]
    Y = [r["vibration_amplitude"] for r in results]
    plt.figure(figsize=(8, 5))
    plt.plot(dp, Y)
    plt.xlabel("Total pressure drop (Pa)")
    plt.ylabel("Vibration amplitude")
    plt.title("Vibration amplitude vs pressure drop")
    plt.grid(True); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()
