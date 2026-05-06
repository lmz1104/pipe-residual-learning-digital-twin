import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

FEATURE_COLS = [
    "Q_m3_s", "D_m", "epsilon_m", "n_bend", "velocity_m_s", "Re",
    "relative_roughness", "friction_factor", "sum_zeta", "dp_phy_Pa",
]


def prepare_features(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y_error = df["error_Pa"].copy()
    y_obs = df["dp_obs_Pa"].copy()
    return X, y_error, y_obs


def evaluate_predictions(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def train_and_compare_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X, y_error, y_obs = prepare_features(df)
    X_train, X_test, yerr_train, yerr_test, _, yobs_test = train_test_split(
        X, y_error, y_obs, test_size=test_size, random_state=random_state
    )

    dp_phy_test = X_test["dp_phy_Pa"].values
    baseline_metrics = evaluate_predictions(yobs_test, dp_phy_test)

    lr = LinearRegression()
    lr.fit(X_train, yerr_train)
    err_pred_lr = lr.predict(X_test)
    dp_corr_lr = dp_phy_test + err_pred_lr
    lr_metrics = evaluate_predictions(yobs_test, dp_corr_lr)

    rf = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
    )
    rf.fit(X_train, yerr_train)
    err_pred_rf = rf.predict(X_test)
    dp_corr_rf = dp_phy_test + err_pred_rf
    rf_metrics = evaluate_predictions(yobs_test, dp_corr_rf)

    metrics_df = pd.DataFrame([
        {"Model": "Physical baseline", **baseline_metrics},
        {"Model": "LinearRegression correction", **lr_metrics},
        {"Model": "RandomForest correction", **rf_metrics},
    ])

    pred_df = pd.DataFrame({
        "dp_obs_true": yobs_test.values,
        "dp_phy_baseline": dp_phy_test,
        "dp_corr_lr": dp_corr_lr,
        "dp_corr_rf": dp_corr_rf,
        "err_true": yerr_test.values,
        "err_pred_lr": err_pred_lr,
        "err_pred_rf": err_pred_rf,
    })

    feature_importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    coef_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "coef": lr.coef_,
    }).sort_values("coef", key=np.abs, ascending=False)

    return {
        "metrics_df": metrics_df,
        "pred_df": pred_df,
        "feature_importance_df": feature_importance_df,
        "coef_df": coef_df,
        "rf_model": rf,
        "lr_model": lr,
    }


def predict_corrected_pressure(model, feature_row: pd.DataFrame) -> float:
    err_pred = model.predict(feature_row[FEATURE_COLS])[0]
    dp_phy = float(feature_row["dp_phy_Pa"].iloc[0])
    return dp_phy + err_pred
