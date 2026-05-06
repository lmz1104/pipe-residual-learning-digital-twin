import pandas as pd
from pipe_research.config import OUTPUT_DIR, DEFAULTS
from pipe_research.physical_model import batch_generate_samples, pressure_drop_module
from pipe_research.synthetic_data import generate_pseudo_dataset
from pipe_research.ml_model import train_and_compare_models
from pipe_research.vibration_model import vibration_amplitude
from pipe_research.evaluation import write_result_summary
from pipe_research.plotting import (
    plot_pressure_drop_curves, plot_pseudo_dataset, plot_true_vs_pred,
    plot_residual_histogram, plot_model_metric_bar, plot_feature_importance,
    plot_vibration_vs_flow, plot_vibration_vs_pressure,
)


def main():
    Q_values = [0.0005 + i * 0.0002 for i in range(30)]

    samples = batch_generate_samples(
        Q_values=Q_values, D=0.05, L=DEFAULTS["L"], rho=DEFAULTS["rho"], mu=DEFAULTS["mu"],
        epsilon=4.5e-5, n_bend=4, zeta_bend=DEFAULTS["zeta_bend"],
        valve_coeffs=DEFAULTS["valve_coeffs"], extra_zeta=DEFAULTS["extra_zeta"],
    )
    plot_pressure_drop_curves(samples, save_path=str(OUTPUT_DIR / "pressure_drop_curves.png"))

    df = generate_pseudo_dataset(
        n_samples=400, L=DEFAULTS["L"], rho=DEFAULTS["rho"], mu=DEFAULTS["mu"],
        zeta_bend=DEFAULTS["zeta_bend"], valve_coeffs=DEFAULTS["valve_coeffs"],
        extra_zeta=DEFAULTS["extra_zeta"], D_range=DEFAULTS["D_range"], Q_range=DEFAULTS["Q_range"],
        epsilon_range=DEFAULTS["epsilon_range"], bend_range=DEFAULTS["bend_range"],
        noise_level=DEFAULTS["noise_level"], seed=DEFAULTS["seed"],
    )
    df.to_csv(OUTPUT_DIR / "pseudo_dataset.csv", index=False, encoding="utf-8-sig")
    plot_pseudo_dataset(df, save_path=str(OUTPUT_DIR / "pseudo_dataset_scatter.png"))

    results = train_and_compare_models(df, test_size=0.2, random_state=DEFAULTS["seed"])
    metrics_df = results["metrics_df"]
    pred_df = results["pred_df"]
    feature_importance_df = results["feature_importance_df"]
    coef_df = results["coef_df"]

    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    write_result_summary(metrics_df, OUTPUT_DIR / "result_summary.md")
    pred_df.to_csv(OUTPUT_DIR / "prediction_results.csv", index=False, encoding="utf-8-sig")
    feature_importance_df.to_csv(OUTPUT_DIR / "rf_feature_importance.csv", index=False, encoding="utf-8-sig")
    coef_df.to_csv(OUTPUT_DIR / "lr_coefficients.csv", index=False, encoding="utf-8-sig")

    plot_true_vs_pred(pred_df, save_path=str(OUTPUT_DIR / "true_vs_pred.png"))
    plot_residual_histogram(pred_df, save_path=str(OUTPUT_DIR / "residual_histogram.png"))
    plot_model_metric_bar(metrics_df, save_prefix=str(OUTPUT_DIR / "metrics_bar"))
    plot_feature_importance(feature_importance_df, save_path=str(OUTPUT_DIR / "feature_importance.png"))

    vib_results = []
    for Q in Q_values:
        flow_result = pressure_drop_module(
            Q=Q, D=0.05, L=DEFAULTS["L"], rho=DEFAULTS["rho"], mu=DEFAULTS["mu"],
            epsilon=4.5e-5, n_bend=4, zeta_bend=DEFAULTS["zeta_bend"],
            valve_coeffs=DEFAULTS["valve_coeffs"], extra_zeta=DEFAULTS["extra_zeta"],
        )
        amp = vibration_amplitude(
            dp=flow_result["dp_total_Pa"], v=flow_result["velocity_m_s"], m=1.0, c=5.0,
            k=1000.0, alpha=1e-5, beta=1e-3, omega=20.0,
        )
        vib_results.append({
            "Q_m3_s": Q,
            "velocity_m_s": flow_result["velocity_m_s"],
            "dp_total_Pa": flow_result["dp_total_Pa"],
            "vibration_amplitude": amp,
        })

    pd.DataFrame(vib_results).to_csv(OUTPUT_DIR / "vibration_results.csv", index=False, encoding="utf-8-sig")
    plot_vibration_vs_flow(vib_results, save_path=str(OUTPUT_DIR / "vibration_vs_flow.png"))
    plot_vibration_vs_pressure(vib_results, save_path=str(OUTPUT_DIR / "vibration_vs_pressure.png"))

    print("运行完成，输出文件保存在：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
