import numpy as np
import pandas as pd
import streamlit as st

from pipe_research.config import DEFAULTS
from pipe_research.physical_model import pressure_drop_module
from pipe_research.synthetic_data import generate_pseudo_dataset
from pipe_research.ml_model import train_and_compare_models, predict_corrected_pressure
from pipe_research.vibration_model import vibration_amplitude


@st.cache_data
def build_training_data():
    return generate_pseudo_dataset(
        n_samples=600, L=DEFAULTS["L"], rho=DEFAULTS["rho"], mu=DEFAULTS["mu"],
        zeta_bend=DEFAULTS["zeta_bend"], valve_coeffs=DEFAULTS["valve_coeffs"],
        extra_zeta=DEFAULTS["extra_zeta"], D_range=DEFAULTS["D_range"], Q_range=DEFAULTS["Q_range"],
        epsilon_range=DEFAULTS["epsilon_range"], bend_range=DEFAULTS["bend_range"],
        noise_level=DEFAULTS["noise_level"], seed=DEFAULTS["seed"],
    )


@st.cache_resource
def train_rf_model():
    df = build_training_data()
    results = train_and_compare_models(df, test_size=0.2, random_state=DEFAULTS["seed"])
    return results["rf_model"]


st.set_page_config(page_title="Pipe System Digital Twin Prototype", layout="wide")
st.title("管路系统数字孪生原型")
st.write("机理模型 + 数据驱动修正 + 流致振动快速评估")

rf_model = train_rf_model()

with st.sidebar:
    st.header("输入参数")
    Q = st.slider("流量 Q (m³/s)", 0.0005, 0.01, 0.002, step=0.0001)
    D = st.slider("管径 D (m)", 0.02, 0.08, 0.05, step=0.001)
    epsilon = st.slider("粗糙度 ε (m)", 1e-5, 5e-5, 4.5e-5, step=1e-6, format="%.6f")
    n_bend = st.slider("弯头数量", 0, 5, 2, step=1)

result = pressure_drop_module(
    Q=Q, D=D, L=DEFAULTS["L"], rho=DEFAULTS["rho"], mu=DEFAULTS["mu"], epsilon=epsilon,
    n_bend=n_bend, zeta_bend=DEFAULTS["zeta_bend"], valve_coeffs=DEFAULTS["valve_coeffs"],
    extra_zeta=DEFAULTS["extra_zeta"],
)

dp_phy = result["dp_total_Pa"]
feature_row = pd.DataFrame([{
    "Q_m3_s": Q,
    "D_m": D,
    "epsilon_m": epsilon,
    "n_bend": n_bend,
    "velocity_m_s": result["velocity_m_s"],
    "Re": result["Re"],
    "relative_roughness": result["relative_roughness"],
    "friction_factor": result["friction_factor"],
    "sum_zeta": result["sum_zeta"],
    "dp_phy_Pa": dp_phy,
}])

dp_corr = predict_corrected_pressure(rf_model, feature_row)
amp = vibration_amplitude(dp=dp_corr, v=result["velocity_m_s"], m=1.0, c=5.0, k=1000.0, alpha=1e-5, beta=1e-3, omega=20.0)

col1, col2, col3 = st.columns(3)
col1.metric("机理压降 (Pa)", f"{dp_phy:.2f}")
col2.metric("修正压降 (Pa)", f"{dp_corr:.2f}")
col3.metric("振动幅值", f"{amp:.6f}")

st.subheader("流动状态")
st.write({
    "velocity_m_s": result["velocity_m_s"],
    "Re": result["Re"],
    "friction_factor": result["friction_factor"],
    "sum_zeta": result["sum_zeta"],
})

st.subheader("风险提示")
if dp_corr > 2e5:
    st.warning("压降偏高，系统阻力可能较大。")
else:
    st.success("压降处于当前阈值以下。")
if amp > 0.01:
    st.warning("振动响应偏高，存在潜在振动风险。")
else:
    st.success("振动响应处于当前阈值以下。")

st.subheader("趋势分析")
Q_scan = np.linspace(0.0005, 0.01, 40)
scan_rows = []
for q in Q_scan:
    r = pressure_drop_module(
        Q=float(q), D=D, L=DEFAULTS["L"], rho=DEFAULTS["rho"], mu=DEFAULTS["mu"], epsilon=epsilon,
        n_bend=n_bend, zeta_bend=DEFAULTS["zeta_bend"], valve_coeffs=DEFAULTS["valve_coeffs"],
        extra_zeta=DEFAULTS["extra_zeta"],
    )
    xq = pd.DataFrame([{
        "Q_m3_s": float(q),
        "D_m": D,
        "epsilon_m": epsilon,
        "n_bend": n_bend,
        "velocity_m_s": r["velocity_m_s"],
        "Re": r["Re"],
        "relative_roughness": r["relative_roughness"],
        "friction_factor": r["friction_factor"],
        "sum_zeta": r["sum_zeta"],
        "dp_phy_Pa": r["dp_total_Pa"],
    }])
    dp_corr_q = predict_corrected_pressure(rf_model, xq)
    amp_q = vibration_amplitude(dp_corr_q, r["velocity_m_s"])
    scan_rows.append({"Q_m3_s": float(q), "dp_phy_Pa": r["dp_total_Pa"], "dp_corr_Pa": dp_corr_q, "amp": amp_q})

scan_df = pd.DataFrame(scan_rows)
st.line_chart(scan_df.set_index("Q_m3_s")[["dp_phy_Pa", "dp_corr_Pa"]])
st.line_chart(scan_df.set_index("Q_m3_s")[["amp"]])
st.caption("说明：该原型用于方法验证与演示，不代表真实工业级在线数字孪生系统。")
