"""Synthetic data generator for a thermo-fluid pipeline residual-learning demo.

The generated observations are pseudo-observations, not field data. The aim is to
create a controlled benchmark where the physical model is approximately correct
but has learnable systematic errors caused by unmodelled losses, fouling-like
roughness effects and nonlinear operating effects.
"""

import random
import numpy as np
import pandas as pd
from .physical_model import pressure_drop_module


def generate_pseudo_dataset(
    n_samples: int,
    L: float = 100.0,
    rho: float = 1000.0,
    mu: float = 1.0e-3,
    zeta_bend: float = 0.9,
    valve_coeffs=None,
    extra_zeta: float = 0.0,
    D_range: tuple = (0.02, 0.08),
    Q_range: tuple = (0.0005, 0.01),
    epsilon_range: tuple = (1e-5, 5e-5),
    bend_range: tuple = (0, 5),
    noise_level: float = 0.035,
    seed: int = 42,
) -> pd.DataFrame:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    data = []

    q_min, q_max = Q_range
    eps_min, eps_max = epsilon_range
    bend_min, bend_max = bend_range

    for _ in range(n_samples):
        D = random.uniform(*D_range)
        Q = random.uniform(*Q_range)
        epsilon = random.uniform(*epsilon_range)
        n_bend = random.randint(*bend_range)

        result = pressure_drop_module(Q, D, L, rho, mu, epsilon, n_bend, zeta_bend, valve_coeffs, extra_zeta)
        dp_phy = result["dp_total_Pa"]

        q_norm = (Q - q_min) / (q_max - q_min)
        eps_norm = (epsilon - eps_min) / (eps_max - eps_min)
        bend_norm = (n_bend - bend_min) / max(1, bend_max - bend_min)
        re_norm = np.log10(result["Re"]) / 6.0

        # Learnable model discrepancy: hidden systematic terms that are not
        # explicitly represented in the mechanistic baseline.
        fouling_bias = 0.055 * eps_norm * q_norm
        local_loss_bias = 0.070 * bend_norm * q_norm ** 2
        operating_nonlinearity = 0.045 * np.sin(2.5 * np.pi * q_norm) * re_norm
        systematic_error_ratio = fouling_bias + local_loss_bias + operating_nonlinearity

        random_noise_ratio = np.random.normal(loc=0.0, scale=noise_level)
        dp_obs = dp_phy * (1.0 + systematic_error_ratio + random_noise_ratio)

        data.append({
            "Q_m3_s": Q,
            "D_m": D,
            "L_m": L,
            "rho_kg_m3": rho,
            "mu_Pa_s": mu,
            "epsilon_m": epsilon,
            "n_bend": n_bend,
            "zeta_bend": zeta_bend,
            "extra_zeta": extra_zeta,
            "velocity_m_s": result["velocity_m_s"],
            "Re": result["Re"],
            "relative_roughness": result["relative_roughness"],
            "friction_factor": result["friction_factor"],
            "sum_zeta": result["sum_zeta"],
            "dp_friction_Pa": result["dp_friction_Pa"],
            "dp_local_Pa": result["dp_local_Pa"],
            "dp_phy_Pa": dp_phy,
            "systematic_error_ratio": systematic_error_ratio,
            "random_noise_ratio": random_noise_ratio,
            "dp_obs_Pa": dp_obs,
            "error_Pa": dp_obs - dp_phy,
            "relative_error": (dp_obs - dp_phy) / dp_phy if dp_phy != 0 else np.nan,
        })

    return pd.DataFrame(data)
