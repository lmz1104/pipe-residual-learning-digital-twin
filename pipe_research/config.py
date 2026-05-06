from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULTS = {
    "L": 100.0,
    "rho": 1000.0,
    "mu": 1.0e-3,
    "zeta_bend": 0.9,
    "valve_coeffs": [0.2, 1.5],
    "extra_zeta": 0.0,
    "D_range": (0.02, 0.08),
    "Q_range": (0.0005, 0.01),
    "epsilon_range": (1e-5, 5e-5),
    "bend_range": (0, 5),
    "noise_level": 0.035,
    "seed": 42,
}
