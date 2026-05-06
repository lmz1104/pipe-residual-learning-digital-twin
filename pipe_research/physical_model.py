import math
from typing import Dict, List, Optional


def reynolds_number(rho: float, v: float, D: float, mu: float) -> float:
    if rho <= 0:
        raise ValueError("rho 必须大于 0")
    if mu <= 0:
        raise ValueError("mu 必须大于 0")
    if D <= 0:
        raise ValueError("D 必须大于 0")
    return rho * abs(v) * D / mu


def velocity_from_flowrate(Q: float, D: float) -> float:
    if D <= 0:
        raise ValueError("D 必须大于 0")
    if Q < 0:
        raise ValueError("Q 必须 >= 0")
    area = math.pi * D ** 2 / 4.0
    return Q / area


def friction_factor(Re: float, epsilon: float, D: float) -> float:
    if Re <= 0:
        raise ValueError("Re 必须大于 0")
    if D <= 0:
        raise ValueError("D 必须大于 0")
    if epsilon < 0:
        raise ValueError("epsilon 不能小于 0")

    rr = epsilon / D
    if Re < 2300:
        return 64.0 / Re

    def f_haaland(re: float, rel_rough: float) -> float:
        term = (rel_rough / 3.7) ** 1.11 + 6.9 / re
        return 1.0 / (-1.8 * math.log10(term)) ** 2

    if 2300 <= Re < 4000:
        f_lam = 64.0 / Re
        f_turb = f_haaland(Re, rr)
        w = (Re - 2300.0) / (4000.0 - 2300.0)
        return (1.0 - w) * f_lam + w * f_turb

    return f_haaland(Re, rr)


def local_loss_coefficient(
    n_bend: int = 0,
    zeta_bend: float = 0.9,
    valve_coeffs: Optional[List[float]] = None,
    extra_zeta: float = 0.0,
) -> float:
    if n_bend < 0:
        raise ValueError("n_bend 必须 >= 0")
    if zeta_bend < 0:
        raise ValueError("zeta_bend 必须 >= 0")
    if extra_zeta < 0:
        raise ValueError("extra_zeta 必须 >= 0")

    total = n_bend * zeta_bend + extra_zeta
    if valve_coeffs is not None:
        for z in valve_coeffs:
            if z < 0:
                raise ValueError("valve_coeffs 中每个局部阻力系数都必须 >= 0")
            total += z
    return total


def darcy_weisbach_dp(f: float, L: float, D: float, rho: float, v: float) -> float:
    if f <= 0:
        raise ValueError("f 必须大于 0")
    if L < 0:
        raise ValueError("L 必须 >= 0")
    if D <= 0:
        raise ValueError("D 必须大于 0")
    if rho <= 0:
        raise ValueError("rho 必须大于 0")
    return f * (L / D) * (rho * v ** 2 / 2.0)


def local_pressure_drop(sum_zeta: float, rho: float, v: float) -> float:
    if sum_zeta < 0:
        raise ValueError("sum_zeta 必须 >= 0")
    if rho <= 0:
        raise ValueError("rho 必须大于 0")
    return sum_zeta * (rho * v ** 2 / 2.0)


def pressure_drop_module(
    Q: float,
    D: float,
    L: float,
    rho: float,
    mu: float,
    epsilon: float,
    n_bend: int = 0,
    zeta_bend: float = 0.9,
    valve_coeffs: Optional[List[float]] = None,
    extra_zeta: float = 0.0,
) -> Dict[str, float]:
    v = velocity_from_flowrate(Q, D)
    Re = reynolds_number(rho, v, D, mu)
    f = friction_factor(Re, epsilon, D)
    dp_friction = darcy_weisbach_dp(f, L, D, rho, v)
    sum_zeta = local_loss_coefficient(n_bend, zeta_bend, valve_coeffs, extra_zeta)
    dp_local = local_pressure_drop(sum_zeta, rho, v)
    dp_total = dp_friction + dp_local
    return {
        "flowrate_m3_s": Q,
        "velocity_m_s": v,
        "Re": Re,
        "relative_roughness": epsilon / D,
        "friction_factor": f,
        "sum_zeta": sum_zeta,
        "dp_friction_Pa": dp_friction,
        "dp_local_Pa": dp_local,
        "dp_total_Pa": dp_total,
        "dp_friction_kPa": dp_friction / 1000.0,
        "dp_local_kPa": dp_local / 1000.0,
        "dp_total_kPa": dp_total / 1000.0,
    }


def batch_generate_samples(Q_values: List[float], D: float, L: float, rho: float, mu: float, epsilon: float,
                           n_bend: int = 0, zeta_bend: float = 0.9, valve_coeffs=None, extra_zeta: float = 0.0):
    results = []
    for Q in Q_values:
        results.append(pressure_drop_module(Q, D, L, rho, mu, epsilon, n_bend, zeta_bend, valve_coeffs, extra_zeta))
    return results
