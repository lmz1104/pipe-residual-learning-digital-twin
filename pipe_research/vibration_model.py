import math


def vibration_amplitude(dp: float, v: float, m: float = 1.0, c: float = 5.0, k: float = 1000.0,
                        alpha: float = 1e-5, beta: float = 1e-3, omega: float = 20.0) -> float:
    if m <= 0:
        raise ValueError("m 必须大于 0")
    if c < 0:
        raise ValueError("c 不能小于 0")
    if k <= 0:
        raise ValueError("k 必须大于 0")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha 和 beta 不能小于 0")
    if omega < 0:
        raise ValueError("omega 不能小于 0")

    A = alpha * dp + beta * v ** 2
    denom = math.sqrt((k - m * omega ** 2) ** 2 + (c * omega) ** 2)
    if denom == 0:
        raise ZeroDivisionError("振动响应分母为 0，请调整参数")
    return A / denom
