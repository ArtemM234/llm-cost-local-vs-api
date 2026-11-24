"""Break-even model (paper, Section 2.5). All money in USD, energy in kWh, time in seconds."""

from __future__ import annotations

import math

SECONDS_PER_DAY = 86_400


def local_cost_per_correct(
    volume_per_day: float, c_hw: float, lifetime_years: float, e_task_kwh: float, p_el: float, acc_local: float
) -> float:
    """C_local(V) / (V · acc_local), C_local(V) = C_hw/(L·365) + V·E_task·p_el."""
    daily = c_hw / (lifetime_years * 365) + volume_per_day * e_task_kwh * p_el
    return daily / (volume_per_day * acc_local)


def api_cost_per_correct(c_task_api: float, acc_api: float) -> float:
    return c_task_api / acc_api


def break_even_volume(
    c_hw: float, lifetime_years: float, e_task_kwh: float, p_el: float,
    acc_local: float, c_task_api: float, acc_api: float,
) -> float:
    """V* = (C_hw/(L·365)) / (acc_local·c_api/acc_api − E_task·p_el); inf if local never wins."""
    denom = acc_local * c_task_api / acc_api - e_task_kwh * p_el
    if denom <= 0 or acc_local <= 0:
        return math.inf
    return (c_hw / (lifetime_years * 365)) / denom


def max_daily_volume(t_task_s: float) -> float:
    """V_max = 86400 / t_task: the laptop runs one task at a time, 24 h a day."""
    return SECONDS_PER_DAY / t_task_s


def owned_cost_per_correct(e_task_kwh: float, p_el: float, t_task_s: float, wear_usd_per_hour: float, acc_local: float) -> float:
    """Scenario «laptop already owned»: marginal cost only (energy + optional wear per busy hour)."""
    return (e_task_kwh * p_el + wear_usd_per_hour * t_task_s / 3600) / acc_local


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)
