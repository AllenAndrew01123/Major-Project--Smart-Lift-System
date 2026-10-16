"""Synthetic passenger traffic for an office building.

Real lift-call logs aren't publicly available, so traffic is generated from a
non-homogeneous Poisson process: every minute of the day has an expected number
of trips (the *rate*), and the actual number is drawn from Poisson(rate).

Traffic components (classic lift-traffic theory, Barney "Elevator Traffic Handbook"):
  * incoming   lobby -> floor   morning peak (~9:00) + return from lunch
  * outgoing   floor -> lobby   lunch out + evening peak (~17:30)
  * interfloor floor -> floor   meetings during working hours
  * background a trickle of floor -> floor trips at all hours (late workers etc.)

Day-to-day variation makes the data realistic and gives forecasting models
something to learn beyond "the average day":
  * weekends are ~15% as busy
  * occupancy varies per day (holidays, rain, ...)
  * peak times shift slightly per day
  * random events (a conference on one floor) add extra traffic
  * tenant changes: a floor's population can change from a given day (drift)
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BuildingConfig

MINUTES = 24 * 60
BIN_S = 300  # forecasting resolution: 5-minute bins
BINS_PER_DAY = 24 * 3600 // BIN_S

SCENARIOS = {  # name -> (start hour, end hour)
    "Morning up-peak": (7.5, 10.0),
    "Lunch (two-way)": (11.75, 14.0),
    "Evening down-peak": (16.5, 19.0),
    "Full day": (0.0, 24.0),
}


@dataclass
class TrafficConfig:
    weekend_factor: float = 0.15
    occupancy_sd: float = 0.12        # day-to-day occupancy variation
    peak_jitter_h: float = 0.25       # day-to-day shift of peak times
    interfloor_trips: float = 1.5     # interfloor trips per person per day
    background_trips: float = 0.05
    event_prob: float = 0.25          # chance of a conference on a weekday
    event_size: int = 50
    # drift: list of (from_day, floor, new_population)
    tenant_changes: list = field(default_factory=list)


def _gauss(t_h, mu, sd):
    """Gaussian bump over time (hours) integrating to 1 trip per person."""
    return np.exp(-0.5 * ((t_h - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def _plateau(t_h, start, end, edge=0.5):
    """Smooth 'working hours' window normalised to integrate to 1."""
    w = 1 / (1 + np.exp(-(t_h - start) / (edge / 4))) / (1 + np.exp((t_h - end) / (edge / 4)))
    return w / (end - start)


def populations(cfg: BuildingConfig, tcfg: TrafficConfig, day: int) -> np.ndarray:
    pop = np.full(cfg.n_floors, cfg.pop_per_floor, dtype=float)
    pop[0] = 0  # nobody works in the lobby
    for from_day, floor, new_pop in tcfg.tenant_changes:
        if day >= from_day:
            pop[floor] = new_pop
    return pop


def day_rates(cfg: BuildingConfig, tcfg: TrafficConfig, day: int, rng: np.random.Generator):
    """Expected trips per minute: dict component -> array [MINUTES, n_floors].

    For incoming the floor index is the destination, for outgoing the origin,
    for interfloor/background the origin.
    """
    t = (np.arange(MINUTES) + 0.5) / 60.0  # minute centres in hours
    weekend = day % 7 >= 5
    pop = populations(cfg, tcfg, day)
    occ = np.clip(rng.normal(1.0, tcfg.occupancy_sd), 0.5, 1.3)
    if weekend:
        occ *= tcfg.weekend_factor
    j = lambda: rng.normal(0, tcfg.peak_jitter_h)  # noqa: E731

    incoming = 0.85 * _gauss(t, 9.0 + j(), 0.45) + 0.35 * _gauss(t, 13.3 + j() / 2, 0.3)
    outgoing = 0.35 * _gauss(t, 12.3 + j() / 2, 0.3) + 0.85 * _gauss(t, 17.6 + j(), 0.55)
    inter = tcfg.interfloor_trips * _plateau(t, 9.75, 17.0)
    back = np.full_like(t, tcfg.background_trips / 24)

    per_person = lambda curve: np.outer(curve, pop * occ) / 60.0  # per hour -> per minute  # noqa: E731
    rates = {
        "incoming": per_person(incoming),
        "outgoing": per_person(outgoing),
        "interfloor": per_person(inter + back),
    }

    if not weekend and rng.random() < tcfg.event_prob:
        floor = rng.integers(1, cfg.n_floors)
        start = rng.uniform(10, 15)
        rates["incoming"][:, floor] += tcfg.event_size * _gauss(t, start, 0.15) / 60
        rates["outgoing"][:, floor] += tcfg.event_size * _gauss(t, start + 1.5, 0.15) / 60
    return rates


def generate_day(cfg: BuildingConfig, tcfg: TrafficConfig, day: int, seed: int, return_rates: bool = False):
    """Sample every passenger of one day: columns day, time (s), origin, dest.

    With return_rates=True also returns the *true* expected hall calls per 5-min bin
    [BINS_PER_DAY, n_floors, 2] - the best possible forecast (used as an oracle).
    """
    rng = np.random.default_rng([seed, day])
    rates = day_rates(cfg, tcfg, day, rng)
    upper = np.arange(1, cfg.n_floors)
    rows = []
    for kind, r in rates.items():
        counts = rng.poisson(r)  # [MINUTES, n_floors]
        minute, floor = np.nonzero(counts)
        n = counts[minute, floor]
        minute, floor = np.repeat(minute, n), np.repeat(floor, n)
        time = minute * 60 + rng.integers(0, 60, size=len(minute))
        if kind == "incoming":
            origin, dest = np.zeros_like(floor), floor
        elif kind == "outgoing":
            origin, dest = floor, np.zeros_like(floor)
        else:  # interfloor: random other upper floor
            origin = floor
            dest = rng.choice(upper, size=len(floor))
            clash = dest == origin
            dest[clash] = (dest[clash] % (cfg.n_floors - 1)) + 1
        rows.append(pd.DataFrame({"time": time, "origin": origin, "dest": dest}))
    df = pd.concat(rows, ignore_index=True)
    df = df[df.origin != df.dest]
    df.insert(0, "day", day)
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    return (df, _rates_to_bins(cfg, rates)) if return_rates else df


def _rates_to_bins(cfg: BuildingConfig, rates: dict) -> np.ndarray:
    """Expected hall calls per (5-min bin, origin floor, direction) from per-minute rates."""
    n = cfg.n_floors
    per_min = np.zeros((MINUTES, n, 2))
    per_min[:, 0, 0] = rates["incoming"].sum(axis=1)   # lobby, going up
    per_min[:, :, 1] += rates["outgoing"]              # each floor, going down
    p_up = np.array([(n - 1 - f) / (n - 2) for f in range(n)])  # interfloor dest uniform over other upper floors
    per_min[:, :, 0] += rates["interfloor"] * p_up
    per_min[:, :, 1] += rates["interfloor"] * (1 - p_up)
    return per_min.reshape(BINS_PER_DAY, BIN_S // 60, n, 2).sum(axis=1).astype(np.float32)


def generate_days(cfg: BuildingConfig, tcfg: TrafficConfig, n_days: int, seed: int, start_day: int = 0,
                  return_rates: bool = False):
    out = [generate_day(cfg, tcfg, d, seed, return_rates) for d in range(start_day, start_day + n_days)]
    if not return_rates:
        return pd.concat(out, ignore_index=True)
    return pd.concat([o[0] for o in out], ignore_index=True), np.stack([o[1] for o in out])


def scenario_window(df_day: pd.DataFrame, scenario: str) -> pd.DataFrame:
    start, end = SCENARIOS[scenario]
    return df_day[(df_day.time >= start * 3600) & (df_day.time < end * 3600)].reset_index(drop=True)


def hall_call_counts(df: pd.DataFrame, n_floors: int, days=None) -> np.ndarray:
    """Count passenger arrivals per 5-min bin, floor and direction.

    Returns array [days, BINS_PER_DAY, n_floors, 2] where the last axis is (up, down).
    This is what the forecasting models learn to predict.
    """
    days = np.sort(df.day.unique()) if days is None else np.asarray(days)
    day_idx = np.searchsorted(days, df.day.to_numpy())
    out = np.zeros((len(days), BINS_PER_DAY, n_floors, 2), dtype=np.float32)
    b = (df.time.to_numpy() // BIN_S).astype(int)
    d = (df.dest.to_numpy() < df.origin.to_numpy()).astype(int)
    np.add.at(out, (day_idx, b, df.origin.to_numpy(), d), 1)
    return out
