"""Benchmark: every algorithm x scenario x test day, using the fixed experiment dataset."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

import pandas as pd

from .config import BuildingConfig
from .dispatch.classical import FCFS, DestinationDispatch, Look, NearestCar
from .experiment import ARTIFACTS, TEST_WEEKDAYS, day_forecasts, load_dataset
from .ml.rl import LSTMRLDispatcher, load_agents
from .sim.core import Simulation
from .traffic import SCENARIOS, scenario_window

CLASSICAL = {D.name: D for D in (FCFS, Look, NearestCar, DestinationDispatch)}


@lru_cache(maxsize=1)
def trained_agents():
    return load_agents(ARTIFACTS / "rl")


@lru_cache(maxsize=64)
def forecast_for(source: str, day: int):
    return day_forecasts(source, [day])[day]


def algorithm_factories(cfg: BuildingConfig, day: int) -> dict[str, Callable]:
    """name -> function creating a fresh dispatcher for `day`."""
    algos = dict(CLASSICAL)
    for (source, lam), agent in trained_agents().items():
        make = lambda agent=agent, source=source, lam=lam: LSTMRLDispatcher(agent, forecast_for(source, day), lam)
        algos[make().name] = make
    return algos


def run_one(cfg: BuildingConfig, dispatcher, day_trips: pd.DataFrame, scenario: str, record_every: int = 0):
    start, end = SCENARIOS[scenario]
    trips = scenario_window(day_trips, scenario)
    sim = Simulation(cfg, dispatcher, trips, t_start=int(start * 3600), t_end=int(end * 3600),
                     record_every=record_every)
    return sim.run()


def benchmark(algorithms: list[str] | None = None, scenarios: list[str] | None = None,
              days: list[int] | None = None, cfg: BuildingConfig = BuildingConfig(), progress=None) -> pd.DataFrame:
    df = load_dataset()[0]
    days = days or TEST_WEEKDAYS[:5]
    scenarios = scenarios or list(SCENARIOS)
    rows = []
    jobs = [(d, s) for d in days for s in scenarios]
    for i, (day, scen) in enumerate(jobs):
        factories = algorithm_factories(cfg, day)
        for name in algorithms or factories:
            m = run_one(cfg, factories[name](), df[df.day == day], scen).metrics()
            rows.append({"algorithm": name, "scenario": scen, "day": day, **m})
        if progress:
            progress((i + 1) / len(jobs))
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    cols = ["avg_wait_s", "p95_wait_s", "long_waits_pct", "avg_journey_s", "energy_kwh", "wh_per_passenger"]
    return results.groupby(["scenario", "algorithm"], sort=False)[cols].agg(["mean", "std"])
