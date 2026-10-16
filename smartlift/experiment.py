"""The fixed synthetic dataset every experiment uses (so results are reproducible).

Days   0-69   training
Days  70-83   validation (early stopping)
Days  84-111  test (same building)       <- benchmark days come from here
Days 112-139  drift: a new tenant moves into the top floors (active-learning experiment)
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import BuildingConfig
from .ml.forecast import WINDOW, DayForecast, HistoricalAverage, NeuralForecaster, Persistence, calendar, flatten
from .traffic import BINS_PER_DAY, TrafficConfig, generate_days, hall_call_counts

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
FORECAST_DIR = ARTIFACTS / "forecast"
SEED = 42
N_DAYS = 140
TRAIN, VAL, TEST, DRIFT = (0, 70), (70, 84), (84, 112), (112, 140)
DRIFT_TENANTS = [(DRIFT[0], 13, 150), (DRIFT[0], 14, 150), (DRIFT[0], 15, 150), (DRIFT[0], 2, 10), (DRIFT[0], 3, 10)]
TEST_WEEKDAYS = [d for d in range(*TEST) if d % 7 < 5]
MODEL_FILES = {"LSTM": "lstm.pt", "Bi-LSTM + GRU": "bilstm_gru.pt"}


def bins(days):
    idx = np.arange(days[0] * BINS_PER_DAY, days[1] * BINS_PER_DAY)
    return idx[idx >= WINDOW]


@lru_cache(maxsize=1)
def load_dataset(cfg: BuildingConfig = BuildingConfig()):
    """Returns (trips DataFrame, hall-call series [T, F*2], true rates [T, F*2], calendar [T, 3])."""
    tcfg = TrafficConfig(tenant_changes=DRIFT_TENANTS)
    df, rates = generate_days(cfg, tcfg, N_DAYS, seed=SEED, return_rates=True)
    series = flatten(hall_call_counts(df, cfg.n_floors, days=range(N_DAYS)))
    return df, series, flatten(rates), calendar(N_DAYS)


@lru_cache(maxsize=1)
def load_forecasters():
    """All forecasters, by name (the neural ones only if they have been trained)."""
    _, series, true_rates, cal = load_dataset()
    end = TRAIN[1] * BINS_PER_DAY
    out = {"Persistence": Persistence(), "Historical average": HistoricalAverage().fit(series[:end], cal[:end])}
    for name, file in MODEL_FILES.items():
        if (FORECAST_DIR / file).exists():
            out[name] = NeuralForecaster.load(FORECAST_DIR / file)
    out["Oracle"] = true_rates  # arrays are indexed directly
    return out


def day_forecasts(source: str, days, cfg: BuildingConfig = BuildingConfig()) -> dict[int, DayForecast]:
    """Forecast of every 5-min bin of each day in `days`, from forecaster `source`."""
    _, series, _, cal = load_dataset()
    model = load_forecasters()[source]
    days = list(days)
    idx = np.concatenate([np.arange(d * BINS_PER_DAY, (d + 1) * BINS_PER_DAY) for d in days])
    preds = model.predict(series, cal, idx) if hasattr(model, "predict") else model[idx]
    preds = np.asarray(preds).reshape(len(days), BINS_PER_DAY, cfg.n_floors, 2)
    return {d: DayForecast(p, source) for d, p in zip(days, preds)}
