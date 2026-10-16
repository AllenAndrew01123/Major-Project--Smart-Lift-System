"""Active-learning experiment on the drift period (run after train_forecasters).

Usage:  python -m scripts.active_learning
"""
import json
import time

import numpy as np

from smartlift.experiment import DRIFT, DRIFT_TENANTS, FORECAST_DIR as OUT, TRAIN, bins, load_dataset
from smartlift.ml.active import run_experiment
from smartlift.ml.forecast import HistoricalAverage, NeuralForecaster, evaluate
from smartlift.traffic import BINS_PER_DAY


def main():
    _, series, true_rates, cal = load_dataset()
    base = NeuralForecaster.load(OUT / "lstm.pt")
    changed = sorted({f for _, f, _ in DRIFT_TENANTS})
    focus = np.array([f * 2 + d for f in changed for d in (0, 1)])
    drift_days = range(*DRIFT)

    t0 = time.time()
    res = run_experiment(base, series, cal, drift_days, replay_idx=bins(TRAIN), focus_cols=focus)
    print(f"done in {time.time() - t0:.0f}s")

    # reference lines: stale historical average and the oracle
    hist = HistoricalAverage().fit(series[:TRAIN[1] * BINS_PER_DAY], cal[:TRAIN[1] * BINS_PER_DAY])
    for name, pred_fn in [("Historical average (stale)", lambda i: hist.predict(series, cal, i)),
                          ("Oracle (true rate)", lambda i: true_rates[i])]:
        days = []
        for d in drift_days:
            i = np.arange(d * BINS_PER_DAY, (d + 1) * BINS_PER_DAY)
            m = evaluate(pred_fn(i), series[i])
            m["focus_rmse"] = float(np.sqrt(((pred_fn(i)[:, focus] - series[i][:, focus]) ** 2).mean()))
            days.append(m)
        res[name] = days

    for name, days in res.items():
        r = [d["rmse"] for d in days]
        f = [d["focus_rmse"] for d in days]
        print(f"{name:28s} rmse first week {np.mean(r[:7]):.4f} last week {np.mean(r[-7:]):.4f} | "
              f"changed floors first {np.mean(f[:7]):.4f} last {np.mean(f[-7:]):.4f}")
    (OUT / "active_learning.json").write_text(json.dumps(
        {"days": list(drift_days), "changed_floors": changed, "budget": 24, "results": res}, indent=2))


if __name__ == "__main__":
    main()
