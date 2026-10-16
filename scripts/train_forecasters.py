"""Generate the synthetic dataset, train the forecasting models and evaluate them.

Dataset splits are defined in smartlift/experiment.py.

Usage:  python -m scripts.train_forecasters
"""
import json
import time

import numpy as np

from smartlift.experiment import FORECAST_DIR, MODEL_FILES, N_DAYS, SEED, TEST, TRAIN, VAL, bins, load_dataset
from smartlift.ml.forecast import HistoricalAverage, NeuralForecaster, Persistence, evaluate
from smartlift.traffic import BINS_PER_DAY

OUT = FORECAST_DIR


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df, series, true_rates, cal = load_dataset()
    print(f"dataset: {len(df)} trips over {N_DAYS} days, series {series.shape} ({time.time() - t0:.1f}s)")

    train_idx, val_idx, test_idx = bins(TRAIN), bins(VAL), bins(TEST)
    hist = HistoricalAverage().fit(series[:TRAIN[1] * BINS_PER_DAY], cal[:TRAIN[1] * BINS_PER_DAY])
    models = {"Persistence": Persistence(), "Historical average": hist}
    for arch in ["LSTM", "Bi-LSTM + GRU"]:
        t0 = time.time()
        m = NeuralForecaster(arch, series.shape[1], seed=SEED).fit(series, cal, train_idx, val_idx)
        print(f"{arch} trained in {time.time() - t0:.0f}s")
        m.save(OUT / MODEL_FILES[arch])
        models[arch] = m

    results = {name: evaluate(m.predict(series, cal, test_idx), series[test_idx]) for name, m in models.items()}
    results["Oracle (true rate)"] = evaluate(true_rates[test_idx], series[test_idx])
    for name, r in results.items():
        print(f"{name:20s} " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()))

    # keep a few test days of predictions for the app's plots
    show = bins((TEST[0], TEST[0] + 7))
    np.savez_compressed(OUT / "test_predictions.npz", idx=show, actual=series[show], true_rate=true_rates[show],
                        **{name: m.predict(series, cal, show) for name, m in models.items()})
    history = {name: m.history for name, m in models.items() if hasattr(m, "history")}
    (OUT / "metrics.json").write_text(json.dumps({"test": results, "history": history}, indent=2))


if __name__ == "__main__":
    main()
