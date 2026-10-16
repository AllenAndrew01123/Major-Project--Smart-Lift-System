"""Run the full benchmark and save artifacts/bench/results.csv.  Usage:  python -m scripts.run_benchmark"""
import time

import pandas as pd

from smartlift.bench import benchmark, summarise
from smartlift.experiment import ARTIFACTS

if __name__ == "__main__":
    t0 = time.time()
    results = benchmark()
    out = ARTIFACTS / "bench"
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "results.csv", index=False)
    pd.set_option("display.width", 250)
    summary = summarise(results)
    print(summary.xs("mean", axis=1, level=1).round(2).to_string())
    print(f"done in {time.time() - t0:.0f}s")
