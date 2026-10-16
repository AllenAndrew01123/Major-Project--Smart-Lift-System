"""Train LSTM + RL agents.  Usage:
    python -m scripts.train_rl                      # the full set: lam sweep + ablations (parallel)
    python -m scripts.train_rl --lam 0.5 --source LSTM --seed 1

Agents that are already trained are skipped.
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from smartlift.experiment import ARTIFACTS
from smartlift.ml import rl

OUT = ARTIFACTS / "rl"
LAMS = [0, 0.25, 0.5, 1, 2]       # trade-off curve with the LSTM forecast
ABLATION_LAM = 0.5                 # ablations: same lam, different forecast
SEEDS = range(5)                   # the ablation comparison is repeated with 5 training seeds
JOBS = ([("LSTM", lam, 0) for lam in LAMS]
        + [(source, ABLATION_LAM, seed) for seed in SEEDS for source in ("LSTM", "Persistence", "Oracle")])


def job(source, lam, seed, episodes):
    torch.set_num_threads(1)  # one core per worker process
    t0 = time.time()
    agent, info = rl.train(lam, source, episodes=episodes, seed=seed,
                           log=lambda msg: print(f"[seed {seed}] {msg}", flush=True))
    rl.save(agent, info, OUT)
    return f"{source} lam={lam:g} seed={seed} done in {time.time() - t0:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float)
    ap.add_argument("--source", default="LSTM", choices=list(rl.SOURCE_NAMES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=250)
    args = ap.parse_args()
    jobs = [(args.source, args.lam, args.seed)] if args.lam is not None else JOBS
    jobs = list(dict.fromkeys(j for j in jobs if not rl.model_path(OUT, *j).exists()))
    print(f"{len(jobs)} agents to train")
    with ProcessPoolExecutor(max_workers=max(1, min(len(jobs), 8))) as pool:
        for msg in pool.map(job, *zip(*jobs), [args.episodes] * len(jobs)):
            print(msg)
