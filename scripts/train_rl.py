"""Train LSTM + RL agents.  Usage:
    python -m scripts.train_rl                      # the full set: lam sweep + ablations (parallel)
    python -m scripts.train_rl --lam 0.5 --source LSTM
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
JOBS = [("LSTM", lam) for lam in LAMS] + [("Persistence", ABLATION_LAM), ("Oracle", ABLATION_LAM)]


def job(source, lam, episodes):
    torch.set_num_threads(1)  # one core per worker process
    t0 = time.time()
    agent, info = rl.train(lam, source, episodes=episodes, log=lambda msg: print(msg, flush=True))
    rl.save(agent, info, OUT)
    return f"{source} lam={lam:g} done in {time.time() - t0:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float)
    ap.add_argument("--source", default="LSTM", choices=list(rl.SOURCE_NAMES))
    ap.add_argument("--episodes", type=int, default=250)
    args = ap.parse_args()
    jobs = [(args.source, args.lam)] if args.lam is not None else JOBS
    with ProcessPoolExecutor(max_workers=min(len(jobs), 7)) as pool:
        for msg in pool.map(job, *zip(*jobs), [args.episodes] * len(jobs)):
            print(msg)
