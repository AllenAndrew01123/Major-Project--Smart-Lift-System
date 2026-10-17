"""Active-learning RNN: adapting the forecaster when the building changes.

Scenario: a new company moves into the top floors. A model trained
on the old traffic is now wrong for those floors. Each day we may spend a small
*budget* of k training samples (windows) to update the model. Which k?

  Frozen   never update (what a pre-trained model does)
  Random   k random windows of the day
  Active (top-k)      the k windows where the model is most *uncertain*, measured with
                      Monte-Carlo dropout: u_t = sum of Var(y_hat_t) over 20 stochastic passes
  Active (per hour)   the most uncertain window in each hour of the day (k = 24). Top-k tends
                      to pick 24 near-identical consecutive peak windows; spreading the budget
                      over the day keeps the samples diverse.
  Full     every window of the day (upper bound, 12x the budget)

Every update mixes in an equal number of old training windows ("replay") so the model
doesn't forget the rest of the building. Evaluation is prequential: each day is
scored *before* the model learns from it.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ..traffic import BINS_PER_DAY
from .forecast import WINDOW, NeuralForecaster, evaluate, windows


def finetune(model: NeuralForecaster, series, cal, idx, replay_idx, rng, steps=30, lr=5e-4):
    opt = torch.optim.Adam(model.net.parameters(), lr=lr)
    loss_fn = nn.PoissonNLLLoss(log_input=True)
    for _ in range(steps):
        batch = np.concatenate([idx, rng.choice(replay_idx, size=len(idx), replace=False)])
        model.net.train()
        loss = loss_fn(model.net(windows(series, cal, batch)), torch.from_numpy(series[batch]))
        opt.zero_grad()
        loss.backward()
        opt.step()


def run_experiment(base: NeuralForecaster, series, cal, drift_days, replay_idx, budget=24, seed=0,
                   focus_cols=None):
    """Returns {strategy: [per-day metrics dict]}."""
    results = {}
    for strategy in ["Frozen", "Random", "Active (top-k)", "Active (per hour)", "Full retrain"]:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        model = copy.deepcopy(base)
        days = []
        for day in drift_days:
            idx = np.arange(day * BINS_PER_DAY, (day + 1) * BINS_PER_DAY)
            idx = idx[idx >= WINDOW]
            mean, var = model.predict_mc(series, cal, idx)
            m = evaluate(model.predict(series, cal, idx), series[idx])
            if focus_cols is not None:
                p = model.predict(series, cal, idx)[:, focus_cols]
                m["focus_rmse"] = float(np.sqrt(((p - series[idx][:, focus_cols]) ** 2).mean()))
            m["mean_uncertainty"] = float(var.sum(1).mean())
            days.append(m)
            if strategy == "Frozen":
                continue
            if strategy == "Random":
                chosen = rng.choice(idx, size=budget, replace=False)
            elif strategy == "Active (top-k)":
                chosen = idx[np.argsort(var.sum(1))[-budget:]]
            elif strategy == "Active (per hour)":
                hours = (idx % BINS_PER_DAY) // 12
                chosen = np.array([idx[hours == h][np.argmax(var.sum(1)[hours == h])] for h in np.unique(hours)])
            else:
                chosen = idx
            finetune(model, series, cal, chosen, replay_idx, rng,
                     steps=30 if strategy != "Full retrain" else 30 * len(idx) // budget)
        results[strategy] = days
    return results
