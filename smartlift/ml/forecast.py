"""Hall-call demand forecasting.

Task: given the last W five-minute bins of hall calls (per floor and direction),
predict the expected number of calls per floor and direction in the *next* bin.
That forecast is part of what the LSTM + RL dispatcher sees when it positions free cars.

Models (the first two architectures are the ones from the original paper):
  * LSTM            LSTM(128) -> Dropout(0.2) -> LSTM(64) -> Dropout(0.2) -> Dense
  * Bi-LSTM + GRU   BiLSTM(128) -> Dropout(0.2) -> GRU(64) -> Dropout(0.2) -> Dense
Baselines:
  * Persistence         "next 5 min looks like the last 5 min"
  * Historical average  mean of the same time-slot on past weekdays / weekends
  * Oracle              the true Poisson rate used to generate the data (noise floor)

Targets are counts, so the networks output a log-rate and are trained with Poisson
negative log-likelihood instead of MSE.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ..traffic import BIN_S, BINS_PER_DAY

WINDOW = 12  # 12 x 5 min = the last hour


# ---------------------------------------------------------------------- data

def calendar(n_days: int, first_day: int = 0) -> np.ndarray:
    """Per-bin features: time of day as sin/cos (so 23:55 is close to 00:00) + weekend flag."""
    b = np.tile(np.arange(BINS_PER_DAY), n_days)
    day = np.repeat(np.arange(first_day, first_day + n_days), BINS_PER_DAY)
    angle = 2 * np.pi * b / BINS_PER_DAY
    return np.stack([np.sin(angle), np.cos(angle), (day % 7 >= 5).astype(float)], axis=1).astype(np.float32)


def flatten(counts: np.ndarray) -> np.ndarray:
    """[days, bins, floors, 2] -> [days*bins, floors*2] time series."""
    d, b, f, k = counts.shape
    return counts.reshape(d * b, f * k)


def windows(series: np.ndarray, cal: np.ndarray, idx: np.ndarray):
    """Model inputs for predicting bins `idx`: the previous WINDOW bins of
    log(1+counts) plus calendar features. Returns tensor [N, WINDOW, F]."""
    feats = np.concatenate([np.log1p(series), cal], axis=1)
    steps = idx[:, None] + np.arange(-WINDOW, 0)[None, :]
    return torch.from_numpy(feats[steps])


# ---------------------------------------------------------------------- models

class LSTMNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_in, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(64, n_out)

    def forward(self, x):
        h, _ = self.lstm1(x)
        h, _ = self.lstm2(self.drop(h))
        return self.head(self.drop(h[:, -1]))  # log-rate


class BiLSTMGRUNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, dropout: float = 0.2):
        super().__init__()
        # bidirectional only over the *input window* (all of it is in the past) - no leakage
        self.bilstm = nn.LSTM(n_in, 128, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(256, 64, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(64, n_out)

    def forward(self, x):
        h, _ = self.bilstm(x)
        h, _ = self.gru(self.drop(h))
        return self.head(self.drop(h[:, -1]))


ARCHITECTURES = {"LSTM": LSTMNet, "Bi-LSTM + GRU": BiLSTMGRUNet}


# ---------------------------------------------------------------------- forecasters
# Every forecaster answers: predict(series, cal, idx) -> expected calls [len(idx), floors*2]
# using only series[:i] for each i in idx.

class Persistence:
    name = "Persistence"

    def predict(self, series, cal, idx):
        return series[idx - 1]


class HistoricalAverage:
    name = "Historical average"

    def fit(self, series, cal):
        slot = np.arange(len(series)) % BINS_PER_DAY
        weekend = cal[:, 2].astype(int)
        self.table = np.zeros((2, BINS_PER_DAY, series.shape[1]), dtype=np.float32)
        for w in (0, 1):
            for s in range(BINS_PER_DAY):
                rows = (weekend == w) & (slot == s)
                if rows.any():
                    self.table[w, s] = series[rows].mean(axis=0)
        return self

    def predict(self, series, cal, idx):
        return self.table[cal[idx, 2].astype(int), idx % BINS_PER_DAY]


class NeuralForecaster:
    def __init__(self, arch: str, n_features: int, seed: int = 0):
        torch.manual_seed(seed)
        self.name = arch
        self.arch = arch
        self.n_out = n_features
        self.net = ARCHITECTURES[arch](n_features + 3, n_features)
        self.history: list[dict] = []

    def fit(self, series, cal, train_idx, val_idx, epochs: int = 40, lr: float = 1e-3, batch: int = 256,
            patience: int = 5, verbose: bool = True):
        x_tr, y_tr = windows(series, cal, train_idx), torch.from_numpy(series[train_idx])
        x_va, y_va = windows(series, cal, val_idx), torch.from_numpy(series[val_idx])
        loss_fn = nn.PoissonNLLLoss(log_input=True)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        best, best_state, bad = np.inf, None, 0
        for epoch in range(epochs):
            self.net.train()
            perm = torch.randperm(len(x_tr))
            total = 0.0
            for i in range(0, len(perm), batch):
                b = perm[i:i + batch]
                loss = loss_fn(self.net(x_tr[b]), y_tr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            self.net.eval()
            with torch.no_grad():
                val = loss_fn(self.net(x_va), y_va).item()
            self.history.append({"epoch": epoch + 1, "train_loss": total / len(perm), "val_loss": val})
            if verbose:
                print(f"  [{self.name}] epoch {epoch + 1:2d} train {total / len(perm):.4f} val {val:.4f}")
            if val < best - 1e-4:
                best, best_state, bad = val, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, series, cal, idx):
        self.net.eval()
        return torch.exp(self.net(windows(series, cal, idx))).numpy()

    @torch.no_grad()
    def predict_mc(self, series, cal, idx, samples: int = 20):
        """Monte-Carlo dropout: keep dropout ON at inference and sample several
        predictions. Mean = forecast, variance = the model's uncertainty."""
        self.net.train()
        x = windows(series, cal, idx)
        preds = torch.stack([torch.exp(self.net(x)) for _ in range(samples)])
        self.net.eval()
        return preds.mean(0).numpy(), preds.var(0).numpy()

    def save(self, path):
        torch.save({"arch": self.arch, "n_out": self.n_out, "state": self.net.state_dict(),
                    "history": self.history}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, weights_only=False)
        f = cls(ckpt["arch"], ckpt["n_out"])
        f.net.load_state_dict(ckpt["state"])
        f.history = ckpt["history"]
        return f


class DayForecast:
    """Pre-computed forecast for every 5-min bin of one day, each made using only data
    available before that bin (no peeking at the future)."""

    def __init__(self, preds: np.ndarray, source: str):
        self.preds = preds  # [BINS_PER_DAY, n_floors, 2] expected hall calls (up, down)
        self.source = source

    def at(self, t: int) -> np.ndarray:
        return self.preds[min(int(t) // BIN_S, BINS_PER_DAY - 1)]


# ---------------------------------------------------------------------- evaluation

def evaluate(pred: np.ndarray, actual: np.ndarray) -> dict:
    """Errors per (bin, floor, direction) cell and for the building total per bin."""
    pred = np.maximum(pred, 1e-6)
    total_p, total_a = pred.sum(1), actual.sum(1)
    busy = total_a >= 10  # percentage error only makes sense when there is traffic
    return {
        "rmse": float(np.sqrt(((pred - actual) ** 2).mean())),
        "mae": float(np.abs(pred - actual).mean()),
        "poisson_nll": float((pred - actual * np.log(pred)).mean()),
        "total_rmse": float(np.sqrt(((total_p - total_a) ** 2).mean())),
        "total_mape_busy": float((np.abs(total_p - total_a)[busy] / total_a[busy]).mean() * 100),
    }
