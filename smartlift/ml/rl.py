"""LSTM + RL dispatcher: one learning agent that makes both group-control decisions.

1. Assignment   a hall call arrives -> which car serves it?
                cost(car) = 60 * Q_assign(phi)   learned: minutes this passenger will wait
                          + lam * extra_wh(car)  physics: extra energy (Wh) of the detour
                Q_assign is trained as a contextual bandit (gamma = 0): the target is the
                passenger's realised wait. The energy term is known exactly at decision time,
                so there is nothing to learn about it.

2. Positioning  a car is free (no passengers, no calls) -> what should it do?
                actions: stay | standby | go to lobby | go to the floor with the most forecast calls
                the state includes the LSTM forecast of hall calls in the next 5 minutes.
                cost between two decisions = system waiting time (person-seconds, everyone
                waiting in the building) + lam * this car's energy (Wh)
                SMDP Q-learning (decisions happen at irregular times):
                    Q(s, a) <- cost + exp(-tau / HORIZON) * min_a' Q_target(s', a')

lam (seconds of waiting per Wh) sets the trade-off: lam = 0 only minimises waiting; a larger
lam accepts some waiting to save energy. Training several lam values gives a trade-off curve.

Ablations use the same agent with a different forecast: "Persistence" (no forecast, just the
last 5 minutes of calls) and "Oracle" (the true demand rate, a perfect forecast).
"""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import torch
from torch import nn

from ..config import BuildingConfig
from ..dispatch.base import assigned_floors, eta
from ..dispatch.classical import NearestCar
from ..sim.core import Lift, Passenger, Simulation, State
from .forecast import DayForecast

N_ASSIGN = 9
N_POSITION = 10
ACTIONS = ["stay", "standby", "lobby", "forecast hotspot"]
STAY, STANDBY, LOBBY, HOTSPOT = range(4)
DECIDE_EVERY = 60     # a free car re-decides at least once a minute
HORIZON = 300         # seconds: discounting time scale for positioning
WAKE_WH = 250 * 60 / 3600  # waking a standby car loses ~1 minute of standby savings (300 W - 50 W)
MAIN_LAM = 0.5        # the lam reported as "the" LSTM + RL result
SOURCE_NAMES = {"LSTM": "LSTM + RL", "Persistence": "RL, no forecast", "Oracle": "RL + oracle forecast"}


# ---------------------------------------------------------------------- assignment

def assign_features(sim: Simulation, lift: Lift, p: Passenger) -> list[float]:
    cfg = sim.cfg
    committed = lift.load + sum(q.assigned == lift.id for ps in sim.waiting for q in ps)
    heading_our_way = (lift.direction == p.direction and (p.origin - lift.floor) * lift.direction >= 0)
    return [
        eta(sim, lift, p.origin, p.direction) / 60,
        lift.load / cfg.capacity,
        committed / cfg.capacity,
        len(lift.car_calls | assigned_floors(sim, lift)) / 5,
        abs(lift.floor - p.origin) / cfg.n_floors,
        float(heading_our_way),
        float(lift.state == State.IDLE and not lift.passengers),
        float(lift.state == State.DWELL),
        float(p.origin == 0),
    ]


def extra_wh(sim: Simulation, lift: Lift, p: Passenger) -> float:
    """Physics estimate of the extra energy (Wh) if `lift` takes passenger `p`:
    floors outside the car's planned route, new stops, and waking it from standby."""
    e = sim.ecfg
    stops = lift.car_calls | assigned_floors(sim, lift) | {lift.floor}
    lo, hi = min(stops), max(stops)
    floors = max(lo - p.origin, p.origin - hi, 0)
    lo, hi = min(lo, p.origin), max(hi, p.origin)
    floors += max(lo - p.dest, p.dest - hi, 0)
    new_stops = (p.origin not in stops) + (p.dest not in stops)
    per_floor_j = e.friction_n * e.floor_height_m / e.efficiency
    car, cw = e.car_mass_kg, e.car_mass_kg + e.balance * sim.cfg.capacity * e.person_kg
    per_stop_j = e.door_cycle_j + 0.5 * (car + cw) * e.speed_ms ** 2 / e.efficiency
    wake = WAKE_WH if lift.state == State.STANDBY else 0.0
    return (floors * per_floor_j + new_stops * per_stop_j) / 3600 + wake


# ---------------------------------------------------------------------- positioning

def hotspot(sim: Simulation, lift: Lift, per_floor: np.ndarray, parked: dict[int, int]) -> int:
    """Floor with the most forecast calls that no other free car is already covering."""
    covered = {f for i, f in parked.items() if i != lift.id}
    order = np.argsort(-per_floor, kind="stable")
    return int(next((f for f in order if f not in covered), order[0]))


def position_features(sim: Simulation, lift: Lift, demand: np.ndarray, hot: int, n_free: int) -> list[float]:
    """demand: forecast hall calls in the next 5 minutes, [n_floors, 2] (up, down)."""
    cfg = sim.cfg
    per_floor = demand.sum(axis=1)
    return [
        lift.floor / cfg.n_floors,
        float(lift.state == State.STANDBY),
        per_floor.sum() / 30,
        demand[0, 0] / 20,
        per_floor[lift.floor] / 5,
        per_floor[hot] / 5,
        abs(hot - lift.floor) / cfg.n_floors,
        (n_free - 1) / cfg.n_lifts,
        sum(other.state == State.STANDBY for other in sim.lifts) / cfg.n_lifts,
        sum(len(w) for w in sim.waiting) / 10,
    ]


# ---------------------------------------------------------------------- networks

def mlp(n_in: int, n_out: int) -> nn.Module:
    return nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_out))


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.assign = mlp(N_ASSIGN, 1)               # phi(car) -> expected wait (minutes)
        self.position = mlp(N_POSITION, len(ACTIONS))  # state -> cost-to-go of each action


# ---------------------------------------------------------------------- dispatcher

class LSTMRLDispatcher(NearestCar):
    """Learned assignment + learned positioning. Cars serve their calls in LOOK order
    (inherited from Nearest Car); idle cars go to the position the agent chose."""

    def __init__(self, agent: Agent, forecast: DayForecast, lam: float, epsilon: float = 0.0, seed: int = 0):
        self.agent, self.forecast, self.lam = agent, forecast, lam
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.name = algo_name(forecast.source, lam)
        self.description = ("Reinforcement learning makes both decisions: which car answers each call "
                            "(learned wait + λ × extra energy) and what free cars do (stay, standby, "
                            f"lobby, or the busiest forecast floor). Forecast: {forecast.source}.")

    def reset(self, sim: Simulation):
        self.decisions: dict[int, list[float]] = {}     # passenger id -> features of the chosen car
        self.transitions: list[tuple] = []              # (s, a, cost, tau, s')
        self.pending: dict[int, tuple] = {}             # lift id -> (s, a, t, wait_area, energy_j)
        self.parked: dict[int, int] = {}                # lift id -> floor it was sent to
        self.last_decision = {lift.id: -DECIDE_EVERY for lift in sim.lifts}
        self.busy_since_decision = {lift.id: True for lift in sim.lifts}
        self.wait_area = 0.0                            # person-seconds waited so far
        self.action_counts = np.zeros(len(ACTIONS), dtype=int)

    def _greedy(self, values: torch.Tensor) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(values)))
        return int(torch.argmin(values))

    # -- assignment
    def on_arrival(self, sim: Simulation, p: Passenger):
        for q in sim.waiting[p.origin]:  # hall button already lit -> same car (if it has room)
            if q is not p and q.direction == p.direction and q.assigned is not None:
                group = sum(r.assigned == q.assigned for r in sim.waiting[p.origin])
                if group + sim.lifts[q.assigned].load < sim.cfg.capacity:
                    p.assigned = q.assigned
                    return
        phis = [assign_features(sim, lift, p) for lift in sim.lifts]
        energy = torch.tensor([extra_wh(sim, lift, p) for lift in sim.lifts])
        with torch.no_grad():
            cost = 60 * self.agent.assign(torch.tensor(phis)).squeeze(-1) + self.lam * energy
        choice = self._greedy(cost)
        p.assigned = choice
        self.decisions[p.id] = phis[choice]
        self.parked.pop(choice, None)
        sim.wake(sim.lifts[choice])

    # -- positioning
    def _free(self, sim: Simulation, lift: Lift) -> bool:
        return not lift.passengers and not assigned_floors(sim, lift)

    def on_tick(self, sim: Simulation):
        self.wait_area += sum(len(w) for w in sim.waiting)
        for lift in sim.lifts:
            if not self._free(sim, lift) or lift.state not in (State.IDLE, State.STANDBY):
                self.busy_since_decision[lift.id] = True
            elif self.busy_since_decision[lift.id] or sim.t - self.last_decision[lift.id] >= DECIDE_EVERY:
                self._decide(sim, lift)

    def _decide(self, sim: Simulation, lift: Lift):
        demand = self.forecast.at(sim.t)
        hot = hotspot(sim, lift, demand.sum(axis=1), self.parked)
        n_free = sum(self._free(sim, other) for other in sim.lifts)
        s = position_features(sim, lift, demand, hot, n_free)
        with torch.no_grad():
            a = self._greedy(self.agent.position(torch.tensor(s)))
        energy_j = lift.energy.total_kwh * 3.6e6
        if lift.id in self.pending:  # close the previous decision of this car
            s0, a0, t0, area0, e0 = self.pending[lift.id]
            cost = (self.wait_area - area0) + self.lam * (energy_j - e0) / 3600
            self.transitions.append((s0, a0, cost / 60, sim.t - t0, s))
        self.pending[lift.id] = (s, a, sim.t, self.wait_area, energy_j)
        self.last_decision[lift.id] = sim.t
        self.busy_since_decision[lift.id] = False
        self.action_counts[a] += 1

        self.parked.pop(lift.id, None)
        if a == STANDBY:
            sim.standby(lift)
        elif a in (LOBBY, HOTSPOT):
            sim.wake(lift)
            self.parked[lift.id] = 0 if a == LOBBY else hot

    def next_target(self, sim: Simulation, lift: Lift):
        target = super().next_target(sim, lift)
        if target is None and self.parked.get(lift.id, lift.floor) != lift.floor:
            return self.parked[lift.id]
        return target

    def assign_experience(self, sim: Simulation):
        """(features, realised wait in minutes) for every assignment whose passenger boarded."""
        by_id = {p.id: p for p in sim.done}
        return [(phi, (by_id[i].t_board - by_id[i].t_arrive) / 60)
                for i, phi in self.decisions.items() if i in by_id]


# ---------------------------------------------------------------------- training

def episode_cost(m: dict, lam: float) -> float:
    """What the agent minimises, for a whole run: total waiting (s) + lam * energy (Wh)."""
    return m["avg_wait_s"] * m["served"] + lam * m["energy_kwh"] * 1000


def train(lam: float, source: str = "LSTM", episodes: int = 250, seed: int = 0, log=print):
    from ..experiment import TRAIN, VAL, day_forecasts, load_dataset
    from ..traffic import SCENARIOS

    cfg = BuildingConfig()
    df = load_dataset()[0]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    agent = Agent()
    target = copy.deepcopy(agent.position)
    opt = torch.optim.Adam([{"params": agent.assign.parameters(), "lr": 1e-3},
                            {"params": agent.position.parameters(), "lr": 3e-4}])
    a_x, a_y, transitions = [], [], []
    train_days = list(range(TRAIN[0] + 1, TRAIN[1]))  # day 0 has no history for the forecaster
    val_days = [d for d in range(*VAL) if d % 7 < 5][:3]
    forecasts = day_forecasts(source, train_days + val_days)
    end = int(SCENARIOS["Full day"][1] * 3600)

    def run(dispatcher, day):
        return Simulation(cfg, dispatcher, df[df.day == day], t_start=0, t_end=end).run()

    def validate():
        runs = [run(LSTMRLDispatcher(agent, forecasts[d], lam), d) for d in val_days]
        ms = [r.metrics() for r in runs]
        return (float(np.mean([episode_cost(m, lam) for m in ms])),
                float(np.mean([m["avg_wait_s"] for m in ms])), float(np.mean([m["energy_kwh"] for m in ms])))

    nc = [run(NearestCar(), d).metrics() for d in val_days]
    baseline = {"val_cost": float(np.mean([episode_cost(m, lam) for m in nc])),
                "val_wait_s": float(np.mean([m["avg_wait_s"] for m in nc])),
                "val_energy_kwh": float(np.mean([m["energy_kwh"] for m in nc]))}
    log(f"Nearest Car on validation days: wait {baseline['val_wait_s']:.2f}s energy {baseline['val_energy_kwh']:.1f} kWh")
    curve, best = [], (np.inf, None)
    for ep in range(episodes):
        epsilon = max(0.02, 0.3 * (1 - ep / (0.6 * episodes)))
        day = int(rng.choice(train_days))
        disp = LSTMRLDispatcher(agent, forecasts[day], lam, epsilon=epsilon, seed=int(rng.integers(1 << 30)))
        sim = run(disp, day)
        m = sim.metrics()
        for x, y in disp.assign_experience(sim):
            a_x.append(x)
            a_y.append(y)
        transitions += disp.transitions
        a_x, a_y, transitions = a_x[-100_000:], a_y[-100_000:], transitions[-100_000:]

        AX, AY = torch.tensor(a_x), torch.tensor(a_y)
        S = torch.tensor([t[0] for t in transitions])
        A = torch.tensor([t[1] for t in transitions])
        C = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
        G = torch.exp(-torch.tensor([t[3] for t in transitions], dtype=torch.float32) / HORIZON)
        S2 = torch.tensor([t[4] for t in transitions])
        for step in range(200):
            b = torch.randint(len(AX), (256,))
            loss_a = nn.functional.mse_loss(agent.assign(AX[b]).squeeze(-1), AY[b])
            b = torch.randint(len(S), (256,))
            with torch.no_grad():  # Double DQN: online net picks a', slow target net scores it
                a2 = agent.position(S2[b]).argmin(dim=1, keepdim=True)
                y = C[b] + G[b] * target(S2[b]).gather(1, a2).squeeze(1)
            q = agent.position(S[b]).gather(1, A[b, None]).squeeze(1)
            loss_p = nn.functional.smooth_l1_loss(q, y)
            opt.zero_grad()
            (loss_a + loss_p).backward()
            opt.step()
            with torch.no_grad():  # target net slowly follows the online net (Polyak averaging)
                for pt, po in zip(target.parameters(), agent.position.parameters()):
                    pt.mul_(0.99).add_(0.01 * po)

        row = {"episode": ep + 1, "epsilon": epsilon, "day": day, "wait_s": m["avg_wait_s"],
               "energy_kwh": m["energy_kwh"], "loss_assign": loss_a.item(), "loss_position": loss_p.item(),
               "actions": disp.action_counts.tolist()}
        if (ep + 1) % 10 == 0:
            row["val_cost"], row["val_wait_s"], row["val_energy_kwh"] = validate()
            if row["val_cost"] < best[0]:  # keep the best checkpoint, the curve is noisy
                best = (row["val_cost"], copy.deepcopy(agent.state_dict()))
            log(f"[{source} lam={lam:g}] ep {ep + 1}: eps {epsilon:.2f} val wait {row['val_wait_s']:.2f}s "
                f"energy {row['val_energy_kwh']:.1f} kWh  actions {row['actions']}")
        curve.append(row)
    agent.load_state_dict(best[1])
    return agent, {"lam": lam, "source": source, "seed": seed, "episodes": episodes, "curve": curve, "best_val_cost": best[0],
                   "nearest_car_val": baseline}


def algo_name(source: str, lam: float) -> str:
    return f"{SOURCE_NAMES.get(source, source)} (λ={lam:g})"


def model_path(out_dir, source: str, lam: float, seed: int = 0):
    return out_dir / f"{source.lower()}_lam{lam:g}_seed{seed}.pt"


def save(agent: Agent, info: dict, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = model_path(out_dir, info["source"], info["lam"], info["seed"])
    torch.save(agent.state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps(info, indent=2))


def load_agents(out_dir) -> dict[tuple[str, float, int], Agent]:
    """Every trained agent in `out_dir`, keyed by (forecast source, lam, seed)."""
    agents = {}
    for path in sorted(out_dir.glob("*_lam*.pt")):
        info = json.loads(path.with_suffix(".json").read_text())
        agent = Agent()
        agent.load_state_dict(torch.load(path))
        agent.eval()
        agents[(info["source"], info["lam"], info["seed"])] = agent
    order = list(SOURCE_NAMES)
    return dict(sorted(agents.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1], kv[0][2])))
