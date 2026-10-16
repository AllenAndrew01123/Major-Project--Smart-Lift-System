import numpy as np
import pytest
import torch

from smartlift.config import BuildingConfig, EnergyConfig
from smartlift.dispatch.classical import FCFS, DestinationDispatch, Look, NearestCar
from smartlift.ml.forecast import WINDOW, DayForecast, NeuralForecaster, calendar
from smartlift.ml.rl import ACTIONS, Agent, LSTMRLDispatcher
from smartlift.sim.core import UP, DOWN, Simulation
from smartlift.sim.energy import EnergyMeter
from smartlift.traffic import BINS_PER_DAY, TrafficConfig, generate_day, hall_call_counts

CFG = BuildingConfig()


def all_dispatchers():
    flat = DayForecast(np.full((BINS_PER_DAY, CFG.n_floors, 2), 0.2, dtype=np.float32), "flat")
    explore = LSTMRLDispatcher(Agent(), flat, lam=0.5, epsilon=1.0)  # random actions: every code path
    return [FCFS(), Look(), NearestCar(), DestinationDispatch(), LSTMRLDispatcher(Agent(), flat, lam=0.5), explore]


def test_single_trip_timing():
    # board at t=0, doors+boarding 6+1 ticks, then 5 floors x 2 ticks
    sim = Simulation(CFG, NearestCar(), [(0, 0, 5)]).run()
    p = sim.done[0]
    assert p.t_board == 0
    assert p.t_alight == CFG.stop_ticks + CFG.ticks_per_passenger + 5 * CFG.ticks_per_floor


@pytest.mark.parametrize("dispatcher", all_dispatchers(), ids=lambda d: d.name)
def test_everyone_delivered_and_capacity_respected(dispatcher):
    day = generate_day(CFG, TrafficConfig(), 0, seed=3)
    trips = day[(day.time >= 8.5 * 3600) & (day.time < 9.5 * 3600)]
    sim = Simulation(CFG, dispatcher, trips, record_every=1).run()
    assert len(sim.done) == len(trips)
    assert all(p.t_arrive <= p.t_board < p.t_alight for p in sim.done)
    assert max(max(s.loads) for s in sim.snapshots) <= CFG.capacity
    assert all(p.lift is not None for p in sim.done)


def test_energy_physics():
    def one_floor(people, direction):
        m = EnergyMeter(CFG, EnergyConfig())
        m.move_floor(people, direction)
        return m.running_j

    assert one_floor(CFG.capacity, UP) > one_floor(0, UP)      # heavy car up costs more
    assert one_floor(0, DOWN) > one_floor(0, UP)               # empty car is lighter than counterweight
    assert one_floor(CFG.capacity // 2, UP) == pytest.approx(one_floor(CFG.capacity // 2, DOWN))  # balanced


def test_traffic_reproducible_and_valid():
    a = generate_day(CFG, TrafficConfig(), 4, seed=7)
    b = generate_day(CFG, TrafficConfig(), 4, seed=7)
    assert a.equals(b)
    assert (a.origin != a.dest).all()
    assert hall_call_counts(a, CFG.n_floors).sum() == len(a)


def test_weekend_quieter():
    weekday = generate_day(CFG, TrafficConfig(), 0, seed=1)
    weekend = generate_day(CFG, TrafficConfig(), 5, seed=1)
    assert len(weekend) < 0.4 * len(weekday)


def test_forecaster_cannot_see_the_future():
    rng = np.random.default_rng(0)
    series = rng.poisson(1.0, size=(2 * BINS_PER_DAY, 4)).astype(np.float32)
    cal = calendar(2)
    model = NeuralForecaster("LSTM", 4)
    idx = np.array([100, 200])
    before = model.predict(series, cal, idx)
    series[100:] = 50  # change everything at/after the first predicted bin
    after = model.predict(series, cal, idx)
    np.testing.assert_allclose(before[0], after[0], rtol=1e-6)
    assert not np.allclose(before[1], after[1])
    assert WINDOW <= 100


def test_mc_dropout_gives_uncertainty():
    torch.manual_seed(0)
    series = np.ones((BINS_PER_DAY, 4), dtype=np.float32)
    model = NeuralForecaster("Bi-LSTM + GRU", 4)
    mean, var = model.predict_mc(series, calendar(1), np.arange(20, 30))
    assert mean.shape == var.shape == (10, 4)
    assert (var > 0).any()


def test_rl_records_position_transitions():
    flat = DayForecast(np.full((BINS_PER_DAY, CFG.n_floors, 2), 0.2, dtype=np.float32), "flat")
    disp = LSTMRLDispatcher(Agent(), flat, lam=1.0, epsilon=1.0, seed=1)
    day = generate_day(CFG, TrafficConfig(), 0, seed=3)
    sim = Simulation(CFG, disp, day[(day.time >= 8 * 3600) & (day.time < 9 * 3600)]).run()
    assert disp.transitions and (disp.action_counts > 0).all()
    s, a, cost, tau, s2 = disp.transitions[0]
    assert 0 <= a < len(ACTIONS) and cost >= 0 and tau > 0 and len(s) == len(s2)
    assert len(disp.assign_experience(sim)) > 0
