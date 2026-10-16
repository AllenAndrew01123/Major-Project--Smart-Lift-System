"""Tick-based multi-lift simulator (1 tick = 1 second).

The simulator owns the *physics* (moving, doors, boarding, energy) and asks a
Dispatcher for every *decision*:

    on_arrival(sim, p)            a passenger pressed a hall button -> optionally assign a lift
    next_target(sim, lift)        an idle lift asks where to go next (None = stay)
    should_stop(sim, lift, floor) a moving lift reached a floor -> stop here?
    can_board(sim, lift, p)       may this waiting passenger enter this car?
    on_tick(sim)                  called once per tick (forecasts, standby, ...)

Lift life cycle:   IDLE -> MOVING -> (arrive) -> DWELL -> IDLE ...
                   IDLE -> STANDBY -> WAKING -> IDLE        (energy saving)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..config import BuildingConfig, EnergyConfig
from .energy import EnergyMeter

UP, DOWN = 1, -1


@dataclass(eq=False)
class Passenger:
    id: int
    t_arrive: int
    origin: int
    dest: int
    assigned: int | None = None   # lift id chosen by the dispatcher (None = any)
    t_board: int | None = None
    t_alight: int | None = None
    lift: int | None = None

    @property
    def direction(self) -> int:
        return UP if self.dest > self.origin else DOWN


class State(Enum):
    IDLE = "idle"
    MOVING = "moving"
    DWELL = "dwell"      # doors open, people boarding/alighting
    STANDBY = "standby"  # powered down
    WAKING = "waking"


@dataclass(eq=False)
class Lift:
    id: int
    energy: EnergyMeter
    floor: int = 0
    direction: int = UP
    state: State = State.IDLE
    timer: int = 0
    target: int | None = None
    passengers: list = field(default_factory=list)
    floors_travelled: int = 0
    stops: int = 0

    @property
    def car_calls(self) -> set[int]:
        return {p.dest for p in self.passengers}

    @property
    def load(self) -> int:
        return len(self.passengers)


@dataclass
class Snapshot:
    """What the UI needs to draw one frame."""
    t: int
    floors: list
    states: list
    loads: list
    directions: list
    waiting: list


class Simulation:
    def __init__(self, cfg: BuildingConfig, dispatcher, trips, ecfg: EnergyConfig | None = None,
                 start_floor: int = 0, record_every: int = 0, t_start: int | None = None,
                 t_end: int | None = None, max_overtime: int = 3 * 3600):
        """trips: iterable of (time, origin, dest), or a DataFrame with those columns.

        The run always covers [t_start, t_end] (so every algorithm is charged the same
        base-load energy for the window) and continues until every passenger is delivered.
        """
        self.cfg = cfg
        self.ecfg = ecfg or EnergyConfig()
        self.dispatcher = dispatcher
        if hasattr(trips, "itertuples"):
            trips = trips[["time", "origin", "dest"]].itertuples(index=False)
        self.arrivals = [Passenger(i, int(t), int(o), int(d)) for i, (t, o, d) in enumerate(trips)]
        self.arrivals.sort(key=lambda p: p.t_arrive)
        self.lifts = [Lift(i, EnergyMeter(cfg, self.ecfg), floor=start_floor) for i in range(cfg.n_lifts)]
        self.waiting: list[list[Passenger]] = [[] for _ in range(cfg.n_floors)]
        self.done: list[Passenger] = []
        first = self.arrivals[0].t_arrive if self.arrivals else 0
        last = self.arrivals[-1].t_arrive if self.arrivals else 0
        self.t = self.t_start = first if t_start is None else t_start
        self.t_end = last if t_end is None else t_end
        self._next = 0
        self.record_every = record_every
        self.snapshots: list[Snapshot] = []
        self.deadline = max(self.t_end, last) + max_overtime
        dispatcher.reset(self)

    # ------------------------------------------------------------------ main loop
    def run(self):
        while (self.t < self.t_end or not self.finished) and self.t < self.deadline:
            self.step()
        return self

    @property
    def finished(self) -> bool:
        return (self._next >= len(self.arrivals)
                and not any(self.waiting)
                and not any(lift.passengers for lift in self.lifts))

    def step(self):
        while self._next < len(self.arrivals) and self.arrivals[self._next].t_arrive <= self.t:
            p = self.arrivals[self._next]
            self._next += 1
            self.waiting[p.origin].append(p)
            self.dispatcher.on_arrival(self, p)
        self.dispatcher.on_tick(self)
        for lift in self.lifts:
            self._step_lift(lift)
            lift.energy.tick(standby=lift.state == State.STANDBY)
        if self.record_every and (self.t - self.t_start) % self.record_every == 0:
            self.snapshots.append(Snapshot(
                self.t, [lift.floor for lift in self.lifts], [lift.state.value for lift in self.lifts],
                [lift.load for lift in self.lifts], [lift.direction for lift in self.lifts],
                [len(w) for w in self.waiting]))
        self.t += 1

    def _step_lift(self, lift: Lift):
        if lift.state == State.STANDBY:
            return
        if lift.state in (State.DWELL, State.WAKING):
            lift.timer -= 1
            if lift.timer > 0:
                return
            lift.state = State.IDLE
        if lift.state == State.MOVING:
            lift.timer -= 1
            if lift.timer > 0:
                return
            lift.floor += lift.direction
            lift.floors_travelled += 1
            lift.energy.move_floor(lift.load, lift.direction)
            if lift.floor == lift.target or self.dispatcher.should_stop(self, lift, lift.floor):
                self._arrive(lift)
            else:
                lift.timer = self.cfg.ticks_per_floor
            return
        # IDLE: ask the dispatcher what to do
        target = self.dispatcher.next_target(self, lift)
        if target is None:
            return
        if target == lift.floor:
            self._arrive(lift)
            return
        lift.target = target
        lift.direction = UP if target > lift.floor else DOWN
        lift.state = State.MOVING
        lift.timer = self.cfg.ticks_per_floor
        lift.energy.start(lift.load)

    # ------------------------------------------------------------------ stopping at a floor
    def _arrive(self, lift: Lift):
        floor = lift.floor
        lift.target = None
        alighting = [p for p in lift.passengers if p.dest == floor]
        for p in alighting:
            lift.passengers.remove(p)
            p.t_alight = self.t
            self.done.append(p)

        lift.direction = self.choose_direction(lift)
        space = self.cfg.capacity - lift.load
        boarding = [p for p in self.waiting[floor]
                    if p.direction == lift.direction and self.dispatcher.can_board(self, lift, p)][:space]
        for p in boarding:
            self.waiting[floor].remove(p)
            p.t_board, p.lift = self.t, lift.id
            lift.passengers.append(p)

        # people assigned to this car who didn't fit get re-assigned
        left_behind = [p for p in self.waiting[floor] if p.assigned == lift.id and p.direction == lift.direction]
        for p in left_behind:
            p.assigned = None
        for p in left_behind:
            self.dispatcher.on_arrival(self, p)

        if alighting or boarding:
            lift.state = State.DWELL
            lift.timer = self.cfg.stop_ticks + self.cfg.ticks_per_passenger * (len(alighting) + len(boarding))
            lift.stops += 1
            lift.energy.door_cycle()
        else:
            lift.state = State.IDLE

    def choose_direction(self, lift: Lift) -> int:
        """Direction a stopped car will serve next (decides who may board)."""
        f, d = lift.floor, lift.direction
        if lift.passengers:
            return d if any((c - f) * d > 0 for c in lift.car_calls) else -d
        here = [p for p in self.waiting[f] if self.dispatcher.can_board(self, lift, p)]
        mine = [p for p in here if p.assigned == lift.id]
        for group in (mine, here):
            dirs = {p.direction for p in group}
            if dirs:
                return d if d in dirs else dirs.pop()
        return d

    # ------------------------------------------------------------------ helpers for dispatchers
    def wake(self, lift: Lift):
        if lift.state == State.STANDBY:
            lift.state = State.WAKING
            lift.timer = self.cfg.wake_ticks

    def standby(self, lift: Lift):
        if lift.state == State.IDLE and not lift.passengers:
            lift.state = State.STANDBY

    def waiting_floors(self, lift: Lift | None = None) -> list[int]:
        """Floors with someone waiting that `lift` is allowed to pick up."""
        return [f for f, ps in enumerate(self.waiting)
                if ps and (lift is None or any(self.dispatcher.can_board(self, lift, p) for p in ps))]

    # ------------------------------------------------------------------ results
    def metrics(self) -> dict:
        served = self.done
        wait = np.array([p.t_board - p.t_arrive for p in served]) if served else np.zeros(1)
        journey = np.array([p.t_alight - p.t_arrive for p in served]) if served else np.zeros(1)
        kwh = sum(lift.energy.total_kwh for lift in self.lifts)
        running = sum(lift.energy.running_j for lift in self.lifts) / 3.6e6
        return {
            "passengers": len(self.arrivals),
            "served": len(served),
            "avg_wait_s": float(wait.mean()),
            "p95_wait_s": float(np.percentile(wait, 95)),
            "long_waits_pct": float((wait > 60).mean() * 100),
            "avg_journey_s": float(journey.mean()),
            "energy_kwh": kwh,
            "running_kwh": running,
            "wh_per_passenger": 1000 * kwh / max(len(served), 1),
            "floors_travelled": sum(lift.floors_travelled for lift in self.lifts),
            "stops": sum(lift.stops for lift in self.lifts),
            "sim_seconds": self.t - self.t_start,
        }
