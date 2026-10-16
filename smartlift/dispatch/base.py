"""Dispatcher interface + shared helpers used by several algorithms."""
from __future__ import annotations

from ..sim.core import UP, Lift, Passenger, Simulation, State


class Dispatcher:
    name = "Base"
    description = ""

    def reset(self, sim: Simulation):
        pass

    def on_arrival(self, sim: Simulation, p: Passenger):
        pass

    def on_tick(self, sim: Simulation):
        pass

    def next_target(self, sim: Simulation, lift: Lift) -> int | None:
        raise NotImplementedError

    def can_board(self, sim: Simulation, lift: Lift, p: Passenger) -> bool:
        return True

    def should_stop(self, sim: Simulation, lift: Lift, floor: int) -> bool:
        """Default 'collective' behaviour: stop for own car calls, and for people
        waiting to go our way who aren't promised to another car."""
        if floor in lift.car_calls:
            return True
        if lift.load >= sim.cfg.capacity:
            return False
        return any(p.direction == lift.direction and p.assigned in (None, lift.id)
                   and self.can_board(sim, lift, p) for p in sim.waiting[floor])


# ---------------------------------------------------------------------- helpers

def look_target(lift: Lift, requests: set[int]) -> int | None:
    """LOOK / elevator algorithm: keep going in the current direction while there
    are requests ahead (head for the farthest one; stops on the way are picked up
    by should_stop), otherwise reverse."""
    f, d = lift.floor, lift.direction
    for direction in (d, -d):
        ahead = [r for r in requests if (r - f) * direction > 0]
        if ahead:
            return max(ahead) if direction == UP else min(ahead)
    return None


def can_serve_here(sim: Simulation, dispatcher: Dispatcher, lift: Lift) -> bool:
    """Is someone waiting at the lift's current floor that it would pick up right now?"""
    if lift.load >= sim.cfg.capacity:
        return False
    return any(dispatcher.can_board(sim, lift, p) and (not lift.passengers or p.direction == lift.direction)
               for p in sim.waiting[lift.floor])


def assigned_floors(sim: Simulation, lift: Lift) -> set[int]:
    return {f for f, ps in enumerate(sim.waiting) if any(p.assigned == lift.id for p in ps)}


def eta(sim: Simulation, lift: Lift, floor: int, direction: int) -> float:
    """Estimated seconds until `lift` can pick up a passenger at `floor` going `direction`,
    assuming it finishes its current sweep first (LOOK order) and stops at every
    floor it has already committed to."""
    cfg = sim.cfg
    stops = lift.car_calls | assigned_floors(sim, lift)
    pos, d = lift.floor, lift.direction
    delay = {State.STANDBY: cfg.wake_ticks, State.WAKING: lift.timer, State.DWELL: lift.timer}.get(lift.state, 0)
    if not stops:
        return delay + abs(floor - pos) * cfg.ticks_per_floor
    ahead = (floor - pos) * d > 0 or (floor == pos and lift.state != State.MOVING)
    if ahead and direction == d:
        dist = abs(floor - pos)
        n_stops = sum(1 for s in stops if 0 < (s - pos) * d < (floor - pos) * d)
    else:  # finish the sweep, turn around, come back
        far = max(stops | {pos}) if d == UP else min(stops | {pos})
        dist = abs(far - pos) + abs(far - floor)
        n_stops = len(stops)
    full = 60 if lift.load >= cfg.capacity else 0
    return delay + dist * cfg.ticks_per_floor + n_stops * (cfg.stop_ticks + 2) + full
