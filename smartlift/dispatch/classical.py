"""Classical (non-ML) dispatching algorithms used as baselines."""
from __future__ import annotations

from ..sim.core import Lift, Passenger, Simulation
from .base import Dispatcher, assigned_floors, can_serve_here, eta, look_target


class FCFS(Dispatcher):
    name = "FCFS"
    description = ("First-come-first-served: a free car takes the oldest waiting call and "
                   "delivers its passengers before taking the next one. No pick-ups on the way.")

    def next_target(self, sim: Simulation, lift: Lift):
        if lift.passengers:
            return lift.passengers[0].dest
        mine = [p for ps in sim.waiting for p in ps if p.assigned == lift.id]
        if mine:
            return mine[0].origin
        free = [p for ps in sim.waiting for p in ps if p.assigned is None]
        if not free:
            return None
        oldest = min(free, key=lambda p: p.t_arrive)
        for q in sim.waiting[oldest.origin]:  # everyone behind the same hall button
            if q.assigned is None and q.direction == oldest.direction:
                q.assigned = lift.id
        return oldest.origin

    def can_board(self, sim, lift, p):
        return p.assigned in (None, lift.id)

    def should_stop(self, sim, lift, floor):
        return floor in lift.car_calls


class Look(Dispatcher):
    name = "LOOK (collective)"
    description = ("Every car independently sweeps up and down, stopping for any call in its "
                   "direction and reversing when nothing is ahead. No coordination between cars.")

    def next_target(self, sim: Simulation, lift: Lift):
        if can_serve_here(sim, self, lift):
            return lift.floor
        requests = set(lift.car_calls)
        if lift.load < sim.cfg.capacity:
            requests |= set(sim.waiting_floors(lift))
        return look_target(lift, requests)


class NearestCar(Dispatcher):
    name = "Nearest Car (ETA)"
    description = ("Group control: each new hall call is assigned to the car with the lowest "
                   "estimated time of arrival; cars then serve their calls in LOOK order.")

    def on_arrival(self, sim: Simulation, p: Passenger):
        for q in sim.waiting[p.origin]:  # hall button already lit -> same car (if it has room)
            if q is not p and q.direction == p.direction and q.assigned is not None:
                group = sum(r.assigned == q.assigned for r in sim.waiting[p.origin])
                if group + sim.lifts[q.assigned].load < sim.cfg.capacity:
                    p.assigned = q.assigned
                    return
        p.assigned = min(sim.lifts, key=lambda lift: self.cost(sim, lift, p)).id
        sim.wake(sim.lifts[p.assigned])

    def cost(self, sim, lift, p):
        return eta(sim, lift, p.origin, p.direction)

    def next_target(self, sim: Simulation, lift: Lift):
        if can_serve_here(sim, self, lift):
            return lift.floor
        requests = set(lift.car_calls)
        if lift.load < sim.cfg.capacity:
            requests |= assigned_floors(sim, lift)
        return look_target(lift, requests)


class DestinationDispatch(NearestCar):
    name = "Destination Dispatch"
    description = ("Passengers enter their destination in the lobby/hall. Each one is given a "
                   "specific car, chosen so that people going to the same floors share a car "
                   "(fewer stops per trip).")

    def on_arrival(self, sim: Simulation, p: Passenger):
        p.assigned = min(sim.lifts, key=lambda lift: self.cost(sim, lift, p)).id
        sim.wake(sim.lifts[p.assigned])

    def cost(self, sim, lift, p):
        stops = lift.car_calls | assigned_floors(sim, lift)
        dests = lift.car_calls | {q.dest for ps in sim.waiting for q in ps if q.assigned == lift.id}
        new_stops = (p.origin not in stops) + (p.dest not in dests)
        committed = lift.load + sum(q.assigned == lift.id for ps in sim.waiting for q in ps)
        crowded = 120 if committed >= sim.cfg.capacity else 0
        return eta(sim, lift, p.origin, p.direction) + new_stops * sim.cfg.stop_ticks * 2 + crowded

    def can_board(self, sim, lift, p):
        return p.assigned == lift.id
