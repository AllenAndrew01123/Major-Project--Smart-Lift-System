"""Quick sanity check: every dispatcher on every scenario of one weekday."""
import time

from smartlift.config import BuildingConfig
from smartlift.dispatch.classical import FCFS, DestinationDispatch, Look, NearestCar
from smartlift.sim.core import Simulation
from smartlift.traffic import SCENARIOS, TrafficConfig, generate_day, scenario_window

cfg = BuildingConfig()
day = generate_day(cfg, TrafficConfig(), 0, seed=1)
for scen, (a, b) in SCENARIOS.items():
    w = scenario_window(day, scen)
    print("==", scen, len(w))
    for D in [FCFS, Look, NearestCar, DestinationDispatch]:
        t0 = time.time()
        m = Simulation(cfg, D(), w, t_start=int(a * 3600), t_end=int(b * 3600)).run().metrics()
        print(f"{D.name:22s} served {m['served']}/{m['passengers']} wait {m['avg_wait_s']:6.1f} "
              f"p95 {m['p95_wait_s']:5.0f} journey {m['avg_journey_s']:6.1f} kWh {m['energy_kwh']:6.2f} "
              f"stops {m['stops']} ({time.time() - t0:.1f}s)")
