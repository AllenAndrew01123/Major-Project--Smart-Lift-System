"""Run the simulator from the terminal (no app): every algorithm on one test day.  Usage:
    python -m scripts.simulate                                   # full day 84, all 5 algorithms
    python -m scripts.simulate --day 85 --scenario "Morning up-peak"
    python -m scripts.simulate --watch "Nearest Car (ETA)"       # also print the lifts second by second
"""
import argparse

from smartlift.bench import algorithm_factories, run_one
from smartlift.config import BuildingConfig
from smartlift.experiment import TEST_WEEKDAYS, load_dataset
from smartlift.ml.rl import MAIN_LAM, algo_name
from smartlift.traffic import SCENARIOS, scenario_window


def hhmm(t):
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=TEST_WEEKDAYS[0])
    ap.add_argument("--scenario", default="Full day", choices=list(SCENARIOS))
    ap.add_argument("--watch", help="algorithm to show second by second")
    args = ap.parse_args()

    cfg = BuildingConfig()
    df = load_dataset()[0]
    trips = df[df.day == args.day]
    print(f"dataset: {len(df)} passengers over {df.day.nunique()} days")
    print(f"day {args.day}, {args.scenario}: {len(scenario_window(trips, args.scenario))} passengers, "
          f"{cfg.n_lifts} lifts, {cfg.n_floors} floors\n")

    algos = {name: make for name, make in algorithm_factories(cfg, args.day).items()
             if "RL" not in name or name == algo_name("LSTM", MAIN_LAM)}
    print(f"{'algorithm':24s} {'avg wait':>9s} {'p95 wait':>9s} {'energy':>10s}")
    for name, make in algos.items():
        m = run_one(cfg, make(), trips, args.scenario).metrics()
        print(f"{name:24s} {m['avg_wait_s']:8.1f}s {m['p95_wait_s']:8.0f}s {m['energy_kwh']:7.1f} kWh")

    if args.watch:
        # replay the busiest 2 minutes, one line every 5 s: each lift's floor, state and load
        sim = run_one(cfg, algos[args.watch](), trips, args.scenario, record_every=5)
        busiest = max(sim.snapshots, key=lambda s: sum(s.waiting)).t
        print(f"\n{args.watch}, {hhmm(busiest - 60)}-{hhmm(busiest + 60)}   (floor/state/people inside)")
        for s in sim.snapshots:
            if abs(s.t - busiest) <= 60:
                lifts = "  ".join(f"L{i}: {f:2d} {st[:4]:4s} {n:2d}"
                                  for i, (f, st, n) in enumerate(zip(s.floors, s.states, s.loads)))
                print(f"{hhmm(s.t)}  {lifts}   waiting {sum(s.waiting)}")
