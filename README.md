# SmartLift: classical vs machine-learning lift dispatching

A revamp of our B.Tech major project *"An Approach Towards Building a Smart Lift System Using Machine
Learning"* (SPIT, 2025). The original slides and paper are in [`legacy/`](legacy/).

**Question:** can machine learning make a group of lifts serve people faster *and* use less energy?

Everything runs on one simulator, so every algorithm (classical or ML) is scored on exactly the same
passengers, on test days the ML models never saw.

```
synthetic traffic ──► simulator ◄──► dispatcher ──► metrics (wait, journey, kWh)
        │                         (classical rules, or LSTM + RL)
        │                               ▲
        └──► hall-call counts ──► LSTM forecast of the next 5 minutes
```

## Quick start

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py          # the interactive app (trained models are included)

pytest                                       # tests
python -m scripts.train_forecasters          # ~1 min: train + evaluate LSTM, Bi-LSTM+GRU
python -m scripts.active_learning            # ~5 min: active-learning experiment
python -m scripts.train_rl                   # ~1 h: 7 LSTM + RL agents in parallel (λ sweep + ablations)
python -m scripts.run_benchmark              # ~1 min: all algorithms × scenarios × 5 test days
```

## What's inside

| Part | File | What it does |
|---|---|---|
| Traffic | `smartlift/traffic.py` | Non-homogeneous Poisson passenger generator: morning up-peak, lunch, evening down-peak, interfloor meetings, weekends, day-to-day occupancy, random conferences, tenant changes |
| Simulator | `smartlift/sim/core.py` | 1 tick = 1 s. Lifts move, stop, open doors, board/alight (capacity limits), standby. Asks a *dispatcher* for every decision |
| Energy | `smartlift/sim/energy.py` | Counterweight physics (only the imbalance is lifted), friction, acceleration, doors, base-load/standby power, split like ISO 25745-2 into running vs non-running energy |
| Classical | `smartlift/dispatch/classical.py` | FCFS, LOOK (collective), Nearest Car (ETA group control), Destination Dispatch |
| Forecasting | `smartlift/ml/forecast.py` | LSTM and Bi-LSTM+GRU (architectures from the paper), Poisson loss, MC dropout, baselines |
| Active learning | `smartlift/ml/active.py` | Uncertainty-driven fine-tuning when the building changes |
| LSTM + RL | `smartlift/ml/rl.py` | The ML dispatcher: learns call assignment and where free cars go (stay / standby / lobby / forecast hotspot), trading waiting against energy with a weight λ |
| Benchmark | `smartlift/bench.py` | Algorithms × scenarios × test days |
| App | `app/streamlit_app.py` | Traffic explorer, live comparison, space-time diagram, model results |

**Setup:** 16 floors (lobby + 15), 4 lifts, 12 people per car, 60 people per floor.
**Dataset:** 140 synthetic days: train 0–69, validation 70–83, test 84–111, drift 112–139.

## Algorithms compared

| Type | Algorithm | Idea |
|---|---|---|
| Classical | FCFS | A free car takes the oldest call; no pick-ups on the way |
| Classical | LOOK (collective) | Every car sweeps up and down on its own, no coordination |
| Classical | Nearest Car (ETA) | Group control: each call goes to the car with the lowest estimated arrival time |
| Classical | Destination Dispatch | Passengers enter their floor; people going to the same floors share a car |
| ML | **LSTM + RL** | One reinforcement-learning agent makes both decisions (below), at several λ |
| Ablation | RL, no forecast | Same agent, but it only sees the last 5 minutes of calls instead of the LSTM forecast |
| Ablation | RL + oracle forecast | Same agent with the true demand rate: the best any forecast could do |

**How LSTM + RL works** (`smartlift/ml/rl.py`). λ is in seconds of waiting per Wh: how much energy matters.

1. *Assignment:* a hall call arrives → choose the car with the lowest
   `predicted wait (Q-network) + λ × extra energy of the detour (physics model)`.
   The Q-network is trained as a contextual bandit (γ = 0) on each passenger's realised wait.
2. *Positioning:* a car is free → **stay**, **standby** (300 W → 50 W), **go to lobby**, or **go to the floor with
   the most forecast calls**. The state includes the LSTM forecast for the next 5 minutes. Cost until the car's next
   decision = everyone's waiting time + λ × the car's energy. SMDP Q-learning (decisions at irregular times,
   discount e^(−τ/300 s)), Double DQN, replay buffer, slowly-updated target network.

Trained on whole simulated *training* days (250 per agent); the checkpoint with the lowest validation
`wait + λ·energy` is kept. Test days are used only in the benchmark.

## Results

Mean of 5 unseen test weekdays (`artifacts/bench/results.csv`; mean ± std and every scenario are in the app).
λ = 0.5 is the headline setting.

**Full day**

| Algorithm | Avg wait (s) | P95 wait (s) | Avg journey (s) | Energy (kWh) |
|---|---:|---:|---:|---:|
| FCFS | 30.3 | 97 | 55.4 | 170.1 |
| LOOK (collective) | 15.4 | 53 | 45.4 | 198.1 |
| Nearest Car (ETA) | 13.5 | 43 | 46.4 | 124.6 |
| Destination Dispatch | 14.1 | 47 | 47.5 | 119.9 |
| LSTM + RL (λ=0) | 10.3 | 36 | 40.0 | 161.8 |
| LSTM + RL (λ=0.25) | 10.4 | 36 | 40.2 | 128.5 |
| **LSTM + RL (λ=0.5)** | 11.3 | 37 | 43.3 | 118.1 |
| LSTM + RL (λ=1) | 15.4 | 44 | 49.0 | 100.5 |
| LSTM + RL (λ=2) | 20.7 | 63 | 56.7 | 87.2 |
| RL, no forecast (λ=0.5) | 11.7 | 39 | 43.6 | 116.6 |
| RL + oracle forecast (λ=0.5) | 11.2 | 36 | 42.8 | 115.4 |

**Morning up-peak (07:30–10:00)**

| Algorithm | Avg wait (s) | P95 wait (s) | Avg journey (s) | Energy (kWh) |
|---|---:|---:|---:|---:|
| FCFS | 18.3 | 40 | 52.0 | 33.2 |
| LOOK (collective) | 6.7 | 27 | 47.8 | 31.8 |
| Nearest Car (ETA) | 12.1 | 30 | 65.7 | 19.3 |
| Destination Dispatch | 11.8 | 30 | 65.5 | 19.3 |
| LSTM + RL (λ=0) | 5.4 | 22 | 46.4 | 35.1 |
| LSTM + RL (λ=0.25) | 6.0 | 25 | 47.5 | 27.8 |
| **LSTM + RL (λ=0.5)** | 8.3 | 28 | 57.4 | 21.5 |
| LSTM + RL (λ=1) | 13.5 | 38 | 66.7 | 17.8 |
| LSTM + RL (λ=2) | 17.5 | 47 | 72.5 | 16.4 |
| RL, no forecast (λ=0.5) | 8.2 | 31 | 56.0 | 22.4 |
| RL + oracle forecast (λ=0.5) | 7.5 | 26 | 55.6 | 22.2 |


**Demand forecasting** (next 5 minutes, per floor & direction, test days 84–111)

| Model | RMSE per cell | Poisson NLL | RMSE building total | MAPE busy bins |
|---|---:|---:|---:|---:|
| Persistence | 0.794 | 0.962 | 4.62 | 25.5% |
| Historical average | 0.808 | 0.065 | 5.46 | 28.9% |
| LSTM | 0.595 | 0.025 | 3.69 | 20.4% |
| Bi-LSTM + GRU | 0.584 | 0.025 | 3.62 | 19.8% |
| Oracle (true rate) | 0.527 | 0.018 | 3.00 | 16.5% |

### Findings

1. **LSTM + RL beats the best classical algorithm on waiting *and* energy at the same time.** At λ = 0.5, full day:
   11.3 s average wait vs 13.5 s for Nearest Car (−16%), 118 kWh vs 125 kWh (−5%), 95th-percentile wait 37 s vs
   43 s. It was better on both on **every one** of the 5 test days (1.2–2.9 s less waiting, 4.8–8.2 kWh less).
2. **λ gives a real trade-off curve.** λ = 0 waits least (10.3 s) but parks cars aggressively (162 kWh);
   λ = 2 uses 30% less energy than Nearest Car (87 kWh) but waits longer (20.7 s). A building owner picks the point.
3. **Coordination matters most for energy.** Group control (Nearest Car) uses **37% less energy** than
   uncoordinated LOOK, whose cars all chase the same calls.
4. **The sequence models are the best forecasters.** Bi-LSTM+GRU closes ~79% of the gap between the historical
   average and the oracle noise floor, because it sees *today's* level (occupancy, early peak, a conference).
   It edges out the plain LSTM, the same ordering the paper claimed but with a far smaller gap.
5. **But most of the dispatching gain comes from RL, not from the forecast.** With the same λ, the agent that
   only sees the last 5 minutes of calls does about as well as the LSTM one (paired over the 5 test days:
   LSTM better on 2, worse on 3; differences within noise). A *perfect* forecast is better on all 5 days, by about
   2.7 kWh at the same waiting time, so better forecasts do help, but the LSTM is not yet accurate enough to show it.
6. **Active learning: the naive version fails, a diverse version helps early.** After a tenant change:
   * Picking the 24 most uncertain windows (MC dropout) is no better than not updating, because they are
     near-duplicates from the same peak.
   * Picking the most uncertain window *per hour* adapts fastest in week 1.
   * Random sampling catches up by week 4.

## Changes from the original project

| Original | Revamp |
|---|---|
| Single-lift sim, lifts run one after another | Multi-lift tick simulator, all lifts run concurrently, with tests |
| Power chart drawn from straight lines + noise (`legacy/code/C.py`) | Energy measured from a physical model inside the simulation |
| LSTM / Bi-LSTM+GRU / active RNN only described | Implemented, trained, evaluated against baselines and an oracle |
| Tabular Q-learning with random up/down moves | LSTM + RL: learned call assignment and car positioning, beats the best classical method on wait and energy |
| No visual interface ("further work") | Streamlit app |

## Limitations

* Synthetic data. The generator follows lift-traffic theory but isn't calibrated to a real building.
* The simulator ignores acceleration profiles and passenger behaviour such as reopening doors or balking.
* Assignment uses γ = 0 (each call is credited only with its own passenger's wait); only positioning uses
  multi-step (SMDP) credit assignment.
* Each agent is trained once (one seed). RL training is noisy, so small differences between agents
  (e.g. LSTM vs no-forecast) partly reflect training luck. Several seeds per agent would tighten this.
* The LSTM's forecasts on training days are in-sample (it was trained on them), so the agent sees slightly
  better forecasts in training than at test time.
* Results are averaged over 5 test days, so small differences (< ~0.5 s) are within noise.
