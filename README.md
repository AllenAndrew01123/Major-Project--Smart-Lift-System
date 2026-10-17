# An Approach Towards Building a Smart Lift System Using Machine Learning

Have you ever noticed that waiting for a lift takes longer at certain hours, or wondered why the lifts are parked
where they are? Behind the panel, a *group controller* decides which lift answers every call, usually with a fixed
rule such as "send the nearest lift". That rule reacts to calls as they come in. It can't see the morning rush
coming, and it keeps idle lifts powered on all night.

This project asks whether lifts can **learn** to do better. An **LSTM** forecasts how many calls each floor will make in
the next 5 minutes, and a **reinforcement-learning** controller uses that forecast to decide which lift answers each
call, where free lifts should wait, and when to power them down. It learns by trial and error in a simulated
16-floor office with 4 lifts, and is then tested against four classical controllers.

**Result:** over a full working day, the ML controller cut the average wait from **13.5 s to 11.0 s (−19%)** and the
lifts' energy use from **125 to 119 kWh (−4%)** compared with the best classical controller (Nearest Car). It was tested
on 5 days of traffic it never saw during training, and it was trained 5 separate times: every version beat Nearest
Car on both waiting and energy.

Every controller, classical or ML, runs in the same simulator on exactly the same passengers, so the only
difference between the results is the decisions.

```
                   synthetic traffic
           passengers │          │ past calls
                      ▼          ▼
results ◄── lift simulator      LSTM forecast (next 5 minutes)
              │      ▲               │
  where lifts │      │ which lift    │
  are, who is ▼      │ goes where    ▼
  waiting    controller: classical rules  OR  ML controller (LSTM + RL)
```

The simulator was written for this project; its design was inspired by
[fourjr/elevator-simulator](https://github.com/fourjr/elevator-simulator).

## Quick start

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py          # the interactive app (trained models are included)

pytest                                       # tests
python -m scripts.train_forecasters          # ~1 min: train + evaluate LSTM, Bi-LSTM+GRU
python -m scripts.active_learning            # ~5 min: active-learning experiment
python -m scripts.train_rl                   # ~35 min: 19 LSTM + RL agents in parallel (λ sweep + 5-seed ablations)
python -m scripts.run_benchmark              # ~4 min: all algorithms × scenarios × 5 test days
```

## What's inside

| Part | File | What it does |
|---|---|---|
| Traffic | `smartlift/traffic.py` | Non-homogeneous Poisson passenger generator: morning up-peak, lunch, evening down-peak, interfloor meetings, weekends, day-to-day occupancy, random conferences, tenant changes |
| Simulator | `smartlift/sim/core.py` | 1 tick = 1 s. Lifts move, stop, open doors, board/alight (capacity limits), standby. Asks a *dispatcher* for every decision |
| Energy | `smartlift/sim/energy.py` | Counterweight physics (only the imbalance is lifted), friction, acceleration, doors, base-load/standby power, split like ISO 25745-2 into running vs non-running energy |
| Classical | `smartlift/dispatch/classical.py` | FCFS, LOOK (collective), Nearest Car (ETA group control), Destination Dispatch |
| Forecasting | `smartlift/ml/forecast.py` | LSTM and Bi-LSTM+GRU, Poisson loss, MC dropout, baselines |
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
| ML | **LSTM + RL (λ = 0.5)** | One reinforcement-learning agent makes both decisions (below) |

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
LSTM + RL is also averaged over **5 training seeds** (no seed was picked).

**Full day**

| Algorithm | Avg wait (s) | P95 wait (s) | Avg journey (s) | Energy (kWh) |
|---|---:|---:|---:|---:|
| FCFS | 30.3 | 97 | 55.4 | 170.1 |
| LOOK (collective) | 15.4 | 53 | 45.4 | 198.1 |
| Nearest Car (ETA) | 13.5 | 43 | 46.4 | 124.6 |
| Destination Dispatch | 14.1 | 47 | 47.5 | 119.9 |
| **LSTM + RL (λ=0.5)** | 11.0 | 36 | 42.6 | 119.1 |

**Morning up-peak (07:30–10:00)**

| Algorithm | Avg wait (s) | P95 wait (s) | Avg journey (s) | Energy (kWh) |
|---|---:|---:|---:|---:|
| FCFS | 18.3 | 40 | 52.0 | 33.2 |
| LOOK (collective) | 6.7 | 27 | 47.8 | 31.8 |
| Nearest Car (ETA) | 12.1 | 30 | 65.7 | 19.3 |
| Destination Dispatch | 11.8 | 30 | 65.5 | 19.3 |
| **LSTM + RL (λ=0.5)** | 7.8 | 27 | 55.7 | 22.6 |

### Experiments behind the LSTM + RL design

These are not separate algorithms; they justify the choices above.

**Choosing λ** (full day, one seed each; Nearest Car: 13.5 s, 124.6 kWh). λ is the price of
energy in seconds of waiting per Wh. Smaller λ waits less but spends more energy; larger λ saves energy but waits longer.
λ = 0.5 was fixed before training (`MAIN_LAM` in `rl.py`), not picked from test results, and it is
the only value that beats Nearest Car on both.

| λ | Avg wait (s) | Energy (kWh) |
|---:|---:|---:|
| 0 | 10.3 | 161.8 |
| 0.25 | 10.4 | 128.5 |
| **0.5 (chosen; mean of 5 seeds)** | 11.0 | 119.1 |
| 1 | 15.4 | 100.5 |
| 2 | 20.7 | 87.2 |

**Does the forecast help? Ablations** (full day, λ = 0.5, mean ± std over 5 training seeds). The same agent with no forecast (only the last 5 minutes of calls) and with a perfect (oracle) forecast:

| Agent | Avg wait (s) | Energy (kWh) | Seeds better than Nearest Car on both |
|---|---:|---:|---:|
| Nearest Car (reference) | 13.5 | 124.6 | – |
| LSTM + RL | 11.0 ± 0.3 | 119.1 ± 1.7 | 5 / 5 |
| RL, no forecast | 11.7 ± 0.8 | 116.5 ± 3.2 | 5 / 5 |
| RL + oracle forecast | 11.0 ± 0.4 | 116.3 ± 1.9 | 5 / 5 |

**Demand forecasting** (next 5 minutes, per floor & direction, test days 84–111)

| Model | RMSE per cell | Poisson NLL | RMSE building total | MAPE busy bins |
|---|---:|---:|---:|---:|
| Persistence | 0.794 | 0.962 | 4.62 | 25.5% |
| Historical average | 0.808 | 0.065 | 5.46 | 28.9% |
| LSTM | 0.595 | 0.025 | 3.69 | 20.4% |
| Bi-LSTM + GRU | 0.584 | 0.025 | 3.62 | 19.8% |
| Oracle (true rate) | 0.527 | 0.018 | 3.00 | 16.5% |

### Findings

1. **LSTM + RL beats the best classical algorithm on waiting *and* energy at the same time.** At λ = 0.5, full day,
   averaged over 5 training seeds: 11.0 s average wait vs 13.5 s for Nearest Car (−19%), 119 kWh vs 125 kWh (−4%),
   95th-percentile wait 36 s vs 43 s. **All 5 seeds** beat Nearest Car on both, so it is not one lucky training run.
2. **λ gives a real trade-off curve.** λ = 0 waits least (10.3 s) but parks cars aggressively (162 kWh);
   λ = 2 uses 30% less energy than Nearest Car (87 kWh) but waits longer (20.7 s). A building owner picks the point.
3. **Coordination matters most for energy.** Group control (Nearest Car) uses **37% less energy** than
   uncoordinated LOOK, whose cars all chase the same calls.
4. **The sequence models are the best forecasters.** Bi-LSTM+GRU closes ~79% of the gap between the historical
   average and the oracle noise floor, because it sees *today's* level (occupancy, early peak, a conference).
   It edges out the plain LSTM, but only by a small margin.
5. **Most of the dispatching gain comes from RL; the forecast adds a smaller, consistent extra.** All three λ = 0.5
   agents beat Nearest Car, even the one without a forecast. Compared with that no-forecast agent, the LSTM agent has
   a lower combined cost (wait + λ·energy) in **4 of 5 seeds**: it waits 0.7 s less for 2.6 kWh more, and its
   results vary 3× less between seeds. A *perfect* forecast beats the LSTM in all 5 seeds (same wait, ~2.8 kWh less),
   so a more accurate forecaster would help further.
6. **Active learning: the naive version fails, a diverse version helps early.** After a tenant change:
   * Picking the 24 most uncertain windows (MC dropout) is no better than not updating, because they are
     near-duplicates from the same peak.
   * Picking the most uncertain window *per hour* adapts fastest in week 1.
   * Random sampling catches up by week 4.

## Limitations

* Synthetic data. The generator follows lift-traffic theory but isn't calibrated to a real building.
* The simulator ignores acceleration profiles and passenger behaviour such as reopening doors or balking.
* Assignment uses γ = 0 (each call is credited only with its own passenger's wait); only positioning uses
  multi-step (SMDP) credit assignment.
* RL training is noisy. The λ = 0.5 comparison uses 5 seeds; the other λ values use one seed each, so their
  points on the trade-off curve are less certain. Only 5 seeds, so the LSTM vs no-forecast gap is suggestive
  rather than statistically strong.
* The LSTM's forecasts on training days are in-sample (it was trained on them), so the agent sees slightly
  better forecasts in training than at test time.
* Results are averaged over 5 test days, so small differences (< ~0.5 s) are within noise.
