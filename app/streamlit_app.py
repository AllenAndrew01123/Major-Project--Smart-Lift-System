"""SmartLift - interactive front end.   Run:  streamlit run app/streamlit_app.py"""
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartlift.bench import algorithm_factories, run_one  # noqa: E402
from smartlift.config import BuildingConfig  # noqa: E402
from smartlift.experiment import ARTIFACTS, TEST_WEEKDAYS, load_dataset  # noqa: E402
from smartlift.ml.rl import ACTIONS, MAIN_LAM, SOURCE_NAMES, algo_name  # noqa: E402
from smartlift.traffic import BIN_S, BINS_PER_DAY, SCENARIOS  # noqa: E402

st.set_page_config(page_title="SmartLift", page_icon="🛗", layout="wide")

# ---------------------------------------------------------------- chart styling
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ML_COLOR, CLASSIC_COLOR, NEUTRAL = SLOTS[0], "#a3a29c", "#77766f"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def style(fig, height=380, **kw):
    kw.setdefault("margin", dict(l=10, r=10, t=80, b=10))
    fig.update_layout(height=height, hovermode="closest",
                      title=dict(y=0.98, yanchor="top"),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0), **kw)
    fig.update_xaxes(showgrid=False, linecolor="rgba(128,128,128,.4)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)", zeroline=False)
    return fig


def is_ml(name: str) -> bool:
    return "RL" in name


MAIN = algo_name("LSTM", MAIN_LAM)


def hhmm(seconds) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}"


def clock(seconds):
    """Seconds since midnight -> datetime, so plotly draws a real time axis."""
    return pd.to_datetime(np.asarray(seconds), unit="s")


def day_label(d: int) -> str:
    return f"Day {d} ({DAYS[d % 7]})"


# ---------------------------------------------------------------- cached data
@st.cache_data(show_spinner="Generating synthetic traffic…")
def dataset():
    df, series, rates, cal = load_dataset()
    return df, series, rates


@st.cache_data(show_spinner="Simulating…")
def simulate(n_lifts, capacity, algo, scenario, day, record_every=0):
    cfg = replace(BuildingConfig(), n_lifts=n_lifts, capacity=capacity)
    df = dataset()[0]
    sim = run_one(cfg, algorithm_factories(cfg, day)[algo](), df[df.day == day], scenario, record_every)
    waits = [p.t_board - p.t_arrive for p in sim.done]
    snaps = pd.DataFrame([{"t": s.t, "floors": s.floors, "states": s.states, "loads": s.loads,
                           "directions": s.directions, "waiting": s.waiting} for s in sim.snapshots])
    trips = pd.DataFrame([(p.t_arrive, p.origin, p.dest, p.lift) for p in sim.done],
                         columns=["t", "origin", "dest", "lift"])
    return sim.metrics(), waits, snaps, trips


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------- per-tab settings
def pick_day(key: str) -> int:
    return st.selectbox("Test day (never seen in training)", TEST_WEEKDAYS, format_func=day_label, key=f"{key}_day")


def run_settings(key: str):
    """Settings for a live simulation, shown at the top of the tab that uses them."""
    c = st.columns(4)
    with c[0]:
        day = pick_day(key)
    order = ["Full day"] + [name for name in SCENARIOS if name != "Full day"]  # full day first
    scenario = c[1].selectbox("Time of day", order, key=f"{key}_scenario",
                              format_func=lambda s: f"{s} ({hhmm(SCENARIOS[s][0] * 3600)}–{hhmm(SCENARIOS[s][1] * 3600)})")
    n_lifts = c[2].slider("Number of lifts", 2, 8, 4, key=f"{key}_lifts")
    capacity = c[3].slider("Car capacity (people)", 6, 20, 12, key=f"{key}_capacity")
    if (n_lifts, capacity) != (4, 12):
        st.caption("⚠️ The RL agent was trained on the default building (4 lifts, 12 people), and the saved "
                   "results use it too. Other sizes test how well it generalises.")
    return day, scenario, n_lifts, capacity


# The algorithms in the app: the 4 classical ones and LSTM + RL at the chosen λ.
# Other λ values and the ablations are experiments, shown only in the RL agent tab.
ALGOS = [n for n in algorithm_factories(BuildingConfig(), TEST_WEEKDAYS[0]) if not is_ml(n) or n == MAIN]

tabs = st.tabs(["Overview", "Traffic", "Compare algorithms", "Watch a run", "RL agent"])

# ================================================================= Overview
with tabs[0]:
    st.header("An approach towards a smart lift system using ML")
    st.subheader("Have you ever noticed that waiting for a lift takes longer at certain hours?")
    st.markdown("""
At 9 in the morning the queue in the lobby never seems to move, yet at 3 in the afternoon a lift is there almost
as soon as you press the button. **Why is that?** And when nobody is using them, **why are the lifts parked where
they are?** Sometimes all of them are up on the 12th floor, just when you're standing in the lobby.

It comes down to two things:

* **Rush hours.** At 9 am almost everyone starts in the lobby and goes up. Each lift fills up, makes many stops on the
  way, and then has to travel all the way back down empty before it can take the next group. Demand arrives faster
  than the lifts can come back, so a queue builds up. At lunch and in the evening the same thing happens in other
  directions.
* **Simple rules.** Behind the panel, a *group controller* decides which lift answers each call, hundreds of times a
  day. Most follow a fixed rule of thumb like *"send the nearest lift"*, and a free lift often just stays wherever
  it last stopped. The rule reacts to calls; it can't see the morning rush coming, and it doesn't notice that a lift
  standing idle with its lights on all night is quietly using electricity.

**So we asked: what if the lifts could *learn*?** Learn when the rush is coming (an **LSTM** forecasts the calls in
the next 5 minutes), where to wait for it, when to power down, and which car should answer each call. They learn by
trial and error in a simulated building (**reinforcement learning**), then face days they have never seen.
""")
    bench = ARTIFACTS / "bench" / "results.csv"
    if bench.exists():
        r = pd.read_csv(bench)
        full = r[r.scenario == "Full day"]
        mean = full.groupby("algorithm", sort=False)[["avg_wait_s", "p95_wait_s", "energy_kwh"]].mean()
        nc = mean.loc["Nearest Car (ETA)"]
        if MAIN in mean.index:
            ml = mean.loc[MAIN]
            seeds = full[full.algorithm == MAIN].groupby("seed")[["avg_wait_s", "energy_kwh"]].mean()
            wins = int(((seeds.avg_wait_s < nc.avg_wait_s) & (seeds.energy_kwh < nc.energy_kwh)).sum())
            st.subheader("Here's what we found")
            st.markdown(f"Against **Nearest Car**, the best classical controller we tested, over a full working day "
                        f"in a 16-floor office:")
            c = st.columns(4)
            c[0].metric("Average wait", f"{ml.avg_wait_s:.1f} s",
                        f"{(ml.avg_wait_s / nc.avg_wait_s - 1) * 100:+.0f}% (was {nc.avg_wait_s:.1f} s)",
                        delta_color="inverse")
            c[1].metric("95% of people waited under", f"{ml.p95_wait_s:.0f} s",
                        f"{(ml.p95_wait_s / nc.p95_wait_s - 1) * 100:+.0f}% (was {nc.p95_wait_s:.0f} s)",
                        delta_color="inverse", help="95th-percentile waiting time")
            c[2].metric("Energy per day", f"{ml.energy_kwh:.0f} kWh",
                        f"{(ml.energy_kwh / nc.energy_kwh - 1) * 100:+.0f}% (was {nc.energy_kwh:.0f} kWh)",
                        delta_color="inverse")
            c[3].metric("Better on both, every time?", f"{wins} / {len(seeds)}",
                        "separately trained versions", delta_color="off",
                        help="The ML controller was trained 5 times from scratch, with different random seeds; each version is counted.")
            how_many = "every version" if wins == len(seeds) else f"{wins} of the {len(seeds)} versions"
            st.markdown("**In short: people wait less, and the lifts use less electricity.** We tested this on "
                        f"5 days of traffic the ML controller had never seen. We also trained it {len(seeds)} separate "
                        f"times, and {how_many} beat Nearest Car on both waiting and energy. Use the tabs above to run "
                        "the algorithms yourself or watch the lifts move.")

    st.subheader("How we tested it")
    st.markdown("""
A simulator (1 tick = 1 second) runs a 16-floor office with 4 lifts on synthetic traffic: a morning rush, lunch,
an evening rush, meetings, and the odd conference. Every controller, classical or ML, faces exactly the same
passengers, and is scored on how long people wait and how much electricity the lifts use.
""")
    _, mid, _ = st.columns([1, 3, 1])  # middle column = centred diagram
    mid.graphviz_chart("""
digraph {
  rankdir=TB; nodesep=0.5; ranksep=0.45; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#eef2f8", color="#b8c2d3", fontname="Helvetica",
        fontsize=12, fontcolor="#1d2433", margin="0.18,0.1"];
  edge [color="#8a94a6", fontname="Helvetica", fontsize=10, fontcolor="#8a94a6", arrowsize=0.7];

  traffic [label=<<b>Synthetic traffic</b><br/><font point-size="10">140 days of office passengers</font>>];
  lstm [label=<<b>LSTM forecast</b><br/><font point-size="10">hall calls in the next 5 minutes</font>>];
  sim [label=<<b>Lift simulator</b><br/><font point-size="10">16 floors · 4 lifts · energy meter</font>>];
  out [label=<<b>Results</b><br/><font point-size="10">waiting time · energy (kWh)</font>>];
  ctrl [shape=plain, style="", label=<
    <table border="0" cellborder="1" cellspacing="0" cellpadding="8" color="#b8c2d3" style="rounded">
      <tr><td colspan="2" bgcolor="#eef2f8"><b>Controller</b><font point-size="10">: plug in one</font></td></tr>
      <tr>
        <td bgcolor="#f5f5f3"><b>Classical rules</b><br/><font point-size="10">FCFS · LOOK<br/>Nearest Car · Destination</font></td>
        <td port="ml" bgcolor="#dce9fb"><b>ML controller</b><br/><font point-size="10">LSTM + RL</font></td>
      </tr>
    </table>>];

  {rank=same; out; sim; lstm;}
  traffic -> sim [label=" passengers"];
  traffic -> lstm [label=" past calls"];
  sim -> out [label=" scores"];
  sim -> ctrl [label=" where lifts are,\n who is waiting"];
  ctrl -> sim [label=" which lift\n goes where"];
  lstm -> ctrl:ml [label=" forecast"];
}""", width="stretch")
    st.markdown("""
**Algorithms**
""")
    descr = {}
    factories = algorithm_factories(BuildingConfig(), TEST_WEEKDAYS[0])
    for name in ALGOS:
        descr[name] = factories[name]().description
    st.dataframe(pd.DataFrame({"Algorithm": descr.keys(), "Type": ["ML" if is_ml(n) else "Classical" for n in descr],
                               "How it works": descr.values()}), hide_index=True, width="stretch")

# ================================================================= Traffic
with tabs[1]:
    df, series, rates = dataset()
    st.header("Synthetic traffic")
    st.markdown("Real lift logs from office buildings aren't public, so we generate the passengers ourselves, "
                "using the patterns described in lift-traffic engineering (Barney, *Elevator Traffic Handbook*).")
    day = pick_day("traffic")
    day_df = df[df.day == day]
    day_bins = slice(day * BINS_PER_DAY, (day + 1) * BINS_PER_DAY)
    per_bin = series[day_bins].sum(1)
    c = st.columns(3)
    c[0].metric("Trips on this day", f"{len(day_df):,}")
    c[1].metric("Busiest 5 minutes", f"{int(per_bin.max())} people", f"at {hhmm(int(per_bin.argmax()) * BIN_S)}",
                delta_color="off")
    c[2].metric("Dataset size", f"{df.day.nunique()} days", f"{len(df):,} trips", delta_color="off")

    s = series[day_bins].reshape(BINS_PER_DAY, -1, 2)
    times = clock(np.arange(BINS_PER_DAY) * BIN_S)
    fig = go.Figure()
    for k, (label, color) in enumerate([("Up calls", SLOTS[0]), ("Down calls", SLOTS[1])]):
        fig.add_bar(x=times, y=s[:, :, k].sum(1), name=label, marker_color=color)
    fig.add_scatter(x=times, y=rates[day_bins].sum(1), name="Expected (from the formula)", mode="lines",
                    line=dict(color=NEUTRAL, width=2))
    fig.update_layout(barmode="stack", bargap=0.1, title=f"Hall calls per 5 minutes, {day_label(day)}",
                      hovermode="x unified")
    fig.update_xaxes(tickformat="%H:%M", hoverformat="%H:%M")
    st.plotly_chart(style(fig), width="stretch")
    st.caption("Bars = the passengers actually generated. Line = the expected number from the formula below. "
               "The bars wobble around the line: that randomness is what a real day looks like.")

    st.subheader("How the traffic is made")
    st.markdown("""
**Step 1: how many trips to expect, minute by minute.** For each floor *f* and minute *t*:
""")
    st.latex(r"\lambda_f(t) \;=\; \underbrace{P_f}_{\text{people on floor}} \times "
             r"\underbrace{o_d}_{\text{today's turnout}} \times "
             r"\underbrace{\textstyle\sum_k a_k\, g(t;\, \mu_k, \sigma_k)}_{\text{daily pattern}}")
    st.markdown(r"""
* $P_f$ = 60 people on each of the 15 upper floors.
* $g(t; \mu, \sigma)$ is a bell curve centred at time $\mu$ with spread $\sigma$, scaled so its area is 1.
  The weight $a$ in front is then simply **trips per person**.

| Pattern | Trip | Peak $\mu$ | Spread $\sigma$ | Trips per person $a$ |
|---|---|---|---|---|
| Morning arrival | lobby → own floor | 09:00 | 27 min | 0.85 |
| Lunch out | own floor → lobby | 12:18 | 18 min | 0.35 |
| Lunch back | lobby → own floor | 13:18 | 18 min | 0.35 |
| Evening departure | own floor → lobby | 17:36 | 33 min | 0.85 |
| Meetings | own floor → another floor | flat, 09:45–17:00 | – | 1.5 |
| Background | own floor → another floor | flat, all day | – | 0.05 |

**Step 2: make every day a little different**, so a forecaster has something to learn beyond "the average day":
* **Turnout** $o_d \sim \mathcal{N}(1,\ 0.12)$: some days are busier (a deadline), some quieter (rain, holidays).
* **Peak times shift** by $\delta \sim \mathcal{N}(0,\ 15\ \text{min})$: the 9:00 rush is sometimes at 8:45 (lunch peaks shift half as much).
* **Weekends** are 15% as busy.
* **Conferences:** on 1 weekday in 4, 50 extra people arrive at a random floor between 10:00 and 15:00 and leave 1.5 h later.
* **A tenant change** from day 112: floors 13–15 fill up (150 people each) and floors 2–3 empty out (10 each).
  Only the active-learning experiment uses those days.

**Step 3: draw the actual passengers.** The number of people who turn up in a minute is random:
""")
    st.latex(r"N_f(t) \sim \text{Poisson}\big(\lambda_f(t)\big)")
    st.markdown(r"""
Each person gets a random second within that minute; for meetings, the destination is a random other floor.
This is a **non-homogeneous Poisson process**: Poisson because many people decide independently when to use
the lift, *non-homogeneous* because the rate $\lambda$ changes through the day.

**Why this matters for the ML.** Because we know $\lambda$ exactly, we also know the *best possible* forecast
(the **oracle**). No model can predict the Poisson randomness itself, so the oracle shows how close the LSTM
gets to perfect, and the RL experiments use it to test what a perfect forecast would be worth.

**How the 140 days are used:** days 0–69 train the models, 70–83 decide when to stop training, 84–111 are the
**test days** (all results in this app come from these), and 112–139 are the tenant-change days.
""")

# ================================================================= Compare
with tabs[2]:
    st.header("Compare algorithms")
    st.caption("Runs a live simulation of every chosen algorithm on the same passengers.")
    day, scenario, n_lifts, capacity = run_settings("compare")
    chosen = st.multiselect("Algorithms", ALGOS, ALGOS)
    st.caption(f"{scenario} on {day_label(day)} · {n_lifts} lifts · capacity {capacity}")
    if chosen:
        rows, waits = [], {}
        prog = st.progress(0.0)
        for i, algo in enumerate(chosen):
            m, w, _, _ = simulate(n_lifts, capacity, algo, scenario, day)
            rows.append({"Algorithm": algo, **m})
            waits[algo] = w
            prog.progress((i + 1) / len(chosen))
        prog.empty()
        res = pd.DataFrame(rows)

        metric_names = {"avg_wait_s": "Average wait (s)", "p95_wait_s": "95th-percentile wait (s)",
                        "avg_journey_s": "Average journey (s)", "energy_kwh": "Energy (kWh)"}
        cols = st.columns(2)
        for i, (key, label) in enumerate(metric_names.items()):
            ordered = res.sort_values(key)
            fig = go.Figure(go.Bar(
                x=ordered[key], y=ordered.Algorithm, orientation="h",
                marker_color=[ML_COLOR if is_ml(n) else CLASSIC_COLOR for n in ordered.Algorithm],
                text=[f"{v:.1f}" for v in ordered[key]], textposition="outside",
                hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
            fig.update_layout(title=label, showlegend=False, bargap=0.35)
            fig.update_yaxes(autorange="reversed")
            cols[i % 2].plotly_chart(style(fig, 330), width="stretch")
        st.caption("Blue = ML-based, gray = classical. Lower is better everywhere.")

        fig = go.Figure()
        for algo in chosen:
            w = np.sort(waits[algo])
            fig.add_scatter(x=w, y=np.arange(1, len(w) + 1) / len(w) * 100, name=algo, mode="lines",
                            line=dict(width=2, color=ML_COLOR if is_ml(algo) else CLASSIC_COLOR,
                                      dash="solid" if is_ml(algo) else "dot"),
                            hovertemplate=algo + ": %{y:.0f}% waited ≤ %{x}s<extra></extra>")
        fig.update_layout(title="Wait-time distribution: % of passengers who waited at most x seconds",
                          xaxis_title="wait (s)", yaxis_title="% of passengers", hovermode="x unified")
        st.plotly_chart(style(fig, 380), width="stretch")

        show = res[["Algorithm", "served", "avg_wait_s", "p95_wait_s", "long_waits_pct", "avg_journey_s",
                    "energy_kwh", "running_kwh", "wh_per_passenger", "floors_travelled", "stops"]]
        st.dataframe(show.round(2), hide_index=True, width="stretch")

    bench = ARTIFACTS / "bench" / "results.csv"
    if bench.exists():
        with st.expander(f"Saved benchmark for {scenario}: mean ± std over all 5 test weekdays "
                         "(default building; LSTM + RL also over 5 training seeds)"):
            r = pd.read_csv(bench)
            r = r[(r.scenario == scenario) & r.algorithm.isin(ALGOS)]
            agg = r.groupby("algorithm", sort=False)[["avg_wait_s", "p95_wait_s", "avg_journey_s", "energy_kwh"]]
            mean, std = agg.mean(), agg.std()
            table = mean.round(1).astype(str) + " ± " + std.round(1).astype(str)
            st.dataframe(table, width="stretch")

# ================================================================= Watch
with tabs[3]:
    st.header("Watch a run")
    day, scenario, n_lifts, capacity = run_settings("watch")
    a, b = SCENARIOS[scenario]
    algo = st.selectbox("Algorithm", ALGOS, index=ALGOS.index(MAIN) if MAIN in ALGOS else 0)
    m, _, snaps, trips = simulate(n_lifts, capacity, algo, scenario, day, record_every=2)
    marks = list(range(int(a * 3600), int(b * 3600) + 1, 300))  # every 5 minutes of the scenario
    default_len = 20 * 60
    starts = [s0 for s0 in marks if s0 + default_len <= marks[-1]] or marks[:1]
    busiest = max(starts, key=lambda s0: ((trips.t >= s0) & (trips.t < s0 + default_len)).sum())
    t0, t1 = st.select_slider("Time window: drag either end (defaults to the busiest 20 minutes)", marks,
                              value=(busiest, min(busiest + default_len, marks[-1])), format_func=hhmm)
    if t1 <= t0:
        t1 = t0 + 300
    win = snaps[(snaps.t >= t0) & (snaps.t <= t1)]

    k = st.columns(4)
    k[0].metric("Average wait", f"{m['avg_wait_s']:.1f} s")
    k[1].metric("95th pct wait", f"{m['p95_wait_s']:.0f} s")
    k[2].metric("Energy", f"{m['energy_kwh']:.1f} kWh")
    k[3].metric("Passengers", m["served"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.05)
    xt = clock(win.t)
    for i in range(n_lifts):
        floors = [f[i] for f in win.floors]
        states = [s[i] for s in win.states]
        loads = [lo[i] for lo in win.loads]
        fig.add_scatter(x=xt, y=floors, mode="lines", name=f"Lift {i + 1}", row=1, col=1,
                        line=dict(width=2, color=SLOTS[i % len(SLOTS)], shape="linear"),
                        customdata=np.stack([states, loads], axis=1),
                        hovertemplate=f"Lift {i + 1}: floor %{{y}} · %{{customdata[0]}} · %{{customdata[1]}} on board<extra></extra>")
    calls = trips[(trips.t >= t0) & (trips.t <= t1)]
    fig.add_scatter(x=clock(calls.t), y=calls.origin, mode="markers", name="hall call",
                    marker=dict(size=5, color=NEUTRAL, symbol="line-ew-open"), row=1, col=1,
                    hovertemplate="call at floor %{y}<extra></extra>")
    fig.add_scatter(x=xt, y=[sum(w) for w in win.waiting], name="people waiting", mode="lines",
                    line=dict(width=2, color=NEUTRAL), fill="tozeroy", row=2, col=1)
    fig.update_yaxes(title="floor", dtick=1, row=1, col=1)
    fig.update_yaxes(title="waiting", row=2, col=1)
    fig.update_xaxes(tickformat="%H:%M", hoverformat="%H:%M:%S")
    fig.update_layout(title=f"Space-time diagram: {algo}", hovermode="x unified")
    st.plotly_chart(style(fig, 620), width="stretch")
    st.caption("Each line is a lift's floor over time; flat segments are stops. Ticks mark when and where "
               "hall calls were made. Gaps between a tick and a line arriving there = waiting.")

    st.subheader("Animation")
    speed = st.select_slider("Playback speed", [1, 2, 5, 10, 20, 50], value=10, format_func=lambda x: f"{x}×")
    if len(win):
        floors_axis = list(range(16))

        def lift_text(row, i):
            arrow = {1: "▲", -1: "▼"}[row.directions[i]] if row.states[i] == "moving" else ""
            return f"{arrow}{row.loads[i]}"

        def frame_traces(row):
            traces = [go.Bar(x=row.waiting, y=floors_axis, orientation="h")]
            for i in range(n_lifts):
                traces.append(go.Scatter(
                    x=[i + 1], y=[row.floors[i]], text=[lift_text(row, i)],
                    marker=dict(opacity=0.3 if row.states[i] == "standby" else 1),
                    hovertext=[f"Lift {i + 1}: {row.states[i]}, {row.loads[i]} on board"]))
            return traces

        skip = max(1, math.ceil(len(win) / 900))  # long windows: keep the animation light
        rows = list(win.iloc[::skip].itertuples())
        first = rows[0]
        fig = make_subplots(rows=1, cols=2, column_widths=[0.3, 0.7], shared_yaxes=True, horizontal_spacing=0.02)
        fig.add_bar(x=first.waiting, y=floors_axis, orientation="h", marker_color=NEUTRAL, name="waiting",
                    hovertemplate="floor %{y}: %{x} waiting<extra></extra>", row=1, col=1)
        for i in range(n_lifts):
            fig.add_scatter(x=[i + 1], y=[first.floors[i]], mode="markers+text", row=1, col=2,
                            marker=dict(size=34, symbol="square", color=SLOTS[i % len(SLOTS)]),
                            text=[lift_text(first, i)], textfont=dict(color="white", size=12),
                            hovertext=[f"Lift {i + 1}"], hoverinfo="text", name=f"Lift {i + 1}")
            fig.add_vline(x=i + 1, line=dict(color="rgba(128,128,128,.25)", width=22), row=1, col=2)  # shaft
        frame_ms = max(20, int(1000 * 2 * skip / speed))  # one frame = 2*skip simulated seconds
        fig.frames = [go.Frame(data=frame_traces(r), name=str(r.t), traces=list(range(1 + n_lifts)),
                               layout=go.Layout(title_text=f"{hhmm(r.t)}:{r.t % 60:02d}  ·  "
                                                           f"{sum(r.waiting)} waiting"))
                      for r in rows]
        play = dict(frame=dict(duration=frame_ms, redraw=True), transition=dict(duration=0), fromcurrent=True)
        fig.update_layout(
            showlegend=False, title=f"{hhmm(first.t)}:00  ·  {sum(first.waiting)} waiting",
            updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.16, xanchor="left", yanchor="top",
                              buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                                       dict(label="❚❚ Pause", method="animate",
                                            args=[[None], dict(frame=dict(duration=0), mode="immediate")])])],
            sliders=[dict(x=0.2, y=-0.13, len=0.8, currentvalue=dict(visible=False), ticklen=0,
                          steps=[dict(method="animate", label=hhmm(r.t) if (r.t - first.t) % 300 == 0 else "",
                                      args=[[str(r.t)], dict(frame=dict(duration=0, redraw=True),
                                                            mode="immediate")]) for r in rows])])
        most_waiting = max(max(r.waiting) for r in rows)
        fig.update_xaxes(range=[most_waiting + 1, 0], autorange=False, title="people waiting", row=1, col=1)
        fig.update_xaxes(range=[0.4, n_lifts + 0.6], dtick=1, title="lift", row=1, col=2)
        fig.update_yaxes(dtick=1, range=[-0.6, 15.6], title="floor", row=1, col=1)
        st.plotly_chart(style(fig, 700, margin=dict(l=10, r=10, t=80, b=120)), width="stretch")
        st.caption("Number in each car = passengers on board, ▲/▼ = moving, faded car = standby. "
                   "Grey bars on the left = people waiting on each floor. Uses the time window chosen above"
                   + (f" (long window: every {skip}th frame shown)." if skip > 1 else "."))

# ================================================================= RL
with tabs[4]:
    st.header("LSTM + RL: the machine-learning controller")
    st.markdown("""
The ML controller is a **reinforcement-learning agent**: a program that learns which decisions work well by trying
them in the simulator and seeing what they cost, rather than being told the right answer. It was trained only on
the *training* days of the dataset, so the results in this app come from days it has never seen.

It makes the same two decisions that any lift group controller has to make, which are described below. Both use one
setting, **λ (lambda)**, which says how much we care about electricity compared with people's time. It is measured in
seconds of waiting per watt-hour: with λ = 0.5, saving 1 Wh is worth making people wait half a second longer in total.
We fixed λ = 0.5 before training; the chart further down shows what other values would have given.
""")

    st.subheader("Decision 1: which lift answers a call?")
    st.markdown("""
Whenever someone presses a hall button, the controller scores every lift and sends the one with the lowest cost:
""")
    st.latex(r"\text{cost}(\text{lift}) \;=\; \underbrace{\hat{w}(\text{lift})}_{\text{predicted wait (s)}}"
             r"\;+\; \lambda \times \underbrace{\Delta E(\text{lift})}_{\text{extra energy (Wh)}}")
    st.markdown("""
The **predicted wait** comes from a small neural network (the *Q-network*). It looks at each lift from the caller's
point of view: its estimated arrival time, how full it is, how many stops it has already promised, how far away it
is, whether it is already heading this way, whether it is idle or has its doors open, and whether the call is in the
lobby. From these it predicts how long the person will actually wait if that lift is chosen.

The **extra energy** is not learned. It is calculated from the same physics as the simulator: the extra floors the lift
would have to travel, the extra stops, and the cost of waking it if it is in standby.

The network learns from experience. After each passenger boards, their real waiting time is known, and the network is
adjusted so that next time its prediction for a similar situation is closer to what really happened. Each call is
judged only by its own passenger's wait (in RL terms a *contextual bandit*, discount γ = 0). This keeps the learning
simple and stable, because the choice of lift mainly affects that passenger.
""")

    st.subheader("Decision 2: what should a free lift do?")
    st.markdown(f"""
When a lift has no passengers and no calls to answer, it has four options: **{" · ".join(ACTIONS)}**.
*Stay* leaves it where it is, *standby* powers it down from 300 W to 50 W, *lobby* sends it to the ground floor, and
*forecast hotspot* sends it to the floor where the LSTM expects the most calls (skipping floors another free lift is
already covering). The lift decides as soon as it becomes free, and again every minute while it stays free.

To choose, it looks at its own floor and whether it is in standby, at **the LSTM forecast of hall calls in the next
5 minutes** (for the whole building, for the lobby, for its own floor and for the busiest floor), at how many other
lifts are free or in standby, and at how many people are waiting right now.

This decision is harder to learn than the first, because its effects arrive later: powering down at 07:50 saves
energy immediately, but someone may wait longer at 08:05. Each choice is therefore charged with everything that
happens until the lift's next decision, namely the waiting time of **everyone** in the building plus λ times the
lift's own energy, and it is learned with Q-learning adapted to decisions that happen at irregular times (SMDP
Q-learning):
""")
    st.latex(r"Q(s, a) \;\leftarrow\; C \;+\; e^{-\tau / 300\,\mathrm{s}} \cdot \min_{a'} Q(s', a')")
    st.markdown("""
Here $Q(s, a)$ is the network's estimate of the total future cost of taking action $a$ in situation $s$. $C$ is the
cost actually paid until the next decision, $\\tau$ is how many seconds that took, and $s'$ is the situation at the next
decision. The factor $e^{-\\tau/300\\,\\mathrm{s}}$ makes costs in the near future count almost fully and costs far ahead
count less (after 5 minutes they count about 37%). Three standard techniques keep the training stable: a **replay
buffer** that reuses past experience, a slowly updated **target network**, and **Double DQN**, which avoids
over-optimistic estimates.
""")

    st.subheader("How it was trained")
    st.markdown("""
Each training run plays 250 simulated days, picked at random from the training days. At first the controller tries
random actions 30% of the time so that it discovers options it would otherwise never try (for example, standby at
night); this falls to 2% as it learns. Every 10 days it is tested on 3 validation days, and the version with the lowest
waiting + λ × energy is kept. To check that the result is not a lucky run, the whole training was repeated 5 times
with different random seeds.
""")

    st.subheader("Choosing λ, and does the forecast help?")
    st.markdown("""
The chart below places every controller by its full-day energy use (left is better) and average wait (lower is
better), so the best place to be is the bottom-left corner. The blue line shows what the ML controller does at
different values of λ: small values buy shorter waits with more energy, large values save energy but make people
wait longer. The two extra markers answer whether the LSTM forecast is useful: the same controller was trained
once with **no forecast** (it only sees the calls of the last 5 minutes) and once with a **perfect forecast** (the
true demand from the traffic formula).
""")
    bench = ARTIFACTS / "bench" / "results.csv"
    if bench.exists():
        full = pd.read_csv(bench)
        full = full[full.scenario == "Full day"]
        r = full.groupby("algorithm", sort=False)[["avg_wait_s", "energy_kwh"]].mean()
        per_seed = full.dropna(subset=["seed"]).groupby(["algorithm", "seed"])[["avg_wait_s", "energy_kwh"]].mean()
        fig = go.Figure()
        sweep = r[r.index.str.startswith(SOURCE_NAMES["LSTM"])]
        fig.add_scatter(x=sweep.energy_kwh, y=sweep.avg_wait_s, mode="lines+markers+text", name="LSTM + RL (λ sweep)",
                        text=[n.split("(")[1].rstrip(")") for n in sweep.index], textposition="top right",
                        line=dict(color=ML_COLOR, width=2), marker=dict(size=9))
        for name, color, symbol in [(algo_name("Persistence", MAIN_LAM), SLOTS[1], "diamond"),
                                    (algo_name("Oracle", MAIN_LAM), SLOTS[2], "star")]:
            if name in r.index:
                fig.add_scatter(x=[r.loc[name, "energy_kwh"]], y=[r.loc[name, "avg_wait_s"]], mode="markers",
                                name=name, marker=dict(size=12, color=color, symbol=symbol))
        for name, color in [(MAIN, ML_COLOR), (algo_name("Persistence", MAIN_LAM), SLOTS[1]),
                            (algo_name("Oracle", MAIN_LAM), SLOTS[2])]:
            if name in per_seed.index.get_level_values(0):
                seeds = per_seed.loc[name]
                fig.add_scatter(x=seeds.energy_kwh, y=seeds.avg_wait_s, mode="markers", showlegend=False,
                                marker=dict(size=7, color=color, opacity=0.35),
                                hovertemplate=name + " seed %{text}: %{y:.1f} s, %{x:.1f} kWh<extra></extra>",
                                text=[f"{int(x)}" for x in seeds.index])
        classic = r[~r.index.map(is_ml)]
        fig.add_scatter(x=classic.energy_kwh, y=classic.avg_wait_s, mode="markers+text", name="classical",
                        text=list(classic.index), textposition="bottom center",
                        marker=dict(size=10, color=CLASSIC_COLOR))
        fig.update_layout(title="Trade-off: full day, mean of 5 test days (down-left is better)",
                          xaxis_title="energy (kWh)", yaxis_title="average wait (s)")
        st.plotly_chart(style(fig, 460), width="stretch")
        st.caption("For λ = 0.5 and the two forecast variants, the large marker is the average of the 5 training runs "
                   "and the faint dots are the individual runs, so you can see how much the result varies. All five "
                   "runs of the ML controller beat Nearest Car on both waiting and energy. Without a forecast the "
                   "controller still beats Nearest Car, so most of the gain comes from RL; the LSTM adds a smaller "
                   "improvement, and a perfect forecast shows there is more to gain from better forecasting.")

    runs = {p.stem: load_json(p) for p in sorted((ARTIFACTS / "rl").glob("*_lam*.json"))}
    if not runs:
        st.warning("Run `python -m scripts.train_rl` first.")
    else:
        st.subheader("Inside one training run")
        st.markdown("Pick a training run to see how it improved over time. The two charts show its waiting time "
                    "and energy on the validation days, measured every 10 training days, with Nearest Car as a "
                    "dashed reference line.")
        labels = {k: f'{algo_name(v["source"], v["lam"])} · seed {v["seed"]}' for k, v in runs.items()}
        main = f"{MAIN} · seed 0"
        key = st.selectbox("Training run", list(runs), format_func=labels.get,
                           index=list(labels.values()).index(main) if main in labels.values() else 0)
        info = runs[key]
        curve = pd.DataFrame(info["curve"])
        val = curve.dropna(subset=["val_wait_s"])
        base = info["nearest_car_val"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Validation wait (s)", "Validation energy (kWh)"])
        for col, (k, b) in enumerate([("val_wait_s", base["val_wait_s"]), ("val_energy_kwh", base["val_energy_kwh"])]):
            fig.add_scatter(x=val.episode, y=val[k], mode="lines+markers", line=dict(color=ML_COLOR, width=2),
                            showlegend=False, row=1, col=col + 1)
            fig.add_hline(y=b, line=dict(color=NEUTRAL, dash="dash"), annotation_text="Nearest Car",
                          annotation_position="top left", row=1, col=col + 1)
        fig.update_xaxes(title_text="episode (one simulated training day)")
        st.plotly_chart(style(fig, 360), width="stretch")
        st.caption("The curves are noisy because every training day is different and the controller is still "
                   "exploring, which is why the best version on the validation days is kept rather than the last.")
