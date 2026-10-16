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
from smartlift.experiment import (ARTIFACTS, DRIFT, DRIFT_TENANTS, TEST, TEST_WEEKDAYS, TRAIN, VAL,  # noqa: E402
                                  load_dataset)
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
    scenario = c[1].selectbox("Time of day", list(SCENARIOS), key=f"{key}_scenario",
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

tabs = st.tabs(["Overview", "Traffic", "Compare algorithms", "Watch a run", "Forecasting models", "RL agent"])

# ================================================================= Overview
with tabs[0]:
    st.header("An approach towards a smart lift system using ML: revamped")
    st.markdown("""
**Question:** can machine learning make a group of lifts serve people faster *and* use less energy?

**How:** a discrete-time simulator (1 tick = 1 s) of a 16-floor office with 4 lifts runs on synthetic traffic.
Every dispatching algorithm, classical or ML, plugs into the same simulator and is scored on the same
unseen test days.
""")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded", fontname="Helvetica", fontsize=11];
  traffic [label="Synthetic traffic\\n(non-homogeneous Poisson,\\npeaks, events, drift)"];
  counts [label="Hall calls per\\n5 min × floor × dir"];
  fc [label="Forecasters\\nHist. avg · LSTM ·\\nBi-LSTM+GRU · Active RNN"];
  sim [label="Simulator\\nlifts · doors · boarding\\nenergy (ISO 25745 split)"];
  disp [label="Dispatchers\\nFCFS · LOOK · Nearest Car ·\\nDestination · LSTM + RL"];
  out [label="Metrics\\nwait · journey · kWh"];
  traffic -> sim; traffic -> counts -> fc -> disp; disp -> sim [dir=both]; sim -> out;
}""")
    bench = ARTIFACTS / "bench" / "results.csv"
    if bench.exists():
        r = pd.read_csv(bench)
        full = r[r.scenario == "Full day"].groupby("algorithm", sort=False)[["avg_wait_s", "energy_kwh"]].mean()
        nc = full.loc["Nearest Car (ETA)"]
        st.subheader("Headline results (full day, mean of 5 test days; LSTM + RL also over 5 training seeds)")
        c = st.columns(4)
        c[0].metric("Best classical wait", f"{nc.avg_wait_s:.1f} s", help="Nearest Car (ETA)")
        if MAIN in full.index:
            ml = full.loc[MAIN]
            c[1].metric(f"{MAIN} wait", f"{ml.avg_wait_s:.1f} s",
                        f"{(ml.avg_wait_s / nc.avg_wait_s - 1) * 100:+.0f}% vs Nearest Car", delta_color="inverse")
            c[2].metric(f"{MAIN} energy", f"{ml.energy_kwh:.0f} kWh",
                        f"{(ml.energy_kwh / nc.energy_kwh - 1) * 100:+.0f}% vs Nearest Car", delta_color="inverse")
        c[3].metric("LOOK → Nearest Car energy", f"{nc.energy_kwh:.0f} kWh",
                    f"{(nc.energy_kwh / full.loc['LOOK (collective)'].energy_kwh - 1) * 100:+.0f}%",
                    delta_color="inverse")
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
    day = pick_day("traffic")
    st.markdown("Real lift logs aren't public, so traffic comes from a **non-homogeneous Poisson process** "
                "built from lift-traffic theory: a morning up-peak, lunch two-way traffic, an evening "
                "down-peak and interfloor meetings. Day-to-day occupancy changes, random conferences and "
                "a mid-dataset tenant change make the data realistic enough for the forecasters to learn from.")
    day_df = df[df.day == day]
    c = st.columns(3)
    c[0].metric("Trips on this day", f"{len(day_df):,}")
    c[1].metric("Busiest 5 minutes", int(series[day * BINS_PER_DAY:(day + 1) * BINS_PER_DAY].sum(1).max()))
    c[2].metric("Dataset size", f"{df.day.nunique()} days", f"{len(df):,} trips", delta_color="off")

    s = series[day * BINS_PER_DAY:(day + 1) * BINS_PER_DAY].reshape(BINS_PER_DAY, -1, 2)
    times = clock(np.arange(BINS_PER_DAY) * BIN_S)
    fig = go.Figure()
    for k, (label, color) in enumerate([("Up calls", SLOTS[0]), ("Down calls", SLOTS[1])]):
        fig.add_bar(x=times, y=s[:, :, k].sum(1), name=label, marker_color=color)
    fig.update_layout(barmode="stack", bargap=0.1, title=f"Hall calls per 5 minutes, {day_label(day)}",
                      hovermode="x unified")
    fig.update_xaxes(tickformat="%H:%M", hoverformat="%H:%M")
    st.plotly_chart(style(fig), width="stretch")

    heat = s.sum(2).T
    fig = go.Figure(go.Heatmap(z=heat, x=times, y=[f"Floor {f}" if f else "Lobby" for f in range(heat.shape[0])],
                               colorscale=[[0, "rgba(42,120,214,0.03)"], [1, "#0d3a73"]],
                               hovertemplate="%{y} at %{x}: %{z} calls<extra></extra>",
                               zmin=0, zmax=float(np.percentile(heat[heat > 0], 98)), zauto=False,
                               colorbar=dict(title="calls")))
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_layout(title="Where the calls come from (origin floor × time; colour capped at the 98th "
                            "percentile so upper floors stay visible next to the lobby)")
    st.plotly_chart(style(fig, 420), width="stretch")

    daily = df.groupby("day").size()
    colors = [NEUTRAL if d % 7 >= 5 else (SLOTS[1] if d >= DRIFT[0] else SLOTS[0]) for d in daily.index]
    fig = go.Figure(go.Bar(x=daily.index, y=daily.values, marker_color=colors,
                           hovertemplate="Day %{x}: %{y} trips<extra></extra>"))
    for (lo, hi), label in [(TRAIN, "train"), (VAL, "val"), (TEST, "test"), (DRIFT, "drift")]:
        fig.add_vrect(x0=lo - 0.5, x1=hi - 0.5, line_width=0, fillcolor="rgba(128,128,128,0.06)",
                      annotation_text=label, annotation_position="top left")
    fig.update_layout(title="Trips per day across the 140-day dataset (gray = weekends, orange = after tenant change)",
                      showlegend=False, bargap=0.15)
    st.plotly_chart(style(fig, 320), width="stretch")
    st.caption("Tenant change (from day 112): " + ", ".join(f"floor {f} → {p} people" for _, f, p in DRIFT_TENANTS))

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

# ================================================================= Forecasting
with tabs[4]:
    st.header("Forecasting hall-call demand")
    st.markdown("""
Predict the number of hall calls **per floor and direction in the next 5 minutes** from the last hour.
The two network architectures are the ones proposed in the original paper; outputs are log-rates trained
with **Poisson negative log-likelihood** because the targets are counts.
* **LSTM**: LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32)
* **Bi-LSTM + GRU**: BiLSTM(128) → Dropout(0.2) → GRU(64) → Dropout(0.2) → Dense(32)
* **Oracle** = the true Poisson rate used to generate the data: the *best any model could possibly do*.
""")
    metrics = load_json(ARTIFACTS / "forecast" / "metrics.json")
    if metrics is None:
        st.warning("Run `python -m scripts.train_forecasters` first.")
    else:
        t = pd.DataFrame(metrics["test"]).T
        t.columns = ["RMSE (per cell)", "MAE (per cell)", "Poisson NLL", "RMSE (building total)",
                     "MAPE % (busy bins)"]
        st.dataframe(t.style.format("{:.3f}").highlight_min(axis=0, color="rgba(42,120,214,.18)"),
                     width="stretch")
        hist = t.loc["Historical average", "RMSE (per cell)"]
        orac = t.loc["Oracle (true rate)", "RMSE (per cell)"]
        best = t.loc["Bi-LSTM + GRU", "RMSE (per cell)"]
        st.info(f"Bi-LSTM + GRU closes **{(hist - best) / (hist - orac) * 100:.0f}%** of the gap between the "
                "historical-average baseline and the theoretical best (oracle). It beats the average because it "
                "sees *today's* level: whether occupancy is high, the peak is early, or there's a conference.")

        pred = np.load(ARTIFACTS / "forecast" / "test_predictions.npz")
        n_show = BINS_PER_DAY * 2
        x = pred["idx"][:n_show]
        xt = [f"{DAYS[(i // BINS_PER_DAY) % 7]} {hhmm((i % BINS_PER_DAY) * BIN_S)}" for i in x]
        which = st.radio("Show", ["Whole building", "Lobby, going up"], horizontal=True)
        col = slice(None) if which == "Whole building" else [0]
        fig = go.Figure()
        fig.add_scatter(x=xt, y=pred["actual"][:n_show, col].sum(1), name="Actual", mode="lines",
                        line=dict(color=NEUTRAL, width=1))
        for name, color, dash in [("Historical average", SLOTS[3], "dot"), ("Bi-LSTM + GRU", SLOTS[0], "solid"),
                                  ("true_rate", SLOTS[2], "dash")]:
            fig.add_scatter(x=xt, y=pred[name][:n_show, col].sum(1), mode="lines",
                            name="Oracle (true rate)" if name == "true_rate" else name,
                            line=dict(color=color, width=2, dash=dash))
        fig.update_layout(title="Forecast vs actual: first two test days", hovermode="x unified",
                          yaxis_title="calls / 5 min")
        fig.update_xaxes(nticks=12)
        st.plotly_chart(style(fig, 400), width="stretch")

        fig = go.Figure()
        for (name, h), color in zip(metrics["history"].items(), SLOTS):
            ep = [r["epoch"] for r in h]
            fig.add_scatter(x=ep, y=[r["train_loss"] for r in h], name=f"{name} train", mode="lines",
                            line=dict(color=color, width=2, dash="dot"))
            fig.add_scatter(x=ep, y=[r["val_loss"] for r in h], name=f"{name} validation", mode="lines",
                            line=dict(color=color, width=2))
        fig.update_layout(title="Training history (Poisson NLL)", xaxis_title="epoch", hovermode="x unified")
        st.plotly_chart(style(fig, 340), width="stretch")

    st.subheader("Active-learning RNN: adapting when the building changes")
    al = load_json(ARTIFACTS / "forecast" / "active_learning.json")
    if al is None:
        st.warning("Run `python -m scripts.active_learning` first.")
    else:
        st.markdown(f"""
From day {DRIFT[0]} a new company fills floors 13–15 and floors 2–3 empty out. The LSTM was trained on the old
building. Each day it may learn from only **{al['budget']} of the day's 288 windows**. Which ones?
*Active* strategies pick the windows where the model is most **uncertain**, estimated with **Monte-Carlo
dropout** (20 stochastic forward passes, uncertainty = variance of the predictions).
""")
        focus = st.radio("Error on", ["Changed floors", "Whole building"], horizontal=True)
        key = "focus_rmse" if focus == "Changed floors" else "rmse"
        styles = {"Frozen": (NEUTRAL, "dot"), "Random": (SLOTS[1], "solid"), "Active (top-k)": (SLOTS[2], "solid"),
                  "Active (per hour)": (SLOTS[0], "solid"), "Full retrain": (SLOTS[3], "solid"),
                  "Oracle (true rate)": (NEUTRAL, "dash")}
        fig = go.Figure()
        for name, (color, dash) in styles.items():
            vals = pd.Series([d[key] for d in al["results"][name]]).rolling(3, min_periods=1).mean()
            fig.add_scatter(x=al["days"], y=vals, name=name, mode="lines", line=dict(color=color, width=2, dash=dash))
        fig.update_layout(title=f"Daily RMSE after the tenant change ({focus.lower()}, 3-day rolling mean)",
                          xaxis_title="day", yaxis_title="RMSE", hovermode="x unified")
        st.plotly_chart(style(fig, 400), width="stretch")
        rows = []
        for name, days in al["results"].items():
            r = [d[key] for d in days]
            rows.append({"Strategy": name, "Week 1": np.mean(r[:7]), "Week 4": np.mean(r[-7:])})
        st.dataframe(pd.DataFrame(rows).set_index("Strategy").style.format("{:.3f}"), width="stretch")
        st.caption("Findings: naive top-k uncertainty sampling is no better than not updating at all, because it "
                   "picks 24 near-identical windows from the same peak. Taking the most uncertain window *per hour* "
                   "adapts fastest in week 1; random sampling catches up by week 4.")

# ================================================================= RL
with tabs[5]:
    st.header("LSTM + RL: the machine-learning dispatcher")
    st.markdown(f"""
One agent, trained by trial and error in the simulator on *training* days only, makes both group-control decisions.
λ (seconds of waiting per Wh) says how much energy matters compared with waiting.

**1. Assignment:** a hall call arrives, which car serves it?
* cost(car) = Q-network's predicted wait (learned) + λ × extra energy of the detour (from the physics model)
* features per car: ETA, load, committed stops, distance, heading our way, idle / doors open, lobby call
* learned as a contextual bandit (γ = 0): the target is the passenger's realised wait

**2. Positioning:** a car is free, what should it do? **{" · ".join(ACTIONS)}**
* state: the car's floor and standby status, **the LSTM forecast of hall calls in the next 5 minutes**
  (building total, lobby, this floor, the busiest floor), other free / standby cars, people waiting
* cost until the car's next decision: everyone's waiting time + λ × this car's energy
* SMDP Q-learning (decisions at irregular times): Q(s,a) ← cost + e^(−τ/300 s) · min Q(s′,a′),
  Double DQN, replay buffer, slowly-updated target network
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
        st.caption("Ablations at the same λ: *no forecast* sees only the last 5 minutes of calls; "
                   "*oracle* sees the true demand rate, the best any forecast could do. The λ = 0.5 agents were "
                   "each trained with 5 seeds: big markers are the mean, faint dots are individual seeds.")

    runs = {p.stem: load_json(p) for p in sorted((ARTIFACTS / "rl").glob("*_lam*.json"))}
    if not runs:
        st.warning("Run `python -m scripts.train_rl` first.")
    else:
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
        acts = np.array(curve.actions.tolist())[-20:].sum(axis=0)
        fig = go.Figure(go.Bar(x=ACTIONS, y=acts / acts.sum() * 100, marker_color=ML_COLOR))
        fig.update_layout(title="What free cars chose (last 20 training days, % of decisions)", yaxis_title="%")
        st.plotly_chart(style(fig, 300), width="stretch")
        st.caption("The best validation checkpoint (lowest wait + λ·energy) is kept. "
                   "Training days include ε-greedy exploration (random actions).")
