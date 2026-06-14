"""Pre-Race — plan and rank a pit strategy before lights-out.

Pick circuit + era, build a driver lineup with base pace derived from green-flag
median laps, inspect tyre-degradation curves, then run the Monte-Carlo engine's
recommend() over candidate strategies and view the MDP pit-policy heatmap.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent          # dashboard/
_ROOT = _HERE.parent                                     # repo root
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme as T

T.apply_theme()
T.register_plotly_template()

# Official race-lap distance per circuit (keyed by CircuitKey == FastF1 EventName).
# Used to auto-fill race length so the user is not asked for a value that is fixed.
# Unknown circuits fall back to a manual slider.
CIRCUIT_LAPS = {
    "Bahrain Grand Prix": 57, "Saudi Arabian Grand Prix": 50,
    "Australian Grand Prix": 58, "Japanese Grand Prix": 53,
    "Chinese Grand Prix": 56, "Miami Grand Prix": 57,
    "Emilia Romagna Grand Prix": 63, "Monaco Grand Prix": 78,
    "Canadian Grand Prix": 70, "Spanish Grand Prix": 66,
    "Austrian Grand Prix": 71, "British Grand Prix": 52,
    "Hungarian Grand Prix": 70, "Belgian Grand Prix": 44,
    "Dutch Grand Prix": 72, "Italian Grand Prix": 53,
    "Azerbaijan Grand Prix": 51, "Singapore Grand Prix": 62,
    "United States Grand Prix": 56, "Mexico City Grand Prix": 71,
    "São Paulo Grand Prix": 71, "Las Vegas Grand Prix": 50,
    "Qatar Grand Prix": 57, "Abu Dhabi Grand Prix": 58,
    "French Grand Prix": 53,
}

# Heavy src imports guarded so the page degrades gracefully.
_ENGINE_OK = True
_ENGINE_ERR = ""
try:
    from src.simulation.engine import RaceSpec, DriverSpec, simulate, recommend
    from src.models.tyre.fit import predict_degradation
    from src.models.pit_strategy.mdp import build_race_model, solve, COMPOUNDS, MAX_AGE
except Exception as e:                                    # pragma: no cover
    _ENGINE_OK = False
    _ENGINE_ERR = str(e)

st.markdown("## Pre-Race strategy lab")
T.section("Setup", sub="Choose the race, build the field, then let the simulator rank strategies.")

if not _ENGINE_OK:
    T.warn(f"Simulation engine unavailable: <code>{_ENGINE_ERR}</code>")
    st.stop()

curves = T.load_tyre_curves()
if curves is None:
    T.warn("models/tyre_curves.joblib is missing — run <code>python -m src.models.tyre.fit</code> first.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar: race + field controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<p class="f1-label">Race setup</p>', unsafe_allow_html=True)
    seasons = T.list_seasons()
    default_season = 2025 if 2025 in seasons else seasons[-1]
    season = st.selectbox("Season (era source)", seasons, index=seasons.index(default_season))
    era = T.era_for_season(season)
    st.caption(f"Regulation era: **{era}** ({'2022–2025 ground-effect' if era == 0 else '2026+ active-aero'})")

    circuits = T.season_circuits(season)
    if not circuits:
        st.error("No races for this season.")
        st.stop()
    rnd = st.selectbox(
        "Circuit",
        list(circuits.keys()),
        format_func=lambda r: f"R{r:02d} · {circuits[r]}",
    )
    circuit = circuits[rnd]

    n_drivers = st.slider("Drivers in field", 4, 24, 22)
    n_rollouts = st.slider("Monte-Carlo rollouts", 200, 3000, 1000, step=200)
    if circuit in CIRCUIT_LAPS:
        n_laps = CIRCUIT_LAPS[circuit]
        st.caption(f"Race distance: **{n_laps} laps** (official, fixed for this circuit).")
    else:
        n_laps = st.slider("Race laps", 30, 78, 57)

# Pre-race base pace from each driver's recent form — NOT this race's own laps.
base = T.form_base_pace(season, rnd)
if base.empty:
    T.warn("Not enough prior data to derive a pre-race base pace for this race.")
    st.stop()

field = base.head(n_drivers).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Driver lineup
# ---------------------------------------------------------------------------

T.section("Grid", f"{circuit} · {season}",
          sub="Base pace = each driver's recent form (last 5 races, gap to field) on this "
              "circuit's baseline — no laps from this race used. Grid from lap-1 order.")

lineup = field[["Driver", "Team", "base_pace_s", "grid_pos"]].copy()
lineup["base_pace_s"] = lineup["base_pace_s"].round(3)
lineup = lineup.sort_values("grid_pos").reset_index(drop=True)
st.dataframe(
    lineup.rename(columns={"base_pace_s": "base pace (s)", "grid_pos": "grid"}),
    width="stretch", hide_index=True,
)
if "baseline_src" in field.columns and len(field):
    st.caption(
        f"Circuit baseline from: **{field['baseline_src'].iloc[0]}**. "
        "Drivers with no recent history fall back to the field-median form delta."
    )

# Session weather (median per race — in-race temp does not swing drastically).
_wx = T.load_race(season, rnd)
if _wx is not None and not _wx.empty:
    _at = _wx["AirTemp"].dropna() if "AirTemp" in _wx.columns else None
    _tt = _wx["TrackTemp"].dropna() if "TrackTemp" in _wx.columns else None
    _bits = []
    if _at is not None and len(_at):
        _bits.append(f"Air **{_at.median():.0f}°C**")
    if _tt is not None and len(_tt):
        _bits.append(f"Track **{_tt.median():.0f}°C**")
    if _bits:
        st.caption("Session conditions: " + " · ".join(_bits))

ego = st.selectbox("Ego driver (strategy target)", lineup["Driver"].tolist())

# default strategy schedule for rivals: simple 1-stop on the half-distance
def _default_strategy(start_compound: str) -> list[tuple[int, str]]:
    pit = int(n_laps * 0.45)
    other = "HARD" if start_compound != "HARD" else "MEDIUM"
    return [(pit, other)]


def _make_field_specs() -> list:
    specs = []
    for _, r in field.iterrows():
        start = "MEDIUM"
        specs.append(DriverSpec(
            driver=r["Driver"], base_pace_s=float(r["base_pace_s"]),
            start_compound=start, strategy=_default_strategy(start),
            grid_pos=int(r["grid_pos"]),
            grid_gap_s=float((r["grid_pos"] - 1) * 0.4),
        ))
    return specs


# ---------------------------------------------------------------------------
# Tyre degradation curves
# ---------------------------------------------------------------------------

T.section("Tyres", "Degradation curves",
          sub="Fuel-corrected lap-time loss vs tyre age, with the fitted 95% confidence band.")

ages = np.arange(1, 36, dtype=float)
fig_tyre = go.Figure()
for comp in ["SOFT", "MEDIUM", "HARD"]:
    try:
        mid, lo, hi = predict_degradation(curves, comp, circuit, era, ages)
    except Exception:
        continue
    color = T.COMPOUND_COLORS[comp]
    rgb = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    fill = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)"
    fig_tyre.add_trace(go.Scatter(x=ages, y=hi, mode="lines", line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"))
    fig_tyre.add_trace(go.Scatter(x=ages, y=lo, mode="lines", line=dict(width=0),
                                  fill="tonexty", fillcolor=fill,
                                  showlegend=False, hoverinfo="skip"))
    fig_tyre.add_trace(go.Scatter(x=ages, y=mid, mode="lines", name=comp,
                                  line=dict(color=color, width=2.4)))
fig_tyre.update_layout(xaxis_title="Tyre age (laps)", yaxis_title="Δ lap time vs fresh (s)")
st.plotly_chart(T.style_fig(fig_tyre, 360), width="stretch")

# ---------------------------------------------------------------------------
# Strategy candidates + recommendation
# ---------------------------------------------------------------------------

T.section("Strategy", "Candidate comparison",
          sub="The MDP proposes; the Monte-Carlo engine disposes — paired common-random-number rollouts.")

p1 = int(n_laps * 0.33)
p2a = int(n_laps * 0.28)
p2b = int(n_laps * 0.62)
candidates = {
    "1-stop M→H":   ("MEDIUM", [(int(n_laps * 0.45), "HARD")]),
    "1-stop S→H":   ("SOFT",   [(p1, "HARD")]),
    "2-stop M→H→H": ("MEDIUM", [(p2a, "HARD"), (p2b, "HARD")]),
    "2-stop S→M→H": ("SOFT",   [(p2a, "MEDIUM"), (p2b, "HARD")]),
}

with st.expander("Add a custom strategy"):
    cc1, cc2, cc3, cc4 = st.columns(4)
    cstart = cc1.selectbox("Start compound", ["SOFT", "MEDIUM", "HARD"], index=1, key="cstart")
    cp1 = cc2.number_input("Pit 1 lap", 1, n_laps, p1, key="cp1")
    cc1b = cc3.selectbox("→ compound", ["SOFT", "MEDIUM", "HARD"], index=2, key="cc1b")
    add_2nd = cc4.checkbox("2nd stop", key="add2")
    stops = [(int(cp1), cc1b)]
    if add_2nd:
        cd1, cd2 = st.columns(2)
        cp2 = cd1.number_input("Pit 2 lap", int(cp1) + 1, n_laps, min(int(cp1) + 18, n_laps - 1), key="cp2")
        cc2b = cd2.selectbox("→ compound ", ["SOFT", "MEDIUM", "HARD"], index=1, key="cc2b")
        stops.append((int(cp2), cc2b))
    if st.checkbox("Include custom strategy in ranking", key="incl_custom"):
        candidates["Custom"] = (cstart, stops)

if st.button("Run simulation", type="primary"):
    specs = _make_field_specs()
    spec = RaceSpec(circuit=circuit, era=era, n_laps=n_laps, drivers=specs)
    with st.spinner(f"Rolling out {n_rollouts:,} races per candidate…"):
        rec = recommend(spec, ego, candidates, n_rollouts=n_rollouts)
        # also a full-field sim with the ego on its best candidate for grid bars
        best = rec.iloc[0]
        ego_idx = next(i for i, d in enumerate(specs) if d.driver == ego)
        specs[ego_idx].start_compound = best["start"]
        specs[ego_idx].strategy = eval(best["stops"])  # noqa: S307 - our own serialized tuples
        field_res = simulate(RaceSpec(circuit=circuit, era=era, n_laps=n_laps, drivers=specs),
                             n_rollouts=n_rollouts)
    st.session_state["prerace"] = {
        "rec": rec, "field": field_res.table(), "ego": ego, "best": best["strategy"],
    }

state = st.session_state.get("prerace")
if state:
    rec = state["rec"]
    st.markdown(f"**Ranked strategies for {state['ego']}** — best: "
                f"<span style='color:{T.RED};font-weight:700'>{state['best']}</span>",
                unsafe_allow_html=True)
    show = rec.copy()
    show["win_prob"] = (show["win_prob"] * 100).round(1)
    show["podium_prob"] = (show["podium_prob"] * 100).round(1)
    show["mean_finish"] = show["mean_finish"].round(2)
    st.dataframe(
        show.rename(columns={"win_prob": "win %", "podium_prob": "podium %",
                             "mean_finish": "mean finish", "p50_time_s": "median time (s)"}),
        width="stretch", hide_index=True,
    )

    # horizontal win/podium bars (red scale)
    cga, cgb = st.columns(2)
    with cga:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=rec["strategy"], x=rec["win_prob"] * 100, orientation="h",
            marker=dict(color=rec["win_prob"], colorscale=[[0, "#3a0e0e"], [1, T.RED]]),
            text=[f"{v*100:.1f}%" for v in rec["win_prob"]], textposition="outside",
        ))
        fig.update_layout(title="Win probability", xaxis_title="%", yaxis=dict(autorange="reversed"))
        st.plotly_chart(T.style_fig(fig, 300), width="stretch")
    with cgb:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=rec["strategy"], x=rec["podium_prob"] * 100, orientation="h",
            marker=dict(color=rec["podium_prob"], colorscale=[[0, "#3a0e0e"], [1, T.RED]]),
            text=[f"{v*100:.1f}%" for v in rec["podium_prob"]], textposition="outside",
        ))
        fig.update_layout(title="Podium probability", xaxis_title="%", yaxis=dict(autorange="reversed"))
        st.plotly_chart(T.style_fig(fig, 300), width="stretch")

    with st.expander("Full-field finishing-order projection (ego on best strategy)"):
        ft = state["field"].copy()
        ft["win_prob"] = (ft["win_prob"] * 100).round(1)
        ft["podium_prob"] = (ft["podium_prob"] * 100).round(1)
        st.dataframe(ft, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# MDP policy heatmap
# ---------------------------------------------------------------------------

T.section("Policy", "MDP optimal pit map",
          sub="Exact backward induction: the optimal action at each (lap, tyre age) for a single car.")

pol_compound = st.selectbox("Tyre compound for policy view", ["SOFT", "MEDIUM", "HARD"], index=1)
two_used = st.checkbox("Second compound already used (rule satisfied)", value=False)

try:
    base_pace = float(field["base_pace_s"].median())
    model = build_race_model(circuit, era, n_laps, base_pace, curves)
    res = solve(model, start_compound=pol_compound)
    ci = COMPOUNDS.index(pol_compound)
    grid = res["policy"][1:, ci, :, int(two_used)]          # [lap, age]
    z = grid.T                                              # [age, lap]
    colorscale = [[0.0, "#2A2A33"], [0.25, "#2A2A33"],
                  [0.25, T.RED], [0.5, T.RED],
                  [0.5, T.COMPOUND_COLORS["MEDIUM"]], [0.75, T.COMPOUND_COLORS["MEDIUM"]],
                  [0.75, T.WHITE], [1.0, T.WHITE]]
    fig_pol = go.Figure(data=go.Heatmap(
        z=z, x=list(range(1, n_laps + 1)), y=list(range(MAX_AGE + 1)),
        zmin=-0.5, zmax=3.5, colorscale=colorscale,
        colorbar=dict(tickvals=[0, 1, 2, 3],
                      ticktext=["stay", "→SOFT", "→MED", "→HARD"]),
    ))
    fig_pol.update_layout(xaxis_title="Lap", yaxis_title=f"Tyre age on {pol_compound}")
    st.plotly_chart(T.style_fig(fig_pol, 400), width="stretch")
    st.markdown(
        "**How to read this map:** each cell is the optimal action for a single car "
        "at that point in the race — the colour says what to do when your tyre has reached "
        "that age (y-axis) on that lap (x-axis): **stay out**, or **pit to SOFT / MEDIUM / HARD**. "
        "Reading up a column shows when the optimal call flips from staying out to pitting — "
        "that boundary is your pit window."
    )
    st.caption(f"MDP optimum from start={pol_compound}: "
               f"{res['n_stops']} stop(s) — {res['stops']} · "
               f"avg lap {res['avg_lap_s']:.2f}s")
except Exception as e:
    T.warn(f"Could not build the MDP policy: <code>{e}</code>")
