"""Pre-Race — predict an UPCOMING (or hypothetical) race before it happens.

The fan picks a circuit and an optional date, sets the starting grid (we seed it
from last year's running of that circuit and let them reorder it), dials in the
expected track temperature / wet conditions, and the race simulator rolls out the
full field thousands of times to give a predicted finishing order with win and
podium chances — plus how many places each driver is expected to gain or lose
from where they start.

This is a true pre-race tool: each driver's base pace comes from RECENT FORM and
never uses any lap from the race we're predicting. Strategy auditing (ranking
candidate pit plans, the optimal-pit map) lives on the Post-Race page instead.
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
    from src.simulation.engine import RaceSpec, DriverSpec, simulate
    from src.models.tyre.fit import predict_degradation
except Exception as e:                                    # pragma: no cover
    _ENGINE_OK = False
    _ENGINE_ERR = str(e)


# ---------------------------------------------------------------------------
# Local data helpers (this page only) — find circuits, the latest lineup, the
# previous running of a circuit, and a past race's finishing order.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _circuit_runnings() -> dict[str, tuple[int, int]]:
    """circuit name -> most recent (season, round) it was run in the data."""
    out: dict[str, tuple[int, int]] = {}
    for s in T.list_seasons():
        for r in T.list_rounds(s):
            c = T.race_circuit(s, r)
            if c is None:
                continue
            if c not in out or (s, r) > out[c]:
                out[c] = (s, r)
    return out


@st.cache_data(show_spinner=False)
def _latest_lineup() -> tuple[int, pd.DataFrame]:
    """(latest_season, DataFrame[Driver, Team]) — the newest season's full entry
    list, taken from that season's most recent race."""
    seasons = T.list_seasons()
    season = seasons[-1]
    rounds = T.list_rounds(season)
    df = T.load_race(season, rounds[-1])
    lineup = df[["Driver", "Team"]].drop_duplicates().reset_index(drop=True)
    return season, lineup


@st.cache_data(show_spinner=False)
def _finish_order(season: int, rnd: int) -> list[str]:
    """Finishing order (winner first) of a past race, derived from lap data.

    No Position column exists, so classify by laps completed (more = ahead) then
    cumulative lap time at the final lap (faster = ahead). DNFs fall to the back.
    """
    df = T.load_race(season, rnd)
    if df is None or df.empty:
        return []
    laps = df.dropna(subset=["LapTimeSeconds"])
    agg = laps.groupby("Driver").agg(
        last_lap=("LapNumber", "max"),
        total_time=("LapTimeSeconds", "sum"),
    ).reset_index()
    agg = agg.sort_values(["last_lap", "total_time"], ascending=[False, True])
    return agg["Driver"].tolist()


def _seed_grid(latest_lineup: pd.DataFrame, prior_season: int, prior_round: int) -> pd.DataFrame:
    """Default starting grid for the latest lineup, seeded from last year's race.

    Drivers who ran last year's race at this circuit keep that finishing order;
    any latest-season driver who did NOT (new / replaced seat) drops to the back,
    in lineup order. Returns DataFrame[Grid, Driver, Team].
    """
    order = _finish_order(prior_season, prior_round)
    present = set(latest_lineup["Driver"])
    seeded = [d for d in order if d in present]                  # last year's order
    extras = [d for d in latest_lineup["Driver"] if d not in seeded]  # new seats -> back
    ordered = seeded + extras
    team_of = dict(zip(latest_lineup["Driver"], latest_lineup["Team"]))
    return pd.DataFrame(
        {"Grid": range(1, len(ordered) + 1),
         "Driver": ordered,
         "Team": [team_of.get(d, "") for d in ordered]}
    )


st.markdown("## Pre-Race prediction")
T.section("Pre-Race", sub="Predict an upcoming or hypothetical race: pick a circuit, set the "
                         "starting grid and conditions, then see the projected finishing order.")

if not _ENGINE_OK:
    T.warn(f"Our race simulator is unavailable: <code>{_ENGINE_ERR}</code>")
    st.stop()

curves = T.load_tyre_curves()
if curves is None:
    T.warn("The tyre-wear model is missing — run <code>python -m src.models.tyre.fit</code> first.")
    st.stop()

runnings = _circuit_runnings()
if not runnings:
    T.warn("No race data found to build a grid from.")
    st.stop()
latest_season, latest_lineup = _latest_lineup()
era = T.era_for_season(latest_season)


# ---------------------------------------------------------------------------
# 1. Circuit + date
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<p class="f1-label">Race setup</p>', unsafe_allow_html=True)
    circuit = st.selectbox("Circuit", sorted(runnings.keys()),
                           help="Pick the track for the race you want to predict.")
    prior_season, prior_round = runnings[circuit]
    st.caption(f"Latest lineup: **{latest_season} season** "
               f"({'2026+ active-aero cars' if era else '2022–2025 cars'}).")
    st.caption(f"Grid seeded from the last race here: **{prior_season} season**.")

    race_date = st.text_input("Race date (YYYY-MM-DD, optional)", key="prerace_date",
                              help="Used only to fetch a weather forecast for the conditions panel.")

    n_drivers = st.slider("Cars in the race", 4, len(latest_lineup), len(latest_lineup))
    n_rollouts = st.slider("Simulation detail (more = steadier numbers)",
                           200, 3000, 1000, step=200,
                           help="How many times we re-run the race to average out luck.")
    if circuit in CIRCUIT_LAPS:
        n_laps = CIRCUIT_LAPS[circuit]
        st.caption(f"Race distance: **{n_laps} laps** (the real length of this race).")
    else:
        n_laps = st.slider("Race laps", 30, 78, 57)


# ---------------------------------------------------------------------------
# Base pace from recent FORM — never uses this race's own laps.
# form_base_pace withholds the target race's laps; we pass last year's running of
# this circuit so the circuit baseline + form are right, then map that pace onto
# the LATEST-season lineup (so new drivers get the field-median fallback pace).
# ---------------------------------------------------------------------------

form = T.form_base_pace(prior_season, prior_round)
if form.empty:
    T.warn("Not enough past data to estimate driver pace for this circuit yet.")
    st.stop()

pace_of = dict(zip(form["Driver"], form["base_pace_s"]))
fallback_pace = float(form["base_pace_s"].median())
baseline_src = str(form["baseline_src"].iloc[0]) if "baseline_src" in form.columns and len(form) else ""

# Latest-season entry list with a base pace for everyone (median for new seats).
entries = latest_lineup.copy()
entries["base_pace_s"] = entries["Driver"].map(pace_of)
entries["is_new"] = entries["base_pace_s"].isna()
entries["base_pace_s"] = entries["base_pace_s"].fillna(fallback_pace)


# ---------------------------------------------------------------------------
# 2. Starting grid — seed from last year, let the user reorder it
# ---------------------------------------------------------------------------

T.section("Starting grid", f"{circuit} · hypothetical {latest_season} race",
          sub="We don't predict qualifying, so you set the grid. It's pre-filled from last "
              "year's finishing order here; drivers new to this race start at the back. "
              "Edit the 'Grid' numbers to reorder, then the simulator uses your order.")

# Build / refresh the seeded grid; recompute when the circuit changes.
seed = _seed_grid(latest_lineup, prior_season, prior_round)
seed = seed[seed["Driver"].isin(entries["Driver"])].reset_index(drop=True)
seed = seed.head(n_drivers).reset_index(drop=True)
seed["Grid"] = range(1, len(seed) + 1)

grid_key = f"grid_{circuit}_{n_drivers}"
edited = st.data_editor(
    seed,
    key=grid_key,
    width="stretch",
    hide_index=True,
    disabled=["Driver", "Team"],
    column_config={
        "Grid": st.column_config.NumberColumn(
            "Grid", min_value=1, max_value=len(seed), step=1,
            help="Starting position. Lower = closer to the front."),
        "Driver": st.column_config.TextColumn("Driver"),
        "Team": st.column_config.TextColumn("Team"),
    },
)

# Resolve the edited grid into clean 1..N starting positions (stable tie-break by
# the original seed order, so duplicate numbers don't crash the sim).
grid_df = edited.copy()
grid_df["Grid"] = pd.to_numeric(grid_df["Grid"], errors="coerce")
grid_df["_seed"] = range(len(grid_df))
grid_df = grid_df.sort_values(["Grid", "_seed"], na_position="last").reset_index(drop=True)
grid_df["grid_pos"] = range(1, len(grid_df) + 1)
# grid_df already carries Driver/Team from the editor; only pull in pace + new-seat flag.
grid_df = grid_df.merge(entries[["Driver", "base_pace_s", "is_new"]], on="Driver", how="left")

new_names = grid_df.loc[grid_df["is_new"] == True, "Driver"].tolist()  # noqa: E712
if new_names:
    st.caption("New / changed seats this season (placed at the back by default, "
               "given a mid-grid pace estimate): **" + ", ".join(new_names) + "**.")
if baseline_src:
    st.caption(f"Driver pace from recent form on this circuit's baseline ({baseline_src}); "
               "no laps from the race we're predicting are used.")


# ---------------------------------------------------------------------------
# 3. Conditions — weather panel + track-temperature control + wet toggle
# ---------------------------------------------------------------------------

T.section("Conditions", "Weather & track temperature",
          sub="Fetch a forecast for your date, or set the track temperature by hand. Hotter "
              "track → tyres wear faster. Use the wet toggle for a rainy race.")


def _med(df, col):
    return float(df[col].median()) if col in df.columns and df[col].notna().any() else None


# Historical session conditions from last year's race here (a sensible default).
_wx = T.load_race(prior_season, prior_round)
hist_track = hist_air = None
wx_meta = {}
if _wx is not None and not _wx.empty:
    hist_track = _med(_wx, "TrackTemp")
    hist_air = _med(_wx, "AirTemp")
    wx_meta = {
        "Humidity": _med(_wx, "Humidity"),
        "WindSpeed": _med(_wx, "WindSpeed"),
        "Rainfall": bool(_wx["Rainfall"].any()) if "Rainfall" in _wx.columns and _wx["Rainfall"].notna().any() else None,
    }

fc = st.session_state.get("prerace_forecast")
with st.expander("Fetch a weather forecast for this race"):
    st.caption("Looks up the forecast for the circuit and date you set in the sidebar "
               "(works up to ~16 days ahead; older dates use the weather archive).")
    if st.button("Fetch forecast", key="fc_btn"):
        if not race_date.strip():
            st.warning("Enter a race date in the sidebar first.")
        else:
            try:
                from src.pipeline.weather import session_weather
                got = session_weather(circuit, race_date.strip(), mode="auto")
                if got.get("source") == "unavailable":
                    st.warning("No forecast available for this circuit/date.")
                else:
                    st.session_state["prerace_forecast"] = got
                    fc = got
                    st.success(f"Got the weather forecast for {race_date.strip()}.")
            except Exception as e:                                # pragma: no cover
                st.warning(f"Forecast fetch failed: {e}")

# Default track temp: forecast estimate > last-year session median > 30°C.
if fc and fc.get("TrackTempEst") is not None:
    default_track = float(fc["TrackTempEst"])
elif hist_track is not None:
    default_track = hist_track
else:
    default_track = 30.0

mcols = st.columns(4)
_air = (fc.get("AirTemp") if fc else None) or hist_air
mcols[0].metric("Air temp", f"{_air:.0f}°C" if _air is not None else "—")
mcols[1].metric("Track temp (last year)", f"{hist_track:.0f}°C" if hist_track is not None else "—")
_hum = (fc.get("Humidity") if fc else None) or wx_meta.get("Humidity")
mcols[2].metric("Humidity", f"{_hum:.0f}%" if _hum is not None else "—")
_wind = (fc.get("WindSpeed") if fc else None) or wx_meta.get("WindSpeed")
mcols[3].metric("Wind", f"{_wind:.0f} km/h" if _wind is not None else "—")

_rain = (fc.get("Rainfall") if fc else wx_meta.get("Rainfall"))
if fc:
    st.caption(f"Showing a **{fc['source']}** forecast for {race_date.strip()}. "
               f"Estimated track temp **{fc.get('TrackTempEst', float('nan')):.0f}°C**.")
if _rain:
    st.caption("Rain was recorded here last time — consider the wet-race toggle below.")

cc1, cc2 = st.columns([3, 2])
track_temp = cc1.slider(
    "Track temperature for the race (°C)",
    10.0, 60.0, float(round(default_track)), step=1.0,
    help="Hotter track wears tyres faster. This feeds the tyre-wear view and the race simulator.",
)
wet = cc2.toggle("Wet race", value=bool(_rain),
                 help="Start everyone on intermediate (wet) tyres.")
if wet:
    T.warn("Wet running is based on limited data — these numbers are <b>indicative only</b>. "
           "A real wet race is unpredictable and usually needs more, earlier stops.")
if hist_track is not None and abs(track_temp - hist_track) >= 1:
    cc1.caption(f"That's **{track_temp - hist_track:+.0f}°C** vs last year's race here "
                f"({hist_track:.0f}°C).")


# ---------------------------------------------------------------------------
# 4. Tyre-degradation curves (useful pre-race)
# ---------------------------------------------------------------------------

T.section("Tyres", "How fast the tyres wear",
          sub=f"Lap-time loss as a tyre ages at **{track_temp:.0f}°C** track temp, with the "
              "model's confidence band. Dashed = last year's measured temperature.")

ages = np.arange(1, 36, dtype=float)
fig_tyre = go.Figure()
for comp in ["SOFT", "MEDIUM", "HARD"]:
    try:
        mid, lo, hi = predict_degradation(curves, comp, circuit, era, ages, track_temp=track_temp)
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
    if hist_track is not None and abs(track_temp - hist_track) >= 1:
        try:
            base_mid, _, _ = predict_degradation(curves, comp, circuit, era, ages, track_temp=hist_track)
            fig_tyre.add_trace(go.Scatter(x=ages, y=base_mid, mode="lines",
                                          name=f"{comp} (last year)", showlegend=False,
                                          line=dict(color=color, width=1.2, dash="dot"),
                                          opacity=0.6, hoverinfo="skip"))
        except Exception:
            pass
fig_tyre.update_layout(xaxis_title="Tyre age (laps)", yaxis_title="Lap time lost vs fresh (s)")
st.plotly_chart(T.style_fig(fig_tyre, 360), width="stretch")
st.markdown(
    "**How to read this chart:** each line is how much *slower* a tyre gets as it ages "
    "(x-axis = laps on that tyre, y-axis = seconds lost vs a fresh set). A line that climbs "
    "steeply wears out sooner, so that tyre needs changing earlier; a flatter line lasts "
    "longer. The shaded band is the range of uncertainty. The **dashed** line is the same "
    "tyre at last year's temperature, so you can see how much the slider changed things."
)


# ---------------------------------------------------------------------------
# 5. Strategy assigned to each driver (shown to the user)
# ---------------------------------------------------------------------------

def _default_strategy(start_compound: str) -> list[tuple[int, str]]:
    """Sensible default 1-stop: switch around the middle of the race."""
    pit = max(1, int(n_laps * 0.45))
    other = "HARD" if start_compound != "HARD" else "MEDIUM"
    return [(pit, other)]


# Wet race -> everyone starts on intermediates (limited-data, indicative only).
start_compound = "INTERMEDIATE" if wet else "MEDIUM"
pit_lap = max(1, int(n_laps * 0.45))
to_compound = "INTERMEDIATE" if wet else "HARD"
strategy_label = (f"Start INTERMEDIATE → INTERMEDIATE (around lap {pit_lap})" if wet
                  else f"Start MEDIUM → HARD (around lap {pit_lap})")


def _make_field_specs(grid: pd.DataFrame) -> list:
    specs = []
    for _, r in grid.iterrows():
        if wet:
            strat = [(pit_lap, "INTERMEDIATE")]
        else:
            strat = _default_strategy(start_compound)
        specs.append(DriverSpec(
            driver=r["Driver"], base_pace_s=float(r["base_pace_s"]),
            start_compound=start_compound, strategy=strat,
            grid_pos=int(r["grid_pos"]),
            grid_gap_s=float((int(r["grid_pos"]) - 1) * 0.4),
        ))
    return specs


# ---------------------------------------------------------------------------
# 6. Simulate -> predicted finishing order
# ---------------------------------------------------------------------------

T.section("Prediction", "Projected finishing order",
          sub="The simulator runs the whole field many times from your grid and conditions. "
              f"Everyone uses the same default plan: **{strategy_label}**.")

st.caption(f"Assumed pit plan for every car: **{strategy_label}**."
           + ("  Wet-race numbers are indicative only." if wet else ""))

if st.button("Predict the race", type="primary"):
    specs = _make_field_specs(grid_df)
    spec = RaceSpec(circuit=circuit, era=era, n_laps=n_laps, drivers=specs, track_temp=track_temp)
    with st.spinner(f"Simulating the race {n_rollouts:,} times…"):
        res = simulate(spec, n_rollouts=n_rollouts)
    tbl = res.table()
    grid_map = dict(zip(grid_df["Driver"], grid_df["grid_pos"]))
    team_map = dict(zip(grid_df["Driver"], grid_df["Team"]))
    tbl["grid_pos"] = tbl["driver"].map(grid_map)
    tbl["Team"] = tbl["driver"].map(team_map)
    # Places gained vs grid: + = moved up the order from where they started.
    tbl["places_gained"] = tbl["grid_pos"] - tbl["mean_finish"]
    st.session_state["prerace_pred"] = {
        "tbl": tbl, "circuit": circuit, "season": latest_season,
        "strategy": strategy_label, "wet": wet,
    }

state = st.session_state.get("prerace_pred")
if state and state.get("circuit") == circuit:
    tbl = state["tbl"].copy()

    # Friendly finishing-order table (SHARED CONTRACT: render via theme helper).
    show = tbl[["driver", "Team", "grid_pos", "win_prob", "podium_prob",
                "mean_finish", "p5_finish", "p95_finish"]].copy()
    show["places_gained"] = (tbl["places_gained"]).round(1)
    friendly = T.friendly_finish_table(show)
    friendly = friendly.rename(columns={"driver": "Driver", "places_gained": "Places gained"})
    friendly.insert(0, "Predicted finish", range(1, len(friendly) + 1))

    st.dataframe(
        friendly, width="stretch", hide_index=True,
        column_config={
            "Win chance": st.column_config.TextColumn(help=T.FRIENDLY_FINISH_HELP["Win chance"]),
            "Podium chance": st.column_config.TextColumn(help=T.FRIENDLY_FINISH_HELP["Podium chance"]),
            "Avg finish": st.column_config.NumberColumn(help=T.FRIENDLY_FINISH_HELP["Avg finish"]),
            "Best case": st.column_config.NumberColumn(help=T.FRIENDLY_FINISH_HELP["Best case"]),
            "Worst case": st.column_config.NumberColumn(help=T.FRIENDLY_FINISH_HELP["Worst case"]),
            "Places gained": st.column_config.NumberColumn(
                help="Starting position minus predicted average finish. "
                     "Positive = expected to gain places; negative = expected to lose places."),
        },
    )
    st.caption(
        "How to read: rows are sorted by predicted finish (top = most likely race winner). "
        "**Win chance** / **Podium chance** are how often that driver won or finished top-3 across "
        "all the simulated races. **Places gained** is how many spots they're expected to move from "
        "their starting position (positive = gains places)."
        + ("  These wet-race figures are indicative only." if state.get("wet") else "")
    )

    # Win / podium bar chart.
    bars = tbl.sort_values("win_prob", ascending=False).head(min(12, len(tbl)))
    cga, cgb = st.columns(2)
    with cga:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=bars["driver"], x=bars["win_prob"] * 100, orientation="h",
            marker=dict(color=bars["win_prob"], colorscale=[[0, "#3a0e0e"], [1, T.RED]]),
            text=[f"{v*100:.0f}%" for v in bars["win_prob"]], textposition="outside",
        ))
        fig.update_layout(title="Win chance", xaxis_title="%", yaxis=dict(autorange="reversed"))
        st.plotly_chart(T.style_fig(fig, 340), width="stretch")
    with cgb:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=bars["driver"], x=bars["podium_prob"] * 100, orientation="h",
            marker=dict(color=bars["podium_prob"], colorscale=[[0, "#3a0e0e"], [1, T.RED]]),
            text=[f"{v*100:.0f}%" for v in bars["podium_prob"]], textposition="outside",
        ))
        fig.update_layout(title="Podium chance", xaxis_title="%", yaxis=dict(autorange="reversed"))
        st.plotly_chart(T.style_fig(fig, 340), width="stretch")
    st.caption(
        "How to read: each bar is one driver. Across thousands of simulated races, a longer / "
        "redder bar means a higher chance of winning (left) or finishing on the podium (right)."
    )

    # Places gained / lost chart.
    gl = tbl.sort_values("places_gained", ascending=False)
    fig_gl = go.Figure()
    fig_gl.add_trace(go.Bar(
        x=gl["driver"], y=gl["places_gained"],
        marker=dict(color=["#43B02A" if v >= 0 else T.RED for v in gl["places_gained"]]),
        text=[f"{v:+.1f}" for v in gl["places_gained"]], textposition="outside",
    ))
    fig_gl.update_layout(title="Places gained / lost vs starting grid",
                         xaxis_title="Driver", yaxis_title="Places (+ = gained)")
    st.plotly_chart(T.style_fig(fig_gl, 340), width="stretch")
    st.caption(
        "How to read: green bars are drivers expected to move UP the order from where they start; "
        "red bars are drivers expected to drop back. Taller bar = bigger expected swing."
    )
else:
    st.info("Set your grid and conditions above, then press **Predict the race**.")
