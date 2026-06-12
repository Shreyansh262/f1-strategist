"""Post-Race — audit a finished race against the models.

Predicted-vs-actual MAE per driver, "was the actual strategy optimal?" against
the simulator, fitted-vs-actual tyre degradation, and error spikes correlated
with SC/VSC (TrackStatus != '1').
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
_ROOT = _HERE.parent
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme as T

T.page_config("Post-Race · F1 Strategist")
T.apply_theme()
T.register_plotly_template()

_OK = True
_ERR = ""
try:
    from src.pipeline.features import build_features, load_encoding_mappings
    from src.pipeline.validate import validate_laps
    from src.models.lap_time.train import get_X_y
    from src.models.tyre.fit import predict_degradation, prepare_stint_laps
    from src.simulation.engine import RaceSpec, DriverSpec, recommend
except Exception as e:                                    # pragma: no cover
    _OK = False
    _ERR = str(e)

st.markdown("## Post-Race analysis")
T.section("Audit", sub="What happened, and was it optimal? Predictions, strategy, tyres, error spikes.")

if not _OK:
    T.warn(f"Analysis modules unavailable: <code>{_ERR}</code>")
    st.stop()

with st.sidebar:
    st.markdown('<p class="f1-label">Race</p>', unsafe_allow_html=True)
    seasons = T.list_seasons()
    season = st.selectbox("Season", seasons, index=len(seasons) - 1)
    circuits = T.season_circuits(season)
    if not circuits:
        st.error("No races.")
        st.stop()
    rnd = st.selectbox("Round", list(circuits.keys()),
                       format_func=lambda r: f"R{r:02d} · {circuits[r]}")

df = T.load_race(season, rnd)
if df is None or df.empty:
    T.warn("Race parquet missing or empty.")
    st.stop()

circuit = str(df["CircuitKey"].iloc[0])
era = T.era_for_season(season)
n_laps = int(df["LapNumber"].max())
st.markdown(f"### {circuit} · {season}")


# ---------------------------------------------------------------------------
# Predicted vs actual MAE per driver
# ---------------------------------------------------------------------------

T.section("Accuracy", "Predicted vs actual MAE per driver",
          sub="LightGBM lap-time prediction error, green-flag laps. Lower is better.")

feats = None
model = T.load_lap_model()
if model is not None:
    try:
        cmap, tmap, dmap = load_encoding_mappings()
        feats = build_features(validate_laps(df.copy()), cmap, tmap, dmap)
        X, y = get_X_y(feats)
        feats = feats.copy()
        feats["yhat"] = model.predict(X)
        feats["abs_err"] = (feats["LapTimeSeconds"] - feats["yhat"]).abs()
    except Exception as e:
        T.warn(f"Lap-time prediction failed: <code>{e}</code>")
        feats = None

if feats is not None:
    # green-flag MAE per driver — TrackStatus carried via merge on lap keys
    meta = df[["Season", "RoundNumber", "Driver", "LapNumber", "TrackStatus"]].copy()
    m = feats.merge(meta, on=["Season", "RoundNumber", "Driver", "LapNumber"], how="left")
    green = m[m["TrackStatus"] == "1"]
    per_drv = green.groupby("Driver")["abs_err"].mean().sort_values()
    overall = green["abs_err"].mean()
    c1, c2 = st.columns([1, 3])
    c1.metric("Green MAE (race)", f"{overall:.2f}s")
    fig = go.Figure(go.Bar(
        x=per_drv.values, y=per_drv.index, orientation="h",
        marker=dict(color=per_drv.values, colorscale=[[0, T.WHITE], [1, T.RED]]),
    ))
    fig.add_vline(x=overall, line=dict(color=T.GREY, dash="dot"))
    fig.update_layout(xaxis_title="Mean abs error (s)", yaxis=dict(autorange="reversed"))
    c2.plotly_chart(T.style_fig(fig, max(300, 22 * len(per_drv))), width="stretch")


# ---------------------------------------------------------------------------
# Was the actual strategy optimal?
# ---------------------------------------------------------------------------

T.section("Strategy", "Was the actual strategy optimal?",
          sub="Extract the winner's actual stops, then rank it against alternatives in the simulator.")

base = T.green_base_pace(season, rnd)
if base.empty or len(base) < 2:
    st.caption("Not enough green-flag pace to run the strategy audit.")
else:
    n_drivers = min(10, len(base))
    field = base.head(n_drivers).reset_index(drop=True)

    # extract actual strategy of the fastest finisher
    df2 = df.sort_values(["Driver", "LapNumber"])
    totals = df2.groupby("Driver").agg(time=("LapTimeSeconds", "sum"),
                                       laps=("LapNumber", "max"))
    finishers = totals[totals["laps"] == totals["laps"].max()].nsmallest(1, "time")
    actual_stops = []
    actual_start = "MEDIUM"
    if len(finishers):
        win_drv = finishers.index[0]
        wd = df2[df2["Driver"] == win_drv]
        comp_runs = wd.groupby((wd["Compound"] != wd["Compound"].shift()).cumsum())
        runs = [(int(g["LapNumber"].min()), g["Compound"].iloc[0]) for _, g in comp_runs]
        runs = [(l, c) for l, c in runs if c in ("SOFT", "MEDIUM", "HARD")]
        if runs:
            actual_start = runs[0][1]
            actual_stops = [(l - 1, c) for l, c in runs[1:]]   # pit at end of prev lap

    def _strat_field():
        specs = []
        for _, r in field.iterrows():
            specs.append(DriverSpec(
                driver=r["Driver"], base_pace_s=float(r["base_pace_s"]),
                start_compound="MEDIUM", strategy=[(int(n_laps * 0.45), "HARD")],
                grid_pos=int(r["grid_pos"]), grid_gap_s=float((r["grid_pos"] - 1) * 0.4),
            ))
        return specs

    ego = field["Driver"].iloc[0]
    cands = {
        "ACTUAL (winner)": (actual_start, actual_stops or [(int(n_laps * 0.45), "HARD")]),
        "1-stop M→H": ("MEDIUM", [(int(n_laps * 0.45), "HARD")]),
        "2-stop M→H→H": ("MEDIUM", [(int(n_laps * 0.28), "HARD"), (int(n_laps * 0.62), "HARD")]),
        "1-stop S→H": ("SOFT", [(int(n_laps * 0.33), "HARD")]),
    }
    if st.button("Run strategy audit", type="primary"):
        with st.spinner("Simulating alternatives…"):
            spec = RaceSpec(circuit=circuit, era=era, n_laps=n_laps, drivers=_strat_field())
            rec = recommend(spec, ego, cands, n_rollouts=1500)
        st.session_state["postrace_rec"] = rec

    rec = st.session_state.get("postrace_rec")
    if rec is not None:
        show = rec.copy()
        show["win_prob"] = (show["win_prob"] * 100).round(1)
        show["podium_prob"] = (show["podium_prob"] * 100).round(1)
        show["mean_finish"] = show["mean_finish"].round(2)
        best = rec["strategy"].iloc[0]
        verdict = ("the actual strategy was simulator-optimal"
                   if best == "ACTUAL (winner)"
                   else f"the simulator preferred **{best}**")
        st.markdown(f"Ranking ego = **{ego}** on the lineup — {verdict}.")
        st.dataframe(
            show.rename(columns={"win_prob": "win %", "podium_prob": "podium %",
                                 "mean_finish": "mean finish"}),
            width="stretch", hide_index=True,
        )


# ---------------------------------------------------------------------------
# Tyre degradation: actual scatter vs fitted curve
# ---------------------------------------------------------------------------

T.section("Tyres", "Actual degradation vs fitted curve",
          sub="Fuel-corrected per-stint deltas at this race, over the fitted curve + 95% band.")

curves = T.load_tyre_curves()
if curves is not None:
    try:
        from src.models.lap_time.train_tft_data import load_tft_data  # noqa: F401
    except Exception:
        pass
    # prepare deltas just from this race's laps
    try:
        from src.pipeline.features import build_features as _bf, add_stint_id
        prepped = validate_laps(df.copy())
        prepped = add_stint_id(prepped)
        feat_full = _bf(prepped, *load_encoding_mappings())
        # carry back the cols prepare_stint_laps needs
        keys = ["Season", "RoundNumber", "Driver", "LapNumber"]
        carry = prepped[keys + ["Compound", "TrackStatus", "TyreLife", "StintID"]].copy()
        carry["TrackStatus"] = carry["TrackStatus"].astype(str)
        merged = feat_full.merge(carry, on=keys, how="left", suffixes=("", "_c"))
        for c in ["Compound", "TrackStatus", "TyreLife", "StintID"]:
            if f"{c}_c" in merged.columns:
                merged[c] = merged[f"{c}_c"]
        stint_laps = prepare_stint_laps(merged)
    except Exception as e:
        stint_laps = pd.DataFrame()
        st.caption(f"Could not prepare actual deltas: {e}")

    pol_comp = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"], index=1, key="tyre_comp")
    fig = go.Figure()
    if not stint_laps.empty:
        s = stint_laps[stint_laps["Compound"] == pol_comp]
        if not s.empty:
            fig.add_trace(go.Scatter(
                x=s["TyreLife"], y=s["Delta"], mode="markers", name="actual deltas",
                marker=dict(color=T.GREY, size=5, opacity=0.5),
            ))
    ages = np.arange(1, 36, dtype=float)
    try:
        mid, lo, hi = predict_degradation(curves, pol_comp, circuit, era, ages)
        color = T.COMPOUND_COLORS[pol_comp]
        rgb = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        fig.add_trace(go.Scatter(x=ages, y=hi, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ages, y=lo, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)",
                                 name="95% CI", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ages, y=mid, mode="lines", name="fitted",
                                 line=dict(color=color, width=2.4)))
    except Exception as e:
        st.caption(f"No fitted curve: {e}")
    fig.update_layout(xaxis_title="Tyre age (laps)", yaxis_title="Δ fuel-corrected lap time (s)")
    st.plotly_chart(T.style_fig(fig, 380), width="stretch")


# ---------------------------------------------------------------------------
# Error spikes vs TrackStatus
# ---------------------------------------------------------------------------

T.section("Incidents", "Error spikes vs track status",
          sub="Field-median lap-time prediction error per lap. Shaded = SC/VSC/yellow "
              "(TrackStatus != '1') — where the model is expected to break.")

if feats is not None:
    meta = df[["Season", "RoundNumber", "Driver", "LapNumber", "TrackStatus"]].copy()
    m = feats.merge(meta, on=["Season", "RoundNumber", "Driver", "LapNumber"], how="left")
    per_lap = m.groupby("LapNumber")["abs_err"].median()
    # track status per lap (any non-green car -> flag the lap)
    status_lap = (
        df.assign(green=lambda x: x["TrackStatus"] == "1")
        .groupby("LapNumber")["green"].mean()
    )
    non_green = status_lap[status_lap < 0.95].index.tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=per_lap.index, y=per_lap.values, mode="lines",
                             line=dict(color=T.WHITE, width=1.8), name="median |error|"))
    for lap in non_green:
        fig.add_vrect(x0=lap - 0.5, x1=lap + 0.5, fillcolor="rgba(255,209,46,0.18)",
                      line_width=0)
    fig.update_layout(xaxis_title="Lap", yaxis_title="Median |error| (s)")
    st.plotly_chart(T.style_fig(fig, 360), width="stretch")
    st.caption(f"{len(non_green)} lap(s) had a non-green car. "
               "Spikes clustering on shaded laps confirm the model breaks under SC/VSC, "
               "exactly as scoped — green-flag MAE is the honest headline metric.")
else:
    st.caption("Lap-time model unavailable, so error spikes cannot be computed.")
