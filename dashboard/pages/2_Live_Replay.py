"""Live Replay — step through a historical race lap by lap.

Pure historical replay from data/raw parquet, no live-timing dependency:
position/gap chart up to the current lap, per-driver tyre-stint timeline,
actual-vs-predicted lap time with the calibration ribbon, and pit-window alerts.
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

T.page_config("Live Replay · F1 Strategist")
T.apply_theme()
T.register_plotly_template()

# Lap-time model is optional — page works without it (predicted overlay hidden).
_LAPMODEL_OK = True
_LAP_ERR = ""
try:
    from src.pipeline.features import build_features, load_encoding_mappings
    from src.pipeline.validate import validate_laps
    from src.models.lap_time.train import get_X_y
except Exception as e:                                    # pragma: no cover
    _LAPMODEL_OK = False
    _LAP_ERR = str(e)

st.markdown("## Live replay")
T.section("Replay", sub="Re-run any 2022–2026 Grand Prix lap by lap. Historical data only.")


# ---------------------------------------------------------------------------
# Race selector
# ---------------------------------------------------------------------------

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
max_lap = int(df["LapNumber"].max())

st.markdown(f"### {circuit} · {season}")

cur_lap = st.slider("Lap", 1, max_lap, min(max_lap, max_lap // 2))

# cumulative race time per driver up to each lap (gap chart)
df = df.sort_values(["Driver", "LapNumber"])
df["cum"] = df.groupby("Driver")["LapTimeSeconds"].cumsum()

upto = df[df["LapNumber"] <= cur_lap]
# leader cumulative time at the current lap
at_lap = upto[upto["LapNumber"] == cur_lap].dropna(subset=["cum"])
if at_lap.empty:
    at_lap = upto.sort_values("LapNumber").groupby("Driver").tail(1)

order = at_lap.sort_values("cum")
leader_cum = order["cum"].iloc[0] if len(order) else 0.0

# KPI strip
k1, k2, k3, k4 = st.columns(4)
k1.metric("Lap", f"{cur_lap} / {max_lap}")
k2.metric("Cars running", f"{order['Driver'].nunique()}")
green_frac = (df[df["LapNumber"] <= cur_lap]["TrackStatus"] == "1").mean()
k3.metric("Green-lap share", f"{green_frac:.0%}")
leader = order["Driver"].iloc[0] if len(order) else "—"
k4.metric("Leader", leader)


# ---------------------------------------------------------------------------
# Position / gap-to-leader chart
# ---------------------------------------------------------------------------

T.section("Track", "Gap to leader",
          sub="Cumulative race-time gap behind the on-track leader, lap by lap.")

# build gap traces for the top drivers at the current lap
top_drivers = order["Driver"].head(10).tolist()
fig_gap = go.Figure()
for i, drv in enumerate(top_drivers):
    d = upto[upto["Driver"] == drv].dropna(subset=["cum"]).sort_values("LapNumber")
    if d.empty:
        continue
    # gap behind the leader's cumulative time at each lap
    leader_by_lap = (
        upto.dropna(subset=["cum"]).groupby("LapNumber")["cum"].min()
    )
    gaps = d["cum"].values - leader_by_lap.reindex(d["LapNumber"]).values
    color = T.RED if i == 0 else (T.WHITE if i == 1 else T.COLORWAY[i % len(T.COLORWAY)])
    fig_gap.add_trace(go.Scatter(
        x=d["LapNumber"], y=gaps, mode="lines", name=drv,
        line=dict(color=color, width=2.2 if i < 2 else 1.3),
    ))
fig_gap.update_layout(xaxis_title="Lap", yaxis_title="Gap to leader (s)")
fig_gap.update_yaxes(autorange="reversed")
st.plotly_chart(T.style_fig(fig_gap, 420), width="stretch")


# ---------------------------------------------------------------------------
# Tyre-stint timeline
# ---------------------------------------------------------------------------

T.section("Tyres", "Stint timeline",
          sub="Coloured bars = compound; bar length = stint length. Marker = current lap.")

order_drivers = order["Driver"].head(12).tolist()
fig_stint = go.Figure()
for drv in order_drivers:
    d = df[df["Driver"] == drv].sort_values("LapNumber")
    # segment by compound runs
    seg = (d["Compound"] != d["Compound"].shift()).cumsum()
    for _, g in d.groupby(seg):
        comp = g["Compound"].iloc[0]
        if pd.isna(comp):
            continue
        start = int(g["LapNumber"].min())
        end = int(g["LapNumber"].max())
        fig_stint.add_trace(go.Bar(
            y=[drv], x=[end - start + 1], base=start - 1, orientation="h",
            marker=dict(color=T.COMPOUND_COLORS.get(comp, T.GREY),
                        line=dict(color=T.BG, width=1)),
            name=comp, showlegend=False,
            hovertemplate=f"{drv} · {comp}<br>laps {start}-{end}<extra></extra>",
        ))
fig_stint.add_vline(x=cur_lap, line=dict(color=T.RED, width=2, dash="dot"))
fig_stint.update_layout(barmode="stack", xaxis_title="Lap",
                        yaxis=dict(autorange="reversed"))
# compound legend chips
st.markdown(
    " ".join(T.compound_chip(c) for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]),
    unsafe_allow_html=True,
)
st.plotly_chart(T.style_fig(fig_stint, max(320, 26 * len(order_drivers))), width="stretch")


# ---------------------------------------------------------------------------
# Actual vs predicted lap time + calibration ribbon
# ---------------------------------------------------------------------------

T.section("Model", "Actual vs predicted",
          sub="Selected driver's measured lap time against the LightGBM prediction, "
              "with the recalibrated ±band.")

sel_driver = st.selectbox("Driver", order["Driver"].tolist())

pred_df = None
if _LAPMODEL_OK and T.load_lap_model() is not None:
    try:
        cmap, tmap, dmap = load_encoding_mappings()
        feats = build_features(validate_laps(df.copy()), cmap, tmap, dmap)
        X, _ = get_X_y(feats)
        model = T.load_lap_model()
        feats = feats.copy()
        feats["yhat"] = model.predict(X)
        pred_df = feats[feats["Driver"] == sel_driver][["LapNumber", "yhat"]]
    except Exception as e:
        st.caption(f"Prediction overlay unavailable: {e}")

d = df[df["Driver"] == sel_driver].sort_values("LapNumber")
fig_pred = go.Figure()
# calibration ribbon around prediction
if pred_df is not None and not pred_df.empty:
    rec = T.read_csv_report("lap_time/tft_recalibration.csv")
    sigma = 1.0
    if rec is not None:
        row = rec[(rec["era"] == era) & (rec["bands"] == "recalibrated")]
        if len(row):
            sigma = float(row["mean_band_width_s"].iloc[0]) / 2.0
    m = pred_df.sort_values("LapNumber")
    fig_pred.add_trace(go.Scatter(x=m["LapNumber"], y=m["yhat"] + sigma, mode="lines",
                                  line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_pred.add_trace(go.Scatter(x=m["LapNumber"], y=m["yhat"] - sigma, mode="lines",
                                  line=dict(width=0), fill="tonexty",
                                  fillcolor="rgba(225,6,0,0.15)",
                                  name="±band", hoverinfo="skip"))
    fig_pred.add_trace(go.Scatter(x=m["LapNumber"], y=m["yhat"], mode="lines",
                                  name="predicted", line=dict(color=T.RED, width=2)))
fig_pred.add_trace(go.Scatter(
    x=d["LapNumber"], y=d["LapTimeSeconds"], mode="markers+lines", name="actual",
    line=dict(color=T.WHITE, width=1.4), marker=dict(size=4, color=T.WHITE),
))
fig_pred.add_vline(x=cur_lap, line=dict(color=T.GREY, width=1, dash="dot"))
fig_pred.update_layout(xaxis_title="Lap", yaxis_title="Lap time (s)")
st.plotly_chart(T.style_fig(fig_pred, 380), width="stretch")


# ---------------------------------------------------------------------------
# Pit-window alert
# ---------------------------------------------------------------------------

T.section("Strategy", "Pit-window alert",
          sub="Compares each running driver's current tyre age against where the tyre "
              "curve starts to bite (degradation > 0.8s vs fresh).")

curves = T.load_tyre_curves()
alerts = []
if curves is not None:
    from src.models.tyre.fit import predict_degradation
    cur_state = at_lap[["Driver", "Compound", "TyreLife"]].dropna()
    ages = np.arange(1, 46, dtype=float)
    for _, r in cur_state.iterrows():
        comp = r["Compound"]
        if comp not in ("SOFT", "MEDIUM", "HARD"):
            continue
        try:
            mid, _, _ = predict_degradation(curves, comp, circuit, era, ages)
        except Exception:
            continue
        cliff_idx = np.argmax(mid > 0.8)
        cliff_age = int(ages[cliff_idx]) if mid.max() > 0.8 else 99
        age = int(r["TyreLife"])
        margin = cliff_age - age
        status = "PIT NOW" if margin <= 0 else ("WINDOW" if margin <= 4 else "OK")
        alerts.append({"Driver": r["Driver"], "Compound": comp, "tyre age": age,
                       "cliff age": cliff_age, "laps to cliff": margin, "status": status})

if alerts:
    adf = pd.DataFrame(alerts).sort_values("laps to cliff")
    def _style(row):
        c = {"PIT NOW": "rgba(225,6,0,0.30)", "WINDOW": "rgba(255,209,46,0.18)"}.get(row["status"], "")
        return [f"background-color:{c}" if c else "" for _ in row]
    st.dataframe(adf.style.apply(_style, axis=1), width="stretch", hide_index=True)
else:
    st.caption("No slick-tyre running drivers at this lap, or tyre curves unavailable.")
