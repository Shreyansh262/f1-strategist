"""Race Replay — step through a past race lap by lap, then project the finish.

Pure historical replay from data/raw parquet, no live-timing dependency:
position/gap chart up to the current lap, per-driver tyre-stint timeline,
actual-vs-predicted lap time with the calibration ribbon, and pit-window alerts.

It also absorbs what used to be the Live page's "Replay projection" tab, so a
past race has ONE home: from the selected lap you can (1) project the finishing
order by simulating the remaining laps thousands of times, and (2) play a
strategy what-if — change ONE car's remaining plan and see if it finishes better.
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

# Engine compounds the user can pick for a what-if (wet two are "indicative").
_PICKABLE_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

st.markdown("## Race replay")
T.section("Replay", sub="Step through any completed 2022–2026 Grand Prix lap by lap, "
                        "then project how it finishes from the lap you're on. "
                        "Historical data only — true live timing is a separate mode.")


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
_at = df["AirTemp"].dropna() if "AirTemp" in df.columns else None
_tt = df["TrackTemp"].dropna() if "TrackTemp" in df.columns else None
_wx_bits = []
if _at is not None and len(_at):
    _wx_bits.append(f"Air **{_at.median():.0f}°C**")
if _tt is not None and len(_tt):
    _wx_bits.append(f"Track **{_tt.median():.0f}°C**")
if _wx_bits:
    st.caption("Session conditions: " + " · ".join(_wx_bits))

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
st.caption(
    "How to read: the leader is the flat red line along the top at 0s. Every other line is how "
    "many seconds behind the leader that driver is on each lap — the axis is flipped so **higher "
    "on the chart = closer to the lead**. A line diving downward = losing time (a pit stop, traffic, "
    "or a slow lap); a line climbing back up = catching the car ahead."
)


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

T.section("Pace", "Actual vs lap-time prediction",
          sub="The selected driver's measured lap time against our lap-time prediction, "
              "with the expected ±band around it.")

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


# ===========================================================================
# Project the finish from here (absorbed from the old Live → Replay tab)
# ===========================================================================

from src.simulation.live_state import from_replay
from src.simulation.engine import DriverSpec, RaceSpec, simulate

# One seed for every simulation on this page — makes the what-if a fair,
# paired (common-random-numbers) comparison against the baseline.
_SIM_SEED = 42
_N_ROLLOUTS = 1500


def _build_spec():
    """Remaining-race RaceSpec from the selected lap, or None with a reason shown."""
    spec = from_replay(season, rnd, cur_lap)
    if spec is None:
        T.warn("Could not build a race state at this lap. Try a different lap or round.")
    return spec


def _flag_banner(spec) -> None:
    if spec.caution is not None:
        cause, elapsed = spec.caution
        dot = {"SC": "🟡", "VSC": "🟠", "RED": "🔴", "YELLOW": "🟡"}.get(cause, "⚠️")
        T.warn(
            f"{dot} <strong>{cause}</strong> flying for {elapsed} lap(s) — the projection "
            "samples when it ends from how long past cautions lasted."
        )
    else:
        st.caption("🟢 Green flag at this lap.")


def _win_bar(res, height: int = 400):
    """Horizontal win-chance bar chart for the top ~10 by average finish."""
    bar_df = res.table().head(10).copy()
    bar_df["win_pct"] = (bar_df["win_prob"] * 100).round(1)
    bar_df = bar_df.sort_values("win_pct")
    fig = go.Figure()
    for _, row in bar_df.iterrows():
        color = T.RED if row["driver"] == bar_df.iloc[-1]["driver"] else T.GREY
        fig.add_trace(go.Bar(
            y=[row["driver"]], x=[row["win_pct"]], orientation="h",
            marker=dict(color=color), showlegend=False,
            hovertemplate=f"{row['driver']}: {row['win_pct']:.1f}%<extra></extra>",
        ))
    fig.update_layout(xaxis_title="Win chance (%)", yaxis_title="")
    return T.style_fig(fig, height)


T.section("Projection", "Project the finish from here",
          sub="Take the race exactly as it stands on this lap and simulate the laps that "
              "are left, many times over, to see how it most likely ends.")

if curves is None:
    T.warn("Tyre curves unavailable — run the tyre fit to enable projections.")
else:
    project_clicked = st.button("PROJECT THE FINISH FROM HERE", key="project_btn")
    if project_clicked:
        spec = _build_spec()
        if spec is not None:
            _flag_banner(spec)
            with st.spinner("Simulating the remaining laps thousands of times…"):
                res = simulate(spec, n_rollouts=_N_ROLLOUTS, seed=_SIM_SEED,
                               device="cpu", curves=curves)
            st.markdown("#### Predicted finishing order")
            st.dataframe(
                T.friendly_finish_table(res.table()),
                hide_index=True, width="stretch",
                column_config={
                    col: st.column_config.Column(help=T.FRIENDLY_FINISH_HELP[col])
                    for col in T.FRIENDLY_FINISH_HELP
                },
            )
            st.caption(
                f"From lap {cur_lap}, simulating the remaining {spec.n_laps} laps "
                f"{_N_ROLLOUTS:,} times — any flag currently flying is included."
            )
            st.markdown("#### Win chance")
            st.plotly_chart(_win_bar(res), width="stretch")
    else:
        st.caption("Press the button to run the projection (it isn't run on every change).")


# ===========================================================================
# Strategy what-if: change ONE car's remaining plan from here
# ===========================================================================

T.section("What-if", "Strategy what-if",
          sub="What if one car pitted differently from this lap on? Pick a car, give it a "
              "new plan for the rest of the race, and compare it fairly against staying out.")

if curves is None:
    st.caption("Tyre curves unavailable — what-if needs them to project the rest of the race.")
elif cur_lap >= max_lap:
    st.caption("The race is at its final lap here — move the lap slider back to try a what-if.")
else:
    # Driver to experiment with (default = current leader).
    wi_drivers = order["Driver"].tolist()
    wi_driver = st.selectbox("Car to change", wi_drivers, key="wi_driver")

    # Remaining strategy controls — pit laps are ABSOLUTE race laps the user reads
    # off the timeline; we convert them to spec-relative laps before simulating.
    c1, c2 = st.columns(2)
    with c1:
        pit1 = st.slider("First pit on lap", cur_lap + 1, max_lap,
                         min(max_lap, cur_lap + max(1, (max_lap - cur_lap) // 2)),
                         key="wi_pit1")
        comp1 = st.selectbox("…onto compound", _PICKABLE_COMPOUNDS, index=2, key="wi_comp1")
    with c2:
        two_stop = st.checkbox("Add a second stop", key="wi_two_stop")
        pit2 = st.slider("Second pit on lap", pit1 + 1, max_lap,
                         min(max_lap, pit1 + max(1, (max_lap - pit1) // 2)),
                         key="wi_pit2", disabled=not two_stop)
        comp2 = st.selectbox("…onto compound", _PICKABLE_COMPOUNDS, index=1,
                             key="wi_comp2", disabled=not two_stop)

    st.caption(
        "INTERMEDIATE / WET are available but only indicative — the wet tyre model is "
        "approximate."
    )

    whatif_clicked = st.button("RUN WHAT-IF", key="whatif_btn")
    if whatif_clicked:
        spec = _build_spec()
        if spec is not None:
            _flag_banner(spec)

            # Spec laps are numbered 1..n_laps over the REMAINING race, so an
            # absolute race lap L maps to spec lap (L - cur_lap).
            new_stops = [(pit1 - cur_lap, comp1)]
            if two_stop and pit2 > pit1:
                new_stops.append((pit2 - cur_lap, comp2))
            new_stops = [(lap, c) for lap, c in new_stops if 1 <= lap <= spec.n_laps]

            try:
                ego_idx = next(i for i, d in enumerate(spec.drivers)
                               if d.driver == wi_driver)
            except StopIteration:
                ego_idx = None

            if ego_idx is None:
                T.warn(f"{wi_driver} is not running at this lap.")
            elif not new_stops:
                T.warn("The chosen pit lap(s) fall outside the remaining race — "
                       "pick laps after the current lap.")
            else:
                # BASELINE = the field as-is (every car continues on its current
                # tyres, no further stops). WHAT-IF = identical field except the
                # chosen car gets the new plan. Both are simulated with the SAME
                # seed, so every random draw (pace noise, degradation, pit noise,
                # overtakes, caution length) is shared lap-for-lap — the only
                # difference between the two outcomes is the strategy, not luck.
                base_drivers = [DriverSpec(**d.__dict__) for d in spec.drivers]
                base_spec = RaceSpec(**{**spec.__dict__, "drivers": base_drivers})

                wi_drivers_spec = [DriverSpec(**d.__dict__) for d in spec.drivers]
                wi_drivers_spec[ego_idx].strategy = new_stops
                wi_spec = RaceSpec(**{**spec.__dict__, "drivers": wi_drivers_spec})

                with st.spinner("Simulating both plans on the same dice…"):
                    base_res = simulate(base_spec, n_rollouts=_N_ROLLOUTS,
                                        seed=_SIM_SEED, device="cpu", curves=curves)
                    wi_res = simulate(wi_spec, n_rollouts=_N_ROLLOUTS,
                                      seed=_SIM_SEED, device="cpu", curves=curves)

                base_avg = base_res.mean_finish[wi_driver]
                wi_avg = wi_res.mean_finish[wi_driver]
                # Lower finish number = better, so improvement = baseline − whatif.
                delta = base_avg - wi_avg

                plan_txt = " then ".join(
                    f"lap {lap + cur_lap} → {c}" for lap, c in new_stops
                )
                st.markdown(f"#### What if **{wi_driver}** pits: {plan_txt}")

                m1, m2, m3 = st.columns(3)
                m1.metric("Avg finish — stay out", f"{base_avg:.2f}")
                m2.metric("Avg finish — new plan", f"{wi_avg:.2f}",
                          delta=f"{delta:+.2f} places",
                          delta_color="normal")
                if delta > 0.05:
                    verdict = f"Better by ~{delta:.2f} place(s)"
                elif delta < -0.05:
                    verdict = f"Worse by ~{abs(delta):.2f} place(s)"
                else:
                    verdict = "About the same"
                m3.metric("Verdict", verdict)

                st.caption(
                    "Positive = the new plan finishes higher up on average. Both plans were "
                    "simulated on the SAME random draws (same seed), so this difference is the "
                    "strategy, not luck — this is purely 'what if they pit differently from here'."
                )

                # Side-by-side win / podium chance for the chosen car.
                cmp_df = pd.DataFrame([
                    {"plan": "Stay out",
                     "win_prob": base_res.win_prob[wi_driver],
                     "podium_prob": base_res.podium_prob[wi_driver],
                     "mean_finish": base_avg},
                    {"plan": "New plan",
                     "win_prob": wi_res.win_prob[wi_driver],
                     "podium_prob": wi_res.podium_prob[wi_driver],
                     "mean_finish": wi_avg},
                ])
                st.markdown("#### This car's chances, side by side")
                st.dataframe(
                    T.friendly_finish_table(cmp_df),
                    hide_index=True, width="stretch",
                    column_config={
                        col: st.column_config.Column(help=T.FRIENDLY_FINISH_HELP[col])
                        for col in T.FRIENDLY_FINISH_HELP
                    },
                )

                st.markdown("#### Full field under the new plan")
                st.dataframe(
                    T.friendly_finish_table(wi_res.table()),
                    hide_index=True, width="stretch",
                    column_config={
                        col: st.column_config.Column(help=T.FRIENDLY_FINISH_HELP[col])
                        for col in T.FRIENDLY_FINISH_HELP
                    },
                )
    else:
        st.caption("Set a new plan above and press the button to compare it against staying out.")
