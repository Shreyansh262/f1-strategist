"""Live — real-time OpenF1 timing and flag-aware finish projection.

Tab 1: Replay projection (offline default) — historical parquet at any lap.
Tab 2: Live (OpenF1) — fetch live snapshot and project from current state.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
_ROOT = _HERE.parent
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme as T

T.apply_theme()
T.register_plotly_template()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown("## Live")
T.section(
    "Live",
    sub="Real-time OpenF1 timing and a flag-aware projection of how the race finishes "
        "from right now.",
)

# ---------------------------------------------------------------------------
# Shared: tyre curves (needed for both tabs)
# ---------------------------------------------------------------------------

curves = T.load_tyre_curves()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_replay, tab_live = st.tabs(["Replay projection", "Live (OpenF1)"])


# ===========================================================================
# TAB 1: Replay projection (offline default)
# ===========================================================================

with tab_replay:

    # ------------------------------------------------------------------
    # Selectors (sidebar mirrors 2_Live_Replay style)
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown('<p class="f1-label">Live — Replay</p>', unsafe_allow_html=True)
        seasons = T.list_seasons()
        if not seasons:
            T.warn("No race data found. Download at least one season.")
            st.stop()
        season = st.selectbox("Season", seasons, index=len(seasons) - 1, key="live_season")
        circuits = T.season_circuits(season)
        if not circuits:
            T.warn("No races found for this season.")
            st.stop()
        rnd = st.selectbox(
            "Round", list(circuits.keys()),
            format_func=lambda r: f"R{r:02d} · {circuits[r]}",
            key="live_round",
        )

    df = T.load_race(season, rnd)
    if df is None or df.empty:
        T.warn("Race parquet missing or empty. Select a different round.")
        st.stop()

    max_lap = int(df["LapNumber"].max())
    default_lap = max(1, int(max_lap * 0.60))
    cur_lap = st.slider("Lap", 1, max_lap, default_lap, key="live_lap")

    # ------------------------------------------------------------------
    # Build spec from replay
    # ------------------------------------------------------------------
    from src.simulation.live_state import from_replay

    spec = from_replay(season, rnd, cur_lap)
    if spec is None:
        T.warn("Could not build race state at this lap. Try a different lap or round.")
        st.stop()

    # ------------------------------------------------------------------
    # Current state panel
    # ------------------------------------------------------------------
    st.markdown("### Current state")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lap", f"{cur_lap} / {spec.n_laps + cur_lap}")
    k2.metric("Cars running", str(len(spec.drivers)))
    leader_name = spec.drivers[0].driver if spec.drivers else "—"
    k3.metric("Leader", leader_name)
    k4.metric("Era", str(spec.era))

    # Flag banner
    if spec.caution is not None:
        cause, elapsed = spec.caution
        color_map = {"SC": "🟡", "VSC": "🟠", "RED": "🔴", "YELLOW": "🟡"}
        dot = color_map.get(cause, "⚠️")
        T.warn(
            f"{dot} <strong>{cause}</strong> — flying for {elapsed} lap(s). "
            "Projection samples when it ends from historical durations."
        )
    else:
        st.caption("🟢 Green flag.")

    # Current order table
    order_rows = [
        {
            "Pos": i + 1,
            "Driver": d.driver,
            "Compound": d.start_compound,
            "Tyre age": d.start_age,
            "Gap to leader (s)": round(d.grid_gap_s, 3),
        }
        for i, d in enumerate(spec.drivers)
    ]
    order_df = pd.DataFrame(order_rows)
    st.markdown(
        " ".join(T.compound_chip(c) for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]),
        unsafe_allow_html=True,
    )
    st.dataframe(order_df, hide_index=True, width="stretch")

    # ------------------------------------------------------------------
    # Projected finish
    # ------------------------------------------------------------------
    st.markdown("### Projected finish")

    if curves is None:
        T.warn("Tyre curves unavailable — run the tyre fit.")
        st.stop()

    from src.simulation.engine import simulate

    with st.spinner("Running Monte Carlo projection…"):
        res = simulate(spec, n_rollouts=1500, seed=42, device="cpu", curves=curves)

    result_df = res.table().copy()
    result_df["win_prob"] = (result_df["win_prob"] * 100).round(1).astype(str) + "%"
    result_df["podium_prob"] = (result_df["podium_prob"] * 100).round(1).astype(str) + "%"
    result_df["mean_finish"] = result_df["mean_finish"].round(2)
    result_df["p5_finish"] = result_df["p5_finish"].round(1)
    result_df["p95_finish"] = result_df["p95_finish"].round(1)

    st.dataframe(result_df, hide_index=True, width="stretch")
    st.caption(
        f"Projecting the remaining {spec.n_laps} laps from lap {cur_lap} — "
        "1 500 Monte Carlo rollouts, flags included."
    )

    # Win-probability bar chart (top ~10 by mean_finish)
    bar_df = res.table().head(10).copy()
    bar_df["win_pct"] = (bar_df["win_prob"] * 100).round(1)
    bar_df = bar_df.sort_values("win_pct")

    fig_win = go.Figure()
    for i, row in bar_df.iterrows():
        color = T.RED if row["driver"] == bar_df.iloc[-1]["driver"] else T.GREY
        fig_win.add_trace(go.Bar(
            y=[row["driver"]],
            x=[row["win_pct"]],
            orientation="h",
            marker=dict(color=color),
            showlegend=False,
            hovertemplate=f"{row['driver']}: {row['win_pct']:.1f}%<extra></extra>",
        ))

    fig_win.update_layout(xaxis_title="Win probability (%)", yaxis_title="")
    st.plotly_chart(T.style_fig(fig_win, 400), width="stretch")


# ===========================================================================
# TAB 2: Live (OpenF1)
# ===========================================================================

with tab_live:
    st.caption(
        "Hits the public OpenF1 API (openf1.org). Works on any session_key — "
        "a finished session returns its final-lap state."
    )

    session_key_input = st.text_input("Session key", value="latest", key="live_sk")
    total_laps_override = st.number_input(
        "Total race laps (override)", min_value=0, value=0, step=1, key="live_laps_override",
        help="Leave 0 to auto-detect from circuit.",
    )
    fetch_btn = st.button("Fetch live snapshot", key="live_fetch_btn")

    # Persist last fetch in session_state so reruns don't lose the data.
    if "live_snap" not in st.session_state:
        st.session_state["live_snap"] = None
    if "live_sk_used" not in st.session_state:
        st.session_state["live_sk_used"] = None

    if fetch_btn:
        try:
            from src.pipeline.openf1 import live_snapshot

            snap = live_snapshot(session_key_input)
            st.session_state["live_snap"] = snap
            st.session_state["live_sk_used"] = session_key_input
        except Exception as exc:
            st.session_state["live_snap"] = None
            T.warn(f"Fetch failed: {exc}")

    snap = st.session_state.get("live_snap")

    if snap is not None:
        try:
            sess = snap["session"]
            drivers_raw = snap.get("drivers", [])

            if not sess.get("available") and not drivers_raw:
                T.warn("No data for that session_key (or offline).")
            else:
                # Session header
                st.markdown(
                    f"**{sess.get('circuit', '—')}** · "
                    f"{sess.get('year', '—')} · "
                    f"{sess.get('session_name', '—')} · "
                    f"Lap **{sess.get('current_lap', '—')}**"
                )

                # Flag banner
                caution_raw = snap.get("caution")
                if caution_raw is not None:
                    cause = caution_raw.get("cause", "SC")
                    elapsed = caution_raw.get("elapsed_laps", 1)
                    color_map = {"SC": "🟡", "VSC": "🟠", "RED": "🔴", "YELLOW": "🟡"}
                    dot = color_map.get(cause, "⚠️")
                    T.warn(
                        f"{dot} <strong>{cause}</strong> — flying for {elapsed} lap(s). "
                        "Projection samples when it ends from historical durations."
                    )
                else:
                    st.caption("🟢 Green flag.")

                # Snapshot driver table
                if drivers_raw:
                    snap_rows = []
                    for d in drivers_raw:
                        snap_rows.append({
                            "driver_number": d.get("driver_number"),
                            "position": d.get("position"),
                            "compound": d.get("compound") or "—",
                            "tyre age": d.get("start_age"),
                            "gap_to_leader_s": d.get("gap_to_leader_s"),
                            "last_lap_s": d.get("last_lap_s"),
                        })
                    snap_df = pd.DataFrame(snap_rows)
                    st.dataframe(snap_df, hide_index=True, width="stretch")

                # Projection
                try:
                    from src.simulation.live_state import from_openf1

                    override_laps = int(total_laps_override) if total_laps_override else None
                    live_spec = from_openf1(
                        session_key_input,
                        total_laps=override_laps,
                        use_cache=True,
                    )

                    if live_spec is not None and curves is not None:
                        from src.simulation.engine import simulate as _simulate

                        st.markdown("### Projected finish")
                        with st.spinner("Running Monte Carlo projection…"):
                            live_res = _simulate(
                                live_spec, n_rollouts=1000, seed=42,
                                device="cpu", curves=curves,
                            )

                        lr_df = live_res.table().copy()
                        lr_df["win_prob"] = (lr_df["win_prob"] * 100).round(1).astype(str) + "%"
                        lr_df["podium_prob"] = (lr_df["podium_prob"] * 100).round(1).astype(str) + "%"
                        lr_df["mean_finish"] = lr_df["mean_finish"].round(2)
                        lr_df["p5_finish"] = lr_df["p5_finish"].round(1)
                        lr_df["p95_finish"] = lr_df["p95_finish"].round(1)

                        st.dataframe(lr_df, hide_index=True, width="stretch")
                        st.caption(
                            f"Projecting the remaining {live_spec.n_laps} laps — "
                            "1 000 Monte Carlo rollouts, flags included."
                        )

                        # Win-prob bar chart
                        lbar_df = live_res.table().head(10).copy()
                        lbar_df["win_pct"] = (lbar_df["win_prob"] * 100).round(1)
                        lbar_df = lbar_df.sort_values("win_pct")

                        fig_lwin = go.Figure()
                        for _, lrow in lbar_df.iterrows():
                            lcolor = T.RED if lrow["driver"] == lbar_df.iloc[-1]["driver"] else T.GREY
                            fig_lwin.add_trace(go.Bar(
                                y=[lrow["driver"]],
                                x=[lrow["win_pct"]],
                                orientation="h",
                                marker=dict(color=lcolor),
                                showlegend=False,
                                hovertemplate=f"{lrow['driver']}: {lrow['win_pct']:.1f}%<extra></extra>",
                            ))

                        fig_lwin.update_layout(xaxis_title="Win probability (%)", yaxis_title="")
                        st.plotly_chart(T.style_fig(fig_lwin, 400), width="stretch")

                    elif curves is None:
                        T.warn("Tyre curves unavailable — run the tyre fit.")
                    elif live_spec is None:
                        st.caption("Live projection unavailable: could not build RaceSpec from snapshot.")

                except Exception as proj_exc:
                    st.caption(f"Live projection unavailable: {proj_exc}")

        except Exception as e:
            st.caption(f"Live projection unavailable: {e}")
