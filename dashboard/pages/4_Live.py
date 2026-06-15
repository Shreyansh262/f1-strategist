"""Live — one true live-race view.

A single button fetches whatever Grand Prix (or Sprint) is happening right now,
shows the current running order by driver name, the flag that is flying, and a
projection of how the race finishes from this point. When nothing is live it says
so plainly and shows the most-recent race's final result with NO projection — a
finished race is never given a win probability.
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
    sub="Tap the button to pull in whatever race is happening right now, see who's "
        "leading, and get a live projection of how it finishes from here.",
)

curves = T.load_tyre_curves()

# Flag dot lookup, shared by the banner.
_FLAG_DOTS = {"SC": "🟡", "VSC": "🟠", "RED": "🔴", "YELLOW": "🟡"}
_CAUSE_WORDS = {
    "SC": "Safety car",
    "VSC": "Virtual safety car",
    "RED": "Red flag",
    "YELLOW": "Yellow flag",
}

# ---------------------------------------------------------------------------
# Session state — persist the last fetch so reruns don't lose it.
# ---------------------------------------------------------------------------

for _k in ("live_status", "live_snap", "live_spec_table", "live_spec_nlaps"):
    if _k not in st.session_state:
        st.session_state[_k] = None
if "live_fetched" not in st.session_state:
    st.session_state["live_fetched"] = False


# ---------------------------------------------------------------------------
# Fetch button — the ONLY place that touches the network.
# ---------------------------------------------------------------------------

if st.button("Fetch live race", key="live_fetch_btn", type="primary"):
    st.session_state["live_fetched"] = True
    # Reset cached payloads from any previous fetch.
    st.session_state["live_status"] = None
    st.session_state["live_snap"] = None
    st.session_state["live_spec_table"] = None
    st.session_state["live_spec_nlaps"] = None
    try:
        from src.pipeline.openf1 import live_race_status, live_snapshot

        status = live_race_status(use_cache=False)
        st.session_state["live_status"] = status

        session_key = status.get("session_key") or 0
        if session_key:
            # Snapshot is needed for BOTH live (order + projection) and no_live
            # (final result, clearly labelled, no percentages).
            try:
                snap = live_snapshot(session_key)
                st.session_state["live_snap"] = snap
            except Exception:
                st.session_state["live_snap"] = None

            # Only project when a race is genuinely live — never percentage a
            # finished race.
            if status.get("state") == "live" and curves is not None:
                try:
                    from src.simulation.engine import simulate
                    from src.simulation.live_state import from_openf1

                    spec = from_openf1(session_key)
                    if spec is not None:
                        res = simulate(
                            spec, n_rollouts=1000, seed=42,
                            device="cpu", curves=curves,
                        )
                        st.session_state["live_spec_table"] = res.table()
                        st.session_state["live_spec_nlaps"] = int(spec.n_laps)
                except Exception:
                    st.session_state["live_spec_table"] = None
    except Exception as exc:
        # Total offline / unexpected failure -> fall back to a no_live message.
        st.session_state["live_status"] = {
            "state": "no_live",
            "session": {},
            "session_key": 0,
            "message": f"Couldn't reach live timing right now ({exc}). "
                       "Check your connection and try again.",
        }


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

status = st.session_state.get("live_status")

if not st.session_state["live_fetched"] or status is None:
    # Idle state — renders without any network or button click.
    T.warn(
        "No race loaded yet. Tap <strong>Fetch live race</strong> above to check "
        "whether a Grand Prix or Sprint is running right now."
    )
    st.caption("This is the only button that goes online — nothing is fetched until you ask.")
    st.stop()


snap = st.session_state.get("live_snap")
drivers_raw = (snap or {}).get("drivers", []) if isinstance(snap, dict) else []
state = status.get("state")


def _order_table(rows: list[dict], live: bool) -> pd.DataFrame:
    """Current running order by driver NAME + team (never bare numbers)."""
    out = []
    for d in rows:
        out.append({
            "Pos": d.get("position"),
            "Driver": d.get("name") or f"#{d.get('driver_number')}",
            "Team": d.get("team") or "—",
            "Tyre": d.get("compound") or "—",
            "Gap to leader (s)": (
                round(d["gap_to_leader_s"], 1)
                if isinstance(d.get("gap_to_leader_s"), (int, float)) else "—"
            ),
        })
    df = pd.DataFrame(out)
    # Drivers without a position sort last; keep the API's order otherwise.
    if not df.empty and "Pos" in df.columns:
        df = df.sort_values("Pos", na_position="last").reset_index(drop=True)
    return df


# ===========================================================================
# LIVE — header, flag, order, flag-aware projection
# ===========================================================================

if state == "live":
    sess = (snap or {}).get("session", {}) if isinstance(snap, dict) else {}
    status_sess = status.get("session", {})

    circuit = sess.get("circuit") or status_sess.get("circuit") or "—"
    year = sess.get("year") or status_sess.get("year") or "—"
    session_name = sess.get("session_name") or status_sess.get("session_name") or "Race"
    current_lap = sess.get("current_lap", "—")

    st.markdown(f"### {circuit} · {year}")
    st.markdown(f"**{session_name}** · live on lap **{current_lap}**")

    # Flag banner.
    caution = (snap or {}).get("caution") if isinstance(snap, dict) else None
    if caution:
        cause = str(caution.get("cause", "SC"))
        elapsed = int(caution.get("elapsed_laps", 1) or 1)
        dot = _FLAG_DOTS.get(cause, "⚠️")
        word = _CAUSE_WORDS.get(cause, cause)
        T.warn(
            f"{dot} <strong>{word}</strong> — out for {elapsed} lap(s) so far. "
            "The projection accounts for the slow-down while it's flying."
        )
    else:
        st.caption("🟢 Green flag — racing.")

    # Current running order (names + team).
    if drivers_raw:
        st.markdown("### Running order")
        st.markdown(
            " ".join(T.compound_chip(c) for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]),
            unsafe_allow_html=True,
        )
        st.dataframe(_order_table(drivers_raw, live=True), hide_index=True, width="stretch")

    # Flag-aware finish projection.
    st.markdown("### Projected finish")
    proj = st.session_state.get("live_spec_table")
    n_laps = st.session_state.get("live_spec_nlaps")

    if curves is None:
        T.warn("Projection unavailable — the tyre model hasn't been built yet.")
    elif proj is None or proj.empty:
        st.caption("Couldn't build a projection from the current live data — try again shortly.")
    else:
        friendly = T.friendly_finish_table(proj)
        st.dataframe(
            friendly,
            hide_index=True,
            width="stretch",
            column_config={
                col: st.column_config.Column(help=help_text)
                for col, help_text in T.FRIENDLY_FINISH_HELP.items()
                if col in friendly.columns
            },
        )
        laps_txt = f"the remaining {n_laps} laps" if n_laps else "the remaining laps"
        st.caption(
            f"Live projection of {laps_txt} — many simulated race-finishes from "
            "the current order, with the flag that's flying right now included."
        )

        # Win-chance bar chart (top ~10 by average finish).
        bar = proj.head(10).copy()
        bar["win_pct"] = (bar["win_prob"].astype(float) * 100).round(1)
        bar = bar.sort_values("win_pct")

        fig = go.Figure()
        leader = bar.iloc[-1]["driver"] if len(bar) else None
        for _, row in bar.iterrows():
            color = T.RED if row["driver"] == leader else T.GREY
            fig.add_trace(go.Bar(
                y=[row["driver"]],
                x=[row["win_pct"]],
                orientation="h",
                marker=dict(color=color),
                showlegend=False,
                hovertemplate=f"{row['driver']}: {row['win_pct']:.1f}%<extra></extra>",
            ))
        fig.update_layout(xaxis_title="Win chance (%)", yaxis_title="")
        st.plotly_chart(T.style_fig(fig, 400), width="stretch")


# ===========================================================================
# NO LIVE — friendly message + (optional) final result, NO projection
# ===========================================================================

else:
    message = status.get("message") or "No live race happening right now."
    T.warn(f"ℹ️ {message}")

    # Optionally show the most-recent race's FINAL order — clearly labelled,
    # with no projection and no percentages.
    if drivers_raw:
        status_sess = status.get("session", {})
        sess = (snap or {}).get("session", {}) if isinstance(snap, dict) else {}
        circuit = status_sess.get("circuit") or sess.get("circuit") or ""
        year = status_sess.get("year") or sess.get("year") or ""
        label = f"{circuit} {year}".strip()

        st.markdown("### Final result" + (f" — {label}" if label else ""))
        st.caption("This race has finished, so there's no live projection — just the result.")
        st.markdown(
            " ".join(T.compound_chip(c) for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]),
            unsafe_allow_html=True,
        )
        st.dataframe(_order_table(drivers_raw, live=False), hide_index=True, width="stretch")
    else:
        st.caption("Tap Fetch live race again later — a session will appear here when one is on.")
