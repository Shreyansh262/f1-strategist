"""Post-Race analysis — a plain-language audit of a finished race.

For casual fans: replay is for *what happened*; this page is for *was it the
best they could do, and how could teams have done better?* It uses our race
simulator and tyre-wear model to:
  - rank the winner's real strategy against alternatives,
  - find, for the top finishers, a better strategy and show the places it
    would have gained,
  - draw an optimal "when to pit" map, and
  - check the tyre-wear model against what the tyres actually did this race.

No model brand names or formulas are shown to the user — just plain English.
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

_OK = True
_ERR = ""
try:
    from src.pipeline.features import build_features, load_encoding_mappings, add_stint_id
    from src.pipeline.validate import validate_laps
    from src.models.tyre.fit import predict_degradation, prepare_stint_laps
    from src.simulation.engine import RaceSpec, DriverSpec, simulate, recommend
    from src.models.pit_strategy.mdp import build_race_model, solve, COMPOUNDS, MAX_AGE
except Exception as e:                                    # pragma: no cover
    _OK = False
    _ERR = str(e)

st.markdown("## Post-Race analysis")
T.section("Audit", sub="Was the real race the best they could do — and how could teams have done better? "
                      "Powered by our race simulator and tyre-wear model.")

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
    T.warn("Race data missing or empty.")
    st.stop()

circuit = str(df["CircuitKey"].iloc[0])
era = T.era_for_season(season)
n_laps = int(df["LapNumber"].max())
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


# ---------------------------------------------------------------------------
# Helpers: extract a driver's ACTUAL strategy from the race laps
# ---------------------------------------------------------------------------

DRY = ("SOFT", "MEDIUM", "HARD")


def _actual_strategy(laps_for_driver: pd.DataFrame) -> tuple[str, list[tuple[int, str]]]:
    """(start_compound, [(pit_at_end_of_lap, compound), ...]) from one driver's laps.

    A "run" is a block of consecutive laps on the same compound; the first run is
    the start tyre, each later run begins one lap after a pit. Only dry compounds
    are kept so the run feeds the (dry-only) simulator cleanly.
    """
    wd = laps_for_driver.sort_values("LapNumber")
    comp_runs = wd.groupby((wd["Compound"] != wd["Compound"].shift()).cumsum())
    runs = [(int(g["LapNumber"].min()), g["Compound"].iloc[0]) for _, g in comp_runs]
    runs = [(l, c) for l, c in runs if c in DRY]
    if not runs:
        return "MEDIUM", []
    start = runs[0][1]
    stops = [(l - 1, c) for l, c in runs[1:]]   # pit happened at end of previous lap
    return start, stops


def _fmt_strategy(start: str, stops: list[tuple[int, str]]) -> str:
    """Plain one-stop/two-stop label, e.g. '1-stop: MEDIUM → HARD (lap 28)'."""
    if not stops:
        return f"no-stop: {start}"
    chain = start
    parts = []
    for lap, comp in stops:
        chain += f" → {comp}"
        parts.append(f"lap {lap}")
    n = len(stops)
    word = {1: "1-stop", 2: "2-stop", 3: "3-stop"}.get(n, f"{n}-stop")
    return f"{word}: {chain} ({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Build the field of finishers with their actual strategies + base pace
# ---------------------------------------------------------------------------

base = T.green_base_pace(season, rnd)

df2 = df.sort_values(["Driver", "LapNumber"])
totals = df2.groupby("Driver").agg(time=("LapTimeSeconds", "sum"),
                                   laps=("LapNumber", "max"))
full_distance = int(totals["laps"].max())
finishers = totals[totals["laps"] == full_distance].sort_values("time")
# actual finishing order (1 = winner) for the cars that went the distance
actual_order = {drv: i + 1 for i, drv in enumerate(finishers.index)}


# ---------------------------------------------------------------------------
# Was the actual strategy optimal? (winner ranked against alternatives)
# ---------------------------------------------------------------------------

T.section("Best-strategy check", "Was the winner's strategy the best one?",
          sub="We take the winner's real tyre plan, then race it against a few sensible "
              "alternatives thousands of times in the simulator to see which comes out on top.")

if base.empty or len(base) < 2:
    st.caption("Not enough clean racing laps to run the best-strategy check.")
else:
    n_drivers = min(10, len(base))
    field = base.head(n_drivers).reset_index(drop=True)

    actual_start, actual_stops = "MEDIUM", []
    win_drv = None
    if len(finishers):
        win_drv = finishers.index[0]
        actual_start, actual_stops = _actual_strategy(df2[df2["Driver"] == win_drv])

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
        "Winner's real plan": (actual_start, actual_stops or [(int(n_laps * 0.45), "HARD")]),
        "1-stop M→H": ("MEDIUM", [(int(n_laps * 0.45), "HARD")]),
        "2-stop M→H→H": ("MEDIUM", [(int(n_laps * 0.28), "HARD"), (int(n_laps * 0.62), "HARD")]),
        "1-stop S→H": ("SOFT", [(int(n_laps * 0.33), "HARD")]),
    }
    if st.button("Run best-strategy check", type="primary"):
        with st.spinner("Simulating alternatives…"):
            spec = RaceSpec(circuit=circuit, era=era, n_laps=n_laps, drivers=_strat_field())
            rec = recommend(spec, ego, cands, n_rollouts=1500)
        st.session_state["postrace_rec"] = rec

    rec = st.session_state.get("postrace_rec")
    if rec is not None:
        best = rec["strategy"].iloc[0]
        verdict = ("the winner's real plan was already the simulator's top pick"
                   if best == "Winner's real plan"
                   else f"the simulator narrowly preferred **{best}**")
        who = f"**{win_drv}**" if win_drv else "the winner"
        st.markdown(f"Comparing strategies for {who} — {verdict}.")
        show = rec.drop(columns=[c for c in ("start", "stops", "p50_time_s") if c in rec.columns])
        st.dataframe(T.friendly_finish_table(show), width="stretch", hide_index=True)
        st.caption("Win chance = how often that plan won across the simulated races · "
                   "Podium chance = top-3 · Avg finish = average position (1 = win). "
                   "Top row is the simulator's pick.")


# ---------------------------------------------------------------------------
# How could teams have done better? — standings delta from a better strategy
# ---------------------------------------------------------------------------

T.section("Room for improvement", "How could teams have done better?",
          sub="For the leading finishers, we read each one's real tyre plan from the race, then "
              "search for a better plan in the simulator. The table shows where they actually "
              "finished versus where a smarter plan would have put them.")

st.caption("This is an honest single-car what-if: we change one driver's plan at a time and "
           "hold every rival on their real strategy, so it shows the upside that was on the "
           "table, not a re-run of the whole race.")

if base.empty or len(finishers) < 3:
    st.caption("Not enough full-distance finishers to run the improvement search.")
else:
    n_top = st.slider("How many of the leading finishers to analyse", 3, 8,
                      min(5, len(finishers)), key="improve_n")

    if st.button("Find better strategies", type="primary", key="improve_btn"):
        # Build a field of every full-distance finisher we have pace for, each on
        # their ACTUAL strategy. base_pace + actual strategy reproduce the race in
        # the simulator closely enough to read off a fair baseline finishing order.
        pace = base.set_index("Driver")
        drivers_in_field = [d for d in finishers.index if d in pace.index]
        if len(drivers_in_field) < 3:
            st.warning("Too few finishers have clean pace data for this search.")
        else:
            actual_strats: dict[str, tuple[str, list[tuple[int, str]]]] = {}
            specs = []
            for d in drivers_in_field:
                start, stops = _actual_strategy(df2[df2["Driver"] == d])
                actual_strats[d] = (start, stops)
                # fall back to a 1-stop if a driver's laps gave no clean dry run
                strat = stops or [(int(n_laps * 0.45), "HARD")]
                r = pace.loc[d]
                specs.append(DriverSpec(
                    driver=d, base_pace_s=float(r["base_pace_s"]),
                    start_compound=start, strategy=strat,
                    grid_pos=int(r["grid_pos"]),
                    grid_gap_s=float((r["grid_pos"] - 1) * 0.4),
                ))

            base_spec = RaceSpec(circuit=circuit, era=era, n_laps=n_laps, drivers=specs)

            # candidate plans to try for each driver (start, stops)
            def _candidates() -> dict[str, tuple[str, list[tuple[int, str]]]]:
                return {
                    "1-stop M→H": ("MEDIUM", [(int(n_laps * 0.45), "HARD")]),
                    "1-stop S→H": ("SOFT", [(int(n_laps * 0.33), "HARD")]),
                    "1-stop M→H (late)": ("MEDIUM", [(int(n_laps * 0.55), "HARD")]),
                    "2-stop M→H→H": ("MEDIUM", [(int(n_laps * 0.28), "HARD"),
                                                (int(n_laps * 0.62), "HARD")]),
                    "2-stop S→M→H": ("SOFT", [(int(n_laps * 0.30), "MEDIUM"),
                                              (int(n_laps * 0.62), "HARD")]),
                }

            targets = drivers_in_field[:n_top]
            rows = []
            with st.spinner(f"Searching better strategies for {len(targets)} drivers…"):
                # 1) baseline projected order: everyone on their real strategy
                base_res = simulate(base_spec, n_rollouts=1200)
                base_finish = base_res.mean_finish

                for d in targets:
                    a_start, a_stops = actual_strats[d]
                    # always include the driver's own real plan in the search so a
                    # genuinely optimal real strategy yields "0 places gained"
                    cands = {"Real plan": (a_start, a_stops or
                                           [(int(n_laps * 0.45), "HARD")])}
                    cands.update(_candidates())
                    ranked = recommend(base_spec, d, cands, n_rollouts=1200)
                    best_name = ranked["strategy"].iloc[0]
                    best_start = ranked["start"].iloc[0]
                    best_stops = eval(ranked["stops"].iloc[0])  # noqa: S307 - our serialized tuples

                    # 2) re-simulate the field with ONLY this driver swapped to the
                    #    suggested plan; read the driver's new projected finish.
                    swapped = [DriverSpec(**{**s.__dict__}) for s in specs]
                    j = next(i for i, s in enumerate(swapped) if s.driver == d)
                    swapped[j].start_compound = best_start
                    swapped[j].strategy = best_stops
                    new_res = simulate(
                        RaceSpec(circuit=circuit, era=era, n_laps=n_laps, drivers=swapped),
                        n_rollouts=1200,
                    )
                    proj_now = base_finish[d]
                    proj_better = new_res.mean_finish[d]
                    gained = int(round(proj_now - proj_better))
                    suggested = (_fmt_strategy(best_start, best_stops)
                                 if best_name != "Real plan" else "— (real plan was best)")
                    rows.append({
                        "Driver": d,
                        "Actual finish": actual_order.get(d, "—"),
                        "Actual strategy": _fmt_strategy(a_start, a_stops),
                        "Suggested strategy": suggested,
                        "Projected finish": round(float(proj_better), 1),
                        "Places gained": max(gained, 0),
                    })

            delta = pd.DataFrame(rows)
            st.session_state["postrace_delta"] = delta

    delta = st.session_state.get("postrace_delta")
    if delta is not None and len(delta):
        st.dataframe(delta, width="stretch", hide_index=True)
        movers = delta[delta["Places gained"] > 0]
        if len(movers):
            top = movers.sort_values("Places gained", ascending=False).iloc[0]
            st.caption(
                f"Biggest opportunity: **{top['Driver']}** could have gained about "
                f"**{int(top['Places gained'])} place(s)** with the {top['Suggested strategy']} plan. "
                "Projected finish is the simulator's average position for that driver with the "
                "suggested plan, rivals unchanged — '0 places gained' means their real plan was "
                "already as good as anything we tried."
            )
        else:
            st.caption("The leading finishers all ran strategies the simulator couldn't beat — "
                       "the front-runners nailed it.")


# ---------------------------------------------------------------------------
# Best pit-stop plan map (optimal "when to pit")
# ---------------------------------------------------------------------------

T.section("Pit plan", "Best pit-stop plan map — when to pit",
          sub="The single-car ideal: for one car, the best call at every point in the race "
              "given how worn its tyres are.")

curves = T.load_tyre_curves()
if curves is None:
    T.warn("Tyre-wear model is missing — the pit-plan map and tyre check are unavailable.")
else:
    pol_compound = st.selectbox("Tyre you started on", ["SOFT", "MEDIUM", "HARD"], index=1,
                                key="pol_comp")
    two_used = st.checkbox("Second tyre type already used (rule satisfied)", value=False)
    try:
        base_pace = float(base["base_pace_s"].median()) if not base.empty else float(
            df2[df2["TrackStatus"] == "1"]["LapTimeSeconds"].median())
        model = build_race_model(circuit, era, n_laps, base_pace, curves)
        res = solve(model, start_compound=pol_compound)
        ci = COMPOUNDS.index(pol_compound)
        grid = res["policy"][1:, ci, :, int(two_used)]      # [lap, age]
        z = grid.T                                          # [age, lap]
        colorscale = [[0.0, "#2A2A33"], [0.25, "#2A2A33"],
                      [0.25, T.RED], [0.5, T.RED],
                      [0.5, T.COMPOUND_COLORS["MEDIUM"]], [0.75, T.COMPOUND_COLORS["MEDIUM"]],
                      [0.75, T.WHITE], [1.0, T.WHITE]]
        fig_pol = go.Figure(data=go.Heatmap(
            z=z, x=list(range(1, n_laps + 1)), y=list(range(MAX_AGE + 1)),
            zmin=-0.5, zmax=3.5, colorscale=colorscale,
            colorbar=dict(tickvals=[0, 1, 2, 3],
                          ticktext=["stay out", "pit→SOFT", "pit→MED", "pit→HARD"]),
        ))
        fig_pol.update_layout(xaxis_title="Lap",
                              yaxis_title=f"Laps on the {pol_compound} tyre")
        st.plotly_chart(T.style_fig(fig_pol, 400), width="stretch")
        st.markdown(
            "**How to read this map:** each square is the best call for a single car at that "
            "moment — the colour tells you what to do once your tyre has done that many laps "
            "(up the side) on that lap of the race (along the bottom): **stay out**, or "
            "**pit for SOFT / MEDIUM / HARD**. Reading up a column shows the lap where the "
            "advice flips from staying out to pitting — that's the ideal pit window."
        )
        st.caption(f"Ideal plan starting on {pol_compound}: "
                   f"{res['n_stops']} stop(s) — {res['stops']} · "
                   f"average lap {res['avg_lap_s']:.2f}s")
    except Exception as e:
        T.warn(f"Could not build the pit-plan map: <code>{e}</code>")


# ---------------------------------------------------------------------------
# Tyre wear: what the tyres actually did vs the model's expectation
# ---------------------------------------------------------------------------

if curves is not None:
    T.section("Tyre check", "How the tyres actually wore vs our model",
              sub="Each dot is a real lap this race (cleaned for fuel weight); the line is what "
                  "our tyre-wear model expected, with its likely range shaded.")

    try:
        prepped = validate_laps(df.copy())
        prepped = add_stint_id(prepped)
        feat_full = build_features(prepped, *load_encoding_mappings())
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
        st.caption(f"Could not prepare the actual tyre data: {e}")

    pol_comp = st.selectbox("Tyre compound", ["SOFT", "MEDIUM", "HARD"], index=1, key="tyre_comp")
    fig = go.Figure()
    if not stint_laps.empty:
        s = stint_laps[stint_laps["Compound"] == pol_comp]
        if not s.empty:
            fig.add_trace(go.Scatter(
                x=s["TyreLife"], y=s["Delta"], mode="markers", name="actual laps",
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
                                 name="likely range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ages, y=mid, mode="lines", name="model expectation",
                                 line=dict(color=color, width=2.4)))
    except Exception as e:
        st.caption(f"No tyre-wear curve available: {e}")
    fig.update_layout(xaxis_title="Laps on the tyre",
                      yaxis_title="Lap time lost vs a fresh tyre (s)")
    st.plotly_chart(T.style_fig(fig, 380), width="stretch")
    st.caption("Dots above the line = the tyre wore faster than expected this race; "
               "below = it lasted better than the model thought.")
