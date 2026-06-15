"""F1 AI Race Strategist — dashboard entrypoint + navigation.

Launch from the repo root:
    venv\\Scripts\\python.exe -m streamlit run dashboard/app.py

Uses st.navigation so the sidebar shows clean labels (Dashboard / Pre-Race /
Race Replay / Post-Race) instead of raw filenames — the landing entry would
otherwise read "app" (the filename). The landing page is the home() function
below; the three analysis pages live in dashboard/pages/. All numbers on the
landing page are read live from reports/ — no hardcoded figures.

NOTE: app.py stays the HF Spaces entrypoint (app_file: dashboard/app.py); only
its internals changed. With st.navigation, set_page_config is called once here
and the page scripts no longer call it themselves.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Make both `dashboard` (for theme) and the repo root (for `src`) importable,
# regardless of Streamlit's CWD.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import streamlit as st

import theme as T


# ---------------------------------------------------------------------------
# Headline KPIs (live from reports/)
# ---------------------------------------------------------------------------

def _kpis() -> dict:
    out = {
        "green_mae_val": None, "green_mae_test": None,
        "n_races": None, "n_laps": None, "sim_cov": None,
    }
    cmp = T.read_csv_report("lap_time/model_comparison.csv")
    if cmp is not None:
        tft = cmp[cmp["model"] == "TFT"]
        v = tft[tft["split"] == "val"]
        t = tft[tft["split"] == "test"]
        if len(v):
            out["green_mae_val"] = float(v["green_mae"].iloc[0])
        if len(t):
            out["green_mae_test"] = float(t["green_mae"].iloc[0])

    # races + laps ingested (per-round parquet only)
    files = sorted(T.DATA_RAW.glob("laps_*_r*.parquet"))
    out["n_races"] = len(files)
    try:
        n = 0
        for f in files:
            n += len(pd.read_parquet(f, columns=["LapNumber"]))
        out["n_laps"] = n
    except Exception:
        out["n_laps"] = None

    # sim coverage from validation_summary.md
    p = T.REPORTS_DIR / "simulation" / "validation_summary.md"
    if p.exists():
        m = re.search(r"Overall:\s*\**([\d.]+)", p.read_text(encoding="utf-8"))
        if m:
            out["sim_cov"] = float(m.group(1))
    return out


# ---------------------------------------------------------------------------
# Landing / overview page (registered as the default nav page)
# ---------------------------------------------------------------------------

def home() -> None:
    T.apply_theme()
    T.register_plotly_template()

    st.markdown(
        f"""
        <div style="padding: 8px 0 0 0;">
          <p style="text-transform:uppercase;letter-spacing:5px;color:{T.RED};
                    font-weight:700;font-size:0.8rem;margin-bottom:0;">
            Race Strategy · Pit Wall</p>
          <h1 style="font-size:3.0rem;margin:2px 0 0 0;line-height:1.05;">
            F1 AI RACE STRATEGIST</h1>
          <p style="color:{T.GREY};font-size:1.1rem;max-width:780px;margin-top:10px;">
            We predict how fast every car laps, how its tyres wear, and the best
            laps to pit — then run the whole race thousands of times to turn all
            of that into <strong style="color:{T.WHITE}">win and podium chances</strong>,
            not a single guess.
          </p>
        </div>
        <hr style="border:none;border-top:1px solid {T.BORDER};margin:18px 0 26px 0;">
        """,
        unsafe_allow_html=True,
    )

    k = _kpis()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        val = f"{k['green_mae_test']:.2f}s" if k["green_mae_test"] else "—"
        st.metric("Lap-time accuracy · 2026 (sec error)", val,
                  help="On average, our lap-time prediction is off by about this many "
                       "seconds per lap on 2026 races (the current rules). Lower is better.")
    with c2:
        val = f"{k['green_mae_val']:.2f}s" if k["green_mae_val"] else "—"
        st.metric("Lap-time accuracy · 2025 (sec error)", val,
                  help="Same lap-time accuracy measured on 2025 races. Lower is better.")
    with c3:
        races = f"{k['n_races']:,}" if k["n_races"] else "—"
        laps = f"{k['n_laps']:,} laps" if k["n_laps"] else ""
        st.metric("Races studied · 2022–2026", races, delta=laps, delta_color="off",
                  help="Number of real Grand Prix races (and total laps) the models "
                       "learned from, covering 2022 through 2026.")
    with c4:
        val = f"{k['sim_cov']:.0%}" if k["sim_cov"] else "—"
        st.metric("Simulator reliability", val,
                  help="How often a driver's real finishing position landed inside the "
                       "range our race simulator predicted. Closer to 80% means the "
                       "simulator's confidence is well calibrated.")

    st.write("")

    # ---- How it works ----
    T.section("What we built", "How it works")

    cols = st.columns(4)
    cards = [
        ("01 · Lap time", "How fast each car laps",
         "We predict each car's lap time — and, just as importantly, how sure we are "
         "about it. Some cars and conditions are far more predictable than others."),
        ("02 · Tyre wear", "How tyres slow down",
         "Tyres lose grip as they age, and they do it differently for each tyre type "
         "and each track. We model that wear — including wet-weather tyres (still a "
         "rough, early estimate)."),
        ("03 · Pit strategy", "The best lap to pit",
         "Pitting too early or too late costs places. We work out the best lap (or laps) "
         "for a car to come in and change tyres."),
        ("04 · Race simulator", "Win and podium chances",
         "We run the whole race thousands of times, mixing in all the uncertainty above, "
         "to turn it into win and podium chances — not one single guess."),
    ]
    for col, (step, title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f"""<div class="f1-card">
                      <p class="step">{step}</p>
                      <h4>{title}</h4>
                      <p>{body}</p>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    st.write("")

    # ---- Page navigation cards ----
    T.section("Explore", "Four ways to use it")

    nav = st.columns(4)
    nav_cards = [
        ("Pre-Race", "Predict an upcoming race",
         "Set the grid and the weather, then simulate the finishing order before the "
         "lights go out."),
        ("Race Replay", "Re-run a past race",
         "Replay any past race lap-by-lap, then try a different strategy mid-race to see "
         "if it would have helped."),
        ("Post-Race", "Audit a finished race",
         "Look back at a finished race: was the strategy the best one, and how could the "
         "teams have done better?"),
        ("Live", "Follow a live race",
         "Follow a race while it's actually running, with a flag-aware projection of the "
         "finish."),
    ]
    for col, (label, title, body) in zip(nav, nav_cards):
        with col:
            st.markdown(
                f"""<div class="f1-card">
                      <p class="step">{label}</p>
                      <h4>{title}</h4>
                      <p>{body}</p>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    st.caption("Open any of these from the sidebar.")

    # sidebar footer
    with st.sidebar:
        st.markdown('<p class="f1-label">F1 AI Strategist</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:{T.GREY};font-size:0.85rem;">'
            "Local replay dashboard — no live-timing dependency. "
            "All models run on CPU.</p>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Navigation — single set_page_config here, then route. Page scripts must NOT
# call set_page_config (st.navigation runs them after this entrypoint).
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="F1 AI Race Strategist",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page(home, title="Dashboard", default=True),
    st.Page("pages/1_Pre_Race.py", title="Pre-Race"),
    st.Page("pages/2_Live_Replay.py", title="Race Replay"),
    st.Page("pages/3_Post_Race.py", title="Post-Race"),
    st.Page("pages/4_Live.py", title="Live"),
]
st.navigation(pages).run()
