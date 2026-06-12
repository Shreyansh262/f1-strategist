"""F1 AI Race Strategist — dashboard landing / overview page.

Launch from the repo root:
    venv\\Scripts\\python.exe -m streamlit run dashboard/app.py

The three analysis pages live in dashboard/pages/ and appear in the sidebar.
All numbers on this page are read live from reports/ — no hardcoded figures.
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

T.page_config("F1 AI Race Strategist")
T.apply_theme()
T.register_plotly_template()


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
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div style="padding: 8px 0 0 0;">
      <p style="text-transform:uppercase;letter-spacing:5px;color:{T.RED};
                font-weight:700;font-size:0.8rem;margin-bottom:0;">
        Race Strategy · Pit Wall</p>
      <h1 style="font-size:3.0rem;margin:2px 0 0 0;line-height:1.05;">
        F1 AI RACE STRATEGIST</h1>
      <p style="color:{T.GREY};font-size:1.1rem;max-width:780px;margin-top:10px;">
        Lap-time prediction across regulation eras with a Temporal Fusion
        Transformer, physics-informed tyre-degradation curves, and an MDP pit
        policy — all fed <em>with their uncertainties</em> into a Monte-Carlo
        race engine that returns pit strategy as <strong style="color:{T.WHITE}">
        win-probability distributions</strong>, not point estimates.
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
    st.metric("TFT green MAE · 2026 test", val,
              help="Mean absolute lap-time error on green-flag laps, 2026 (new-regulation) test set.")
with c2:
    val = f"{k['green_mae_val']:.2f}s" if k["green_mae_val"] else "—"
    st.metric("TFT green MAE · 2025 val", val)
with c3:
    races = f"{k['n_races']:,}" if k["n_races"] else "—"
    laps = f"{k['n_laps']:,} laps" if k["n_laps"] else ""
    st.metric("Races ingested · 2022–2026", races, delta=laps, delta_color="off")
with c4:
    val = f"{k['sim_cov']:.0%}" if k["sim_cov"] else "—"
    st.metric("Sim replay coverage", val,
              help="Share of drivers' actual finishing positions inside the simulated central-80% band (target 80%).")

st.write("")

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------

T.section("Pipeline", "How it works")

cols = st.columns(4)
cards = [
    ("01 · Lap time", "Temporal Fusion Transformer",
     "A sequence model over laps-within-a-stint. Quantile outputs give calibrated "
     "uncertainty and an explicit Era feature handles the 2022→2026 regulation shift."),
    ("02 · Tyres", "Degradation curves",
     "Fuel-corrected quadratic deg per compound × circuit × era from scipy curve-fit, "
     "with analytic 95% confidence bands and hierarchical pooling for thin cells."),
    ("03 · Pit policy", "Exact MDP",
     "Backward induction over (lap × compound × tyre-age × two-compound-rule) gives "
     "the optimal single-car pit policy — an interpretable heatmap, the strategy proposer."),
    ("04 · Strategy", "Monte-Carlo engine",
     "Composes all three models with their uncertainties over thousands of vectorised "
     "rollouts → win / podium probability distributions. The MDP proposes, the sim disposes."),
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

# ---------------------------------------------------------------------------
# Page navigation
# ---------------------------------------------------------------------------

T.section("Explore", "Three analysis modes")

nav = st.columns(3)
nav_cards = [
    ("Pre-Race", "Plan a strategy",
     "Pick a circuit and era, build a driver lineup, inspect tyre-degradation curves, "
     "and rank candidate pit strategies by win probability against the field. View the "
     "MDP pit-policy heatmap."),
    ("Live Replay", "Re-run history lap-by-lap",
     "Step through any 2022–2026 race. Track position and gaps, per-driver tyre stints, "
     "actual-vs-predicted lap time with the calibration ribbon, and pit-window alerts."),
    ("Post-Race", "Audit what happened",
     "Per-driver predicted-vs-actual MAE, 'was the actual strategy optimal?' against the "
     "simulator, fitted-vs-actual tyre degradation, and error spikes correlated with SC/VSC."),
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
st.info("Use the sidebar to open **Pre-Race**, **Live Replay**, or **Post-Race**.", icon="🏁")

# sidebar footer
with st.sidebar:
    st.markdown('<p class="f1-label">F1 AI Strategist</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{T.GREY};font-size:0.85rem;">'
        "Local replay dashboard — no live-timing dependency. "
        "All models run on CPU.</p>",
        unsafe_allow_html=True,
    )
