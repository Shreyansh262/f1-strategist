"""Shared dashboard theme — F1 pit-wall aesthetic.

One place for the CSS injection, the Plotly dark template, the F1 colour
constants, and the cached data/model loaders every page reuses.

Design language (deliberate, NOT a generic AI look):
    - near-black carbon background, slightly lighter panels, hairline borders
    - F1 red (#E10600) for primary highlights / key numbers / active state
    - real F1 compound colours: SOFT red, MEDIUM yellow, HARD white
    - Titillium Web (F1's own typeface family is condensed sans; this is the
      closest free Google font), uppercase tracking on section labels
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MAPPINGS_DIR = PROJECT_ROOT / "data" / "mappings"

# ---------------------------------------------------------------------------
# Colour system
# ---------------------------------------------------------------------------

BG          = "#0E0E10"   # page carbon
BG_ALT      = "#15151E"   # F1 dark navy-black
PANEL       = "#1F1F27"   # cards / panels
BORDER      = "#2A2A33"   # hairline borders / grid lines
RED         = "#E10600"   # F1 primary red
WHITE       = "#F7F4F1"   # off-white text
GREY        = "#9498A1"   # muted secondary text
GREY_DIM    = "#5A5E66"

# Real F1 compound convention
COMPOUND_COLORS = {
    "SOFT":   "#E10600",
    "MEDIUM": "#FFD12E",
    "HARD":   "#F7F4F1",
    "INTERMEDIATE": "#43B02A",
    "WET":    "#1E6FE0",
}

# Restrained categorical palette (no rainbow, no purple)
COLORWAY = ["#E10600", "#F7F4F1", "#FFD12E", "#9498A1", "#43B02A", "#1E6FE0",
            "#E07B39", "#7A7E86"]


# ---------------------------------------------------------------------------
# CSS injection + Streamlit chrome hiding
# ---------------------------------------------------------------------------

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Titillium Web', -apple-system, 'Segoe UI', sans-serif;
}}
.stApp {{
    background: {BG};
    color: {WHITE};
}}

/* hide streamlit chrome */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header[data-testid="stHeader"] {{
    background: transparent;
    height: 0;
}}
.stDeployButton {{display: none;}}

/* sidebar */
section[data-testid="stSidebar"] {{
    background: {BG_ALT};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .stMarkdown {{ color: {GREY}; }}

/* headings */
h1, h2, h3, h4 {{
    color: {WHITE};
    font-weight: 700;
    letter-spacing: 0.5px;
}}
h1 {{ font-weight: 900; }}

/* section label helper */
.f1-label {{
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: 0.72rem;
    font-weight: 700;
    color: {RED};
    margin: 0 0 0.2rem 0;
}}
.f1-sub {{
    color: {GREY};
    font-size: 0.95rem;
    margin-top: -0.2rem;
}}

/* a thin red rule */
.f1-rule {{
    height: 3px; width: 56px; background: {RED};
    margin: 0.4rem 0 1.2rem 0; border: none;
}}

/* KPI metric cards */
div[data-testid="stMetric"] {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-left: 3px solid {RED};
    border-radius: 6px;
    padding: 14px 16px;
}}
div[data-testid="stMetric"] label p {{
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 0.70rem !important;
    color: {GREY} !important;
    font-weight: 600;
}}
div[data-testid="stMetricValue"] {{
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    color: {WHITE};
    font-size: 1.9rem;
}}
div[data-testid="stMetricDelta"] {{ font-size: 0.8rem; }}

/* ------------------------------------------------------------------ *
 * EQUAL-HEIGHT CARD ROWS
 *
 * Streamlit renders a column row as a deeply nested wrapper chain:
 *   stHorizontalBlock
 *     > stColumn
 *         > [stVerticalBlockBorderWrapper]   (only when border=True)
 *             > div
 *                 > stVerticalBlock
 *                     > stElementContainer        (one per element)
 *                         > stMarkdown
 *                             > div               (markdown body)
 *                                 > .f1-card
 * A plain `height:100%` on .f1-card only works if EVERY ancestor in that
 * chain is also a full-height flex container — otherwise the percentage has
 * no resolved parent height to inherit and the card collapses to content.
 * The previous fix only stretched the top two wrappers, so cards still grew
 * with their text. Here we make the whole chain a vertical flexbox and let
 * the markdown container that actually holds .f1-card grow to fill (flex:1).
 * A min-height fallback guarantees a short card still lines up if any future
 * Streamlit DOM change breaks an intermediate flex link.
 * ------------------------------------------------------------------ */
div[data-testid="stHorizontalBlock"] {{ align-items: stretch; }}
div[data-testid="stColumn"] {{ display: flex; flex-direction: column; }}
/* stretch every wrapper between the column and the card to 100% height,
   each itself a column flexbox so the next level can also stretch */
div[data-testid="stColumn"] > div,
div[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] > div,
div[data-testid="stColumn"] [data-testid="stVerticalBlock"] {{
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
}}
/* the element + markdown wrappers that directly hold .f1-card must grow to
   fill the leftover column height so the card inside can be height:100% */
div[data-testid="stColumn"] [data-testid="stVerticalBlock"]
    > div:has(.f1-card),
div[data-testid="stColumn"] [data-testid="stElementContainer"]:has(.f1-card),
div[data-testid="stColumn"] [data-testid="stMarkdown"]:has(.f1-card),
div[data-testid="stColumn"] [data-testid="stMarkdown"]:has(.f1-card) > div {{
    display: flex;
    flex-direction: column;
    flex: 1 1 auto;
    height: 100%;
}}

/* info / panel card */
.f1-card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 18px 20px;
    height: 100%;
    /* flex item so it fills the stretched wrapper chain above; the
       min-height fallback equalises short cards even if a flex link breaks */
    flex: 1 1 auto;
    min-height: 190px;
}}
.f1-card h4 {{ margin-top: 0; color: {WHITE}; }}
.f1-card p {{ color: {GREY}; font-size: 0.92rem; line-height: 1.5; }}
.f1-card .step {{
    color: {RED}; font-weight: 700; font-size: 0.72rem;
    letter-spacing: 2px; text-transform: uppercase;
}}

/* a styled warning/empty state */
.f1-warn {{
    background: rgba(225,6,0,0.06);
    border: 1px solid {RED};
    border-radius: 6px;
    padding: 16px 18px;
    color: {WHITE};
}}

/* dataframes */
.stDataFrame {{ border: 1px solid {BORDER}; border-radius: 6px; }}

/* buttons */
.stButton button {{
    background: {RED};
    color: {WHITE};
    border: none;
    border-radius: 4px;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
.stButton button:hover {{ background: #ff1a12; color: #fff; }}

/* tabs */
button[data-baseweb="tab"] {{ color: {GREY}; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {WHITE}; }}
div[data-baseweb="tab-highlight"] {{ background: {RED}; }}

/* compound chip */
.chip {{
    display:inline-block; padding:2px 10px; border-radius:10px;
    font-size:0.74rem; font-weight:700; letter-spacing:1px; margin-right:4px;
}}
</style>
"""


def apply_theme(page_title: str = "F1 AI Race Strategist", icon: str = "🏁") -> None:
    """Call once at the top of every page (after set_page_config)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_config(title: str) -> None:
    st.set_page_config(
        page_title=title,
        page_icon="🏁",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def section(label: str, title: str | None = None, sub: str | None = None) -> None:
    """Uppercase red kicker + optional title + thin rule."""
    st.markdown(f'<p class="f1-label">{label}</p>', unsafe_allow_html=True)
    if title:
        st.markdown(f"### {title}")
    st.markdown('<hr class="f1-rule">', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<p class="f1-sub">{sub}</p>', unsafe_allow_html=True)


def warn(msg: str) -> None:
    st.markdown(f'<div class="f1-warn">{msg}</div>', unsafe_allow_html=True)


def compound_chip(compound: str) -> str:
    c = COMPOUND_COLORS.get(compound, GREY)
    fg = "#0E0E10" if compound in ("MEDIUM", "HARD") else "#fff"
    return f'<span class="chip" style="background:{c};color:{fg}">{compound}</span>'


# ---------------------------------------------------------------------------
# Fan-friendly tables — plain labels for simulation finishing-order output.
# SHARED CONTRACT: every page renders sim/recommend tables through
# friendly_finish_table() so a casual fan never sees raw column names like
# "p95_finish". Do not remove or rename these without updating all pages.
# ---------------------------------------------------------------------------

FRIENDLY_FINISH_LABELS = {
    "driver": "Driver",
    "strategy": "Strategy",
    "start": "Start tyre",
    "win_prob": "Win chance",
    "podium_prob": "Podium chance",
    "mean_finish": "Avg finish",
    "p5_finish": "Best case",
    "p95_finish": "Worst case",
    "p50_time_s": "Median race time (s)",
    "base_pace_s": "Base pace (s/lap)",
    "grid_pos": "Grid",
}

# Tooltip text for the less-obvious columns (use with st.dataframe column_config
# or a small caption beneath the table).
FRIENDLY_FINISH_HELP = {
    "Win chance": "How often this driver finished 1st across all simulated races.",
    "Podium chance": "How often this driver finished in the top 3.",
    "Avg finish": "Average finishing position across simulated races (1 = win).",
    "Best case": "A realistic good day — they finished at least this high in 95% of sims.",
    "Worst case": "A realistic bad day — they finished no worse than this in 95% of sims.",
}


def friendly_finish_table(df: pd.DataFrame,
                          pct_cols: tuple[str, ...] = ("win_prob", "podium_prob")) -> pd.DataFrame:
    """Return a copy of a sim/recommend table with fan-friendly columns.

    - probability columns -> "63.0%" strings
    - average finish rounded to 2dp; best/worst-case positions to whole numbers
    - columns renamed via FRIENDLY_FINISH_LABELS (unknown columns left untouched)
    """
    out = df.copy()
    for c in pct_cols:
        if c in out.columns:
            out[c] = (out[c].astype(float) * 100).round(1).astype(str) + "%"
    if "mean_finish" in out.columns:
        out["mean_finish"] = out["mean_finish"].astype(float).round(2)
    for c in ("p5_finish", "p95_finish"):
        if c in out.columns:
            out[c] = out[c].astype(float).round(0).astype(int)
    if "p50_time_s" in out.columns:
        out["p50_time_s"] = out["p50_time_s"].astype(float).round(1)
    rename = {k: v for k, v in FRIENDLY_FINISH_LABELS.items() if k in out.columns}
    return out.rename(columns=rename)


# ---------------------------------------------------------------------------
# Plotly template
# ---------------------------------------------------------------------------

def register_plotly_template() -> None:
    pio.templates["f1"] = go.layout.Template(
        layout=dict(
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font=dict(family="Titillium Web, sans-serif", color=WHITE, size=13),
            colorway=COLORWAY,
            xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER,
                       tickcolor=BORDER, color=GREY),
            yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER,
                       tickcolor=BORDER, color=GREY),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=GREY)),
            margin=dict(l=50, r=20, t=40, b=40),
            hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER,
                            font=dict(color=WHITE, family="Titillium Web")),
            title=dict(font=dict(color=WHITE, size=16)),
        )
    )
    pio.templates.default = "f1"


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(template="f1", height=height)
    return fig


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_tyre_curves():
    p = MODELS_DIR / "tyre_curves.joblib"
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_resource(show_spinner=False)
def load_lap_model():
    p = MODELS_DIR / "chosen_lap.joblib"
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_resource(show_spinner=False)
def load_pit_loss_table():
    p = PROJECT_ROOT / "data" / "pit_loss.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def list_seasons() -> list[int]:
    seasons = set()
    for f in glob.glob(str(DATA_RAW / "laps_*_r*.parquet")):
        m = re.search(r"laps_(\d+)_r(\d+)", Path(f).stem)
        if m:
            seasons.add(int(m.group(1)))
    return sorted(seasons)


@st.cache_data(show_spinner=False)
def list_rounds(season: int) -> list[int]:
    rounds = []
    for f in glob.glob(str(DATA_RAW / f"laps_{season}_r*.parquet")):
        m = re.search(rf"laps_{season}_r(\d+)", Path(f).stem)
        if m:
            rounds.append(int(m.group(1)))
    return sorted(rounds)


@st.cache_data(show_spinner=False)
def load_race(season: int, rnd: int) -> pd.DataFrame | None:
    p = DATA_RAW / f"laps_{season}_r{rnd:02d}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = df.drop_duplicates(subset=["Season", "RoundNumber", "Driver", "LapNumber"])
    df["TrackStatus"] = df["TrackStatus"].astype(str)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def race_circuit(season: int, rnd: int) -> str | None:
    df = load_race(season, rnd)
    if df is None or df.empty:
        return None
    return str(df["CircuitKey"].iloc[0])


@st.cache_data(show_spinner=False)
def season_circuits(season: int) -> dict[int, str]:
    """round -> circuit name for a season."""
    out = {}
    for rnd in list_rounds(season):
        c = race_circuit(season, rnd)
        if c:
            out[rnd] = c
    return out


def era_for_season(season: int) -> int:
    return 0 if season <= 2025 else 1


@st.cache_data(show_spinner=False)
def read_csv_report(rel_path: str) -> pd.DataFrame | None:
    p = REPORTS_DIR / rel_path
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def green_base_pace(season: int, rnd: int) -> pd.DataFrame:
    """Per-driver green-flag median pace for a race — base pace for the sim.

    Returns DataFrame[Driver, Team, base_pace_s, grid_pos, n_green].
    grid_pos derived from lap-1 cumulative order proxy (lap-1 time rank).
    """
    df = load_race(season, rnd)
    if df is None or df.empty:
        return pd.DataFrame()
    green = df[
        (df["TrackStatus"] == "1")
        & df["PitInTime"].isna() & df["PitOutTime"].isna()
        & df["LapTimeSeconds"].between(60, 150)
    ]
    agg = (
        green.groupby(["Driver", "Team"])["LapTimeSeconds"]
        .agg(base_pace_s="median", n_green="size")
        .reset_index()
    )
    # grid proxy: order of cumulative time at the end of lap 1
    lap1 = df[df["LapNumber"] == 1][["Driver", "LapTimeSeconds"]].copy()
    lap1 = lap1.dropna().sort_values("LapTimeSeconds")
    grid = {d: i + 1 for i, d in enumerate(lap1["Driver"].tolist())}
    agg["grid_pos"] = agg["Driver"].map(grid).fillna(99).astype(int)
    agg = agg.dropna(subset=["base_pace_s"])
    return agg.sort_values("base_pace_s").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pre-race base pace from recent FORM (does not use the target race's own laps)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _race_green_field(season: int, rnd: int):
    """(per-driver green median df, field green median) for one race, or None.

    Helper for form_base_pace. The per-driver median minus the field median is a
    circuit-neutral 'how fast was this driver vs the field that day' signal.
    """
    df = load_race(season, rnd)
    if df is None or df.empty:
        return None
    green = df[
        (df["TrackStatus"] == "1")
        & df["PitInTime"].isna() & df["PitOutTime"].isna()
        & df["LapTimeSeconds"].between(60, 150)
    ]
    if green.empty:
        return None
    per = (green.groupby(["Driver", "Team"])["LapTimeSeconds"]
           .median().reset_index(name="med"))
    return per, float(green["LapTimeSeconds"].median())


@st.cache_data(show_spinner=False)
def _races_before(season: int, rnd: int) -> list[tuple[int, int]]:
    """All (season, round) strictly before the target race, newest first."""
    out = []
    for s in list_seasons():
        for r in list_rounds(s):
            if (s, r) < (season, rnd):
                out.append((s, r))
    out.sort(reverse=True)
    return out


@st.cache_data(show_spinner=False)
def form_base_pace(season: int, rnd: int, n_recent: int = 5) -> pd.DataFrame:
    """Per-driver base pace for a PRE-race plan, WITHOUT using the target race's
    own laps (removes the circular "plan a race we already have the laps for").

        base_pace_s(driver) = circuit_baseline + recent_form_delta(driver)

      - recent_form_delta: median over the driver's last `n_recent` races of
        (their green median − that race's field green median). Gap-to-field, so it
        is comparable across circuits of very different lap length.
      - circuit_baseline: field green median from the most recent PRIOR running of
        this circuit. Falls back to the target race's field median ONLY if the
        circuit has never been raced before in the dataset (flagged in baseline_src).

    Entry list + grid come from the target race — both legitimately known before
    lights-out (entries from the weekend, grid from qualifying); only in-race PACE
    is withheld. Returns DataFrame[Driver, Team, base_pace_s, grid_pos, n_form, baseline_src].
    """
    target = load_race(season, rnd)
    if target is None or target.empty:
        return pd.DataFrame()

    entries = target[["Driver", "Team"]].drop_duplicates()
    lap1 = (target[target["LapNumber"] == 1][["Driver", "LapTimeSeconds"]]
            .dropna().sort_values("LapTimeSeconds"))
    grid = {d: i + 1 for i, d in enumerate(lap1["Driver"].tolist())}

    priors = _races_before(season, rnd)
    drivers = list(entries["Driver"])
    deltas: dict[str, list[float]] = {d: [] for d in drivers}
    counts: dict[str, int] = {d: 0 for d in drivers}
    for s, r in priors:
        if all(counts[d] >= n_recent for d in drivers):
            break
        res = _race_green_field(s, r)
        if res is None:
            continue
        per, fmed = res
        for _, row in per.iterrows():
            d = row["Driver"]
            if d in counts and counts[d] < n_recent:
                deltas[d].append(float(row["med"]) - fmed)
                counts[d] += 1

    form_delta = {d: (float(np.median(v)) if v else np.nan) for d, v in deltas.items()}
    finite = [v for v in form_delta.values() if pd.notna(v)]
    fallback_delta = float(np.median(finite)) if finite else 0.0

    circuit = str(target["CircuitKey"].iloc[0])
    baseline = None
    baseline_src = "most recent prior running of this circuit"
    for s, r in priors:
        cdf = load_race(s, r)
        if cdf is None or cdf.empty or str(cdf["CircuitKey"].iloc[0]) != circuit:
            continue
        res = _race_green_field(s, r)
        if res is not None:
            baseline = res[1]
            break
    if baseline is None:
        res = _race_green_field(season, rnd)
        baseline = res[1] if res is not None else 90.0
        baseline_src = "this race (circuit has no prior running in the data)"

    rows = []
    for _, e in entries.iterrows():
        d = e["Driver"]
        fd = form_delta.get(d, np.nan)
        if pd.isna(fd):
            fd = fallback_delta
        rows.append({
            "Driver": d, "Team": e["Team"],
            "base_pace_s": baseline + fd,
            "grid_pos": grid.get(d, 99),
            "n_form": counts.get(d, 0),
            "baseline_src": baseline_src,
        })
    return pd.DataFrame(rows).sort_values("base_pace_s").reset_index(drop=True)
