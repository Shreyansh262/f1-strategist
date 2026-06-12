"""Phase 7 — Export deploy bundle for Hugging Face Spaces.

Creates deploy_bundle/ at the repo root containing everything needed to run
the Streamlit dashboard on Hugging Face Spaces:

    deploy_bundle/
      dashboard/                    <- copied verbatim (app.py is the HF entrypoint)
      src/                          <- copied (ingest.py excluded — not needed at serve time)
      models/                       <- chosen_lap.joblib, tyre_curves.joblib, tft_calibration.json
      data/
        pit_loss.json
        mappings/                   <- circuit_map.json, team_map.json, driver_map.json
        raw/                        <- curated parquet subset (see DATA_STRATEGY below)
      reports/
        lap_time/                   <- CSVs needed by the Overview KPI strip
        simulation/                 <- demo_recommendation.csv, validation_summary.md
      requirements-deploy.txt
      README.md                     <- HF Spaces YAML front-matter + project README
      .streamlit/config.toml        <- copied from dashboard/.streamlit/config.toml

DATA_STRATEGY:
  The dashboard reads per-race parquet files from data/raw/ (laps_{year}_r{nn}.parquet)
  to drive all three pages.  Full seasons are NOT needed — only the per-round files
  so a user can pick any race in the replay / post-race pages.  The large "full" season
  files are training artefacts used only in train.py and are excluded.

  Included:
    - data/raw/laps_2025_r*.parquet  (22 files, ~0.6 MB total) — full 2025 season for
      rich live-replay / post-race demos
    - data/raw/laps_2026_r*.parquet  (all available 2026 files) — the new-era test set
    - data/raw/laps_2025_full.parquet — overview page counts n_laps via the full file
      if present (graceful: falls back to per-round sum)

  Excluded:
    - laps_2022/2023/2024 per-round and full files  — 2022-2024 are train/val seasons;
      including them adds ~3MB with no dashboard value (user can't see them better than 2025)
    - tft_lap.ckpt / tft_lap.pt  — TFT checkpoint NOT needed at serve time.  The engine
      samples distributional summaries (sigma from tft_calibration.json) and predictions
      use chosen_lap.joblib (LightGBM).  Checkpoint is 4.5 MB and requires pytorch-forecasting.
    - rf_lap.joblib / bayesian_ridge_lap.joblib / lgbm_lap.joblib  — only chosen_lap.joblib
      is loaded at runtime (the others are training artefacts).

Run from the repo root:
    python scripts/export_deploy_bundle.py

Re-running is idempotent — the bundle is wiped and rebuilt each time.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BUNDLE       = PROJECT_ROOT / "deploy_bundle"

# ---------------------------------------------------------------------------
# Files / directories to include
# ---------------------------------------------------------------------------

MODEL_ARTIFACTS = [
    "models/chosen_lap.joblib",
    "models/tyre_curves.joblib",
    "models/tft_calibration.json",
]

DATA_FILES = [
    "data/pit_loss.json",
    "data/mappings/circuit_map.json",
    "data/mappings/team_map.json",
    "data/mappings/driver_map.json",
]

REPORT_FILES_LAP_TIME = [
    "reports/lap_time/model_comparison.csv",
    "reports/lap_time/tft_recalibration.csv",
    "reports/lap_time/tft_calibration.csv",
    "reports/lap_time/tft_breakdown.csv",
    "reports/lap_time/per_circuit_mae.csv",
    "reports/lap_time/per_era_mae.csv",
    "reports/lap_time/shap_summary.png",
]

REPORT_FILES_SIMULATION = [
    "reports/simulation/validation_summary.md",
    "reports/simulation/validation.csv",
    "reports/simulation/demo_recommendation.csv",
]

# Source dirs whose Python files are copied (no training deps excluded)
SRC_MODULES_TO_COPY = [
    "src/__init__.py",
    "src/simulation/__init__.py",
    "src/simulation/engine.py",
    "src/models/__init__.py",
    "src/models/tyre/fit.py",
    # src/models/tyre/__init__.py — may not exist (implicit namespace pkg); created below
    "src/models/pit_strategy/pit_loss.py",
    "src/models/pit_strategy/mdp.py",
    # src/models/pit_strategy/__init__.py — may not exist; created below
    "src/models/lap_time/__init__.py",
    "src/models/lap_time/train.py",        # needed for get_X_y() by the dashboard
    "src/models/lap_time/train_tft_data.py",  # needed for load_tft_data / load_encoding_mappings carryback
    "src/pipeline/__init__.py",
    "src/pipeline/features.py",
    "src/pipeline/validate.py",
    # Explicitly excluded (not needed at serve time, bring heavy deps):
    #   src/pipeline/ingest.py         — fastf1 import
    #   src/models/lap_time/train_tft.py    — pytorch-forecasting
    #   src/models/lap_time/recalibrate.py  — pytorch-forecasting
    #   src/models/lap_time/evaluate.py     — only for training/eval
    #   src/pipeline/splits.py             — training only
    #   src/api/main.py                    — FastAPI, not used in Streamlit deploy
    #   src/simulation/validate_sim.py     — training evaluation only
]

# Parquet selection logic is in _collect_parquet() below.

# ---------------------------------------------------------------------------
# Size manifest helper
# ---------------------------------------------------------------------------

def _size_str(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    if nbytes < 1024 ** 2:
        return f"{nbytes/1024:.1f} KB"
    return f"{nbytes/1024/1024:.2f} MB"


def _manifest(bundle: Path) -> None:
    """Print a size breakdown of everything in the bundle."""
    categories: dict[str, int] = {}
    total = 0
    for p in sorted(bundle.rglob("*")):
        if p.is_dir():
            continue
        size = p.stat().st_size
        total += size
        rel = p.relative_to(bundle)
        top = str(rel.parts[0]) if len(rel.parts) > 1 else "root"
        categories[top] = categories.get(top, 0) + size

    print("\n=== Deploy bundle size manifest ===")
    for cat, sz in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:<28} {_size_str(sz)}")
    print(f"  {'TOTAL':<28} {_size_str(total)}")
    print()


# ---------------------------------------------------------------------------
# Parquet collection
# ---------------------------------------------------------------------------

def _collect_parquet(src_raw: Path, dst_raw: Path) -> None:
    """Copy the curated subset of raw parquet files to the bundle."""
    copied = 0
    skipped = 0

    for pq in sorted(src_raw.glob("*.parquet")):
        name = pq.name
        # Include all 2025 per-round files, all 2026 files, and laps_2025_full
        if (
            (name.startswith("laps_2025_r") and name.endswith(".parquet"))
            or (name.startswith("laps_2026_") and name.endswith(".parquet"))
            or name == "laps_2025_full.parquet"
        ):
            dst = dst_raw / name
            shutil.copy2(pq, dst)
            copied += 1
        else:
            skipped += 1

    print(f"  Parquet: {copied} files copied, {skipped} excluded (2022-2024 train seasons)")


# ---------------------------------------------------------------------------
# HF Space README with YAML front-matter
# ---------------------------------------------------------------------------

_HF_README = """\
---
title: F1 AI Race Strategist
emoji: 🏎️
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: "1.55.0"
app_file: dashboard/app.py
pinned: false
license: mit
short_description: Lap-time TFT · Tyre curves · MDP pit policy · Monte-Carlo engine
---

# F1 AI Race Strategist

An end-to-end AI system for Formula 1 race strategy, built as a portfolio project.

**Three-page Streamlit dashboard:**
- **Pre-Race** — predict lap-time distributions, rank candidate pit strategies by win probability via Monte-Carlo simulation, view the MDP optimal-pit heatmap.
- **Live Replay** — step through any 2022–2026 Grand Prix lap-by-lap with actual-vs-predicted overlay and pit-window alerts.
- **Post-Race** — audit a finished race: predicted-vs-actual MAE per driver, "was the actual strategy optimal?", tyre-deg actual vs fitted, and error spikes vs SC/VSC.

**Stack:** Temporal Fusion Transformer (lap time + uncertainty) · scipy curve-fit tyre degradation · NumPy MDP (backward induction) · PyTorch vectorised Monte-Carlo engine · LightGBM (chosen serving model).

## Source code

Full project: [github.com/Shreyansh262/f1-strategist](https://github.com/Shreyansh262/f1-strategist)
"""


# ---------------------------------------------------------------------------
# requirements-deploy.txt
# ---------------------------------------------------------------------------

_REQUIREMENTS_DEPLOY = """\
# Minimal requirements for HF Spaces Streamlit deployment.
# TFT checkpoint is NOT loaded at serve time (engine uses distributional summaries;
# dashboard uses LightGBM chosen_lap.joblib). pytorch-forecasting and lightning
# are therefore excluded — this cuts ~500 MB from the installed footprint.
#
# torch CPU wheel is required because engine.py uses torch tensors for vectorised
# Monte-Carlo rollouts (device="cpu"). The --extra-index-url adds the CPU-only
# wheel index alongside PyPI so non-torch packages (streamlit, plotly, pandas, …)
# still resolve from PyPI. Using --index-url instead would replace PyPI entirely
# and cause those packages to fail to install on the PyTorch CPU index.
# torch==2.11.0+cpu resolves only from the CPU index due to the +cpu local version.
# (~240 MB wheel vs ~900 MB CUDA build).
#
# fastf1 is NOT required: the dashboard reads pre-built parquet files from data/raw/
# and never calls fastf1 at runtime (fastf1 is only used in src/pipeline/ingest.py
# which is intentionally excluded from the deploy bundle).
#
# Pinned versions match the dev environment; loosen if a future HF base image conflicts.

streamlit==1.55.0
plotly==6.8.0
pandas==2.3.3
numpy==2.4.3
scipy==1.17.1
scikit-learn==1.8.0
lightgbm==4.6.0
joblib==1.5.3

# CPU-only PyTorch — required for engine.py Monte-Carlo rollouts.
# HF Spaces free tier has no GPU; this wheel is ~240 MB vs 900 MB CUDA build.
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.11.0+cpu
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Building deploy bundle at: {BUNDLE}")

    # Wipe and recreate
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    errors: list[str] = []

    # ---- Model artifacts -----------------------------------------------------
    print("\n[1/5] Model artifacts")
    for rel in MODEL_ARTIFACTS:
        src = PROJECT_ROOT / rel
        dst = BUNDLE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  copied {rel} ({_size_str(src.stat().st_size)})")
        else:
            errors.append(f"MISSING model artifact: {rel}")
            print(f"  MISSING {rel} — will warn at runtime")

    # ---- Data files (non-parquet) --------------------------------------------
    print("\n[2/5] Data files")
    for rel in DATA_FILES:
        src = PROJECT_ROOT / rel
        dst = BUNDLE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  copied {rel}")
        else:
            errors.append(f"MISSING data file: {rel}")
            print(f"  MISSING {rel}")

    # ---- Parquet (curated subset) --------------------------------------------
    print("\n[3/5] Raw parquet (curated subset)")
    src_raw = PROJECT_ROOT / "data" / "raw"
    dst_raw = BUNDLE / "data" / "raw"
    dst_raw.mkdir(parents=True, exist_ok=True)
    if src_raw.exists():
        _collect_parquet(src_raw, dst_raw)
    else:
        errors.append("MISSING data/raw directory")

    # ---- Reports -------------------------------------------------------------
    print("\n[4/5] Reports")
    for rel in REPORT_FILES_LAP_TIME + REPORT_FILES_SIMULATION:
        src = PROJECT_ROOT / rel
        dst = BUNDLE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  copied {rel}")
        else:
            print(f"  skipped (absent) {rel}")

    # ---- Dashboard + src -----------------------------------------------------
    print("\n[5/5] Dashboard and src code")

    # Copy dashboard/ entirely (theme.py, app.py, pages/, .streamlit/)
    src_dashboard = PROJECT_ROOT / "dashboard"
    dst_dashboard = BUNDLE / "dashboard"
    if src_dashboard.exists():
        shutil.copytree(
            src_dashboard,
            dst_dashboard,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        print(f"  copied dashboard/ ({sum(1 for _ in dst_dashboard.rglob('*') if _.is_file())} files)")
    else:
        errors.append("MISSING dashboard/ directory")

    # Copy selected src/ modules (exclude training + fastf1 deps)
    copied_src = 0
    for rel in SRC_MODULES_TO_COPY:
        # Skip comment-only entries (lines starting with #)
        if rel.startswith("#"):
            continue
        src = PROJECT_ROOT / rel
        dst = BUNDLE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
            copied_src += 1
        else:
            print(f"  skipped {rel} (absent in source — will be created as stub below)")

    # Ensure __init__.py stubs exist for packages that don't have them in source.
    # Python 3 supports implicit namespace packages, but explicit __init__.py
    # prevents edge-case import issues on some hosting environments.
    _stub_inits = [
        "src/models/tyre/__init__.py",
        "src/models/pit_strategy/__init__.py",
    ]
    for rel in _stub_inits:
        dst = BUNDLE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.write_text("# auto-generated stub\n", encoding="utf-8")
            copied_src += 1
    print(f"  copied/created {copied_src} src modules (stubs for missing __init__.py)")

    # ---- HF Space generated files (requirements, README, .streamlit) ----------

    # requirements-deploy.txt
    (BUNDLE / "requirements-deploy.txt").write_text(_REQUIREMENTS_DEPLOY, encoding="utf-8")
    # HF Spaces uses requirements.txt as the pip install target
    (BUNDLE / "requirements.txt").write_text(_REQUIREMENTS_DEPLOY, encoding="utf-8")
    print("  wrote requirements-deploy.txt + requirements.txt (identical)")

    # README.md with HF YAML front-matter
    (BUNDLE / "README.md").write_text(_HF_README, encoding="utf-8")
    print("  wrote README.md (HF Spaces YAML front-matter)")

    # Copy .streamlit/config.toml from dashboard/
    streamlit_src = PROJECT_ROOT / "dashboard" / ".streamlit" / "config.toml"
    streamlit_dst = BUNDLE / ".streamlit" / "config.toml"
    streamlit_dst.parent.mkdir(parents=True, exist_ok=True)
    if streamlit_src.exists():
        shutil.copy2(streamlit_src, streamlit_dst)
        print("  copied .streamlit/config.toml")
    else:
        print("  skipped .streamlit/config.toml (not found)")

    # ---- Purge __pycache__ from bundle (local .pyc files are platform-specific) --
    removed_caches = 0
    for pycache in list(BUNDLE.rglob("__pycache__")):
        if pycache.is_dir():
            shutil.rmtree(pycache)
            removed_caches += 1
    if removed_caches:
        print(f"\n[cleanup] Removed {removed_caches} __pycache__ directories")

    # ---- Size manifest -------------------------------------------------------
    _manifest(BUNDLE)

    # ---- Error summary -------------------------------------------------------
    if errors:
        print("=== WARNINGS (missing artifacts — will degrade gracefully at runtime) ===")
        for e in errors:
            print(f"  {e}")
        print()
    else:
        print("All required artifacts present.")

    print(f"Bundle ready at: {BUNDLE}")
    print("Next step: run DEPLOY.md instructions (create HF Space -> push bundle).")


if __name__ == "__main__":
    main()
