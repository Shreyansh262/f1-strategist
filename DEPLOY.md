# Phase 7 — Deployment Guide

Target: Hugging Face Spaces (Streamlit SDK).
Fallback: Render (see end of this file).

---

## Why Hugging Face Spaces over Render

| Factor | HF Spaces | Render free |
|---|---|---|
| RAM (free tier) | 16 GB | 512 MB |
| Disk | 50 GB | ~1 GB ephemeral |
| Sleep | No (persistent) | After 15 min inactivity |
| GPU (free) | No (CPU) | No |
| Cold start | ~60 s on first load | ~60 s after sleep |
| Custom domain | Yes (paid) | Yes (free) |

The simulation engine holds 1000-rollout × 20-driver × 60-lap torch tensors in
memory — well under 1 GB but far above Render's 512 MB limit.
HF Spaces free tier is the correct choice.

> HF Spaces free-tier numbers as of the HF documentation (June 2025).
> Verify current limits at https://huggingface.co/docs/hub/spaces-overview before
> deploying, as these numbers can change.

---

## Resource expectations

| Component | At rest | Peak (1000-rollout sim) |
|---|---|---|
| Dashboard + libs | ~400 MB RSS | ~400 MB |
| chosen_lap.joblib (LightGBM) | ~1 MB | ~1 MB |
| tyre_curves.joblib | <1 MB | <1 MB |
| torch tensors (sim) | ~0 | ~200-400 MB |
| **Total estimated peak** | | **~800 MB - 1 GB** |

Well within the 16 GB HF free-tier RAM limit.

The TFT checkpoint (`tft_lap.ckpt`, 4.5 MB) is **not** included in the bundle.
The engine samples distributional summaries from `tft_calibration.json` + the
recalibration CSV rather than running TFT network inference per lap.
Predictions in the Live Replay / Post-Race pages use `chosen_lap.joblib`
(LightGBM). `pytorch-forecasting` and `lightning` are therefore not required.

---

## Exact deploy steps

### Prerequisites
- Git and `git-lfs` installed locally (run `git lfs install` once if not done).
- A Hugging Face account at https://huggingface.co.
- The `huggingface_hub` CLI or plain git push (either works).

### Step 1 — Run the export script

From the repo root with the project venv active:

```
python scripts/export_deploy_bundle.py
```

This writes `deploy_bundle/` (~6-10 MB total). Review the printed size manifest.
The script is idempotent — re-run any time models or data are updated.

### Step 2 — Create the HF Space

1. Go to https://huggingface.co/new-space.
2. Fill in:
   - **Owner**: your HF username.
   - **Space name**: `f1-strategist` (or any name you like).
   - **SDK**: **Streamlit** (not Docker — the Streamlit SDK is simpler and the
     `requirements.txt` already handles the CPU torch wheel).
   - **Hardware**: CPU Basic (free).
   - **Visibility**: Public (required for free tier; set Private if you upgrade).
3. Click **Create Space**. HF creates an empty git repo at
   `https://huggingface.co/spaces/<username>/f1-strategist`.

### Step 3 — Push the bundle

```bash
cd deploy_bundle

# Initialise as a git repo pointing at your HF Space
git init
git branch -M main
git remote add origin https://huggingface.co/spaces/<your-username>/f1-strategist

# Track large files with Git LFS (parquet + joblib are ~1 MB each, fine without
# LFS, but good practice; HF requires LFS for files > 10 MB)
git lfs install
git lfs track "*.parquet"
git lfs track "*.joblib"
git add .gitattributes

# Commit and push
git add .
git commit -m "Phase 7: initial deploy"
git push -u origin main
```

**Authentication:** HF git remotes do not accept account passwords. Create a
WRITE-scope access token at https://huggingface.co/settings/tokens and use it
as the password when git prompts (username = your HF username). Alternatively,
run `pip install -U huggingface_hub` then `hf auth login` (older CLIs:
`huggingface-cli login`) before pushing.

HF will detect `requirements.txt`, install deps, find `dashboard/app.py`
(the `app_file` in `README.md` front-matter), and launch Streamlit.
HF Spaces natively supports a subdirectory `app_file` — no root wrapper needed.
Build logs are visible in the Space's "Logs" tab.

### Step 4 — Verify

1. Wait ~3-5 min for the build and first launch.
2. Open `https://huggingface.co/spaces/<username>/f1-strategist`.
3. The Overview page should show the KPI strip (TFT MAE values from the CSV).
4. Open Live Replay → choose Season 2025 → any round → confirm laps load.
5. Open Pre-Race → confirm tyre curves render.
6. The sidebar must list **Pre-Race**, **Live Replay**, **Post-Race** — if pages
   are missing, `app_file` in `README.md` front-matter is not `dashboard/app.py`
   (Streamlit discovers `pages/` relative to the main script, so the entrypoint
   must be `dashboard/app.py`, not a root wrapper).

If the app shows a missing-artifact warning for a `.joblib` file, the Git LFS
pointer was not resolved. Re-check `git lfs push --all origin main`.

---

## SDK choice rationale

**Streamlit SDK** (not Docker) was chosen because:
1. HF Spaces natively understands Streamlit — `dashboard/app.py` + `requirements.txt`.
   The `app_file: dashboard/app.py` front-matter in `README.md` points HF at the real
   app; HF Spaces natively supports subdirectory entrypoints so no root wrapper is needed.
2. No Dockerfile maintenance, no port binding, no CMD.
3. The only non-standard requirement is the `--extra-index-url` line for the CPU torch
   wheel, which pip handles fine in `requirements.txt`.
4. The dashboard already runs correctly with `streamlit run`, so no adapter layer
   is needed.

The CPU-only torch wheel (`torch==2.11.0+cpu`, ~240 MB) keeps the installed
footprint reasonable. `requirements.txt` uses `--extra-index-url https://download.pytorch.org/whl/cpu`
(not `--index-url`) so that PyPI remains available for non-torch packages — `--index-url`
would replace PyPI entirely and cause streamlit, plotly, pandas etc. to fail to install.
`torch==2.11.0+cpu` resolves only from the CPU index due to the `+cpu` local version specifier.
HF pip-installs this once and caches it.

---

## Updating the deployed app

To update models or data after retraining:

```bash
python scripts/export_deploy_bundle.py   # rebuild the bundle
cd deploy_bundle
git add .
git commit -m "Update: <describe change>"
git push
```

HF will rebuild the Space automatically.

---

## Fallback: Render

If HF Spaces is unavailable or you need a custom domain on the free tier, use
Render (https://render.com).

**Key differences:**
- Render free tier provides only ~512 MB RAM. The Monte-Carlo simulation engine
  (torch tensor rollouts) will likely OOM at 1000 rollouts × 20 drivers.
  Reduce `n_rollouts` default to 200-300 in `dashboard/pages/1_Pre_Race.py`
  (the slider default), or use Render's Starter plan ($7/mo, 512 MB → 1 GB).
- Render requires a `Procfile` or a `render.yaml`. A minimal `Procfile`:

  ```
  web: streamlit run dashboard/app.py --server.port=$PORT --server.headless=true
  ```

- Models and data must be committed to the repo (or fetched via `render.yaml`
  build command). The same bundle structure applies.
- Render free tier sleeps after 15 min of inactivity (30–60 s cold start).

For a portfolio demo, HF Spaces is strictly better unless you need Render's
custom domain on the free tier.
