"""FastAPI serving layer for the F1 AI Race Strategist — Phase 6.

Exposes the trained models as a JSON API:

    GET  /health                          — artifact load status
    POST /predict/lap-time                — point prediction + uncertainty interval
    GET  /tyre/degradation/{circuit}/{compound}
                                          — degradation curve arrays (mid/lo/hi)
    POST /simulate/race                   — Monte Carlo race → win/podium probs
    POST /predict/strategy                — rank candidate strategies for one driver

All artefacts are loaded once at startup via the FastAPI lifespan.  If an
artefact file is absent the relevant endpoint returns HTTP 503 so the rest of
the API stays live.

TFT is optional: set ENABLE_TFT=1 to attempt loading the Lightning checkpoint.
pytorch_forecasting is never imported at module top level so the API boots fine
on hosts that only have sklearn/lgbm/numpy.

CPU-only serving — the sim engine's device defaults to "cpu".
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.models.pit_strategy.pit_loss import load_pit_loss
from src.pipeline.features import (
    COMPOUND_ORDER,
    ERA_BOUNDARY,
    FUEL_BURN_PER_LAP,
    FUEL_EFFECT_PER_KG,
    FUEL_LOAD_START_KG,
    MODEL_FEATURE_COLUMNS,
    add_engine_maker,
    engine_for,
    load_encoding_mappings,
)
from src.simulation.engine import (
    DriverSpec,
    RaceSpec,
    SimResult,
    default_sigma,
    recommend,
    simulate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Shared artefact store (populated at startup)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}

# Calibration constants (Z for the 0.1/0.9 quantiles)
_Z80: float = 1.2816


# ---------------------------------------------------------------------------
# Lifespan: load all artefacts once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401
    """Load model artefacts at startup; release nothing (stateless objects)."""
    log.info("Loading model artefacts …")

    # ---- lap-time model (sklearn-API joblib) --------------------------------
    chosen_path = MODELS_DIR / "chosen_lap.joblib"
    if chosen_path.exists():
        try:
            _state["lap_model"] = joblib.load(chosen_path)
            log.info("Loaded chosen_lap.joblib")
        except Exception as exc:
            log.warning("Failed to load chosen_lap.joblib: %s", exc)
    else:
        log.warning("chosen_lap.joblib not found — /predict/lap-time will 503")

    # ---- frozen encodings ---------------------------------------------------
    try:
        circuit_map, team_map, driver_map = load_encoding_mappings()
        _state["circuit_map"] = circuit_map
        _state["team_map"] = team_map
        _state["driver_map"] = driver_map
        log.info(
            "Loaded mappings: %d circuits, %d teams, %d drivers",
            len(circuit_map), len(team_map), len(driver_map),
        )
    except Exception as exc:
        log.warning("Failed to load encoding mappings: %s", exc)

    # ---- tyre curves --------------------------------------------------------
    curves_path = MODELS_DIR / "tyre_curves.joblib"
    if curves_path.exists():
        try:
            _state["tyre_curves"] = joblib.load(curves_path)
            log.info("Loaded tyre_curves.joblib (%d cells)", len(_state["tyre_curves"]))
        except Exception as exc:
            log.warning("Failed to load tyre_curves.joblib: %s", exc)
    else:
        log.warning("tyre_curves.joblib not found — /tyre/degradation will 503")

    # ---- calibration json ---------------------------------------------------
    cal_path = MODELS_DIR / "tft_calibration.json"
    if cal_path.exists():
        try:
            with open(cal_path) as f:
                _state["calibration"] = json.load(f)
            log.info("Loaded tft_calibration.json")
        except Exception as exc:
            log.warning("Failed to load tft_calibration.json: %s", exc)

    # ---- recalibration csv --------------------------------------------------
    recal_path = PROJECT_ROOT / "reports" / "lap_time" / "tft_recalibration.csv"
    if recal_path.exists():
        try:
            _state["recalibration"] = pd.read_csv(recal_path)
            log.info("Loaded tft_recalibration.csv")
        except Exception as exc:
            log.warning("Failed to load tft_recalibration.csv: %s", exc)

    # ---- pit-loss table -----------------------------------------------------
    try:
        _state["pit_loss"] = load_pit_loss()
        log.info("Loaded pit_loss table (%d circuits)", len(_state["pit_loss"]) - 1)
    except Exception as exc:
        log.warning("Failed to load pit_loss table: %s", exc)

    # ---- optional TFT -------------------------------------------------------
    if os.getenv("ENABLE_TFT", "0") == "1":
        _load_tft()

    log.info("Startup complete. Loaded: %s", sorted(_state.keys()))
    yield
    # nothing to tear down


def _load_tft() -> None:
    """Attempt to load the TFT checkpoint (optional, CPU-safe)."""
    ckpt_path = MODELS_DIR / "tft_lap.ckpt"
    if not ckpt_path.exists():
        log.warning("ENABLE_TFT=1 but tft_lap.ckpt not found")
        return
    try:
        # Late import so pytorch_forecasting is never required at module level.
        import torch
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import QuantileLoss

        tft = TemporalFusionTransformer.load_from_checkpoint(
            str(ckpt_path),
            map_location="cpu",
            loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            logging_metrics=torch.nn.ModuleList(),
        )
        tft.eval()
        _state["tft"] = tft
        log.info("TFT loaded from %s", ckpt_path)
    except Exception as exc:
        log.warning("TFT load failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="F1 AI Race Strategist API",
    description=(
        "Serves the trained lap-time, tyre-degradation, and Monte Carlo "
        "race-simulation models for the F1 AI Race Strategist portfolio project."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' when the core artefacts loaded, 'degraded' otherwise")
    artefacts: dict[str, bool] = Field(..., description="Per-artefact load status")
    tft_enabled: bool = Field(..., description="Whether ENABLE_TFT=1 was set at startup")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ok",
                    "artefacts": {
                        "lap_model": True,
                        "tyre_curves": True,
                        "calibration": True,
                        "recalibration": True,
                        "pit_loss": True,
                        "circuit_map": True,
                        "team_map": True,
                        "driver_map": True,
                        "tft": False,
                    },
                    "tft_enabled": False,
                }
            ]
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    """Return API health and which model artefacts successfully loaded."""
    keys = [
        "lap_model", "tyre_curves", "calibration", "recalibration",
        "pit_loss", "circuit_map", "team_map", "driver_map", "tft",
    ]
    artefacts = {k: k in _state for k in keys}
    core_ok = all(artefacts[k] for k in ["lap_model", "tyre_curves", "pit_loss"])
    return HealthResponse(
        status="ok" if core_ok else "degraded",
        artefacts=artefacts,
        tft_enabled=os.getenv("ENABLE_TFT", "0") == "1",
    )


# ---------------------------------------------------------------------------
# POST /predict/lap-time
# ---------------------------------------------------------------------------

class LapTimePredictRequest(BaseModel):
    """All fields known at the start of the lap — no future information."""

    circuit: str = Field(..., description="CircuitKey (FastF1 EventName), e.g. 'Bahrain Grand Prix'")
    team: str = Field(..., description="Constructor name, e.g. 'McLaren'")
    compound: str = Field(..., description="Tyre compound: SOFT | MEDIUM | HARD | INTERMEDIATE | WET")
    tyre_life: int = Field(..., ge=1, description="Laps completed on this set (TyreLife)")
    lap_number: int = Field(..., ge=1, description="Race lap number")
    total_laps: int = Field(..., ge=1, description="Total race laps (for NormLapNumber)")
    season: int = Field(..., ge=2022, description="Season year — determines era and engine supplier")
    track_temp: float = Field(35.0, description="Track surface temperature (°C)")
    air_temp: float = Field(25.0, description="Ambient air temperature (°C)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "circuit": "Bahrain Grand Prix",
                    "team": "McLaren",
                    "compound": "MEDIUM",
                    "tyre_life": 12,
                    "lap_number": 18,
                    "total_laps": 57,
                    "season": 2025,
                    "track_temp": 38.0,
                    "air_temp": 26.0,
                }
            ]
        }
    }


class LapTimePredictResponse(BaseModel):
    predicted_lap_time_s: float = Field(..., description="Point prediction (model median), seconds")
    interval_lo_s: float = Field(
        ...,
        description=(
            "Lower bound of the ±1σ uncertainty interval (seconds). "
            "σ is derived from the recalibrated TFT conformal band width: "
            "σ = (band_width + 2·shift) / (2·z₀.₈).  This is a marginal "
            "interval around the point prediction, not a quantile from the "
            "sklearn model itself."
        ),
    )
    interval_hi_s: float = Field(..., description="Upper bound of the ±1σ uncertainty interval")
    sigma_s: float = Field(..., description="Per-lap pace σ used for the interval")
    circuit_encoded: int = Field(..., description="-1 = unseen circuit (sentinel)")
    team_encoded: int = Field(..., description="-1 = unseen team (sentinel)")
    era: int = Field(..., description="Regulation era: 0=2022-2025, 1=2026+")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_lap_time_s": 92.341,
                    "interval_lo_s": 91.161,
                    "interval_hi_s": 93.521,
                    "sigma_s": 1.18,
                    "circuit_encoded": 2,
                    "team_encoded": 7,
                    "era": 0,
                }
            ]
        }
    }


def _build_lap_row(req: LapTimePredictRequest) -> pd.DataFrame:
    """Map a human-readable request to the MODEL_FEATURE_COLUMNS vector."""
    circuit_map = _state.get("circuit_map", {})
    team_map = _state.get("team_map", {})
    driver_map = _state.get("driver_map", {})

    era = 1 if req.season >= ERA_BOUNDARY else 0
    fuel_load = max(FUEL_LOAD_START_KG - (req.lap_number - 1) * FUEL_BURN_PER_LAP, 0.0)
    tyre_age_sq = req.tyre_life ** 2
    tyre_age_cubed = req.tyre_life ** 3
    compound_encoded = COMPOUND_ORDER.get(req.compound.upper(), -1)
    circuit_encoded = circuit_map.get(req.circuit, -1)
    team_encoded = team_map.get(req.team, -1)
    # Driver unknown at serving time — use -1 sentinel (same as a rookie)
    driver_encoded = -1
    engine_maker = engine_for(req.team, req.season)
    from src.pipeline.features import ENGINE_ORDER
    engine_encoded = ENGINE_ORDER.get(engine_maker, -1) if engine_maker else -1
    norm_lap = req.lap_number / req.total_laps
    tyre_life_val = req.tyre_life
    stint_phase = 0 if tyre_life_val <= 8 else (1 if tyre_life_val <= 15 else 2)
    fuel_effect = fuel_load * FUEL_EFFECT_PER_KG
    compound_x_tyre = compound_encoded * tyre_life_val

    row = {
        "CircuitEncoded": circuit_encoded,
        "TeamEncoded": team_encoded,
        "DriverEncoded": driver_encoded,
        "EngineEncoded": engine_encoded,
        "CompoundEncoded": compound_encoded,
        "Era": era,
        "TyreLife": tyre_life_val,
        "TyreAgeSq": tyre_age_sq,
        "TyreAgeCubed": tyre_age_cubed,
        "CompoundXTyreLife": compound_x_tyre,
        "FuelLoad": fuel_load,
        "FuelEffect": fuel_effect,
        "TrackTemp": req.track_temp,
        "AirTemp": req.air_temp,
        "NormLapNumber": norm_lap,
        "StintPhase": stint_phase,
    }
    return pd.DataFrame([row])[MODEL_FEATURE_COLUMNS]


def _lap_sigma(era: int) -> float:
    """Derive per-lap σ from the recalibrated TFT band (mirrors engine.default_sigma)."""
    try:
        cal = _state["calibration"]
        rep: pd.DataFrame = _state["recalibration"]
        raw = rep[(rep["era"] == era) & (rep["bands"] == "raw")].iloc[0]
        width = float(raw["mean_band_width_s"]) + 2.0 * float(cal[f"era{era}_shift_s"])
        return width / (2.0 * _Z80)
    except Exception:
        return 1.0


@app.post("/predict/lap-time", response_model=LapTimePredictResponse, tags=["Prediction"])
def predict_lap_time(req: LapTimePredictRequest) -> LapTimePredictResponse:
    """Predict lap time with an uncertainty interval.

    The point prediction comes from the chosen sklearn model (LightGBM pipeline
    stored in models/chosen_lap.joblib).  The ±1σ interval is derived from the
    recalibrated TFT conformal band width (models/tft_calibration.json +
    reports/lap_time/tft_recalibration.csv) — the same σ the Monte Carlo engine
    uses per lap.  If calibration artefacts are absent the interval falls back
    to σ=1.0s.
    """
    if "lap_model" not in _state:
        raise HTTPException(status_code=503, detail="Lap-time model artefact not loaded")

    era = 1 if req.season >= ERA_BOUNDARY else 0
    circuit_map = _state.get("circuit_map", {})
    team_map = _state.get("team_map", {})

    X = _build_lap_row(req)
    prediction = float(_state["lap_model"].predict(X)[0])

    sigma = _lap_sigma(era)
    return LapTimePredictResponse(
        predicted_lap_time_s=round(prediction, 4),
        interval_lo_s=round(prediction - sigma, 4),
        interval_hi_s=round(prediction + sigma, 4),
        sigma_s=round(sigma, 4),
        circuit_encoded=int(circuit_map.get(req.circuit, -1)),
        team_encoded=int(team_map.get(req.team, -1)),
        era=era,
    )


# ---------------------------------------------------------------------------
# GET /tyre/degradation/{circuit}/{compound}
# ---------------------------------------------------------------------------

class TyreDegradationResponse(BaseModel):
    circuit: str
    compound: str
    era: int
    age_array: list[int] = Field(..., description="Tyre ages queried (integer laps)")
    mid_s: list[float] = Field(..., description="Median degradation delta relative to fresh tyre (s)")
    lo_s: list[float] = Field(..., description="Lower bound of the 95 % CI band")
    hi_s: list[float] = Field(..., description="Upper bound of the 95 % CI band")
    pool_level: str = Field(
        ...,
        description="Hierarchical pooling level used: 'circuit', 'era', or 'compound'",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "circuit": "Bahrain Grand Prix",
                    "compound": "MEDIUM",
                    "era": 0,
                    "age_array": [1, 5, 10, 15, 20, 25, 30],
                    "mid_s": [0.0, 0.13, 0.45, 0.93, 1.55, 2.30, 3.18],
                    "lo_s": [-0.12, 0.01, 0.29, 0.73, 1.28, 1.94, 2.72],
                    "hi_s": [0.12, 0.25, 0.61, 1.13, 1.82, 2.66, 3.64],
                    "pool_level": "circuit",
                }
            ]
        }
    }


@app.get(
    "/tyre/degradation/{circuit}/{compound}",
    response_model=TyreDegradationResponse,
    tags=["Tyre"],
)
def tyre_degradation(
    circuit: str,
    compound: str,
    era: Annotated[int, Query(ge=0, le=1, description="Era: 0=2022-2025, 1=2026+")] = 0,
    max_age: Annotated[int, Query(ge=1, le=60, description="Maximum tyre age to evaluate")] = 40,
) -> TyreDegradationResponse:
    """Return the fitted tyre-degradation curve (mid/lo/hi) for a compound at a circuit.

    The curve is the fuel-corrected within-stint lap-time delta relative to a
    fresh tyre (age=1), fit via the Phase 3 hierarchical curve model.  Pooling
    falls back compound×era → compound if the circuit cell has too few stints.

    Path parameters are URL-encoded (spaces as %20 or +).
    """
    if "tyre_curves" not in _state:
        raise HTTPException(status_code=503, detail="Tyre-curve artefact not loaded")

    from src.models.tyre.fit import predict_degradation

    compound_upper = compound.upper()
    if compound_upper not in ("SOFT", "MEDIUM", "HARD"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported compound '{compound}'. Use SOFT, MEDIUM, or HARD.",
        )

    age_array = np.arange(1, max_age + 1, dtype=float)
    try:
        mid, lo, hi = predict_degradation(
            _state["tyre_curves"], compound_upper, circuit, era, age_array
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Determine which pool level was used for this cell
    curves: pd.DataFrame = _state["tyre_curves"]
    sel = curves[
        (curves["Compound"] == compound_upper)
        & (curves["CircuitKey"] == circuit)
        & (curves["Era"] == era)
    ]
    if sel.empty:
        sel = curves[(curves["Compound"] == compound_upper) & (curves["Era"] == era)]
    pool_level = str(sel.iloc[0]["pool_level"]) if not sel.empty else "compound"

    return TyreDegradationResponse(
        circuit=circuit,
        compound=compound_upper,
        era=era,
        age_array=[int(a) for a in age_array],
        mid_s=[round(float(v), 4) for v in mid],
        lo_s=[round(float(v), 4) for v in lo],
        hi_s=[round(float(v), 4) for v in hi],
        pool_level=pool_level,
    )


# ---------------------------------------------------------------------------
# POST /simulate/race
# ---------------------------------------------------------------------------

_MAX_ROLLOUTS = 5_000
_DEFAULT_ROLLOUTS = 1_000


class DriverSpecRequest(BaseModel):
    driver: str = Field(..., description="Driver identifier (any string label)")
    base_pace_s: float = Field(..., gt=0.0, description="Green-flag base pace (s) at zero tyre delta and mean fuel")
    start_compound: str = Field("MEDIUM", description="Starting compound: SOFT | MEDIUM | HARD")
    strategy: list[tuple[int, str]] = Field(
        default_factory=list,
        description="Pit stops: list of [pit_lap, compound] pairs, e.g. [[20, 'HARD']]",
    )
    grid_pos: int = Field(1, ge=1, description="Starting grid position (1 = pole)")
    grid_gap_s: float = Field(0.0, description="Cumulative time handicap at the start (s)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "driver": "VER",
                    "base_pace_s": 90.2,
                    "start_compound": "MEDIUM",
                    "strategy": [[22, "HARD"]],
                    "grid_pos": 1,
                    "grid_gap_s": 0.0,
                }
            ]
        }
    }


class RaceSimRequest(BaseModel):
    circuit: str = Field(..., description="CircuitKey (FastF1 EventName)")
    era: int = Field(..., ge=0, le=1, description="Regulation era: 0=2022-2025, 1=2026+")
    n_laps: int = Field(..., ge=10, le=78, description="Total race distance in laps")
    drivers: list[DriverSpecRequest] = Field(..., min_length=2, description="All drivers in the race")
    n_rollouts: int = Field(
        _DEFAULT_ROLLOUTS,
        ge=100,
        le=_MAX_ROLLOUTS,
        description=f"Monte Carlo rollouts (capped at {_MAX_ROLLOUTS})",
    )
    seed: int | None = Field(42, description="RNG seed for reproducibility (null = random)")
    pit_loss_s: float | None = Field(None, description="Override per-circuit pit loss (s)")
    sigma_lap_s: float | None = Field(None, description="Override per-lap pace σ (s)")
    overtake_prob: float | None = Field(
        None, ge=0.0, le=1.0, description="Override per-lap overtake probability"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "circuit": "Bahrain Grand Prix",
                    "era": 0,
                    "n_laps": 57,
                    "n_rollouts": 500,
                    "seed": 42,
                    "drivers": [
                        {
                            "driver": "VER",
                            "base_pace_s": 90.1,
                            "start_compound": "MEDIUM",
                            "strategy": [[22, "HARD"]],
                            "grid_pos": 1,
                            "grid_gap_s": 0.0,
                        },
                        {
                            "driver": "NOR",
                            "base_pace_s": 90.3,
                            "start_compound": "MEDIUM",
                            "strategy": [[20, "HARD"]],
                            "grid_pos": 2,
                            "grid_gap_s": 0.3,
                        },
                        {
                            "driver": "LEC",
                            "base_pace_s": 90.5,
                            "start_compound": "SOFT",
                            "strategy": [[15, "HARD"]],
                            "grid_pos": 3,
                            "grid_gap_s": 0.6,
                        },
                    ],
                }
            ]
        }
    }


class DriverSimResult(BaseModel):
    driver: str
    win_prob: float = Field(..., description="Fraction of rollouts won (0–1)")
    podium_prob: float = Field(..., description="Fraction of rollouts in top 3")
    mean_finish: float = Field(..., description="Mean finishing position across rollouts")
    p5_finish: float = Field(..., description="5th-percentile finishing position (best tail)")
    p95_finish: float = Field(..., description="95th-percentile finishing position (worst tail)")


class RaceSimResponse(BaseModel):
    circuit: str
    era: int
    n_laps: int
    n_rollouts: int
    results: list[DriverSimResult] = Field(
        ..., description="Per-driver probabilities, sorted by mean finish position"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "circuit": "Bahrain Grand Prix",
                    "era": 0,
                    "n_laps": 57,
                    "n_rollouts": 500,
                    "results": [
                        {
                            "driver": "VER",
                            "win_prob": 0.552,
                            "podium_prob": 0.841,
                            "mean_finish": 1.72,
                            "p5_finish": 1.0,
                            "p95_finish": 3.0,
                        }
                    ],
                }
            ]
        }
    }


@app.post("/simulate/race", response_model=RaceSimResponse, tags=["Simulation"])
def simulate_race(req: RaceSimRequest) -> RaceSimResponse:
    """Run a Monte Carlo race simulation and return per-driver win / podium probabilities.

    The engine samples per-lap pace noise from the recalibrated TFT band,
    tyre degradation from the Phase 3 CI curves, pit-stop execution noise, and
    circuit-class overtaking probability.  n_rollouts is capped at 5,000 for
    latency.  Compound names are case-insensitive.
    """
    if "tyre_curves" not in _state:
        raise HTTPException(status_code=503, detail="Tyre-curve artefact not loaded")
    if "pit_loss" not in _state and req.pit_loss_s is None:
        raise HTTPException(status_code=503, detail="Pit-loss artefact not loaded and no override provided")

    n_rollouts = min(req.n_rollouts, _MAX_ROLLOUTS)

    driver_specs = [
        DriverSpec(
            driver=d.driver,
            base_pace_s=d.base_pace_s,
            start_compound=d.start_compound.upper(),
            strategy=[(int(lap), comp.upper()) for lap, comp in d.strategy],
            grid_pos=d.grid_pos,
            grid_gap_s=d.grid_gap_s,
        )
        for d in req.drivers
    ]

    spec = RaceSpec(
        circuit=req.circuit,
        era=req.era,
        n_laps=req.n_laps,
        drivers=driver_specs,
        pit_loss_s=req.pit_loss_s,
        sigma_lap_s=req.sigma_lap_s,
        overtake_prob=req.overtake_prob,
    )

    result: SimResult = simulate(
        spec,
        n_rollouts=n_rollouts,
        seed=req.seed,
        device="cpu",
        curves=_state.get("tyre_curves"),
    )

    table = result.table()
    results = [
        DriverSimResult(
            driver=str(row["driver"]),
            win_prob=round(float(row["win_prob"]), 4),
            podium_prob=round(float(row["podium_prob"]), 4),
            mean_finish=round(float(row["mean_finish"]), 3),
            p5_finish=round(float(row["p5_finish"]), 1),
            p95_finish=round(float(row["p95_finish"]), 1),
        )
        for _, row in table.iterrows()
    ]
    return RaceSimResponse(
        circuit=req.circuit,
        era=req.era,
        n_laps=req.n_laps,
        n_rollouts=n_rollouts,
        results=results,
    )


# ---------------------------------------------------------------------------
# POST /predict/strategy
# ---------------------------------------------------------------------------

class PitStop(BaseModel):
    pit_lap: int = Field(..., ge=1, description="Lap at the END of which the driver pits")
    compound: str = Field(..., description="Compound fitted on exit: SOFT | MEDIUM | HARD")


class CandidateStrategy(BaseModel):
    name: str = Field(..., description="Human-readable strategy label, e.g. '2-stop M-H-H'")
    start_compound: str = Field("MEDIUM", description="Starting compound: SOFT | MEDIUM | HARD")
    stops: list[PitStop] = Field(default_factory=list, description="Ordered list of pit stops")


class StrategyPredictRequest(BaseModel):
    circuit: str = Field(..., description="CircuitKey (FastF1 EventName)")
    era: int = Field(..., ge=0, le=1, description="Regulation era: 0=2022-2025, 1=2026+")
    n_laps: int = Field(..., ge=10, le=78)
    ego_driver: DriverSpecRequest = Field(..., description="The driver whose strategy to rank")
    rivals: list[DriverSpecRequest] = Field(
        default_factory=list,
        description="Rival drivers on their fixed strategies (used as backdrop for the simulation)",
    )
    candidate_strategies: list[CandidateStrategy] = Field(
        ..., min_length=1, description="Strategies to evaluate and rank for the ego driver"
    )
    n_rollouts: int = Field(_DEFAULT_ROLLOUTS, ge=100, le=_MAX_ROLLOUTS)
    seed: int | None = Field(42)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "circuit": "Bahrain Grand Prix",
                    "era": 0,
                    "n_laps": 57,
                    "ego_driver": {
                        "driver": "VER",
                        "base_pace_s": 90.1,
                        "start_compound": "MEDIUM",
                        "strategy": [[22, "HARD"]],
                        "grid_pos": 1,
                        "grid_gap_s": 0.0,
                    },
                    "rivals": [
                        {
                            "driver": "NOR",
                            "base_pace_s": 90.3,
                            "start_compound": "MEDIUM",
                            "strategy": [[20, "HARD"]],
                            "grid_pos": 2,
                            "grid_gap_s": 0.3,
                        }
                    ],
                    "candidate_strategies": [
                        {
                            "name": "1-stop M->H (L22)",
                            "start_compound": "MEDIUM",
                            "stops": [{"pit_lap": 22, "compound": "HARD"}],
                        },
                        {
                            "name": "2-stop M->H->H",
                            "start_compound": "MEDIUM",
                            "stops": [
                                {"pit_lap": 15, "compound": "HARD"},
                                {"pit_lap": 35, "compound": "HARD"},
                            ],
                        },
                    ],
                    "n_rollouts": 500,
                    "seed": 42,
                }
            ]
        }
    }


class StrategyRanking(BaseModel):
    strategy: str
    start_compound: str
    stops: str = Field(..., description="String representation of stop list, e.g. '[(22, HARD)]'")
    win_prob: float
    podium_prob: float
    mean_finish: float
    p50_time_s: float = Field(..., description="Median total race time across rollouts (s)")


class StrategyPredictResponse(BaseModel):
    circuit: str
    ego_driver: str
    n_rollouts: int
    rankings: list[StrategyRanking] = Field(
        ..., description="Strategies ranked by mean finish position (best first)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "circuit": "Bahrain Grand Prix",
                    "ego_driver": "VER",
                    "n_rollouts": 500,
                    "rankings": [
                        {
                            "strategy": "1-stop M->H (L22)",
                            "start_compound": "MEDIUM",
                            "stops": "[(22, 'HARD')]",
                            "win_prob": 0.612,
                            "podium_prob": 0.882,
                            "mean_finish": 1.55,
                            "p50_time_s": 5423.7,
                        }
                    ],
                }
            ]
        }
    }


@app.post("/predict/strategy", response_model=StrategyPredictResponse, tags=["Simulation"])
def predict_strategy(req: StrategyPredictRequest) -> StrategyPredictResponse:
    """Rank candidate strategies for one ego driver via paired Monte Carlo rollouts.

    Rivals keep their fixed strategies throughout.  The ego driver's strategy is
    swapped per candidate while the same RNG seed is reused across candidates
    (common random numbers) — differences in the rankings are due to strategy,
    not random variation.  The response is sorted by mean finish position
    (ascending = better).
    """
    if "tyre_curves" not in _state:
        raise HTTPException(status_code=503, detail="Tyre-curve artefact not loaded")
    if "pit_loss" not in _state:
        raise HTTPException(status_code=503, detail="Pit-loss artefact not loaded")

    n_rollouts = min(req.n_rollouts, _MAX_ROLLOUTS)

    def _to_driver_spec(d: DriverSpecRequest) -> DriverSpec:
        return DriverSpec(
            driver=d.driver,
            base_pace_s=d.base_pace_s,
            start_compound=d.start_compound.upper(),
            strategy=[(int(lap), comp.upper()) for lap, comp in d.strategy],
            grid_pos=d.grid_pos,
            grid_gap_s=d.grid_gap_s,
        )

    ego_spec = _to_driver_spec(req.ego_driver)
    rival_specs = [_to_driver_spec(r) for r in req.rivals]
    all_drivers = [ego_spec] + rival_specs

    spec = RaceSpec(
        circuit=req.circuit,
        era=req.era,
        n_laps=req.n_laps,
        drivers=all_drivers,
    )

    candidate_strategies: dict[str, tuple[str, list[tuple[int, str]]]] = {
        c.name: (
            c.start_compound.upper(),
            [(s.pit_lap, s.compound.upper()) for s in c.stops],
        )
        for c in req.candidate_strategies
    }

    df = recommend(
        spec,
        ego_driver=req.ego_driver.driver,
        candidate_strategies=candidate_strategies,
        n_rollouts=n_rollouts,
        seed=req.seed,
    )

    rankings = [
        StrategyRanking(
            strategy=str(row["strategy"]),
            start_compound=str(row["start"]),
            stops=str(row["stops"]),
            win_prob=round(float(row["win_prob"]), 4),
            podium_prob=round(float(row["podium_prob"]), 4),
            mean_finish=round(float(row["mean_finish"]), 3),
            p50_time_s=round(float(row["p50_time_s"]), 2),
        )
        for _, row in df.iterrows()
    ]
    return StrategyPredictResponse(
        circuit=req.circuit,
        ego_driver=req.ego_driver.driver,
        n_rollouts=n_rollouts,
        rankings=rankings,
    )
