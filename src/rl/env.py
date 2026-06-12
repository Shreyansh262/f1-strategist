"""Gymnasium environment wrapping the F1 race simulation — Phase 5 (RL stretch).

The Phase 4b engine (`src/simulation/engine.py`) simulates whole races vectorized
across rollouts. RL needs the opposite: a single race stepped lap-by-lap so the
agent can decide, at each lap, whether to pit and onto which compound. This module
implements that light per-lap transition by REUSING the engine's components rather
than calling `simulate()`:

    base pace + fuel delta     engine constants (FUEL_*), engine._fuel_delta logic
    tyre degradation           src.models.tyre.fit.predict_degradation (mid + CI band)
    per-lap pace noise          σ from engine.default_sigma (recalibrated TFT band)
    pit loss                    src.models.pit_strategy.pit_loss.pit_loss_for
    overtaking / dirty air      engine's STRIKE_GAP_S / HOLD_GAP_S / OVERTAKE_PROB rules

Nothing here hard-codes a magic number that already lives in engine.py — they are
imported, so the env and the showcase engine cannot drift apart.

State / action / reward (Section 10.5, mirroring the MDP in mdp.py)
-------------------------------------------------------------------
Observation (Box, float32) = the MDP state vector + recent lap-time deltas:
    [ norm_lap, norm_tyre_age, compound_onehot(3), norm_position,
      gap_ahead_s_clipped, norm_laps_remaining, era,
      last3 lap-time deltas vs the ego's own running median ]
Actions (Discrete(4), exactly the MDP action set ACTIONS):
    0 stay_out | 1 pit_SOFT | 2 pit_MEDIUM | 3 pit_HARD
Reward: -race_time_for_this_lap / 1000 each step (so the return ≈ -total_time/1000),
    plus a terminal position bonus (+ for places gained vs grid). Episode = one race.

The two-compound rule is enforced the way the MDP does it: a terminal penalty if the
ego never ran a second slick compound (an illegal classification).

CPU-friendly, single rollout. `gymnasium.utils.env_checker.check_env(RaceEnv())`
passes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import joblib
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.models.pit_strategy.mdp import ACTIONS, OUT_LAP_AGE
from src.models.pit_strategy.pit_loss import load_pit_loss, pit_loss_for
from src.models.tyre.fit import predict_degradation
from src.simulation.engine import (
    COMPOUNDS,
    FUEL_BURN_PER_LAP,
    FUEL_EFFECT_PER_KG,
    FUEL_LOAD_START_KG,
    HOLD_GAP_S,
    OVERTAKE_PROB,
    PIT_SIGMA_S,
    STRIKE_GAP_S,
    circuit_class,
    default_sigma,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

MAX_AGE: int = 45          # tyre-age cap, mirrors mdp.MAX_AGE
GAP_CLIP_S: float = 30.0   # clip gap_ahead for the observation
N_RECENT: int = 3          # recent lap-time deltas in the obs


# ---------------------------------------------------------------------------
# Rival field
# ---------------------------------------------------------------------------

@dataclass
class RivalSpec:
    """A fixed-strategy rival. Mirrors engine.DriverSpec minus the ego control."""
    name: str
    base_pace_s: float
    start_compound: str
    strategy: list[tuple[int, str]]      # (pit at END of lap, compound)
    grid_gap_s: float


@dataclass
class EgoSpec:
    """The agent's car. Strategy is decided online by the policy, not fixed."""
    name: str = "EGO"
    base_pace_s: float = 90.3
    start_compound: str = "MEDIUM"
    grid_gap_s: float = 0.9


@dataclass
class RaceConfig:
    """One race the env can roll out. A realistic default field is built below."""
    circuit: str = "Bahrain Grand Prix"
    era: int = 0
    n_laps: int = 57
    ego: EgoSpec = field(default_factory=EgoSpec)
    rivals: list[RivalSpec] = field(default_factory=list)
    pit_loss_s: float | None = None
    sigma_lap_s: float | None = None
    overtake_prob: float | None = None


def default_field(ego_grid: int = 3) -> RaceConfig:
    """A realistic 6-car Bahrain field (same shape as engine._demo), ego mid-grid.

    Rivals run plausible 1-/2-stop strategies; pace spread ~1.5s front-to-back.
    The ego starts on MEDIUM from ~P3 — enough traffic to make pit timing matter.
    """
    rivals = [
        RivalSpec("FAST1", 90.0, "MEDIUM", [(20, "HARD")], 0.0),
        RivalSpec("FAST2", 90.1, "MEDIUM", [(22, "HARD")], 0.3),
        RivalSpec("MID2", 90.6, "SOFT", [(15, "HARD")], 1.2),
        RivalSpec("SLOW1", 91.2, "MEDIUM", [(25, "HARD")], 1.5),
        RivalSpec("SLOW2", 91.4, "HARD", [(30, "MEDIUM")], 1.8),
    ]
    ego = EgoSpec(name="EGO", base_pace_s=90.3, start_compound="MEDIUM",
                  grid_gap_s=0.3 * (ego_grid - 1))
    return RaceConfig(circuit="Bahrain Grand Prix", era=0, n_laps=57,
                      ego=ego, rivals=rivals)


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------

class RaceEnv(gym.Env):
    """Lap-by-lap single-ego race env. Standard Gymnasium API.

    Parameters
    ----------
    config : RaceConfig | None
        The race to simulate. None -> default_field().
    curves : pd.DataFrame | None
        Tyre curves. None -> load models/tyre_curves.joblib once. Pass a synthetic
        frame in tests to avoid the artifact dependency.
    seed : int | None
        Convenience seed; reset(seed=) overrides it per episode.
    """

    metadata = {"render_modes": []}

    # observation layout (kept as one place so train/eval agree):
    #   norm_lap(1) norm_age(1) compound_onehot(3) norm_pos(1) gap_ahead(1)
    #   norm_laps_remaining(1) era(1) recent_deltas(N_RECENT)
    _OBS_DIM = 1 + 1 + 3 + 1 + 1 + 1 + 1 + N_RECENT

    def __init__(
        self,
        config: RaceConfig | None = None,
        curves: pd.DataFrame | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else default_field()
        if curves is None:
            curves = joblib.load(MODELS_DIR / "tyre_curves.joblib")
        self.curves = curves

        self._n_rivals = len(self.config.rivals)
        self._n_cars = self._n_rivals + 1
        self._ego_idx = 0                      # ego is index 0 in the car arrays

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._OBS_DIM,), dtype=np.float32
        )

        self._seed = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # resolved per-race scalars (set in reset)
        self._pit_loss = 0.0
        self._sigma = 1.0
        self._p_pass = 0.0
        self._fuel = np.zeros(self.config.n_laps + 1)

        # precompute deterministic rival lap schedules + degradation
        self._rival_base = np.array([r.base_pace_s for r in self.config.rivals])
        self._rival_grid_gap = np.array([r.grid_gap_s for r in self.config.rivals])
        self._rival_deg = self._rival_degradation()   # [n_rivals, n_laps] mid only

        self.reset(seed=seed)

    # -- setup helpers -------------------------------------------------------

    def _fuel_delta(self) -> np.ndarray:
        """Fuel lap-time delta per lap (1-indexed at [1..n_laps]); reuses engine constants."""
        laps = np.arange(1, self.config.n_laps + 1)
        load = np.maximum(FUEL_LOAD_START_KG - (laps - 1) * FUEL_BURN_PER_LAP, 0.0)
        d = (load - load.mean()) * FUEL_EFFECT_PER_KG
        return np.concatenate([[0.0], d])          # pad index 0

    def _rival_schedule(self, rival: RivalSpec) -> tuple[np.ndarray, np.ndarray]:
        """(compound_idx, tyre_age) per lap for a fixed-strategy rival."""
        L = self.config.n_laps
        comp = np.empty(L + 1, dtype=int)
        age = np.empty(L + 1, dtype=int)
        pits = dict(rival.strategy)
        c, a = COMPOUNDS.index(rival.start_compound), OUT_LAP_AGE
        for lap in range(1, L + 1):
            comp[lap], age[lap] = c, a
            if lap in pits:
                c, a = COMPOUNDS.index(pits[lap]), OUT_LAP_AGE
            else:
                a += 1
        return comp, age

    def _rival_degradation(self) -> np.ndarray:
        """Deterministic mid degradation delta per (rival, lap)."""
        L = self.config.n_laps
        out = np.zeros((self._n_rivals, L + 1))
        for i, r in enumerate(self.config.rivals):
            comp, age = self._rival_schedule(r)
            for ci in range(len(COMPOUNDS)):
                m = comp == ci
                if m.any():
                    mid, _, _ = predict_degradation(
                        self.curves, COMPOUNDS[ci], self.config.circuit,
                        self.config.era, age[m].astype(float),
                    )
                    out[i, m] = mid
        return out

    def _rival_pit_mask(self) -> np.ndarray:
        """[n_rivals, n_laps+1] bool: True on the lap a rival pits (end of lap)."""
        L = self.config.n_laps
        mask = np.zeros((self._n_rivals, L + 1), dtype=bool)
        for i, r in enumerate(self.config.rivals):
            for pit_lap, _ in r.strategy:
                if 1 <= pit_lap <= L:
                    mask[i, pit_lap] = True
        return mask

    # -- Gymnasium API -------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        cfg = self.config
        self._pit_loss = (
            cfg.pit_loss_s if cfg.pit_loss_s is not None
            else pit_loss_for(load_pit_loss(), cfg.circuit)
        )
        self._sigma = (
            cfg.sigma_lap_s if cfg.sigma_lap_s is not None
            else default_sigma(cfg.circuit, cfg.era)
        )
        self._p_pass = (
            cfg.overtake_prob if cfg.overtake_prob is not None
            else OVERTAKE_PROB[circuit_class(cfg.circuit)]
        )
        self._fuel = self._fuel_delta()
        self._rival_pits = self._rival_pit_mask()

        # cumulative race time per car; index 0 = ego.
        self._cum = np.concatenate([[cfg.ego.grid_gap_s], self._rival_grid_gap]).astype(float)
        self._grid_order = np.argsort(self._cum).argsort()  # 0-based grid rank per car
        self._ego_grid_pos = int(self._grid_order[self._ego_idx]) + 1

        # ego tyre state
        self._lap = 1
        self._ego_comp = COMPOUNDS.index(cfg.ego.start_compound)
        self._ego_age = OUT_LAP_AGE
        self._ego_stint = 0
        self._used_compounds = {self._ego_comp}
        self._ego_lap_times: list[float] = []
        self._terminated = False

        return self._observation(), {}

    def step(self, action: int):
        if self._terminated:
            raise RuntimeError("step() called on a terminated episode; call reset().")
        cfg = self.config
        L = cfg.n_laps
        lap = self._lap
        action = int(action)

        # --- ego lap time for THIS lap (pit applied at end of lap) ----------
        pit_this_lap = action != 0
        ego_mid, ego_lo, ego_hi = predict_degradation(
            self.curves, COMPOUNDS[self._ego_comp], cfg.circuit, cfg.era,
            float(self._ego_age),
        )
        deg_sigma = max((float(ego_hi) - float(ego_lo)) / (2 * 1.96), 0.0)
        ego_noise = self._rng.normal(0.0, self._sigma)
        deg_draw = self._rng.normal(0.0, deg_sigma)
        ego_lap_time = (
            cfg.ego.base_pace_s + self._fuel[lap]
            + float(ego_mid) + deg_draw + ego_noise
        )
        if pit_this_lap:
            ego_lap_time += self._pit_loss + self._rng.normal(0.0, PIT_SIGMA_S)

        # --- rival lap times for THIS lap -----------------------------------
        rival_noise = self._rng.normal(0.0, self._sigma, size=self._n_rivals)
        rival_lap = (
            self._rival_base + self._fuel[lap] + self._rival_deg[:, lap] + rival_noise
        )
        rival_lap = rival_lap + self._rival_pits[:, lap] * (
            self._pit_loss + self._rng.normal(0.0, PIT_SIGMA_S, size=self._n_rivals)
        )

        lap_times = np.concatenate([[ego_lap_time], rival_lap])
        proposed = self._cum + lap_times
        self._cum = self._resolve_overtaking(self._cum, proposed, lap)
        self._ego_lap_times.append(ego_lap_time)

        # --- advance ego tyre state -----------------------------------------
        if pit_this_lap:
            self._ego_comp = action - 1            # ACTIONS[1:]=pit_SOFT/MED/HARD
            self._ego_age = OUT_LAP_AGE
            self._ego_stint += 1
            self._used_compounds.add(self._ego_comp)
        else:
            self._ego_age = min(self._ego_age + 1, MAX_AGE)

        # --- reward ----------------------------------------------------------
        reward = -ego_lap_time / 1000.0

        self._lap += 1
        terminated = self._lap > L
        truncated = False

        if terminated:
            self._terminated = True
            reward += self._terminal_bonus()

        obs = self._observation()
        info = {
            "lap": lap,
            "ego_lap_time": ego_lap_time,
            "ego_position": self._ego_position(),
            "pitted": pit_this_lap,
        }
        if terminated:
            info["race_time_s"] = float(self._cum[self._ego_idx])
            info["finish_position"] = self._ego_position()
            info["used_two_compounds"] = len(self._used_compounds) >= 2
        return obs, float(reward), terminated, truncated, info

    # -- mechanics -----------------------------------------------------------

    def _resolve_overtaking(self, cum: np.ndarray, proposed: np.ndarray, lap: int) -> np.ndarray:
        """Single-lap dirty-air model mirroring engine.simulate's pass logic.

        Walk the field front-to-back by start-of-lap order; a car that would jump
        the car ahead needs to be within STRIKE_GAP_S AND win a Bernoulli(p_pass),
        else it is held HOLD_GAP_S behind. Same constants as the showcase engine.
        """
        order = np.argsort(cum)
        resolved = proposed.copy()
        min_ahead = -1e9
        for idx in order:
            t = resolved[idx]
            would_pass = t < min_ahead
            close = abs(t - min_ahead) < STRIKE_GAP_S
            allowed = self._rng.random() < self._p_pass
            if would_pass and not (close and allowed):
                t = min_ahead + HOLD_GAP_S
            resolved[idx] = t
            min_ahead = max(min_ahead, t)
        return resolved

    def _ego_position(self) -> int:
        """1-based track position of the ego from cumulative time."""
        return int(np.argsort(np.argsort(self._cum))[self._ego_idx]) + 1

    def _gap_ahead(self) -> float:
        """Seconds to the car directly ahead of the ego (0 if leading)."""
        order = np.argsort(self._cum)
        pos = int(np.where(order == self._ego_idx)[0][0])
        if pos == 0:
            return 0.0
        ahead = order[pos - 1]
        return float(self._cum[self._ego_idx] - self._cum[ahead])

    def _terminal_bonus(self) -> float:
        """Position-gain bonus + two-compound-rule penalty (mirrors the MDP rule)."""
        finish = self._ego_position()
        gained = self._ego_grid_pos - finish      # positive = moved up
        bonus = 0.1 * gained
        if len(self._used_compounds) < 2:
            bonus -= 5.0                           # illegal race (DSQ-like penalty)
        return bonus

    def _observation(self) -> np.ndarray:
        cfg = self.config
        L = cfg.n_laps
        lap = min(self._lap, L)
        comp_onehot = np.zeros(3, dtype=np.float32)
        comp_onehot[self._ego_comp] = 1.0

        recent = self._ego_lap_times[-N_RECENT:]
        if recent:
            med = float(np.median(self._ego_lap_times))
            deltas = [t - med for t in recent]
        else:
            deltas = []
        deltas = ([0.0] * (N_RECENT - len(deltas))) + deltas   # left-pad

        obs = np.array(
            [
                lap / L,                                   # norm_lap
                self._ego_age / MAX_AGE,                   # norm_tyre_age
                *comp_onehot,                              # compound one-hot
                self._ego_position() / self._n_cars,       # norm_position
                np.clip(self._gap_ahead(), 0.0, GAP_CLIP_S) / GAP_CLIP_S,
                (L - lap) / L,                             # norm_laps_remaining
                float(cfg.era),                            # era
                *deltas,                                   # recent lap-time deltas
            ],
            dtype=np.float32,
        )
        return obs

    # -- expose state for the MDP-policy baseline ---------------------------

    def mdp_state(self) -> tuple[int, int, int, int]:
        """(lap, compound_idx, tyre_age, used_two_compounds) — the MDP's state tuple."""
        return (
            min(self._lap, self.config.n_laps),
            self._ego_comp,
            min(self._ego_age, MAX_AGE),
            int(len(self._used_compounds) >= 2),
        )
