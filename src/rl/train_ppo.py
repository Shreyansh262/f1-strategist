"""PPO trainer for the RL pit agent — Phase 5 (STRETCH).

Trains a PPO (MlpPolicy) agent inside the RaceEnv (src/rl/env.py). PPO with an MLP
policy is mostly CPU-bound; the sensible use of Kaggle's 2x T4 is parallel envs
(SubprocVecEnv) and running the seeds across sessions/processes, not a big net on
the GPU. The --device flag is honoured but defaults to cpu.

Artifacts (all under models/rl/):
    ppo_pit_seed{S}.zip          final policy
    ppo_pit_seed{S}_{steps}.zip  checkpoints every --checkpoint-freq steps
    tb/                          tensorboard logs (episode reward etc.)

Run a short local SMOKE (NOT training — the real runs happen on Kaggle):
    python -m src.rl.train_ppo --steps 10000 --n-envs 2 --seed 0

Full run (Kaggle, per seed, ~3h each on CPU cores):
    python -m src.rl.train_ppo --steps 3000000 --n-envs 8 --seed 0
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

# mlflow is not used here (sb3 logs to tensorboard); keep the file-store guard
# consistent with the rest of the project in case mlflow gets imported transitively.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "rl"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def make_env_fn(seed: int, rank: int):
    """Factory for a seeded RaceEnv (used by the vectorized env)."""
    from src.rl.env import RaceEnv

    def _init():
        env = RaceEnv(seed=seed + rank)
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(n_envs: int, seed: int):
    """SubprocVecEnv for n_envs>1, DummyVecEnv otherwise (cheaper for the smoke test)."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = [make_env_fn(seed, r) for r in range(n_envs)]
    if n_envs > 1:
        return SubprocVecEnv(fns)
    return DummyVecEnv(fns)


def train(
    steps: int,
    seed: int = 0,
    n_envs: int = 4,
    device: str = "cpu",
    checkpoint_freq: int = 500_000,
    learning_rate: float = 3e-4,
) -> Path:
    """Train PPO and return the path to the saved final policy."""
    import importlib.util

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    vec = build_vec_env(n_envs, seed)

    # Tensorboard logging if available (it is on Kaggle); skip cleanly otherwise so
    # the local CPU smoke test runs without the extra dependency.
    tb_log = None
    if importlib.util.find_spec("tensorboard") is not None:
        tb_dir = MODELS_DIR / "tb"
        tb_dir.mkdir(exist_ok=True)
        tb_log = str(tb_dir)
    else:
        log.warning("tensorboard not installed — training without tensorboard logs")

    model = PPO(
        "MlpPolicy",
        vec,
        seed=seed,
        device=device,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=256,
        gamma=0.999,            # long-horizon race: minimal discounting
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=tb_log,
    )

    # CheckpointCallback counts in *steps per env*, so divide to honour the global freq.
    ckpt_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=str(MODELS_DIR),
        name_prefix=f"ppo_pit_seed{seed}",
    )
    log.info(
        "PPO training: steps=%d seed=%d n_envs=%d device=%s ckpt_freq=%d",
        steps, seed, n_envs, device, checkpoint_freq,
    )
    model.learn(total_timesteps=steps, callback=ckpt_cb, progress_bar=False,
                tb_log_name=f"seed{seed}")

    out = MODELS_DIR / f"ppo_pit_seed{seed}.zip"
    model.save(out)
    log.info("Saved final policy -> %s", out)
    vec.close()
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO trainer for the RL pit agent.")
    p.add_argument("--steps", type=int, default=3_000_000,
                   help="total timesteps (use ~5k-10k for a local smoke test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--checkpoint-freq", type=int, default=500_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train(
        steps=args.steps,
        seed=args.seed,
        n_envs=args.n_envs,
        device=args.device,
        checkpoint_freq=args.checkpoint_freq,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
