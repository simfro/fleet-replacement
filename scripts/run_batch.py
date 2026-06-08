"""Run many episodes of FleetReplacementEnv using a chosen agent policy
and record each episode to disk.

Results are written to an auto-named sub-directory of ``results/``:

    results/batch_<config-stem>_<agent>_<YYYYMMDD_HHMMSS>/

A copy of the config YAML is placed in the output directory so the
exact parameters used are transparent.  Each episode is saved as a
pickle file ``episode_<seed:04d>.pkl`` that can be loaded later with
:meth:`EpisodeRecord.load`.

Usage
-----
    python scripts/run_batch.py
    python scripts/run_batch.py --config configs/env.yaml --n-episodes 50 --seed-start 0
    python scripts/run_batch.py --agent myopic --config configs/baseline.yaml --n-episodes 100
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

# Linopy raises a harmless UserWarning when building models with variables
# that have different coordinate sets ("Perform outer join").  Suppress it
# for batch runs where thousands of models are built and the noise is costly.
warnings.filterwarnings("ignore", category=UserWarning, module="linopy")

# Allow running directly from the repository root without a package install.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fleet_replacement.config import load_env_config  # noqa: E402
from fleet_replacement.envs.fleet_replacement import FleetReplacementEnv  # noqa: E402
from fleet_replacement.episode import EpisodeRecord, EpisodeRecorder  # noqa: E402
from fleet_replacement.policies import LookaheadAgent, MyopicAgent  # noqa: E402

# ---------------------------------------------------------------------------
# Episode runner (silent)
# ---------------------------------------------------------------------------


def _run_episode_silent(
    config,
    seed: int,
    agent_name: str = "lookahead",
) -> EpisodeRecord:
    """Run one episode without any console output and return the record."""
    env = EpisodeRecorder(FleetReplacementEnv(config=config))

    if agent_name == "myopic":
        agent: LookaheadAgent | MyopicAgent = MyopicAgent(env)
    else:
        agent = LookaheadAgent(config)

    obs, _ = env.reset(seed=seed)

    while True:
        if agent_name == "lookahead":
            try:
                action = agent.select_action(obs)
            except RuntimeError:
                action = np.zeros(len(obs["fleet"]["is_electric"]), dtype=np.int64)
        else:
            action = agent.select_action(obs)

        obs, _reward, terminated, truncated, _info = env.step(action)

        if terminated or truncated:
            break

    return env.episode_record


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _print_summary(rewards: list[float], output_dir: pathlib.Path) -> None:
    rewards_arr = np.array(rewards)
    width = 42
    print(f"\n{'─' * width}")
    print(f"  Batch summary  ({len(rewards)} episodes)")
    print(f"{'─' * width}")
    print(f"  Min    : {rewards_arr.min():>16,.2f}")
    print(f"  Mean   : {rewards_arr.mean():>16,.2f}")
    print(f"  Std    : {rewards_arr.std():>16,.2f}")
    print(f"  Max    : {rewards_arr.max():>16,.2f}")
    print(f"{'─' * width}")
    print(f"  Output : {output_dir}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run FleetReplacementEnv for many episodes using the LookaheadAgent "
            "and save each EpisodeRecord as a pickle."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "env.yaml"),
        help="Path to the environment YAML config file.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=20,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First random seed.  Episodes use seeds [seed-start, seed-start + n-episodes).",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="lookahead",
        choices=["lookahead", "myopic"],
        help="Agent policy to use for all episodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = pathlib.Path(args.config).resolve()
    config = load_env_config(str(config_path))

    # Auto-generate output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / "results" / f"batch_{config_path.stem}_{args.agent}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config into the output folder for transparency
    shutil.copy(config_path, output_dir / config_path.name)

    print(f"Config     : {config_path}")
    print(f"Agent      : {args.agent}")
    print(
        f"Episodes   : {args.n_episodes}  (seeds {args.seed_start} – {args.seed_start + args.n_episodes - 1})"
    )
    if args.agent == "lookahead":
        print(f"Horizon    : {config.lookahead.horizon} periods")
    print(f"Fleet size : {config.vehicle_management.fleet_size} vehicles")
    print(
        f"Episode    : {config.simulation_period.base_year}"
        f" -> {config.simulation_period.final_year}"
        f"  ({config.simulation_period.final_year - config.simulation_period.base_year} steps)"
    )
    print(f"Output     : {output_dir}\n")

    rewards: list[float] = []
    failed_seeds: list[int] = []

    seeds = range(args.seed_start, args.seed_start + args.n_episodes)

    with tqdm(total=args.n_episodes, unit="ep", dynamic_ncols=True) as bar:
        for seed in seeds:
            try:
                record = _run_episode_silent(config, seed, agent_name=args.agent)
            except Exception as exc:
                tqdm.write(f"[WARN]  seed={seed}  failed: {exc}")
                failed_seeds.append(seed)
                bar.update(1)
                continue

            record.save(str(output_dir / f"episode_{seed:04d}.pkl"))

            total_r = record.total_reward
            n_elec = int(record.n_electric[-1])
            fleet_sz = record.fleet_size
            rewards.append(total_r)

            bar.set_postfix(
                seed=seed,
                reward=f"{total_r:,.0f}",
                electric=f"{n_elec}/{fleet_sz}",
            )
            bar.update(1)

    if failed_seeds:
        print(f"\n[WARN]  {len(failed_seeds)} episode(s) failed: seeds {failed_seeds}")

    if rewards:
        _print_summary(rewards, output_dir)


if __name__ == "__main__":
    main()
