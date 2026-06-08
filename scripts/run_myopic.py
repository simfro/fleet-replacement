"""
Run a complete episode of FleetReplacementEnv using the MyopicAgent policy.

The agent selects replacement decisions by maximising the one-step expected
value per vehicle slot (current sale proceeds + discounted next-year operating
profit), using frozen current market prices.  Detailed per-step output shows
the year, per-slot decisions, fleet state, market prices, and reward breakdown.

Usage
-----
    python scripts/run_myopic.py
    python scripts/run_myopic.py --config configs/env.yaml --seed 42
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

# Allow running directly from the repository root without a package install.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fleet_replacement.config import load_env_config  # noqa: E402
from fleet_replacement.envs.fleet_replacement import FleetReplacementEnv  # noqa: E402
from fleet_replacement.episode import EpisodeRecord, EpisodeRecorder  # noqa: E402
from fleet_replacement.policies import MyopicAgent  # noqa: E402

# ---------------------------------------------------------------------------
# Display helpers (identical to run_lookahead.py)
# ---------------------------------------------------------------------------

_ACTION_NAMES = {0: "Keep", 1: "Replace DT", 2: "Replace BET"}
_VT_NAMES = {0: "DT", 1: "BET"}
_REWARD_LABELS = {
    "revenue": "Revenue",
    "diesel_cost": "Diesel cost",
    "electricity_cost": "Electricity cost",
    "salary_cost": "Salary cost",
    "interest_cost": "Interest cost",
    "depreciation_cost": "Depreciation cost",
    "sale_result": "Sale result",
    "capex": "Capex",
    "opex": "Opex",
    "total_reward": "Total reward",
}


def _print_header(title: str, width: int = 60) -> None:
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


def _print_initial_state(obs: dict) -> None:
    fleet = obs["fleet"]
    info = obs["information_state"]
    _print_header("Initial fleet state")
    _print_fleet(fleet)
    _print_market(info)


def _print_fleet(fleet: dict) -> None:
    is_electric = fleet["is_electric"]
    ages = fleet["age"]
    prices = fleet["purchase_price"]
    n = len(is_electric)
    n_elec = int(np.sum(is_electric))
    n_diesel = n - n_elec

    print("\nFleet")
    print(f"  Total vehicles : {n}")
    print(f"  Diesel trucks  : {n_diesel}")
    print(f"  Electric trucks: {n_elec}")
    print(f"  Average age    : {float(np.mean(ages)):.2f}")
    print("  Slots")
    for i in range(n):
        vtype = _VT_NAMES[int(is_electric[i])]
        print(
            f"    [{i:>2d}]  {vtype:<3s}  age {int(ages[i]):>2d}  "
            f"purchase {float(prices[i]):>12,.0f}"
        )


def _print_market(info: dict) -> None:
    print("\nMarket")
    print(f"  Year                  : {int(info['current_year'][0])}")
    print(f"  Diesel energy price   : {float(info['energy_price_diesel'][0]):.3f}")
    print(f"  Electricity price     : {float(info['energy_price_electricity'][0]):.3f}")
    print(f"  DT purchase price     : {float(info['purchase_price_DT'][0]):>12,.0f}")
    print(f"  BET purchase price    : {float(info['purchase_price_BET'][0]):>12,.0f}")
    print(f"  BET productivity      : {float(info['productivity_BET'][0]):.3f}")


def _print_step(
    step: int,
    prev_obs: dict,
    action: np.ndarray,
    obs: dict,
    reward: float,
    reward_components: dict[str, float],
    terminated: bool,
) -> None:
    year_from = int(prev_obs["information_state"]["current_year"][0])
    year_to = int(obs["information_state"]["current_year"][0])
    _print_header(f"Step {step:>3d}  |  {year_from} -> {year_to}")

    fleet_prev = prev_obs["fleet"]
    print("\nDecisions")
    for i, a in enumerate(action):
        vtype = _VT_NAMES[int(fleet_prev["is_electric"][i])]
        age = int(fleet_prev["age"][i])
        decision_str = _ACTION_NAMES[int(a)]
        print(f"  slot {i:>2d}  {vtype:<3s}  age {age:>2d}  ->  {decision_str}")

    _print_fleet(obs["fleet"])
    _print_market(obs["information_state"])

    print("\nReward breakdown")
    for key, label in _REWARD_LABELS.items():
        if key == "total_reward":
            print(f"  {'-' * 36}")
        val = reward_components.get(key, 0.0)
        sign = "+" if val >= 0 else ""
        print(f"  {label:<22s}  {sign}{val:>14,.2f}")

    if terminated:
        print("\n  [Episode terminated]")


def _print_episode_summary(record: EpisodeRecord, final_obs: dict) -> None:
    _print_header("Episode summary")
    fleet = final_obs["fleet"]
    is_electric = fleet["is_electric"]
    ages = fleet["age"]
    n = len(is_electric)
    n_elec = int(np.sum(is_electric))
    print(f"\n  Total reward     : {record.total_reward:>16,.2f}")
    print(f"  Steps            : {record.n_steps}")
    print(f"  Final fleet size : {n}")
    print(f"  Diesel trucks    : {n - n_elec}")
    print(f"  Electric trucks  : {n_elec}")
    print(f"  Average age      : {float(np.mean(ages)):.2f}")
    print()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    env: EpisodeRecorder, agent: MyopicAgent, seed: int
) -> EpisodeRecord:
    """Run one full episode and return an :class:`EpisodeRecord`."""
    obs, _ = env.reset(seed=seed)
    _print_initial_state(obs)

    step = 0

    while True:
        step += 1
        prev_obs = {
            "fleet": {k: v.copy() for k, v in obs["fleet"].items()},
            "information_state": {
                k: v.copy() for k, v in obs["information_state"].items()
            },
        }

        action = agent.select_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        _print_step(
            step=step,
            prev_obs=prev_obs,
            action=action,
            obs=obs,
            reward=float(reward),
            reward_components=info["reward"],
            terminated=terminated,
        )

        if terminated or truncated:
            break

    record = env.episode_record
    _print_episode_summary(record, obs)
    return record


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FleetReplacementEnv for one episode using the MyopicAgent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "env.yaml"),
        help="Path to the environment YAML config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the environment.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="PATH",
        help="If given, save the EpisodeRecord to this path (pickle). "
        "Parent directories are created automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_env_config(args.config)

    print(f"Config     : {args.config}")
    print(f"Seed       : {args.seed}")
    print(f"Fleet size : {config.vehicle_management.fleet_size} vehicles")
    print(
        f"Episode    : {config.simulation_period.base_year}"
        f" -> {config.simulation_period.final_year}"
        f"  ({config.simulation_period.final_year - config.simulation_period.base_year} steps)"
    )

    env = EpisodeRecorder(FleetReplacementEnv(config=config))
    agent = MyopicAgent(env)

    record = run_episode(env, agent, seed=args.seed)

    if args.save:
        save_path = pathlib.Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        record.save(str(save_path))
        print(f"Episode record saved to: {args.save}")


if __name__ == "__main__":
    main()
