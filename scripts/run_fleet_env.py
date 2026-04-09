from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

# Allow running this script directly from the repository root without installation.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _load_fleet_replacement_env_class():
    env_module_path = SRC_PATH / "fleet_replacement" / "envs" / "fleet_replacement.py"
    spec = importlib.util.spec_from_file_location(
        "fleet_replacement_env_module", env_module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load environment module from {env_module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FleetReplacementEnv


FleetReplacementEnv = _load_fleet_replacement_env_class()


def summarize_observation(obs: dict) -> None:
    fleet = obs["fleet"]
    info = obs["information_state"]

    n_electric = int(np.sum(fleet["is_electric"]))
    n_diesel = int(fleet["is_electric"].shape[0] - n_electric)

    print("Fleet summary")
    print(f"  Vehicles total: {fleet['is_electric'].shape[0]}")
    print(f"  Diesel trucks:  {n_diesel}")
    print(f"  Electric trucks:{n_electric}")
    print(f"  Avg age:        {float(np.mean(fleet['age'])):.2f} years")

    print("\nMarket state")
    print(f"  Current year: {int(info['current_year'][0])}")
    print(f"  Diesel energy price:      {float(info['energy_price_diesel'][0]):.3f}")
    print(
        f"  Electricity energy price: {float(info['energy_price_electricity'][0]):.3f}"
    )
    print(f"  Purchase price DT:        {float(info['purchase_price_DT'][0]):,.0f}")
    print(f"  Purchase price BET:       {float(info['purchase_price_BET'][0]):,.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FleetReplacementEnv skeleton loop"
    )
    parser.add_argument("--config", type=str, default="configs/env.yaml")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = FleetReplacementEnv(config_path=args.config)

    obs, info = env.reset(seed=args.seed)
    summarize_observation(obs)

    print("\nStepping environment")
    total_reward = 0.0
    for step_idx in range(args.max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"  Step {step_idx + 1}: action={action}, reward={reward:.3f}, "
            f"terminated={terminated}, truncated={truncated}"
        )
        if terminated or truncated:
            print("  Episode ended early.")
            break

    print(f"\nTotal reward: {total_reward:.3f}")


if __name__ == "__main__":
    main()
