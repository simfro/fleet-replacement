from enum import Enum
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fleet_replacement.config import EnvConfig, load_env_config


class Actions(Enum):
    KEEP = 0
    REPLACE_DT = 1
    REPLACE_BET = 2


class FleetReplacementEnv(gym.Env):
    def __init__(
        self,
        config: EnvConfig | None = None,
        config_path: str | Path | None = None,
        render_mode=None,
    ):
        if config is None:
            default_config_path = (
                Path(__file__).resolve().parents[3] / "configs" / "env.yaml"
            )
            config = load_env_config(config_path or default_config_path)

        self.config = config
        self.fleet_size = self.config.vehicle_management.fleet_size
        self.max_vehicle_age = self.config.vehicle_management.max_age
        self.start_year = self.config.simulation_period.base_year
        self.final_year = self.config.simulation_period.final_year
        self._current_year = self.start_year
        self.render_mode = render_mode

        # is_electric: 0 = Diesel Truck (DT), 1 = Battery Electric Truck (BET)
        self.observation_space = spaces.Dict(
            {
                "fleet": spaces.Dict(
                    {
                        "is_electric": spaces.MultiBinary(self.fleet_size),
                        "age": spaces.Box(
                            low=0,
                            high=self.max_vehicle_age,
                            shape=(self.fleet_size,),
                            dtype=np.int32,
                        ),
                        "purchase_price": spaces.Box(
                            low=0.0,
                            high=np.finfo(np.float32).max,
                            shape=(self.fleet_size,),
                            dtype=np.float32,
                        ),
                    }
                ),
                "information_state": spaces.Dict(
                    {
                        "current_year": spaces.Box(
                            low=self.start_year,
                            high=self.final_year,
                            shape=(1,),
                            dtype=np.int32,
                        ),
                        "energy_price_diesel": spaces.Box(
                            low=0.0,
                            high=np.finfo(np.float32).max,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        "energy_price_electricity": spaces.Box(
                            low=0.0,
                            high=np.finfo(np.float32).max,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        "purchase_price_DT": spaces.Box(
                            low=0.0,
                            high=np.finfo(np.float32).max,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        "purchase_price_BET": spaces.Box(
                            low=0.0,
                            high=np.finfo(np.float32).max,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                    }
                ),
            }
        )

        self._fleet = self._initial_fleet(self.fleet_size)
        self._info_state = self._initial_info_state()

        # One action per vehicle:
        # Actions.KEEP = keep, Actions.REPLACE_DT = replace with DT,
        # Actions.REPLACE_BET = replace with BET (medium battery).
        self.action_space = spaces.MultiDiscrete(
            np.full(self.fleet_size, len(Actions), dtype=np.int64)
        )

    def _get_obs(self):
        return {"fleet": self._fleet, "information_state": self._info_state}

    def _get_info(self):
        return {
            "current_year": self._current_year,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Deterministic reset: all diesel vehicles purchased at start_year.
        self._current_year = self.start_year
        self._fleet = self._initial_fleet(self.fleet_size)
        self._info_state = self._initial_info_state()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _initial_fleet(self, fleet_size: int):
        return {
            "is_electric": np.zeros(fleet_size, dtype=np.int8),
            "age": np.zeros(fleet_size, dtype=np.int32),
            "purchase_price": np.ones(fleet_size, dtype=np.float32)
            * self.config.DT_price.initial_price,
        }

    def _initial_info_state(self):
        return {
            "current_year": self.start_year,
            "energy_price_diesel": self.config.diesel_price.initial_price,
            "energy_price_electricity": self.config.electricity_price.initial_price,
            "purchase_price_DT": self.config.DT_price.initial_price,
            "purchase_price_BET": self.config.BET_price.initial_price,
        }

    def step(self, action):
        # TODO: Implement me
        # Update the fleet according to the action.
        self._update_fleet(action)

    def _update_fleet(self, action):
        for idx, a in enumerate(action):
            if a == Actions.KEEP.value:
                self._fleet["age"][idx] += 1
            elif a == Actions.REPLACE_DT.value:
                self._fleet["is_electric"][idx] = 0
                self._fleet["age"][idx] = 0
                self._fleet["purchase_price"][idx] = self._info_state[
                    "purchase_price_DT"
                ]
            elif a == Actions.REPLACE_BET.value:
                self._fleet["is_electric"][idx] = 1
                self._fleet["age"][idx] = 0
                self._fleet["purchase_price"][idx] = self._info_state[
                    "purchase_price_BET"
                ]
            else:
                raise ValueError(f"Invalid action {a} for vehicle {idx}")
