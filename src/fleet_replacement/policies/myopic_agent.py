"""
MyopicAgent: policy that selects fleet replacement actions by simulating one
environment step per candidate action and picking the highest reward.

For each vehicle slot the agent evaluates three candidate actions
{KEEP, REPLACE_DT, REPLACE_BET} independently.  For each candidate it
deep-copies the current environment state, calls ``env.step(test_action)``,
and selects the action that maximises the immediate reward returned by the
environment.  The environment is the single source of truth for the reward
calculation; no reward logic is duplicated here.

Because the fleet replacement reward is additive across vehicle slots,
per-slot greedy evaluation is globally optimal for the one-step horizon.
Prices are frozen at the current environment state (no forecast).

Note
----
``FleetReplacementEnv`` computes the step reward entirely from the pre-step
fleet state and current market prices, so the reward returned by a simulated
step is deterministic given the current observation.  The stochastic price
transitions only affect the *next* observation, not the current reward.

Usage
-----
>>> from fleet_replacement.config import load_env_config
>>> from fleet_replacement.envs.fleet_replacement import FleetReplacementEnv
>>> from fleet_replacement.episode import EpisodeRecorder
>>> from fleet_replacement.policies.myopic_agent import MyopicAgent
>>>
>>> config = load_env_config("configs/env.yaml")
>>> env = EpisodeRecorder(FleetReplacementEnv(config=config))
>>> agent = MyopicAgent(env)
>>>
>>> obs, info = env.reset(seed=0)
>>> while True:
...     action = agent.select_action(obs)
...     obs, reward, terminated, truncated, info = env.step(action)
...     if terminated or truncated:
...         break
"""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np

# Action codes match FleetReplacementEnv.Actions enum values.
_KEEP = 0
_REPLACE_DT = 1
_REPLACE_BET = 2


class MyopicAgent:
    """Policy agent that selects actions by simulating one environment step.

    For each vehicle slot the agent deep-copies the current environment state
    and calls ``env.step(test_action)`` for each candidate action, using the
    reward returned by the environment to rank the choices.  The action that
    maximises the immediate reward is selected.

    Decisions are made independently per slot; since the fleet replacement
    reward is additive across slots, per-slot greedy evaluation is globally
    optimal for the one-step horizon.

    Parameters
    ----------
    env:
        The Gymnasium environment (or wrapper) used for the actual episode.
        ``env.unwrapped`` is accessed internally so that deep-copying for
        simulation does not include wrapper-level state such as episode
        recording buffers.
    """

    def __init__(self, env: gym.Env) -> None:
        # Store a reference to the base (non-wrapped) environment.
        # Deep-copying this during select_action gives a clean snapshot of
        # the current environment state for simulation.
        self._base_env = env.unwrapped
        self._max_age = self._base_env.config.vehicle_management.max_age

    def select_action(self, obs: dict) -> np.ndarray:
        """Return the greedy one-step action for all fleet slots.

        For each slot the agent simulates each candidate action in a deep copy
        of the current environment and picks the one with the highest immediate
        reward.  Decisions are built up sequentially so that each slot's
        simulation reflects the already-committed actions for earlier slots.

        Parameters
        ----------
        obs:
            Observation dict produced by ``FleetReplacementEnv._get_obs()``.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(fleet_size,)`` with values in ``{0, 1, 2}``:
            0 = Keep, 1 = Replace with DT, 2 = Replace with BET.
        """
        age_arr = obs["fleet"]["age"]
        fleet_size = len(age_arr)
        max_age = self._max_age

        # Build action array incrementally (sequential per-slot greedy).
        # Each slot's simulation uses the decisions already committed for
        # earlier slots, with KEEP as the placeholder for later slots.
        chosen = np.zeros(fleet_size, dtype=np.int64)

        for i in range(fleet_size):
            age = float(age_arr[i])

            # KEEP is only valid while the vehicle is below max_age.
            candidates = [_REPLACE_DT, _REPLACE_BET]
            if age < max_age:
                candidates.append(_KEEP)

            best_action, best_reward = candidates[0], -float("inf")

            for a in candidates:
                test_action = chosen.copy()
                test_action[i] = a

                # Simulate one step without affecting the real environment.
                sim = copy.deepcopy(self._base_env)
                _, reward, _, _, _ = sim.step(test_action)

                if reward > best_reward:
                    best_reward, best_action = reward, a

            chosen[i] = best_action

        return chosen
