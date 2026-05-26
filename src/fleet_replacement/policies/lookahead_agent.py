"""
LookaheadAgent: policy that wraps the deterministic lookahead planning model.

At every environment step the agent:
  1. Converts the Gymnasium observation into a ``FleetReplacementData`` package.
  2. Constructs a linear price/productivity forecast from the current info-state.
  3. Builds and solves the MILP planning model over the configured horizon.
  4. Returns the first-period decisions as a Gymnasium action array.

Usage
-----
>>> from fleet_replacement.config import load_env_config
>>> from fleet_replacement.envs.fleet_replacement import FleetReplacementEnv
>>> from fleet_replacement.policies import LookaheadAgent
>>>
>>> config = load_env_config("configs/env.yaml")
>>> env = FleetReplacementEnv(config=config)
>>> agent = LookaheadAgent(config)
>>>
>>> obs, info = env.reset(seed=0)
>>> while True:
...     action = agent.select_action(obs)
...     obs, reward, terminated, truncated, info = env.step(action)
...     if terminated or truncated:
...         break
"""

from __future__ import annotations

import numpy as np

from fleet_replacement.config import EnvConfig
from fleet_replacement.policies.lookahead_model import (
    best_immediate_actions,
    build_model,
    Decision,
    FleetReplacementData,
    ForecastParams,
    make_forecast,
    ModelParams,
    solve,
)

# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------

# Maps the planning model's Decision strings to the integer action codes
# expected by FleetReplacementEnv (matching the Actions enum in fleet_replacement.py).
_DECISION_TO_INT: dict[Decision, int] = {
    Decision.KEEP: 0,
    Decision.REPLACE_DT: 1,
    Decision.REPLACE_BET: 2,
}


def _decisions_to_action(decisions: list[Decision]) -> np.ndarray:
    """Convert a list of :class:`Decision` values to a Gymnasium action array.

    Returns an ``np.int64`` array of shape ``(fleet_size,)`` with values in
    ``{0, 1, 2}`` matching ``Actions.KEEP``, ``Actions.REPLACE_DT``, and
    ``Actions.REPLACE_BET`` respectively.
    """
    return np.array([_DECISION_TO_INT[d] for d in decisions], dtype=np.int64)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class LookaheadAgent:
    """Policy agent that uses a deterministic MILP lookahead to select fleet actions.

    At each step the agent builds and solves a ``horizon``-period planning model
    from the current observation, then returns the first-period decisions as a
    Gymnasium-compatible action array.

    Parameters
    ----------
    config:
        Loaded environment configuration, including the ``lookahead`` section
        that specifies the planning horizon and price forecast growth rates.

    Examples
    --------
    >>> agent = LookaheadAgent(config)
    >>> action = agent.select_action(obs)   # np.ndarray, shape (fleet_size,)
    """

    def __init__(self, config: EnvConfig) -> None:
        self._config = config
        self._model_params = ModelParams.from_env_config(
            config, config.lookahead.horizon
        )
        fc = config.lookahead.forecast_rates
        self._forecast_params = ForecastParams(
            diesel_price_growth=fc.diesel_price_growth,
            electricity_price_growth=fc.electricity_price_growth,
            purchase_price_growth_DT=fc.purchase_price_growth_DT,
            purchase_price_growth_BET=fc.purchase_price_growth_BET,
            BET_productivity_growth=fc.BET_productivity_growth,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def horizon(self) -> int:
        """Planning horizon in number of periods."""
        return self._model_params.horizon

    @property
    def model_params(self) -> ModelParams:
        """Read-only view of the model parameters used by this agent."""
        return self._model_params

    @property
    def forecast_params(self) -> ForecastParams:
        """Read-only view of the forecast growth rates used by this agent."""
        return self._forecast_params

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def select_action(self, obs: dict) -> np.ndarray:
        """Return the optimal first-period action for all fleet slots.

        Builds and solves the lookahead MILP from the given observation, then
        maps the first-period decisions to a Gymnasium action array.

        Parameters
        ----------
        obs:
            Observation dict produced by ``FleetReplacementEnv._get_obs()``.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(fleet_size,)`` with values in ``{0, 1, 2}``:
            0 = Keep, 1 = Replace with DT, 2 = Replace with BET.

        Raises
        ------
        RuntimeError
            If the MILP solver does not find an optimal solution.
        """
        data = FleetReplacementData.from_env_observation(obs, self._model_params)
        forecast = make_forecast(
            data.info_state,
            self._model_params.horizon,
            self._model_params.BET_productivity_max,
            self._forecast_params,
        )
        model = build_model(data, forecast)
        solve(model)
        decisions = best_immediate_actions(model, data)
        return _decisions_to_action(decisions)
