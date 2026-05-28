"""
Episode history recording for the fleet replacement simulation.

``EpisodeRecord`` collects every step of a simulation run into a structured
object with array-valued properties that are convenient for analysis and
visualisation.

Typical workflow
----------------
>>> record = run_episode(env, agent, seed=42)   # from scripts/run_lookahead.py

>>> record.years          # array of years for each step
>>> record.n_electric     # electric vehicles per step
>>> record.actions        # (n_steps, fleet_size) action array

>>> df = record.to_dataframe()   # tidy per-step DataFrame (requires pandas)
>>> record.save("results/episode_42.pkl")

>>> record2 = EpisodeRecord.load("results/episode_42.pkl")
"""

from __future__ import annotations

import pathlib
import pickle
from dataclasses import dataclass

import gymnasium
import numpy as np

# ---------------------------------------------------------------------------
# Per-step data
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """All data recorded at a single environment step.

    Fleet arrays (``is_electric``, ``age``, ``purchase_price``) and
    market prices capture the state *at the start of the step* — i.e. what
    the agent observed before making its decision.

    Attributes
    ----------
    year : int
        Simulation year at the start of the step.
    action : np.ndarray
        Per-slot action integers, shape ``(fleet_size,)``.
        0 = Keep, 1 = Replace DT, 2 = Replace BET.
    is_electric : np.ndarray
        Per-slot electric flag (0/1), shape ``(fleet_size,)``.
    age : np.ndarray
        Per-slot vehicle age in years, shape ``(fleet_size,)``.
    purchase_price : np.ndarray
        Per-slot original purchase price, shape ``(fleet_size,)``.
    energy_price_diesel : float
        Observed diesel energy price (SEK/l or configured unit).
    energy_price_electricity : float
        Observed electricity price.
    purchase_price_DT : float
        Market price of a new diesel truck.
    purchase_price_BET : float
        Market price of a new electric truck.
    productivity_BET : float
        Relative BET productivity (1.0 = parity with DT).
    reward : float
        Total step reward.
    reward_components : dict[str, float]
        Itemised reward breakdown (revenue, costs, sale result, …).
    """

    year: int
    action: np.ndarray
    is_electric: np.ndarray
    age: np.ndarray
    purchase_price: np.ndarray
    energy_price_diesel: float
    energy_price_electricity: float
    purchase_price_DT: float
    purchase_price_BET: float
    productivity_BET: float
    reward: float
    reward_components: dict[str, float]


# ---------------------------------------------------------------------------
# Episode record
# ---------------------------------------------------------------------------


class EpisodeRecord:
    """Complete history of one simulation episode.

    Stores a list of :class:`StepRecord` instances and exposes them as
    NumPy arrays via read-only properties for easy vectorised access.

    Parameters
    ----------
    steps : list[StepRecord]
        One entry per environment step, ordered chronologically.
    seed : int
        Random seed used for the episode.
    """

    def __init__(
        self,
        steps: list[StepRecord],
        seed: int,
        bet_t0: float | None = None,
        bet_k: float | None = None,
    ) -> None:
        self.steps = steps
        self.seed = seed
        self.bet_t0 = bet_t0
        self.bet_k = bet_k

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    @property
    def n_steps(self) -> int:
        """Number of steps in the episode."""
        return len(self.steps)

    @property
    def fleet_size(self) -> int:
        """Number of fleet slots."""
        return len(self.steps[0].is_electric) if self.steps else 0

    # ------------------------------------------------------------------
    # Time axis
    # ------------------------------------------------------------------

    @property
    def years(self) -> np.ndarray:
        """Simulation year at the start of each step. Shape: ``(n_steps,)``."""
        return np.array([s.year for s in self.steps], dtype=np.int32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    @property
    def rewards(self) -> np.ndarray:
        """Total step reward. Shape: ``(n_steps,)``."""
        return np.array([s.reward for s in self.steps])

    @property
    def cumulative_rewards(self) -> np.ndarray:
        """Cumulative reward over the episode. Shape: ``(n_steps,)``."""
        return np.cumsum(self.rewards)

    @property
    def total_reward(self) -> float:
        """Sum of all step rewards."""
        return float(np.sum(self.rewards))

    def reward_component(self, key: str) -> np.ndarray:
        """Return per-step values for a single reward component.

        Parameters
        ----------
        key : str
            One of the keys in ``StepRecord.reward_components``, e.g.
            ``"revenue"``, ``"total_reward"``, ``"sale_result"`` …

        Returns
        -------
        np.ndarray
            Shape ``(n_steps,)``.
        """
        return np.array([s.reward_components.get(key, 0.0) for s in self.steps])

    # ------------------------------------------------------------------
    # Fleet composition
    # ------------------------------------------------------------------

    @property
    def n_electric(self) -> np.ndarray:
        """Number of electric vehicles per step. Shape: ``(n_steps,)``."""
        return np.array(
            [int(np.sum(s.is_electric)) for s in self.steps], dtype=np.int32
        )

    @property
    def n_diesel(self) -> np.ndarray:
        """Number of diesel vehicles per step. Shape: ``(n_steps,)``."""
        return np.array(
            [self.fleet_size - int(np.sum(s.is_electric)) for s in self.steps],
            dtype=np.int32,
        )

    @property
    def avg_age(self) -> np.ndarray:
        """Fleet average age per step. Shape: ``(n_steps,)``."""
        return np.array([float(np.mean(s.age)) for s in self.steps])

    @property
    def fleet_is_electric(self) -> np.ndarray:
        """Per-slot electric flag at the start of each step.
        Shape: ``(n_steps, fleet_size)``."""
        return np.stack([s.is_electric for s in self.steps])

    @property
    def fleet_ages(self) -> np.ndarray:
        """Per-slot vehicle age at the start of each step.
        Shape: ``(n_steps, fleet_size)``."""
        return np.stack([s.age for s in self.steps])

    @property
    def fleet_purchase_prices(self) -> np.ndarray:
        """Per-slot original purchase price at the start of each step.
        Shape: ``(n_steps, fleet_size)``."""
        return np.stack([s.purchase_price for s in self.steps])

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    @property
    def actions(self) -> np.ndarray:
        """Per-slot action per step. Shape: ``(n_steps, fleet_size)``.
        Values: 0 = Keep, 1 = Replace DT, 2 = Replace BET."""
        return np.stack([s.action for s in self.steps])

    # ------------------------------------------------------------------
    # Market prices
    # ------------------------------------------------------------------

    @property
    def energy_prices_diesel(self) -> np.ndarray:
        """Diesel energy price observed at each step. Shape: ``(n_steps,)``."""
        return np.array([s.energy_price_diesel for s in self.steps])

    @property
    def energy_prices_electricity(self) -> np.ndarray:
        """Electricity energy price observed at each step. Shape: ``(n_steps,)``."""
        return np.array([s.energy_price_electricity for s in self.steps])

    @property
    def purchase_prices_DT(self) -> np.ndarray:
        """New DT market price observed at each step. Shape: ``(n_steps,)``."""
        return np.array([s.purchase_price_DT for s in self.steps])

    @property
    def purchase_prices_BET(self) -> np.ndarray:
        """New BET market price observed at each step. Shape: ``(n_steps,)``."""
        return np.array([s.purchase_price_BET for s in self.steps])

    @property
    def productivities_BET(self) -> np.ndarray:
        """BET productivity observed at each step. Shape: ``(n_steps,)``."""
        return np.array([s.productivity_BET for s in self.steps])

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------

    def to_dataframe(self):
        """Return a tidy per-step DataFrame with all scalar fields.

        Reward components are included as separate columns.  Fleet-level
        aggregates (``n_electric``, ``n_diesel``, ``avg_age``) are included;
        per-slot arrays (``actions``, ``fleet_ages``, …) are not — access
        them via the array properties instead.

        Requires ``pandas``.

        Returns
        -------
        pandas.DataFrame
            One row per step, indexed by year.
        """
        import pandas as pd

        rows = []
        for s in self.steps:
            row: dict = {
                "year": s.year,
                "n_electric": int(np.sum(s.is_electric)),
                "n_diesel": self.fleet_size - int(np.sum(s.is_electric)),
                "avg_age": float(np.mean(s.age)),
                "energy_price_diesel": s.energy_price_diesel,
                "energy_price_electricity": s.energy_price_electricity,
                "purchase_price_DT": s.purchase_price_DT,
                "purchase_price_BET": s.purchase_price_BET,
                "productivity_BET": s.productivity_BET,
                "reward": s.reward,
            }
            row.update(s.reward_components)
            rows.append(row)

        return pd.DataFrame(rows).set_index("year")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Pickle the record to disk.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``"results/episode_42.pkl"``).
            Parent directories are created automatically.
        """
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> EpisodeRecord:
        """Load a pickled record from disk.

        Parameters
        ----------
        path : str or Path
            Path to a file previously written by :meth:`save`.

        Returns
        -------
        EpisodeRecord
        """
        with pathlib.Path(path).open("rb") as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EpisodeRecord("
            f"seed={self.seed}, "
            f"n_steps={self.n_steps}, "
            f"fleet_size={self.fleet_size}, "
            f"total_reward={self.total_reward:,.0f})"
        )


# ---------------------------------------------------------------------------
# Gymnasium wrapper
# ---------------------------------------------------------------------------


class EpisodeRecorder(gymnasium.Wrapper):
    """Gymnasium wrapper that records the full step history of each episode.

    Wraps any ``FleetReplacementEnv`` (or compatible env) and builds an
    :class:`EpisodeRecord` automatically as the episode progresses.  This is
    the idiomatic Gymnasium way to add recording: the wrapper intercepts
    ``reset()`` and ``step()`` without modifying the underlying environment.

    The recorder is policy-agnostic — it works with the lookahead agent,
    a random policy, or any other callable that returns valid actions.

    Parameters
    ----------
    env : gymnasium.Env
        The environment to wrap.

    Examples
    --------
    >>> env = FleetReplacementEnv(config=config)
    >>> env = EpisodeRecorder(env)          # wrap once; stack with other wrappers freely
    >>>
    >>> obs, info = env.reset(seed=42)
    >>> while True:
    ...     action = agent.select_action(obs)
    ...     obs, reward, terminated, truncated, info = env.step(action)
    ...     if terminated or truncated:
    ...         break
    >>>
    >>> record = env.episode_record         # EpisodeRecord with full history
    >>> record.save("results/ep42.pkl")
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        self._steps: list[StepRecord] = []
        self._seed: int = 0
        self._prev_obs: dict | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._steps = []
        self._seed = seed if seed is not None else 0
        self._prev_obs = obs
        return obs, info

    def step(self, action):
        if self._prev_obs is None:
            raise RuntimeError("Call reset() before step().")

        # Snapshot the state the agent observed before acting.
        fleet_snap = {k: v.copy() for k, v in self._prev_obs["fleet"].items()}
        info_snap = {
            k: v.copy() for k, v in self._prev_obs["information_state"].items()
        }

        obs, reward, terminated, truncated, info = self.env.step(action)

        self._steps.append(
            StepRecord(
                year=int(info_snap["current_year"][0]),
                action=np.asarray(action, dtype=np.int64).copy(),
                is_electric=fleet_snap["is_electric"].copy(),
                age=fleet_snap["age"].copy(),
                purchase_price=fleet_snap["purchase_price"].copy(),
                energy_price_diesel=float(info_snap["energy_price_diesel"][0]),
                energy_price_electricity=float(
                    info_snap["energy_price_electricity"][0]
                ),
                purchase_price_DT=float(info_snap["purchase_price_DT"][0]),
                purchase_price_BET=float(info_snap["purchase_price_BET"][0]),
                productivity_BET=float(info_snap["productivity_BET"][0]),
                reward=float(reward),
                reward_components=dict(info["reward"]),
            )
        )

        self._prev_obs = obs
        return obs, reward, terminated, truncated, info

    @property
    def episode_record(self) -> EpisodeRecord:
        """The :class:`EpisodeRecord` for the current (or most recently completed) episode.

        Raises
        ------
        RuntimeError
            If no steps have been recorded yet (i.e. ``reset()`` was called but
            ``step()`` has not been).
        """
        if not self._steps:
            raise RuntimeError("No steps recorded yet. Call step() at least once.")
        return EpisodeRecord(
            steps=list(self._steps),
            seed=self._seed,
            bet_t0=self.env._bet_t0,
            bet_k=self.env._bet_k,
        )
