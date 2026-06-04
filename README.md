# Fleet Replacement

A simulation and optimisation framework for studying the electrification of a heavy-truck fleet under stochastic market conditions. A fleet manager must decide each year, for every vehicle in the fleet, whether to **keep** it, **replace it with a new diesel truck (DT)**, or **replace it with a battery electric truck (BET)**. The environment is compatible with the [Gymnasium](https://gymnasium.farama.org/) API.

---

## Table of Contents

1. [Overview](#overview)
2. [Simulation Environment](#simulation-environment)
   - [State space](#state-space)
   - [Action space](#action-space)
   - [Stochastic processes](#stochastic-processes)
   - [Reward model](#reward-model)
3. [Lookahead Agent](#lookahead-agent)
   - [Planning model](#planning-model)
   - [Decision variables and constraints](#decision-variables-and-constraints)
   - [Objective function](#objective-function)
   - [Forecast model](#forecast-model)
4. [Installation](#installation)
5. [Configuration](#configuration)
   - [YAML structure](#yaml-structure)
   - [Preset configurations](#preset-configurations)
6. [Usage](#usage)
   - [Single episode with lookahead agent](#single-episode-with-lookahead-agent)
   - [Batch runs](#batch-runs)
   - [Environment only (custom or random policy)](#environment-only-custom-or-random-policy)
   - [Direct MILP API](#direct-milp-api)
7. [Project structure](#project-structure)
8. [Notebooks](#notebooks)

---

## Overview

The simulation covers the period **2026–2046** (20 years). The fleet consists of 8 trucks. At each annual decision step the manager observes the current fleet composition and market conditions, then chooses a replacement action for every vehicle slot. Profits accumulate over the episode and are discounted to present value.

The core research question is: _when_ and _how fast_ should an operator transition from diesel to electric trucks, given that BET purchase prices are currently high but falling and BET operational productivity is currently below diesel but improving?

The framework contains two interacting components:

| Component             | Description                                                                         |
| --------------------- | ----------------------------------------------------------------------------------- |
| `FleetReplacementEnv` | Stochastic Gymnasium environment that simulates the fleet and market                |
| `LookaheadAgent`      | Decision-making policy: solves a deterministic MILP over a rolling planning horizon |

---

## Simulation Environment

**Class:** `fleet_replacement.envs.fleet_replacement.FleetReplacementEnv`

The environment follows the standard Gymnasium loop: `reset()` returns an initial observation, `step(action)` advances the simulation by one year and returns `(obs, reward, terminated, truncated, info)`.

### State space

The observation is a dict with two sub-dicts:

**`fleet`** — one entry per vehicle slot:

| Key                       | Type                | Description                                                 |
| ------------------------- | ------------------- | ----------------------------------------------------------- |
| `fleet["is_electric"]`    | `int[fleet_size]`   | 1 = BET, 0 = DT                                             |
| `fleet["age"]`            | `int[fleet_size]`   | Vehicle age in years (0 – `max_age`)                        |
| `fleet["purchase_price"]` | `float[fleet_size]` | Original purchase price when the vehicle was acquired (SEK) |

**`info_state`** — current market conditions:

| Key                        | Description                                     |
| -------------------------- | ----------------------------------------------- |
| `current_year`             | Simulation year                                 |
| `energy_price_diesel`      | Diesel fuel price (SEK/L)                       |
| `energy_price_electricity` | Electricity price (SEK/kWh)                     |
| `purchase_price_DT`        | New diesel truck purchase price (SEK)           |
| `purchase_price_BET`       | New battery electric truck purchase price (SEK) |
| `productivity_BET`         | BET productivity relative to DT (0–1; DT = 1.0) |

Vehicles are initialised with staggered ages at episode start to avoid artificial bulk-replacement events.

### Action space

Each step requires an integer action for every vehicle slot: a `MultiDiscrete` array of shape `(fleet_size,)` with values in `{0, 1, 2}`.

| Code | Action        | Effect                                          |
| ---- | ------------- | ----------------------------------------------- |
| `0`  | `KEEP`        | Vehicle remains; age increments by 1            |
| `1`  | `REPLACE_DT`  | Scrap current vehicle; enter a new DT at age 0  |
| `2`  | `REPLACE_BET` | Scrap current vehicle; enter a new BET at age 0 |

A vehicle that reaches `max_age` (7 years) is **forced to be replaced** regardless of the chosen action.

### Stochastic processes

Market state variables evolve between steps. The processes are defined in `src/fleet_replacement/envs/models/exogenous_process.py`.

**Energy prices — Geometric Brownian Motion (GBM)**

$$P_t = P_{t-1} \exp\!\left(\mu - \tfrac{1}{2}\sigma^2 + \sigma Z_t\right), \quad Z_t \sim \mathcal{N}(0,1)$$

Applied to both diesel price ($\mu$ = `growth_rate`, $\sigma$ = `volatility`) and electricity price.

**BET purchase price — mean-reverting process**

$$P_t^{\text{BET}} = P_{t-1}^{\text{BET}} + \lambda\!\left(\bar{P} - P_{t-1}^{\text{BET}}\right) + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,\,\sigma_{\text{BET}}^2)$$

where $\lambda$ = `mean_reversion_strength` (0.2), $\bar{P}$ = `long_term_mean` (1 800 000 SEK), $\sigma_{\text{BET}}$ = `purchase_price_volatility` (100 000 SEK).

**BET productivity — logistic growth (stochastic per episode)**

$$\text{prod}(t;\, k, t_0) = p_{\min} + (p_{\max} - p_{\min}) \cdot \frac{1}{1 + e^{-k(t - t_0)}}$$

The parameters $k \in [k_{\min},\, k_{\max}]$ and $t_0 \in [t_{0,\min},\, t_{0,\max}]$ are drawn **once per episode** from uniform distributions, representing different possible technology-adoption trajectories. The default range is $k \in [0.15, 0.60]$, $t_0 \in [2030, 2042]$.

**DT purchase price** remains constant across the episode (no drift or volatility).

### Reward model

The reward at each step is the **total annual profit** across all fleet slots, computed by `src/fleet_replacement/envs/models/reward_model.py`.

$$r_t = \sum_{s=1}^{N} \Bigl[\text{Revenue}_s - \text{OPEX}_s - \text{CAPEX}_s + \text{SaleResult}_s\Bigr]$$

**Revenue** depends on vehicle type: DT earns full revenue, BET earns revenue scaled by `productivity_BET`:

$$\text{Revenue}_s = \text{income\_per\_km} \times \text{annual\_mileage} \times \begin{cases} 1 & \text{DT} \\ \text{productivity\_BET} & \text{BET} \end{cases}$$

Default: 30 SEK/km × 130 000 km = 3 900 000 SEK/year per DT slot.

**OPEX** per slot:

$$\text{OPEX}_s = \underbrace{c_{\text{fuel}} \times d \times p_{\text{energy}}}_{\text{energy cost}} + \underbrace{\frac{w}{N}}_{\text{driver salary share}}$$

where $c_{\text{fuel}}$ is fuel/electricity consumption per km, $d$ is annual mileage, and $w$ = `driver_salary_annual`.

**CAPEX** per slot (annual cost of owning the vehicle):

$$\text{CAPEX}_s = \underbrace{P_{\text{orig}} \cdot f_{\text{loan}} \cdot r_{\text{interest}}}_{\text{interest}} + \underbrace{\frac{P_{\text{orig}}}{L_{\text{econ}}}}_{\text{depreciation}}$$

where $f_{\text{loan}}$ = `loan_fraction` (0.8), $r_{\text{interest}}$ = `interest_rate` (4.1 %), $L_{\text{econ}}$ = `economic_lifetime` (5 years).

**Sale result** when a vehicle is replaced: gain or loss relative to book value.

$$\text{SaleResult}_s = V_{\text{res}} - V_{\text{book}}$$

The residual value $V_{\text{res}}$ tracks market depreciation with an elasticity adjustment based on current market prices. At the terminal year all remaining vehicles are notionally sold at residual value.

---

## Lookahead Agent

**Class:** `fleet_replacement.policies.LookaheadAgent`

At every step the agent solves a deterministic **Mixed Integer Linear Programme (MILP)** over a rolling `horizon`-year planning window (default 10 years). It then executes the first-period decisions from the optimal plan, re-solves next step with updated observations, and repeats — a receding-horizon (model predictive control) structure.

The MILP is built with [linopy](https://linopy.readthedocs.io/) and solved with the [HiGHS](https://highs.dev/) solver.

### Planning model

**Source:** `src/fleet_replacement/policies/lookahead_model.py`

The model treats the fleet as a set of $N$ independent slots. Each slot $s$ at time $t$ is occupied by exactly one vehicle of type $v \in \{\text{DT}, \text{BET}\}$ at age $a \in \{0, \ldots, a_{\max}\}$.

### Decision variables and constraints

| Variable        | Shape                | Type   | Meaning                                                                                           |
| --------------- | -------------------- | ------ | ------------------------------------------------------------------------------------------------- |
| `R[t, s, v, a]` | `(H, N, 2, a_max+1)` | Binary | Indicator: slot $s$ at time $t$ contains vehicle of type $v$ with age $a$                         |
| `x[t, s, d]`    | `(H-1, N, 2)`        | Binary | Replacement decision $d \in \{\text{REPLACE\_DT}, \text{REPLACE\_BET}\}$ for slot $s$ at time $t$ |

Key constraints:

1. **Unique vehicle per slot:** $\sum_{v,a} R[t,s,v,a] = 1 \quad \forall t, s$
2. **At most one replacement per slot per year:** $x[t,s,\text{DT}] + x[t,s,\text{BET}] \leq 1$
3. **Forced replacement at max age:** vehicles at $a = a_{\max}$ must leave the fleet
4. **Age transitions (keep):** if kept, $R[t+1,s,v,a+1] \geq R[t,s,v,a] - x[t,s,\cdot]$
5. **Replacement transitions:** a replacement decision causes $R[t+1,s,v',0]$ to be set for the chosen type $v'$

The initial fleet state (from the environment observation) is fixed as the $t=0$ boundary condition.

### Objective function

Maximise discounted cumulative profit over the planning horizon $H$:

$$\max \sum_{t=0}^{H-1} \gamma^t \sum_{s} \Bigl[\text{Revenue}_{t,s} - \text{EnergyCost}_{t,s} - \text{InterestCost}_{t,s} - \text{DepreciationCost}_{t,s} + \text{SaleResult}_{t,s}\Bigr] + \gamma^H \sum_s \text{TerminalSale}_s$$

where $\gamma$ = `discount_factor` (default 0.96). All terms are linear in the binary variables `R` and `x`, making the model a MILP.

Purchase prices in the objective use the vehicle's **original purchase price** for existing vehicles (known from the observation) and the **forecasted purchase price at the replacement year** for future acquisitions.

### Forecast model

Since the MILP is deterministic, market prices must be forecast over the planning horizon. Linear growth rates are used for each component (`forecast_rates` in the config):

| Quantity           | Forecast                                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| Diesel price       | $P_{\text{DT,energy}}(t) = P_0 \cdot (1 + r_{\text{diesel}} \cdot t)$                                                     |
| Electricity price  | $P_{\text{elec}}(t) = P_0 \cdot (1 + r_{\text{elec}} \cdot t)$                                                            |
| DT purchase price  | $P_{\text{DT}}(t) = P_{\text{DT}}(0) \cdot (1 + r_{\text{DT}} \cdot t)$                                                   |
| BET purchase price | Gap-closure: $P_{\text{BET}}(t) = P_{\text{DT}}(t) + \max\!\left(0,\; \Delta_0 \cdot (1 - r_{\text{gap}} \cdot t)\right)$ |
| BET productivity   | $\text{prod}(t) = \min\!\left(p_{\max},\; \text{prod}_0 \cdot (1 + r_{\text{prod}} \cdot t)\right)$                       |

where $\Delta_0 = P_{\text{BET}}(0) - P_{\text{DT}}(0)$ is the current price gap. The growth rates are set by the chosen config preset and represent the agent's _beliefs_ about future market developments.

---

## Installation

**Prerequisites:** Python 3.10+, a virtual environment (recommended).

```bash
# Clone the repository and install in editable mode
git clone <repo-url>
cd fleet-replacement
pip install -e .
```

All dependencies (`gymnasium`, `linopy`, `highspy`, `pyyaml`, `tqdm`, `pygame`) are declared in `pyproject.toml` and installed automatically.

**Verify the installation:**

```python
import gymnasium
env = gymnasium.make("FleetReplacement-v0")
obs, info = env.reset()
print(obs["info_state"]["current_year"])  # 2026
```

---

## Configuration

### YAML structure

All parameters live in a single YAML file. The full schema mirrors the sections below. Pass the path to `load_env_config()` or to the CLI scripts via `--config`.

```yaml
simulation_period:
  base_year: 2026
  final_year: 2046

diesel_price:
  initial_price: 18.0 # SEK/L
  growth_rate: 0.02 # annual drift for GBM
  volatility: 0.02 # annual volatility for GBM

electricity_price:
  initial_price: 3.0 # SEK/kWh
  growth_rate: 0.0
  volatility: 0.04

DT_price:
  initial_price: 1800000.0 # SEK

BET_price:
  initial_price: 3200000.0 # SEK
  long_term_mean: 1800000.0
  mean_reversion_strength: 0.2
  purchase_price_volatility: 100000.0

BET_productivity:
  start: 0.70 # initial productivity relative to DT
  max: 0.95 # asymptotic maximum
  k_min: 0.15 # logistic steepness — lower bound (sampled per episode)
  k_max: 0.60 # logistic steepness — upper bound
  t0_min: 2030 # inflection year — lower bound
  t0_max: 2042 # inflection year — upper bound

operational:
  fuel_consumption_l_per_km: 0.6
  electricity_consumption_kwh_per_km: 2.5
  driver_salary_annual: 800000 # SEK, shared across fleet
  income_per_km: 30 # SEK/km
  annual_mileage_km: 130000

economic:
  interest_rate: 0.041
  loan_fraction: 0.8
  economic_lifetime: 5 # years (for linear book depreciation)
  discount_factor: 0.96

vehicle_management:
  fleet_size: 8
  max_age: 7 # vehicles at this age must be replaced
  max_replacements: 1 # max replacements per step (currently informational)

residual_value:
  initial_depreciation: 0.2 # immediate value drop at purchase
  annual_depreciation_rate: 0.04
  market_elasticity: 0.8 # sensitivity of resale value to current market price
  floor_fraction: 0.1 # minimum resale value as fraction of new price

lookahead:
  horizon: 10 # planning horizon in years
  forecast_rates:
    diesel_price_growth: 0.00
    electricity_price_growth: 0.00
    purchase_price_growth_DT: 0.00
    BET_price_gap_closure_rate: 0.00
    BET_productivity_growth: 0.00
```

> **Note:** YAML files must use spaces for indentation. Tabs cause a parse error.

### Preset configurations

Three presets are provided in `configs/`. They share all simulation parameters and differ only in the lookahead `forecast_rates`:

| Config              | `diesel_price_growth` | `BET_price_gap_closure_rate` | `BET_productivity_growth` | Description                                 |
| ------------------- | --------------------- | ---------------------------- | ------------------------- | ------------------------------------------- |
| `baseline.yaml`     | 0.00                  | 0.00                         | 0.00                      | Flat forecasts — agent expects no change    |
| `conservative.yaml` | 0.01                  | 0.02                         | 0.01                      | Modest improvement in BET economics         |
| `optimistic.yaml`   | 0.03                  | 0.05                         | 0.05                      | Rapid BET price drop and productivity gains |

The `forecast_rates` affect only the **agent's planning beliefs**, not the stochastic simulation itself.

---

## Usage

### Single episode with lookahead agent

Run one full 20-year episode with the lookahead agent and print step-by-step output:

```bash
python scripts/run_lookahead.py --config configs/baseline.yaml --seed 42
```

Output includes the action taken at each step, the resulting fleet composition, current market prices, and the annual reward breakdown.

### Batch runs

Run multiple episodes and save results for analysis:

```bash
python scripts/run_batch.py --config configs/baseline.yaml --n-episodes 100 --seed 0
```

Results are saved as pickle files in `results/batch_<config-name>_<timestamp>/`. Each file contains an `EpisodeRecord` with the full trajectory.

To run all three presets sequentially on Windows:

```bat
scripts\run_all_configs.bat
```

### Environment only (custom or random policy)

Interact with the environment directly without the lookahead agent:

```bash
python scripts/run_fleet_env.py --config configs/baseline.yaml --max-steps 5 --seed 42
```

This steps through the environment with random actions and prints fleet and market summaries, which is useful for verifying a new configuration or exploring the state space.

To integrate a custom policy in Python:

```python
from fleet_replacement.config import load_env_config
from fleet_replacement.envs.fleet_replacement import FleetReplacementEnv

config = load_env_config("configs/baseline.yaml")
env = FleetReplacementEnv(config=config)

obs, info = env.reset(seed=42)
done = False
while not done:
    action = my_policy(obs)          # your policy here
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### Direct MILP API

Build and solve the planning model without the agent wrapper:

```python
from fleet_replacement.config import load_env_config
from fleet_replacement.envs.fleet_replacement import FleetReplacementEnv
from fleet_replacement.policies.lookahead_model import (
    build_model, solve, best_immediate_actions,
    FleetReplacementData, ForecastParams, make_forecast, ModelParams,
)

config = load_env_config("configs/baseline.yaml")
env = FleetReplacementEnv(config=config)
obs, _ = env.reset(seed=0)

# Construct data and forecast from the current observation
forecast = make_forecast(obs["info_state"], config.lookahead.forecast_rates)
data = FleetReplacementData.from_obs(obs, config)

# Build, solve, extract first-period actions
model = build_model(data, forecast)
solution = solve(model)
decisions = best_immediate_actions(solution)  # list of Decision enums
```

---

## Project structure

```
fleet-replacement/
│
├── configs/
│   ├── baseline.yaml          # flat-forecast preset
│   ├── conservative.yaml      # modest BET improvement preset
│   └── optimistic.yaml        # aggressive BET improvement preset
│
├── notebooks/
│   ├── lookahead_agent.ipynb          # demo: run agent, plot fleet composition
│   ├── lookahead_model_prototyping.ipynb  # low-level MILP construction walkthrough
│   ├── analyse_episode.ipynb          # load a single EpisodeRecord, visualise
│   ├── analyse_batch.ipynb            # aggregate statistics over a batch
│   ├── compare_batches.ipynb          # compare results across config presets
│   ├── stochastic_processes_analysis.ipynb  # calibrate price process parameters
│   ├── 14_workshop_linopy.ipynb       # linopy optimisation tutorial
│   ├── linopy_model_usage.ipynb       # advanced linopy patterns
│   └── plot_helpers.py                # shared visualisation utilities
│
├── scripts/
│   ├── run_lookahead.py       # single episode, verbose output
│   ├── run_batch.py           # N-episode batch, saves EpisodeRecord pickles
│   ├── run_fleet_env.py       # environment test with random actions
│   ├── run.py                 # minimal one-step skeleton
│   └── run_all_configs.bat    # run all three presets sequentially (Windows)
│
├── src/fleet_replacement/
│   ├── config.py              # load_env_config, EnvConfig dataclasses
│   ├── episode.py             # EpisodeRecord: trajectory storage
│   │
│   ├── envs/
│   │   ├── fleet_replacement.py       # FleetReplacementEnv (main environment)
│   │   ├── grid_world.py              # auxiliary GridWorldEnv
│   │   └── models/
│   │       ├── exogenous_process.py   # stochastic price and productivity models
│   │       └── reward_model.py        # annual profit calculation
│   │
│   └── policies/
│       ├── lookahead_agent.py         # LookaheadAgent (gymnasium policy)
│       ├── lookahead_model.py         # MILP model: build_model, solve, best_immediate_actions
│       └── LookaheadModel.jl          # Julia prototype of the planning model
│
├── results/                   # batch output (created at runtime)
├── figures/                   # generated plots
├── pyproject.toml
└── README.md
```

---

## Notebooks

| Notebook                              | Purpose                                                                        |
| ------------------------------------- | ------------------------------------------------------------------------------ |
| `lookahead_agent.ipynb`               | End-to-end demo of the lookahead agent with fleet composition and profit plots |
| `lookahead_model_prototyping.ipynb`   | Step-by-step construction and inspection of the linopy MILP                    |
| `analyse_episode.ipynb`               | Load a saved `EpisodeRecord`, compute KPIs, visualise trajectory               |
| `analyse_batch.ipynb`                 | Load a batch of episodes, compute aggregate statistics and distributions       |
| `compare_batches.ipynb`               | Side-by-side comparison of results from different config presets               |
| `stochastic_processes_analysis.ipynb` | Exploratory analysis of price process calibration                              |
| `14_workshop_linopy.ipynb`            | Introduction to the linopy optimisation framework                              |
| `linopy_model_usage.ipynb`            | Advanced patterns for building the fleet MILP with linopy                      |
