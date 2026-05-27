"""Plotting helpers for the linopy fleet replacement notebook."""

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap, to_rgb
from fleet_replacement.policies.lookahead_model import make_forecast, InfoState


def plot_forecasts(forecast, info_state, params) -> None:
    """1×3 grid: purchase prices, energy prices (twin y), BET productivity."""
    years = info_state.year + np.arange(params.horizon)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(
        years, forecast.purchase_price_DT / 1e6, marker="o", markersize=4, label="DT"
    )
    axes[0].plot(
        years,
        forecast.purchase_price_BET / 1e6,
        marker="o",
        markersize=4,
        color="tab:green",
        label="BET",
    )
    axes[0].set_title("Purchase price")
    axes[0].set_ylabel("MSEK")
    axes[0].legend()

    ax_diesel = axes[1]
    ax_elec = ax_diesel.twinx()
    (l1,) = ax_diesel.plot(
        years,
        forecast.energy_price_diesel,
        marker="o",
        markersize=4,
        color="tab:orange",
        label="Diesel",
    )
    (l2,) = ax_elec.plot(
        years,
        forecast.energy_price_electricity,
        marker="o",
        markersize=4,
        color="tab:purple",
        label="Electricity",
    )
    ax_diesel.set_title("Energy price")
    ax_diesel.set_ylabel("SEK / litre  (Diesel)", color="tab:orange")
    ax_elec.set_ylabel("SEK / kWh  (Electricity)", color="tab:purple")
    ax_diesel.tick_params(axis="y", labelcolor="tab:orange")
    ax_elec.tick_params(axis="y", labelcolor="tab:purple")
    ax_diesel.legend(handles=[l1, l2])

    axes[2].plot(
        years, forecast.productivity_BET, marker="o", markersize=4, color="tab:red"
    )
    axes[2].set_title("BET productivity (relative to DT = 1)")
    axes[2].set_ylabel("Fraction")
    axes[2].set_ylim(0, 1.05)

    for ax in [axes[0], ax_diesel, axes[2]]:
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Linear forecasts over the planning horizon", fontsize=13)
    plt.tight_layout()
    plt.show()


_DECISIONS = ["Keep", "Replace_DT", "Replace_BET"]
_DECISION_COLOR_MAP = {"Keep": "white", "Replace_DT": "black", "Replace_BET": "green"}


def plot_decisions(active_decisions) -> None:
    """Heatmap of optimal decisions per (fleet slot, planning period)."""
    df = active_decisions.copy()
    df["decision_code"] = df["decision"].map({d: i for i, d in enumerate(_DECISIONS)})
    pivot = df.pivot(index="slot", columns="time", values="decision_code")
    cmap = ListedColormap([_DECISION_COLOR_MAP[d] for d in _DECISIONS])

    fig, ax = plt.subplots(
        figsize=(max(6, len(pivot.columns) * 0.7), max(3, len(pivot) * 0.6))
    )
    sns.heatmap(
        pivot,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(_DECISIONS) - 0.5,
        linewidths=0.5,
        linecolor="grey",
        cbar=False,
        ax=ax,
    )
    patches = [
        mpatches.Patch(facecolor=_DECISION_COLOR_MAP[d], edgecolor="grey", label=d)
        for d in _DECISIONS
    ]
    ax.legend(
        handles=patches, bbox_to_anchor=(1.01, 1), loc="upper left", title="Decision"
    )
    ax.set_title("Optimal decisions per fleet slot over the planning horizon")
    ax.set_xlabel("Planning period (t)")
    ax.set_ylabel("Fleet slot")
    plt.tight_layout()
    plt.show()


_VEHICLE_TYPES = ["DT", "BET"]
_FLEET_COLOR_MAP = {"DT": "black", "BET": "green"}


def plot_fleet_composition(model) -> None:
    """Heatmap of vehicle type and age per (fleet slot, planning period).

    Colour indicates vehicle type; opacity fades with age so newer vehicles
    appear vivid and older ones more transparent.
    """
    r_sol = model.solution["R"]
    active_fleet = (
        r_sol.fillna(0)
        .to_series()
        .rename("active")
        .reset_index()
        .query("active > 0.5")[["time", "slot", "vehicle_type", "vehicle_age"]]
        .reset_index(drop=True)
    )

    pivot_type = active_fleet.pivot(index="slot", columns="time", values="vehicle_type")
    pivot_age = active_fleet.pivot(
        index="slot", columns="time", values="vehicle_age"
    ).astype(int)

    n_slots, n_times = pivot_type.shape
    times = pivot_type.columns.tolist()
    slots = pivot_type.index.tolist()

    # RGB triples derived from the named colors in _FLEET_COLOR_MAP
    _rgb = {vt: to_rgb(_FLEET_COLOR_MAP[vt]) for vt in _VEHICLE_TYPES}

    # Alpha: age 0 → fully opaque, oldest vehicle → alpha_min (still distinguishable)
    alpha_min, alpha_max = 0.35, 1.0
    max_age = max(int(pivot_age.values.max()), 1)

    # Build RGBA image array (rows = fleet slots, columns = planning periods)
    rgba = np.ones((n_slots, n_times, 4))  # default: opaque white
    for i in range(n_slots):
        for j in range(n_times):
            vtype = str(pivot_type.iloc[i, j])
            age = int(pivot_age.iloc[i, j])
            rgba[i, j, :3] = _rgb.get(vtype, (1.0, 1.0, 1.0))
            rgba[i, j, 3] = alpha_max - (age / max_age) * (alpha_max - alpha_min)

    fig, ax = plt.subplots(figsize=(max(6, n_times * 0.7), max(3, n_slots * 0.6)))
    ax.set_facecolor("white")
    ax.imshow(rgba, aspect="auto", extent=(-0.5, n_times - 0.5, n_slots - 0.5, -0.5))

    # Age annotations
    for i in range(n_slots):
        for j in range(n_times):
            ax.text(
                j,
                i,
                str(int(pivot_age.iloc[i, j])),
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )

    # Cell grid lines drawn on top of the image
    for x in np.arange(-0.5, n_times, 1):
        ax.axvline(x, color="grey", linewidth=0.5)
    for y in np.arange(-0.5, n_slots, 1):
        ax.axhline(y, color="grey", linewidth=0.5)

    ax.set_xticks(range(n_times))
    ax.set_xticklabels(times)
    ax.set_yticks(range(n_slots))
    ax.set_yticklabels(slots)
    ax.set_xlabel("Planning period (t)")
    ax.set_ylabel("Fleet slot")
    ax.set_title("Fleet composition over the planning horizon  (cell value = age)")

    patches = [
        mpatches.Patch(facecolor=_FLEET_COLOR_MAP[vt], edgecolor="grey", label=vt)
        for vt in _VEHICLE_TYPES
    ]
    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        title="Vehicle type",
    )
    plt.tight_layout()
    plt.show()


def plot_episode_fleet_composition(record) -> None:
    """Heatmap of vehicle type and age per (fleet slot, simulation year).

    Equivalent to :func:`plot_fleet_composition` but takes an
    ``EpisodeRecord`` from the stochastic simulation instead of a linopy
    model solution.  Colour indicates vehicle type (DT = dark, BET = green);
    opacity fades with age so newer vehicles appear vivid.  The cell text
    shows the vehicle age.

    Parameters
    ----------
    record : EpisodeRecord
        Episode history collected by ``EpisodeRecorder``.
    """
    # (n_steps, fleet_size) — observations at the START of each step
    is_electric = record.fleet_is_electric  # 0 = DT, 1 = BET
    ages = record.fleet_ages  # integer vehicle ages
    years = record.years  # simulation year per step

    n_steps, n_slots = is_electric.shape

    _rgb = {vt: to_rgb(_FLEET_COLOR_MAP[vt]) for vt in _VEHICLE_TYPES}
    alpha_min, alpha_max = 0.35, 1.0
    max_age = max(int(ages.max()), 1)

    # Build RGBA image: rows = fleet slots, columns = simulation years
    rgba = np.ones((n_slots, n_steps, 4))
    for j in range(n_steps):
        for i in range(n_slots):
            vtype = "BET" if int(is_electric[j, i]) == 1 else "DT"
            age = int(ages[j, i])
            rgba[i, j, :3] = _rgb[vtype]
            rgba[i, j, 3] = alpha_max - (age / max_age) * (alpha_max - alpha_min)

    # Thin out x-tick labels when the episode is long so they don't overlap.
    tick_step = max(1, n_steps // 20)

    fig, ax = plt.subplots(figsize=(max(8, n_steps * 0.35), max(3, n_slots * 0.6)))
    ax.set_facecolor("white")
    ax.imshow(rgba, aspect="auto", extent=(-0.5, n_steps - 0.5, n_slots - 0.5, -0.5))

    # Age annotations (skip when there are many columns to keep the chart readable)
    if n_steps <= 40:
        for j in range(n_steps):
            for i in range(n_slots):
                ax.text(
                    j,
                    i,
                    str(int(ages[j, i])),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

    # Cell grid lines
    for x in np.arange(-0.5, n_steps, 1):
        ax.axvline(x, color="grey", linewidth=0.5)
    for y in np.arange(-0.5, n_slots, 1):
        ax.axhline(y, color="grey", linewidth=0.5)

    ax.set_xticks(range(0, n_steps, tick_step))
    ax.set_xticklabels(years[::tick_step], rotation=45, ha="right")
    ax.set_yticks(range(n_slots))
    ax.set_yticklabels(range(n_slots))
    ax.set_xlabel("Simulation year")
    ax.set_ylabel("Fleet slot")
    ax.set_title(
        f"Fleet composition over the episode  (seed={record.seed}, "
        f"cell value = age)",
    )

    patches = [
        mpatches.Patch(facecolor=_FLEET_COLOR_MAP[vt], edgecolor="grey", label=vt)
        for vt in _VEHICLE_TYPES
    ]
    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        title="Vehicle type",
    )
    plt.tight_layout()
    plt.show()


def plot_episode_information_state(record, agent, forecast_step: int = 1) -> None:
    """1x3 figure: purchase prices, energy prices (twin y), BET productivity.

    Thick lines show the realised information-state trajectory from *record*.
    Thin semi-transparent lines show the agent's linear forecast computed from
    every ``forecast_step``-th observation, illustrating how the agent's beliefs
    about future prices evolved over the episode.

    Parameters
    ----------
    record : EpisodeRecord
        Episode history collected by ``EpisodeRecorder``.
    agent : LookaheadAgent
        The agent used during the episode; provides ``horizon``,
        ``model_params.BET_productivity_max``, and ``forecast_params``.
    forecast_step : int
        Draw a forecast fan starting every this many steps (default 1 = every step).
    """
    years = record.years
    n_steps = len(years)
    horizon = agent.horizon

    # Reconstruct forecasts at every forecast_step-th observed info state.
    _fc_kw = dict(alpha=0.2, linewidth=0.9, zorder=1)
    forecasts = []
    for i in range(0, n_steps, forecast_step):
        s = record.steps[i]
        info_state = InfoState(
            year=int(s.year),
            energy_price_diesel=float(s.energy_price_diesel),
            energy_price_electricity=float(s.energy_price_electricity),
            purchase_price_DT=float(s.purchase_price_DT),
            purchase_price_BET=float(s.purchase_price_BET),
            productivity_BET=float(s.productivity_BET),
        )
        fc = make_forecast(
            info_state,
            horizon,
            agent.model_params.BET_productivity_max,
            agent.forecast_params,
        )
        fc_years = s.year + np.arange(horizon)
        forecasts.append((fc_years, fc))

    purchase_dt = record.purchase_prices_DT / 1e6
    purchase_bet = record.purchase_prices_BET / 1e6
    energy_diesel = record.energy_prices_diesel
    energy_elec = record.energy_prices_electricity
    productivity = record.productivities_BET

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _proxy_label = f"Forecast (every {forecast_step} steps)"

    # --- Panel 0: Purchase prices ---
    ax = axes[0]
    for fc_years, fc in forecasts:
        ax.plot(fc_years, fc.purchase_price_DT / 1e6, color="tab:gray", **_fc_kw)
        ax.plot(fc_years, fc.purchase_price_BET / 1e6, color="tab:green", **_fc_kw)
    (l_dt,) = ax.plot(
        years,
        purchase_dt,
        color="tab:gray",
        linewidth=2,
        marker="o",
        markersize=3,
        zorder=3,
        label="DT",
    )
    (l_bet,) = ax.plot(
        years,
        purchase_bet,
        color="tab:green",
        linewidth=2,
        marker="o",
        markersize=3,
        zorder=3,
        label="BET",
    )
    fc_proxy = mlines.Line2D(
        [], [], color="grey", linewidth=0.9, alpha=0.7, label=_proxy_label
    )
    ax.legend(handles=[l_dt, l_bet, fc_proxy])
    ax.set_title("Purchase price")
    ax.set_ylabel("MSEK")
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.3)

    # --- Panel 1: Energy prices (dual y-axis) ---
    ax_d = axes[1]
    ax_e = ax_d.twinx()
    for fc_years, fc in forecasts:
        ax_d.plot(fc_years, fc.energy_price_diesel, color="tab:orange", **_fc_kw)
        ax_e.plot(fc_years, fc.energy_price_electricity, color="tab:purple", **_fc_kw)
    (l_diesel,) = ax_d.plot(
        years,
        energy_diesel,
        color="tab:orange",
        linewidth=2,
        marker="o",
        markersize=3,
        zorder=3,
        label="Diesel",
    )
    (l_elec,) = ax_e.plot(
        years,
        energy_elec,
        color="tab:purple",
        linewidth=2,
        marker="o",
        markersize=3,
        zorder=3,
        label="Electricity",
    )
    ax_d.set_ylabel("SEK / litre  (Diesel)", color="tab:orange")
    ax_e.set_ylabel("SEK / kWh  (Electricity)", color="tab:purple")
    ax_d.tick_params(axis="y", labelcolor="tab:orange")
    ax_e.tick_params(axis="y", labelcolor="tab:purple")
    fc_proxy2 = mlines.Line2D(
        [], [], color="grey", linewidth=0.9, alpha=0.7, label=_proxy_label
    )
    ax_d.legend(handles=[l_diesel, l_elec, fc_proxy2])
    ax_d.set_title("Energy price")
    ax_d.set_xlabel("Year")
    ax_d.grid(True, alpha=0.3)

    # --- Panel 2: BET productivity ---
    ax = axes[2]
    for fc_years, fc in forecasts:
        ax.plot(fc_years, fc.productivity_BET, color="tab:red", **_fc_kw)
    (l_prod,) = ax.plot(
        years,
        productivity,
        color="tab:red",
        linewidth=2,
        marker="o",
        markersize=3,
        zorder=3,
        label="Actual",
    )
    fc_proxy3 = mlines.Line2D(
        [], [], color="tab:red", linewidth=0.9, alpha=0.7, label=_proxy_label
    )
    ax.legend(handles=[l_prod, fc_proxy3])
    ax.set_title("BET productivity (relative to DT = 1)")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Information state evolution  (seed={record.seed})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()
