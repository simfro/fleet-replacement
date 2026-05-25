"""Plotting helpers for the linopy fleet replacement notebook."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap


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
    """Heatmap of vehicle type (colour) and age (annotation) per (fleet slot, planning period)."""
    r_sol = model.solution["R"]
    active_fleet = (
        r_sol.fillna(0)
        .to_series()
        .rename("active")
        .reset_index()
        .query("active > 0.5")[["time", "slot", "vehicle_type", "vehicle_age"]]
        .reset_index(drop=True)
    )
    active_fleet["vtype_code"] = active_fleet["vehicle_type"].map(
        {vt: i for i, vt in enumerate(_VEHICLE_TYPES)}
    )
    pivot_type = active_fleet.pivot(index="slot", columns="time", values="vtype_code")
    pivot_age = active_fleet.pivot(index="slot", columns="time", values="vehicle_age")
    cmap_fleet = ListedColormap([_FLEET_COLOR_MAP[vt] for vt in _VEHICLE_TYPES])

    fig, ax = plt.subplots(
        figsize=(max(6, len(pivot_type.columns) * 0.7), max(3, len(pivot_type) * 0.6))
    )
    sns.heatmap(
        pivot_type,
        cmap=cmap_fleet,
        vmin=-0.5,
        vmax=len(_VEHICLE_TYPES) - 0.5,
        linewidths=0.5,
        linecolor="grey",
        cbar=False,
        annot=pivot_age.values,
        fmt="d",
        annot_kws={"color": "white", "fontsize": 9},
        ax=ax,
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
    ax.set_title("Fleet composition over the planning horizon  (cell value = age)")
    ax.set_xlabel("Planning period (t)")
    ax.set_ylabel("Fleet slot")
    plt.tight_layout()
    plt.show()
