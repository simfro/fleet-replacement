from fleet_replacement.config import EnvConfig

REWARD_COMPONENT_KEYS = (
    "revenue",
    "diesel_cost",
    "electricity_cost",
    "salary_cost",
    "interest_cost",
    "depreciation_cost",
    "sale_result",
    "capex",
    "opex",
    "total_reward",
)


def compute_reward(fleet, info_state, action, config: EnvConfig) -> dict[str, float]:
    """Calculate the reward for a single step.

    Parameters
    ----------
    fleet:
        Dict with keys ``is_electric``, ``age``, and ``purchase_price``.
    info_state:
        Dict with current market prices and BET productivity.
    action:
        Per-vehicle action array (0 = keep, 1 = replace DT, 2 = replace BET).
    config:
        Environment configuration.
    """
    _KEEP = 0

    fleet_size = len(fleet["is_electric"])
    revenue = 0.0
    diesel_cost = 0.0
    electricity_cost = 0.0
    interest_cost = 0.0
    depreciation_cost = 0.0
    sale_result = 0.0

    for i in range(fleet_size):
        if fleet["is_electric"][i]:
            revenue += _revenue_BET(info_state, config)
            electricity_cost += _electricity_cost(info_state, config)
        else:
            revenue += _revenue_DT(config)
            diesel_cost += _diesel_cost(info_state, config)

        interest_cost += _interest_cost(fleet["purchase_price"][i], config)
        depreciation_cost += _annual_depreciation(
            age=fleet["age"][i],
            purchase_price=fleet["purchase_price"][i],
            config=config,
        )

        if action[i] != _KEEP:
            sale_result += _sale_result(
                purchase_price=fleet["purchase_price"][i],
                age=fleet["age"][i],
                is_electric=fleet["is_electric"][i],
                info_state=info_state,
                config=config,
            )

    salary_cost = config.operational.driver_salary_annual * fleet_size

    capex = interest_cost + depreciation_cost
    opex = diesel_cost + electricity_cost + salary_cost
    total_reward = revenue - (capex + opex) + sale_result

    reward_dict = {
        "revenue": revenue,
        "diesel_cost": diesel_cost,
        "electricity_cost": electricity_cost,
        "salary_cost": salary_cost,
        "interest_cost": interest_cost,
        "depreciation_cost": depreciation_cost,
        "sale_result": sale_result,
        "capex": capex,
        "opex": opex,
        "total_reward": total_reward,
    }

    # Sanity check, reward_dict shall have exactly the keys in REWARD_COMPONENT_KEYS.
    # Update both if you add/remove components.
    assert reward_dict.keys() == set(REWARD_COMPONENT_KEYS), (
        f"compute_reward keys {set(reward_dict.keys())} do not match "
        f"REWARD_COMPONENT_KEYS {set(REWARD_COMPONENT_KEYS)}"
    )
    return reward_dict


def _sale_result(
    purchase_price, age, is_electric, info_state, config: EnvConfig
) -> float:
    return _residual_value(
        purchase_price, age, is_electric, info_state, config
    ) - _book_value(purchase_price, age, config)


def _residual_value(
    purchase_price, age, is_electric, info_state, config: EnvConfig
) -> float:
    P0 = purchase_price + 0.01  # Avoid zero division
    P_new = info_state["purchase_price_BET" if is_electric else "purchase_price_DT"]

    d0 = config.residual_value.initial_depreciation
    r = config.residual_value.annual_depreciation_rate

    # Core linear piece after immediate drop
    frac = max(0.0, (1 - d0) - r * age)

    # Market adjustment elasticity (0=none, 1=proportional)
    k = config.residual_value.market_elasticity
    market_adj = (P_new / P0) ** k

    value = P0 * frac * market_adj

    # Floor at a fraction of purchase price to avoid zero or negative residual values for old vehicles
    floor_val = config.residual_value.floor_fraction * P0

    return max(value, floor_val)


def _book_value(purchase_price, age, config: EnvConfig) -> float:
    return max(0.0, purchase_price * (1 - age / config.economic.economic_lifetime))


def _annual_depreciation(age, purchase_price, config: EnvConfig) -> float:
    if age > config.economic.economic_lifetime:
        return 0.0
    return purchase_price / config.economic.economic_lifetime


def _revenue_BET(info_state, config: EnvConfig) -> float:
    return (
        info_state["productivity_BET"]
        * config.operational.income_per_km
        * config.operational.annual_mileage_km
    )


def _revenue_DT(config: EnvConfig) -> float:
    return config.operational.income_per_km * config.operational.annual_mileage_km


def _interest_cost(purchase_price, config: EnvConfig) -> float:
    return (
        purchase_price * config.economic.loan_fraction * config.economic.interest_rate
    )


def _electricity_cost(info_state, config: EnvConfig) -> float:
    mileage = config.operational.annual_mileage_km * info_state["productivity_BET"]
    electricity_consumption = (
        mileage * config.operational.electricity_consumption_kwh_per_km
    )
    return electricity_consumption * info_state["energy_price_electricity"]


def _diesel_cost(info_state, config: EnvConfig) -> float:
    mileage = config.operational.annual_mileage_km
    fuel_consumption = mileage * config.operational.fuel_consumption_l_per_km
    return fuel_consumption * info_state["energy_price_diesel"]
