function POMDPs.reward(
        mdp::FleetReplacementMDP, s::State, d::ReplacementDecisionSet, sp::State)::Float64
    sum(enumerate(sp.fleet)) do (i, vehicle)
        reward = revenue(mdp, vehicle, sp.info_state)
        reward -= energy_cost(mdp, vehicle, sp.info_state)
        reward -= salary_cost(mdp, vehicle, sp.info_state)
        reward -= interest_cost(mdp, vehicle)
        reward -= annual_depreciation(mdp, vehicle)
        if d.decisions[i] isa Keep
            reward
        elseif d.decisions[i] isa Replace
            reward += sale_result(mdp, vehicle, sp)
        else
            reward
        end
    end
end

# ----------------------------
# Revenue component
# ----------------------------

function revenue(mdp::FleetReplacementMDP, ::DieselVehicle, ::InfoState)
    return mdp.income_per_km * mdp.annual_mileage_km
end

function revenue(mdp::FleetReplacementMDP, ::ElectricVehicle, info_state::InfoState)
    return mdp.income_per_km *
           mdp.annual_mileage_km *
           info_state.productivity_BET
end

# ----------------------------
# OPEX components
# ----------------------------

function energy_cost(
        mdp::FleetReplacementMDP, ::DieselVehicle, info_state::InfoState)::Float64
    fuel_cost = mdp.annual_mileage_km *
                mdp.fuel_consumption_l_per_km *
                info_state.energy_price_diesel
    return fuel_cost
end

function energy_cost(
        mdp::FleetReplacementMDP, ::ElectricVehicle, info_state::InfoState)::Float64
    electricity_cost = info_state.productivity_BET *
                       mdp.annual_mileage_km *
                       mdp.el_consumption_kWh_per_km *
                       info_state.energy_price_electricity
    return electricity_cost
end

function salary_cost(mdp::FleetReplacementMDP, ::DieselVehicle, ::InfoState)::Float64
    return mdp.driver_salary_annual
end

function salary_cost(
        mdp::FleetReplacementMDP, ::ElectricVehicle, info_state::InfoState)::Float64
    return mdp.driver_salary_annual * info_state.productivity_BET
end

# ----------------------------
# CAPEX components
# ----------------------------

function interest_cost(mdp::FleetReplacementMDP, v::AbstractVehicle)::Float64
    return mdp.loan_fraction * purchase_price(v) * mdp.interest_rate
end

function annual_depreciation(mdp::FleetReplacementMDP, v::AbstractVehicle)::Float64
    return age(v) > mdp.economic_lifetime ? 0.0 :
           purchase_price(v) / mdp.economic_lifetime
end

function sale_result(mdp::FleetReplacementMDP, v::AbstractVehicle, s::State)::Float64
    return residual_value(mdp, v, s.info_state) -
           book_value(mdp, v)
end

function book_value(mdp::FleetReplacementMDP, v::AbstractVehicle)::Float64
    return max(0, purchase_price(v) * (1 - age(v) / mdp.economic_lifetime))
end

function residual_value(
        mdp::FleetReplacementMDP,
        v::AbstractVehicle,
        info_state::InfoState
)
    P0 = purchase_price(v) + 0.01 # Price of the current vehicle at purchase
    P_new = purchase_price(info_state, typeof(v)) # Price of a new vehicle at current time

    # Core linear piece after immediate drop
    frac = max(0.0,
        (1 - mdp.residual_initial_depreciation) -
        mdp.residual_annual_depreciation_rate * age(v))

    # Market adjustment elasticity (0=no effect, 1=full proportionality)
    market_adj = (P_new / P0)^mdp.residual_market_elasticity

    value = P0 * frac * market_adj

    # Floor at a fraction of today's new price
    floor_val = mdp.residual_floor_fraction * P0

    return max(value, floor_val)
end