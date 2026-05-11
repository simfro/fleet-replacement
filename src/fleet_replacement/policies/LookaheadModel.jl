module LookaheadModel

import JuMP
import HiGHS
import InteractiveUtils: subtypes
using AutoHashEquals
using Plots

export create_fleet_replacement_data, solve_fleet_replacement, metric_share_of_BET

"""
    AbstractVehicle
Parent type for all vehicles in the fleet replacement problem.
"""
abstract type AbstractVehicle end

@auto_hash_equals struct DT <: AbstractVehicle
    age::Int
end
Base.show(io::IO, dt::DT) = print(io, "DT(", dt.age, ")")
Base.show(io::IO, ::Type{DT}) = print(io, "DT")

@auto_hash_equals struct BET <: AbstractVehicle
    age::Int
end
Base.show(io::IO, bet::BET) = print(io, "BET(", bet.age, ")")
Base.show(io::IO, ::Type{BET}) = print(io, "BET")

"""
    AbstractDecision
Parent type for decisions in the fleet replacement problem.
"""
abstract type AbstractDecision end

"""
    Keep <: AbstractDecision
A decision to keep the current vehicle.
"""
@auto_hash_equals struct Keep <: AbstractDecision end
Base.show(io::IO, ::Keep) = print(io, "Keep")

"""
    Replace{v <: AbstractVehicle} <: AbstractDecision
A decision to replace a vehicle with a new vehicle of type `v`.

"""
struct Replace{v<:AbstractVehicle} <: AbstractDecision end
Base.:(==)(::Replace{T}, ::Replace{T}) where {T} = true
Base.:(==)(::Replace{T1}, ::Replace{T2}) where {T1,T2} = false
Base.hash(::Replace{T}, h::UInt) where {T} = hash(T, h)
Base.show(io::IO, ::Replace{T}) where {T} = print(io, "Replace{", T, "}")

"""
    FleetReplacementParams

Parameters for the fleet replacement problem.
"""
@kwdef struct FleetReplacementParams
    horizon::Int
    max_age::Int
    # Operational parameters
    income_per_km::Int
    annual_mileage_km::Int
    fuel_consumption_l_per_km::Float64
    el_consumption_kWh_per_km::Float64
    # average_speed::Int
    driver_salary_annual::Int
    # Forecast parameters
    diesel_price_growth::Float64
    electricity_price_growth::Float64
    purchase_price_growth_DT::Float64
    purchase_price_growth_BET::Float64
    BET_productivity_max::Float64
    BET_productivity_growth::Float64
    # Financial parameters
    interest_rate::Float64
    loan_fraction::Float64
    economic_lifetime::Int
    # Discount factor
    γ::Float64
end

struct InfoState
    year::Int
    energy_price_diesel::Float64
    energy_price_electricity::Float64
    purchase_price_DT::Float64
    purchase_price_BET::Float64
    productivity_BET::Float64
    initial_fleet_purchase_prices::Vector{Float64}
end

"""
    FleetReplacementData

Data structure that hold the index sets for the fleet replacement problem.
"""
struct FleetReplacementData
    params::FleetReplacementParams              # Model parameters
    initial_fleet::Vector{AbstractVehicle}      # Initial fleet composition
    info_state::InfoState                       # Initial info state
    time_span::UnitRange{Int}                   # Time periods
    fleet_slots::UnitRange{Int}                 # Fleet slots
    vehicles::Vector{AbstractVehicle}           # All possible vehicles
    decisions::Vector{AbstractDecision}         # All possible decisions
end

"""
    create_fleet_replacement_data(params::FleetReplacementParams)

Create FleetReplacementData from parameters, automatically generating vehicles and decisions.
"""
function create_fleet_replacement_data(
    params::FleetReplacementParams,
    initial_fleet::Vector{<:AbstractVehicle},
    info_state::InfoState,
)
    # Create time span for the planning horizon
    time_span = 0:params.horizon-1

    # Create fleet slots
    fleet_slots = 1:length(initial_fleet)

    # Generate all vehicles of all types for ages 0 to max_age
    vehicles = AbstractVehicle[]
    for VehicleType in subtypes(AbstractVehicle)
        for age = 0:params.max_age
            push!(vehicles, VehicleType(age))
        end
    end

    # Generate decisions: Keep + Replace with each vehicle type
    decisions = AbstractDecision[Keep()]
    for VehicleType in subtypes(AbstractVehicle)
        push!(decisions, Replace{VehicleType}())
    end

    return FleetReplacementData(
        params,
        initial_fleet,
        info_state,
        time_span,
        fleet_slots,
        vehicles,
        decisions,
    )
end

# --- FORECAST FUNCTIONS ---

# Energy price functions
energy_price(::DT, t, data) =
    data.info_state.energy_price_diesel * (1 + data.params.diesel_price_growth * t)
energy_price(::BET, t, data) =
    data.info_state.energy_price_electricity *
    (1 + data.params.electricity_price_growth * t)

# Purchase price functions. Projection
purchase_price(t::Int, ::Type{DT}, data) = max(
    data.info_state.purchase_price_DT / 3, # Never below one third of initial price
    data.info_state.purchase_price_DT * (1 + data.params.purchase_price_growth_DT * t),
)
function purchase_price(t::Int, v::DT, slot::Int, data)
    if v.age < t # if vehicle is purchased during the planning horizon
        return purchase_price(t, DT, data)
    else # if vehicle is already in the fleet at the start of the planning horizon
        #Historical purchase prices from simulation must be forwarded to lookaheadmodel.
        return data.info_state.initial_fleet_purchase_prices[slot]
    end
end
purchase_price(t::Int, ::Type{BET}, data) = max(
    data.info_state.purchase_price_BET / 3, # Never below one third of initial price
    data.info_state.purchase_price_BET * (1 + data.params.purchase_price_growth_BET * t),
)
function purchase_price(t::Int, v::BET, slot::Int, data)
    if v.age < t # if vehicle is purchased during the planning horizon
        return purchase_price(t, BET, data)
    else # if vehicle is already in the fleet at the start of the planning horizon
        return data.info_state.initial_fleet_purchase_prices[slot]
    end
end

function productivity(::BET, t, data)::Float64
    # planning_year = data.info_state.year
    # ρ_start = data.info_state.productivity_BET
    # ρ_max = data.params.BET_productivity_max
    # k = data.params.BET_productivity_k
    # t₀ = data.params.BET_productivity_t₀

    # base = 1 / (1 + exp(-k * ((t + planning_year) - t₀)))
    # ρ = ρ_start + (ρ_max - ρ_start) * base
    ρ = data.info_state.productivity_BET * (1 + data.params.BET_productivity_growth * t)
    # Cap productivity between reasonable bounds (e.g., 10% minimum, max specified)
    ρ_min = 0.1  # Minimum 10% productivity 
    ρ = clamp(ρ, ρ_min, data.params.BET_productivity_max)
    return ρ
end
productivity(::DT, t, data)::Float64 = 1.0

# ----------------------------------------------

"""
    add_fleet_variables(model, data)

Add binary variables R[t, slot, v] to the JuMP model indicating whether vehicle v is in 
fleet slot `slot` at time t.
"""
function add_fleet_variables(model, data)
    JuMP.@variable(model, R[data.time_span, data.fleet_slots, data.vehicles], Bin)
end

"""
    add_decision_variables(model, data)

Add binary variables x[t, slot, v, d] to the JuMP model indicating whether decision d 
is taken for vehicle v in fleet slot `slot` at time t.
"""
function add_decision_variables(model, data)
    JuMP.@variable(
        model,
        x[data.time_span[1:end-1], data.fleet_slots, data.vehicles, data.decisions],
        Bin
    )
end

"""
    add_initial_fleet_composition(model, data)

Add constraints to set the initial fleet composition in the optimization model.
"""
function add_initial_fleet_composition(model, data)
    R = model[:R]

    JuMP.@constraint(
        model,
        con_initial_fleet_composition[slot in data.fleet_slots],
        R[data.time_span[1], slot, data.initial_fleet[slot]] == 1
    )
end

"""
    add_variable_feasibility_constraints(model, data)

Adds feasibility constraints to the optimization model to ensure valid variable assignments 
for fleet replacement decisions:

1. Ensures that each fleet slot has exactly one vehicle assigned at each time period.
2. Enforces that each fleet slot has exactly one decision per vehicle per time period.
3. Prevents decisions from being made for vehicles that are not present in the fleet slot at
    a given time.

These constraints guarantee the logical consistency of vehicle assignments and decision 
variables throughout the planning horizon.
"""
function add_variable_feasibility_constraints(model, data)
    R = model[:R]
    x = model[:x]

    # Each fleet slot must have exactly one vehicle at each time period
    JuMP.@constraint(
        model,
        con_single_vehicle_per_fleet_slot_and_time_period[
            t in data.time_span,
            slot in data.fleet_slots,
        ],
        sum(R[t, slot, v] for v in data.vehicles) == 1
    )

    # Each fleet slot must have exactly one decision per vehicle per time period
    JuMP.@constraint(
        model,
        con_single_decision_per_fleet_slot_and_time_period[
            t in data.time_span[1:end-1],
            slot in data.fleet_slots,
        ],
        sum(x[t, slot, v, d] for v in data.vehicles, d in data.decisions) == 1
    )

    # No decision can be made for a vehicle that is not in the fleet slot
    JuMP.@constraint(
        model,
        con_no_decision_for_nonexistent_vehicle[
            t in data.time_span[1:end-1],
            slot in data.fleet_slots,
            v in data.vehicles,
            d in data.decisions,
        ],
        !R[t, slot, v] --> {x[t, slot, v, d] == 0}
    )
end

"""
    add_replace_at_max_age_constraints(model, data)
Add constraints to ensure that vehicles are replaced when they reach their maximum age.
Implemented by not allowing the "Keep" decision for vehicles at their maximum age.
"""
function add_replace_at_max_age_constraints(model, data)
    R = model[:R]
    x = model[:x]
    max_age = data.params.max_age
    max_age_vehicles = [v for v in data.vehicles if v.age == max_age]

    JuMP.@constraint(
        model,
        con_replace_at_max_age[
            t in data.time_span[1:end-1],
            slot in data.fleet_slots,
            v in max_age_vehicles,
        ],
        x[t, slot, v, Keep()] == 0
    )
end

"""
    add_transition_function_constraints(model, data)

Adds transition function constraints to the given JuMP model for fleet replacement decisions.

# Arguments
- `model`: A JuMP model containing decision variables and parameters for the fleet replacement problem.
- `data`: A data structure holding information about vehicles, time periods, fleet slots, and model parameters.

# Description
This function enforces two types of transition constraints:
- **Keep-transitions**: If a vehicle is kept in a fleet slot, its age increases by one in the next period, provided it has not reached the maximum allowed age.
- **Replacement-transitions**: If a vehicle is replaced, the fleet slot receives a new vehicle of the chosen type at age zero in the next period.

The constraints ensure the correct evolution of the fleet composition over time according to the decisions made.
"""
function add_transition_function_constraints(model, data)
    R = model[:R]
    x = model[:x]

    # Keep-transitions: If we keep a vehicle, it ages by 1 in the next period
    non_max_age_vehicles = [v for v in data.vehicles if v.age < data.params.max_age]

    JuMP.@constraint(
        model,
        con_keep_transitions[
            t in data.time_span[1:end-1],
            slot in data.fleet_slots,
            v in non_max_age_vehicles,
        ],
        x[t, slot, v, Keep()] --> {R[t+1, slot, typeof(v)(v.age + 1)] == 1}
    )

    # Replacement-transitions: If we replace a vehicle, the slot gets a new vehicle of the chosen type at age 0

    JuMP.@constraint(
        model,
        con_replace_transitions[
            t in data.time_span[1:end-1],
            slot in data.fleet_slots,
            v in data.vehicles,
            VehicleType in subtypes(AbstractVehicle),
        ],
        x[t, slot, v, Replace{VehicleType}()] --> {R[t+1, slot, VehicleType(0)] == 1}
    )
end

"""
    add_objective_term_revenue(model, data)

Add the discounted revenue objective term to the model. Revenue is calculated as 
income_per_km × annual_mileage_km × γ^t for each vehicle assignment.
"""
function add_objective_term_revenue(model, data)
    R = model[:R]
    income_per_km = data.params.income_per_km
    annual_mileage_km = data.params.annual_mileage_km
    γ = data.params.γ

    JuMP.@expression(
        model,
        obj_revenue,
        income_per_km *
        annual_mileage_km *
        sum(
            γ^t * R[t, slot, v] * productivity(v, t, data) for t in data.time_span,
            slot in data.fleet_slots, v in data.vehicles
        )
    )
end

function add_objective_term_energy_cost(model, data)
    energy_consumption(::DT, data)::Float64 = data.params.fuel_consumption_l_per_km
    energy_consumption(::BET, data)::Float64 = data.params.el_consumption_kWh_per_km
    R = model[:R]
    annual_mileage_km = data.params.annual_mileage_km
    γ = data.params.γ

    JuMP.@expression(
        model,
        obj_energy_cost,
        annual_mileage_km * sum(
            γ^t *
            R[t, slot, v] *
            productivity(v, t, data) *
            energy_consumption(v, data) *
            energy_price(v, t, data) for t in data.time_span, slot in data.fleet_slots,
            v in data.vehicles
        )
    )
end

function add_objective_term_salary_cost(model, data)
    driver_salary_annual = data.params.driver_salary_annual
    γ = data.params.γ
    R = model[:R]

    JuMP.@expression(
        model,
        obj_salary_cost,
        driver_salary_annual * sum(
            γ^t * R[t, slot, v] * productivity(v, t, data) for t in data.time_span,
            slot in data.fleet_slots, v in data.vehicles
        )
    )
end

function add_objective_term_interest_cost(model, data)
    loan_fraction = data.params.loan_fraction
    interest_rate = data.params.interest_rate
    γ = data.params.γ
    R = model[:R]

    # vehicles within economic lifetime incur interest cost
    vehicles_within_economic_lifetime =
        [v for v in data.vehicles if v.age < data.params.economic_lifetime]

    JuMP.@expression(
        model,
        obj_interest_cost,
        loan_fraction *
        interest_rate *
        sum(
            γ^t * R[t, slot, v] * purchase_price(t, v, slot, data) for t in data.time_span,
            slot in data.fleet_slots, v in vehicles_within_economic_lifetime
        )
    )
end

function add_objective_term_depreciation_cost(model, data)
    R = model[:R]
    economic_lifetime = data.params.economic_lifetime
    γ = data.params.γ

    vehicles_within_economic_lifetime =
        [v for v in data.vehicles if v.age < data.params.economic_lifetime]

    JuMP.@expression(
        model,
        obj_depreciation_cost,
        sum(
            γ^t * R[t, slot, v] * (purchase_price(t, v, slot, data) / economic_lifetime) for
            t in data.time_span, slot in data.fleet_slots,
            v in vehicles_within_economic_lifetime
        )
    )
end

# ---------------------------------
#           SALE RESULT
# ---------------------------------

# Book value for a vehicle of a specific type and age
book_value(t::Int, v::AbstractVehicle, slot::Int, data) =
    max(0, purchase_price(t, v, slot, data) * (1 - v.age / data.params.economic_lifetime))

function residual_value(
    t::Int,
    v::AbstractVehicle,
    slot::Int,
    data,
    # TODO: Make these parameters instead of hardcoded defaults 
    d0::Real=0.20,
    r::Real=0.04,
    k::Real=1.0,
    floor_frac::Real=0.1,
)
    P0 = purchase_price(t, v, slot, data) + 0.01 # Price of the current vehicle at purchase
    P_new = purchase_price(t, v, slot, data) # Price of a new vehicle at current time

    # Core linear piece after immediate drop
    frac = max(0.0, (1 - d0) - r * v.age)

    # Market adjustment elasticity (0=no effect, 1=full proportionality)
    market_adj = (P_new / P0)^k

    value = P0 * frac * market_adj

    # Floor at a fraction of today's new price
    floor_val = floor_frac * P0

    return max(value, floor_val)
end

function add_objective_term_sale_result(model, data)
    x = model[:x]
    γ = data.params.γ

    # Get all replacement decisions
    replacement_decisions = [d for d in data.decisions if d isa Replace]

    JuMP.@expression(
        model,
        obj_sale_result,
        sum(
            γ^t *
            x[t, slot, v, d] *
            (residual_value(t, v, slot, data) - book_value(t, v, slot, data)) for
            t in data.time_span[1:end-1], slot in data.fleet_slots, v in data.vehicles,
            d in replacement_decisions
        )
    )
end

"""
    add_objective_term_terminal_year_sale(model, data)

Add the terminal year sale objective term to the model. This accounts for the residual value
of vehicles remaining in the fleet at the end of the planning horizon.
"""
function add_objective_term_terminal_year_sale(model, data)
    # TODO: Consider if this should be an constraint instead. Would require new "sell" decission
    R = model[:R]
    γ = data.params.γ

    JuMP.@expression(
        model,
        obj_terminal_year_sale,
        sum(
            γ^data.time_span[end] *
            R[data.time_span[end], slot, v] *
            (
                residual_value(data.time_span[end], v, slot, data) -
                book_value(data.time_span[end], v, slot, data)
            ) for slot in data.fleet_slots, v in data.vehicles
        ),
    )
end

function construct_objective(model)
    JuMP.@objective(
        model,
        Max,
        model[:obj_revenue] - model[:obj_energy_cost] - model[:obj_salary_cost] -
        model[:obj_interest_cost] - model[:obj_depreciation_cost] +
        model[:obj_sale_result] +
        model[:obj_terminal_year_sale]
    )
end

# --------------------------------
#        PLOT RESULTS
# --------------------------------
function plot_energy_price_trends(data)
    years = data.time_span
    diesel_prices = [energy_price(DT(0), t, data) for t in years]
    electricity_prices = [energy_price(BET(0), t, data) for t in years]

    plt = Plots.plot(
        years,
        diesel_prices,
        label="Diesel Price",
        xlabel="Year",
        ylabel="Price (SEK/l)",
        title="Fuel Price Trends",
        legend=:topleft,
        lw=2,
    )

    Plots.plot!(
        twinx(),
        years,
        electricity_prices,
        ylabel="Price (SEK/kWh)",
        label="Electricity Price",
        lw=2,
        color=:red,
        linestyle=:dash,
    )

    return plt
end
function plot_purchase_price(data)
    years = data.time_span
    DT_prices = [purchase_price(t, DT, data) for t in years]
    BET_prices = [purchase_price(t, BET, data) for t in years]

    plt = Plots.plot(
        years,
        [DT_prices BET_prices],
        label=["Diesel" "Electric"],
        xlabel="Year",
        ylabel="Price (SEK)",
        title="Vehicle Purchase Price Trends",
        legend=:topleft,
        lw=2,
    )
    return plt
end
function plot_BET_productivity(data)
    years = data.time_span
    productivity_BET = [productivity(BET(0), t, data) for t in years]

    plt = Plots.plot(
        years,
        productivity_BET,
        label="BET Productivity",
        xlabel="Year",
        ylabel="Productivity",
        title="BET Productivity Over Time",
        legend=:topleft,
        lw=2,
    )
    return plt
end

function plot_replacements(model, data)
    replacement_to_int(x::Keep) = 1
    replacement_to_int(x::Replace{DT}) = 2
    replacement_to_int(x::Replace{BET}) = 3

    x = model[:x]
    x_sol = JuMP.value.(x)

    n_vehicles = length(data.fleet_slots)

    T = length(data.time_span) - 1
    replacement_matrix = zeros(Int, n_vehicles, T)

    for (t_idx, t) in enumerate(data.time_span[1:end-1])
        for slot in data.fleet_slots, v in data.vehicles, d in data.decisions
            if x_sol[t, slot, v, d] > 0.5
                replacement_matrix[slot, t_idx] = replacement_to_int(d)  # Convert decision to int
            end
        end
    end

    years = collect(data.time_span[1:end-1])  # Years for the decision periods

    cmap = [:white, :black, :green]  # white=no decision, black=Keep, green=Replace
    plt = plot()
    heatmap!(
        plt,
        years,
        1:n_vehicles,
        replacement_matrix,
        xlabel="Year",
        ylabel="Vehicle Index",
        title="Replacement decisions",
        yticks=1:n_vehicles,
        xticks=years,
        color=cmap,
        clims=(1, 3),
        colorbar=false,
        yflip=true,
        legend=false,
    )

    vline!(plt, years[1]-0.5:1:(years[end]+0.5), c=:gray, ls=:dash, label=false)
    hline!(plt, 0.5:1:(n_vehicles+0.5), c=:gray, ls=:dash, label=false)

    return plt
end

function plot_fleet(model, data)
    R = model[:R]
    R_sol = JuMP.value.(R)

    # Extract fleet state matrix
    n_vehicles = length(data.fleet_slots)
    T = length(data.time_span)
    years = collect(data.time_span)

    # Create matrix to store vehicle types and ages
    fleet_matrix = zeros(Int, n_vehicles, T)
    age_matrix = zeros(Int, n_vehicles, T)

    # Map vehicle types to integers
    vehicle_to_int(v::DT) = 1
    vehicle_to_int(v::BET) = 2

    # Fill the matrices
    for (t_idx, t) in enumerate(data.time_span), slot = 1:n_vehicles
        for v in data.vehicles
            if R_sol[t, slot, v] > 0.5
                fleet_matrix[slot, t_idx] = vehicle_to_int(v)
                age_matrix[slot, t_idx] = v.age
                break
            end
        end
    end

    cmap = [:white, :black, :green]  # white=empty, black=DT, green=BET
    plt = plot()
    heatmap!(
        plt,
        years,
        1:n_vehicles,
        fleet_matrix,
        xlabel="Year",
        ylabel="Vehicle Index",
        title="Fleet Composition",
        yticks=1:n_vehicles,
        xticks=years,
        color=cmap,
        clims=(0, 2),
        colorbar=false,
        yflip=true,
        legend=false,
    )

    # Add age annotations on each cell
    for (t_idx, t) in enumerate(years), slot = 1:n_vehicles
        age = age_matrix[slot, t_idx]
        if fleet_matrix[slot, t_idx] > 0  # Only annotate if there's a vehicle
            annotate!(plt, t, slot, text("$age", 8, :white, :center))
        end
    end

    vline!(plt, years[1]-0.5:1:(years[end]+0.5), c=:gray, ls=:dash, label=false)
    hline!(plt, 0.5:1:(n_vehicles+0.5), c=:gray, ls=:dash, label=false)

    return plt
end

function plot_results(model, data)
    plt1 = plot_energy_price_trends(data)
    plt2 = plot_purchase_price(data)
    plt3 = plot_BET_productivity(data)
    plt4 = plot_fleet(model, data)
    plt5 = plot_replacements(model, data)

    return plot(
        plt1,
        plt2,
        plt3,
        plt4,
        plt5,
        layout=@layout([[a; b; c] [d; e]]),
        size=(900, 800),
    )
end

function metric_share_of_BET(model, data)
    R = model[:R]
    R_sol = JuMP.value.(R)
    n_vehicles = length(data.fleet_slots)
    T = length(data.time_span)
    shares = Float64[]
    for t in data.time_span
        total_BET = sum(
            R_sol[t, slot, v] for slot in data.fleet_slots, v in data.vehicles if v isa BET;
            init=0.0,
        )
        push!(shares, total_BET / n_vehicles)
    end
    return shares
end

# return the best immediate action for each vehicle in the initial fleet
function best_immediate_actions(model, data)
    x = model[:x]
    x_sol = JuMP.value.(x)
    n_vehicles = length(data.fleet_slots)
    best_actions = Vector{AbstractDecision}(undef, n_vehicles)
    t = data.time_span[1] # first time period
    for slot in data.fleet_slots, v in data.vehicles, d in data.decisions
        if x_sol[t, slot, v, d] > 0.5
            best_actions[slot] = d
        end
    end
    return best_actions
end

function solve_fleet_replacement(data)
    model = JuMP.Model(HiGHS.Optimizer)
    add_fleet_variables(model, data)
    add_decision_variables(model, data)
    add_initial_fleet_composition(model, data)
    add_variable_feasibility_constraints(model, data)
    add_replace_at_max_age_constraints(model, data)
    add_transition_function_constraints(model, data)
    add_objective_term_revenue(model, data)
    add_objective_term_energy_cost(model, data)
    add_objective_term_salary_cost(model, data)
    add_objective_term_interest_cost(model, data)
    add_objective_term_depreciation_cost(model, data)
    add_objective_term_sale_result(model, data)
    add_objective_term_terminal_year_sale(model, data)
    construct_objective(model)
    JuMP.optimize!(model)
    if !JuMP.is_solved_and_feasible(model; dual=false)
        error("""
        The model was not solved correctly:
        termination_status : $(JuMP.termination_status(model))
        primal_status      : $(JuMP.primal_status(model))
        dual_status        : $(JuMP.dual_status(model))
        raw_status         : $(JuMP.raw_status(model))
        """)
    end
    return model
end

end # module