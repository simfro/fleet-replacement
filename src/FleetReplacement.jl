"Environment module for the fleet replacement problem using POMDPs.jl"
module FleetReplacement

import POMDPs
import POMDPTools
import Parameters
import TOML
using Distributions: Normal, mean
using Random: AbstractRNG
using StaticArrays: SVector

export FleetReplacementMDP,
       State,
       InfoState,
       DieselVehicle,
       ElectricVehicle

include("types.jl")

# Load default parameters from TOML file
function _load_default_params()
    # Try to find scenario_parameters.toml relative to this file
    project_root = dirname(dirname(@__FILE__))
    toml_path = joinpath(project_root, "data", "scenario_parameters.toml")

    if !isfile(toml_path)
        error("Could not find scenario_parameters.toml at $toml_path")
    end

    TOML.parsefile(toml_path)
end

const _DEFAULT_PARAMS = _load_default_params()

" 
    FleetReplacementMDP defines the Markov Decision Process for the fleet replacement problem. 
    
    Create with: `FleetReplacementMDP(s₀)` using default parameters from data/scenario_parameters.toml
"
Parameters.@with_kw struct FleetReplacementMDP <: POMDPs.MDP{State, ReplacementDecisionSet}
    # Scenario time frame
    base_year::Int = Int(_DEFAULT_PARAMS["simulation_period"]["base_year"])
    final_year::Int = Int(_DEFAULT_PARAMS["simulation_period"]["final_year"])
    # Diesel price parameters
    initial_diesel_price::Float64 = _DEFAULT_PARAMS["diesel_price"]["initial_price"]
    diesel_price_growth_rate::Float64 = _DEFAULT_PARAMS["diesel_price"]["growth_rate"]
    diesel_price_volatility::Float64 = _DEFAULT_PARAMS["diesel_price"]["volatility"]
    # Electricity price parameters
    initial_electricity_price::Float64 = _DEFAULT_PARAMS["electricity_price"]["initial_price"]
    electricity_price_growth_rate::Float64 = _DEFAULT_PARAMS["electricity_price"]["growth_rate"]
    electricity_price_volatility::Float64 = _DEFAULT_PARAMS["electricity_price"]["volatility"]
    # DT purchase price parameters
    initial_DT_price::Float64 = _DEFAULT_PARAMS["diesel_vehicle"]["initial_price"]
    # BET purchase price parameters
    initial_BET_price::Float64 = _DEFAULT_PARAMS["electric_vehicle"]["initial_price"]
    BET_long_term_mean::Float64 = _DEFAULT_PARAMS["electric_vehicle"]["long_term_mean"]
    BET_mean_reversion_strength::Float64 = _DEFAULT_PARAMS["electric_vehicle"]["mean_reversion_strength"]
    BET_purchase_price_volatility::Float64 = _DEFAULT_PARAMS["electric_vehicle"]["purchase_price_volatility"]
    # BET productivity parameters
    BET_productivity_start::Float64 = _DEFAULT_PARAMS["BET_productivity"]["start"]
    BET_productivity_max::Float64 = _DEFAULT_PARAMS["BET_productivity"]["max"]
    BET_productivity_k::Float64 = _DEFAULT_PARAMS["BET_productivity"]["k"]
    BET_productivity_t₀::Int = Int(_DEFAULT_PARAMS["BET_productivity"]["t0"])
    # Operational data
    fuel_consumption_l_per_km::Float64 = _DEFAULT_PARAMS["operational"]["fuel_consumption_l_per_km"]
    el_consumption_kWh_per_km::Float64 = _DEFAULT_PARAMS["operational"]["electricity_consumption_kWh_per_km"]
    driver_salary_annual::Int = Int(_DEFAULT_PARAMS["operational"]["driver_salary_annual"])
    income_per_km::Int = Int(_DEFAULT_PARAMS["operational"]["income_per_km"])
    annual_mileage_km::Int = Int(_DEFAULT_PARAMS["operational"]["annual_mileage_km"])
    # Loan parameters
    interest_rate::Float64 = _DEFAULT_PARAMS["loan"]["interest_rate"]
    loan_fraction::Float64 = _DEFAULT_PARAMS["loan"]["fraction"]
    economic_lifetime::Int = Int(_DEFAULT_PARAMS["loan"]["economic_lifetime"])
    # Vehicle management
    max_age::Int = Int(_DEFAULT_PARAMS["vehicle_management"]["max_age"])
    max_replacements::Int = Int(_DEFAULT_PARAMS["vehicle_management"]["max_replacements"])
    # Residual value parameters
    residual_initial_depreciation::Float64 = _DEFAULT_PARAMS["residual_value"]["initial_depreciation"]
    residual_annual_depreciation_rate::Float64 = _DEFAULT_PARAMS["residual_value"]["annual_depreciation_rate"]
    residual_market_elasticity::Float64 = _DEFAULT_PARAMS["residual_value"]["market_elasticity"]
    residual_floor_fraction::Float64 = _DEFAULT_PARAMS["residual_value"]["floor_fraction"]
    # Discount factor
    γ::Float64 = _DEFAULT_PARAMS["discount"]["gamma"]
    # Initial state (required - no default)
    s₀::State
end

POMDPs.discount(mdp::FleetReplacementMDP) = mdp.γ

# POMDPs.statetype(::FleetReplacementMDP) = State # Might not be necessary

function POMDPs.initialstate(mdp::FleetReplacementMDP)
    POMDPTools.Deterministic(mdp.s₀)
end

function POMDPs.isterminal(mdp::FleetReplacementMDP, s::State)
    s.info_state.current_year > mdp.final_year
end

include("actions.jl")
include("transition.jl")
include("reward.jl")
include("visualization.jl")

if false
    include("../scripts/__intellisense_includes.jl")
end

end # module FleetReplacement
