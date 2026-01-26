# Test script for render function
using FleetReplacement
using POMDPs
using POMDPTools
import TOML

# Create initial state from parameters
params = TOML.parsefile(joinpath(@__DIR__, "..", "data", "scenario_parameters.toml"))
initial_fleet = [
    DieselVehicle(3, params["diesel_vehicle"]["initial_price"]),
    DieselVehicle(7, params["diesel_vehicle"]["initial_price"]),
    ElectricVehicle(1, params["electric_vehicle"]["initial_price"])
]
initial_info_state = InfoState(
    Int(params["simulation_period"]["base_year"]),
    params["diesel_price"]["initial_price"],
    params["electricity_price"]["initial_price"],
    params["diesel_vehicle"]["initial_price"],
    params["electric_vehicle"]["initial_price"],
    params["BET_productivity"]["start"]
)
s₀ = State(initial_fleet, initial_info_state)

# Create MDP
mdp = FleetReplacementMDP(s₀ = s₀)

println("="^70)
println("TEST 1: Render state only")
println("="^70)

# Render the state only
vis = POMDPTools.render(mdp, (s = s₀,))
display(vis)

println("\n")
println("="^70)
println("TEST 2: Render with action and reward")
println("="^70)

# Test with action
a = FleetReplacement.ReplacementDecisionSet([
    FleetReplacement.Keep(1),
    FleetReplacement.Replace{ElectricVehicle}(2),
    FleetReplacement.Keep(3)
])
vis2 = POMDPTools.render(mdp, (s = s₀, a = a, r = -125000.0, t = 1))
display(vis2)

println("\n")
println("="^70)
println("TEST 3: Render with state transition")
println("="^70)

# Create next state after transition
next_info_state = InfoState(
    2026,
    18.5,   # Updated diesel price
    1.85,   # Updated electricity price
    1250000.0,
    1400000.0,
    0.92
)
next_fleet = [
    DieselVehicle(4, params["diesel_vehicle"]["initial_price"]),
    ElectricVehicle(0, params["electric_vehicle"]["initial_price"]),  # Replaced!
    ElectricVehicle(2, params["electric_vehicle"]["initial_price"])
]
sp = State(next_fleet, next_info_state)

vis3 = POMDPTools.render(mdp, (s = s₀, a = a, sp = sp, r = -125000.0, t = 1))
display(vis3)

println("\n✅ All render tests completed!")
