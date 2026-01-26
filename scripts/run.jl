import POMDPs
import POMDPTools
import TOML

using FleetReplacement

params = TOML.parsefile(joinpath(@__DIR__, "..", "data", "scenario_parameters.toml"))

# initial_fleet = [DieselVehicle(0, params["diesel_vehicle"]["initial_price"]) for _ in 1:4]
initial_fleet = [DieselVehicle(0, params["diesel_vehicle"]["initial_price"]),
    ElectricVehicle(0, params["electric_vehicle"]["initial_price"])]
initial_info_state = InfoState(
    Int(params["simulation_period"]["base_year"]),
    params["diesel_price"]["initial_price"],
    params["electricity_price"]["initial_price"],
    params["diesel_vehicle"]["initial_price"],
    params["electric_vehicle"]["initial_price"],
    params["BET_productivity"]["start"]
)

s₀ = State(initial_fleet, initial_info_state)

# Create MDP with defaults - only pass s₀
mdp = FleetReplacementMDP(s₀ = s₀)

# POMDPs.actions(mdp, mdp.s₀)
# d = POMDPs.actions(mdp, mdp.s₀)[5]

# ds = POMDPTools.DisplaySimulator()
# POMDPs.simulate(ds, mdp, POMDPTools.RandomPolicy(mdp))

hs = POMDPTools.HistoryRecorder()
history = POMDPs.simulate(hs, mdp, POMDPTools.RandomPolicy(mdp))
POMDPTools.render(mdp, history.hist[1])