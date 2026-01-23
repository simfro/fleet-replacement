POMDPs.actiontype(::FleetReplacementMDP) = ReplacementDecisionSet

function POMDPs.actions(mdp::FleetReplacementMDP, s::State)::Vector{ReplacementDecisionSet}
    vehicle_decisions = allowed_vehicle_decisions(mdp, s)
    all_decision_sets = decision_combinations(vehicle_decisions)
    filtered_decision_sets = filter_decision_sets(all_decision_sets, s, mdp)
    return filtered_decision_sets
end

"""
    allowed_vehicle_decisions(
        mdp::FleetReplacementMDP, s::State
    )::Vector{Vector{AbstractReplacementDecision}}

Generates a list of allowed decisions for each vehicle in the fleet based on its age.
"""
function allowed_vehicle_decisions(
        mdp::FleetReplacementMDP, s::State)::Vector{Vector{AbstractReplacementDecision}}
    vehicle_decisions = Vector{Vector{AbstractReplacementDecision}}()
    for (index, vehicle) in enumerate(s.fleet)
        if age(vehicle) >= mdp.max_age
            push!(
                vehicle_decisions,
                [
                    Replace{DieselVehicle}(index),
                    Replace{ElectricVehicle}(index)
                ]
            )
        elseif age(vehicle) < mdp.max_age
            push!(
                vehicle_decisions,
                [
                    Keep(index),
                    Replace{DieselVehicle}(index),
                    Replace{ElectricVehicle}(index)
                ]
            )
        end
    end
    return vehicle_decisions
end

"""
    decision_combinations(
        vehicle_decisions::Vector{Vector{AbstractReplacementDecision}}
    )::Vector{ReplacementDecisionSet}

Generates all possible combinations of replacement decisions for the fleet.
"""
function decision_combinations(vehicle_decisions::Vector{Vector{AbstractReplacementDecision}})::Vector{ReplacementDecisionSet}
    decision_combinations = Iterators.product(vehicle_decisions...)
    all_decision_sets = vec([ReplacementDecisionSet(collect(combo))
                             for combo in decision_combinations])
    return all_decision_sets
end

function filter_decision_sets(decision_sets::Vector{ReplacementDecisionSet}, s::State,
        mdp::FleetReplacementMDP)::Vector{ReplacementDecisionSet}
    num_must_replace = count(v -> age(v) >= mdp.max_age, s.fleet)
    max_replacements = mdp.max_replacements
    filter!(
        d -> count(x -> x isa Replace, d.decisions) >= num_must_replace &&
            count(x -> x isa Replace, d.decisions) <=
            max(num_must_replace, max_replacements),
        decision_sets
    )
    return decision_sets
end
