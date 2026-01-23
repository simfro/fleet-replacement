# using POMDPTools: ImplicitDistribution

function POMDPs.transition(mdp::FleetReplacementMDP, s::State,
        d::ReplacementDecisionSet)
    return POMDPTools.ImplicitDistribution(s, d) do s, d, rng
        new_fleet = update_fleet(s, d)
        new_info_state = update_info_state(mdp, s, rng)
        return State(new_fleet, new_info_state)
    end
end

# ----------------------------
# Vehicle fleet update functions
# ----------------------------

function update_fleet(s::State, d::ReplacementDecisionSet)::Vector{AbstractVehicle}
    return [update_vehicle_slot(s.fleet[i], d, s.info_state)
            for (i, d) in enumerate(d.decisions)]
end

function update_vehicle_slot(
        v::AbstractVehicle,
        ::Keep,
        ::InfoState
)::AbstractVehicle
    return typeof(v)(age(v) + 1, purchase_price(v))
end

function update_vehicle_slot(
        ::AbstractVehicle,
        ::Replace{T},
        info_state::InfoState
)::AbstractVehicle where {T <: AbstractVehicle}
    return T(0, purchase_price(info_state, T))
end

# ----------------------------
# Information state update functions
# ----------------------------

function update_info_state(mdp::FleetReplacementMDP, s::State, rng::AbstractRNG)::InfoState
    next_year = s.info_state.current_year + 1
    new_energy_price_diesel = update_diesel_price(
        rng,
        s.info_state.energy_price_diesel,
        mdp.diesel_price_growth_rate,
        mdp.diesel_price_volatility
    )
    new_energy_price_electricity = update_electricity_price(
        rng,
        s.info_state.energy_price_electricity,
        mdp.electricity_price_growth_rate,
        mdp.electricity_price_volatility
    )
    new_purchase_price_DT = update_DT_purchase_price(
        rng,
        s.info_state.purchase_price_DT
    )
    new_purchase_price_BET = update_BET_purchase_price(
        rng,
        s.info_state.purchase_price_BET,
        mdp.BET_long_term_mean,
        mdp.BET_mean_reversion_strength,
        mdp.BET_purchase_price_volatility
    )
    new_productivity_BET = update_BET_productivity(
        rng,
        s.info_state.productivity_BET,
        next_year,
        mdp.BET_productivity_start,
        mdp.BET_productivity_max,
        mdp.BET_productivity_k,
        mdp.BET_productivity_t₀
    )

    return InfoState(
        next_year,
        new_energy_price_diesel,
        new_energy_price_electricity,
        new_purchase_price_DT,
        new_purchase_price_BET,
        new_productivity_BET
    )
end

function update_diesel_price(
        rng,
        diesel_price::Float64,      # Current diesel price
        μ::Float64,                 # Growth rate
        σ::Float64                  # Volatility
)::Float64
    ε = rand(rng, Normal(0, 1))
    drift = (μ - 0.5 * σ^2)
    diffusion = σ * ε
    new_price = diesel_price * exp(drift + diffusion)
    return new_price
end

function update_electricity_price(
        rng,
        electricity_price::Float64, # Current electricity price
        μ::Float64,                 # Growth rate
        σ::Float64                  # Volatility
)::Float64
    ε = rand(rng, Normal(0, 1))
    drift = (μ - 0.5 * σ^2)
    diffusion = σ * ε
    new_price = electricity_price * exp(drift + diffusion)
    return new_price
end

function update_DT_purchase_price(
        rng,
        purchase_price::Float64
)::Float64
    return purchase_price
end

function update_BET_purchase_price(
        rng,
        purchase_price::Float64,
        μ::Float64, # Long-term mean
        θ::Float64, # Mean reversion strength
        σ::Float64  # Volatility
)::Float64
    ε = rand(rng, Normal(0, σ))
    new_price = purchase_price + θ * (μ - purchase_price) + ε
    return new_price
end

function update_BET_productivity(
        rng,
        p_current::Float64, # Current productivity
        t::Int,             # Current year
        P_start::Float64,   # Initial productivity
        P_max::Float64,     # Maximum productivity
        k::Float64,         # Growth rate
        t₀::Int             # Year when productivity starts to increase
)::Float64
    base = 1 / (1 + exp(-k * (t - t₀)))
    next_p = P_start + (P_max - P_start) * base
    return next_p
end