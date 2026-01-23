# ----------------------------
# State Types
# ----------------------------

abstract type AbstractVehicle end

struct VehicleBase
    age::Int
    purchase_price::Float64
end

age(v::AbstractVehicle) = v.base.age
purchase_price(v::AbstractVehicle) = v.base.purchase_price

struct DieselVehicle <: AbstractVehicle
    base::VehicleBase
end
function Base.show(io::IO, dv::DieselVehicle)
    print(io, "DT(age=$(age(dv)), price=$(purchase_price(dv)))")
end
function DieselVehicle(age::Int, purchase_price::Float64)
    DieselVehicle(VehicleBase(age, purchase_price))
end

struct ElectricVehicle <: AbstractVehicle
    base::VehicleBase
end
function Base.show(io::IO, ev::ElectricVehicle)
    print(io, "BET(age=$(age(ev)), price=$(purchase_price(ev)))")
end
function ElectricVehicle(age::Int, purchase_price::Float64)
    ElectricVehicle(VehicleBase(age, purchase_price))
end

struct InfoState
    current_year::Int
    energy_price_diesel::Float64
    energy_price_electricity::Float64
    purchase_price_DT::Float64
    purchase_price_BET::Float64
    productivity_BET::Float64
end

""" 
    Contains all the information necessary for making replacement decisions. 
    Separated into vehicle fleet and info state for clarity. 
"""
struct State{N, V <: AbstractVehicle}
    fleet::SVector{N, V}
    info_state::InfoState
end
"""
    State(fleet::SVector{N, V}, info_state::InfoState) where {N, V <: AbstractVehicle}

Construct a `State` from a any AbstractVector of vehicles by converting it to an `SVector`.
"""
function State(fleet::AbstractVector{V}, info_state::InfoState) where {V <: AbstractVehicle}
    N = length(fleet)
    State{N, V}(SVector{N, V}(fleet), info_state)
end

function purchase_price(info_state::InfoState, ::Type{DieselVehicle})
    info_state.purchase_price_DT
end
function purchase_price(info_state::InfoState, ::Type{ElectricVehicle})
    info_state.purchase_price_BET
end

# ----------------------------
# Action Types and Sets
# ----------------------------

abstract type AbstractReplacementDecision end

struct Keep <: AbstractReplacementDecision
    index::Int
end
Base.show(io::IO, ::Keep) = print(io, "Keep")

struct Replace{T <: AbstractVehicle} <: AbstractReplacementDecision
    index::Int
end
Base.show(io::IO, ::Replace{T}) where {T} = print(io, "Replace($(T))")

struct ReplacementDecisionSet
    decisions::SVector{N, <:AbstractReplacementDecision} where {N}
end
Base.show(io::IO, d::ReplacementDecisionSet) = begin
    decision_str = join(d.decisions, ", ")
    print(io, "[$decision_str]")
end

"""
    ReplacementDecisionSet(decisions::AbstractVector)

Create a `ReplacementDecisionSet` from a regular vector of replacement decisions.
"""
function ReplacementDecisionSet(decisions::AbstractVector{<:AbstractReplacementDecision})
    return ReplacementDecisionSet(SVector(decisions...))
end