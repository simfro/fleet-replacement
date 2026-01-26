# Visualization for FleetReplacementMDP
# Implements POMDPTools.render interface

import POMDPTools

"""
    FleetVisualization

Custom type for rendering the fleet replacement MDP state.
Allows flexible display with different MIME types.
"""
struct FleetVisualization
    mdp::FleetReplacementMDP
    step::NamedTuple
end

# Vehicle icons
const DIESEL_ICON = "🚛"
const ELECTRIC_ICON = "⚡🚛"
const ARROW_ICON = "→"
const KEEP_ICON = "✓"
const REPLACE_DT_ICON = "🔄🚛"
const REPLACE_BET_ICON = "🔄⚡"

"""
    POMDPTools.render(mdp::FleetReplacementMDP, step::NamedTuple)

Render the current state of the fleet replacement problem.

# Arguments
- `mdp`: The FleetReplacementMDP instance
- `step`: A NamedTuple containing simulation step data (s, a, sp, r, t, etc.)

# Returns
A `FleetVisualization` object that can be displayed as text or HTML.
"""
function POMDPTools.render(mdp::FleetReplacementMDP, step::NamedTuple = NamedTuple())
    FleetVisualization(mdp, step)
end

"""
    format_vehicle_icon(v::AbstractVehicle)

Returns the appropriate icon for the vehicle type.
"""
format_vehicle_icon(::DieselVehicle) = DIESEL_ICON
format_vehicle_icon(::ElectricVehicle) = ELECTRIC_ICON

"""
    format_decision_icon(d::AbstractReplacementDecision)

Returns the appropriate icon for the replacement decision.
"""
format_decision_icon(::Keep) = KEEP_ICON * " Keep"
format_decision_icon(::Replace{DieselVehicle}) = REPLACE_DT_ICON * " → DT"
format_decision_icon(::Replace{ElectricVehicle}) = REPLACE_BET_ICON * " → BET"

"""
    vehicle_type_str(v::AbstractVehicle)

Returns a short string representation of vehicle type.
"""
vehicle_type_str(::DieselVehicle) = "DT"
vehicle_type_str(::ElectricVehicle) = "BET"

# Text/Plain display
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, vis::FleetVisualization)
    mdp = vis.mdp
    step = vis.step

    # Header
    println(io, "╔══════════════════════════════════════════════════════════════════╗")
    println(io, "║              FLEET REPLACEMENT MDP - DASHBOARD                   ║")
    println(io, "╠══════════════════════════════════════════════════════════════════╣")

    # Time step info
    if haskey(step, :t)
        println(io, "║  Time Step: $(step.t)")
    end

    # State section
    state = get(step, :s, get(step, :sp, nothing))
    action = get(step, :a, nothing)

    if !isnothing(state)
        info = state.info_state

        # InfoState panel
        println(io, "╠══════════════════════════════════════════════════════════════════╣")
        println(
            io, "║  📅 YEAR: $(info.current_year)   ($(mdp.base_year) - $(mdp.final_year))")
        println(io, "╠══════════════════════════════════════════════════════════════════╣")
        println(io, "║  📊 MARKET CONDITIONS                                            ║")
        println(io,
            "║  ├─ Diesel Price:       $(lpad(round(info.energy_price_diesel, digits=2), 8)) SEK/L")
        println(io,
            "║  ├─ Electricity Price:  $(lpad(round(info.energy_price_electricity, digits=2), 8)) SEK/kWh")
        println(io,
            "║  ├─ DT Purchase Price:  $(lpad(round(info.purchase_price_DT/1000, digits=0), 8)) kSEK")
        println(io,
            "║  ├─ BET Purchase Price: $(lpad(round(info.purchase_price_BET/1000, digits=0), 8)) kSEK")
        println(io,
            "║  └─ BET Productivity:   $(lpad(round(info.productivity_BET * 100, digits=1), 8))%")

        # Fleet section
        println(io, "╠══════════════════════════════════════════════════════════════════╣")
        println(io, "║  🚚 FLEET STATUS                                                 ║")
        println(io, "╠══════════════════════════════════════════════════════════════════╣")

        for (i, vehicle) in enumerate(state.fleet)
            icon = format_vehicle_icon(vehicle)
            type_str = vehicle_type_str(vehicle)
            age_str = "Age: $(lpad(age(vehicle), 2))"

            # Age warning
            age_warning = age(vehicle) >= mdp.max_age ? " ⚠️ MAX" : ""

            # Decision for this vehicle (if action available)
            decision_str = ""
            if !isnothing(action) && i <= length(action.decisions)
                decision = action.decisions[i]
                decision_str = " │ $(format_decision_icon(decision))"
            end

            println(io, "║  [$i] $icon $type_str  │  $age_str$age_warning$decision_str")
        end

        # Fleet summary
        n_diesel = count(v -> v isa DieselVehicle, state.fleet)
        n_electric = count(v -> v isa ElectricVehicle, state.fleet)
        println(io, "╠══════════════════════════════════════════════════════════════════╣")
        println(io,
            "║  Fleet: $(n_diesel) DT / $(n_electric) BET (Total: $(length(state.fleet)))")
    end

    # Reward section
    if haskey(step, :r)
        println(io, "╠══════════════════════════════════════════════════════════════════╣")
        println(io, "║  💰 REWARD: $(round(step.r, digits=2)) SEK")
    end

    # Next state section (if transitioning)
    if haskey(step, :sp) && haskey(step, :s)
        sp = step.sp
        sp_info = sp.info_state

        n_diesel_new = count(v -> v isa DieselVehicle, sp.fleet)
        n_electric_new = count(v -> v isa ElectricVehicle, sp.fleet)

        println(io, "╠══════════════════════════════════════════════════════════════════╣")
        println(io, "║  📈 NEXT STATE (Year $(sp_info.current_year))")
        println(io, "║  └─ Fleet: $(n_diesel_new) DT / $(n_electric_new) BET")
    end

    # Terminal state check
    if !isnothing(state) && POMDPs.isterminal(mdp, state)
        println(io, "╠══════════════════════════════════════════════════════════════════╣")
        println(io, "║  🏁 TERMINAL STATE REACHED                                       ║")
    end

    println(io, "╚══════════════════════════════════════════════════════════════════╝")
end

# HTML display for Jupyter notebooks
function Base.show(io::IO, ::MIME{Symbol("text/html")}, vis::FleetVisualization)
    mdp = vis.mdp
    step = vis.step

    state = get(step, :s, get(step, :sp, nothing))
    action = get(step, :a, nothing)
    reward = get(step, :r, nothing)
    t = get(step, :t, nothing)

    html = """
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; border: 2px solid #333; border-radius: 8px; padding: 16px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);">
        <h2 style="margin: 0 0 16px 0; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px;">
            🚛 Fleet Replacement MDP
        </h2>
    """

    if !isnothing(t)
        html *= """<div style="color: #7f8c8d; margin-bottom: 12px;">Time Step: <strong>$t</strong></div>"""
    end

    if !isnothing(state)
        info = state.info_state

        # InfoState card
        html *= """
        <div style="background: white; border-radius: 6px; padding: 12px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="margin: 0 0 8px 0; color: #27ae60;">📅 Year $(info.current_year)</h3>
            <table style="width: 100%; font-size: 14px;">
                <tr><td>⛽ Diesel Price:</td><td style="text-align: right;"><strong>$(round(info.energy_price_diesel, digits=2)) SEK/L</strong></td></tr>
                <tr><td>⚡ Electricity Price:</td><td style="text-align: right;"><strong>$(round(info.energy_price_electricity, digits=2)) SEK/kWh</strong></td></tr>
                <tr><td>🚛 DT Price:</td><td style="text-align: right;"><strong>$(round(info.purchase_price_DT/1000, digits=0)) kSEK</strong></td></tr>
                <tr><td>⚡🚛 BET Price:</td><td style="text-align: right;"><strong>$(round(info.purchase_price_BET/1000, digits=0)) kSEK</strong></td></tr>
                <tr><td>📈 BET Productivity:</td><td style="text-align: right;"><strong>$(round(info.productivity_BET * 100, digits=1))%</strong></td></tr>
            </table>
        </div>
        """

        # Fleet visualization
        html *= """
        <div style="background: white; border-radius: 6px; padding: 12px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="margin: 0 0 8px 0; color: #e74c3c;">🚚 Fleet</h3>
            <div style="display: flex; flex-direction: column; gap: 8px;">
        """

        for (i, vehicle) in enumerate(state.fleet)
            icon = format_vehicle_icon(vehicle)
            type_str = vehicle_type_str(vehicle)
            v_age = age(vehicle)

            age_color = v_age >= mdp.max_age ? "#e74c3c" : "#27ae60"
            bg_color = vehicle isa DieselVehicle ? "#fff3cd" : "#d4edda"

            decision_html = ""
            if !isnothing(action) && i <= length(action.decisions)
                decision = action.decisions[i]
                dec_color = decision isa Keep ? "#27ae60" : "#3498db"
                decision_html = """<span style="margin-left: 12px; padding: 2px 8px; background: $(dec_color); color: white; border-radius: 4px; font-size: 12px;">$(format_decision_icon(decision))</span>"""
            end

            html *= """
            <div style="display: flex; align-items: center; padding: 8px; background: $(bg_color); border-radius: 4px;">
                <span style="font-size: 24px; margin-right: 8px;">$icon</span>
                <span style="font-weight: bold; width: 40px;">$type_str</span>
                <span style="color: $(age_color); margin-left: 12px;">Age: <strong>$v_age</strong>$(v_age >= mdp.max_age ? " ⚠️" : "")</span>
                $decision_html
            </div>
            """
        end

        n_diesel = count(v -> v isa DieselVehicle, state.fleet)
        n_electric = count(v -> v isa ElectricVehicle, state.fleet)

        html *= """
            </div>
            <div style="margin-top: 8px; font-size: 12px; color: #7f8c8d;">
                Summary: $(n_diesel) Diesel / $(n_electric) Electric ($(length(state.fleet)) total)
            </div>
        </div>
        """
    end

    if !isnothing(reward)
        reward_color = reward >= 0 ? "#27ae60" : "#e74c3c"
        html *= """
        <div style="background: $(reward_color); color: white; border-radius: 6px; padding: 12px; text-align: center;">
            <strong>💰 Reward: $(round(reward, digits=2)) SEK</strong>
        </div>
        """
    end

    if !isnothing(state) && POMDPs.isterminal(mdp, state)
        html *= """
        <div style="background: #9b59b6; color: white; border-radius: 6px; padding: 12px; margin-top: 12px; text-align: center;">
            <strong>🏁 Terminal State Reached</strong>
        </div>
        """
    end

    html *= "</div>"

    print(io, html)
end

# Fallback show for generic IO
function Base.show(io::IO, vis::FleetVisualization)
    show(io, MIME"text/plain"(), vis)
end
