using Printf

# ANSI color codes
const COLORS = (
    reset = "\e[0m",
    bold = "\e[1m",
    dim = "\e[2m",
    blue = "\e[34m",
    cyan = "\e[36m",
    green = "\e[32m",
    yellow = "\e[33m",
    red = "\e[31m",
    magenta = "\e[35m",
    bg_blue = "\e[44m",
    bg_cyan = "\e[46m"
)

"""
    get_vehicle_icon(vehicle::AbstractVehicle) -> Tuple{String, String}
    
Returns a Unicode icon and label for a vehicle based on its type.
"""
function get_vehicle_icon(v::DieselVehicle)
    return "🚛", "DT"
end

function get_vehicle_icon(v::ElectricVehicle)
    return "⚡", "BET"
end

"""
    get_age_indicator(age::Int, max_age::Int) -> Tuple{String, String}
    
Returns a visual health indicator based on vehicle age.
"""
function get_age_indicator(age::Int, max_age::Int)
    ratio = min(age / max_age, 1.0)

    if ratio < 0.3
        return "█████ $(COLORS.green)●$(COLORS.reset)", "Excellent"
    elseif ratio < 0.5
        return "████░ $(COLORS.green)●$(COLORS.reset)", "Good"
    elseif ratio < 0.7
        return "███░░ $(COLORS.yellow)●$(COLORS.reset)", "Fair"
    elseif ratio < 0.9
        return "██░░░ $(COLORS.yellow)●$(COLORS.reset)", "Aging"
    else
        return "█░░░░ $(COLORS.red)●$(COLORS.reset)", "Critical"
    end
end

"""
    format_price(price::Float64) -> String
    
Formats a price with appropriate units (k for thousands, M for millions).
"""
function format_price(price::Float64)
    if price >= 1_000_000
        return @sprintf("%.1f M", price/1_000_000)
    elseif price >= 1_000
        return @sprintf("%.0f k", price/1_000)
    else
        return @sprintf("%.0f", price)
    end
end

"""
    POMDPTools.render(mdp::FleetReplacementMDP, step::Dict) -> Nothing
    
Render a beautiful dashboard-style visualization of the fleet state.

The `step` dictionary should contain:
- `:s` : State object (required)
- `:a` : Action taken (optional)
- `:r` : Reward received (optional)
"""
function POMDPTools.render(mdp::FleetReplacementMDP, step::NamedTuple)
    # Convert NamedTuple to Dict for compatibility
    POMDPTools.render(mdp, Dict(pairs(step)))
end

function POMDPTools.render(mdp::FleetReplacementMDP, step::Dict)
    state = get(step, :s, nothing)

    if state === nothing
        println("$(COLORS.red)Error: No state in step dictionary$(COLORS.reset)")
        return nothing
    end

    info = state.info_state
    fleet = state.fleet
    max_age = mdp.max_age

    # --- HEADER ---
    println()
    println("$(COLORS.bold)$(COLORS.cyan)╔══════════════════════════════════════════════════════════════════════════════╗$(COLORS.reset)")
    println("$(COLORS.bold)$(COLORS.cyan)║$(COLORS.reset)  $(COLORS.bold)FLEET REPLACEMENT DASHBOARD$(COLORS.reset)  $(repeat(" ", 50)) $(COLORS.bold)$(COLORS.cyan)║$(COLORS.reset)")
    println("$(COLORS.bold)$(COLORS.cyan)╚══════════════════════════════════════════════════════════════════════════════╝$(COLORS.reset)")
    println()

    # --- EXOGENOUS MARKET CONDITIONS ---
    println("$(COLORS.bold)$(COLORS.blue)┌─ MARKET CONDITIONS$(COLORS.reset) " *
            repeat("─", 60) * "┐")

    year_str = "Year $(COLORS.bold)$(COLORS.yellow)$(info.current_year)$(COLORS.reset)"
    diesel_price_str = "Diesel: $(COLORS.bold)$(COLORS.red)$(format_price(info.energy_price_diesel))/L$(COLORS.reset)"
    electric_price_str = "Electric: $(COLORS.bold)$(COLORS.green)$(format_price(info.energy_price_electricity))/kWh$(COLORS.reset)"

    bet_prod_pct = round(info.productivity_BET * 100, digits = 1)
    productivity_str = "BET Productivity: $(COLORS.bold)$(bet_prod_pct)%$(COLORS.reset)"

    println("│ $(rpad(year_str, 74))|")
    println("│ $(rpad(diesel_price_str, 74))|")
    println("│ $(rpad(electric_price_str, 74))|")
    println("│ $(rpad(productivity_str, 74))|")
    println("│ $(repeat(" ", 74))|")

    dt_purchase_str = "DT Purchase: $(COLORS.bold)$(format_price(info.purchase_price_DT))$(COLORS.reset)"
    bet_purchase_str = "BET Purchase: $(COLORS.bold)$(format_price(info.purchase_price_BET))$(COLORS.reset)"

    println("│ $(rpad(dt_purchase_str, 74))|")
    println("│ $(rpad(bet_purchase_str, 74))|")
    println("$(COLORS.blue)└$(repeat("─", 76))┘$(COLORS.reset)")
    println()

    # --- FLEET STATUS ---
    println("$(COLORS.bold)$(COLORS.magenta)┌─ FLEET STATUS ($(length(fleet)) vehicles)$(COLORS.reset) " *
            repeat("─", 55) * "┐")

    if length(fleet) == 0
        println("│ $(rpad("No vehicles in fleet", 74))|")
    else
        for (idx, vehicle) in enumerate(fleet)
            icon, label = get_vehicle_icon(vehicle)
            age_bar, age_status = get_age_indicator(age(vehicle), max_age)

            # Vehicle info line
            vehicle_line = "$(icon) #$(idx) [$label] - Age: $(age(vehicle))y $(age_bar)  $(age_status)"
            println("│ $(rpad(vehicle_line, 74))|")

            # Purchase price line
            price_line = "  └─ Purchased at: $(COLORS.bold)$(format_price(purchase_price(vehicle)))$(COLORS.reset)"
            println("│ $(rpad(price_line, 74))|")
        end
    end

    println("$(COLORS.magenta)└$(repeat("─", 76))┘$(COLORS.reset)")
    println()

    # --- ACTION AND REWARD INFO ---
    if get(step, :a, nothing) !== nothing || get(step, :r, nothing) !== nothing
        println("$(COLORS.bold)$(COLORS.yellow)┌─ STEP INFORMATION$(COLORS.reset) " *
                repeat("─", 58) * "┐")

        if get(step, :a, nothing) !== nothing
            action_str = "Action: $(COLORS.bold)$(step[:a])$(COLORS.reset)"
            println("│ $(rpad(action_str, 74))|")
        end

        if get(step, :r, nothing) !== nothing
            reward = step[:r]
            reward_color = reward >= 0 ? COLORS.green : COLORS.red
            reward_str = "Reward: $(COLORS.bold)$(reward_color)$(round(reward, digits=2))$(COLORS.reset)"
            println("│ $(rpad(reward_str, 74))|")
        end

        println("$(COLORS.yellow)└$(repeat("─", 76))┘$(COLORS.reset)")
        println()
    end

    return nothing
end
