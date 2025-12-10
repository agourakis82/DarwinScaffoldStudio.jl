"""
    PLDLADegradationModel

Multi-mechanism degradation model for PLDLA scaffolds.

Incorporates:
- Hydrolytic degradation (first-order kinetics)
- Autocatalysis (pH-dependent acceleration)
- Temperature effects (Arrhenius)
- Crystallinity effects (three-phase model: MAF, RAF, crystalline)
- Mechanical property evolution (Mw-dependent)
- Mass loss kinetics (oligomer dissolution)

Calibrated with experimental data from:
- Kaique Hergesel (2025) - PLDLA/TEC scaffolds for meniscus

References:
- Lyu & Untereker (2009) - Degradation kinetics
- Siepmann et al. (2005) - Autocatalysis in PLGA
- Auras Lab (2023) - Three-phase model
- Weir et al. (2004) - Mw-mechanical properties

Author: Darwin Scaffold Studio
Date: December 2025
"""
module PLDLADegradationModel

export DegradationState, PLDLAParameters, EnvironmentConditions
export simulate_degradation, predict_properties_at_time
export validate_against_experimental, plot_degradation_curves

using Statistics
using Printf

# =============================================================================
# Types
# =============================================================================

"""
State of scaffold degradation at a given time point.
"""
struct DegradationState
    time_days::Float64

    # Molecular properties
    Mw::Float64              # Weight-average molecular weight (kg/mol)
    Mn::Float64              # Number-average molecular weight (kg/mol)
    PDI::Float64             # Polydispersity index (Mw/Mn)

    # Thermal properties
    Tg::Float64              # Glass transition temperature (°C)
    crystallinity::Float64   # Crystalline fraction (0-1)

    # Physical properties
    mass_fraction::Float64   # Remaining mass fraction (0-1)
    porosity::Float64        # Porosity (0-1)

    # Internal environment
    pH_internal::Float64     # pH inside scaffold bulk

    # Mechanical properties
    E_modulus::Float64       # Compressive modulus (MPa)

    # Kinetic parameter
    k_effective::Float64     # Effective degradation rate (1/day)
end

"""
Material parameters for PLDLA variants.
"""
Base.@kwdef struct PLDLAParameters
    # === Molecular (from Kaique Hergesel data) ===
    Mw_initial::Float64 = 94.4      # kg/mol (GPC)
    Mn_initial::Float64 = 51.3      # kg/mol (GPC)
    Tg_initial::Float64 = 54.0      # °C (DSC)
    crystallinity_initial::Float64 = 0.05  # ~5% (semi-crystalline)

    # === Morphology ===
    porosity_initial::Float64 = 0.395    # 39.5% (SEM analysis)
    pore_size_um::Float64 = 120.3        # μm mean pore size
    thickness_mm::Float64 = 2.0          # Scaffold thickness

    # === Kinetics (calibrated from Kaique 90-day data) ===
    k_base::Float64 = 0.023              # 1/day at pH 7.4, 37°C
    Ea_hydrolysis::Float64 = 65000.0     # J/mol (literature: 58-100 kJ/mol)

    # === Mechanical (from compression tests) ===
    E_initial::Float64 = 2.67            # MPa (compression modulus)

    # === Plasticizer (TEC) ===
    TEC_weight_percent::Float64 = 0.0    # 0, 1, or 2%

    # === Material identity ===
    name::String = "PLDLA"
end

"""
Environmental conditions for degradation.
"""
Base.@kwdef struct EnvironmentConditions
    pH::Float64 = 7.4                # PBS buffer
    temperature_C::Float64 = 37.0    # Body temperature
    enzyme_activity::Float64 = 0.0   # Relative enzyme activity (0-1)
    mechanical_stress_MPa::Float64 = 0.0  # Applied stress
end

# =============================================================================
# Kinetic Models
# =============================================================================

"""
Calculate effective degradation rate constant.

RECALIBRATED VERSION v2 - Fitted to Kaique's experimental data.

Key insight: The original model over-predicted degradation because:
1. Multiplicative factors accumulated too aggressively
2. Kaique's porous scaffolds (39.5% porosity) don't accumulate acid like microspheres
3. The base k=0.023/day already captures most of the degradation physics

This version uses MINIMAL corrections to preserve the good fit at 90 days
while improving intermediate points (30, 60 days).

Combines factors:
1. Temperature (Arrhenius) - unchanged
2. pH (autocatalysis) - GREATLY REDUCED for porous scaffolds
3. Crystallinity - SIMPLIFIED (only slows, doesn't accelerate)
4. Geometry - MINIMAL for thin porous samples
5. Plasticizer (TEC) - MINIMAL (data shows similar k values)
"""
function calculate_k_effective(
    params::PLDLAParameters,
    env::EnvironmentConditions,
    state::DegradationState
)
    k = params.k_base

    # === 1. Temperature Effect (Arrhenius) ===
    # Well-established, keep unchanged
    T_ref_K = 310.15  # 37°C
    T_actual_K = env.temperature_C + 273.15
    R = 8.314  # J/(mol·K)

    if abs(env.temperature_C - 37.0) > 0.1
        k *= exp((params.Ea_hydrolysis / R) * (1/T_ref_K - 1/T_actual_K))
    end

    # === 2. pH Effect (Autocatalysis) - MINIMAL ===
    # For porous scaffolds, acid diffuses out easily
    # Only apply small correction
    ΔpH = env.pH - state.pH_internal

    if ΔpH > 0.5  # Only if significant acidification
        # Very small factor: max ~10% acceleration
        k *= 1.0 + 0.1 * min(ΔpH, 1.0)
    end

    # === 3. Crystallinity Effect - SIMPLIFIED ===
    # Crystallinity SLOWS degradation (water exclusion)
    # But don't over-correct - crystals also have defects
    X_c = state.crystallinity
    X_c_initial = params.crystallinity_initial

    # Only slow down if crystallinity increased significantly
    if X_c > X_c_initial + 0.1
        # Max 20% slowdown even at high crystallinity
        k *= max(0.8, 1.0 - 0.3 * (X_c - X_c_initial))
    end

    # === 4. Geometry Effect - MINIMAL ===
    # Kaique's scaffolds are thin and porous
    # Negligible bulk effect
    # k *= 1.0 (effectively disabled)

    # === 5. Plasticizer Effect (TEC) - MINIMAL ===
    # Kaique's data shows similar k for all TEC levels
    # Only very slight acceleration
    k *= 1.0 + 0.02 * params.TEC_weight_percent

    # === 6. Enzyme Effect (if present) ===
    if env.enzyme_activity > 0
        k *= 1.0 + 2.0 * env.enzyme_activity
    end

    # === 7. Mechanical Stress Effect ===
    if env.mechanical_stress_MPa > 0
        k *= 1.0 + 0.05 * env.mechanical_stress_MPa
    end

    return k
end

# =============================================================================
# Property Prediction Functions
# =============================================================================

"""
Predict Tg using modified Fox-Flory relationship.

Tg decreases with:
1. Decreasing Mw (more chain ends)
2. Increasing plasticizer content
3. Increasing crystallinity (slight increase due to constraint)
"""
function predict_Tg(
    Mw_current::Float64,
    params::PLDLAParameters
)
    # Fox-Flory: Tg = Tg∞ - K/Mn
    # Simplified power law: Tg ∝ Mw^α
    α = 0.35  # Empirical exponent for PLA family

    Tg_base = params.Tg_initial

    # Mw effect
    Mw_ratio = Mw_current / params.Mw_initial
    Tg = Tg_base * Mw_ratio^α

    # TEC plasticizer depression
    # Literature: ~5°C drop per 1% TEC
    Tg -= 5.0 * params.TEC_weight_percent

    # Physical lower bound
    return max(Tg, -20.0)
end

"""
Predict crystallinity evolution during degradation.

Crystallinity INCREASES during degradation because:
1. Amorphous regions degrade first → enriches crystalline
2. Short chains crystallize more easily
3. Degradation-induced crystallization
"""
function predict_crystallinity(
    Mw_current::Float64,
    time_days::Float64,
    params::PLDLAParameters
)
    Mw_ratio = Mw_current / params.Mw_initial
    X_c_initial = params.crystallinity_initial

    if Mw_ratio > 0.4
        # Early phase: slow crystallinity increase
        # Amorphous regions preferentially degrading
        X_c = X_c_initial * (1.0 + 0.5 * (1.0 - Mw_ratio))
    elseif Mw_ratio > 0.15
        # Mid phase: faster crystallization
        # Short chains have higher mobility for crystal formation
        X_c = X_c_initial + 0.3 * (1.0 - Mw_ratio)
    else
        # Late phase: maximum crystallinity reached
        # Most amorphous material degraded
        X_c = X_c_initial + 0.5
    end

    # Physical upper bound (~60% for degraded PLA)
    return min(X_c, 0.60)
end

"""
Predict internal pH based on acid accumulation.

pH drops because:
1. Hydrolysis produces carboxylic acid end groups
2. Lactic acid oligomers accumulate
3. Low porosity limits diffusion out
"""
function predict_pH_internal(
    Mw_current::Float64,
    porosity::Float64,
    params::PLDLAParameters,
    env::EnvironmentConditions
)
    # Degradation extent
    degradation_fraction = 1.0 - Mw_current / params.Mw_initial

    # Accumulation factor
    # Low porosity = poor diffusion = more accumulation
    # Thick samples = longer diffusion path
    diffusion_resistance = (1.0 - porosity) * (1.0 + 0.1 * params.thickness_mm)
    accumulation = diffusion_resistance * degradation_fraction

    # pH drop (literature: can reach pH 1.5 in microspheres!)
    # For scaffolds with porosity, less extreme
    ΔpH_max = 2.5  # Maximum pH drop
    ΔpH = ΔpH_max * accumulation^1.5

    # pKa of lactic acid ≈ 3.86
    pH_minimum = 4.0

    return max(env.pH - ΔpH, pH_minimum)
end

"""
Predict mass loss.

Mass loss only occurs when:
1. Mw drops below critical value (~10 kg/mol)
2. Oligomers become water-soluble
3. Fragments can diffuse out
"""
function predict_mass_loss(
    Mw_current::Float64,
    porosity::Float64,
    time_days::Float64,
    params::PLDLAParameters
)
    # Critical Mw for dissolution
    Mw_critical = 10.0  # kg/mol (oligomers become soluble)

    if Mw_current >= Mw_critical
        # Phase 1: No mass loss (only molecular degradation)
        return 0.0
    else
        # Phase 2: Erosion begins
        # Rate depends on:
        # - How far below critical Mw
        # - Porosity (diffusion paths)
        # - Time since onset

        excess_degradation = (Mw_critical - Mw_current) / Mw_critical
        porosity_factor = 0.5 + 0.5 * porosity  # Higher porosity = faster loss

        mass_loss = excess_degradation * porosity_factor

        return min(mass_loss, 0.95)  # Maximum 95% loss
    end
end

"""
Predict porosity evolution.

Porosity increases due to:
1. Microporosity from chain scission (before mass loss)
2. Macroporosity from mass erosion (after Mw < critical)
"""
function predict_porosity(
    Mw_current::Float64,
    mass_loss::Float64,
    params::PLDLAParameters
)
    p_initial = params.porosity_initial

    # Degradation extent
    degradation_fraction = 1.0 - Mw_current / params.Mw_initial

    # Microporosity increase (swelling, crazing, chain scission)
    Δp_micro = 0.15 * degradation_fraction

    # Macroporosity increase (from mass erosion)
    Δp_macro = 0.4 * mass_loss

    # Total porosity
    porosity = p_initial + Δp_micro + Δp_macro

    return min(porosity, 0.98)  # Physical limit
end

"""
Predict compressive modulus.

Modulus decreases due to:
1. Decreasing Mw (chain entanglement loss)
2. Increasing porosity (less material)

But can be partially offset by:
3. Increasing crystallinity (stiffer crystals)
"""
function predict_modulus(
    Mw_current::Float64,
    crystallinity::Float64,
    porosity::Float64,
    params::PLDLAParameters
)
    E_initial = params.E_initial

    # Mw effect (power law)
    # E ∝ Mw^n where n ≈ 0.5-1.0
    n = 0.75
    Mw_ratio = Mw_current / params.Mw_initial
    E = E_initial * Mw_ratio^n

    # Crystallinity effect (stiffening)
    # ~20% increase per 10% crystallinity gain
    ΔX_c = crystallinity - params.crystallinity_initial
    E *= 1.0 + 2.0 * ΔX_c

    # Porosity effect (Gibson-Ashby)
    # E_scaffold/E_solid ∝ (1-porosity)^2
    p_initial = params.porosity_initial
    if porosity > p_initial
        porosity_factor = ((1 - porosity) / (1 - p_initial))^2
        E *= porosity_factor
    end

    # TEC softening effect
    E *= 1.0 - 0.15 * params.TEC_weight_percent

    return max(E, 0.01)  # Minimum physical value
end

# =============================================================================
# Main Simulation Function
# =============================================================================

"""
    simulate_degradation(params, env, time_points)

Simulate degradation of PLDLA scaffold over time.

# Arguments
- `params::PLDLAParameters`: Material parameters
- `env::EnvironmentConditions`: Environmental conditions
- `time_points::Vector{Float64}`: Time points in days

# Returns
- `Vector{DegradationState}`: State at each time point
"""
function simulate_degradation(
    params::PLDLAParameters,
    env::EnvironmentConditions,
    time_points::Vector{Float64}
)
    states = DegradationState[]

    # Initial state
    initial_Tg = params.Tg_initial - 5.0 * params.TEC_weight_percent
    initial_E = params.E_initial * (1.0 - 0.15 * params.TEC_weight_percent)

    state = DegradationState(
        0.0,                          # time
        params.Mw_initial,            # Mw
        params.Mn_initial,            # Mn
        params.Mw_initial / params.Mn_initial,  # PDI
        initial_Tg,                   # Tg
        params.crystallinity_initial, # crystallinity
        1.0,                          # mass fraction
        params.porosity_initial,      # porosity
        env.pH,                       # pH internal (initially = medium)
        initial_E,                    # E modulus
        params.k_base                 # k effective
    )
    push!(states, state)

    # Time integration
    for i in 2:length(time_points)
        t = time_points[i]
        dt = t - time_points[i-1]
        prev = states[end]

        # Calculate effective rate
        k_eff = calculate_k_effective(params, env, prev)

        # === Molecular Weight Decay (First-Order) ===
        Mw_new = prev.Mw * exp(-k_eff * dt)

        # PDI evolution (broadens initially, then narrows)
        PDI_new = if Mw_new / params.Mw_initial > 0.3
            prev.PDI * (1.0 + 0.01 * dt)  # Broadening
        else
            max(1.2, prev.PDI * 0.99)     # Narrowing (oligomers uniform)
        end
        PDI_new = min(PDI_new, 3.0)

        Mn_new = Mw_new / PDI_new

        # === Derived Properties ===
        Tg_new = predict_Tg(Mw_new, params)
        crystallinity_new = predict_crystallinity(Mw_new, t, params)

        # Need iterative update for coupled properties
        porosity_temp = predict_porosity(Mw_new, prev.mass_fraction < 1.0 ? 1.0 - prev.mass_fraction : 0.0, params)
        pH_new = predict_pH_internal(Mw_new, porosity_temp, params, env)
        mass_loss = predict_mass_loss(Mw_new, porosity_temp, t, params)
        porosity_new = predict_porosity(Mw_new, mass_loss, params)
        E_new = predict_modulus(Mw_new, crystallinity_new, porosity_new, params)

        state_new = DegradationState(
            t,
            Mw_new,
            Mn_new,
            PDI_new,
            Tg_new,
            crystallinity_new,
            1.0 - mass_loss,
            porosity_new,
            pH_new,
            E_new,
            k_eff
        )
        push!(states, state_new)
    end

    return states
end

"""
Predict properties at a specific time point.
"""
function predict_properties_at_time(
    params::PLDLAParameters,
    env::EnvironmentConditions,
    time_days::Float64
)
    time_points = collect(0.0:1.0:time_days)
    if time_points[end] != time_days
        push!(time_points, time_days)
    end

    states = simulate_degradation(params, env, time_points)
    return states[end]
end

# =============================================================================
# Validation Functions
# =============================================================================

"""
Experimental data from Kaique Hergesel (2025).
"""
const KAIQUE_EXPERIMENTAL_DATA = Dict(
    "PLDLA" => Dict(
        :Mw => Dict(0 => 94.4, 30 => 52.7, 60 => 35.9, 90 => 11.8),
        :Mn => Dict(0 => 51.3, 30 => 25.4, 60 => 18.3, 90 => 7.9),
        :Tg => Dict(0 => 54.0, 30 => 54.0, 60 => 48.0, 90 => 36.0),
        :E => Dict(0 => 2.67),
    ),
    "PLDLA/TEC1%" => Dict(
        :Mw => Dict(0 => 85.8, 30 => 31.6, 60 => 22.4, 90 => 12.1),
        :Mn => Dict(0 => 45.0, 30 => 19.3, 60 => 11.7, 90 => 8.1),
        :Tg => Dict(0 => 49.0, 30 => 49.0, 60 => 38.0, 90 => 41.0),
        :E => Dict(0 => 1.90),
    ),
    "PLDLA/TEC2%" => Dict(
        :Mw => Dict(0 => 68.4, 30 => 26.9, 60 => 19.4, 90 => 8.4),
        :Mn => Dict(0 => 32.7, 30 => 15.0, 60 => 12.6, 90 => 6.6),
        :Tg => Dict(0 => 46.0, 30 => 44.0, 60 => 22.0, 90 => 35.0),
        :E => Dict(0 => 1.60),
    )
)

"""
    validate_against_experimental(material_name)

Compare model predictions with Kaique's experimental data.

Returns dict with errors and statistics.
"""
function validate_against_experimental(material_name::String="PLDLA")
    # Get experimental data
    exp_data = KAIQUE_EXPERIMENTAL_DATA[material_name]

    # Setup parameters
    TEC_pct = if material_name == "PLDLA"
        0.0
    elseif material_name == "PLDLA/TEC1%"
        1.0
    else
        2.0
    end

    params = PLDLAParameters(
        Mw_initial = exp_data[:Mw][0],
        Mn_initial = exp_data[:Mn][0],
        Tg_initial = exp_data[:Tg][0] + 5.0 * TEC_pct,  # Base Tg before TEC
        E_initial = exp_data[:E][0] / (1.0 - 0.15 * TEC_pct),  # Base E before TEC
        TEC_weight_percent = TEC_pct,
        name = material_name
    )

    env = EnvironmentConditions()
    time_points = Float64[0, 30, 60, 90]

    # Run simulation
    states = simulate_degradation(params, env, time_points)

    # Compare
    results = Dict{Symbol, Vector{NamedTuple}}()

    # Mw comparison
    mw_comparison = NamedTuple[]
    for (i, t) in enumerate(time_points)
        t_int = Int(t)
        if haskey(exp_data[:Mw], t_int)
            exp_val = exp_data[:Mw][t_int]
            pred_val = states[i].Mw
            error_pct = abs(pred_val - exp_val) / exp_val * 100
            push!(mw_comparison, (time=t_int, experimental=exp_val, predicted=pred_val, error_pct=error_pct))
        end
    end
    results[:Mw] = mw_comparison

    # Tg comparison
    tg_comparison = NamedTuple[]
    for (i, t) in enumerate(time_points)
        t_int = Int(t)
        if haskey(exp_data[:Tg], t_int)
            exp_val = exp_data[:Tg][t_int]
            pred_val = states[i].Tg
            error_pct = abs(pred_val - exp_val) / exp_val * 100
            push!(tg_comparison, (time=t_int, experimental=exp_val, predicted=pred_val, error_pct=error_pct))
        end
    end
    results[:Tg] = tg_comparison

    # Calculate summary statistics
    mw_errors = [r.error_pct for r in mw_comparison if r.time > 0]
    tg_errors = [r.error_pct for r in tg_comparison if r.time > 0]

    summary = Dict(
        :material => material_name,
        :Mw_mean_error => isempty(mw_errors) ? NaN : mean(mw_errors),
        :Mw_max_error => isempty(mw_errors) ? NaN : maximum(mw_errors),
        :Tg_mean_error => isempty(tg_errors) ? NaN : mean(tg_errors),
        :Tg_max_error => isempty(tg_errors) ? NaN : maximum(tg_errors),
    )

    return Dict(:comparisons => results, :summary => summary, :states => states)
end

"""
Print validation results in formatted table.
"""
function print_validation_results(results::Dict)
    summary = results[:summary]
    comparisons = results[:comparisons]

    println("\n" * "="^70)
    println("VALIDATION: $(summary[:material])")
    println("="^70)

    # Mw table
    println("\nMolecular Weight (Mw, kg/mol):")
    println("-"^50)
    @printf("  %8s | %12s | %10s | %8s\n", "Time(d)", "Experimental", "Predicted", "Error")
    println("  " * "-"^46)
    for r in comparisons[:Mw]
        @printf("  %8d | %12.1f | %10.1f | %7.1f%%\n", r.time, r.experimental, r.predicted, r.error_pct)
    end
    @printf("\n  Mean Error: %.1f%%\n", summary[:Mw_mean_error])

    # Tg table
    println("\nGlass Transition Temperature (Tg, °C):")
    println("-"^50)
    @printf("  %8s | %12s | %10s | %8s\n", "Time(d)", "Experimental", "Predicted", "Error")
    println("  " * "-"^46)
    for r in comparisons[:Tg]
        @printf("  %8d | %12.1f | %10.1f | %7.1f%%\n", r.time, r.experimental, r.predicted, r.error_pct)
    end
    @printf("\n  Mean Error: %.1f%%\n", summary[:Tg_mean_error])

    println("\n" * "="^70)
end

# =============================================================================
# Plotting Functions (ASCII for terminal)
# =============================================================================

"""
Generate ASCII plot of degradation curves.
"""
function plot_degradation_curves(states::Vector{DegradationState}; width=60, height=15)
    times = [s.time_days for s in states]
    Mw_values = [s.Mw for s in states]
    Mw_initial = Mw_values[1]

    # Normalize to percentage
    Mw_pct = Mw_values ./ Mw_initial .* 100

    println("\nMolecular Weight Decay (% of initial)")
    println("="^(width+10))

    # Create grid
    y_max = 100.0
    y_min = 0.0
    x_max = maximum(times)
    x_min = 0.0

    for row in height:-1:1
        y_val = y_min + (y_max - y_min) * row / height

        # Y-axis label
        @printf("%6.0f |", y_val)

        # Plot points
        line = fill(' ', width)
        for (i, (t, mw)) in enumerate(zip(times, Mw_pct))
            x_pos = Int(round((t - x_min) / (x_max - x_min) * (width - 1))) + 1
            y_lower = y_min + (y_max - y_min) * (row - 1) / height
            y_upper = y_min + (y_max - y_min) * row / height

            if y_lower <= mw < y_upper
                line[x_pos] = '*'
            end
        end
        println(String(line))
    end

    # X-axis
    println("       +" * "-"^width)
    @printf("        0%s%d days\n", " "^(width-10), Int(x_max))
end

"""
Print comprehensive degradation summary.
"""
function print_degradation_summary(states::Vector{DegradationState}, params::PLDLAParameters)
    println("\n" * "="^70)
    println("DEGRADATION SIMULATION SUMMARY: $(params.name)")
    println("="^70)

    println("\nInitial Conditions:")
    println("  Mw = $(params.Mw_initial) kg/mol")
    println("  Tg = $(params.Tg_initial)°C")
    println("  Porosity = $(params.porosity_initial * 100)%")
    println("  TEC = $(params.TEC_weight_percent)%")

    println("\nTime Evolution:")
    println("-"^70)
    @printf("  %6s | %8s | %6s | %6s | %8s | %6s | %6s\n",
            "Day", "Mw", "Tg", "X_c", "Mass", "Poros", "E")
    @printf("  %6s | %8s | %6s | %6s | %8s | %6s | %6s\n",
            "", "(kg/mol)", "(°C)", "(%)", "(%)", "(%)", "(MPa)")
    println("  " * "-"^66)

    for s in states
        @printf("  %6.0f | %8.1f | %6.1f | %6.1f | %8.1f | %6.1f | %6.2f\n",
                s.time_days, s.Mw, s.Tg, s.crystallinity*100,
                s.mass_fraction*100, s.porosity*100, s.E_modulus)
    end

    # Key milestones
    println("\nKey Milestones:")

    # Half-life
    for (i, s) in enumerate(states)
        if s.Mw <= params.Mw_initial / 2
            println("  Mw half-life: ~$(Int(s.time_days)) days")
            break
        end
    end

    # Tg below body temperature
    for s in states
        if s.Tg < 37.0
            println("  Tg < 37°C at: ~$(Int(s.time_days)) days (material softens in vivo)")
            break
        end
    end

    # Mass loss onset
    for s in states
        if s.mass_fraction < 0.99
            println("  Mass loss onset: ~$(Int(s.time_days)) days")
            break
        end
    end

    println("\n" * "="^70)
end

end # module
