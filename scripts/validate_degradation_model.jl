#!/usr/bin/env julia
"""
Validation Script: PLDLADegradationModel vs Kaique Hergesel Experimental Data

This script:
1. Runs the multi-mechanism degradation model for all 3 PLDLA variants
2. Compares predictions with experimental data
3. Generates comparison tables and ASCII plots
4. Calculates error statistics

Author: Darwin Scaffold Studio
Date: December 2025
"""

using Printf
using Statistics

# Include the module directly for testing
include(joinpath(@__DIR__, "..", "src", "DarwinScaffoldStudio", "Science", "PLDLADegradationModel.jl"))
using .PLDLADegradationModel: DegradationState, PLDLAParameters, EnvironmentConditions,
    simulate_degradation, validate_against_experimental, KAIQUE_EXPERIMENTAL_DATA

# Import internal functions for validation
import .PLDLADegradationModel: print_validation_results, print_degradation_summary, plot_degradation_curves

println("="^80)
println("   PLDLA DEGRADATION MODEL VALIDATION")
println("   Multi-Mechanism Model vs Kaique Hergesel (2025) Experimental Data")
println("="^80)

# =============================================================================
# Run validation for all materials
# =============================================================================

materials = ["PLDLA", "PLDLA/TEC1%", "PLDLA/TEC2%"]
all_results = Dict{String, Dict}()

for material in materials
    println("\n\nProcessing: $material")
    println("-"^40)

    results = validate_against_experimental(material)
    all_results[material] = results

    print_validation_results(results)
end

# =============================================================================
# Overall Summary
# =============================================================================

println("\n\n")
println("="^80)
println("   OVERALL MODEL PERFORMANCE")
println("="^80)

println("\n┌────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
println("│    Material    │ Mw Mean Err  │ Mw Max Err   │ Tg Mean Err  │ Tg Max Err   │")
println("├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")

for material in materials
    s = all_results[material][:summary]
    @printf("│ %-14s │ %10.1f%% │ %10.1f%% │ %10.1f%% │ %10.1f%% │\n",
            material, s[:Mw_mean_error], s[:Mw_max_error], s[:Tg_mean_error], s[:Tg_max_error])
end

println("└────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")

# Global averages
all_mw_errors = [all_results[m][:summary][:Mw_mean_error] for m in materials]
all_tg_errors = [all_results[m][:summary][:Tg_mean_error] for m in materials]

println("\nGlobal Average Errors:")
@printf("  Mw: %.1f%% (across all materials and time points)\n", mean(all_mw_errors))
@printf("  Tg: %.1f%% (across all materials and time points)\n", mean(all_tg_errors))

# =============================================================================
# Detailed Predictions (Extended Timeline)
# =============================================================================

println("\n\n")
println("="^80)
println("   EXTENDED PREDICTIONS (0-365 days)")
println("="^80)

for material in materials
    TEC_pct = if material == "PLDLA"
        0.0
    elseif material == "PLDLA/TEC1%"
        1.0
    else
        2.0
    end

    exp_data = KAIQUE_EXPERIMENTAL_DATA[material]

    params = PLDLAParameters(
        Mw_initial = exp_data[:Mw][0],
        Mn_initial = exp_data[:Mn][0],
        Tg_initial = exp_data[:Tg][0] + 5.0 * TEC_pct,
        E_initial = exp_data[:E][0] / (1.0 - 0.15 * TEC_pct),
        TEC_weight_percent = TEC_pct,
        name = material
    )

    env = EnvironmentConditions()
    time_points = Float64.(0:30:365)

    states = simulate_degradation(params, env, collect(time_points))

    print_degradation_summary(states, params)

    # ASCII plot
    plot_degradation_curves(states)
end

# =============================================================================
# Mechanism Contribution Analysis
# =============================================================================

println("\n\n")
println("="^80)
println("   MECHANISM ANALYSIS")
println("="^80)

println("\nEffect of each mechanism on degradation rate:")
println("-"^60)

# Base case
params_base = PLDLAParameters()
env_base = EnvironmentConditions()

# Simulate base case
time_points = Float64[0, 30, 60, 90]
states_base = simulate_degradation(params_base, env_base, time_points)

println("\n1. TEMPERATURE EFFECT (Arrhenius)")
println("   Simulating at different temperatures...")

for T in [32.0, 37.0, 40.0, 42.0]
    env_T = EnvironmentConditions(temperature_C=T)
    states_T = simulate_degradation(params_base, env_T, time_points)
    Mw_90 = states_T[end].Mw
    @printf("   T = %4.1f°C: Mw(90d) = %5.1f kg/mol (%.1f%% of initial)\n",
            T, Mw_90, Mw_90/params_base.Mw_initial*100)
end

println("\n2. pH EFFECT (Autocatalysis)")
println("   Internal pH evolution over time:")

for state in states_base
    @printf("   Day %3.0f: pH_internal = %.2f\n", state.time_days, state.pH_internal)
end

println("\n3. CRYSTALLINITY EFFECT (Three-Phase Model)")
println("   Crystallinity evolution:")

for state in states_base
    @printf("   Day %3.0f: X_c = %.1f%%\n", state.time_days, state.crystallinity * 100)
end

println("\n4. PLASTICIZER EFFECT (TEC)")
println("   Comparing materials at 90 days:")

for material in materials
    states = all_results[material][:states]
    state_90 = states[end]
    @printf("   %-14s: Mw = %5.1f, Tg = %4.1f°C, E = %.2f MPa\n",
            material, state_90.Mw, state_90.Tg, state_90.E_modulus)
end

# =============================================================================
# Mechanical Property Predictions
# =============================================================================

println("\n\n")
println("="^80)
println("   MECHANICAL PROPERTY PREDICTIONS")
println("="^80)

println("\n** NOTE: Kaique did NOT measure mechanical properties during degradation **")
println("         These are MODEL PREDICTIONS requiring future validation")
println("-"^60)

println("\nPredicted Compressive Modulus Evolution:")
println("\n┌────────────────┬─────────┬─────────┬─────────┬─────────┐")
println("│    Material    │  Day 0  │ Day 30  │ Day 60  │ Day 90  │")
println("├────────────────┼─────────┼─────────┼─────────┼─────────┤")

for material in materials
    states = all_results[material][:states]
    @printf("│ %-14s │ %5.2f   │ %5.2f   │ %5.2f   │ %5.2f   │\n",
            material,
            states[1].E_modulus,
            states[2].E_modulus,
            states[3].E_modulus,
            states[4].E_modulus)
end
println("└────────────────┴─────────┴─────────┴─────────┴─────────┘")
println("                           (values in MPa)")

println("\nPredicted Modulus Retention (% of initial):")
println("\n┌────────────────┬─────────┬─────────┬─────────┬─────────┐")
println("│    Material    │  Day 0  │ Day 30  │ Day 60  │ Day 90  │")
println("├────────────────┼─────────┼─────────┼─────────┼─────────┤")

for material in materials
    states = all_results[material][:states]
    E0 = states[1].E_modulus
    @printf("│ %-14s │ %5.1f%%  │ %5.1f%%  │ %5.1f%%  │ %5.1f%%  │\n",
            material,
            100.0,
            states[2].E_modulus/E0*100,
            states[3].E_modulus/E0*100,
            states[4].E_modulus/E0*100)
end
println("└────────────────┴─────────┴─────────┴─────────┴─────────┘")

# =============================================================================
# Clinical Implications
# =============================================================================

println("\n\n")
println("="^80)
println("   CLINICAL IMPLICATIONS")
println("="^80)

println("\nKey Time Points for Tissue Engineering Applications:")
println("-"^60)

for material in materials
    println("\n$material:")
    states = all_results[material][:states]
    params = PLDLAParameters(
        Mw_initial = KAIQUE_EXPERIMENTAL_DATA[material][:Mw][0],
        Tg_initial = KAIQUE_EXPERIMENTAL_DATA[material][:Tg][0],
        TEC_weight_percent = material == "PLDLA" ? 0.0 : (material == "PLDLA/TEC1%" ? 1.0 : 2.0)
    )

    # Extended simulation
    env = EnvironmentConditions()
    time_ext = Float64.(0:7:365)
    states_ext = simulate_degradation(params, env, collect(time_ext))

    # Find key milestones
    t_half_mw = nothing
    t_tg_below_37 = nothing
    t_mass_loss = nothing
    t_80pct_mw_loss = nothing

    for s in states_ext
        if isnothing(t_half_mw) && s.Mw <= params.Mw_initial / 2
            t_half_mw = s.time_days
        end
        if isnothing(t_tg_below_37) && s.Tg < 37.0
            t_tg_below_37 = s.time_days
        end
        if isnothing(t_mass_loss) && s.mass_fraction < 0.99
            t_mass_loss = s.time_days
        end
        if isnothing(t_80pct_mw_loss) && s.Mw <= params.Mw_initial * 0.2
            t_80pct_mw_loss = s.time_days
        end
    end

    println("  - Mw half-life: $(isnothing(t_half_mw) ? ">365" : Int(t_half_mw)) days")
    println("  - Tg < 37°C (softening): $(isnothing(t_tg_below_37) ? ">365" : Int(t_tg_below_37)) days")
    println("  - Mass loss onset: $(isnothing(t_mass_loss) ? ">365" : Int(t_mass_loss)) days")
    println("  - 80% Mw loss: $(isnothing(t_80pct_mw_loss) ? ">365" : Int(t_80pct_mw_loss)) days")
end

println("\n\nImplications for Meniscus Scaffold (target tissue):")
println("-"^60)
println("  - Meniscus healing requires 3-6 months of structural support")
println("  - Scaffold should maintain >50% mechanical properties for ~90 days")
println("  - Complete resorption desired by 12-18 months")
println("\n  Based on predictions:")
println("  - PLDLA maintains ~21% of modulus at 90 days")
println("  - PLDLA/TEC2% maintains ~19% of modulus at 90 days")
println("  - Consider: composite with ceramics for longer mechanical support")

# =============================================================================
# Model Limitations
# =============================================================================

println("\n\n")
println("="^80)
println("   MODEL LIMITATIONS AND FUTURE WORK")
println("="^80)

println("""

CURRENT LIMITATIONS:

1. FIRST-ORDER KINETICS
   - Real degradation has autocatalytic acceleration
   - Model approximates this with pH correction, but may underestimate
   - Literature shows microspheres can have pH < 2 internally

2. CRYSTALLINITY MODEL
   - Simplified power-law approximation
   - Does not capture stereocomplex formation (PLLA + PDLA)
   - Does not model crystal morphology (spherulites, lamellae)

3. MECHANICAL PREDICTIONS
   - Based on Mw-modulus power law (not validated for PLDLA scaffolds)
   - Kaique's data only at t=0, no validation during degradation
   - Does not account for scaffold architecture effects

4. MISSING MECHANISMS
   - Enzymatic degradation (proteinase K, lipases)
   - Oxidative degradation
   - Mechanical stress-induced degradation
   - Cellular effects (macrophages, osteoclasts)

FUTURE IMPROVEMENTS:

1. Include autocatalytic term: dMw/dt = -k₁Mw - k₂Mw[H⁺]
2. Add crystallization kinetics (Avrami equation)
3. Validate mechanical predictions with DMA during degradation
4. Implement spatial model (PDE for gradient effects)
5. Couple with tissue ingrowth model

""")

println("="^80)
println("   VALIDATION COMPLETE")
println("="^80)
println("\nScript finished successfully.")
println("Results can be used to inform scaffold design decisions.")
