#!/usr/bin/env julia
"""
validate_newton_2025_data.jl

Validação dos módulos com dados densos do Newton 2025.
41 polímeros × 25 pontos = 1025 pontos de dados

Testa:
1. NEAT-GP - Descoberta simbólica (evolve_gp!)
2. QuaternionPhysics - Trajetórias (quaternion_trajectory)
3. Granger Causality - Com dados suficientes (25 pontos)
"""

using LinearAlgebra
using Statistics
using Printf
using Dates

println("="^70)
println("  VALIDAÇÃO COM DADOS DENSOS - Newton 2025")
println("  41 polímeros × 25 pontos = 1025 pontos totais")
println("="^70)
println()

# ============================================================================
# LOAD NEWTON 2025 DATABASE
# ============================================================================

include(joinpath(@__DIR__, "..", "data", "literature", "newton_2025_database.jl"))

println("✓ Database carregado: $(length(NEWTON_2025_POLYMERS)) polímeros")
println()

# ============================================================================
# LOAD MODULES
# ============================================================================

include(joinpath(@__DIR__, "..", "src", "DarwinScaffoldStudio", "Science", "NEATGP.jl"))
include(joinpath(@__DIR__, "..", "src", "DarwinScaffoldStudio", "Science", "QuaternionPhysics.jl"))
include(joinpath(@__DIR__, "..", "src", "DarwinScaffoldStudio", "Science", "DeepScientificDiscovery.jl"))

using .NEATGP
using .QuaternionPhysics
using .DeepScientificDiscovery

println("✓ Módulos carregados")
println()

# ============================================================================
# 1. VALIDAÇÃO NEAT-GP
# ============================================================================

println("="^70)
println("  1. NEAT-GP - Descoberta Simbólica de Equações")
println("="^70)
println()

# Test on subset of polymers (faster)
test_polymers = NEWTON_2025_POLYMERS[1:10]

neatgp_results = NamedTuple{(:name, :model, :r2, :rmse, :mw0, :timescale),
                            Tuple{String, Symbol, Float64, Float64, Float64, Float64}}[]

for polymer in test_polymers
    series = generate_best_fit_series(polymer; n_points=25)

    try
        # Prepare data for NEAT-GP
        times = collect(Float64, series.times)
        MW = collect(Float64, series.MW)

        # Create GP configuration
        config = GPConfig(
            population_size=30,
            max_generations=20,
            input_names=["t"]
        )

        # Create population
        pop = GPPopulation(config)

        # Evolve
        best = evolve_gp!(pop, times, MW; verbose=false)

        # Calculate R²
        if best !== nothing
            r2 = 1.0 - best.mse / var(MW)
            rmse = sqrt(best.mse)
        else
            r2 = 0.0
            rmse = Inf
        end

        push!(neatgp_results, (
            name=polymer.name,
            model=series.model,
            r2=max(0.0, r2),
            rmse=rmse,
            mw0=polymer.initial_mw_kda,
            timescale=polymer.degradation_timescale_days
        ))

        print(".")

    catch e
        println("\n  [WARN] NEAT-GP erro em $(polymer.name): $(typeof(e))")
        push!(neatgp_results, (
            name=polymer.name,
            model=series.model,
            r2=0.0,
            rmse=Inf,
            mw0=polymer.initial_mw_kda,
            timescale=polymer.degradation_timescale_days
        ))
    end
end
println()

# Statistics
r2_values = [r.r2 for r in neatgp_results]
rmse_values = [r.rmse for r in neatgp_results if isfinite(r.rmse)]
successful = count(r -> r.r2 > 0.95, neatgp_results)

println("\nNEAT-GP Results (10 polímeros teste):")
println("-"^50)
println(@sprintf("  R² médio: %.4f ± %.4f", mean(r2_values), std(r2_values)))
println(@sprintf("  R² mediano: %.4f", median(r2_values)))
if !isempty(rmse_values)
    println(@sprintf("  RMSE médio: %.2f kDa", mean(rmse_values)))
end
println(@sprintf("  R² > 0.95: %d/%d (%.1f%%)",
    successful, length(neatgp_results), 100 * successful / length(neatgp_results)))
println()

# ============================================================================
# 2. VALIDAÇÃO QUATERNION PHYSICS
# ============================================================================

println("="^70)
println("  2. QuaternionPhysics - Espaço de Fase Quaterniônico")
println("="^70)
println()

quaternion_results = NamedTuple{(:name, :model, :arc_length, :mean_curvature, :is_geodesic),
                               Tuple{String, Symbol, Float64, Float64, Bool}}[]

for polymer in NEWTON_2025_POLYMERS
    series = generate_best_fit_series(polymer; n_points=25)

    try
        # Create quaternion trajectory
        times = collect(Float64, series.times)
        Mn_values = collect(Float64, series.MW)

        # Estimate crystallinity from MW decay
        Xc_values = 0.5 .* (1.0 .- Mn_values ./ maximum(Mn_values)) .+ 0.1

        # Estimate enthalpy (proportional to crystallinity)
        H_values = 100.0 .* Xc_values

        # Create trajectory using correct function
        trajectory = quaternion_trajectory(times, Mn_values, Xc_values, H_values)

        # Analyze
        arc_length = trajectory.total_arc_length
        mean_curv = mean(trajectory.curvature)
        is_geodesic = maximum(trajectory.curvature) < 0.5

        push!(quaternion_results, (
            name=polymer.name,
            model=series.model,
            arc_length=arc_length,
            mean_curvature=mean_curv,
            is_geodesic=is_geodesic
        ))

    catch e
        if polymer.id <= 3
            println("  [DEBUG] Quaternion erro em $(polymer.name): $(typeof(e))")
        end
        push!(quaternion_results, (
            name=polymer.name,
            model=series.model,
            arc_length=NaN,
            mean_curvature=NaN,
            is_geodesic=false
        ))
    end
end

# Statistics
valid_results = filter(r -> !isnan(r.arc_length), quaternion_results)
geodesic_count = count(r -> r.is_geodesic, valid_results)

println("Quaternion Analysis Results:")
println("-"^50)
println(@sprintf("  Trajetórias analisadas: %d/%d", length(valid_results), length(NEWTON_2025_POLYMERS)))

if !isempty(valid_results)
    arc_lengths = [r.arc_length for r in valid_results]
    curvatures = [r.mean_curvature for r in valid_results]

    println(@sprintf("  Comprimento de arco médio: %.4f ± %.4f", mean(arc_lengths), std(arc_lengths)))
    println(@sprintf("  Curvatura média: %.4f ± %.4f", mean(curvatures), std(curvatures)))
    println(@sprintf("  Trajetórias geodésicas: %d/%d (%.1f%%)",
        geodesic_count, length(valid_results), 100 * geodesic_count / length(valid_results)))
end
println()

# ============================================================================
# 3. VALIDAÇÃO GRANGER CAUSALITY
# ============================================================================

println("="^70)
println("  3. Granger Causality - Com 25 pontos (> 20 mínimo)")
println("="^70)
println()

granger_results = NamedTuple{(:name, :model, :granger_fstat, :granger_pvalue, :significant),
                             Tuple{String, Symbol, Float64, Float64, Bool}}[]

for polymer in NEWTON_2025_POLYMERS
    series = generate_best_fit_series(polymer; n_points=25)

    try
        # Create time series
        MW = collect(Float64, series.MW)
        times = collect(Float64, series.times)

        # Calculate dMW/dt
        dMW_dt = vcat([0.0], diff(MW) ./ diff(times))

        # Test Granger causality: does dMW/dt Granger-cause MW?
        result = granger_causality_test(dMW_dt, MW; max_lag=3)

        # Get F-statistic (field is F_statistic not f_statistic)
        fstat = result.F_statistic

        push!(granger_results, (
            name=polymer.name,
            model=series.model,
            granger_fstat=fstat,
            granger_pvalue=result.p_value,
            significant=result.p_value < 0.05
        ))

    catch e
        if polymer.id <= 3
            println("  [DEBUG] Granger erro em $(polymer.name): $(typeof(e))")
        end
        push!(granger_results, (
            name=polymer.name,
            model=series.model,
            granger_fstat=NaN,
            granger_pvalue=1.0,
            significant=false
        ))
    end
end

# Statistics
significant_count = count(r -> r.significant, granger_results)
valid_granger = filter(r -> !isnan(r.granger_fstat), granger_results)

println("Granger Causality Results:")
println("-"^50)
println(@sprintf("  Testes válidos: %d/%d", length(valid_granger), length(NEWTON_2025_POLYMERS)))
println(@sprintf("  Relações significativas (p < 0.05): %d (%.1f%%)",
    significant_count, 100 * significant_count / length(granger_results)))

if !isempty(valid_granger)
    fstats = [r.granger_fstat for r in valid_granger]
    pvalues = [r.granger_pvalue for r in valid_granger]
    println(@sprintf("  F-statistic médio: %.2f ± %.2f", mean(fstats), std(fstats)))
    println(@sprintf("  p-value médio: %.4f", mean(pvalues)))
end
println()

# By scission mode
chain_end_sig = count(r -> r.significant && r.model == :chain_end, granger_results)
random_sig = count(r -> r.significant && r.model == :random, granger_results)
n_chain_end = count(r -> r.model == :chain_end, granger_results)
n_random = count(r -> r.model == :random, granger_results)

println("Por modo de cisão:")
println(@sprintf("  Chain-end: %d/%d significativas (%.1f%%)",
    chain_end_sig, n_chain_end, 100 * chain_end_sig / max(1, n_chain_end)))
println(@sprintf("  Random: %d/%d significativas (%.1f%%)",
    random_sig, n_random, 100 * random_sig / max(1, n_random)))
println()

# ============================================================================
# RESUMO FINAL
# ============================================================================

println("="^70)
println("  RESUMO FINAL - VALIDAÇÃO NEWTON 2025")
println("="^70)
println()

# Calculate overall validation scores
neatgp_valid = !isempty(r2_values) && mean(r2_values) > 0.80
quaternion_valid = length(valid_results) > 30
granger_valid = significant_count > 0

println("Módulos Validados:")
println("-"^50)
println(@sprintf("  NEAT-GP: %s (R² médio = %.4f)",
    neatgp_valid ? "✓ VALIDADO" : "⚠ PARCIAL",
    isempty(r2_values) ? 0.0 : mean(r2_values)))
println(@sprintf("  QuaternionPhysics: %s (%d/%d trajetórias)",
    quaternion_valid ? "✓ VALIDADO" : "⚠ PARCIAL",
    length(valid_results), length(NEWTON_2025_POLYMERS)))
println(@sprintf("  Granger Causality: %s (%d/%d significativas)",
    granger_valid ? "✓ VALIDADO" : "⚠ PARCIAL",
    significant_count, length(granger_results)))
println()

# Data density comparison
println("Comparação com validação anterior:")
println("-"^50)
println("  Antes (datasets esparsos):")
println("    - 24 polímeros × 4-7 pontos = ~130 pontos")
println("    - Granger: 0 relações detectadas (dados insuficientes)")
println()
println("  Agora (Newton 2025):")
println(@sprintf("    - 41 polímeros × 25 pontos = %d pontos", 41 * 25))
println(@sprintf("    - Granger: %d relações detectadas", significant_count))
println()

# Statistical improvement
if significant_count > 0
    println("MELHORIA: Granger causality agora funciona com dados densos!")
else
    println("NOTA: Mesmo com 25 pontos, relações causais podem não existir")
    println("      (MW e dMW/dt são trivialmente relacionados por definição)")
end
println()

# Save results
results_file = joinpath(@__DIR__, "..", "docs", "NEWTON_2025_VALIDATION_RESULTS.md")
open(results_file, "w") do f
    write(f, "# Resultados de Validação - Newton 2025\n\n")
    write(f, "**Data:** $(today())\n\n")
    write(f, "**Fonte:** Cheng et al., Newton 1, 100168 (2025)\n\n")

    write(f, "## Estatísticas Gerais\n\n")
    write(f, "- Total de polímeros: 41\n")
    write(f, "- Pontos por polímero: 25\n")
    write(f, "- Total de pontos: 1025\n\n")

    write(f, "## NEAT-GP (10 polímeros teste)\n\n")
    write(f, "| Métrica | Valor |\n")
    write(f, "|---------|-------|\n")
    write(f, @sprintf("| R² médio | %.4f ± %.4f |\n",
        isempty(r2_values) ? 0.0 : mean(r2_values),
        isempty(r2_values) ? 0.0 : std(r2_values)))
    write(f, @sprintf("| R² mediano | %.4f |\n",
        isempty(r2_values) ? 0.0 : median(r2_values)))
    write(f, "\n")

    write(f, "## QuaternionPhysics\n\n")
    write(f, "| Métrica | Valor |\n")
    write(f, "|---------|-------|\n")
    write(f, @sprintf("| Trajetórias válidas | %d/%d |\n", length(valid_results), 41))
    write(f, @sprintf("| Geodésicas | %d (%.1f%%) |\n", geodesic_count,
        100 * geodesic_count / max(1, length(valid_results))))
    write(f, "\n")

    write(f, "## Granger Causality\n\n")
    write(f, "| Métrica | Valor |\n")
    write(f, "|---------|-------|\n")
    write(f, @sprintf("| Testes válidos | %d/%d |\n", length(valid_granger), 41))
    write(f, @sprintf("| Relações significativas | %d (%.1f%%) |\n",
        significant_count, 100 * significant_count / max(1, length(granger_results))))
    write(f, "\n")

    write(f, "## Conclusão\n\n")
    if neatgp_valid && quaternion_valid
        write(f, "Validação bem sucedida com dados densos do Newton 2025.\n")
    else
        write(f, "Validação parcial - alguns módulos precisam de ajustes.\n")
    end
end

println("Resultados salvos em: $results_file")
println()
println("="^70)
println("  VALIDAÇÃO COMPLETA")
println("="^70)
