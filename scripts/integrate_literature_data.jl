#!/usr/bin/env julia
"""
Script para integrar dados de degradação PLDLA da literatura.

Este script será atualizado com os dados encontrados pelos agentes de busca.
"""

using Printf
using Statistics

# =============================================================================
# DADOS EXPERIMENTAIS DO KAIQUE (baseline)
# =============================================================================

KAIQUE_DATA = Dict(
    "PLDLA" => (
        source = "Kaique PhD thesis",
        Mn = [51.3, 25.4, 18.3, 7.9],
        Mw = [94.4, 52.7, 35.9, 11.8],
        Tg = [54.0, 54.0, 48.0, 36.0],
        t = [0.0, 30.0, 60.0, 90.0],
        conditions = "37°C, PBS pH 7.4"
    ),
    "PLDLA/TEC1%" => (
        source = "Kaique PhD thesis",
        Mn = [45.0, 19.3, 11.7, 8.1],
        Mw = [85.8, 31.6, 22.4, 12.1],
        Tg = [49.0, 49.0, 38.0, 41.0],
        t = [0.0, 30.0, 60.0, 90.0],
        conditions = "37°C, PBS pH 7.4, 1% TEC"
    ),
    "PLDLA/TEC2%" => (
        source = "Kaique PhD thesis",
        Mn = [32.7, 15.0, 12.6, 6.6],
        Mw = [68.4, 26.9, 19.4, 8.4],
        Tg = [46.0, 44.0, 22.0, 35.0],  # NOTE: t=60 Tg=22°C is anomalous!
        t = [0.0, 30.0, 60.0, 90.0],
        conditions = "37°C, PBS pH 7.4, 2% TEC"
    )
)

# =============================================================================
# DADOS DA LITERATURA (a serem preenchidos pelos agentes)
# =============================================================================

# Placeholder - será atualizado com dados reais
LITERATURE_DATA = Dict{String, NamedTuple}()

# Template para adicionar novos datasets:
#=
LITERATURE_DATA["Author2024_PLLA"] = (
    source = "Author et al., Journal, 2024",
    doi = "10.1000/xxxxx",
    Mn = [100.0, 80.0, 60.0, 40.0, 20.0],  # kg/mol
    Mw = [...],  # kg/mol
    Tg = [...],  # °C
    t = [0, 30, 60, 90, 120],  # days
    conditions = "37°C, PBS pH 7.4",
    notes = "Any relevant notes"
)
=#

# =============================================================================
# FOX-FLORY PARAMETERS FROM LITERATURE
# =============================================================================

FOX_FLORY_PARAMS = Dict(
    # Placeholder - será atualizado
    "PLA_generic" => (
        Tg_inf = 55.0,  # °C
        K = 55.0,       # kg/mol
        source = "Literature compilation"
    )
)

# =============================================================================
# HYDROLYSIS RATE CONSTANTS FROM LITERATURE
# =============================================================================

HYDROLYSIS_RATES = Dict(
    # Placeholder - será atualizado
    "PMC3359772" => (
        k = 0.020,  # day⁻¹
        T = 37.0,   # °C
        polymer = "PLA",
        source = "PMC3359772"
    )
)

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

"""
Calculate effective hydrolysis rate from Mn decay data.
"""
function calculate_k_from_data(Mn::Vector{Float64}, t::Vector{Float64})
    # For first-order decay: Mn(t) = Mn0 * exp(-k*t)
    # ln(Mn/Mn0) = -k*t

    Mn0 = Mn[1]
    rates = Float64[]

    for i in 2:length(t)
        if Mn[i] > 0 && t[i] > 0
            k_apparent = -log(Mn[i]/Mn0) / t[i]
            push!(rates, k_apparent)
        end
    end

    return mean(rates), std(rates)
end

"""
Analyze all datasets and compare rate constants.
"""
function analyze_all_datasets()
    println("="^70)
    println("ANALYSIS OF DEGRADATION RATE CONSTANTS")
    println("="^70)

    all_datasets = merge(KAIQUE_DATA, LITERATURE_DATA)

    println("\n┌────────────────────────┬──────────────────┬──────────────────┐")
    println("│ Dataset                │ k_eff (day⁻¹)    │ Half-life (days) │")
    println("├────────────────────────┼──────────────────┼──────────────────┤")

    k_values = Float64[]

    for (name, data) in all_datasets
        k_mean, k_std = calculate_k_from_data(data.Mn, data.t)
        t_half = log(2) / k_mean
        push!(k_values, k_mean)

        @printf("│ %-22s │ %.4f ± %.4f   │ %6.1f           │\n",
                name[1:min(22,length(name))], k_mean, k_std, t_half)
    end

    println("└────────────────────────┴──────────────────┴──────────────────┘")

    println("\nSummary statistics:")
    @printf("  Mean k: %.4f ± %.4f day⁻¹\n", mean(k_values), std(k_values))
    @printf("  Range: %.4f - %.4f day⁻¹\n", minimum(k_values), maximum(k_values))

    return k_values
end

"""
Identify outliers in Tg data.
"""
function analyze_tg_anomalies()
    println("\n" * "="^70)
    println("ANALYSIS OF Tg ANOMALIES")
    println("="^70)

    all_datasets = merge(KAIQUE_DATA, LITERATURE_DATA)

    for (name, data) in all_datasets
        if haskey(data, :Tg)
            # Check for non-monotonic Tg
            Tg = data.Tg
            for i in 2:length(Tg)-1
                if Tg[i] < Tg[i-1] && Tg[i] < Tg[i+1]
                    println("\n⚠ ANOMALY DETECTED in $name:")
                    @printf("  t=%d: Tg=%.1f°C (minimum)\n", Int(data.t[i]), Tg[i])
                    @printf("  t=%d: Tg=%.1f°C (before)\n", Int(data.t[i-1]), Tg[i-1])
                    @printf("  t=%d: Tg=%.1f°C (after)\n", Int(data.t[i+1]), Tg[i+1])
                    println("  This could indicate:")
                    println("    - Measurement artifact (cold crystallization)")
                    println("    - Plasticizer accumulation at this time point")
                    println("    - Phase separation")
                end
            end
        end
    end
end

# =============================================================================
# MAIN
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("Analyzing degradation datasets...\n")

    k_values = analyze_all_datasets()
    analyze_tg_anomalies()

    println("\n" * "="^70)
    println("Aguardando dados da literatura para atualizar este script...")
    println("="^70)
end
