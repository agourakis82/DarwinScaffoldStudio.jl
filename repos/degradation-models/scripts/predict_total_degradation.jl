#!/usr/bin/env julia
# Predict total degradation time for PLDLA scaffolds

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using DegradationModels
using Printf

println("="^75)
println("  PREVISÃO DE DEGRADAÇÃO TOTAL - PLDLA 70:30 Scaffolds")
println("="^75)

# Train model
println("\nTreinando modelo SOTA...")
model = train(PLDLANeuralODEFast, epochs=2000, verbose=false)
println("Modelo treinado com sucesso!")

# Define degradation thresholds
# Mn < 5 kg/mol: scaffold loses mechanical integrity
# Mn < 2 kg/mol: essentially fully degraded (oligomers)
const Mn_INTEGRITY_LOSS = 5.0   # kg/mol - perda de integridade mecânica
const Mn_FULL_DEGRAD = 2.0     # kg/mol - degradação completa

"""
Find time to reach target Mn using binary search.
"""
function find_degradation_time(model, Mn0::Float64, Mn_target::Float64;
                                T::Float64=37.0, condition::Symbol=:in_vitro,
                                region::Union{Symbol,Nothing}=nothing,
                                t_max::Float64=500.0)
    t_low, t_high = 0.0, t_max

    # Check if target is reachable
    Mn_final = predict(model, Mn0, t_max, T=T, condition=condition, region=region)
    if Mn_final > Mn_target
        return NaN  # Not reachable in t_max
    end

    for _ in 1:50
        t_mid = (t_low + t_high) / 2
        Mn_mid = predict(model, Mn0, t_mid, T=T, condition=condition, region=region)

        if Mn_mid > Mn_target
            t_low = t_mid
        else
            t_high = t_mid
        end

        if abs(t_high - t_low) < 0.5
            break
        end
    end

    return (t_low + t_high) / 2
end

# =============================================================================
# ANÁLISE PARA DIFERENTES Mn0
# =============================================================================

println("\n" * "="^75)
println("  1. TEMPO PARA PERDA DE INTEGRIDADE MECÂNICA (Mn < 5 kg/mol)")
println("="^75)

conditions = [
    (:in_vitro, nothing, "In vitro (PBS, 37°C)"),
    (:in_vivo, :subcutaneous, "In vivo subcutâneo (35.5°C)"),
    (:in_vivo, :bone, "In vivo osso (37°C)"),
    (:in_vivo, :muscle, "In vivo músculo (37°C)"),
    (:in_vivo, :cartilage, "In vivo cartilagem (35°C)"),
    (:in_vivo, :inflammation, "In vivo inflamação (39°C)"),
]

println("\n  Mn0 = 50 kg/mol (típico PLDLA 3D-printed):")
println("  " * "-"^65)
@printf("  %-30s  %12s  %15s\n", "Condição", "t (dias)", "t (semanas)")
println("  " * "-"^65)

for (cond, region, label) in conditions
    t_degrad = find_degradation_time(model, 50.0, Mn_INTEGRITY_LOSS,
                                      condition=cond, region=region)
    if isnan(t_degrad)
        @printf("  %-30s  %12s  %15s\n", label, "> 500", "> 71")
    else
        @printf("  %-30s  %12.0f  %15.1f\n", label, t_degrad, t_degrad/7)
    end
end

println("\n  Mn0 = 40 kg/mol (PLDLA + TEC):")
println("  " * "-"^65)
for (cond, region, label) in conditions
    t_degrad = find_degradation_time(model, 40.0, Mn_INTEGRITY_LOSS,
                                      condition=cond, region=region)
    if isnan(t_degrad)
        @printf("  %-30s  %12s  %15s\n", label, "> 500", "> 71")
    else
        @printf("  %-30s  %12.0f  %15.1f\n", label, t_degrad, t_degrad/7)
    end
end

# =============================================================================
# DEGRADAÇÃO COMPLETA
# =============================================================================

println("\n" * "="^75)
println("  2. TEMPO PARA DEGRADAÇÃO COMPLETA (Mn < 2 kg/mol)")
println("="^75)

println("\n  Mn0 = 50 kg/mol:")
println("  " * "-"^65)
@printf("  %-30s  %12s  %15s\n", "Condição", "t (dias)", "t (meses)")
println("  " * "-"^65)

for (cond, region, label) in conditions
    t_degrad = find_degradation_time(model, 50.0, Mn_FULL_DEGRAD,
                                      condition=cond, region=region, t_max=1000.0)
    if isnan(t_degrad)
        @printf("  %-30s  %12s  %15s\n", label, "> 1000", "> 33")
    else
        @printf("  %-30s  %12.0f  %15.1f\n", label, t_degrad, t_degrad/30)
    end
end

# =============================================================================
# CURVA DE DEGRADAÇÃO COMPLETA
# =============================================================================

println("\n" * "="^75)
println("  3. CURVA DE DEGRADAÇÃO DETALHADA (Mn0 = 50 kg/mol)")
println("="^75)

println("\n  In vitro vs In vivo (osso):")
println("  " * "-"^65)
@printf("  %10s  %15s  %15s  %15s\n", "Tempo", "In vitro", "In vivo osso", "Diferença")
@printf("  %10s  %15s  %15s  %15s\n", "(dias)", "(kg/mol)", "(kg/mol)", "(%)")
println("  " * "-"^65)

for t in [0, 7, 14, 21, 30, 45, 60, 75, 90, 120, 150, 180]
    Mn_vitro = predict(model, 50.0, Float64(t), condition=:in_vitro)
    Mn_vivo = predict(model, 50.0, Float64(t), condition=:in_vivo, region=:bone)
    diff = (Mn_vitro - Mn_vivo) / Mn_vitro * 100

    @printf("  %10d  %15.1f  %15.1f  %+14.1f%%\n", t, Mn_vitro, Mn_vivo, -diff)
end

# =============================================================================
# PREVISÃO COM INCERTEZA
# =============================================================================

println("\n" * "="^75)
println("  4. DEGRADAÇÃO COM INTERVALO DE CONFIANÇA (95%)")
println("="^75)

println("\n  In vivo osso (Mn0 = 50 kg/mol):")
println("  " * "-"^65)
@printf("  %10s  %12s  %20s  %15s\n", "Tempo", "Mn médio", "95% CI", "Status")
println("  " * "-"^65)

for t in [30, 60, 90, 120, 150]
    r = predict(model, 50.0, Float64(t), condition=:in_vivo, region=:bone,
                with_uncertainty=true)

    status = if r.Mn > 20
        "Funcional"
    elseif r.Mn > 10
        "Degradando"
    elseif r.Mn > 5
        "Crítico"
    else
        "Reabsorvido"
    end

    @printf("  %8d d  %10.1f    [%5.1f - %5.1f]    %s\n",
            t, r.Mn, r.lower, r.upper, status)
end

# =============================================================================
# RECOMENDAÇÕES CLÍNICAS
# =============================================================================

println("\n" * "="^75)
println("  5. RECOMENDAÇÕES PARA DESIGN DE SCAFFOLDS")
println("="^75)

t_bone = find_degradation_time(model, 50.0, Mn_INTEGRITY_LOSS,
                                condition=:in_vivo, region=:bone)
t_cart = find_degradation_time(model, 50.0, Mn_INTEGRITY_LOSS,
                                condition=:in_vivo, region=:cartilage)

println("""

  PLDLA 70:30 3D-Printed Scaffold (Mn0 ≈ 50 kg/mol):

  ┌─────────────────────────────────────────────────────────────────────┐
  │  APLICAÇÃO              │  TEMPO SUPORTE  │  RECOMENDAÇÃO           │
  ├─────────────────────────────────────────────────────────────────────┤
  │  Regeneração óssea      │  ~$(round(Int, t_bone)) dias      │  Adequado para fraturas   │
  │  Cartilagem articular   │  ~$(round(Int, t_cart)) dias     │  Considerar Mn0 maior    │
  │  Liberação de fármaco   │  30-90 dias     │  Excelente controle      │
  │  Engenharia de tecidos  │  60-120 dias    │  Bom para regeneração    │
  └─────────────────────────────────────────────────────────────────────┘

  FATORES QUE ACELERAM A DEGRADAÇÃO:
  • Inflamação local: reduz t½ em ~30%
  • Temperatura elevada: +2°C → ~20% mais rápido
  • Enzimas (in vivo): ~35% mais rápido que in vitro

  FATORES QUE RETARDAM A DEGRADAÇÃO:
  • Cartilagem (35°C): ~20% mais lento que osso
  • Maior Mn0 inicial: prolonga suporte mecânico
  • Maior cristalinidade: reduz taxa de hidrólise

""")

println("="^75)
println("  Modelo SOTA pronto para otimização de scaffolds!")
println("="^75)
