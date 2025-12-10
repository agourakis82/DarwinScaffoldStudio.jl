#!/usr/bin/env julia
"""
Análise de Sensibilidade Morris (Elementary Effects)

Implementação do método Morris para identificar parâmetros mais influentes
no modelo de degradação de scaffolds poliméricos.

O método Morris é um método de screening global que:
1. Requer menos simulações que Sobol
2. Identifica parâmetros importantes vs não-importantes
3. Distingue efeitos lineares vs não-lineares

Referências:
- Morris (1991): Factorial sampling plans for preliminary computational experiments
- Campolongo et al. (2007): An effective screening design

Author: Darwin Scaffold Studio
Date: 2025-12-10
"""

using Printf
using Statistics
using Random
using Dates

# Incluir o módulo
include(joinpath(@__DIR__, "..", "src", "DarwinScaffoldStudio", "Science", "UnifiedScaffoldTissueModel.jl"))
using .UnifiedScaffoldTissueModel

Random.seed!(42)

println("="^90)
println("  ANÁLISE DE SENSIBILIDADE MORRIS")
println("  Método dos Efeitos Elementares para Modelo de Degradação")
println("="^90)

# ============================================================================
# DEFINIÇÃO DOS PARÂMETROS
# ============================================================================

"""
Estrutura para definição de parâmetro com ranges.
"""
struct ParameterDef
    name::String
    symbol::String
    min::Float64
    max::Float64
    unit::String
    description::String
end

# Parâmetros do modelo de degradação
const PARAMETERS = [
    ParameterDef("k0", "k₀", 0.005, 0.050, "/dia", "Taxa base de degradação"),
    ParameterDef("Ea", "Eₐ", 70.0, 95.0, "kJ/mol", "Energia de ativação"),
    ParameterDef("autocatalysis", "α", 0.01, 0.15, "-", "Fator de autocatálise"),
    ParameterDef("crystallinity", "Xc", 0.0, 0.70, "-", "Cristalinidade inicial"),
    ParameterDef("Mn_initial", "Mn₀", 30.0, 200.0, "kg/mol", "Massa molar inicial"),
    ParameterDef("porosity", "φ", 0.50, 0.85, "-", "Porosidade do scaffold"),
    ParameterDef("water_uptake", "w", 0.001, 0.05, "/dia", "Taxa de absorção de água"),
]

const N_PARAMS = length(PARAMETERS)

# ============================================================================
# FUNÇÕES DO MÉTODO MORRIS
# ============================================================================

"""
Gera trajetória Morris no espaço de parâmetros normalizado [0,1].
"""
function generate_morris_trajectory(n_params::Int, p::Int=4)
    # p = número de níveis (tipicamente 4)
    levels = collect(0:1/(p-1):1)

    # Ponto inicial aleatório
    x_base = rand(levels, n_params)

    # Matriz de trajetória (n_params+1 pontos)
    trajectory = zeros(n_params + 1, n_params)
    trajectory[1, :] = x_base

    # Ordem aleatória de perturbação
    order = randperm(n_params)

    # Incremento
    Δ = 1.0 / (p - 1)

    for i in 1:n_params
        trajectory[i+1, :] = trajectory[i, :]
        param_idx = order[i]

        # Perturbar para cima ou para baixo
        if trajectory[i, param_idx] + Δ <= 1.0
            trajectory[i+1, param_idx] += Δ
        else
            trajectory[i+1, param_idx] -= Δ
        end
    end

    return trajectory, order
end

"""
Desnormaliza parâmetros do espaço [0,1] para valores físicos.
"""
function denormalize_params(x_norm::Vector{Float64})::Dict{String, Float64}
    params = Dict{String, Float64}()
    for (i, p) in enumerate(PARAMETERS)
        params[p.name] = p.min + x_norm[i] * (p.max - p.min)
    end
    return params
end

"""
Função objetivo: calcula Mn em tempo fixo.
"""
function model_output(params::Dict{String, Float64}; t::Float64=90.0)::Float64
    scaffold = ScaffoldDesign(
        Mn_initial = params["Mn_initial"],
        porosity = params["porosity"],
        crystallinity = params["crystallinity"],
        k0 = params["k0"],
        Ea = params["Ea"],
        autocatalysis = params["autocatalysis"],
        polymer_type = :PLDLA  # tipo base
    )

    # Retorna Mn normalizado (fração do inicial)
    Mn = calculate_Mn_advanced(scaffold, t; use_polymer_params=false)
    return Mn / params["Mn_initial"]
end

"""
Calcula efeitos elementares (EE) para uma trajetória.
"""
function compute_elementary_effects(trajectory::Matrix{Float64}, order::Vector{Int})
    n_points = size(trajectory, 1)
    n_params = size(trajectory, 2)

    EE = zeros(n_params)

    # Calcular output para cada ponto da trajetória
    outputs = zeros(n_points)
    for i in 1:n_points
        params = denormalize_params(trajectory[i, :])
        outputs[i] = model_output(params)
    end

    # Calcular EE para cada parâmetro
    Δ = 1.0 / 3  # para p=4 níveis
    for i in 1:n_params
        param_idx = order[i]
        EE[param_idx] = (outputs[i+1] - outputs[i]) / Δ
    end

    return EE
end

"""
Executa análise Morris completa.
"""
function morris_analysis(n_trajectories::Int=20)
    println("\n📊 Executando análise Morris com $n_trajectories trajetórias...")
    println("-"^70)

    # Armazenar todos os efeitos elementares
    all_EE = zeros(n_trajectories, N_PARAMS)

    for r in 1:n_trajectories
        trajectory, order = generate_morris_trajectory(N_PARAMS)
        EE = compute_elementary_effects(trajectory, order)
        all_EE[r, :] = EE

        if r % 5 == 0
            @printf("  Trajetória %d/%d concluída\n", r, n_trajectories)
        end
    end

    # Calcular estatísticas Morris
    # μ* = média dos valores absolutos dos EE (importância global)
    # σ = desvio padrão dos EE (não-linearidade/interações)

    μ_star = vec(mean(abs.(all_EE), dims=1))
    σ = vec(std(all_EE, dims=1))
    μ = vec(mean(all_EE, dims=1))

    return μ_star, σ, μ, all_EE
end

# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

println("\n📋 PARÂMETROS ANALISADOS:")
println("-"^70)
for (i, p) in enumerate(PARAMETERS)
    @printf("  %d. %s (%s): [%.3f, %.3f] %s - %s\n",
            i, p.name, p.symbol, p.min, p.max, p.unit, p.description)
end

# Executar análise
μ_star, σ, μ, all_EE = morris_analysis(30)

# Ordenar por importância (μ*)
sorted_idx = sortperm(μ_star, rev=true)

println("\n\n" * "="^90)
println("  RESULTADOS DA ANÁLISE DE SENSIBILIDADE")
println("="^90)

println("\n┌──────────────────┬──────────┬──────────┬──────────┬─────────────────────────────┐")
println("│ Parâmetro        │    μ*    │    σ     │   μ*/σ   │ Interpretação               │")
println("├──────────────────┼──────────┼──────────┼──────────┼─────────────────────────────┤")

for i in sorted_idx
    p = PARAMETERS[i]
    ratio = σ[i] > 0.001 ? μ_star[i] / σ[i] : Inf

    # Interpretação baseada em μ* e σ
    if μ_star[i] > 0.1
        if σ[i] / μ_star[i] > 0.5
            interp = "Importante + não-linear"
        else
            interp = "Importante + linear"
        end
    elseif μ_star[i] > 0.05
        interp = "Moderadamente importante"
    else
        interp = "Pouco importante"
    end

    @printf("│ %-16s │  %6.3f  │  %6.3f  │  %6.2f  │ %-27s │\n",
            "$(p.name) ($(p.symbol))", μ_star[i], σ[i], ratio, interp)
end
println("└──────────────────┴──────────┴──────────┴──────────┴─────────────────────────────┘")

# Análise visual (texto)
println("\n📊 RANKING DE IMPORTÂNCIA DOS PARÂMETROS:")
println("-"^70)

max_bar = 50
max_μ = maximum(μ_star)

for (rank, i) in enumerate(sorted_idx)
    p = PARAMETERS[i]
    bar_len = round(Int, μ_star[i] / max_μ * max_bar)
    bar = "█" ^ bar_len
    @printf("  %d. %-12s │%s│ μ*=%.3f\n", rank, p.symbol, bar, μ_star[i])
end

# Interpretação física
println("\n\n" * "="^90)
println("  INTERPRETAÇÃO FÍSICA")
println("="^90)

println("\n🔬 PARÂMETROS MAIS INFLUENTES:")
println("-"^70)

top3 = sorted_idx[1:min(3, length(sorted_idx))]
for (rank, i) in enumerate(top3)
    p = PARAMETERS[i]
    println("\n  $rank. $(p.name) ($(p.symbol)):")
    println("     - Importância (μ*): $(round(μ_star[i], digits=3))")
    println("     - Não-linearidade (σ): $(round(σ[i], digits=3))")

    # Interpretação específica
    if p.name == "k0"
        println("     - Taxa base de hidrólise: controle direto da velocidade de degradação")
        println("     - Altamente sensível: pequenas variações causam grandes mudanças")
    elseif p.name == "crystallinity"
        println("     - Cristalinidade: barreira física à penetração de água")
        println("     - Não-linear: efeito mais forte em altas cristalinidades")
    elseif p.name == "Mn_initial"
        println("     - Massa molar inicial: define escala de degradação")
        println("     - Linear: relação proporcional com Mn final")
    elseif p.name == "autocatalysis"
        println("     - Autocatálise: feedback positivo de produtos ácidos")
        println("     - Não-linear: efeito acelera com degradação avançada")
    elseif p.name == "Ea"
        println("     - Energia de ativação: sensibilidade à temperatura")
        println("     - Efeito Arrhenius exponencial")
    end
end

println("\n\n⚠️ PARÂMETROS MENOS INFLUENTES:")
println("-"^70)
bottom = sorted_idx[end-1:end]
for i in bottom
    p = PARAMETERS[i]
    println("  - $(p.name) ($(p.symbol)): μ*=$(round(μ_star[i], digits=3))")
end
println("  → Estes parâmetros podem ser fixados em valores típicos sem grande perda de precisão")

# Recomendações para calibração
println("\n\n" * "="^90)
println("  RECOMENDAÇÕES PARA CALIBRAÇÃO")
println("="^90)

println("\n📋 PRIORIDADE DE CALIBRAÇÃO:")
println("-"^70)
println("  1. ALTA: k0, cristalinidade - calibrar com dados experimentais GPC + DSC")
println("  2. MÉDIA: autocatálise - ajustar com dados de pH do meio")
println("  3. BAIXA: Ea, porosidade - usar valores da literatura")

println("\n📋 DADOS EXPERIMENTAIS NECESSÁRIOS:")
println("-"^70)
println("  - GPC: Mn vs tempo (para k0, autocatálise)")
println("  - DSC: cristalinidade inicial e durante degradação")
println("  - pH do meio: correlacionar com autocatálise")
println("  - Micro-CT: porosidade real (se diferente do design)")

println("\n" * "="^90)
println("  Análise Morris concluída - $(Dates.now())")
println("="^90)

# Salvar resultados
results_file = joinpath(@__DIR__, "..", "docs", "MORRIS_SENSITIVITY_ANALYSIS.md")
open(results_file, "w") do f
    println(f, "# Análise de Sensibilidade Morris")
    println(f, "")
    println(f, "Data: $(Dates.today())")
    println(f, "")
    println(f, "## Método")
    println(f, "- Morris Elementary Effects (1991)")
    println(f, "- 30 trajetórias")
    println(f, "- 4 níveis por parâmetro")
    println(f, "")
    println(f, "## Resultados")
    println(f, "")
    println(f, "| Parâmetro | μ* | σ | Interpretação |")
    println(f, "|-----------|-----|---|---------------|")
    for i in sorted_idx
        p = PARAMETERS[i]
        interp = μ_star[i] > 0.1 ? "Importante" : (μ_star[i] > 0.05 ? "Moderado" : "Baixo")
        println(f, "| $(p.name) | $(round(μ_star[i], digits=3)) | $(round(σ[i], digits=3)) | $interp |")
    end
    println(f, "")
    println(f, "## Conclusões")
    println(f, "")
    println(f, "Os parâmetros mais importantes para a degradação são:")
    for i in top3
        println(f, "1. **$(PARAMETERS[i].name)**: μ* = $(round(μ_star[i], digits=3))")
    end
end

println("\nResultados salvos em: $results_file")
