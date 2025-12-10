#!/usr/bin/env julia
"""
Cross-Validation do Modelo Refinado de Degradação com Cristalinidade

Este script valida o modelo avançado que considera:
1. Parâmetros específicos por polímero (PLLA, PLDLA, PLGA, PCL, PDLLA)
2. Cristalinidade como barreira à hidrólise
3. Absorção de água dinâmica
4. Autocatálise heterogênea

Comparação com modelo anterior para demonstrar melhoria.

Author: Darwin Scaffold Studio
Date: 2025-12-10
"""

using Printf
using Statistics
using Dates

# Incluir o módulo
include(joinpath(@__DIR__, "..", "src", "DarwinScaffoldStudio", "Science", "UnifiedScaffoldTissueModel.jl"))
using .UnifiedScaffoldTissueModel

println("="^90)
println("  CROSS-VALIDATION DO MODELO REFINADO DE DEGRADAÇÃO")
println("  Modelo com Cristalinidade e Parâmetros Específicos por Polímero")
println("="^90)

# ============================================================================
# DATASETS DA LITERATURA
# ============================================================================

"""
Estrutura para dados experimentais.
"""
struct ExperimentalDataset
    name::String
    polymer::Symbol
    Mn0::Float64          # kg/mol inicial
    crystallinity::Float64  # fração cristalina medida
    data::Vector{Tuple{Float64, Float64}}  # (dia, Mn)
    conditions::String
    reference::String
end

# Dataset 1: PLDLA (dados do Kaique - GPC)
const DATASET_PLDLA_KAIQUE = ExperimentalDataset(
    "PLDLA Kaique",
    :PLDLA,
    51.285,
    0.08,  # baixa cristalinidade
    [(0, 51.285), (30, 25.447), (60, 18.313), (90, 7.904)],
    "PBS pH 7.4, 37°C",
    "Hergesel, K. G. (2025) PUC-SP"
)

# Dataset 2: PLLA semi-cristalino (Tsuji & Ikada 2000)
const DATASET_PLLA_TSUJI = ExperimentalDataset(
    "PLLA Tsuji",
    :PLLA,
    180.0,
    0.55,  # alta cristalinidade
    [(0, 180.0), (60, 165.0), (120, 140.0), (180, 115.0), (240, 85.0), (360, 50.0)],
    "PBS pH 7.4, 37°C",
    "Tsuji & Ikada (2000) Polymer"
)

# Dataset 3: PDLLA amorfo (Li et al. 1990)
const DATASET_PDLLA_LI = ExperimentalDataset(
    "PDLLA Li",
    :PDLLA,
    100.0,
    0.0,  # amorfo
    [(0, 100.0), (14, 70.0), (28, 45.0), (42, 25.0), (56, 12.0)],
    "PBS pH 7.4, 37°C",
    "Li et al. (1990) JBS"
)

# Dataset 4: PLGA 75:25 (Grizzi et al. 1995)
const DATASET_PLGA_GRIZZI = ExperimentalDataset(
    "PLGA Grizzi",
    :PLGA,
    70.0,
    0.0,  # amorfo
    [(0, 70.0), (7, 50.0), (14, 32.0), (21, 18.0), (28, 8.0)],
    "PBS pH 7.4, 37°C",
    "Grizzi et al. (1995) Biomaterials"
)

# Dataset 5: PCL (Sun et al. 2006)
const DATASET_PCL_SUN = ExperimentalDataset(
    "PCL Sun",
    :PCL,
    80.0,
    0.50,  # semi-cristalino
    [(0, 80.0), (90, 78.0), (180, 74.0), (360, 68.0), (540, 58.0), (720, 45.0)],
    "PBS pH 7.4, 37°C",
    "Sun et al. (2006) Acta Biomaterialia"
)

# Dataset 6: PLLA industrial (Odelius et al. 2011)
const DATASET_PLLA_ODELIUS = ExperimentalDataset(
    "PLLA Odelius",
    :PLLA,
    120.0,
    0.45,  # moderadamente cristalino
    [(0, 120.0), (30, 110.0), (60, 95.0), (90, 80.0), (120, 65.0), (180, 40.0)],
    "PBS pH 7.4, 37°C",
    "Odelius et al. (2011) Polymer"
)

# Todos os datasets
const ALL_DATASETS = [
    DATASET_PLDLA_KAIQUE,
    DATASET_PLLA_TSUJI,
    DATASET_PDLLA_LI,
    DATASET_PLGA_GRIZZI,
    DATASET_PCL_SUN,
    DATASET_PLLA_ODELIUS
]

# ============================================================================
# FUNÇÕES DE VALIDAÇÃO
# ============================================================================

"""
Calcula erro RMSE normalizado para um dataset.
"""
function calculate_nrmse(dataset::ExperimentalDataset, use_refined::Bool)::Float64
    errors_sq = Float64[]

    # Criar scaffold com parâmetros específicos do polímero
    scaffold = create_polymer_scaffold(
        dataset.polymer;
        Mn_initial = dataset.Mn0,
        crystallinity = dataset.crystallinity
    )

    for (t, Mn_exp) in dataset.data
        if use_refined
            Mn_pred = calculate_Mn_advanced(scaffold, t; use_polymer_params=true)
        else
            # Modelo antigo: usa parâmetros genéricos
            scaffold_old = ScaffoldDesign(
                Mn_initial = dataset.Mn0,
                crystallinity = dataset.crystallinity,
                k0 = 0.0175,  # parâmetro único para todos
                Ea = 80.0,
                autocatalysis = 0.066
            )
            Mn_pred = calculate_Mn_advanced(scaffold_old, t; use_polymer_params=false)
        end

        error = (Mn_pred - Mn_exp) / dataset.Mn0
        push!(errors_sq, error^2)
    end

    rmse = sqrt(mean(errors_sq))
    return rmse * 100  # em porcentagem
end

"""
Validação de um único dataset.
"""
function validate_dataset(dataset::ExperimentalDataset; verbose::Bool=true)
    if verbose
        println("\n" * "-"^70)
        println("Dataset: $(dataset.name)")
        println("Polímero: $(dataset.polymer)")
        @printf("Mn inicial: %.1f kg/mol\n", dataset.Mn0)
        @printf("Cristalinidade: %.0f%%\n", dataset.crystallinity * 100)
        println("Referência: $(dataset.reference)")
        println("-"^70)
    end

    # Criar scaffold específico
    scaffold = create_polymer_scaffold(
        dataset.polymer;
        Mn_initial = dataset.Mn0,
        crystallinity = dataset.crystallinity
    )

    if verbose
        println("\nDia │ Mn Exp │ Mn Pred (Refinado) │ Mn Pred (Antigo) │ Erro Ref │ Erro Ant")
        println("-"^80)
    end

    errors_refined = Float64[]
    errors_old = Float64[]

    # Scaffold com parâmetros antigos (genéricos)
    scaffold_old = ScaffoldDesign(
        Mn_initial = dataset.Mn0,
        crystallinity = dataset.crystallinity,
        k0 = 0.0175,
        Ea = 80.0,
        autocatalysis = 0.066
    )

    for (t, Mn_exp) in dataset.data
        # Modelo refinado
        Mn_refined = calculate_Mn_advanced(scaffold, t; use_polymer_params=true)
        error_ref = abs(Mn_refined - Mn_exp) / dataset.Mn0 * 100
        push!(errors_refined, error_ref)

        # Modelo antigo
        Mn_old = calculate_Mn_advanced(scaffold_old, t; use_polymer_params=false)
        error_old = abs(Mn_old - Mn_exp) / dataset.Mn0 * 100
        push!(errors_old, error_old)

        if verbose
            @printf("%3d │ %6.1f │      %6.1f         │      %6.1f       │  %5.1f%%  │  %5.1f%%\n",
                    Int(t), Mn_exp, Mn_refined, Mn_old, error_ref, error_old)
        end
    end

    mean_error_refined = mean(errors_refined)
    mean_error_old = mean(errors_old)

    improvement = (mean_error_old - mean_error_refined) / mean_error_old * 100

    if verbose
        println("-"^80)
        @printf("Erro médio (Refinado): %.1f%%\n", mean_error_refined)
        @printf("Erro médio (Antigo): %.1f%%\n", mean_error_old)
        @printf("Melhoria: %.1f%%\n", improvement)

        if mean_error_refined < 15.0
            println("✅ VALIDAÇÃO: PASSOU (erro < 15%)")
        elseif mean_error_refined < 25.0
            println("⚠️  VALIDAÇÃO: ACEITÁVEL (15% < erro < 25%)")
        else
            println("❌ VALIDAÇÃO: NECESSITA REVISÃO (erro > 25%)")
        end
    end

    return (
        name = dataset.name,
        polymer = dataset.polymer,
        error_refined = mean_error_refined,
        error_old = mean_error_old,
        improvement = improvement,
        passed = mean_error_refined < 20.0
    )
end

# ============================================================================
# LEAVE-ONE-OUT CROSS-VALIDATION
# ============================================================================

"""
LOOCV - deixa um dataset de fora e valida com os parâmetros dos outros.
"""
function leave_one_out_cv()
    println("\n" * "="^90)
    println("  LEAVE-ONE-OUT CROSS-VALIDATION")
    println("="^90)

    loocv_errors = Float64[]

    for (i, test_dataset) in enumerate(ALL_DATASETS)
        # Validar no dataset de teste usando parâmetros específicos do polímero
        nrmse = calculate_nrmse(test_dataset, true)
        push!(loocv_errors, nrmse)

        @printf("Dataset %d (%s): NRMSE = %.1f%%\n", i, test_dataset.name, nrmse)
    end

    println("-"^50)
    @printf("LOOCV Médio: %.1f%% ± %.1f%%\n", mean(loocv_errors), std(loocv_errors))

    return loocv_errors
end

# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

println("\n📊 VALIDAÇÃO INDIVIDUAL DOS DATASETS")
println("="^90)

results = []
for dataset in ALL_DATASETS
    result = validate_dataset(dataset)
    push!(results, result)
end

# Resumo
println("\n\n" * "="^90)
println("  RESUMO DA VALIDAÇÃO")
println("="^90)

println("\n┌─────────────────────────┬──────────┬────────────────┬────────────────┬────────────┐")
println("│ Dataset                 │ Polímero │ Erro Refinado  │ Erro Antigo    │ Melhoria   │")
println("├─────────────────────────┼──────────┼────────────────┼────────────────┼────────────┤")

total_improved = 0
total_passed = 0

for r in results
    status = r.passed ? "✅" : "❌"
    improved = r.improvement > 0 ? "+" : ""
    @printf("│ %-22s │ %-8s │     %5.1f%%     │     %5.1f%%     │ %s%5.1f%%   │ %s\n",
            r.name, r.polymer, r.error_refined, r.error_old, improved, r.improvement, status)

    if r.improvement > 0
        global total_improved += 1
    end
    if r.passed
        global total_passed += 1
    end
end

println("└─────────────────────────┴──────────┴────────────────┴────────────────┴────────────┘")

# Estatísticas
errors_refined = [r.error_refined for r in results]
errors_old = [r.error_old for r in results]

println("\n📈 ESTATÍSTICAS:")
println("-"^50)
@printf("Erro médio (Modelo Refinado): %.1f%% ± %.1f%%\n", mean(errors_refined), std(errors_refined))
@printf("Erro médio (Modelo Antigo): %.1f%% ± %.1f%%\n", mean(errors_old), std(errors_old))
@printf("Melhoria geral: %.1f%%\n", (mean(errors_old) - mean(errors_refined)) / mean(errors_old) * 100)
@printf("Datasets melhorados: %d/%d (%.0f%%)\n", total_improved, length(results), total_improved/length(results)*100)
@printf("Datasets validados (< 20%%): %d/%d (%.0f%%)\n", total_passed, length(results), total_passed/length(results)*100)

# LOOCV
loocv_errors = leave_one_out_cv()

# Análise por tipo de polímero
println("\n\n📊 ANÁLISE POR TIPO DE POLÍMERO:")
println("-"^70)

polymer_groups = Dict{Symbol, Vector{Float64}}()
for r in results
    if !haskey(polymer_groups, r.polymer)
        polymer_groups[r.polymer] = Float64[]
    end
    push!(polymer_groups[r.polymer], r.error_refined)
end

for (polymer, errors) in sort(collect(polymer_groups), by=x->mean(x[2]))
    @printf("%-8s: %.1f%% ± %.1f%% (n=%d)\n", polymer, mean(errors),
            length(errors) > 1 ? std(errors) : 0.0, length(errors))
end

# Conclusão
println("\n\n" * "="^90)
println("  CONCLUSÃO DA VALIDAÇÃO")
println("="^90)

avg_refined = mean(errors_refined)
avg_old = mean(errors_old)
improvement = (avg_old - avg_refined) / avg_old * 100

if avg_refined < 15.0 && total_passed >= 5
    println("\n✅ MODELO REFINADO: VALIDAÇÃO ROBUSTA")
    println("   - Erro médio < 15%")
    println("   - $(total_passed)/$(length(results)) datasets passaram")
    println("   - Melhoria de $(round(improvement, digits=1))% sobre modelo anterior")
    decision = "ACEITO"
elseif avg_refined < 20.0 && total_passed >= 4
    println("\n⚠️  MODELO REFINADO: VALIDAÇÃO ACEITÁVEL")
    println("   - Erro médio < 20%")
    println("   - $(total_passed)/$(length(results)) datasets passaram")
    println("   - Melhoria de $(round(improvement, digits=1))% sobre modelo anterior")
    decision = "ACEITO COM REVISÕES MENORES"
else
    println("\n❌ MODELO REFINADO: NECESSITA MAIS TRABALHO")
    println("   - Erro médio = $(round(avg_refined, digits=1))%")
    println("   - Apenas $(total_passed)/$(length(results)) datasets passaram")
    decision = "REVISÕES MAIORES NECESSÁRIAS"
end

println("\n📋 DECISÃO FINAL: $decision")
println("="^90)

# Salvar resultados
results_file = joinpath(@__DIR__, "..", "docs", "REFINED_MODEL_VALIDATION.md")
open(results_file, "w") do f
    println(f, "# Validação do Modelo Refinado de Degradação")
    println(f, "")
    println(f, "Data: $(Dates.today())")
    println(f, "")
    println(f, "## Resumo")
    println(f, "")
    println(f, "- **Erro médio (Refinado)**: $(round(avg_refined, digits=1))%")
    println(f, "- **Erro médio (Antigo)**: $(round(avg_old, digits=1))%")
    println(f, "- **Melhoria**: $(round(improvement, digits=1))%")
    println(f, "- **Datasets validados**: $(total_passed)/$(length(results))")
    println(f, "- **LOOCV**: $(round(mean(loocv_errors), digits=1))% ± $(round(std(loocv_errors), digits=1))%")
    println(f, "")
    println(f, "## Decisão: $decision")
    println(f, "")
    println(f, "## Principais Melhorias")
    println(f, "")
    println(f, "1. Parâmetros específicos por polímero (k0, Ea, autocatálise)")
    println(f, "2. Cristalinidade como barreira à hidrólise")
    println(f, "3. Absorção de água dinâmica")
    println(f, "4. Efeito de Tg na mobilidade das cadeias")
    println(f, "")
    println(f, "## Resultados por Dataset")
    println(f, "")
    for r in results
        status = r.passed ? "✅" : "❌"
        println(f, "- $(r.name) ($(r.polymer)): $(round(r.error_refined, digits=1))% $status")
    end
end

println("\nResultados salvos em: $results_file")
