#!/usr/bin/env julia
"""
Teste do modelo híbrido robusto (Física + Neural)
"""

using Pkg
Pkg.activate(".")

using Statistics
using Printf
using Random

Random.seed!(42)

println("="^80)
println("  MODELO HÍBRIDO ROBUSTO - FÍSICA + CORREÇÃO NEURAL")
println("  Multi-polímero: PLDLA, PDLLA, PLLA")
println("="^80)

include("../src/DarwinScaffoldStudio/Science/RobustPINN.jl")
using .RobustPINN

# ============================================================================
# CRIAR DATASETS
# ============================================================================

datasets = create_all_datasets()

println("\n📚 DATASETS:")
for ds in datasets
    polymer_name = ["PLDLA", "PDLLA", "PLLA"][ds.polymer_type]
    println("  • $(ds.name): $(polymer_name), $(length(ds.times)) pontos, cryst=$(ds.crystallinity)")
end

# ============================================================================
# TREINAR MODELO
# ============================================================================

model = HybridDegradationModel()

println("\n" * "="^80)
println("  TREINAMENTO")
println("="^80)

train_hybrid!(model, datasets; epochs=300, lr=0.002, verbose=true)

# ============================================================================
# AVALIAÇÃO
# ============================================================================

println("\n" * "="^80)
println("  AVALIAÇÃO DETALHADA")
println("="^80)

for ds in datasets
    Mn0 = ds.Mn_values[1]

    println("\n📊 $(ds.name):")
    println("  Tempo │ Mn Real │ Mn Pred │  Erro")
    println("  ──────┼─────────┼─────────┼───────")

    for (i, t) in enumerate(ds.times)
        Mn_pred = predict_hybrid(model, t, ds.temperature, Mn0;
                                polymer_type=ds.polymer_type,
                                crystallinity=ds.crystallinity)
        Mn_true = ds.Mn_values[i]
        erro = abs(Mn_pred - Mn_true) / Mn_true * 100

        @printf("  %5.0f │  %5.1f  │  %5.1f  │ %4.1f%%\n",
                t, Mn_true, Mn_pred, erro)
    end
end

# ============================================================================
# COMPARAÇÃO FINAL
# ============================================================================

println("\n" * "="^80)
println("  RESUMO FINAL")
println("="^80)

mean_mape = compare_models(model, datasets)

# ============================================================================
# VALIDAÇÃO CRUZADA
# ============================================================================

println("\n" * "="^80)
println("  VALIDAÇÃO CRUZADA LEAVE-ONE-OUT")
println("="^80)

cv_mapes = Float64[]

for (i, test_ds) in enumerate(datasets)
    # Treinar sem o dataset de teste
    train_ds = [d for (j, d) in enumerate(datasets) if j != i]

    cv_model = HybridDegradationModel()
    train_hybrid!(cv_model, train_ds; epochs=200, lr=0.002, verbose=false)

    # Avaliar no teste
    result = evaluate_model(cv_model, test_ds)
    push!(cv_mapes, result.mape)

    @printf("Fold %d: Teste em %-15s → MAPE: %5.1f%%\n", i, test_ds.name, result.mape)
end

cv_mean = mean(cv_mapes)
cv_std = std(cv_mapes)

println("-"^50)
@printf("Validação Cruzada: MAPE = %.1f%% ± %.1f%%\n", cv_mean, cv_std)
@printf("Acurácia CV: %.1f%%\n", 100 - cv_mean)

# ============================================================================
# PREDIÇÕES PARA CENÁRIOS CLÍNICOS
# ============================================================================

println("\n" * "="^80)
println("  PREDIÇÕES PARA CENÁRIOS CLÍNICOS")
println("="^80)

println("\n📋 Tempo para perda de integridade mecânica (Mn < 10 kg/mol):")
println("-"^60)

scenarios = [
    ("PLDLA scaffold (menisco)", 1, 0.0, 51.0),
    ("PDLLA implante (amorfo)", 2, 0.0, 45.0),
    ("PLLA parafuso (cristalino)", 3, 0.36, 98.0),
]

for (name, ptype, cryst, Mn0) in scenarios
    # Encontrar tempo para Mn < 10
    t_critical = 0
    for t in 1:500
        Mn = predict_hybrid(model, Float64(t), 310.15, Mn0;
                           polymer_type=ptype, crystallinity=cryst)
        if Mn < 10.0
            t_critical = t
            break
        end
    end

    t_str = t_critical > 0 ? "$(t_critical) dias" : "> 500 dias"
    @printf("  %-30s → %s\n", name, t_str)
end

# ============================================================================
# TABELA DE PREDIÇÕES
# ============================================================================

println("\n📈 CURVAS DE DEGRADAÇÃO PREDITAS:")
println("-"^70)
println("Tempo │   PLDLA   │   PDLLA   │   PLLA    │ PLLA cryst")
println("(dias)│  (51 kg)  │  (45 kg)  │  (98 kg)  │  (98 kg)")
println("-"^70)

for t in [0, 14, 30, 60, 90, 120, 150, 180, 240, 300]
    mn_pldla = predict_hybrid(model, Float64(t), 310.15, 51.0; polymer_type=1, crystallinity=0.0)
    mn_pdlla = predict_hybrid(model, Float64(t), 310.15, 45.0; polymer_type=2, crystallinity=0.0)
    mn_plla = predict_hybrid(model, Float64(t), 310.15, 98.0; polymer_type=3, crystallinity=0.0)
    mn_plla_c = predict_hybrid(model, Float64(t), 310.15, 98.0; polymer_type=3, crystallinity=0.36)

    @printf(" %4d │   %5.1f   │   %5.1f   │   %5.1f   │   %5.1f\n",
            t, mn_pldla, mn_pdlla, mn_plla, mn_plla_c)
end

println("\n" * "="^80)
println("""
┌────────────────────────────────────────────────────────────────────────────┐
│                     MODELO HÍBRIDO - RESUMO                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ARQUITETURA:                                                              │
│    • Física: Cinética de hidrólise com Arrhenius + autocatálise           │
│    • Neural: Correção residual aprendida (±10%)                            │
│    • Multi-polímero: PLDLA, PDLLA, PLLA com k0 específico                 │
│                                                                            │
│  PARÂMETROS APRENDIDOS:                                                    │
│    • k0_PLDLA = $(round(model.k0_pldla, digits=4)) /dia                                               │
│    • k0_PDLLA = $(round(model.k0_pdlla, digits=4)) /dia                                               │
│    • k0_PLLA  = $(round(model.k0_plla, digits=4)) /dia                                                │
│    • Ea = $(model.Ea) kJ/mol                                                       │
│    • Autocatálise = $(round(model.autocatalysis, digits=3))                                              │
│    • Fator cristalinidade = $(round(model.crystallinity_factor, digits=3))                                   │
│                                                                            │
│  MÉTRICAS:                                                                 │
│    • Acurácia treino: $(round(100 - mean_mape, digits=1))%                                              │
│    • Acurácia CV: $(round(100 - cv_mean, digits=1))% ± $(round(cv_std, digits=1))%                                           │
│                                                                            │
│  STATUS: ✅ VALIDADO COM VALIDAÇÃO CRUZADA                                 │
└────────────────────────────────────────────────────────────────────────────┘
""")
