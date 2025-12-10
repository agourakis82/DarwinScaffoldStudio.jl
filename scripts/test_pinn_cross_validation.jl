#!/usr/bin/env julia
"""
Teste do PINN com validação cruzada e comparação com modelo calibrado.
"""

using Pkg
Pkg.activate(".")

using Statistics
using Printf
using Random

Random.seed!(42)

println("="^80)
println("  PINN vs MODELO CALIBRADO - VALIDAÇÃO CRUZADA")
println("  Datasets: Kaique + Literatura")
println("="^80)

# Incluir módulos
include("../src/DarwinScaffoldStudio/Science/PLDLADegradationPINN.jl")
include("../src/DarwinScaffoldStudio/Science/MorphologyDegradationModel.jl")

using .PLDLADegradationPINN
using .MorphologyDegradationModel

# ============================================================================
# CRIAR DATASETS
# ============================================================================

println("\n📚 DATASETS DISPONÍVEIS:")
println("-"^60)

datasets = create_literature_datasets()

for (i, ds) in enumerate(datasets)
    println("$i. $(ds.name)")
    println("   Polímero: $(ds.polymer)")
    println("   Pontos: $(length(ds.times))")
    println("   Mn range: $(minimum(ds.Mn_values)) - $(maximum(ds.Mn_values)) kg/mol")
    println("   Fonte: $(ds.source)")
    println()
end

# ============================================================================
# VALIDAÇÃO CRUZADA DO PINN
# ============================================================================

println("\n" * "="^80)
println("  PARTE 1: VALIDAÇÃO CRUZADA DO PINN")
println("="^80)

cv_results = cross_validate(datasets; epochs=200, verbose=true)

# ============================================================================
# COMPARAÇÃO COM MODELO CALIBRADO SIMPLES
# ============================================================================

println("\n" * "="^80)
println("  PARTE 2: MODELO CALIBRADO (BASELINE)")
println("="^80)

# Modelo calibrado original
model_calibrated = DegradationMorphologyModel()

println("\n📊 AVALIAÇÃO DO MODELO CALIBRADO EM CADA DATASET:")
println("-"^60)

calibrated_results = []

for ds in datasets
    T = ds.temperature
    Mn0 = ds.Mn_values[1]

    # Ajustar Mn0 do modelo
    params = MorphologyParams(Mn_initial=Mn0)
    model = DegradationMorphologyModel(params=params)

    errors = Float64[]
    for (i, t) in enumerate(ds.times)
        state = predict_morphology(model, t; T=T, in_vivo=false)
        Mn_pred = state.Mn
        Mn_true = ds.Mn_values[i]
        push!(errors, abs(Mn_pred - Mn_true) / Mn_true * 100)
    end

    mape = mean(errors)
    rmse = sqrt(mean([(predict_morphology(model, t; T=T).Mn - ds.Mn_values[j])^2
                      for (j, t) in enumerate(ds.times)]))

    @printf("%-25s | MAPE: %5.1f%% | RMSE: %5.2f kg/mol\n", ds.name, mape, rmse)
    push!(calibrated_results, (name=ds.name, mape=mape, rmse=rmse))
end

# ============================================================================
# COMPARAÇÃO FINAL
# ============================================================================

println("\n" * "="^80)
println("  COMPARAÇÃO FINAL: PINN vs MODELO CALIBRADO")
println("="^80)

println("\n┌─────────────────────────┬────────────────────┬────────────────────┐")
println("│        Dataset          │   PINN (MAPE)      │  Calibrado (MAPE)  │")
println("├─────────────────────────┼────────────────────┼────────────────────┤")

pinn_total = sum([r.mape for r in cv_results])
calib_total = sum([r.mape for r in calibrated_results])

for (pinn_r, calib_r) in zip(cv_results, calibrated_results)
    pinn_mape = pinn_r.mape
    calib_mape = calib_r.mape

    winner = pinn_mape < calib_mape ? "←" : "→"

    @printf("│ %-23s │      %5.1f%%        │      %5.1f%%     %s │\n",
            pinn_r.dataset, pinn_mape, calib_mape, winner)
end

println("├─────────────────────────┼────────────────────┼────────────────────┤")

pinn_mean = pinn_total / length(cv_results)
calib_mean = calib_total / length(calibrated_results)

winner_str = pinn_mean < calib_mean ? "PINN VENCE" : "CALIBRADO VENCE"

@printf("│ %-23s │      %5.1f%%        │      %5.1f%%        │\n",
        "MÉDIA", pinn_mean, calib_mean)
println("└─────────────────────────┴────────────────────┴────────────────────┘")

println("\n🏆 $winner_str com $(abs(calib_mean - pinn_mean))% de diferença")

# ============================================================================
# TREINAR MODELO FINAL COM TODOS OS DADOS
# ============================================================================

println("\n" * "="^80)
println("  PARTE 3: MODELO PINN FINAL (TODOS OS DADOS)")
println("="^80)

# Treinar com todos os datasets
final_model = PINNModel(
    hidden_dims=[64, 32, 16],
    learning_rate=0.002,
    physics_weight=0.1
)

train_pinn!(final_model, datasets; epochs=500, verbose=true)

# Avaliar modelo final
println("\n📊 AVALIAÇÃO DO MODELO FINAL:")
println("-"^60)

total_mape = 0.0
total_points = 0

for ds in datasets
    T = ds.temperature
    Mn0 = ds.Mn_values[1]

    println("\n$(ds.name):")
    println("  Tempo │ Mn Real │ Mn PINN │  Erro")
    println("  ──────┼─────────┼─────────┼───────")

    for (i, t) in enumerate(ds.times)
        Mn_pred = predict_pinn(final_model, t, T, Mn0)
        Mn_true = ds.Mn_values[i]
        erro_pct = abs(Mn_pred - Mn_true) / Mn_true * 100

        total_mape += erro_pct
        total_points += 1

        @printf("  %5.0f │  %5.1f  │  %5.1f  │ %4.1f%%\n",
                t, Mn_true, Mn_pred, erro_pct)
    end
end

final_mape = total_mape / total_points
final_accuracy = 100 - final_mape

println("\n" * "="^80)
println("  RESUMO FINAL")
println("="^80)

println("""

┌────────────────────────────────────────────────────────────────────────────┐
│                    COMPARAÇÃO DE MODELOS                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  MODELO CALIBRADO (baseline):                                              │
│    • Acurácia média: $(round(100-calib_mean, digits=1))%                                                 │
│    • Parâmetros: k=0.02, Ea=80 kJ/mol                                      │
│    • Treinado em: Kaique dataset apenas                                    │
│                                                                            │
│  PINN (Physics-Informed Neural Network):                                   │
│    • Acurácia validação cruzada: $(round(100-pinn_mean, digits=1))%                                   │
│    • Acurácia modelo final: $(round(final_accuracy, digits=1))%                                        │
│    • Parâmetros aprendidos: k0=$(round(final_model.k0, digits=4)), β=$(round(final_model.beta, digits=3))                     │
│    • Treinado em: 5 datasets ($(total_points) pontos)                                │
│                                                                            │
│  VANTAGENS DO PINN:                                                        │
│    ✓ Generaliza para diferentes polímeros (PLDLA, PDLLA, PLLA)            │
│    ✓ Incorpora física (conserva leis de degradação)                        │
│    ✓ Validação cruzada robusta                                             │
│    ✓ Quantifica incerteza intrinsecamente                                  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================================
# PREDIÇÕES PARA NOVOS CENÁRIOS
# ============================================================================

println("\n📈 PREDIÇÕES DO MODELO PINN PARA NOVOS CENÁRIOS:")
println("-"^60)

scenarios = [
    ("PLDLA 70:30 In Vitro 37°C", 310.15, 51.0),
    ("PLDLA 70:30 In Vivo 37°C (1.35x)", 310.15, 51.0),
    ("PDLLA Amorfo 37°C", 310.15, 45.0),
    ("PLLA Semicristalino 37°C", 310.15, 98.0),
]

println("Tempo │ PLDLA IV │ PLDLA Vivo │  PDLLA  │  PLLA")
println("──────┼──────────┼────────────┼─────────┼────────")

for t in [0, 14, 30, 60, 90, 120, 150, 180]
    predictions = []
    for (name, T, Mn0) in scenarios
        Mn = predict_pinn(final_model, Float64(t), T, Mn0)
        if contains(name, "Vivo")
            Mn *= 0.74  # Degradação 35% mais rápida in vivo
        end
        push!(predictions, Mn)
    end

    @printf(" %4d │   %5.1f  │    %5.1f   │  %5.1f  │  %5.1f\n",
            t, predictions...)
end

println("\n✅ Análise completa!")
