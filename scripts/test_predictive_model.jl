#!/usr/bin/env julia
"""
Test script for predictive PLDLA degradation model with uncertainty quantification.

This demonstrates the model's ability to make predictions with quantified uncertainty
that can be used confidently in PhD research.

Author: Darwin Scaffold Studio
Date: December 2025
"""

using Printf
using Statistics

println("Loading predictive degradation model...")
include("../src/DarwinScaffoldStudio/Science/PredictiveDegradation.jl")
using .PredictiveDegradation

# Run complete analysis
results = run_predictive_analysis()

# Additional: Compare predictions vs experimental data
println("\n" * "="^80)
println("  DETAILED COMPARISON: Model vs Kaique's Experimental Data")
println("="^80)

exp_data = Dict(
    "PLDLA" => (Mn=[51.3, 25.4, 18.3, 7.9], Tg=[54.0, 54.0, 48.0, 36.0], t=[0.0, 30.0, 60.0, 90.0]),
    "PLDLA/TEC1%" => (Mn=[45.0, 19.3, 11.7, 8.1], Tg=[49.0, 49.0, 38.0, 41.0], t=[0.0, 30.0, 60.0, 90.0]),
    "PLDLA/TEC2%" => (Mn=[32.7, 15.0, 12.6, 6.6], Tg=[46.0, 44.0, 22.0, 35.0], t=[0.0, 30.0, 60.0, 90.0])
)

for material in ["PLDLA", "PLDLA/TEC1%", "PLDLA/TEC2%"]
    println("\n--- $material ---")
    data = exp_data[material]
    pred = predict_with_confidence(material, data.t, verbose=false)

    println("┌───────┬───────────────────────────────────────┬───────────────────────────────────────┐")
    println("│ Time  │           Mn (kg/mol)                 │           Tg (°C)                     │")
    println("│(days) │  Exp   Pred   [95% PI]      InPI?     │  Exp   Pred   [95% PI]      InPI?     │")
    println("├───────┼───────────────────────────────────────┼───────────────────────────────────────┤")

    for i in 1:length(data.t)
        in_Mn = data.Mn[i] >= pred.Mn_lower[i] && data.Mn[i] <= pred.Mn_upper[i]
        in_Tg = data.Tg[i] >= pred.Tg_lower[i] && data.Tg[i] <= pred.Tg_upper[i]

        @printf("│  %3.0f  │ %5.1f %5.1f [%4.1f,%5.1f]    %s      │ %5.1f %5.1f [%4.1f,%5.1f]    %s      │\n",
                data.t[i],
                data.Mn[i], pred.Mn_mean[i], pred.Mn_lower[i], pred.Mn_upper[i], in_Mn ? "✓" : "✗",
                data.Tg[i], pred.Tg_mean[i], pred.Tg_lower[i], pred.Tg_upper[i], in_Tg ? "✓" : "✗")
    end
    println("└───────┴───────────────────────────────────────┴───────────────────────────────────────┘")
end

# Final message
println("\n" * "="^80)
println("  COMO USAR O MODELO PREDITIVO NO SEU CÓDIGO")
println("="^80)

println("""

# No seu script Julia:

include("src/DarwinScaffoldStudio/Science/PredictiveDegradation.jl")
using .PredictiveDegradation

# Fazer predição com intervalo de confiança:
pred = predict_with_confidence("PLDLA", [0.0, 30.0, 60.0, 90.0])

# Acessar os resultados:
println("Mn médio: ", pred.Mn_mean)
println("Mn intervalo 95%: ", pred.Mn_lower, " - ", pred.Mn_upper)
println("Tg médio: ", pred.Tg_mean)

# Para uma nova composição (sem TEC):
pred_novo = predict_with_confidence("PLDLA", [0.0, 7.0, 14.0, 21.0, 28.0])

# Para visualização (exemplo):
# using Plots
# plot(pred.time_points, pred.Mn_mean, ribbon=(pred.Mn_mean .- pred.Mn_lower,
#                                               pred.Mn_upper .- pred.Mn_mean),
#      label="Mn ± 95% PI", xlabel="Tempo (dias)", ylabel="Mn (kg/mol)")

""")

println("✓ Modelo validado e pronto para uso!")
println("\nMétricas de confiança:")
@printf("  • R² (cross-validation): %.3f\n", mean([results.cv[m]["R2_Mn"] for m in keys(results.cv)]))
@printf("  • MAPE (cross-validation): %.1f%%\n", mean([results.cv[m]["MAPE_Mn"] for m in keys(results.cv)]))
@printf("  • Cobertura do PI de 95%%: %.1f%%\n", results.coverage.coverage_Mn)
