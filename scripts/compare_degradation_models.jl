"""
Comparação dos três modelos de degradação de PLDLA.

1. ConservativeDegradation - Empírico com autocatálise saturante
2. BronstedDegradation - Brønsted-Lowry + VFT
3. ThermodynamicDegradation - Primeiros princípios (Eyring + Fick)
"""

using Printf
using Statistics

# Carregar os três módulos
include("../src/DarwinScaffoldStudio/Science/ConservativeDegradation.jl")
include("../src/DarwinScaffoldStudio/Science/BronstedDegradation.jl")
include("../src/DarwinScaffoldStudio/Science/ThermodynamicDegradation.jl")

using .ConservativeDegradation
using .BronstedDegradation
using .ThermodynamicDegradation

println("\n" * "="^80)
println("       COMPARAÇÃO DOS MODELOS DE DEGRADAÇÃO")
println("       Empírico vs Brønsted-Lowry vs Termodinâmico")
println("="^80)

# Dados experimentais de referência (Kaique)
datasets = [
    ("Kaique_PLDLA", [51.3, 25.4, 18.3, 7.9], [0.0, 30.0, 60.0, 90.0]),
    ("Kaique_TEC1", [45.0, 19.3, 11.7, 8.1], [0.0, 30.0, 60.0, 90.0]),
    ("Kaique_TEC2", [32.7, 15.0, 12.6, 6.6], [0.0, 30.0, 60.0, 90.0]),
]

results = Dict{String, Dict{String, Float64}}()

for (name, Mn_exp, t_exp) in datasets
    println("\n" * "="^80)
    println("  Dataset: $name")
    println("="^80)

    # Prever com cada modelo
    pred_cons = ConservativeDegradation.predict_conservative(name)
    pred_bron = BronstedDegradation.predict_bronsted(name)
    pred_thermo = ThermodynamicDegradation.predict_thermodynamic(name)

    println("\n┌─────────┬──────────┬──────────┬──────────┬──────────┐")
    println("│ Time(d) │ Mn_exp   │ Conserv  │ Brønsted │ Thermo   │")
    println("├─────────┼──────────┼──────────┼──────────┼──────────┤")

    errors_cons = Float64[]
    errors_bron = Float64[]
    errors_thermo = Float64[]

    for i in 1:length(t_exp)
        err_c = abs(pred_cons.Mn[i] - Mn_exp[i]) / Mn_exp[i] * 100
        err_b = abs(pred_bron["Mn"][i] - Mn_exp[i]) / Mn_exp[i] * 100
        err_t = abs(pred_thermo["Mn"][i] - Mn_exp[i]) / Mn_exp[i] * 100

        push!(errors_cons, err_c)
        push!(errors_bron, err_b)
        push!(errors_thermo, err_t)

        @printf("│ %7.0f │ %8.1f │ %8.1f │ %8.1f │ %8.1f │\n",
                t_exp[i], Mn_exp[i], pred_cons.Mn[i], pred_bron["Mn"][i], pred_thermo["Mn"][i])
    end
    println("└─────────┴──────────┴──────────┴──────────┴──────────┘")

    mape_cons = mean(errors_cons[2:end])
    mape_bron = mean(errors_bron[2:end])
    mape_thermo = mean(errors_thermo[2:end])

    println("\n  MAPE:")
    @printf("    Conservative: %.1f%%\n", mape_cons)
    @printf("    Brønsted:     %.1f%%\n", mape_bron)
    @printf("    Thermo:       %.1f%%\n", mape_thermo)

    results[name] = Dict(
        "Conservative" => mape_cons,
        "Bronsted" => mape_bron,
        "Thermodynamic" => mape_thermo
    )
end

# Sumário final
println("\n" * "="^80)
println("       SUMÁRIO FINAL")
println("="^80)

println("\n┌────────────────────┬──────────────┬──────────────┬──────────────┐")
println("│ Dataset            │ Conservative │   Brønsted   │ Thermodynamic│")
println("├────────────────────┼──────────────┼──────────────┼──────────────┤")

for name in ["Kaique_PLDLA", "Kaique_TEC1", "Kaique_TEC2"]
    r = results[name]
    @printf("│ %-18s │ %10.1f%% │ %10.1f%% │ %10.1f%% │\n",
            name, r["Conservative"], r["Bronsted"], r["Thermodynamic"])
end

# Médias
mean_cons = mean([results[n]["Conservative"] for n in keys(results)])
mean_bron = mean([results[n]["Bronsted"] for n in keys(results)])
mean_thermo = mean([results[n]["Thermodynamic"] for n in keys(results)])

println("├────────────────────┼──────────────┼──────────────┼──────────────┤")
@printf("│ %-18s │ %10.1f%% │ %10.1f%% │ %10.1f%% │\n",
        "MÉDIA", mean_cons, mean_bron, mean_thermo)
println("└────────────────────┴──────────────┴──────────────┴──────────────┘")

# Análise
println("\n" * "="^80)
println("       ANÁLISE COMPARATIVA")
println("="^80)

best_model = argmin([mean_cons, mean_bron, mean_thermo])
models = ["Conservative", "Brønsted-Lowry", "Thermodynamic"]

println("\n┌─────────────────────────────────────────────────────────────────────────────────┐")
println("│  MELHOR MODELO: $(models[best_model])")
println("├─────────────────────────────────────────────────────────────────────────────────┤")
println("│                                                                                 │")
println("│  CONSERVATIVE (Empírico):                                                       │")
println("│    + Simples de implementar                                                     │")
println("│    + Bom ajuste com parâmetros calibrados                                       │")
println("│    - Sem fundamentação teórica                                                  │")
println("│    - Parâmetros não transferíveis                                               │")
println("│                                                                                 │")
println("│  BRØNSTED-LOWRY:                                                                │")
println("│    + Fundamentação química (catálise ácida)                                     │")
println("│    + pH local explica autocatálise                                              │")
println("│    + VFT melhor que Arrhenius perto de Tg                                       │")
println("│    - Ainda requer calibração de k₀                                              │")
println("│                                                                                 │")
println("│  THERMODYNAMIC (Primeiros Princípios):                                          │")
println("│    + Baseado em teoria (Eyring, Fick, termodinâmica)                            │")
println("│    + Parâmetros têm significado físico                                          │")
println("│    + Potencialmente mais transferível                                           │")
println("│    - Requer fator de acessibilidade (ajuste)                                    │")
println("│    - Mais complexo computacionalmente                                           │")
println("│                                                                                 │")
println("└─────────────────────────────────────────────────────────────────────────────────┘")

println("\n┌─────────────────────────────────────────────────────────────────────────────────┐")
println("│  RECOMENDAÇÃO PARA DISSERTAÇÃO                                                  │")
println("├─────────────────────────────────────────────────────────────────────────────────┤")
println("│                                                                                 │")
println("│  1. Usar BRØNSTED-LOWRY como modelo principal                                   │")
println("│     - Combina precisão com fundamentação química                                │")
println("│     - Explica mecanisticamente a autocatálise                                   │")
println("│                                                                                 │")
println("│  2. Apresentar THERMODYNAMIC como contribuição teórica                          │")
println("│     - Mostra entendimento profundo da físico-química                            │")
println("│     - Valores de ΔG°, ΔH‡, ΔS‡ publicáveis                                      │")
println("│                                                                                 │")
println("│  3. Usar CONSERVATIVE para validação cruzada                                    │")
println("│     - Benchmark empírico                                                        │")
println("│                                                                                 │")
println("└─────────────────────────────────────────────────────────────────────────────────┘")

println("\n✓ Comparação completa!")
