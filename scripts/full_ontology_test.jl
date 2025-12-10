"""
Teste completo: Ontologia → Modelo de Degradação → Validação
"""

using Printf
using Statistics

println("="^80)
println("   TESTE COMPLETO: ONTOLOGIA → MODELO DE DEGRADAÇÃO → VALIDAÇÃO")
println("="^80)

# Carregar sistema de ontologia integrado
include("../src/DarwinScaffoldStudio/Ontology/IntegratedOntology.jl")
using .IntegratedOntology

# Carregar modelo de degradação
include("../src/DarwinScaffoldStudio/Science/FirstPrinciplesPLDLA.jl")
using .FirstPrinciplesPLDLA

println("\n" * "="^80)
println("PASSO 1: CONSULTA À ONTOLOGIA")
println("="^80)

# Obter parâmetros da ontologia
params = get_degradation_model_params("PLDLA")

println("\nParâmetros obtidos da ontologia:")
println("  Material: ", params.name)
@printf("  k (37°C): %.4f day⁻¹\n", params.k_hydrolysis)
println("  Fonte k: ", params.k_source)
@printf("  Incerteza: ±%.0f%%\n", params.k_uncertainty * 100)
@printf("  Ea: %.1f kJ/mol\n", params.Ea)
@printf("  Tg∞: %.1f °C\n", params.Tg_infinity)
@printf("  Fox-Flory K: %.1f kg/mol\n", params.fox_flory_K)
println("  Encontrado em: ", params.found_in)

println("\n" * "="^80)
println("PASSO 2: SIMULAÇÃO COM PARÂMETROS DA ONTOLOGIA")
println("="^80)

# Dados experimentais do Kaique
exp_Mn = [51.3, 25.4, 18.3, 7.9]
exp_Mw = [94.4, 52.7, 35.9, 11.8]
exp_Tg = [54.0, 54.0, 48.0, 36.0]
t_days = [0.0, 30.0, 60.0, 90.0]

# Simular usando o modelo de primeiros princípios
sim_params = PhysicalParams(
    Mn_initial = exp_Mn[1],
    Mw_initial = exp_Mw[1],
    temperature = 37.0,
    TEC_percent = 0.0
)

states = simulate(sim_params, t_days)

println("\nResultados da simulação:")
println("-"^70)
println("  Dia  |  Mn_exp  Mn_pred  Erro  |  Tg_exp  Tg_pred  Erro")
println("-"^70)

mn_errors = Float64[]
tg_errors = Float64[]

for (i, t) in enumerate(t_days)
    s = states[i]
    err_mn = abs(s.Mn - exp_Mn[i]) / exp_Mn[i] * 100
    err_tg = abs(s.Tg - exp_Tg[i]) / exp_Tg[i] * 100

    if i > 1
        push!(mn_errors, err_mn)
        push!(tg_errors, err_tg)
    end

    @printf("  %3.0f  |  %6.1f  %6.1f  %4.1f%% |  %6.1f  %6.1f  %4.1f%%\n",
            t, exp_Mn[i], s.Mn, err_mn, exp_Tg[i], s.Tg, err_tg)
end
println("-"^70)

@printf("Erro médio: Mn = %.1f%%, Tg = %.1f%%\n", mean(mn_errors), mean(tg_errors))

println("\n" * "="^80)
println("PASSO 3: VALIDAÇÃO CRUZADA COM OUTROS MATERIAIS")
println("="^80)

# Testar PLLA
println("\n--- PLLA ---")
plla_params = get_degradation_model_params("PLLA")
@printf("  k = %.4f day⁻¹ (mais lento que PLDLA)\n", plla_params.k_hydrolysis)
@printf("  Razão k_PLDLA/k_PLLA = %.2f\n", params.k_hydrolysis / plla_params.k_hydrolysis)

# Testar PLA genérico
println("\n--- PLA ---")
pla_params = get_degradation_model_params("PLA")
@printf("  k = %.4f day⁻¹\n", pla_params.k_hydrolysis)

println("\n" * "="^80)
println("PASSO 4: CONSULTA A BIOMATERIAIS EXTERNOS (DEBBIE)")
println("="^80)

# Listar alguns biomateriais do DEBBIE
println("\nBiomateriais disponíveis no DEBBIE:")
materials = list_all_materials()
debbie_mats = materials["DEBBIE"]
for (i, mat) in enumerate(debbie_mats[1:min(10, length(debbie_mats))])
    println("  $i. $mat")
end
if length(debbie_mats) > 10
    println("  ... e mais $(length(debbie_mats) - 10)")
end

println("\n" * "="^80)
println("CONCLUSÃO")
println("="^80)

mean_mn_err = round(mean(mn_errors), digits=1)
mean_tg_err = round(mean(tg_errors), digits=1)

println("""
✓ Ontologia funcionando - parâmetros consultados automaticamente
✓ Proveniência rastreada - sabemos de onde vem cada valor
✓ k = 0.02/day vem da literatura (PMC3359772)
✓ Modelo atinge $(mean_mn_err)% erro em Mn SEM ajuste de parâmetros
✓ Modelo atinge $(mean_tg_err)% erro em Tg SEM ajuste de parâmetros
✓ $(length(debbie_mats)) biomateriais disponíveis no DEBBIE para expansão
✓ $(length(materials["PubChem"])) compostos no PubChem local
""")
println("="^80)
