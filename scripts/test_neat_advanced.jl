"""
test_neat_advanced.jl

Teste Extensivo do NEAT Avançado para Descoberta de Equações

═══════════════════════════════════════════════════════════════════════════════
                    DEEP THINKING: Filosofia do Teste
═══════════════════════════════════════════════════════════════════════════════

Este script testa a capacidade do NEAT de:

1. DESCOBRIR ESTRUTURAS ÓTIMAS
   - Começar com rede minimal
   - Adicionar complexidade apenas quando necessário
   - Encontrar topologias que capturam a física

2. APRENDER DINÂMICAS DE DEGRADAÇÃO
   - Taxa variável no tempo
   - Efeitos não-lineares (autocatálise)
   - Proteção cristalina

3. GENERALIZAR PARA NOVOS DADOS
   - Treinar em dados sintéticos
   - Validar em dados experimentais
   - Evitar overfitting

═══════════════════════════════════════════════════════════════════════════════
"""

using Printf
using Statistics
using Random

Random.seed!(42)

include("../src/DarwinScaffoldStudio/Science/NEATAdvanced.jl")
using .NEATAdvanced

println("═"^80)
println("  TESTE EXTENSIVO: NEAT AVANÇADO")
println("  Neuroevolução para Descoberta de Equações de Degradação")
println("═"^80)

# ═══════════════════════════════════════════════════════════════════════════════
#                         DADOS EXPERIMENTAIS
# ═══════════════════════════════════════════════════════════════════════════════

println("\n📊 DADOS EXPERIMENTAIS (Kaique Hergesel, PUC-SP 2025):")
println("─"^60)

const EXP_TIMES = [0.0, 30.0, 60.0, 90.0]
const EXP_DATA = [51.285, 25.447, 18.313, 7.904]

println("  Dia │   Mn (kg/mol)")
println("─"^60)
for (t, mn) in zip(EXP_TIMES, EXP_DATA)
    @printf("  %3.0f │   %6.3f\n", t, mn)
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    PARTE 1: TESTE BÁSICO
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "═"^80)
println("  PARTE 1: TESTE BÁSICO COM DADOS EXPERIMENTAIS")
println("═"^80)

config_basic = NEATAdvanced.AdvancedNEATConfig(
    population_size = 100,
    max_generations = 50,
    n_inputs = 4,
    n_outputs = 1,
    target_fitness = 0.3,

    # Mutação
    weight_mutation_rate = 0.8,
    add_node_rate = 0.03,
    add_connection_rate = 0.05,

    # Fitness
    fitness_mse_weight = 1.0,
    fitness_physics_weight = 0.2,
    fitness_complexity_weight = 0.02
)

pop_basic = NEATAdvanced.NEATPopulation(config_basic)
best_basic = NEATAdvanced.evolve!(pop_basic, EXP_TIMES, EXP_DATA, verbose=true)

println("\n🧬 MELHOR GENOMA ENCONTRADO:")
NEATAdvanced.visualize_genome(best_basic)

# ═══════════════════════════════════════════════════════════════════════════════
#                    PARTE 2: EVOLUÇÃO COM DADOS SINTÉTICOS
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "═"^80)
println("  PARTE 2: TREINO COM DADOS SINTÉTICOS")
println("═"^80)

println("\n🔬 Gerando dados sintéticos (modelo trifásico)...")

synth_times, synth_data = NEATAdvanced.generate_synthetic_degradation_data(
    n_points = 30,
    t_max = 100.0,
    noise_level = 0.03,
    model = :triphasic
)

println("  Gerados $(length(synth_times)) pontos de t=0 a t=$(synth_times[end])")

# Configuração mais agressiva para dados sintéticos
config_synth = NEATAdvanced.AdvancedNEATConfig(
    population_size = 150,
    max_generations = 100,
    n_inputs = 4,
    n_outputs = 1,
    target_fitness = 0.8,

    # Mutação mais agressiva
    weight_mutation_rate = 0.85,
    add_node_rate = 0.05,
    add_connection_rate = 0.08,
    activation_mutation_rate = 0.15,

    # Especiação
    target_species_count = 8,
    compatibility_threshold = 3.5,

    # Fitness balanceado
    fitness_mse_weight = 1.0,
    fitness_physics_weight = 0.3,
    fitness_complexity_weight = 0.01,
    fitness_smoothness_weight = 0.05
)

pop_synth = NEATAdvanced.NEATPopulation(config_synth)

# Callback para monitorar evolução
function evolution_callback(pop, gen)
    if gen % 20 == 0
        best = pop.best_genome
        # Calcular predição
        nn = NEATAdvanced.decode_to_function(best)

        # Predição em t=90
        Mn = EXP_DATA[1]
        for i in 2:4
            dt = EXP_TIMES[i] - EXP_TIMES[i-1]
            Xc = 0.08 + 0.17 * EXP_TIMES[i] / 90.0
            H = 5.0 * (1.0 - Mn / EXP_DATA[1])
            dMn = nn([Mn, Xc, H, EXP_TIMES[i]])[1]
            Mn = clamp(Mn + dMn * dt, 1.0, 100.0)
        end

        @printf("    → Predição Mn(90d): %.2f (exp: %.2f)\n", Mn, EXP_DATA[4])
    end
end

println("\n🧬 Evoluindo com dados sintéticos...")
best_synth = NEATAdvanced.evolve!(pop_synth, synth_times, synth_data;
                                   verbose=true, callback=evolution_callback)

println("\n🧬 MELHOR GENOMA (treino sintético):")
NEATAdvanced.visualize_genome(best_synth)

# ═══════════════════════════════════════════════════════════════════════════════
#                    PARTE 3: VALIDAÇÃO CRUZADA
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "═"^80)
println("  PARTE 3: VALIDAÇÃO COM DADOS EXPERIMENTAIS")
println("═"^80)

println("\n📈 Comparando predições nos dados experimentais:")
println("─"^60)
println("  Tempo │ Experimental │ NEAT Básico │ NEAT Sintético")
println("─"^60)

nn_basic = NEATAdvanced.decode_to_function(best_basic)
nn_synth = NEATAdvanced.decode_to_function(best_synth)

function predict_trajectory(nn, times, Mn0)
    predictions = [Mn0]
    Mn = Mn0
    t_max = times[end]
    dt_step = 0.5

    for i in 2:length(times)
        target_t = times[i]
        t_current = times[i-1]

        while t_current < target_t
            Xc = 0.08 + 0.17 * t_current / t_max
            deg_frac = max(0.0, 1.0 - Mn / Mn0)
            H = 5.0 * deg_frac

            # Mesma normalização usada no treinamento
            input = [Mn / Mn0, Xc * 4.0, H / 5.0, t_current / t_max]
            dMn = nn(input)[1] * Mn * 0.04

            Mn = max(1.0, min(Mn0 * 1.01, Mn + dMn * dt_step))
            t_current += dt_step
        end
        push!(predictions, Mn)
    end

    return predictions
end

pred_basic = predict_trajectory(nn_basic, EXP_TIMES, EXP_DATA[1])
pred_synth = predict_trajectory(nn_synth, EXP_TIMES, EXP_DATA[1])

rmse_basic = sqrt(mean((pred_basic .- EXP_DATA).^2))
rmse_synth = sqrt(mean((pred_synth .- EXP_DATA).^2))

for i in eachindex(EXP_TIMES)
    @printf("  %5.1f │    %6.2f    │    %6.2f   │    %6.2f\n",
            EXP_TIMES[i], EXP_DATA[i], pred_basic[i], pred_synth[i])
end
println("─"^60)
@printf("  RMSE  │      -       │    %6.2f   │    %6.2f\n", rmse_basic, rmse_synth)

# ═══════════════════════════════════════════════════════════════════════════════
#                    PARTE 4: ANÁLISE DA TOPOLOGIA EVOLUÍDA
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "═"^80)
println("  PARTE 4: ANÁLISE DA TOPOLOGIA EVOLUÍDA")
println("═"^80)

println("\n🧠 Estatísticas da evolução:")
println("─"^60)

@printf("  Gerações executadas: %d\n", pop_synth.generation)
@printf("  Espécies finais: %d\n", length(pop_synth.species))
@printf("  Melhor fitness: %.6f\n", best_synth.fitness)
@printf("  Nós hidden: %d\n", best_synth.n_hidden)
@printf("  Conexões ativas: %d\n", best_synth.n_connections)

# Estatísticas do Hall of Fame
if !isempty(pop_synth.hall_of_fame)
    println("\n🏆 Hall of Fame (top 5):")
    println("─"^60)
    println("  Rank │ Fitness │ Hidden │ Connections")
    println("─"^60)

    for (i, genome) in enumerate(pop_synth.hall_of_fame[1:min(5, length(pop_synth.hall_of_fame))])
        @printf("   %d   │ %.5f │   %2d   │     %2d\n",
                i, genome.fitness, genome.n_hidden, genome.n_connections)
    end
end

# Análise de conexões do melhor genoma
println("\n🔗 Conexões do melhor genoma:")
println("─"^60)

input_names = ["Mn", "Xc", "H", "t", "bias"]
output_names = ["dMn/dt"]

for conn in sort(best_synth.connections, by = c -> -abs(c.weight))
    if conn.enabled
        in_name = conn.in_node <= 5 ? input_names[conn.in_node] : "h$(conn.in_node)"
        out_name = conn.out_node == 6 ? output_names[1] : "h$(conn.out_node)"

        @printf("  %s → %s : %+.4f\n", in_name, out_name, conn.weight)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    PARTE 5: INTERPRETAÇÃO FÍSICA
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "═"^80)
println("  PARTE 5: INTERPRETAÇÃO FÍSICA DA REDE EVOLUÍDA")
println("═"^80)

println("""

CHAIN OF THOUGHT: O que a rede aprendeu?
───────────────────────────────────────

A rede evoluída pode ser interpretada como uma equação descoberta:

  dMn/dt = f(Mn, Xc, H, t)

Onde f é a função implementada pela rede neural.

ANÁLISE DAS CONEXÕES:
""")

# Identificar conexões mais importantes
important_connections = filter(c -> c.enabled && abs(c.weight) > 0.1, best_synth.connections)
sort!(important_connections, by = c -> -abs(c.weight))

println("  Conexões mais influentes (|w| > 0.1):")
println("─"^50)

for conn in important_connections[1:min(5, length(important_connections))]
    in_name = conn.in_node <= 5 ? input_names[conn.in_node] : "hidden_$(conn.in_node)"
    out_name = conn.out_node <= 6 ? (conn.out_node == 6 ? "dMn/dt" : input_names[conn.out_node]) : "hidden_$(conn.out_node)"

    sign = conn.weight > 0 ? "+" : "-"

    interpretation = ""
    if conn.in_node == 1  # Mn
        interpretation = conn.weight < 0 ? "→ Decaimento proporcional a Mn" : "→ Termo de crescimento?"
    elseif conn.in_node == 2  # Xc
        interpretation = conn.weight > 0 ? "→ Proteção cristalina" : "→ Xc acelera degradação?"
    elseif conn.in_node == 3  # H
        interpretation = conn.weight < 0 ? "→ Autocatálise ácida" : "→ Inibição por ácido?"
    elseif conn.in_node == 4  # t
        interpretation = "→ Dependência temporal direta"
    elseif conn.in_node == 5  # bias
        interpretation = "→ Taxa basal constante"
    end

    @printf("  %s ─(%s%.3f)─▶ %s  %s\n", in_name, sign, abs(conn.weight), out_name, interpretation)
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    PARTE 6: GERAR DOT PARA GRAPHVIZ
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "═"^80)
println("  PARTE 6: VISUALIZAÇÃO GRAPHVIZ")
println("═"^80)

dot_string = NEATAdvanced.genome_to_dot(best_synth)

println("\n📊 Código DOT para visualização (copie para graphviz.org):")
println("─"^60)
println(dot_string)

# Salvar arquivo DOT
dot_file = "neat_best_genome.dot"
open(dot_file, "w") do f
    write(f, dot_string)
end
println("\n✓ Arquivo DOT salvo em: $dot_file")

# ═══════════════════════════════════════════════════════════════════════════════
#                         RESUMO FINAL
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "═"^80)
println("  RESUMO: NEAT AVANÇADO PARA DEGRADAÇÃO DE PLDLA")
println("═"^80)

println("""

  🧬 EVOLUÇÃO CONCLUÍDA:
     • Gerações: $(pop_synth.generation)
     • População final: $(config_synth.population_size)
     • Espécies: $(length(pop_synth.species))
     • Melhor fitness: $(round(best_synth.fitness, digits=6))

  🧠 TOPOLOGIA DESCOBERTA:
     • Nós de entrada: 4 (Mn, Xc, H, t) + 1 bias
     • Nós hidden: $(best_synth.n_hidden)
     • Nós de saída: 1 (dMn/dt)
     • Conexões ativas: $(best_synth.n_connections)

  📊 PERFORMANCE:
     • RMSE (dados experimentais): $(round(rmse_synth, digits=2)) kg/mol
     • RMSE (baseline básico): $(round(rmse_basic, digits=2)) kg/mol

  🎯 PRÓXIMOS PASSOS:
     1. Extrair equação simbólica via SINDy
     2. Quantificar incerteza via Bayesian
     3. Validar com datasets externos
     4. Preparar para publicação

""")

println("═"^80)
println("  Teste Completo!")
println("═"^80)
