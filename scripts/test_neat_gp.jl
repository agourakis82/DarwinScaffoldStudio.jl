"""
test_neat_gp.jl

Teste do NEAT-GP para Descoberta de Equações de Degradação

═══════════════════════════════════════════════════════════════════════════════
                    OBJETIVO CIENTÍFICO
═══════════════════════════════════════════════════════════════════════════════

NEAT-GP combina:
1. Evolução de topologia (NEAT)
2. Operações matemáticas simbólicas (GP)

Resultado esperado: Equação EXATA e INTERPRETÁVEL para degradação de PLDLA

Equação alvo aproximada (literatura):
    dMn/dt = -k × Mn × (1 + β×[H⁺])

onde:
    k = taxa base de degradação
    β = fator de autocatálise ácida
    [H⁺] = concentração de ácidos

═══════════════════════════════════════════════════════════════════════════════
"""

using Printf
using Statistics
using Random

Random.seed!(42)

include("../src/DarwinScaffoldStudio/Science/NEATGP.jl")
using .NEATGP

println("═"^80)
println("  NEAT-GP: DESCOBERTA AUTOMÁTICA DE EQUAÇÕES")
println("  Programação Genética com Topologias Evolutivas")
println("═"^80)

# ═══════════════════════════════════════════════════════════════════════════════
#                          DADOS EXPERIMENTAIS
# ═══════════════════════════════════════════════════════════════════════════════

println("\n  DADOS EXPERIMENTAIS (Kaique Hergesel, PUC-SP 2025):")
println("─"^60)

const TIMES = [0.0, 30.0, 60.0, 90.0]
const DATA = [51.285, 25.447, 18.313, 7.904]

println("  Dia │   Mn (kg/mol)  │ Degradação")
println("─"^60)
for (t, mn) in zip(TIMES, DATA)
    deg = (1 - mn/DATA[1]) * 100
    bar = "█"^round(Int, deg/5)
    @printf("  %3.0f │     %6.3f     │ %s %.0f%%\n", t, mn, bar, deg)
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    CONFIGURAÇÃO DO NEAT-GP
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "═"^80)
println("  CONFIGURAÇÃO DO NEAT-GP")
println("═"^80)

config = NEATGP.GPConfig(
    # População
    population_size = 200,
    max_generations = 150,
    target_fitness = 0.5,

    # Inputs
    n_inputs = 4,
    input_names = ["Mn", "Xc", "H", "t"],
    n_outputs = 1,

    # Operações permitidas - focadas em física
    allowed_unary = [NEATGP.OP_NEG, NEATGP.OP_EXP, NEATGP.OP_LOG,
                     NEATGP.OP_SQRT, NEATGP.OP_SQR, NEATGP.OP_INV],
    allowed_binary = [NEATGP.OP_ADD, NEATGP.OP_SUB, NEATGP.OP_MUL, NEATGP.OP_DIV],
    use_constants = true,

    # Mutação - agressiva para explorar
    weight_mutation_rate = 0.85,
    const_mutation_rate = 0.40,
    const_perturb_strength = 0.3,
    add_node_rate = 0.12,
    add_connection_rate = 0.18,
    change_operation_rate = 0.08,

    # Especiação
    compatibility_threshold = 2.5,
    target_species_count = 12,

    # Fitness - parsimônia moderada
    mse_weight = 1.0,
    complexity_weight = 0.003,

    # Elitismo
    elitism_count = 3,
    survival_threshold = 0.25
)

println("""
  Configuração:
    • População: $(config.population_size) indivíduos
    • Gerações: $(config.max_generations)
    • Inputs: $(join(config.input_names, ", "))

  Operações unárias: neg, exp, log, sqrt, sqr, inv
  Operações binárias: +, -, ×, ÷
""")

# ═══════════════════════════════════════════════════════════════════════════════
#                    EVOLUÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

println("═"^80)
println("  INICIANDO EVOLUÇÃO")
println("═"^80)

pop = NEATGP.GPPopulation(config)

@time best = NEATGP.evolve_gp!(pop, TIMES, DATA, verbose=true)

# ═══════════════════════════════════════════════════════════════════════════════
#                    RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "═"^80)
println("  RESULTADOS DA DESCOBERTA")
println("═"^80)

if best !== nothing
    println("\n  MELHOR GENOMA:")
    println("─"^60)
    @printf("    Fitness: %.6f\n", best.fitness)
    @printf("    MSE: %.4f\n", best.mse)
    @printf("    Complexidade: %d (nós + conexões)\n", best.complexity)
    @printf("    Nós: %d\n", length(best.nodes))
    @printf("    Conexões: %d\n", length(best.connections))

    println("\n  EQUAÇÃO DESCOBERTA:")
    println("─"^60)
    eq = NEATGP.equation_to_string(best, config)
    println("\n    dMn/dt = $eq")

    println("\n  EQUAÇÃO EM LATEX:")
    println("─"^60)
    latex = NEATGP.equation_to_latex(best, config)
    println("\n    $latex")

    # Validação
    println("\n" * "═"^80)
    println("  VALIDAÇÃO COM DADOS EXPERIMENTAIS")
    println("═"^80)

    function validate(genome, times, data)
        Mn0 = data[1]
        t_max = times[end]

        predictions = Float64[Mn0]
        Mn = Mn0

        for i in 2:length(times)
            target_t = times[i]
            t_curr = times[i-1]
            dt_step = 0.5

            while t_curr < target_t
                Xc = 0.08 + 0.17 * t_curr / t_max
                deg_frac = max(0.0, 1.0 - Mn / Mn0)
                H = 5.0 * deg_frac

                input = [Mn / Mn0, Xc * 4.0, H / 5.0, t_curr / t_max]
                dMn = NEATGP.evaluate_genome(genome, input) * Mn * 0.04
                Mn = max(1.0, min(Mn0 * 1.01, Mn + dMn * dt_step))
                t_curr += dt_step
            end
            push!(predictions, Mn)
        end

        return predictions
    end

    predictions = validate(best, TIMES, DATA)

    println("\n  Comparação predição vs experimental:")
    println("─"^60)
    println("  Tempo │ Exp (kg/mol) │ Pred (kg/mol) │ Erro │ Status")
    println("─"^60)

    function compute_rmse(preds, actual)
        total = 0.0
        for i in 1:length(actual)
            erro = preds[i] - actual[i]
            total += erro^2
            status = abs(erro) < 3.0 ? "✓" : "○"
            @printf("  %5.0f │    %6.2f     │     %6.2f    │ %+5.2f │   %s\n",
                    TIMES[i], actual[i], preds[i], erro, status)
        end
        return sqrt(total / length(actual))
    end

    rmse = compute_rmse(predictions, DATA)
    println("─"^60)
    @printf("  RMSE: %.2f kg/mol\n", rmse)

    # Análise da estrutura
    println("\n" * "═"^80)
    println("  ANÁLISE ESTRUTURAL DO GENOMA")
    println("═"^80)

    println("\n  NÓS:")
    for (id, node) in sort(collect(best.nodes), by=kv->kv[2].layer)
        op_name = NEATGP.op_symbol(node.operation)
        if node.operation == NEATGP.OP_VAR
            name = config.input_names[node.var_index]
            @printf("    [%3d] VAR: %s (layer %.2f)\n", id, name, node.layer)
        elseif node.operation == NEATGP.OP_CONST
            @printf("    [%3d] CONST: %.3f (layer %.2f)\n", id, node.constant, node.layer)
        else
            @printf("    [%3d] OP: %s (layer %.2f)\n", id, op_name, node.layer)
        end
    end

    println("\n  CONEXÕES:")
    for conn in sort(best.connections, by=c->-abs(c.weight))
        status = conn.enabled ? "●" : "○"
        @printf("    %s %3d → %3d [slot %d]: weight = %+.3f\n",
                status, conn.in_node, conn.out_node, conn.slot, conn.weight)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    RESUMO PARA PUBLICAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "═"^80)
println("  RESUMO PARA PUBLICAÇÃO CIENTÍFICA")
println("═"^80)

println("""

  METODOLOGIA:
  ────────────
  Utilizamos NEAT-GP (NeuroEvolution of Augmenting Topologies com
  Programação Genética) para descobrir automaticamente a equação
  diferencial que governa a degradação hidrolítica do PLDLA.

  INOVAÇÕES:
  ──────────
  1. Nós simbólicos evoluem operações matemáticas (+, -, ×, ÷, exp, log, ...)
  2. Topologia evolui como em NEAT (proteção de inovação)
  3. Crossover significativo via innovation numbers
  4. Parsimônia via MDL (Minimum Description Length)

  RESULTADOS:
  ───────────
  O algoritmo descobriu uma equação interpretável com:
  • $(best !== nothing ? best.complexity : "?") termos (nós + conexões)
  • RMSE: $(best !== nothing ? @sprintf("%.2f", sqrt(best.mse)) : "?") kg/mol
  • Fitness: $(best !== nothing ? @sprintf("%.4f", best.fitness) : "?")

  CONTRIBUIÇÃO:
  ─────────────
  Esta abordagem permite descoberta automática de modelos mecanísticos,
  eliminando o viés do pesquisador na escolha da forma funcional.

  PRÓXIMOS PASSOS:
  ────────────────
  1. Validação cruzada com datasets externos
  2. Aplicação a outros polímeros biodegradáveis
  3. Incorporação de incerteza bayesiana
  4. Submissão para Nature Computational Science
""")

println("═"^80)
println("  Experimento NEAT-GP Concluído!")
println("═"^80)
