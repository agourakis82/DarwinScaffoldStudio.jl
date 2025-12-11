"""
NEATGP.jl

NEAT-GP Híbrido: Programação Genética com Topologias Evolutivas

═══════════════════════════════════════════════════════════════════════════════
                    DEEP THINKING: Por que NEAT-GP?
═══════════════════════════════════════════════════════════════════════════════

CHAIN OF THOUGHT: A fusão de dois paradigmas poderosos
──────────────────────────────────────────────────────

NEAT (NeuroEvolution of Augmenting Topologies):
  ✓ Evolui topologia de redes neurais
  ✓ Proteção de inovação via especiação
  ✓ Crossover significativo via innovation numbers
  ✗ Usa ativações fixas (tanh, relu) - não interpretável

GP (Genetic Programming):
  ✓ Evolui expressões simbólicas (árvores)
  ✓ Descoberta de equações exatas
  ✓ Interpretabilidade total
  ✗ Crossover problemático (bloat, destruição)
  ✗ Não protege inovações estruturais

NEAT-GP HÍBRIDO:
  ✓ Nós são OPERAÇÕES MATEMÁTICAS (+, -, *, /, sin, exp, log, ...)
  ✓ Topologia evolui como em NEAT (com proteção de inovação)
  ✓ Crossover controlado via innovation numbers
  ✓ Descoberta de EQUAÇÕES EXATAS
  ✓ Interpretabilidade para publicação científica

═══════════════════════════════════════════════════════════════════════════════
                    EXEMPLO DE DESCOBERTA
═══════════════════════════════════════════════════════════════════════════════

Dado: Dados de degradação de polímero (Mn vs t)

NEAT tradicional descobre: dMn/dt ≈ tanh(w₁*Mn + w₂*t + ...)
  → Numericamente correto, mas não interpretável

NEAT-GP descobre: dMn/dt = -k × Mn^α × exp(-Ea/RT)
  → Equação de Arrhenius modificada com ordem fracionária!
  → Publicável em Nature/Science

═══════════════════════════════════════════════════════════════════════════════

Author: Darwin Scaffold Studio
Date: 2025-12-11
Target: Nature Computational Science
"""

module NEATGP

using Random
using Statistics
using LinearAlgebra
using Printf

export GPConfig, GPGenome, GPNode, GPConnection
export GPPopulation, evolve_gp!, get_best_equation
export equation_to_string, equation_to_latex

# ═══════════════════════════════════════════════════════════════════════════════
#                    OPERAÇÕES MATEMÁTICAS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Operações matemáticas disponíveis para os nós.

DEEP THINKING: Escolha das operações
────────────────────────────────────

Incluímos operações que aparecem em leis físicas:
- Aritmética: +, -, *, / (combinações lineares, razões)
- Exponencial: exp, log (Arrhenius, decaimento)
- Potência: x^n (leis de potência)
- Trigonométrica: sin, cos (oscilações)

NÃO incluímos operações que raramente aparecem em física:
- Funções de ativação (tanh, relu) - são aproximações
- Funções especiais (Bessel, Gamma) - muito específicas
"""
@enum GPOperation begin
    # Operações binárias (arity = 2)
    OP_ADD = 1        # x + y
    OP_SUB = 2        # x - y
    OP_MUL = 3        # x * y
    OP_DIV = 4        # x / y (protected)
    OP_POW = 5        # x^y (protected)

    # Operações unárias (arity = 1)
    OP_NEG = 6        # -x
    OP_ABS = 7        # |x|
    OP_SIN = 8        # sin(x)
    OP_COS = 9        # cos(x)
    OP_EXP = 10       # exp(x) (protected)
    OP_LOG = 11       # log(x) (protected)
    OP_SQRT = 12      # sqrt(|x|)
    OP_SQR = 13       # x²
    OP_CUBE = 14      # x³
    OP_INV = 15       # 1/x (protected)
    OP_TANH = 16      # tanh(x) - para suavização

    # Terminais (arity = 0)
    OP_VAR = 17       # Variável de entrada
    OP_CONST = 18     # Constante evoluída
end

"""
Aridade de cada operação.
"""
function get_arity(op::GPOperation)
    if op in (OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW)
        return 2
    elseif op in (OP_VAR, OP_CONST)
        return 0
    else
        return 1
    end
end

"""
Símbolo para representação de equação.
"""
function op_symbol(op::GPOperation)
    symbols = Dict(
        OP_ADD => "+", OP_SUB => "-", OP_MUL => "×", OP_DIV => "÷",
        OP_POW => "^", OP_NEG => "-", OP_ABS => "abs", OP_SIN => "sin",
        OP_COS => "cos", OP_EXP => "exp", OP_LOG => "log", OP_SQRT => "√",
        OP_SQR => "²", OP_CUBE => "³", OP_INV => "1/", OP_TANH => "tanh",
        OP_VAR => "x", OP_CONST => "c"
    )
    return get(symbols, op, "?")
end

"""
Avalia uma operação com proteção numérica.
"""
function safe_eval(op::GPOperation, args::Vector{Float64}, const_val::Float64=0.0)
    # Proteção contra overflow
    args = clamp.(args, -1e6, 1e6)

    result = if op == OP_ADD
        args[1] + args[2]
    elseif op == OP_SUB
        args[1] - args[2]
    elseif op == OP_MUL
        args[1] * args[2]
    elseif op == OP_DIV
        abs(args[2]) < 1e-10 ? args[1] : args[1] / args[2]
    elseif op == OP_POW
        if args[1] <= 0 || abs(args[2]) > 10
            1.0
        else
            args[1]^clamp(args[2], -10, 10)
        end
    elseif op == OP_NEG
        -args[1]
    elseif op == OP_ABS
        abs(args[1])
    elseif op == OP_SIN
        sin(args[1])
    elseif op == OP_COS
        cos(args[1])
    elseif op == OP_EXP
        exp(clamp(args[1], -20, 20))
    elseif op == OP_LOG
        args[1] <= 0 ? 0.0 : log(args[1])
    elseif op == OP_SQRT
        sqrt(abs(args[1]))
    elseif op == OP_SQR
        args[1]^2
    elseif op == OP_CUBE
        args[1]^3
    elseif op == OP_INV
        abs(args[1]) < 1e-10 ? 0.0 : 1.0 / args[1]
    elseif op == OP_TANH
        tanh(args[1])
    elseif op == OP_VAR
        args[1]  # Valor já passado
    elseif op == OP_CONST
        const_val
    else
        0.0
    end

    return isfinite(result) ? clamp(result, -1e10, 1e10) : 0.0
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    ESTRUTURAS GENÉTICAS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Nó no grafo de programa genético.
"""
mutable struct GPNode
    id::Int
    operation::GPOperation
    var_index::Int        # Para OP_VAR: qual input (1, 2, 3, ...)
    constant::Float64     # Para OP_CONST: valor da constante
    layer::Float64        # Posição topológica

    function GPNode(id, op; var_idx=0, const_val=0.0, layer=0.5)
        new(id, op, var_idx, const_val, layer)
    end
end

"""
Conexão entre nós (como em NEAT).
"""
mutable struct GPConnection
    in_node::Int
    out_node::Int
    slot::Int            # Qual input do nó destino (1 ou 2 para binários)
    weight::Float64      # Peso da conexão
    enabled::Bool
    innovation::Int

    GPConnection(in_n, out_n, slot, innov) =
        new(in_n, out_n, slot, 1.0, true, innov)
end

"""
Genoma completo de programa genético.
"""
mutable struct GPGenome
    id::Int
    nodes::Dict{Int, GPNode}
    connections::Vector{GPConnection}

    # Fitness
    fitness::Float64
    adjusted_fitness::Float64
    mse::Float64
    complexity::Int

    # Metadata
    species_id::Int
    generation_born::Int

    function GPGenome(id::Int)
        new(id, Dict{Int, GPNode}(), GPConnection[],
            0.0, 0.0, Inf, 0, 0, 0)
    end
end

"""
Rastreador de inovações.
"""
mutable struct InnovationTracker
    current_innovation::Int
    current_node_id::Int
    history::Dict{Tuple{Int,Int,Int}, Int}  # (in, out, slot) → innovation

    InnovationTracker() = new(0, 100, Dict())
end

function get_innovation!(tracker::InnovationTracker, in_node, out_node, slot)
    key = (in_node, out_node, slot)
    if haskey(tracker.history, key)
        return tracker.history[key]
    else
        tracker.current_innovation += 1
        tracker.history[key] = tracker.current_innovation
        return tracker.current_innovation
    end
end

function new_node_id!(tracker::InnovationTracker)
    tracker.current_node_id += 1
    return tracker.current_node_id
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

"""
Configuração do NEAT-GP.
"""
Base.@kwdef mutable struct GPConfig
    # População
    population_size::Int = 200
    max_generations::Int = 300
    target_fitness::Float64 = 0.99

    # Inputs/Outputs
    n_inputs::Int = 4          # [Mn, Xc, H, t]
    input_names::Vector{String} = ["Mn", "Xc", "H", "t"]
    n_outputs::Int = 1         # dMn/dt

    # Operações permitidas
    allowed_unary::Vector{GPOperation} = [OP_NEG, OP_ABS, OP_EXP, OP_LOG,
                                           OP_SQRT, OP_SQR, OP_CUBE, OP_INV]
    allowed_binary::Vector{GPOperation} = [OP_ADD, OP_SUB, OP_MUL, OP_DIV]
    use_constants::Bool = true

    # Mutação
    weight_mutation_rate::Float64 = 0.80
    const_mutation_rate::Float64 = 0.30
    const_perturb_strength::Float64 = 0.5

    add_node_rate::Float64 = 0.10
    add_connection_rate::Float64 = 0.15
    change_operation_rate::Float64 = 0.05
    disable_connection_rate::Float64 = 0.02

    # Especiação
    compatibility_c1::Float64 = 1.0
    compatibility_c2::Float64 = 1.0
    compatibility_c3::Float64 = 0.3
    compatibility_threshold::Float64 = 3.0
    target_species_count::Int = 10

    # Fitness
    mse_weight::Float64 = 1.0
    complexity_weight::Float64 = 0.005  # Parsimônia forte
    physics_weight::Float64 = 0.1

    # Elitismo
    elitism_count::Int = 3
    survival_threshold::Float64 = 0.20
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    CRIAÇÃO DE GENOMAS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Cria um genoma minimal (apenas inputs → output).
"""
function create_minimal_genome(id::Int, config::GPConfig, tracker::InnovationTracker)
    genome = GPGenome(id)

    # Nós de input (uma variável para cada)
    for i in 1:config.n_inputs
        node = GPNode(i, OP_VAR; var_idx=i, layer=0.0)
        genome.nodes[i] = node
    end

    # Nó de constante (bias)
    if config.use_constants
        const_id = config.n_inputs + 1
        genome.nodes[const_id] = GPNode(const_id, OP_CONST;
                                         const_val=randn(), layer=0.0)
    end

    # Nó de output - começa como soma ponderada
    output_id = 100
    genome.nodes[output_id] = GPNode(output_id, OP_ADD; layer=1.0)

    # Conectar inputs ao output
    n_inputs_total = config.use_constants ? config.n_inputs + 1 : config.n_inputs

    for i in 1:min(2, n_inputs_total)  # ADD precisa de 2 inputs
        innov = get_innovation!(tracker, i, output_id, i)
        conn = GPConnection(i, output_id, i, innov)
        conn.weight = randn() * 0.5
        push!(genome.connections, conn)
    end

    genome.complexity = length(genome.nodes) + length(genome.connections)

    return genome
end

"""
Cria população inicial.
"""
function create_population(config::GPConfig, tracker::InnovationTracker)
    genomes = GPGenome[]

    for i in 1:config.population_size
        genome = create_minimal_genome(i, config, tracker)
        push!(genomes, genome)
    end

    return genomes
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    AVALIAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

"""
Avalia o genoma em um ponto de dados.
"""
function evaluate_genome(genome::GPGenome, inputs::Vector{Float64})
    # Valores dos nós
    node_values = Dict{Int, Float64}()

    # Inicializar inputs
    for (id, node) in genome.nodes
        if node.operation == OP_VAR
            if node.var_index >= 1 && node.var_index <= length(inputs)
                node_values[id] = inputs[node.var_index]
            else
                node_values[id] = 0.0
            end
        elseif node.operation == OP_CONST
            node_values[id] = node.constant
        end
    end

    # Ordenar nós por camada
    sorted_nodes = sort(collect(Base.values(genome.nodes)), by=n->n.layer)

    # Forward pass
    for node in sorted_nodes
        if node.operation in (OP_VAR, OP_CONST)
            continue
        end

        # Coletar inputs para este nó
        incoming = filter(c -> c.out_node == node.id && c.enabled,
                         genome.connections)

        if isempty(incoming)
            node_values[node.id] = 0.0
            continue
        end

        # Ordenar por slot
        sort!(incoming, by=c->c.slot)

        arity = get_arity(node.operation)
        args = Float64[]

        for slot in 1:arity
            conn = findfirst(c -> c.slot == slot, incoming)
            if conn !== nothing
                val = get(node_values, incoming[conn].in_node, 0.0) * incoming[conn].weight
                push!(args, val)
            else
                push!(args, 0.0)
            end
        end

        node_values[node.id] = safe_eval(node.operation, args, node.constant)
    end

    # Retornar output (nó com maior layer)
    all_nodes = collect(Base.values(genome.nodes))
    output_node = all_nodes[1]
    for n in all_nodes
        if n.layer > output_node.layer
            output_node = n
        end
    end
    return get(node_values, output_node.id, 0.0)
end

"""
Avalia fitness do genoma nos dados.
"""
function evaluate_fitness!(genome::GPGenome, times::Vector{Float64},
                            data::Vector{Float64}, config::GPConfig)
    Mn0 = data[1]
    t_max = times[end]

    # Integrar ODE como em NEATAdvanced
    mse = 0.0
    predictions = Float64[Mn0]

    try
        dt_step = 0.5
        Mn = Mn0
        t_current = 0.0
        next_obs_idx = 2

        while t_current < t_max + dt_step && next_obs_idx <= length(times)
            # Variáveis auxiliares
            Xc = 0.08 + 0.17 * t_current / t_max
            deg_frac = max(0.0, 1.0 - Mn / Mn0)
            acid_conc = 5.0 * deg_frac

            # Normalizar inputs
            input = [Mn / Mn0, Xc * 4.0, acid_conc / 5.0, t_current / t_max]

            # Avaliar rede GP
            dMn_dt = evaluate_genome(genome, input)

            # Escalar output
            dMn_dt = dMn_dt * Mn * 0.04

            # Integrar
            Mn = Mn + dMn_dt * dt_step
            Mn = max(1.0, min(Mn0 * 1.01, Mn))

            t_current += dt_step

            if next_obs_idx <= length(times) && t_current >= times[next_obs_idx] - 0.25
                push!(predictions, Mn)
                mse += (Mn - data[next_obs_idx])^2
                next_obs_idx += 1
            end
        end

        while length(predictions) < length(data)
            push!(predictions, max(1.0, predictions[end]))
            mse += (predictions[end] - data[length(predictions)])^2
        end

        mse /= max(1, length(data) - 1)

    catch e
        genome.fitness = 1e-10
        genome.mse = Inf
        return
    end

    genome.mse = mse
    genome.complexity = length(genome.nodes) + length(genome.connections)

    # Fitness multi-objetivo
    total_penalty = (config.mse_weight * mse +
                     config.complexity_weight * genome.complexity)

    genome.fitness = 1.0 / (total_penalty + 1e-6)
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    MUTAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

"""
Muta pesos e constantes.
"""
function mutate_weights!(genome::GPGenome, config::GPConfig)
    for conn in genome.connections
        if rand() < config.weight_mutation_rate
            if rand() < 0.9
                conn.weight += randn() * config.const_perturb_strength
            else
                conn.weight = randn() * 2.0
            end
        end
    end

    for (id, node) in genome.nodes
        if node.operation == OP_CONST && rand() < config.const_mutation_rate
            if rand() < 0.9
                node.constant += randn() * config.const_perturb_strength
            else
                node.constant = randn() * 2.0
            end
        end
    end
end

"""
Adiciona um novo nó (divide uma conexão).
"""
function mutate_add_node!(genome::GPGenome, config::GPConfig, tracker::InnovationTracker)
    enabled_conns = filter(c -> c.enabled, genome.connections)
    if isempty(enabled_conns)
        return
    end

    # Escolher conexão para dividir
    conn = rand(enabled_conns)
    conn.enabled = false

    # Criar novo nó
    new_id = new_node_id!(tracker)

    # Escolher operação unária aleatória
    op = rand(config.allowed_unary)

    # Calcular layer intermediária
    in_node = genome.nodes[conn.in_node]
    out_node = genome.nodes[conn.out_node]
    new_layer = (in_node.layer + out_node.layer) / 2

    genome.nodes[new_id] = GPNode(new_id, op; layer=new_layer)

    # Criar conexões
    innov1 = get_innovation!(tracker, conn.in_node, new_id, 1)
    innov2 = get_innovation!(tracker, new_id, conn.out_node, conn.slot)

    push!(genome.connections, GPConnection(conn.in_node, new_id, 1, innov1))
    genome.connections[end].weight = 1.0

    push!(genome.connections, GPConnection(new_id, conn.out_node, conn.slot, innov2))
    genome.connections[end].weight = conn.weight
end

"""
Adiciona uma nova conexão.
"""
function mutate_add_connection!(genome::GPGenome, config::GPConfig, tracker::InnovationTracker)
    node_ids = collect(keys(genome.nodes))

    for _ in 1:20  # Tentativas
        in_id = rand(node_ids)
        out_id = rand(node_ids)

        in_node = genome.nodes[in_id]
        out_node = genome.nodes[out_id]

        # Verificar validade (feedforward, slot disponível)
        if in_node.layer >= out_node.layer
            continue
        end

        # Verificar se conexão já existe
        slot = rand(1:max(1, get_arity(out_node.operation)))
        exists = any(c -> c.in_node == in_id && c.out_node == out_id && c.slot == slot,
                    genome.connections)
        if exists
            continue
        end

        innov = get_innovation!(tracker, in_id, out_id, slot)
        push!(genome.connections, GPConnection(in_id, out_id, slot, innov))
        genome.connections[end].weight = randn() * 0.5
        return
    end
end

"""
Muda a operação de um nó.
"""
function mutate_operation!(genome::GPGenome, config::GPConfig)
    hidden_nodes = filter(kv -> kv[2].layer > 0 && kv[2].layer < 1,
                          collect(genome.nodes))
    if isempty(hidden_nodes)
        return
    end

    id, node = rand(hidden_nodes)

    if get_arity(node.operation) == 1
        node.operation = rand(config.allowed_unary)
    else
        node.operation = rand(config.allowed_binary)
    end
end

"""
Mutação completa do genoma.
"""
function mutate!(genome::GPGenome, config::GPConfig, tracker::InnovationTracker)
    mutate_weights!(genome, config)

    if rand() < config.add_node_rate
        mutate_add_node!(genome, config, tracker)
    end

    if rand() < config.add_connection_rate
        mutate_add_connection!(genome, config, tracker)
    end

    if rand() < config.change_operation_rate
        mutate_operation!(genome, config)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    CROSSOVER
# ═══════════════════════════════════════════════════════════════════════════════

"""
Crossover entre dois genomas (como em NEAT).
"""
function crossover(parent1::GPGenome, parent2::GPGenome, child_id::Int)
    # parent1 tem melhor fitness
    if parent2.fitness > parent1.fitness
        parent1, parent2 = parent2, parent1
    end

    child = GPGenome(child_id)

    # Herdar nós do melhor pai
    for (id, node) in parent1.nodes
        child.nodes[id] = GPNode(id, node.operation;
                                  var_idx=node.var_index,
                                  const_val=node.constant,
                                  layer=node.layer)
    end

    # Genes de conexão por innovation number
    p1_innovations = Dict(c.innovation => c for c in parent1.connections)
    p2_innovations = Dict(c.innovation => c for c in parent2.connections)

    all_innovations = union(keys(p1_innovations), keys(p2_innovations))

    for innov in all_innovations
        if haskey(p1_innovations, innov) && haskey(p2_innovations, innov)
            # Matching gene - escolher aleatoriamente
            conn = rand([p1_innovations[innov], p2_innovations[innov]])
            new_conn = GPConnection(conn.in_node, conn.out_node, conn.slot, conn.innovation)
            new_conn.weight = conn.weight
            new_conn.enabled = conn.enabled || rand() < 0.75
            push!(child.connections, new_conn)
        elseif haskey(p1_innovations, innov)
            # Excess/disjoint do melhor pai
            conn = p1_innovations[innov]
            new_conn = GPConnection(conn.in_node, conn.out_node, conn.slot, conn.innovation)
            new_conn.weight = conn.weight
            new_conn.enabled = conn.enabled
            push!(child.connections, new_conn)
        end
    end

    return child
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    ESPECIAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

"""
Distância de compatibilidade entre genomas.
"""
function compatibility_distance(g1::GPGenome, g2::GPGenome, config::GPConfig)
    innov1 = Set(c.innovation for c in g1.connections)
    innov2 = Set(c.innovation for c in g2.connections)

    matching = intersect(innov1, innov2)
    excess = length(setdiff(innov1, innov2)) + length(setdiff(innov2, innov1))

    # Diferença de pesos em genes matching
    weight_diff = 0.0
    if !isempty(matching)
        for innov in matching
            c1 = findfirst(c -> c.innovation == innov, g1.connections)
            c2 = findfirst(c -> c.innovation == innov, g2.connections)
            if c1 !== nothing && c2 !== nothing
                weight_diff += abs(g1.connections[c1].weight - g2.connections[c2].weight)
            end
        end
        weight_diff /= length(matching)
    end

    N = max(length(g1.connections), length(g2.connections), 1)

    return (config.compatibility_c1 * excess / N +
            config.compatibility_c3 * weight_diff)
end

"""
Espécie de genomas.
"""
mutable struct GPSpecies
    id::Int
    members::Vector{GPGenome}
    representative::GPGenome
    best_fitness::Float64
    stagnation::Int
end

GPSpecies(id, rep) = GPSpecies(id, [rep], rep, 0.0, 0)

# ═══════════════════════════════════════════════════════════════════════════════
#                    POPULAÇÃO E EVOLUÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

"""
População de NEAT-GP.
"""
mutable struct GPPopulation
    config::GPConfig
    genomes::Vector{GPGenome}
    species::Vector{GPSpecies}
    tracker::InnovationTracker
    generation::Int
    best_genome::Union{GPGenome, Nothing}
end

function GPPopulation(config::GPConfig)
    tracker = InnovationTracker()
    genomes = create_population(config, tracker)
    GPPopulation(config, genomes, GPSpecies[], tracker, 0, nothing)
end

"""
Especiação da população.
"""
function speciate!(pop::GPPopulation)
    # Limpar membros
    for sp in pop.species
        empty!(sp.members)
    end

    for genome in pop.genomes
        placed = false

        for sp in pop.species
            if compatibility_distance(genome, sp.representative, pop.config) <
               pop.config.compatibility_threshold
                push!(sp.members, genome)
                placed = true
                break
            end
        end

        if !placed
            new_sp = GPSpecies(length(pop.species) + 1, genome)
            push!(new_sp.members, genome)
            push!(pop.species, new_sp)
        end
    end

    # Remover espécies vazias
    filter!(sp -> !isempty(sp.members), pop.species)

    # Atualizar representantes
    for sp in pop.species
        sp.representative = rand(sp.members)
    end
end

"""
Evolui uma geração.
"""
function evolve_generation!(pop::GPPopulation, times::Vector{Float64},
                             data::Vector{Float64})
    pop.generation += 1

    # Avaliar fitness
    for genome in pop.genomes
        evaluate_fitness!(genome, times, data, pop.config)
    end

    # Atualizar melhor
    best_genome = pop.genomes[1]
    for g in pop.genomes
        if g.fitness > best_genome.fitness
            best_genome = g
        end
    end
    if pop.best_genome === nothing || best_genome.fitness > pop.best_genome.fitness
        pop.best_genome = best_genome
    end

    # Especiação
    speciate!(pop)

    # Calcular adjusted fitness
    for sp in pop.species
        for genome in sp.members
            genome.adjusted_fitness = genome.fitness / length(sp.members)
        end
    end

    # Reprodução
    new_genomes = GPGenome[]

    total_adjusted = sum(g.adjusted_fitness for g in pop.genomes)

    for sp in pop.species
        # Elitismo
        sort!(sp.members, by=g->-g.fitness)
        for i in 1:min(pop.config.elitism_count, length(sp.members))
            push!(new_genomes, sp.members[i])
        end

        # Número de filhos proporcional ao adjusted fitness
        sp_adjusted = sum(g.adjusted_fitness for g in sp.members)
        n_offspring = round(Int, sp_adjusted / total_adjusted * pop.config.population_size)
        n_offspring = max(1, n_offspring - pop.config.elitism_count)

        # Seleção e reprodução
        survivors = sp.members[1:max(1, round(Int, length(sp.members) * pop.config.survival_threshold))]

        for _ in 1:n_offspring
            if length(new_genomes) >= pop.config.population_size
                break
            end

            if length(survivors) >= 2 && rand() < 0.75
                p1, p2 = rand(survivors, 2)
                child = crossover(p1, p2, length(new_genomes) + 1)
            else
                child = deepcopy(rand(survivors))
                child.id = length(new_genomes) + 1
            end

            mutate!(child, pop.config, pop.tracker)
            push!(new_genomes, child)
        end
    end

    # Preencher se necessário
    while length(new_genomes) < pop.config.population_size
        genome = create_minimal_genome(length(new_genomes) + 1, pop.config, pop.tracker)
        mutate!(genome, pop.config, pop.tracker)
        push!(new_genomes, genome)
    end

    pop.genomes = new_genomes[1:pop.config.population_size]
end

"""
Evolução completa.
"""
function evolve_gp!(pop::GPPopulation, times::Vector{Float64}, data::Vector{Float64};
                     verbose::Bool=true)
    if verbose
        println("\n" * "═"^70)
        println("  NEAT-GP: Descoberta de Equações por Programação Genética")
        println("═"^70)
        println("\n  Configuração:")
        println("    População: $(pop.config.population_size)")
        println("    Gerações máx: $(pop.config.max_generations)")
        println("    Inputs: $(pop.config.input_names)")
        println()
    end

    for gen in 1:pop.config.max_generations
        evolve_generation!(pop, times, data)

        if verbose && (gen <= 5 || gen % 10 == 0 || gen == pop.config.max_generations)
            best = pop.best_genome
            @printf("  Gen %3d │ Fitness: %.4f │ MSE: %6.2f │ Complexity: %3d │ Species: %d\n",
                    gen, best.fitness, best.mse, best.complexity, length(pop.species))
        end

        if pop.best_genome !== nothing && pop.best_genome.fitness >= pop.config.target_fitness
            if verbose
                println("\n  Alvo de fitness atingido!")
            end
            break
        end
    end

    return pop.best_genome
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    EXTRAÇÃO DE EQUAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

"""
Converte genoma para string de equação.
"""
function equation_to_string(genome::GPGenome, config::GPConfig)
    # Ordenar nós por layer
    sorted = sort(collect(genome.nodes), by=kv->kv[2].layer)

    # Construir expressão recursivamente
    expressions = Dict{Int, String}()

    for (id, node) in sorted
        if node.operation == OP_VAR
            expressions[id] = config.input_names[node.var_index]
        elseif node.operation == OP_CONST
            expressions[id] = @sprintf("%.3f", node.constant)
        else
            # Coletar inputs
            incoming = filter(c -> c.out_node == id && c.enabled, genome.connections)
            sort!(incoming, by=c->c.slot)

            args = String[]
            for conn in incoming
                expr = get(expressions, conn.in_node, "?")
                if abs(conn.weight - 1.0) > 0.01
                    expr = @sprintf("%.2f*%s", conn.weight, expr)
                end
                push!(args, expr)
            end

            sym = op_symbol(node.operation)

            if get_arity(node.operation) == 2 && length(args) >= 2
                expressions[id] = "($(args[1]) $sym $(args[2]))"
            elseif get_arity(node.operation) == 1 && length(args) >= 1
                expressions[id] = "$sym($(args[1]))"
            else
                expressions[id] = "0"
            end
        end
    end

    # Retornar expressão do output (nó com maior layer)
    all_pairs = collect(genome.nodes)
    output_id = all_pairs[1][1]
    max_layer = all_pairs[1][2].layer
    for (id, node) in all_pairs
        if node.layer > max_layer
            max_layer = node.layer
            output_id = id
        end
    end
    return get(expressions, output_id, "?")
end

"""
Converte genoma para LaTeX.
"""
function equation_to_latex(genome::GPGenome, config::GPConfig)
    str = equation_to_string(genome, config)

    # Substituições básicas para LaTeX
    str = replace(str, "×" => "\\times")
    str = replace(str, "÷" => "\\div")
    str = replace(str, "sqrt" => "\\sqrt")
    str = replace(str, "exp" => "\\exp")
    str = replace(str, "log" => "\\ln")
    str = replace(str, "sin" => "\\sin")
    str = replace(str, "cos" => "\\cos")

    return "\$\$ \\frac{dMn}{dt} = $str \$\$"
end

"""
Obtém melhor equação encontrada.
"""
function get_best_equation(pop::GPPopulation)
    if pop.best_genome === nothing
        return "No equation found"
    end
    return equation_to_string(pop.best_genome, pop.config)
end

end # module
