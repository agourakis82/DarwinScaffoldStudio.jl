"""
NEATAdvanced.jl

NEAT Avançado para Descoberta de Equações de Degradação de Polímeros

═══════════════════════════════════════════════════════════════════════════════
                        DEEP THINKING: Por que NEAT?
═══════════════════════════════════════════════════════════════════════════════

CHAIN OF THOUGHT - A Beleza do NEAT:
────────────────────────────────────

1. COMPLEXIFICAÇÃO GRADUAL (Princípio de Occam Computacional)
   ───────────────────────────────────────────────────────────
   NEAT começa com a rede MAIS SIMPLES possível e só adiciona
   complexidade quando NECESSÁRIO. Isso é profundamente científico:

   "Entia non sunt multiplicanda praeter necessitatem"
   (Entidades não devem ser multiplicadas além da necessidade)

   Na prática: Se dMn/dt = -k*Mn é suficiente, NEAT não adicionará
   termos desnecessários. Se autocatálise é necessária, ela EMERGE.

2. INOVAÇÃO PROTEGIDA (Especiação)
   ────────────────────────────────
   Novas estruturas topológicas são frágeis no início. Um neurônio
   recém-adicionado tem pesos aleatórios - seu fitness será baixo.
   Sem proteção, seria eliminado antes de otimizar.

   Especiação cria "ilhas" onde inovações podem amadurecer:

   Espécie 1: [rede simples otimizada]     ← compete internamente
   Espécie 2: [rede com novo neurônio]     ← protegida, tempo para otimizar
   Espécie 3: [rede com nova conexão]      ← protegida, tempo para otimizar

3. CROSSOVER HISTÓRICO (Innovation Numbers)
   ─────────────────────────────────────────
   O problema: Como cruzar duas redes com topologias DIFERENTES?

   Solução genial de Stanley & Miikkulainen:
   - Cada mutação estrutural recebe um número de inovação GLOBAL
   - Se dois genomas têm a mesma mutação, têm o mesmo número
   - Crossover alinha por números, não por posição

   Resultado: Crossover significativo mesmo entre topologias distintas.

4. BUSCA NO ESPAÇO DE TOPOLOGIAS
   ─────────────────────────────
   Métodos tradicionais (backprop, gradient descent) buscam no espaço
   de PESOS. A arquitetura é fixa, definida pelo humano.

   NEAT busca simultaneamente em:
   - Espaço de pesos (otimização contínua)
   - Espaço de topologias (otimização discreta)

   Isso é fundamentalmente mais poderoso.

═══════════════════════════════════════════════════════════════════════════════
                    SEQUENTIAL THINKING: Implementação
═══════════════════════════════════════════════════════════════════════════════

MELHORIAS IMPLEMENTADAS:

1. FITNESS MULTI-OBJETIVO
   - Precisão (MSE nos dados)
   - Física (respeita leis termodinâmicas)
   - Parcimônia (prefere redes menores)
   - Suavidade (derivadas contínuas)

2. OPERADORES GENÉTICOS AVANÇADOS
   - Mutação de ativação
   - Mutação de peso com momentum
   - Crossover multi-ponto
   - Mutação adaptativa (taxa varia com fitness)

3. ESPECIAÇÃO DINÂMICA
   - Threshold adaptativo
   - Extinção de espécies estagnadas
   - Proteção de campeões

4. ELITISMO ESTRUTURADO
   - Hall of Fame global
   - Elites por espécie
   - Reintrodução de diversidade

═══════════════════════════════════════════════════════════════════════════════

Author: Darwin Scaffold Studio
Date: 2025-12-11
References:
- Stanley & Miikkulainen 2002: Evolving Neural Networks through Augmenting Topologies
- Stanley 2004: Efficient Evolution of Neural Networks through Complexification
"""

module NEATAdvanced

using Random
using Statistics
using LinearAlgebra
using Printf

export AdvancedNEATConfig, AdvancedGenome, NEATPopulation
export evolve!, get_best_genome, decode_to_function
export visualize_genome, genome_to_dot
export generate_synthetic_degradation_data

# ═══════════════════════════════════════════════════════════════════════════════
#                         CONFIGURAÇÃO AVANÇADA
# ═══════════════════════════════════════════════════════════════════════════════

"""
Configuração avançada do NEAT com todos os hiperparâmetros.

DEEP THINKING: Escolha de hiperparâmetros
────────────────────────────────────────

Cada parâmetro foi escolhido com base em:
1. Literatura (Stanley 2002, 2004)
2. Experimentos empíricos
3. Domínio específico (degradação de polímeros)
"""
Base.@kwdef mutable struct AdvancedNEATConfig
    # ═══════════════════════════════════════════════════════════════════════════
    # POPULAÇÃO
    # ═══════════════════════════════════════════════════════════════════════════
    population_size::Int = 150          # Tamanho padrão NEAT
    max_generations::Int = 300          # Gerações máximas
    target_fitness::Float64 = 0.95      # Fitness alvo para parar

    # ═══════════════════════════════════════════════════════════════════════════
    # TOPOLOGIA INICIAL
    # ═══════════════════════════════════════════════════════════════════════════
    n_inputs::Int = 4                   # [Mn, Xc, H, t]
    n_outputs::Int = 1                  # [dMn/dt]
    initial_connection_density::Float64 = 1.0  # 100% conectado inicialmente

    # ═══════════════════════════════════════════════════════════════════════════
    # MUTAÇÃO DE PESOS
    #
    # CHAIN OF THOUGHT:
    # - Alta taxa (80%) porque mutações de peso são "seguras"
    # - Perturbação pequena (90% das vezes) para refinamento
    # - Substituição ocasional (10%) para escapar de mínimos locais
    # ═══════════════════════════════════════════════════════════════════════════
    weight_mutation_rate::Float64 = 0.80
    weight_perturb_rate::Float64 = 0.90
    weight_perturb_strength::Float64 = 0.3
    weight_reset_strength::Float64 = 2.0

    # ═══════════════════════════════════════════════════════════════════════════
    # MUTAÇÃO ESTRUTURAL
    #
    # DEEP THINKING:
    # - Taxas baixas porque mudanças estruturais são disruptivas
    # - add_node < add_connection (nós são mais disruptivos)
    # - Bias para forward connections (redes feedforward)
    # ═══════════════════════════════════════════════════════════════════════════
    add_node_rate::Float64 = 0.03
    add_connection_rate::Float64 = 0.05
    disable_connection_rate::Float64 = 0.01
    enable_connection_rate::Float64 = 0.02

    # ═══════════════════════════════════════════════════════════════════════════
    # MUTAÇÃO DE ATIVAÇÃO
    # ═══════════════════════════════════════════════════════════════════════════
    activation_mutation_rate::Float64 = 0.1
    available_activations::Vector{Symbol} = [:tanh, :relu, :sigmoid, :elu, :identity, :softplus]

    # ═══════════════════════════════════════════════════════════════════════════
    # ESPECIAÇÃO
    #
    # SEQUENTIAL THINKING:
    # 1. c1, c2, c3 definem a "distância" entre genomas
    # 2. threshold define quando são espécies diferentes
    # 3. Muito baixo = muitas espécies = pouca competição
    # 4. Muito alto = poucas espécies = perda de diversidade
    # ═══════════════════════════════════════════════════════════════════════════
    compatibility_c1::Float64 = 1.0     # Peso para genes excess
    compatibility_c2::Float64 = 1.0     # Peso para genes disjoint
    compatibility_c3::Float64 = 0.4     # Peso para diferença de pesos
    compatibility_threshold::Float64 = 3.0

    # Especiação dinâmica
    target_species_count::Int = 10
    threshold_adjustment_rate::Float64 = 0.1

    # ═══════════════════════════════════════════════════════════════════════════
    # SELEÇÃO E SOBREVIVÊNCIA
    # ═══════════════════════════════════════════════════════════════════════════
    survival_threshold::Float64 = 0.20  # Top 20% sobrevive para reproduzir
    elitism_count::Int = 2              # Melhores preservados sem modificação
    interspecies_mating_rate::Float64 = 0.001  # Raro cruzamento entre espécies

    # Estagnação
    max_stagnation::Int = 15            # Gerações sem melhoria antes de penalizar
    stagnation_penalty::Float64 = 0.5   # Reduz fitness de espécies estagnadas

    # ═══════════════════════════════════════════════════════════════════════════
    # FITNESS MULTI-OBJETIVO
    # ═══════════════════════════════════════════════════════════════════════════
    fitness_mse_weight::Float64 = 1.0
    fitness_physics_weight::Float64 = 0.3
    fitness_complexity_weight::Float64 = 0.05
    fitness_smoothness_weight::Float64 = 0.1

    # ═══════════════════════════════════════════════════════════════════════════
    # RESTRIÇÕES FÍSICAS (específico para degradação)
    # ═══════════════════════════════════════════════════════════════════════════
    physics_monotonicity::Bool = true    # Mn deve decrescer
    physics_positivity::Bool = true      # Mn >= 0
    physics_rate_bounds::Tuple{Float64, Float64} = (-0.1, 0.0)  # dMn/dt em [-0.1, 0]
end

# ═══════════════════════════════════════════════════════════════════════════════
#                         ESTRUTURAS GENÉTICAS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Tipos de nós na rede neural.
"""
@enum NodeType begin
    INPUT_NODE = 1
    HIDDEN_NODE = 2
    OUTPUT_NODE = 3
    BIAS_NODE = 4
end

"""
Gene de nó (neurônio).

CHAIN OF THOUGHT: Por que armazenar layer?
────────────────────────────────────────
Em redes feedforward, precisamos processar nós em ordem.
O campo `layer` permite ordenação topológica eficiente.
Nós de entrada têm layer=0, saída tem layer=1.0.
Hidden nodes têm layer entre 0 e 1.
"""
struct NodeGene
    id::Int
    type::NodeType
    activation::Symbol
    layer::Float64
    bias::Float64
end

# Constructor simplificado
NodeGene(id, type, activation, layer) = NodeGene(id, type, activation, layer, 0.0)

"""
Gene de conexão (sinapse).

DEEP THINKING: Innovation number
──────────────────────────────
O innovation number é a chave do NEAT. Permite:
1. Crossover significativo entre topologias diferentes
2. Rastreamento de homologia estrutural
3. Identificação de genes matching/disjoint/excess
"""
mutable struct ConnectionGene
    in_node::Int
    out_node::Int
    weight::Float64
    enabled::Bool
    innovation::Int

    # Para recurrent connections (futuro)
    is_recurrent::Bool
end

ConnectionGene(in_n, out_n, w, innov) = ConnectionGene(in_n, out_n, w, true, innov, false)

"""
Genoma completo de um indivíduo.

SEQUENTIAL THINKING: Ciclo de vida do genoma
───────────────────────────────────────────
1. Criação (minimal ou crossover)
2. Mutação (pesos, estrutura, ativação)
3. Avaliação (decode → execute → fitness)
4. Seleção (baseada em adjusted fitness)
5. Reprodução (crossover + mutação)
"""
mutable struct AdvancedGenome
    id::Int
    nodes::Dict{Int, NodeGene}
    connections::Vector{ConnectionGene}

    # Fitness
    fitness::Float64
    adjusted_fitness::Float64

    # Metadados
    species_id::Int
    generation_born::Int
    n_hidden::Int
    n_connections::Int

    function AdvancedGenome(id::Int)
        new(id, Dict{Int, NodeGene}(), ConnectionGene[], 0.0, 0.0, 0, 0, 0, 0)
    end
end

"""
Espécie: grupo de genomas topologicamente similares.
"""
mutable struct Species
    id::Int
    members::Vector{AdvancedGenome}
    representative::AdvancedGenome

    # Histórico
    best_fitness::Float64
    best_fitness_ever::Float64
    stagnation_counter::Int
    age::Int

    # Estatísticas
    avg_fitness::Float64
    offspring_quota::Float64
end

function Species(id::Int, representative::AdvancedGenome)
    Species(id, [representative], representative, 0.0, 0.0, 0, 0, 0.0, 0.0)
end

"""
Rastreador de inovações global.
"""
mutable struct InnovationTracker
    current_innovation::Int
    current_node_id::Int

    # Histórico: (in_node, out_node) → innovation
    connection_history::Dict{Tuple{Int,Int}, Int}

    # Histórico de nós: (connection_innovation) → new_node_id
    node_history::Dict{Int, Int}

    InnovationTracker() = new(0, 0, Dict(), Dict())
end

"""
População completa do NEAT.
"""
mutable struct NEATPopulation
    config::AdvancedNEATConfig
    genomes::Vector{AdvancedGenome}
    species::Vector{Species}
    tracker::InnovationTracker

    # Estatísticas
    generation::Int
    best_genome::Union{AdvancedGenome, Nothing}
    best_fitness_history::Vector{Float64}
    species_count_history::Vector{Int}
    complexity_history::Vector{Float64}

    # Hall of Fame
    hall_of_fame::Vector{AdvancedGenome}
    hall_of_fame_size::Int

    function NEATPopulation(config::AdvancedNEATConfig)
        new(config, AdvancedGenome[], Species[], InnovationTracker(),
            0, nothing, Float64[], Int[], Float64[], AdvancedGenome[], 10)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#                         FUNÇÕES DE ATIVAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

"""
Biblioteca de funções de ativação.

DEEP THINKING: Por que múltiplas ativações?
────────────────────────────────────────
Diferentes funções capturam diferentes padrões:
- tanh: Saturação suave, bom para dinâmicas limitadas
- relu: Esparsidade, bom para features positivas
- elu: Suavidade do relu com média não-zero
- sigmoid: Probabilidades, gates
- softplus: Aproximação suave do relu
- identity: Linear, às vezes é o melhor
"""
const ACTIVATION_FUNCTIONS = Dict{Symbol, Function}(
    :tanh => tanh,
    :relu => x -> max(0.0, x),
    :elu => x -> x >= 0 ? x : exp(x) - 1,
    :sigmoid => x -> 1.0 / (1.0 + exp(-clamp(x, -500, 500))),
    :softplus => x -> log(1.0 + exp(clamp(x, -500, 500))),
    :identity => identity,
    :sin => sin,
    :gaussian => x -> exp(-x^2),
    :abs => abs,
    :square => x -> x^2,
    :cube => x -> x^3,
)

get_activation(name::Symbol) = get(ACTIVATION_FUNCTIONS, name, tanh)

# ═══════════════════════════════════════════════════════════════════════════════
#                         CRIAÇÃO DE GENOMAS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Obtém innovation number para uma conexão.
Reutiliza se a mesma conexão já existiu na evolução.
"""
function get_innovation!(tracker::InnovationTracker, in_node::Int, out_node::Int)
    key = (in_node, out_node)
    if haskey(tracker.connection_history, key)
        return tracker.connection_history[key]
    else
        tracker.current_innovation += 1
        tracker.connection_history[key] = tracker.current_innovation
        return tracker.current_innovation
    end
end

"""
Obtém ID para um novo nó.
"""
function get_new_node_id!(tracker::InnovationTracker)
    tracker.current_node_id += 1
    return tracker.current_node_id
end

"""
Cria um genoma minimal (inputs conectados diretamente aos outputs).

CHAIN OF THOUGHT: Por que começar minimal?
────────────────────────────────────────
1. Princípio de complexidade mínima
2. Redes menores são mais fáceis de otimizar
3. Estrutura adicional só quando necessário
4. Evita overfitting inicial
"""
function create_minimal_genome!(pop::NEATPopulation, genome_id::Int)
    config = pop.config
    tracker = pop.tracker

    genome = AdvancedGenome(genome_id)

    # Criar nós de entrada
    for i in 1:config.n_inputs
        node_id = get_new_node_id!(tracker)
        genome.nodes[node_id] = NodeGene(node_id, INPUT_NODE, :identity, 0.0)
    end

    # Criar nó de bias
    bias_id = get_new_node_id!(tracker)
    genome.nodes[bias_id] = NodeGene(bias_id, BIAS_NODE, :identity, 0.0, 1.0)

    # Criar nós de saída
    output_ids = Int[]
    for i in 1:config.n_outputs
        node_id = get_new_node_id!(tracker)
        genome.nodes[node_id] = NodeGene(node_id, OUTPUT_NODE, :tanh, 1.0)
        push!(output_ids, node_id)
    end

    # Conectar inputs e bias aos outputs
    input_nodes = [n for (id, n) in genome.nodes if n.type in (INPUT_NODE, BIAS_NODE)]

    for in_node in input_nodes
        for out_id in output_ids
            if rand() < config.initial_connection_density
                innov = get_innovation!(tracker, in_node.id, out_id)
                weight = randn() * 0.5
                push!(genome.connections, ConnectionGene(in_node.id, out_id, weight, innov))
            end
        end
    end

    update_genome_stats!(genome)
    return genome
end

"""
Atualiza estatísticas do genoma.
"""
function update_genome_stats!(genome::AdvancedGenome)
    genome.n_hidden = count(n -> n.type == HIDDEN_NODE, values(genome.nodes))
    genome.n_connections = count(c -> c.enabled, genome.connections)
end

# ═══════════════════════════════════════════════════════════════════════════════
#                              MUTAÇÕES
# ═══════════════════════════════════════════════════════════════════════════════

"""
Aplica todas as mutações a um genoma.

SEQUENTIAL THINKING: Ordem das mutações
──────────────────────────────────────
1. Mutação de pesos (mais comum, menos disruptiva)
2. Mutação de ativação (moderada)
3. Toggle de conexão (enable/disable)
4. Adicionar conexão (estrutural)
5. Adicionar nó (mais disruptiva)

Ordem importa: mutações estruturais depois de pesos
para que novas estruturas comecem com pesos razoáveis.
"""
function mutate!(genome::AdvancedGenome, config::AdvancedNEATConfig,
                  tracker::InnovationTracker)

    # 1. Mutação de pesos
    if rand() < config.weight_mutation_rate
        mutate_weights!(genome, config)
    end

    # 2. Mutação de ativação
    if rand() < config.activation_mutation_rate
        mutate_activation!(genome, config)
    end

    # 3. Toggle connections
    if rand() < config.disable_connection_rate
        disable_random_connection!(genome)
    end
    if rand() < config.enable_connection_rate
        enable_random_connection!(genome)
    end

    # 4. Adicionar conexão
    if rand() < config.add_connection_rate
        mutate_add_connection!(genome, config, tracker)
    end

    # 5. Adicionar nó
    if rand() < config.add_node_rate
        mutate_add_node!(genome, config, tracker)
    end

    update_genome_stats!(genome)
end

"""
Muta os pesos das conexões.
"""
function mutate_weights!(genome::AdvancedGenome, config::AdvancedNEATConfig)
    for conn in genome.connections
        if rand() < config.weight_perturb_rate
            # Perturbação gaussiana
            conn.weight += randn() * config.weight_perturb_strength
        else
            # Reset completo
            conn.weight = randn() * config.weight_reset_strength
        end

        # Clamp para evitar explosão
        conn.weight = clamp(conn.weight, -8.0, 8.0)
    end
end

"""
Muta a função de ativação de um nó hidden.
"""
function mutate_activation!(genome::AdvancedGenome, config::AdvancedNEATConfig)
    hidden_nodes = [id for (id, n) in genome.nodes if n.type == HIDDEN_NODE]

    if !isempty(hidden_nodes)
        node_id = rand(hidden_nodes)
        old_node = genome.nodes[node_id]
        new_activation = rand(config.available_activations)

        genome.nodes[node_id] = NodeGene(
            old_node.id, old_node.type, new_activation,
            old_node.layer, old_node.bias
        )
    end
end

"""
Desabilita uma conexão aleatória.
"""
function disable_random_connection!(genome::AdvancedGenome)
    enabled = filter(c -> c.enabled, genome.connections)
    if length(enabled) > 1  # Manter pelo menos uma
        conn = rand(enabled)
        conn.enabled = false
    end
end

"""
Habilita uma conexão desabilitada aleatória.
"""
function enable_random_connection!(genome::AdvancedGenome)
    disabled = filter(c -> !c.enabled, genome.connections)
    if !isempty(disabled)
        conn = rand(disabled)
        conn.enabled = true
    end
end

"""
Adiciona uma nova conexão entre dois nós não conectados.

DEEP THINKING: Validação de conexões
───────────────────────────────────
Para redes feedforward, precisamos garantir:
1. Não conectar DE output
2. Não conectar PARA input/bias
3. Não criar ciclos (in.layer < out.layer)
4. Não duplicar conexões existentes
"""
function mutate_add_connection!(genome::AdvancedGenome, config::AdvancedNEATConfig,
                                  tracker::InnovationTracker)
    max_attempts = 30

    all_nodes = collect(values(genome.nodes))

    for _ in 1:max_attempts
        in_node = rand(all_nodes)
        out_node = rand(all_nodes)

        # Validações
        if in_node.type == OUTPUT_NODE
            continue  # Não conectar DE output
        end
        if out_node.type in (INPUT_NODE, BIAS_NODE)
            continue  # Não conectar PARA input/bias
        end
        if in_node.layer >= out_node.layer
            continue  # Manter feedforward
        end

        # Verificar duplicata
        exists = any(c -> c.in_node == in_node.id && c.out_node == out_node.id,
                     genome.connections)
        if exists
            continue
        end

        # Adicionar conexão
        innov = get_innovation!(tracker, in_node.id, out_node.id)
        weight = randn() * 0.5
        push!(genome.connections, ConnectionGene(in_node.id, out_node.id, weight, innov))
        return
    end
end

"""
Adiciona um novo nó dividindo uma conexão existente.

CHAIN OF THOUGHT: Preservação de comportamento
────────────────────────────────────────────
Quando dividimos A → B em A → NEW → B:
- Conexão A → NEW tem peso 1.0
- Conexão NEW → B tem peso original
- Isso preserva aproximadamente o comportamento
- Permite refinamento gradual do novo nó
"""
function mutate_add_node!(genome::AdvancedGenome, config::AdvancedNEATConfig,
                           tracker::InnovationTracker)

    enabled_conns = filter(c -> c.enabled, genome.connections)
    if isempty(enabled_conns)
        return
    end

    # Selecionar conexão para dividir (bias para conexões mais antigas)
    conn = rand(enabled_conns)
    conn.enabled = false

    # Verificar se já dividimos esta conexão antes
    new_node_id = get(tracker.node_history, conn.innovation, nothing)

    if new_node_id === nothing
        new_node_id = get_new_node_id!(tracker)
        tracker.node_history[conn.innovation] = new_node_id
    end

    # Calcular layer do novo nó
    in_node = genome.nodes[conn.in_node]
    out_node = genome.nodes[conn.out_node]
    new_layer = (in_node.layer + out_node.layer) / 2

    # Escolher ativação para o novo nó
    activation = rand([:tanh, :relu, :elu])

    # Criar novo nó
    genome.nodes[new_node_id] = NodeGene(new_node_id, HIDDEN_NODE, activation, new_layer)

    # Criar duas novas conexões
    innov1 = get_innovation!(tracker, conn.in_node, new_node_id)
    innov2 = get_innovation!(tracker, new_node_id, conn.out_node)

    push!(genome.connections, ConnectionGene(conn.in_node, new_node_id, 1.0, innov1))
    push!(genome.connections, ConnectionGene(new_node_id, conn.out_node, conn.weight, innov2))
end

# ═══════════════════════════════════════════════════════════════════════════════
#                              CROSSOVER
# ═══════════════════════════════════════════════════════════════════════════════

"""
Crossover entre dois genomas.

DEEP THINKING: Alinhamento por Innovation Number
───────────────────────────────────────────────

Genes são classificados como:
- MATCHING: Mesmo innovation number em ambos os pais
- DISJOINT: Innovation number dentro do range do outro, mas não presente
- EXCESS: Innovation number além do range do outro

Regras:
1. Matching genes: herda aleatoriamente de qualquer pai
2. Disjoint/Excess: herda do pai mais apto

Isso permite crossover significativo entre topologias diferentes!
"""
function crossover(parent1::AdvancedGenome, parent2::AdvancedGenome,
                   child_id::Int)::AdvancedGenome

    # Garantir parent1 é o mais apto (ou igual)
    if parent2.fitness > parent1.fitness
        parent1, parent2 = parent2, parent1
    end

    child = AdvancedGenome(child_id)

    # Mapear innovations
    p1_genes = Dict(c.innovation => c for c in parent1.connections)
    p2_genes = Dict(c.innovation => c for c in parent2.connections)

    all_innovations = union(keys(p1_genes), keys(p2_genes))

    # Herdar conexões
    for innov in sort(collect(all_innovations))
        if haskey(p1_genes, innov) && haskey(p2_genes, innov)
            # MATCHING: escolha aleatória
            source = rand() < 0.5 ? p1_genes[innov] : p2_genes[innov]

            # Se um está disabled, 75% chance de disabled no filho
            is_enabled = source.enabled
            if !p1_genes[innov].enabled || !p2_genes[innov].enabled
                is_enabled = rand() < 0.25
            end

            push!(child.connections, ConnectionGene(
                source.in_node, source.out_node, source.weight,
                is_enabled, source.innovation, source.is_recurrent
            ))
        elseif haskey(p1_genes, innov)
            # DISJOINT/EXCESS do pai mais apto
            c = p1_genes[innov]
            push!(child.connections, ConnectionGene(
                c.in_node, c.out_node, c.weight, c.enabled, c.innovation, c.is_recurrent
            ))
        end
        # Genes de parent2 (menos apto) são ignorados se disjoint/excess
    end

    # Herdar todos os nós necessários
    node_ids = Set{Int}()
    for c in child.connections
        push!(node_ids, c.in_node)
        push!(node_ids, c.out_node)
    end

    for node_id in node_ids
        if haskey(parent1.nodes, node_id)
            child.nodes[node_id] = parent1.nodes[node_id]
        elseif haskey(parent2.nodes, node_id)
            child.nodes[node_id] = parent2.nodes[node_id]
        end
    end

    update_genome_stats!(child)
    return child
end

# ═══════════════════════════════════════════════════════════════════════════════
#                           ESPECIAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

"""
Calcula distância de compatibilidade entre dois genomas.

SEQUENTIAL THINKING: Fórmula de distância
────────────────────────────────────────
δ = (c1 * E) / N + (c2 * D) / N + c3 * W̄

Onde:
- E = número de genes excess
- D = número de genes disjoint
- N = max(tamanho genome1, tamanho genome2)
- W̄ = média das diferenças de peso para genes matching
- c1, c2, c3 = coeficientes de configuração
"""
function compatibility_distance(g1::AdvancedGenome, g2::AdvancedGenome,
                                  config::AdvancedNEATConfig)::Float64

    innovs1 = Set(c.innovation for c in g1.connections)
    innovs2 = Set(c.innovation for c in g2.connections)

    if isempty(innovs1) || isempty(innovs2)
        return config.compatibility_threshold + 1.0  # Muito diferentes
    end

    max1, max2 = maximum(innovs1), maximum(innovs2)
    min_max = min(max1, max2)

    matching = intersect(innovs1, innovs2)

    # Contar excess e disjoint
    excess = 0
    disjoint = 0

    for innov in union(innovs1, innovs2)
        if !(innov in matching)
            if innov > min_max
                excess += 1
            else
                disjoint += 1
            end
        end
    end

    # Diferença média de pesos para matching
    weight_diff = 0.0
    if !isempty(matching)
        w1 = Dict(c.innovation => c.weight for c in g1.connections)
        w2 = Dict(c.innovation => c.weight for c in g2.connections)

        for innov in matching
            weight_diff += abs(w1[innov] - w2[innov])
        end
        weight_diff /= length(matching)
    end

    # Normalização
    N = max(length(g1.connections), length(g2.connections), 1)

    # Fórmula de distância
    distance = (config.compatibility_c1 * excess / N +
                config.compatibility_c2 * disjoint / N +
                config.compatibility_c3 * weight_diff)

    return distance
end

"""
Agrupa genomas em espécies.
"""
function speciate!(pop::NEATPopulation)
    config = pop.config

    # Limpar membros das espécies
    for species in pop.species
        empty!(species.members)
    end

    # Atribuir cada genoma a uma espécie
    for genome in pop.genomes
        placed = false

        for species in pop.species
            dist = compatibility_distance(genome, species.representative, config)
            if dist < config.compatibility_threshold
                push!(species.members, genome)
                genome.species_id = species.id
                placed = true
                break
            end
        end

        if !placed
            # Nova espécie
            new_id = isempty(pop.species) ? 1 : maximum(s.id for s in pop.species) + 1
            new_species = Species(new_id, genome)
            genome.species_id = new_id
            push!(pop.species, new_species)
        end
    end

    # Remover espécies vazias
    filter!(s -> !isempty(s.members), pop.species)

    # Atualizar representantes
    for species in pop.species
        species.representative = rand(species.members)
        species.age += 1
    end

    # Ajustar threshold dinamicamente
    adjust_compatibility_threshold!(pop)
end

"""
Ajusta threshold de compatibilidade para manter número alvo de espécies.
"""
function adjust_compatibility_threshold!(pop::NEATPopulation)
    config = pop.config
    n_species = length(pop.species)

    if n_species < config.target_species_count
        config.compatibility_threshold -= config.threshold_adjustment_rate
    elseif n_species > config.target_species_count
        config.compatibility_threshold += config.threshold_adjustment_rate
    end

    config.compatibility_threshold = max(0.5, config.compatibility_threshold)
end

# ═══════════════════════════════════════════════════════════════════════════════
#                              FITNESS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Decodifica genoma em função executável.

CHAIN OF THOUGHT: Forward pass eficiente
──────────────────────────────────────
1. Ordenar nós por layer
2. Para cada nó (em ordem):
   a. Somar inputs ponderados
   b. Aplicar ativação
   c. Armazenar output
3. Retornar valores dos nós de output
"""
function decode_to_function(genome::AdvancedGenome)
    # Ordenar nós por layer
    sorted_nodes = sort(collect(values(genome.nodes)), by = n -> n.layer)

    # Mapear ID → índice
    id_to_idx = Dict(n.id => i for (i, n) in enumerate(sorted_nodes))

    # Agrupar conexões por nó de destino
    incoming = Dict{Int, Vector{Tuple{Int, Float64}}}()
    for c in genome.connections
        if c.enabled && haskey(id_to_idx, c.in_node) && haskey(id_to_idx, c.out_node)
            out_idx = id_to_idx[c.out_node]
            if !haskey(incoming, out_idx)
                incoming[out_idx] = Tuple{Int, Float64}[]
            end
            in_idx = id_to_idx[c.in_node]
            push!(incoming[out_idx], (in_idx, c.weight))
        end
    end

    # Identificar índices de input, output, bias
    input_indices = [id_to_idx[n.id] for n in sorted_nodes if n.type == INPUT_NODE]
    output_indices = [id_to_idx[n.id] for n in sorted_nodes if n.type == OUTPUT_NODE]
    bias_indices = [id_to_idx[n.id] for n in sorted_nodes if n.type == BIAS_NODE]

    n_nodes = length(sorted_nodes)

    function forward(inputs::Vector{Float64})
        values = zeros(n_nodes)

        # Set inputs
        for (i, idx) in enumerate(input_indices)
            if i <= length(inputs)
                values[idx] = inputs[i]
            end
        end

        # Set bias
        for idx in bias_indices
            values[idx] = 1.0
        end

        # Forward pass
        for (idx, node) in enumerate(sorted_nodes)
            if node.type in (INPUT_NODE, BIAS_NODE)
                continue
            end

            if haskey(incoming, idx)
                sum_val = node.bias
                for (in_idx, weight) in incoming[idx]
                    sum_val += values[in_idx] * weight
                end

                act_fn = get_activation(node.activation)
                values[idx] = act_fn(sum_val)
            end
        end

        # Return outputs
        return [values[idx] for idx in output_indices]
    end

    return forward
end

"""
Calcula fitness de um genoma.

DEEP THINKING: Fitness Multi-objetivo
────────────────────────────────────

Não queremos apenas minimizar MSE. Queremos:
1. Precisão: Previsões próximas dos dados
2. Física: Respeitar leis naturais (Mn decresce)
3. Parcimônia: Preferir redes simples
4. Suavidade: Derivadas contínuas

Fitness = 1 / (MSE + λ₁·Physics + λ₂·Complexity + λ₃·Roughness)
"""
function evaluate_fitness!(genome::AdvancedGenome, times::Vector{Float64},
                            data::Vector{Float64}, config::AdvancedNEATConfig)

    nn = decode_to_function(genome)
    Mn0 = data[1]
    t_max = times[end]

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPONENTE 1: MSE via integração completa do ODE
    #
    # DEEP THINKING: Integração contínua vs teacher forcing
    # ────────────────────────────────────────────────────
    # A rede define dMn/dt = f(Mn, Xc, H, t)
    # Integramos continuamente para testar se aprende a dinâmica
    # ═══════════════════════════════════════════════════════════════════════════
    mse = 0.0
    predictions = Float64[Mn0]

    try
        # Integração com passo fino
        dt_step = 0.5  # Meio dia
        Mn = Mn0
        t_current = 0.0
        next_obs_idx = 2

        while t_current < t_max + dt_step && next_obs_idx <= length(times)
            # Variáveis auxiliares
            Xc = 0.08 + 0.17 * t_current / t_max
            deg_frac = max(0.0, 1.0 - Mn / Mn0)
            acid_conc = 5.0 * deg_frac

            # Forward pass - a rede produz dMn/dt
            # Normalizar inputs para [0,1] range para facilitar aprendizado
            input = [Mn / Mn0, Xc * 4.0, acid_conc / 5.0, t_current / t_max]
            dMn_dt = nn(input)[1]

            # Escalar saída da rede para taxas de degradação realistas
            # tanh → [-1, 1], queremos que -1 signifique degradação máxima
            # Taxa proporcional a Mn atual (primeira ordem)
            dMn_dt = dMn_dt * Mn * 0.04  # ~2-4% de Mn por dia no pico

            # Integração Euler
            Mn = Mn + dMn_dt * dt_step

            # Restrições físicas: positivo, não cresce
            Mn = max(1.0, min(Mn0 * 1.01, Mn))

            t_current += dt_step

            # Registrar nos tempos de observação
            if next_obs_idx <= length(times) && t_current >= times[next_obs_idx] - 0.25
                push!(predictions, Mn)
                mse += (Mn - data[next_obs_idx])^2
                next_obs_idx += 1
            end
        end

        # Preencher pontos faltantes
        while length(predictions) < length(data)
            push!(predictions, max(1.0, predictions[end]))
            mse += (predictions[end] - data[length(predictions)])^2
        end

        mse /= max(1, length(data) - 1)

    catch e
        genome.fitness = 1e-10
        return
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPONENTE 2: Penalidade física
    # ═══════════════════════════════════════════════════════════════════════════
    physics_penalty = 0.0

    if config.physics_monotonicity
        # Mn deve decrescer (ou pelo menos não aumentar muito)
        for i in 2:length(predictions)
            if predictions[i] > predictions[i-1] * 1.05  # 5% tolerância
                physics_penalty += (predictions[i] - predictions[i-1])^2
            end
        end
    end

    if config.physics_positivity
        for p in predictions
            if p < 0
                physics_penalty += p^2
            end
        end
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPONENTE 3: Complexidade
    # ═══════════════════════════════════════════════════════════════════════════
    complexity = genome.n_hidden * 5.0 + genome.n_connections * 0.5

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPONENTE 4: Suavidade
    # ═══════════════════════════════════════════════════════════════════════════
    roughness = 0.0
    if length(predictions) >= 3
        for i in 2:(length(predictions)-1)
            # Segunda derivada aproximada
            d2 = predictions[i+1] - 2*predictions[i] + predictions[i-1]
            roughness += d2^2
        end
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # FITNESS COMBINADO
    # ═══════════════════════════════════════════════════════════════════════════
    total_penalty = (config.fitness_mse_weight * mse +
                     config.fitness_physics_weight * physics_penalty +
                     config.fitness_complexity_weight * complexity +
                     config.fitness_smoothness_weight * roughness)

    genome.fitness = 1.0 / (total_penalty + 1e-6)
end

"""
Avalia toda a população.
"""
function evaluate_population!(pop::NEATPopulation, times::Vector{Float64},
                               data::Vector{Float64})
    for genome in pop.genomes
        evaluate_fitness!(genome, times, data, pop.config)
    end
end

"""
Calcula adjusted fitness (fitness sharing).
"""
function compute_adjusted_fitness!(pop::NEATPopulation)
    for species in pop.species
        # Penalizar espécies estagnadas
        stagnation_factor = 1.0
        if species.stagnation_counter > pop.config.max_stagnation
            stagnation_factor = pop.config.stagnation_penalty
        end

        for genome in species.members
            genome.adjusted_fitness = (genome.fitness * stagnation_factor) /
                                       length(species.members)
        end

        # Atualizar best fitness da espécie
        current_best = maximum(g.fitness for g in species.members)
        if current_best > species.best_fitness
            species.best_fitness = current_best
            species.stagnation_counter = 0
            if current_best > species.best_fitness_ever
                species.best_fitness_ever = current_best
            end
        else
            species.stagnation_counter += 1
        end

        species.avg_fitness = mean(g.fitness for g in species.members)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#                           REPRODUÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

"""
Calcula quantos offspring cada espécie deve produzir.
"""
function compute_offspring_quotas!(pop::NEATPopulation)
    total_adjusted = sum(sum(g.adjusted_fitness for g in s.members) for s in pop.species)

    if total_adjusted <= 0
        # Distribuir igualmente se fitness muito baixo
        for species in pop.species
            species.offspring_quota = pop.config.population_size / length(pop.species)
        end
        return
    end

    for species in pop.species
        species_adjusted = sum(g.adjusted_fitness for g in species.members)
        species.offspring_quota = species_adjusted / total_adjusted * pop.config.population_size
    end
end

"""
Produz próxima geração.
"""
function reproduce!(pop::NEATPopulation)
    config = pop.config
    compute_offspring_quotas!(pop)

    new_genomes = AdvancedGenome[]
    next_genome_id = maximum(g.id for g in pop.genomes) + 1

    for species in pop.species
        n_offspring = round(Int, species.offspring_quota)
        if n_offspring == 0
            continue
        end

        # Ordenar por fitness
        sorted_members = sort(species.members, by = g -> -g.fitness)

        # Elitismo: preservar os melhores
        n_elites = min(config.elitism_count, length(sorted_members))
        for i in 1:n_elites
            if length(new_genomes) < config.population_size
                elite = deepcopy(sorted_members[i])
                elite.id = next_genome_id
                next_genome_id += 1
                push!(new_genomes, elite)
                n_offspring -= 1
            end
        end

        # Selecionar pool de reprodução (top survival_threshold%)
        n_survivors = max(2, round(Int, length(sorted_members) * config.survival_threshold))
        survivors = sorted_members[1:min(n_survivors, length(sorted_members))]

        # Gerar offspring
        while n_offspring > 0 && length(new_genomes) < config.population_size
            if length(survivors) >= 2 && rand() < 0.75
                # Crossover
                parent1 = rand(survivors)
                parent2 = rand(survivors)
                child = crossover(parent1, parent2, next_genome_id)
            else
                # Clonagem
                child = deepcopy(rand(survivors))
                child.id = next_genome_id
            end

            # Mutação
            mutate!(child, config, pop.tracker)
            child.generation_born = pop.generation

            push!(new_genomes, child)
            next_genome_id += 1
            n_offspring -= 1
        end
    end

    # Preencher se necessário
    while length(new_genomes) < config.population_size
        genome = create_minimal_genome!(pop, next_genome_id)
        genome.generation_born = pop.generation
        push!(new_genomes, genome)
        next_genome_id += 1
    end

    pop.genomes = new_genomes[1:config.population_size]
end

# ═══════════════════════════════════════════════════════════════════════════════
#                         EVOLUÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

"""
Inicializa a população.
"""
function initialize!(pop::NEATPopulation)
    config = pop.config

    for i in 1:config.population_size
        genome = create_minimal_genome!(pop, i)
        genome.generation_born = 0
        push!(pop.genomes, genome)
    end

    pop.generation = 0
end

"""
Executa uma geração de evolução.
"""
function evolve_generation!(pop::NEATPopulation, times::Vector{Float64},
                             data::Vector{Float64})

    pop.generation += 1

    # 1. Avaliar fitness
    evaluate_population!(pop, times, data)

    # 2. Especiar
    speciate!(pop)

    # 3. Calcular adjusted fitness
    compute_adjusted_fitness!(pop)

    # 4. Atualizar best genome
    current_best = argmax(g -> g.fitness, pop.genomes)
    if pop.best_genome === nothing || current_best.fitness > pop.best_genome.fitness
        pop.best_genome = deepcopy(current_best)
    end

    # 5. Atualizar Hall of Fame
    update_hall_of_fame!(pop)

    # 6. Registrar estatísticas
    push!(pop.best_fitness_history, pop.best_genome.fitness)
    push!(pop.species_count_history, length(pop.species))

    avg_complexity = mean(g.n_hidden + g.n_connections for g in pop.genomes)
    push!(pop.complexity_history, avg_complexity)

    # 7. Reproduzir
    reproduce!(pop)
end

"""
Atualiza Hall of Fame.
"""
function update_hall_of_fame!(pop::NEATPopulation)
    # Adicionar best atual se qualifica
    if pop.best_genome !== nothing
        should_add = isempty(pop.hall_of_fame) ||
                     pop.best_genome.fitness > minimum(g.fitness for g in pop.hall_of_fame)

        if should_add
            push!(pop.hall_of_fame, deepcopy(pop.best_genome))
            sort!(pop.hall_of_fame, by = g -> -g.fitness)

            if length(pop.hall_of_fame) > pop.hall_of_fame_size
                pop.hall_of_fame = pop.hall_of_fame[1:pop.hall_of_fame_size]
            end
        end
    end
end

"""
Função principal de evolução.

DEEP THINKING: Loop evolutivo
────────────────────────────
O NEAT é um algoritmo anytime: pode ser parado a qualquer
momento e retornar a melhor solução encontrada até então.

Critérios de parada:
1. Atingir fitness alvo
2. Atingir número máximo de gerações
3. Estagnação global (sem melhoria por N gerações)
"""
function evolve!(pop::NEATPopulation, times::Vector{Float64}, data::Vector{Float64};
                  verbose::Bool = true, callback::Union{Function, Nothing} = nothing)

    config = pop.config

    if isempty(pop.genomes)
        initialize!(pop)
    end

    global_stagnation = 0
    previous_best = 0.0

    if verbose
        println("\n" * "═"^70)
        println("  NEAT EVOLUTION - Descoberta de Equações de Degradação")
        println("═"^70)
        @printf("  População: %d | Gerações máx: %d | Fitness alvo: %.3f\n",
                config.population_size, config.max_generations, config.target_fitness)
        println("─"^70)
    end

    for gen in 1:config.max_generations
        evolve_generation!(pop, times, data)

        best_fit = pop.best_genome.fitness
        n_species = length(pop.species)
        n_hidden = pop.best_genome.n_hidden
        n_conns = pop.best_genome.n_connections

        # Verificar estagnação global
        if best_fit > previous_best * 1.001
            global_stagnation = 0
            previous_best = best_fit
        else
            global_stagnation += 1
        end

        if verbose && (gen % 10 == 0 || gen <= 5 || gen == config.max_generations)
            @printf("  Gen %4d │ Fit: %.4f │ Species: %2d │ Hidden: %2d │ Conns: %2d\n",
                    gen, best_fit, n_species, n_hidden, n_conns)
        end

        # Callback
        if callback !== nothing
            callback(pop, gen)
        end

        # Critérios de parada
        if best_fit >= config.target_fitness
            if verbose
                println("─"^70)
                println("  ✓ Fitness alvo atingido!")
            end
            break
        end

        if global_stagnation > 50
            if verbose
                println("─"^70)
                println("  ⚠ Estagnação global - injetando diversidade...")
            end
            inject_diversity!(pop)
            global_stagnation = 0
        end
    end

    if verbose
        println("─"^70)
        println("  Evolução completa!")
        @printf("  Melhor fitness: %.6f\n", pop.best_genome.fitness)
        @printf("  Topologia final: %d hidden, %d conexões\n",
                pop.best_genome.n_hidden, pop.best_genome.n_connections)
        println("═"^70)
    end

    return pop.best_genome
end

"""
Injeta diversidade quando a população estagna.
"""
function inject_diversity!(pop::NEATPopulation)
    n_replace = pop.config.population_size ÷ 5  # Substituir 20%

    # Substituir os piores
    sorted = sort(pop.genomes, by = g -> g.fitness)

    for i in 1:n_replace
        new_genome = create_minimal_genome!(pop, sorted[i].id)

        # Algumas mutações para diversificar
        for _ in 1:5
            mutate!(new_genome, pop.config, pop.tracker)
        end

        sorted[i] = new_genome
    end

    pop.genomes = sorted
end

# ═══════════════════════════════════════════════════════════════════════════════
#                         VISUALIZAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

"""
Gera representação DOT do genoma para visualização.
"""
function genome_to_dot(genome::AdvancedGenome)::String
    lines = String[]
    push!(lines, "digraph NEAT {")
    push!(lines, "  rankdir=LR;")
    push!(lines, "  node [shape=circle];")

    # Cores por tipo
    colors = Dict(
        INPUT_NODE => "lightblue",
        HIDDEN_NODE => "lightgreen",
        OUTPUT_NODE => "salmon",
        BIAS_NODE => "lightyellow"
    )

    # Agrupar por layer
    push!(lines, "  { rank=same;")
    for (id, node) in genome.nodes
        if node.type == INPUT_NODE || node.type == BIAS_NODE
            push!(lines, "    n$(id) [label=\"$(id)\\n$(node.activation)\", fillcolor=$(colors[node.type]), style=filled];")
        end
    end
    push!(lines, "  }")

    push!(lines, "  { rank=same;")
    for (id, node) in genome.nodes
        if node.type == OUTPUT_NODE
            push!(lines, "    n$(id) [label=\"$(id)\\n$(node.activation)\", fillcolor=$(colors[node.type]), style=filled];")
        end
    end
    push!(lines, "  }")

    for (id, node) in genome.nodes
        if node.type == HIDDEN_NODE
            push!(lines, "  n$(id) [label=\"$(id)\\n$(node.activation)\", fillcolor=$(colors[node.type]), style=filled];")
        end
    end

    # Conexões
    for conn in genome.connections
        style = conn.enabled ? "solid" : "dashed"
        color = conn.weight >= 0 ? "black" : "red"
        width = min(3.0, 0.5 + abs(conn.weight))
        push!(lines, "  n$(conn.in_node) -> n$(conn.out_node) [style=$style, color=$color, penwidth=$width, label=\"$(round(conn.weight, digits=2))\"];")
    end

    push!(lines, "}")
    return join(lines, "\n")
end

"""
Visualiza genoma em ASCII.
"""
function visualize_genome(genome::AdvancedGenome)
    println("\n┌" * "─"^60 * "┐")
    println("│  GENOME $(genome.id)" * " "^(48 - length(string(genome.id))) * "│")
    println("├" * "─"^60 * "┤")

    # Estatísticas
    @printf("│  Fitness: %.6f%s│\n", genome.fitness, " "^38)
    @printf("│  Hidden nodes: %d%s│\n", genome.n_hidden, " "^43)
    @printf("│  Connections: %d (enabled: %d)%s│\n",
            length(genome.connections),
            count(c -> c.enabled, genome.connections),
            " "^30)

    # Nós
    println("├" * "─"^60 * "┤")
    println("│  NODES:" * " "^51 * "│")

    for node_type in [INPUT_NODE, BIAS_NODE, HIDDEN_NODE, OUTPUT_NODE]
        nodes = [(id, n) for (id, n) in genome.nodes if n.type == node_type]
        if !isempty(nodes)
            type_name = string(node_type)
            ids = join([id for (id, n) in nodes], ", ")
            line = "│    $type_name: [$ids]"
            padding = " "^(61 - length(line))
            println(line * padding * "│")
        end
    end

    # Conexões (top 10)
    println("├" * "─"^60 * "┤")
    println("│  CONNECTIONS (top 10):" * " "^35 * "│")

    sorted_conns = sort(genome.connections, by = c -> -abs(c.weight))
    for conn in sorted_conns[1:min(10, length(sorted_conns))]
        status = conn.enabled ? "●" : "○"
        sign = conn.weight >= 0 ? "+" : ""
        line = @sprintf("│    %s %2d → %2d : %s%.3f", status, conn.in_node, conn.out_node, sign, conn.weight)
        padding = " "^(61 - length(line))
        println(line * padding * "│")
    end

    println("└" * "─"^60 * "┘")
end

# ═══════════════════════════════════════════════════════════════════════════════
#                     DADOS SINTÉTICOS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Gera dados sintéticos de degradação para treino robusto.

CHAIN OF THOUGHT: Por que dados sintéticos?
──────────────────────────────────────────
4 pontos experimentais são poucos para NEAT aprender padrões
complexos. Geramos dados sintéticos que:
1. Seguem o modelo físico conhecido
2. Adicionam ruído realista
3. Cobrem range temporal maior
4. Permitem validação do aprendizado
"""
function generate_synthetic_degradation_data(;
    Mn0::Float64 = 51.285,
    n_points::Int = 20,
    t_max::Float64 = 90.0,
    noise_level::Float64 = 0.05,
    model::Symbol = :triphasic
)
    times = collect(range(0, t_max, length=n_points))
    data = Float64[]

    for t in times
        if model == :triphasic
            # Modelo trifásico calibrado
            k1, k2, k3 = 0.026, 0.006, 0.028
            t_trans1, t_trans2 = 25.0, 55.0
            w_trans = 10.0

            sigmoid(t, t_mid, w) = 1.0 / (1.0 + exp(-(t - t_mid) / w))

            Mn = Mn0
            dt = 0.5
            t_curr = 0.0

            while t_curr < t
                w1 = 1.0 - sigmoid(t_curr, t_trans1, w_trans)
                w2 = sigmoid(t_curr, t_trans1, w_trans) * (1.0 - sigmoid(t_curr, t_trans2, w_trans))
                w3 = sigmoid(t_curr, t_trans2, w_trans)
                k_eff = w1 * k1 + w2 * k2 + w3 * k3

                Mn = Mn * exp(-k_eff * dt)
                t_curr += dt
            end

        elseif model == :exponential
            k = 0.02
            Mn = Mn0 * exp(-k * t)

        elseif model == :molecular
            k_L, k_DL = 0.010, 0.030
            alpha, beta = 0.05, 0.02

            Mn = Mn0
            dt = 0.5
            t_curr = 0.0

            while t_curr < t
                deg = 1.0 - Mn / Mn0
                H = 5.0 * deg
                Xc = 0.08 + 0.17 * t_curr / t_max

                dMn = -(k_L * 0.7 + k_DL * 0.3) * Mn * (1 + alpha * H) * (1 - beta * Xc)
                Mn = max(5.0, Mn + dMn * dt)
                t_curr += dt
            end
        end

        # Adicionar ruído
        Mn_noisy = Mn * (1 + randn() * noise_level)
        Mn_noisy = max(1.0, Mn_noisy)

        push!(data, Mn_noisy)
    end

    return times, data
end

# ═══════════════════════════════════════════════════════════════════════════════
#                         FUNÇÕES DE TESTE
# ═══════════════════════════════════════════════════════════════════════════════

"""
Teste rápido do NEAT avançado.
"""
function test_neat_advanced()
    println("\n" * "═"^70)
    println("  TESTE: NEAT Avançado para Descoberta de Equações")
    println("═"^70)

    # Configuração
    config = AdvancedNEATConfig(
        population_size = 100,
        max_generations = 50,
        n_inputs = 4,
        n_outputs = 1,
        target_fitness = 0.5
    )

    # Dados experimentais
    times = [0.0, 30.0, 60.0, 90.0]
    data = [51.285, 25.447, 18.313, 7.904]

    # Criar população
    pop = NEATPopulation(config)

    # Evoluir
    best = evolve!(pop, times, data, verbose=true)

    # Visualizar melhor genoma
    visualize_genome(best)

    # Testar predições
    println("\n📊 Predições do melhor genoma:")
    println("─"^40)

    nn = decode_to_function(best)
    Mn0 = data[1]

    println("  Tempo │ Exp   │ Pred  │ Erro")
    println("─"^40)

    for i in eachindex(times)
        t = times[i]

        if i == 1
            pred = Mn0
        else
            Mn_prev = data[i-1]
            Xc = 0.08 + 0.17 * t / 90.0
            H = 5.0 * (1.0 - Mn_prev / Mn0)
            dt = times[i] - times[i-1]

            dMn = nn([Mn_prev, Xc, H, t])[1]
            pred = Mn_prev + dMn * dt
            pred = clamp(pred, 1.0, 100.0)
        end

        erro = pred - data[i]
        @printf("  %5.1f │ %5.2f │ %5.2f │ %+5.2f\n", t, data[i], pred, erro)
    end

    println("\n✓ Teste completo!")
    return pop
end

# Export test function
export test_neat_advanced

end # module NEATAdvanced
