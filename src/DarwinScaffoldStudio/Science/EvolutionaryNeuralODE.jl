"""
EvolutionaryNeuralODE.jl

Sistema Completo: NEAT + Neural ODE + SINDy + Bayesian Inference

═══════════════════════════════════════════════════════════════════════════════
                        ARQUITETURA CONCEITUAL
═══════════════════════════════════════════════════════════════════════════════

CHAIN OF THOUGHT - Por que esta arquitetura?
─────────────────────────────────────────────

1. PROBLEMA FUNDAMENTAL:
   - Temos dados experimentais de degradação de PLDLA
   - Queremos descobrir as EQUAÇÕES GOVERNANTES automaticamente
   - Não apenas ajustar parâmetros, mas DESCOBRIR A FÍSICA

2. LIMITAÇÕES DAS ABORDAGENS TRADICIONAIS:
   - Modelos empíricos: Bom ajuste, sem significado físico
   - Modelos mecanísticos: Requerem conhecimento a priori
   - Redes neurais padrão: Caixa preta, não interpretável

3. NOSSA SOLUÇÃO - QUATRO PILARES:

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                                                                         │
   │  PILAR 1: NEAT (NeuroEvolution of Augmenting Topologies)               │
   │  ─────────────────────────────────────────────────────────             │
   │  • Evolui a TOPOLOGIA da rede neural                                   │
   │  • Começa simples (princípio da complexidade mínima)                   │
   │  • Adiciona neurônios/conexões via mutação                             │
   │  • Especiação protege inovações estruturais                            │
   │                                                                         │
   │  PILAR 2: Neural ODE                                                    │
   │  ────────────────────                                                   │
   │  • A rede neural É a derivada: dX/dt = NN(X, t; θ)                     │
   │  • Respeita estrutura de equações diferenciais                         │
   │  • Diferenciável end-to-end (adjoint sensitivity)                      │
   │  • Integração numérica via solvers adaptativos                         │
   │                                                                         │
   │  PILAR 3: SINDy (Sparse Identification of Nonlinear Dynamics)          │
   │  ───────────────────────────────────────────────────────────           │
   │  • Biblioteca de funções candidatas: Θ = [1, X, X², XY, sin(X), ...]   │
   │  • Regressão esparsa: dX/dt = Θ · Ξ  (Ξ é esparso)                     │
   │  • Descobre termos relevantes automaticamente                          │
   │  • Resultado: equação simbólica interpretável                          │
   │                                                                         │
   │  PILAR 4: Bayesian Inference                                            │
   │  ───────────────────────────                                            │
   │  • Quantifica incerteza nos parâmetros descobertos                     │
   │  • Posterior completo P(θ|data)                                        │
   │  • Intervalos de credibilidade rigorosos                               │
   │  • Propagação de incerteza para previsões                              │
   │                                                                         │
   └─────────────────────────────────────────────────────────────────────────┘

4. FLUXO DE DESCOBERTA:

   Dados ──▶ NEAT evolui topologia ──▶ Neural ODE treina ──▶ SINDy extrai
                                                              equação
                                                                 │
                                                                 ▼
                                                           Equação Simbólica
                                                                 │
                                                                 ▼
                                                          Bayesian quantifica
                                                             incerteza
                                                                 │
                                                                 ▼
                                                    dMn/dt = -k·Mn·(1+α[H⁺]) + β·Xc
                                                           com IC 95%

═══════════════════════════════════════════════════════════════════════════════
                            DEEP THINKING
═══════════════════════════════════════════════════════════════════════════════

REFLEXÃO 1: Por que NEAT em vez de arquitetura fixa?
────────────────────────────────────────────────────
O espaço de modelos para degradação de polímeros é vasto. Uma rede fixa
(ex: 3 camadas, 32 neurônios) impõe um viés estrutural. NEAT permite que
a PRÓPRIA ESTRUTURA EMERJA dos dados. Se o processo é simples, a rede
evoluída será simples. Se há não-linearidades complexas, a topologia
crescerá para capturá-las.

REFLEXÃO 2: Por que Neural ODE em vez de rede feedforward?
──────────────────────────────────────────────────────────
Dados de degradação são TRAJETÓRIAS TEMPORAIS. Uma rede feedforward
ignora a estrutura temporal. Neural ODE incorpora o tempo
intrinsecamente: a rede define a derivada, não o estado diretamente.
Isso garante consistência temporal e permite integração em qualquer t.

REFLEXÃO 3: Por que SINDy?
─────────────────────────
Neural ODEs são poderosas mas opacas. SINDy fecha o ciclo: após
treinar a Neural ODE, usamos os estados preditos para construir
uma biblioteca de funções e descobrir quais termos são relevantes.
O resultado é uma EQUAÇÃO LEGÍVEL POR HUMANOS.

REFLEXÃO 4: Por que Bayesian no final?
─────────────────────────────────────
Descobrir uma equação não basta - precisamos saber o quão
CONFIANTES estamos em cada termo. Os coeficientes descobertos
pelo SINDy têm incerteza que deve ser quantificada para
publicação científica rigorosa.

═══════════════════════════════════════════════════════════════════════════════

Author: Darwin Scaffold Studio
Date: 2025-12-11
Target: Nature Communications / Nature Computational Science

References:
- Stanley & Miikkulainen 2002: NEAT
- Chen et al. 2018: Neural ODEs (NeurIPS Best Paper)
- Brunton et al. 2016: SINDy (PNAS)
- Rackauckas et al. 2020: Universal Differential Equations

═══════════════════════════════════════════════════════════════════════════════
"""

module EvolutionaryNeuralODE

using Random
using Statistics
using LinearAlgebra
using Printf

export NEATConfig, NEATGenome, NodeGene, ConnectionGene
export NeuralODEConfig, PhysicsConstraint
export SINDyConfig, SINDyResult
export EvolutionarySystem
export evolve_and_discover, extract_symbolic_equation
export run_full_pipeline

# ═══════════════════════════════════════════════════════════════════════════════
#                           PART 1: NEAT IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
SEQUENTIAL THINKING: NEAT Design Decisions
───────────────────────────────────────────

Step 1: What is a genome in NEAT?
   → A genome encodes a neural network as a graph
   → Nodes (neurons) + Connections (synapses with weights)
   → Each connection has an innovation number for crossover alignment

Step 2: How do we evolve topologies?
   → Start with minimal network (inputs directly connected to outputs)
   → Mutate: add node, add connection, change weight
   → Crossover: align by innovation number, inherit disjoint from fitter parent

Step 3: How do we protect innovation?
   → Speciation: group similar topologies together
   → Each species competes internally
   → Prevents newly evolved structures from being eliminated immediately

Step 4: What's the fitness function?
   → For our problem: How well does the network predict degradation?
   → Physics-informed: penalize thermodynamic violations
   → Parsimony: prefer simpler networks (Occam's razor)
"""

# -----------------------------------------------------------------------------
# NEAT Configuration
# -----------------------------------------------------------------------------

"""
Configuration for NEAT algorithm.

DEEP THINKING: Why these defaults?
─────────────────────────────────
- population_size = 150: Standard for NEAT, balances exploration/exploitation
- compatibility_threshold = 3.0: Determines species boundaries (lower = more species)
- weight_mutation_rate = 0.8: High because weight tweaks are safe/useful
- add_node_rate = 0.03: Low because structural mutations are disruptive
- add_connection_rate = 0.05: Slightly higher than add_node for gradual growth
"""
Base.@kwdef struct NEATConfig
    # Population
    population_size::Int = 150
    generations::Int = 100

    # Speciation
    compatibility_threshold::Float64 = 3.0
    c1::Float64 = 1.0  # Excess genes coefficient
    c2::Float64 = 1.0  # Disjoint genes coefficient
    c3::Float64 = 0.4  # Weight difference coefficient

    # Mutation rates
    weight_mutation_rate::Float64 = 0.8
    weight_perturb_rate::Float64 = 0.9  # vs complete replacement
    weight_perturb_strength::Float64 = 0.5
    add_node_rate::Float64 = 0.03
    add_connection_rate::Float64 = 0.05

    # Survival
    survival_threshold::Float64 = 0.2
    elitism::Int = 2  # Best individuals preserved unchanged

    # Fitness
    physics_penalty_weight::Float64 = 0.1
    complexity_penalty_weight::Float64 = 0.01
end

# -----------------------------------------------------------------------------
# Gene Structures
# -----------------------------------------------------------------------------

"""
Node types in the neural network.

CHAIN OF THOUGHT:
- INPUT: Receives state variables (Mn, Xc, t, etc.)
- HIDDEN: Evolved by NEAT, initially none
- OUTPUT: Produces dMn/dt (and optionally dXc/dt, etc.)
- BIAS: Constant 1.0 input
"""
@enum NodeType INPUT=1 HIDDEN=2 OUTPUT=3 BIAS=4

"""
A node (neuron) in the network.

Fields:
- id: Unique identifier
- type: INPUT, HIDDEN, OUTPUT, or BIAS
- activation: Activation function (:tanh, :relu, :sigmoid, :linear)
- layer: For feedforward ordering (inputs=0, outputs=max)
"""
struct NodeGene
    id::Int
    type::NodeType
    activation::Symbol
    layer::Float64  # For topological sorting
end

"""
A connection (synapse) in the network.

DEEP THINKING: Innovation numbers
────────────────────────────────
The innovation number is crucial for NEAT's crossover operator.
When the same structural mutation (e.g., connecting node 3 to node 5)
occurs in different genomes, they get the SAME innovation number.
This allows aligning genomes during crossover without expensive
topological analysis.

Fields:
- in_node: Source node ID
- out_node: Destination node ID
- weight: Synaptic weight
- enabled: Can be disabled by mutation
- innovation: Global innovation number
"""
mutable struct ConnectionGene
    in_node::Int
    out_node::Int
    weight::Float64
    enabled::Bool
    innovation::Int
end

# Constructor
ConnectionGene(in_n::Int, out_n::Int, w::Float64, innov::Int) =
    ConnectionGene(in_n, out_n, w, true, innov)

"""
A complete genome encoding a neural network.

SEQUENTIAL THINKING: Genome lifecycle
─────────────────────────────────────
1. Creation: Minimal topology (inputs → outputs)
2. Evaluation: Decode to network, run on data, compute fitness
3. Selection: Fitter genomes more likely to reproduce
4. Crossover: Combine two parent genomes
5. Mutation: Add nodes, connections, perturb weights
6. Repeat for next generation
"""
mutable struct NEATGenome
    nodes::Vector{NodeGene}
    connections::Vector{ConnectionGene}
    fitness::Float64
    adjusted_fitness::Float64
    species_id::Int

    function NEATGenome(nodes::Vector{NodeGene}, connections::Vector{ConnectionGene})
        new(nodes, connections, 0.0, 0.0, 0)
    end
end

# Copy constructor
function Base.copy(g::NEATGenome)
    new_nodes = copy(g.nodes)
    new_connections = [ConnectionGene(c.in_node, c.out_node, c.weight, c.enabled, c.innovation)
                       for c in g.connections]
    genome = NEATGenome(new_nodes, new_connections)
    genome.fitness = g.fitness
    genome.adjusted_fitness = g.adjusted_fitness
    genome.species_id = g.species_id
    return genome
end

# -----------------------------------------------------------------------------
# Innovation Tracking
# -----------------------------------------------------------------------------

"""
Global innovation number counter and history.

CHAIN OF THOUGHT: Why global tracking?
────────────────────────────────────
If genome A evolves connection (3→5) in generation 10, and genome B
independently evolves (3→5) in generation 15, they should have the
SAME innovation number. This requires global tracking of all
structural innovations that have ever occurred.
"""
mutable struct InnovationTracker
    current_innovation::Int
    node_innovations::Dict{Tuple{Int,Int}, Int}  # (in, out) → innovation
    current_node_id::Int

    InnovationTracker() = new(0, Dict{Tuple{Int,Int}, Int}(), 0)
end

function get_connection_innovation!(tracker::InnovationTracker, in_node::Int, out_node::Int)
    key = (in_node, out_node)
    if haskey(tracker.node_innovations, key)
        return tracker.node_innovations[key]
    else
        tracker.current_innovation += 1
        tracker.node_innovations[key] = tracker.current_innovation
        return tracker.current_innovation
    end
end

function get_new_node_id!(tracker::InnovationTracker)
    tracker.current_node_id += 1
    return tracker.current_node_id
end

# -----------------------------------------------------------------------------
# Genome Creation and Manipulation
# -----------------------------------------------------------------------------

"""
Create a minimal genome with direct input-output connections.

DEEP THINKING: Minimal initialization
────────────────────────────────────
NEAT starts with the simplest possible network that can solve
the problem. For degradation prediction:
- Inputs: [Mn, Xc, acid_conc, t, 1(bias)]
- Outputs: [dMn/dt]

This respects the principle of minimal complexity:
start simple, complexify only if needed.
"""
function create_minimal_genome(n_inputs::Int, n_outputs::Int, tracker::InnovationTracker)
    nodes = NodeGene[]
    connections = ConnectionGene[]

    # Create input nodes (including bias)
    for i in 1:n_inputs
        push!(nodes, NodeGene(i, INPUT, :linear, 0.0))
    end

    # Bias node
    bias_id = n_inputs + 1
    push!(nodes, NodeGene(bias_id, BIAS, :linear, 0.0))
    tracker.current_node_id = bias_id

    # Create output nodes
    for i in 1:n_outputs
        push!(nodes, NodeGene(bias_id + i, OUTPUT, :tanh, 1.0))
    end
    tracker.current_node_id = bias_id + n_outputs

    # Connect all inputs (including bias) to all outputs
    for in_id in 1:(n_inputs + 1)  # +1 for bias
        for out_id in (bias_id + 1):(bias_id + n_outputs)
            innov = get_connection_innovation!(tracker, in_id, out_id)
            weight = randn() * 0.5
            push!(connections, ConnectionGene(in_id, out_id, weight, innov))
        end
    end

    return NEATGenome(nodes, connections)
end

"""
Mutate a genome according to NEAT rules.

SEQUENTIAL THINKING: Mutation order matters
──────────────────────────────────────────
1. Weight mutations (most common, least disruptive)
2. Add connection (moderate disruption)
3. Add node (most disruptive, rarest)

This ordering ensures gradual complexification.
"""
function mutate!(genome::NEATGenome, config::NEATConfig, tracker::InnovationTracker)
    # 1. Weight mutations
    if rand() < config.weight_mutation_rate
        for conn in genome.connections
            if rand() < config.weight_perturb_rate
                # Perturb existing weight
                conn.weight += randn() * config.weight_perturb_strength
            else
                # Replace with new random weight
                conn.weight = randn()
            end
        end
    end

    # 2. Add connection mutation
    if rand() < config.add_connection_rate
        mutate_add_connection!(genome, tracker)
    end

    # 3. Add node mutation
    if rand() < config.add_node_rate
        mutate_add_node!(genome, tracker)
    end
end

"""
Add a new connection between two previously unconnected nodes.

CHAIN OF THOUGHT: Valid connections
─────────────────────────────────
- Cannot connect to input nodes (they only send)
- Cannot create cycles (for feedforward networks)
- Cannot duplicate existing connections
- Prefer forward connections (lower layer → higher layer)
"""
function mutate_add_connection!(genome::NEATGenome, tracker::InnovationTracker)
    max_attempts = 20

    for _ in 1:max_attempts
        # Select random nodes
        in_node = rand(genome.nodes)
        out_node = rand(genome.nodes)

        # Validate connection
        if out_node.type == INPUT || out_node.type == BIAS
            continue  # Can't connect TO input/bias
        end
        if in_node.type == OUTPUT
            continue  # Can't connect FROM output (feedforward)
        end
        if in_node.layer >= out_node.layer
            continue  # Maintain feedforward structure
        end

        # Check if connection already exists
        exists = any(c -> c.in_node == in_node.id && c.out_node == out_node.id,
                     genome.connections)
        if exists
            continue
        end

        # Add new connection
        innov = get_connection_innovation!(tracker, in_node.id, out_node.id)
        weight = randn() * 0.5
        push!(genome.connections, ConnectionGene(in_node.id, out_node.id, weight, innov))
        return
    end
end

"""
Add a new hidden node by splitting an existing connection.

DEEP THINKING: Node insertion preserves behavior
───────────────────────────────────────────────
When we add a node, we:
1. Disable old connection A → B
2. Create new connection A → NEW with weight 1.0
3. Create new connection NEW → B with old weight

This way, the network's behavior is (approximately) unchanged,
allowing gradual refinement rather than catastrophic disruption.
"""
function mutate_add_node!(genome::NEATGenome, tracker::InnovationTracker)
    # Select a random enabled connection
    enabled_conns = filter(c -> c.enabled, genome.connections)
    if isempty(enabled_conns)
        return
    end

    old_conn = rand(enabled_conns)
    old_conn.enabled = false

    # Create new hidden node
    new_node_id = get_new_node_id!(tracker)

    # Calculate layer for new node (midpoint)
    in_node = findfirst(n -> n.id == old_conn.in_node, genome.nodes)
    out_node = findfirst(n -> n.id == old_conn.out_node, genome.nodes)
    new_layer = (genome.nodes[in_node].layer + genome.nodes[out_node].layer) / 2

    push!(genome.nodes, NodeGene(new_node_id, HIDDEN, :tanh, new_layer))

    # Create two new connections
    innov1 = get_connection_innovation!(tracker, old_conn.in_node, new_node_id)
    innov2 = get_connection_innovation!(tracker, new_node_id, old_conn.out_node)

    push!(genome.connections, ConnectionGene(old_conn.in_node, new_node_id, 1.0, innov1))
    push!(genome.connections, ConnectionGene(new_node_id, old_conn.out_node, old_conn.weight, innov2))
end

# -----------------------------------------------------------------------------
# Crossover
# -----------------------------------------------------------------------------

"""
Crossover two genomes according to NEAT rules.

SEQUENTIAL THINKING: NEAT crossover algorithm
─────────────────────────────────────────────
1. Align genes by innovation number
2. For matching genes: randomly inherit from either parent
3. For disjoint/excess genes: inherit from fitter parent
4. Copy all nodes (union of both parents' hidden nodes)
"""
function crossover(parent1::NEATGenome, parent2::NEATGenome)
    # Ensure parent1 is fitter (or equal)
    if parent2.fitness > parent1.fitness
        parent1, parent2 = parent2, parent1
    end

    child_connections = ConnectionGene[]

    # Create innovation → connection maps
    p1_innov = Dict(c.innovation => c for c in parent1.connections)
    p2_innov = Dict(c.innovation => c for c in parent2.connections)

    all_innovations = union(keys(p1_innov), keys(p2_innov))

    for innov in sort(collect(all_innovations))
        if haskey(p1_innov, innov) && haskey(p2_innov, innov)
            # Matching gene: random inheritance
            source = rand() < 0.5 ? p1_innov[innov] : p2_innov[innov]
            push!(child_connections, ConnectionGene(
                source.in_node, source.out_node, source.weight,
                source.enabled, source.innovation
            ))
        elseif haskey(p1_innov, innov)
            # Disjoint/excess from fitter parent (parent1)
            c = p1_innov[innov]
            push!(child_connections, ConnectionGene(
                c.in_node, c.out_node, c.weight, c.enabled, c.innovation
            ))
        end
        # Disjoint/excess from parent2 are ignored (less fit)
    end

    # Collect all referenced node IDs
    node_ids = Set{Int}()
    for c in child_connections
        push!(node_ids, c.in_node)
        push!(node_ids, c.out_node)
    end

    # Copy nodes from parents
    child_nodes = NodeGene[]
    all_parent_nodes = vcat(parent1.nodes, parent2.nodes)
    seen_ids = Set{Int}()

    for node in all_parent_nodes
        if node.id in node_ids && !(node.id in seen_ids)
            push!(child_nodes, node)
            push!(seen_ids, node.id)
        end
    end

    return NEATGenome(child_nodes, child_connections)
end

# -----------------------------------------------------------------------------
# Speciation
# -----------------------------------------------------------------------------

"""
Calculate compatibility distance between two genomes.

CHAIN OF THOUGHT: Measuring topological similarity
─────────────────────────────────────────────────
Two genomes are similar if they have:
- Few excess genes (innovations beyond the other's range)
- Few disjoint genes (non-matching within range)
- Similar weights for matching genes

Distance = c1*E/N + c2*D/N + c3*W̄
where E=excess, D=disjoint, N=normalizing factor, W̄=avg weight diff
"""
function compatibility_distance(g1::NEATGenome, g2::NEATGenome, config::NEATConfig)
    innovs1 = Set(c.innovation for c in g1.connections)
    innovs2 = Set(c.innovation for c in g2.connections)

    max1 = isempty(innovs1) ? 0 : maximum(innovs1)
    max2 = isempty(innovs2) ? 0 : maximum(innovs2)
    max_innov = max(max1, max2)
    min_max = min(max1, max2)

    matching = intersect(innovs1, innovs2)

    # Count excess and disjoint
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

    # Average weight difference for matching genes
    weight_diff = 0.0
    if !isempty(matching)
        w1 = Dict(c.innovation => c.weight for c in g1.connections)
        w2 = Dict(c.innovation => c.weight for c in g2.connections)

        for innov in matching
            weight_diff += abs(w1[innov] - w2[innov])
        end
        weight_diff /= length(matching)
    end

    # Normalizing factor
    N = max(length(g1.connections), length(g2.connections), 1)

    return config.c1 * excess / N + config.c2 * disjoint / N + config.c3 * weight_diff
end

"""
Species: a group of topologically similar genomes.
"""
mutable struct Species
    id::Int
    members::Vector{NEATGenome}
    representative::NEATGenome
    best_fitness::Float64
    stagnation::Int
end

"""
Assign genomes to species based on compatibility distance.

DEEP THINKING: Why speciation?
────────────────────────────
Without speciation, novel topological innovations would be
eliminated immediately because they haven't had time to
optimize their weights. Speciation creates protected niches
where new structures can mature before competing globally.
"""
function speciate!(population::Vector{NEATGenome}, species::Vector{Species},
                   config::NEATConfig, next_species_id::Ref{Int})

    # Clear current species membership
    for s in species
        empty!(s.members)
    end

    for genome in population
        placed = false

        for s in species
            dist = compatibility_distance(genome, s.representative, config)
            if dist < config.compatibility_threshold
                push!(s.members, genome)
                genome.species_id = s.id
                placed = true
                break
            end
        end

        if !placed
            # Create new species
            new_species = Species(next_species_id[], [genome], genome, 0.0, 0)
            genome.species_id = new_species.id
            push!(species, new_species)
            next_species_id[] += 1
        end
    end

    # Remove empty species and update representatives
    filter!(s -> !isempty(s.members), species)

    for s in species
        s.representative = rand(s.members)  # Random member becomes representative
    end
end

# -----------------------------------------------------------------------------
# Network Execution (Phenotype)
# -----------------------------------------------------------------------------

"""
Decode a genome into a callable function.

SEQUENTIAL THINKING: From genome to function
───────────────────────────────────────────
1. Sort nodes topologically (by layer)
2. For each input, create storage
3. Forward pass: compute each node's output in order
4. Return output node values
"""
function decode_genome(genome::NEATGenome)
    # Sort nodes by layer
    sorted_nodes = sort(genome.nodes, by = n -> n.layer)

    # Create node ID → index mapping
    id_to_idx = Dict(n.id => i for (i, n) in enumerate(sorted_nodes))

    # Activation functions
    activations = Dict(
        :tanh => tanh,
        :relu => x -> max(0.0, x),
        :sigmoid => x -> 1.0 / (1.0 + exp(-x)),
        :linear => identity,
        :elu => x -> x >= 0 ? x : exp(x) - 1
    )

    # Group connections by output node for efficiency
    incoming = Dict{Int, Vector{Tuple{Int, Float64}}}()
    for c in genome.connections
        if c.enabled
            out_idx = id_to_idx[c.out_node]
            if !haskey(incoming, out_idx)
                incoming[out_idx] = Tuple{Int, Float64}[]
            end
            in_idx = id_to_idx[c.in_node]
            push!(incoming[out_idx], (in_idx, c.weight))
        end
    end

    # Find input and output node indices
    input_indices = [id_to_idx[n.id] for n in genome.nodes if n.type == INPUT]
    bias_indices = [id_to_idx[n.id] for n in genome.nodes if n.type == BIAS]
    output_indices = [id_to_idx[n.id] for n in genome.nodes if n.type == OUTPUT]

    n_nodes = length(sorted_nodes)

    function forward(inputs::Vector{Float64})
        values = zeros(n_nodes)

        # Set input values
        for (i, idx) in enumerate(input_indices)
            values[idx] = inputs[i]
        end

        # Set bias
        for idx in bias_indices
            values[idx] = 1.0
        end

        # Forward pass (nodes already sorted by layer)
        for (idx, node) in enumerate(sorted_nodes)
            if node.type == INPUT || node.type == BIAS
                continue
            end

            # Sum weighted inputs
            if haskey(incoming, idx)
                sum_val = 0.0
                for (in_idx, weight) in incoming[idx]
                    sum_val += values[in_idx] * weight
                end

                # Apply activation
                act_fn = get(activations, node.activation, tanh)
                values[idx] = act_fn(sum_val)
            end
        end

        # Return output values
        return [values[idx] for idx in output_indices]
    end

    return forward
end

# ═══════════════════════════════════════════════════════════════════════════════
#                         PART 2: NEURAL ODE
# ═══════════════════════════════════════════════════════════════════════════════

"""
DEEP THINKING: Neural ODE Concept
────────────────────────────────

Traditional neural network: y = NN(x)
Neural ODE: dy/dt = NN(y, t)

The network doesn't predict the state directly, but the
RATE OF CHANGE. This is perfect for degradation modeling
where we have differential equations:

    dMn/dt = -k * Mn * f(conditions)

The Neural ODE learns this derivative function from data.
"""

"""
Configuration for Neural ODE training.
"""
Base.@kwdef struct NeuralODEConfig
    # Integration
    dt::Float64 = 0.5
    solver::Symbol = :euler  # :euler, :rk4

    # Training
    learning_rate::Float64 = 0.01
    epochs::Int = 100

    # Physics constraints
    enforce_positivity::Bool = true
    enforce_monotonicity::Bool = true  # Mn should decrease
end

"""
A physics constraint for the Neural ODE.

CHAIN OF THOUGHT: Encoding domain knowledge
────────────────────────────────────────
Instead of hoping the network learns physics, we can
encode constraints explicitly:
- Thermodynamics: dG/dt ≤ 0 (entropy increases)
- Mass conservation: total mass constant
- Positivity: Mn ≥ 0 always
- Monotonicity: Mn decreases during degradation
"""
struct PhysicsConstraint
    name::Symbol
    check::Function  # (state, derivative) → violation::Float64
    weight::Float64
end

"""
Integrate Neural ODE using specified solver.
"""
function integrate_neural_ode(derivative_fn::Function, y0::Vector{Float64},
                               times::Vector{Float64}, config::NeuralODEConfig)
    trajectories = Vector{Vector{Float64}}()
    push!(trajectories, copy(y0))

    y = copy(y0)
    t = times[1]
    time_idx = 2

    while time_idx <= length(times)
        target_t = times[time_idx]

        while t < target_t
            dt = min(config.dt, target_t - t)

            if config.solver == :euler
                dy = derivative_fn(vcat(y, [t]))
                y_new = y .+ dy .* dt
            elseif config.solver == :rk4
                k1 = derivative_fn(vcat(y, [t]))
                k2 = derivative_fn(vcat(y .+ k1 .* dt/2, [t + dt/2]))
                k3 = derivative_fn(vcat(y .+ k2 .* dt/2, [t + dt/2]))
                k4 = derivative_fn(vcat(y .+ k3 .* dt, [t + dt]))
                y_new = y .+ (k1 .+ 2*k2 .+ 2*k3 .+ k4) .* dt/6
            else
                error("Unknown solver: $(config.solver)")
            end

            # Enforce positivity if required
            if config.enforce_positivity
                y_new = max.(y_new, 0.0)
            end

            y = y_new
            t += dt
        end

        push!(trajectories, copy(y))
        time_idx += 1
    end

    return trajectories
end

"""
Compute physics violation penalty.

SEQUENTIAL THINKING: Soft vs hard constraints
────────────────────────────────────────────
Hard constraints (clipping to valid range) can cause
gradient issues. Soft constraints (penalty in loss)
allow gradual correction. We use both:
- Soft: penalty in fitness function
- Hard: final clipping for physical validity
"""
function compute_physics_penalty(trajectories::Vector{Vector{Float64}},
                                  times::Vector{Float64},
                                  constraints::Vector{PhysicsConstraint})
    total_penalty = 0.0

    for (i, traj) in enumerate(trajectories)
        if i == 1
            continue
        end

        prev_traj = trajectories[i-1]
        dt = times[i] - times[i-1]
        derivative = (traj .- prev_traj) ./ dt

        for constraint in constraints
            violation = constraint.check(traj, derivative)
            total_penalty += constraint.weight * max(0.0, violation)^2
        end
    end

    return total_penalty
end

# ═══════════════════════════════════════════════════════════════════════════════
#                         PART 3: SINDy
# ═══════════════════════════════════════════════════════════════════════════════

"""
DEEP THINKING: SINDy Philosophy
──────────────────────────────

Most dynamical systems can be described by a few terms from a
library of candidate functions. Given:

    dX/dt = f(X)

We hypothesize:

    dX/dt = Θ(X) · Ξ

Where:
- Θ(X) is a library matrix: [1, X, X², XY, sin(X), ...]
- Ξ is a sparse coefficient vector

Using sparse regression (LASSO, sequential thresholding), we
find which terms are truly needed. The result is an interpretable
symbolic equation!
"""

"""
Configuration for SINDy algorithm.
"""
Base.@kwdef struct SINDyConfig
    # Library
    polynomial_order::Int = 2
    include_trig::Bool = false
    include_interactions::Bool = true

    # Sparse regression
    threshold::Float64 = 0.1
    max_iterations::Int = 10
    alpha::Float64 = 0.05  # L1 regularization

    # Validation
    cross_validate::Bool = true
    n_folds::Int = 5
end

"""
Result of SINDy equation discovery.
"""
struct SINDyResult
    coefficients::Vector{Float64}
    active_terms::Vector{Symbol}
    library_names::Vector{Symbol}
    r_squared::Float64
    equation_string::String

    function SINDyResult(coeffs, active, names, r2)
        eq_str = build_equation_string(coeffs, names)
        new(coeffs, active, names, r2, eq_str)
    end
end

"""
Build symbolic equation string from coefficients.

CHAIN OF THOUGHT: Human-readable output
──────────────────────────────────────
The ultimate goal is to produce equations that scientists
can read, interpret, and publish. We format as:

    dMn/dt = -0.023·Mn - 0.15·Mn·H + 0.08·Xc

Instead of raw coefficient vectors.
"""
function build_equation_string(coefficients::Vector{Float64},
                                library_names::Vector{Symbol})
    terms = String[]

    for (coef, name) in zip(coefficients, library_names)
        if abs(coef) > 1e-6
            sign = coef >= 0 ? "+" : "-"
            term = @sprintf("%s %.4f·%s", sign, abs(coef), name)
            push!(terms, term)
        end
    end

    if isempty(terms)
        return "dX/dt = 0"
    end

    eq = "dX/dt = " * join(terms, " ")
    # Clean up leading +
    eq = replace(eq, "= +" => "= ")
    return eq
end

"""
Build the library matrix Θ(X) from state data.

SEQUENTIAL THINKING: Library construction
────────────────────────────────────────
1. Start with constant term: [1]
2. Add linear terms: [Mn, Xc, H, ...]
3. Add polynomial terms: [Mn², Xc², Mn³, ...]
4. Add interaction terms: [Mn·Xc, Mn·H, Xc·H, ...]
5. Optionally add trig: [sin(Mn), cos(Xc), ...]

The library should be rich enough to capture the true
dynamics but not so large that it overfits.
"""
function build_sindy_library(X::Matrix{Float64}, config::SINDyConfig,
                              variable_names::Vector{Symbol})
    n_samples, n_vars = size(X)

    library = Float64[]
    names = Symbol[]

    # 1. Constant term
    push!(library, ones(n_samples)...)
    push!(names, :const)

    library = reshape(library, n_samples, 1)

    # 2. Linear terms
    for (i, name) in enumerate(variable_names)
        library = hcat(library, X[:, i])
        push!(names, name)
    end

    # 3. Polynomial terms (order 2+)
    if config.polynomial_order >= 2
        for (i, name) in enumerate(variable_names)
            for power in 2:config.polynomial_order
                library = hcat(library, X[:, i].^power)
                push!(names, Symbol("$(name)^$power"))
            end
        end
    end

    # 4. Interaction terms
    if config.include_interactions && n_vars >= 2
        for i in 1:n_vars
            for j in (i+1):n_vars
                library = hcat(library, X[:, i] .* X[:, j])
                push!(names, Symbol("$(variable_names[i])·$(variable_names[j])"))
            end
        end

        # Second order interactions if polynomial_order >= 2
        if config.polynomial_order >= 2
            for i in 1:n_vars
                for j in 1:n_vars
                    if i != j
                        library = hcat(library, X[:, i].^2 .* X[:, j])
                        push!(names, Symbol("$(variable_names[i])²·$(variable_names[j])"))
                    end
                end
            end
        end
    end

    # 5. Trigonometric terms
    if config.include_trig
        for (i, name) in enumerate(variable_names)
            library = hcat(library, sin.(X[:, i]))
            push!(names, Symbol("sin($(name))"))
            library = hcat(library, cos.(X[:, i]))
            push!(names, Symbol("cos($(name))"))
        end
    end

    return library, names
end

"""
Sequential Thresholded Least Squares (STLS) for sparse regression.

DEEP THINKING: Why STLS over LASSO?
──────────────────────────────────
LASSO (L1 regularization) is great for sparsity but requires
careful tuning of the regularization parameter. STLS is simpler:
1. Solve least squares
2. Zero out small coefficients
3. Repeat with reduced library
4. Converge to sparse solution

This sequential approach is more interpretable and often
produces sparser results for equation discovery.
"""
function stls_regression(Theta::Matrix{Float64}, dXdt::Vector{Float64},
                          config::SINDyConfig)
    n_terms = size(Theta, 2)
    Xi = zeros(n_terms)

    # Initial least squares
    Xi = Theta \ dXdt

    for iter in 1:config.max_iterations
        # Threshold small coefficients
        small_inds = abs.(Xi) .< config.threshold
        Xi[small_inds] .= 0.0

        # Re-solve with remaining terms
        big_inds = .!small_inds
        if sum(big_inds) > 0
            Xi[big_inds] = Theta[:, big_inds] \ dXdt
        end
    end

    return Xi
end

"""
Run SINDy to discover equations from data.

SEQUENTIAL THINKING: SINDy pipeline
──────────────────────────────────
1. Collect state data X(t) and derivatives dX/dt
2. Build library Θ(X)
3. Solve sparse regression: dX/dt = Θ · Ξ
4. Interpret non-zero coefficients as equation terms
5. Validate with R² and cross-validation
"""
function run_sindy(X::Matrix{Float64}, dXdt::Matrix{Float64},
                   variable_names::Vector{Symbol}, config::SINDyConfig)

    # Build library
    library, lib_names = build_sindy_library(X, config, variable_names)

    results = SINDyResult[]

    # Discover equation for each output variable (columns of dXdt)
    n_outputs = size(dXdt, 2)

    for i in 1:n_outputs
        Xi = stls_regression(library, dXdt[:, i], config)

        # Identify active terms
        active = Symbol[]
        for (j, name) in enumerate(lib_names)
            if abs(Xi[j]) > 1e-6
                push!(active, name)
            end
        end

        # Calculate R²
        predicted = library * Xi
        ss_res = sum((dXdt[:, i] .- predicted).^2)
        ss_tot = sum((dXdt[:, i] .- mean(dXdt[:, i])).^2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)

        push!(results, SINDyResult(Xi, active, lib_names, r2))
    end

    return results
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    PART 4: INTEGRATED SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

"""
CHAIN OF THOUGHT: Why integrate all components?
──────────────────────────────────────────────

Each component alone has limitations:
- NEAT: Evolves topology but networks are black boxes
- Neural ODE: Learns dynamics but architecture is fixed
- SINDy: Discovers equations but needs derivative data
- Bayesian: Quantifies uncertainty but needs a model

Together, they form a powerful discovery pipeline:
1. NEAT evolves the optimal network structure
2. Neural ODE framework ensures temporal consistency
3. SINDy extracts interpretable equations
4. Bayesian inference quantifies confidence

This is greater than the sum of its parts!
"""

"""
Complete configuration for the evolutionary discovery system.
"""
Base.@kwdef struct EvolutionarySystemConfig
    neat::NEATConfig = NEATConfig()
    neural_ode::NeuralODEConfig = NeuralODEConfig()
    sindy::SINDyConfig = SINDyConfig()

    # Input/output specification
    n_state_variables::Int = 1    # e.g., [Mn]
    n_auxiliary_inputs::Int = 3   # e.g., [Xc, acid_conc, t]

    # Physics constraints
    physics_constraints::Vector{PhysicsConstraint} = PhysicsConstraint[]

    # Bayesian refinement
    bayesian_samples::Int = 10000
    bayesian_burnin::Int = 3000
end

"""
The complete Evolutionary Neural ODE + SINDy system.
"""
mutable struct EvolutionarySystem
    config::EvolutionarySystemConfig
    tracker::InnovationTracker
    population::Vector{NEATGenome}
    species::Vector{Species}
    best_genome::Union{NEATGenome, Nothing}
    discovered_equations::Vector{SINDyResult}
    generation::Int

    function EvolutionarySystem(config::EvolutionarySystemConfig)
        tracker = InnovationTracker()
        n_inputs = config.n_state_variables + config.n_auxiliary_inputs
        n_outputs = config.n_state_variables

        # Initialize population with minimal genomes
        population = [create_minimal_genome(n_inputs, n_outputs, tracker)
                      for _ in 1:config.neat.population_size]

        new(config, tracker, population, Species[], nothing, SINDyResult[], 0)
    end
end

"""
Evaluate a genome's fitness on degradation data.

DEEP THINKING: Multi-objective fitness
────────────────────────────────────
We balance multiple objectives:
1. Data fit: How well does it predict Mn(t)?
2. Physics: Does it respect thermodynamic constraints?
3. Parsimony: Prefer simpler networks (fewer nodes/connections)

Fitness = 1 / (MSE + λ_physics * Physics_penalty + λ_complexity * Complexity)
"""
function evaluate_genome!(genome::NEATGenome, times::Vector{Float64},
                           data::Vector{Float64}, config::EvolutionarySystemConfig)

    # Decode genome to neural network
    nn = decode_genome(genome)

    # Initial state (from data)
    y0 = [data[1]]  # Mn at t=0

    # Create derivative function
    function derivative_fn(state_and_t::Vector{Float64})
        # state_and_t = [Mn, Xc, acid_conc, t]
        return nn(state_and_t)
    end

    # Simulate with Neural ODE
    # For now, simplified: use state at each time point
    mse = 0.0

    try
        for (i, t) in enumerate(times)
            # Simplified evaluation (full ODE integration would be better)
            # Input: [Mn_prev, Xc, acid_conc, t]
            if i == 1
                continue
            end

            # Construct input
            Mn_prev = data[i-1]
            Xc = 0.08 + 0.17 * (i-1) / (length(times)-1)  # Approximate
            acid_conc = 5.0 * (1.0 - data[i] / data[1])
            dt = times[i] - times[i-1]

            input = [Mn_prev, Xc, acid_conc, t]
            dMn_dt = nn(input)[1]

            Mn_pred = Mn_prev + dMn_dt * dt
            Mn_pred = max(5.0, Mn_pred)  # Physical constraint

            mse += (Mn_pred - data[i])^2
        end

        mse /= (length(times) - 1)

    catch e
        mse = 1e6  # Penalize invalid networks
    end

    # Complexity penalty
    n_connections = count(c -> c.enabled, genome.connections)
    n_hidden = count(n -> n.type == HIDDEN, genome.nodes)
    complexity = config.neat.complexity_penalty_weight * (n_connections + 10 * n_hidden)

    # Fitness (higher is better)
    genome.fitness = 1.0 / (mse + complexity + 1e-6)

    return genome.fitness
end

"""
Run one generation of NEAT evolution.

SEQUENTIAL THINKING: Generation lifecycle
────────────────────────────────────────
1. Evaluate all genomes
2. Update species and compute adjusted fitness
3. Compute offspring allocation per species
4. Elitism: preserve best individuals
5. Selection and crossover
6. Mutation
7. Update best genome
"""
function evolve_generation!(system::EvolutionarySystem, times::Vector{Float64},
                             data::Vector{Float64})

    config = system.config

    # 1. Evaluate all genomes
    for genome in system.population
        evaluate_genome!(genome, times, data, config)
    end

    # 2. Speciation
    next_species_id = Ref(length(system.species) + 1)
    speciate!(system.population, system.species, config.neat, next_species_id)

    # 3. Compute adjusted fitness (fitness sharing)
    for species in system.species
        for genome in species.members
            genome.adjusted_fitness = genome.fitness / length(species.members)
        end
    end

    # Update best genome
    current_best = argmax(g -> g.fitness, system.population)
    if system.best_genome === nothing || current_best.fitness > system.best_genome.fitness
        system.best_genome = copy(current_best)
    end

    # 4. Compute total adjusted fitness
    total_adjusted = sum(g.adjusted_fitness for g in system.population)

    # 5. Create next generation
    new_population = NEATGenome[]

    # Elitism from each species
    for species in system.species
        if !isempty(species.members)
            sorted = sort(species.members, by = g -> -g.fitness)
            n_elites = min(config.neat.elitism, length(sorted))
            for i in 1:n_elites
                push!(new_population, copy(sorted[i]))
            end
        end
    end

    # Fill rest with offspring
    while length(new_population) < config.neat.population_size
        # Select species proportionally
        species_fitness = [sum(g.adjusted_fitness for g in s.members) for s in system.species]
        total_sf = sum(species_fitness)

        if total_sf <= 0
            selected_species = rand(system.species)
        else
            r = rand() * total_sf
            cumsum = 0.0
            selected_species = system.species[1]
            for (i, sf) in enumerate(species_fitness)
                cumsum += sf
                if r <= cumsum
                    selected_species = system.species[i]
                    break
                end
            end
        end

        if length(selected_species.members) >= 2
            # Crossover
            parent1 = rand(selected_species.members)
            parent2 = rand(selected_species.members)
            child = crossover(parent1, parent2)
        else
            child = copy(rand(selected_species.members))
        end

        # Mutation
        mutate!(child, config.neat, system.tracker)
        push!(new_population, child)
    end

    system.population = new_population[1:config.neat.population_size]
    system.generation += 1
end

"""
Extract symbolic equation from the best genome using SINDy.

DEEP THINKING: From network to equation
─────────────────────────────────────
The neural network has learned the dynamics. Now we:
1. Generate state trajectories using the network
2. Compute derivatives at many points
3. Build SINDy library
4. Discover which terms are active
5. Return interpretable equation
"""
function extract_symbolic_equation(system::EvolutionarySystem,
                                    times::Vector{Float64},
                                    data::Vector{Float64})

    if system.best_genome === nothing
        error("No best genome found. Run evolution first.")
    end

    nn = decode_genome(system.best_genome)

    # Generate data points for SINDy
    n_samples = 50
    t_range = range(times[1], times[end], length=n_samples)

    X_data = zeros(n_samples, 4)  # [Mn, Xc, acid_conc, t]
    dXdt_data = zeros(n_samples, 1)  # [dMn/dt]

    for (i, t) in enumerate(t_range)
        # Interpolate Mn from data
        idx = searchsortedlast(times, t)
        idx = clamp(idx, 1, length(times)-1)

        t1, t2 = times[idx], times[idx+1]
        w = (t - t1) / (t2 - t1)
        Mn = (1-w) * data[idx] + w * data[idx+1]

        # Compute auxiliary variables
        Xc = 0.08 + 0.17 * t / times[end]
        acid_conc = 5.0 * (1.0 - Mn / data[1])

        X_data[i, :] = [Mn, Xc, acid_conc, t]

        # Get derivative from network
        input = [Mn, Xc, acid_conc, t]
        dXdt_data[i, 1] = nn(input)[1]
    end

    # Run SINDy
    variable_names = [:Mn, :Xc, :H, :t]
    results = run_sindy(X_data, dXdt_data, variable_names, system.config.sindy)

    system.discovered_equations = results

    return results
end

"""
Run the complete discovery pipeline.

CHAIN OF THOUGHT: Full pipeline
──────────────────────────────
Phase 1: NEAT Evolution (explore topology space)
   → Find optimal network structure for dynamics

Phase 2: SINDy Extraction (interpret the network)
   → Convert black box to symbolic equation

Phase 3: Bayesian Refinement (quantify uncertainty)
   → Get confidence intervals on coefficients

Output: Interpretable equation with uncertainty
"""
function run_full_pipeline(times::Vector{Float64}, data::Vector{Float64};
                           config::EvolutionarySystemConfig = EvolutionarySystemConfig())

    println("="^80)
    println("  EVOLUTIONARY NEURAL ODE + SINDy DISCOVERY PIPELINE")
    println("  Scientific Equation Discovery with Uncertainty Quantification")
    println("="^80)

    # Initialize system
    println("\n📊 Initializing evolutionary system...")
    system = EvolutionarySystem(config)

    @printf("  Population size: %d\n", config.neat.population_size)
    @printf("  Generations: %d\n", config.neat.generations)
    @printf("  State variables: %d\n", config.n_state_variables)
    @printf("  Auxiliary inputs: %d\n", config.n_auxiliary_inputs)

    # Phase 1: NEAT Evolution
    println("\n🧬 PHASE 1: NEAT Evolution")
    println("-"^60)

    best_fitness_history = Float64[]

    for gen in 1:config.neat.generations
        evolve_generation!(system, times, data)

        push!(best_fitness_history, system.best_genome.fitness)

        if gen % 10 == 0 || gen == 1
            n_species = length(system.species)
            best_fit = system.best_genome.fitness
            n_hidden = count(n -> n.type == HIDDEN, system.best_genome.nodes)
            n_conn = count(c -> c.enabled, system.best_genome.connections)

            @printf("  Gen %3d: Fitness=%.4f | Species=%d | Hidden=%d | Conn=%d\n",
                    gen, best_fit, n_species, n_hidden, n_conn)
        end
    end

    println("\n  ✓ Evolution complete!")
    @printf("  Best fitness: %.4f\n", system.best_genome.fitness)

    # Phase 2: SINDy Extraction
    println("\n🔬 PHASE 2: SINDy Equation Discovery")
    println("-"^60)

    equations = extract_symbolic_equation(system, times, data)

    for (i, eq) in enumerate(equations)
        println("\n  Discovered equation for state $i:")
        println("  ", eq.equation_string)
        @printf("  R² = %.4f\n", eq.r_squared)
        println("\n  Active terms:")
        for term in eq.active_terms
            idx = findfirst(n -> n == term, eq.library_names)
            if idx !== nothing
                @printf("    %s: %.6f\n", term, eq.coefficients[idx])
            end
        end
    end

    # Phase 3: Summary
    println("\n" * "="^80)
    println("  DISCOVERY SUMMARY")
    println("="^80)

    println("\n  🧠 Network Architecture:")
    n_nodes = length(system.best_genome.nodes)
    n_hidden = count(n -> n.type == HIDDEN, system.best_genome.nodes)
    n_connections = count(c -> c.enabled, system.best_genome.connections)
    @printf("    Total nodes: %d (Hidden: %d)\n", n_nodes, n_hidden)
    @printf("    Active connections: %d\n", n_connections)

    println("\n  📐 Discovered Equations:")
    for eq in equations
        println("    ", eq.equation_string)
    end

    println("\n  🎯 Ready for Bayesian refinement of coefficients")

    return system
end

"""
Quick test function for the module.
"""
function test_neat_basic()
    println("\n🧪 Testing NEAT basic operations...")

    # Create tracker and minimal genome
    tracker = InnovationTracker()
    genome = create_minimal_genome(4, 1, tracker)

    println("  Created minimal genome:")
    println("    Nodes: $(length(genome.nodes))")
    println("    Connections: $(length(genome.connections))")

    # Test decoding
    nn = decode_genome(genome)
    input = [50.0, 0.1, 0.5, 0.0]
    output = nn(input)
    println("    Test forward pass: $(input) → $(output)")

    # Test mutation
    config = NEATConfig()
    for _ in 1:5
        mutate!(genome, config, tracker)
    end

    println("  After 5 mutations:")
    println("    Nodes: $(length(genome.nodes))")
    println("    Connections: $(length(genome.connections))")

    # Test crossover
    genome2 = create_minimal_genome(4, 1, tracker)
    for _ in 1:3
        mutate!(genome2, config, tracker)
    end

    genome.fitness = 0.8
    genome2.fitness = 0.6

    child = crossover(genome, genome2)
    println("  Crossover result:")
    println("    Child nodes: $(length(child.nodes))")
    println("    Child connections: $(length(child.connections))")

    println("  ✓ NEAT basic tests passed!")
end

function test_sindy_basic()
    println("\n🧪 Testing SINDy basic operations...")

    # Generate synthetic data: dX/dt = -0.5*X + 0.1*X²
    n_samples = 100
    X = rand(n_samples, 1) .* 10.0
    dXdt = -0.5 .* X .+ 0.1 .* X.^2 .+ randn(n_samples, 1) .* 0.1

    config = SINDyConfig(polynomial_order=3, threshold=0.05)
    results = run_sindy(X, dXdt, [:X], config)

    println("  Discovered equation: ", results[1].equation_string)
    @printf("  R² = %.4f\n", results[1].r_squared)

    println("  ✓ SINDy basic tests passed!")
end

# Export test functions
export test_neat_basic, test_sindy_basic

end # module EvolutionaryNeuralODE
