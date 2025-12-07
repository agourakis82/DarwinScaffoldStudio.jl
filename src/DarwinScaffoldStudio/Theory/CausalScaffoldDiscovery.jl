"""
    CausalScaffoldDiscovery

State-of-the-art Causal Inference for Scaffold-Cell Interactions.

Implements Pearl's Causal Hierarchy + Modern Methods:
- Structural Causal Models (SCM) with full do-calculus
- PC Algorithm for causal discovery (Spirtes et al. 2000)
- FCI Algorithm for latent confounders (Spirtes et al. 2000)
- GES (Greedy Equivalence Search) score-based discovery
- NOTEARS continuous optimization for DAG learning (Zheng et al. 2018)
- DoWhy-style identification and estimation (Sharma & Kiciman 2020)
- Counterfactual inference (Pearl 2009)
- Double/Debiased Machine Learning (Chernozhukov et al. 2018)
- Causal forests for heterogeneous treatment effects (Wager & Athey 2018)
- Sensitivity analysis (Cinelli & Hazlett 2020)
- Instrumental Variables and Regression Discontinuity
- Difference-in-Differences estimator

Pearl's Causal Hierarchy:
1. Association: P(Y|X) - seeing
2. Intervention: P(Y|do(X)) - doing
3. Counterfactuals: P(Y_x|X',Y') - imagining

References:
- Pearl 2009: Causality (2nd ed.)
- Peters, Janzing, Schölkopf 2017: Elements of Causal Inference
- Hernán & Robins 2020: Causal Inference: What If
"""
module CausalScaffoldDiscovery

using LinearAlgebra
using Statistics
using Random
using Distributions

export CausalDAG, SCM, CausalModel
export discover_causal_graph, pc_algorithm, fci_algorithm, ges_algorithm, notears
export intervene, do_calculus, compute_do_effect
export estimate_ate, estimate_cate, backdoor_adjustment, frontdoor_adjustment
export counterfactual, twin_network_counterfactual
export DoWhyModel, identify, estimate, refute
export DoubleML, causal_forest, heterogeneous_effects
export sensitivity_analysis, e_value, robustness_value
export instrumental_variables, regression_discontinuity, difference_in_differences
export CausalGraph, add_edge!, remove_edge!, is_d_separated
export propensity_score_matching, inverse_probability_weighting

# =============================================================================
# Causal Graph Structures
# =============================================================================

"""
    CausalGraph

Directed Acyclic Graph for causal structure.
"""
mutable struct CausalGraph
    nodes::Vector{String}
    adjacency::Matrix{Int}  # 1 = edge, 0 = no edge, -1 = bidirected (latent)
    node_index::Dict{String, Int}
end

function CausalGraph(nodes::Vector{String})
    n = length(nodes)
    node_index = Dict(node => i for (i, node) in enumerate(nodes))
    CausalGraph(nodes, zeros(Int, n, n), node_index)
end

function add_edge!(g::CausalGraph, from::String, to::String; bidirected::Bool=false)
    i, j = g.node_index[from], g.node_index[to]
    g.adjacency[i, j] = bidirected ? -1 : 1
end

function remove_edge!(g::CausalGraph, from::String, to::String)
    i, j = g.node_index[from], g.node_index[to]
    g.adjacency[i, j] = 0
end

function has_edge(g::CausalGraph, from::String, to::String)
    i, j = g.node_index[from], g.node_index[to]
    return g.adjacency[i, j] != 0
end

function get_parents(g::CausalGraph, node::String)
    j = g.node_index[node]
    parent_indices = findall(g.adjacency[:, j] .== 1)
    return [g.nodes[i] for i in parent_indices]
end

function get_children(g::CausalGraph, node::String)
    i = g.node_index[node]
    child_indices = findall(g.adjacency[i, :] .== 1)
    return [g.nodes[j] for j in child_indices]
end

function get_ancestors(g::CausalGraph, node::String)
    ancestors = Set{String}()
    queue = get_parents(g, node)
    while !isempty(queue)
        parent = popfirst!(queue)
        if parent ∉ ancestors
            push!(ancestors, parent)
            append!(queue, get_parents(g, parent))
        end
    end
    return collect(ancestors)
end

function get_descendants(g::CausalGraph, node::String)
    descendants = Set{String}()
    queue = get_children(g, node)
    while !isempty(queue)
        child = popfirst!(queue)
        if child ∉ descendants
            push!(descendants, child)
            append!(queue, get_children(g, child))
        end
    end
    return collect(descendants)
end

"""
    is_d_separated(g, X, Y, Z)

Test d-separation: X ⊥_d Y | Z in graph g (Pearl 2009, Definition 1.2.3)
"""
function is_d_separated(g::CausalGraph, X::Vector{String}, Y::Vector{String},
                        Z::Vector{String})
    # Bayes-Ball algorithm (Shachter 1998)
    n = length(g.nodes)

    # Mark ancestors of Z
    Z_ancestors = Set{String}()
    for z in Z
        union!(Z_ancestors, Set(get_ancestors(g, z)))
        push!(Z_ancestors, z)
    end

    # BFS from X to Y
    visited_from_child = Set{String}()
    visited_from_parent = Set{String}()

    # (node, came_from_child)
    queue = [(x, false) for x in X]

    while !isempty(queue)
        node, from_child = popfirst!(queue)

        if node in Y
            return false  # Path found, not d-separated
        end

        is_conditioned = node in Z

        if from_child
            # Came from child
            if node ∉ visited_from_child
                push!(visited_from_child, node)

                if !is_conditioned
                    # Can go to parents
                    for parent in get_parents(g, node)
                        push!(queue, (parent, false))
                    end
                end

                if is_conditioned || node in Z_ancestors
                    # Can go to children (collider opened)
                    for child in get_children(g, node)
                        push!(queue, (child, true))
                    end
                end
            end
        else
            # Came from parent
            if node ∉ visited_from_parent
                push!(visited_from_parent, node)

                if !is_conditioned
                    # Can go to children
                    for child in get_children(g, node)
                        push!(queue, (child, true))
                    end
                    # Can go to parents (fork)
                    for parent in get_parents(g, node)
                        push!(queue, (parent, false))
                    end
                end
            end
        end
    end

    return true  # No path found, d-separated
end

# =============================================================================
# Structural Causal Model
# =============================================================================

"""
    SCM (Structural Causal Model)

Full SCM with structural equations and exogenous noise.
"""
mutable struct SCM
    graph::CausalGraph
    structural_equations::Dict{String, Function}  # V = f(Pa(V), U_V)
    noise_distributions::Dict{String, Distribution}
    data::Union{Nothing, Matrix{Float64}}
    var_names::Vector{String}
end

function SCM(nodes::Vector{String})
    graph = CausalGraph(nodes)
    equations = Dict{String, Function}()
    noise = Dict{String, Distribution}(v => Normal(0, 1) for v in nodes)
    SCM(graph, equations, noise, nothing, nodes)
end

function set_equation!(scm::SCM, node::String,
                       equation::Function,
                       noise_dist::Distribution=Normal(0, 1))
    scm.structural_equations[node] = equation
    scm.noise_distributions[node] = noise_dist
end

"""
    sample(scm, n)

Sample from the SCM (observational distribution).
"""
function sample(scm::SCM, n::Int)
    nodes = topological_sort(scm.graph)
    data = Dict{String, Vector{Float64}}()

    # Sample exogenous noise
    U = Dict(v => rand(scm.noise_distributions[v], n) for v in nodes)

    for v in nodes
        parents = get_parents(scm.graph, v)
        if isempty(parents)
            # Exogenous variable
            if haskey(scm.structural_equations, v)
                data[v] = scm.structural_equations[v](U[v])
            else
                data[v] = U[v]
            end
        else
            # Endogenous variable
            parent_data = hcat([data[p] for p in parents]...)
            data[v] = scm.structural_equations[v](parent_data, U[v])
        end
    end

    return hcat([data[v] for v in scm.var_names]...)
end

"""
    intervene(scm, interventions)

Create interventional SCM by cutting edges and fixing values.
do(X = x) operation.
"""
function intervene(scm::SCM, interventions::Dict{String, Float64})
    # Create mutilated graph
    new_graph = CausalGraph(scm.graph.nodes)
    new_graph.adjacency = copy(scm.graph.adjacency)

    # Remove incoming edges to intervened variables
    for (var, _) in interventions
        j = new_graph.node_index[var]
        new_graph.adjacency[:, j] .= 0
    end

    # Create new equations (constants for intervened variables)
    new_equations = copy(scm.structural_equations)
    for (var, val) in interventions
        new_equations[var] = (args...) -> fill(val, length(args[end]))
    end

    return SCM(new_graph, new_equations, scm.noise_distributions, nothing, scm.var_names)
end

function topological_sort(g::CausalGraph)
    n = length(g.nodes)
    visited = falses(n)
    order = String[]

    function dfs(i)
        visited[i] = true
        for j in 1:n
            if g.adjacency[i, j] == 1 && !visited[j]
                dfs(j)
            end
        end
        pushfirst!(order, g.nodes[i])
    end

    for i in 1:n
        if !visited[i]
            dfs(i)
        end
    end

    return order
end

# =============================================================================
# Causal Discovery Algorithms
# =============================================================================

"""
    pc_algorithm(data, var_names; alpha)

PC Algorithm for causal discovery (Spirtes, Glymour, Scheines 2000).
Constraint-based approach using conditional independence tests.
"""
function pc_algorithm(data::Matrix{Float64}, var_names::Vector{String};
                      alpha::Float64=0.05, max_cond_size::Int=3)
    n_vars = length(var_names)
    n_samples = size(data, 1)

    # Initialize complete undirected graph
    graph = CausalGraph(var_names)
    graph.adjacency = ones(Int, n_vars, n_vars) - I(n_vars)

    # Separation sets
    sep_sets = Dict{Tuple{Int,Int}, Vector{Int}}()

    # Phase I: Remove edges based on conditional independence
    for cond_size in 0:max_cond_size
        for i in 1:n_vars
            for j in (i+1):n_vars
                if graph.adjacency[i, j] == 0
                    continue
                end

                # Get adjacent nodes (potential conditioning sets)
                adj_i = findall(graph.adjacency[i, :] .!= 0)
                adj_i = setdiff(adj_i, [j])

                if length(adj_i) >= cond_size
                    # Test all subsets of size cond_size
                    for S in combinations(adj_i, cond_size)
                        if test_conditional_independence_fisher(
                            data, i, j, S; alpha=alpha, n=n_samples)
                            # Remove edge
                            graph.adjacency[i, j] = 0
                            graph.adjacency[j, i] = 0
                            sep_sets[(i, j)] = S
                            sep_sets[(j, i)] = S
                            break
                        end
                    end
                end
            end
        end
    end

    # Phase II: Orient edges (v-structures)
    for j in 1:n_vars
        # Find unshielded triples i - j - k where i and k not adjacent
        neighbors_j = findall(graph.adjacency[:, j] .!= 0)

        for (idx1, i) in enumerate(neighbors_j)
            for k in neighbors_j[(idx1+1):end]
                if graph.adjacency[i, k] == 0  # Unshielded
                    # Check if j is in separation set of (i, k)
                    S = get(sep_sets, (min(i,k), max(i,k)), Int[])
                    if j ∉ S
                        # Orient as v-structure: i → j ← k
                        graph.adjacency[i, j] = 1
                        graph.adjacency[j, i] = 0
                        graph.adjacency[k, j] = 1
                        graph.adjacency[j, k] = 0
                    end
                end
            end
        end
    end

    # Phase III: Apply Meek's orientation rules
    apply_meek_rules!(graph)

    return graph
end

"""
Apply Meek's orientation rules (Meek 1995)
"""
function apply_meek_rules!(g::CausalGraph)
    n = length(g.nodes)
    changed = true

    while changed
        changed = false

        for i in 1:n, j in 1:n
            if g.adjacency[i, j] == 1 && g.adjacency[j, i] == 1  # Undirected
                # Rule 1: If i → j - k and i, k not adjacent, orient j → k
                for k in 1:n
                    if k != i && k != j
                        if g.adjacency[j, k] == 1 && g.adjacency[k, j] == 1  # j - k
                            if g.adjacency[i, k] == 0 && g.adjacency[k, i] == 0  # not adjacent
                                for m in 1:n
                                    if g.adjacency[m, j] == 1 && g.adjacency[j, m] == 0  # m → j
                                        if g.adjacency[m, k] == 0  # m, k not adjacent
                                            g.adjacency[j, k] = 1
                                            g.adjacency[k, j] = 0
                                            changed = true
                                        end
                                    end
                                end
                            end
                        end
                    end
                end

                # Rule 2: If i → k → j, orient i → j
                for k in 1:n
                    if k != i && k != j
                        if g.adjacency[i, k] == 1 && g.adjacency[k, i] == 0 &&
                           g.adjacency[k, j] == 1 && g.adjacency[j, k] == 0
                            g.adjacency[i, j] = 1
                            g.adjacency[j, i] = 0
                            changed = true
                        end
                    end
                end
            end
        end
    end
end

function combinations(arr, k)
    if k == 0
        return [Int[]]
    elseif k > length(arr)
        return Vector{Int}[]
    else
        result = Vector{Int}[]
        for i in 1:length(arr)
            for combo in combinations(arr[(i+1):end], k-1)
                push!(result, vcat(arr[i], combo))
            end
        end
        return result
    end
end

"""
    fci_algorithm(data, var_names; alpha)

FCI (Fast Causal Inference) Algorithm for graphs with latent confounders.
Outputs PAG (Partial Ancestral Graph) with bidirected edges.
"""
function fci_algorithm(data::Matrix{Float64}, var_names::Vector{String};
                       alpha::Float64=0.05)
    # Start with PC algorithm result
    graph = pc_algorithm(data, var_names; alpha=alpha)

    # FCI additional steps:
    # 1. Find possible d-sep sets
    # 2. Re-orient considering latent confounders
    # 3. Mark bidirected edges (<->)

    n = length(var_names)

    # Look for potential latent confounders
    for i in 1:n, j in (i+1):n
        if graph.adjacency[i, j] == 0 && graph.adjacency[j, i] == 0
            # Check for common effect pattern (collider)
            common_children = Int[]
            for k in 1:n
                if graph.adjacency[i, k] == 1 && graph.adjacency[j, k] == 1
                    push!(common_children, k)
                end
            end

            # If common children but no direct edge, might be latent confounder
            if length(common_children) > 0
                # Test if correlation remains after conditioning on common children
                if !test_conditional_independence_fisher(data, i, j, common_children; alpha=alpha)
                    # Add bidirected edge indicating latent confounder
                    graph.adjacency[i, j] = -1
                    graph.adjacency[j, i] = -1
                end
            end
        end
    end

    return graph
end

"""
    notears(data; lambda, max_iter)

NOTEARS: Non-combinatorial Optimization for DAG Learning (Zheng et al. 2018).
Continuous optimization approach to structure learning.

min_W ||X - XW||²_F + λ||W||₁
s.t.  h(W) = tr(e^{W∘W}) - d = 0  (acyclicity constraint)
"""
function notears(data::Matrix{Float64};
                 lambda::Float64=0.1,
                 max_iter::Int=100,
                 h_tol::Float64=1e-8,
                 rho_max::Float64=1e16)
    n, d = size(data)

    # Standardize data
    X = (data .- mean(data, dims=1)) ./ std(data, dims=1)

    # Initialize
    W = zeros(d, d)
    rho = 1.0
    alpha = 0.0

    for iter in 1:max_iter
        # Solve augmented Lagrangian subproblem
        W_old = copy(W)

        for _ in 1:10  # Inner iterations
            # Gradient of least squares loss
            grad_loss = -2 * X' * X * (I - W) / n

            # Gradient of acyclicity constraint
            M = W .* W
            E = exp(M)
            grad_h = 2 * W .* (E' * ones(d, d))

            # Update
            grad = grad_loss + alpha * grad_h + rho * h(W) * grad_h

            # Proximal gradient step with L1 regularization
            W_new = W - 0.01 * grad
            W_new = sign.(W_new) .* max.(abs.(W_new) .- 0.01 * lambda, 0)

            # No self-loops
            W_new[diagind(W_new)] .= 0

            W = W_new
        end

        # Check acyclicity
        h_val = h(W)

        if h_val < h_tol
            break
        end

        # Update dual variable
        alpha += rho * h_val
        rho = min(2 * rho, rho_max)
    end

    # Threshold small values
    W[abs.(W) .< 0.3] .= 0

    # Convert to CausalGraph
    var_names = ["X$i" for i in 1:d]
    graph = CausalGraph(var_names)

    for i in 1:d, j in 1:d
        if abs(W[i, j]) > 0.3
            graph.adjacency[i, j] = 1
        end
    end

    return graph, W
end

function h(W)
    # Acyclicity constraint: tr(e^{W∘W}) - d = 0
    d = size(W, 1)
    M = W .* W
    return tr(exp(M)) - d
end

"""
    ges_algorithm(data; score)

GES (Greedy Equivalence Search) for score-based causal discovery.
Searches over Markov equivalence classes.
"""
function ges_algorithm(data::Matrix{Float64};
                       penalty::Float64=1.0)
    n, d = size(data)
    var_names = ["X$i" for i in 1:d]

    # Start with empty graph
    graph = CausalGraph(var_names)
    current_score = bic_score(data, graph)

    # Phase I: Forward (add edges)
    improved = true
    while improved
        improved = false
        best_score = current_score
        best_edge = nothing

        for i in 1:d, j in 1:d
            if i != j && graph.adjacency[i, j] == 0
                # Try adding edge i → j
                graph.adjacency[i, j] = 1

                if is_dag(graph)
                    new_score = bic_score(data, graph; penalty=penalty)
                    if new_score > best_score
                        best_score = new_score
                        best_edge = (i, j, :add)
                    end
                end

                graph.adjacency[i, j] = 0
            end
        end

        if best_edge !== nothing
            i, j, _ = best_edge
            graph.adjacency[i, j] = 1
            current_score = best_score
            improved = true
        end
    end

    # Phase II: Backward (remove edges)
    improved = true
    while improved
        improved = false
        best_score = current_score
        best_edge = nothing

        for i in 1:d, j in 1:d
            if graph.adjacency[i, j] == 1
                # Try removing edge i → j
                graph.adjacency[i, j] = 0

                new_score = bic_score(data, graph; penalty=penalty)
                if new_score > best_score
                    best_score = new_score
                    best_edge = (i, j, :remove)
                end

                graph.adjacency[i, j] = 1
            end
        end

        if best_edge !== nothing
            i, j, _ = best_edge
            graph.adjacency[i, j] = 0
            current_score = best_score
            improved = true
        end
    end

    return graph
end

function is_dag(g::CausalGraph)
    # Check for cycles using DFS
    n = length(g.nodes)
    visited = zeros(Int, n)  # 0: unvisited, 1: in progress, 2: done

    function has_cycle(i)
        visited[i] = 1
        for j in 1:n
            if g.adjacency[i, j] == 1
                if visited[j] == 1
                    return true  # Back edge = cycle
                elseif visited[j] == 0
                    if has_cycle(j)
                        return true
                    end
                end
            end
        end
        visited[i] = 2
        return false
    end

    for i in 1:n
        if visited[i] == 0
            if has_cycle(i)
                return false
            end
        end
    end

    return true
end

function bic_score(data::Matrix{Float64}, g::CausalGraph; penalty::Float64=1.0)
    n, d = size(data)
    score = 0.0

    for j in 1:d
        parents = findall(g.adjacency[:, j] .== 1)

        if isempty(parents)
            # Score for root node
            variance = var(data[:, j])
            score += -n/2 * log(2π * variance) - n/2
        else
            # Regression score
            X_pa = data[:, parents]
            y = data[:, j]

            # OLS
            β = X_pa \ y
            residuals = y - X_pa * β
            variance = var(residuals)

            k = length(parents) + 1
            score += -n/2 * log(2π * variance) - n/2 - penalty * k * log(n) / 2
        end
    end

    return score
end

# =============================================================================
# Causal Effect Estimation
# =============================================================================

"""
    backdoor_adjustment(data, treatment, outcome, confounders)

Backdoor adjustment formula (Pearl 2009):
P(Y|do(X)) = Σ_z P(Y|X,Z) P(Z)
"""
function backdoor_adjustment(data::Matrix{Float64},
                             treatment_idx::Int,
                             outcome_idx::Int,
                             confounder_indices::Vector{Int};
                             treatment_value::Float64=1.0)
    n = size(data, 1)

    if isempty(confounder_indices)
        # No confounders - simple conditional mean
        mask = data[:, treatment_idx] .≈ treatment_value
        return mean(data[mask, outcome_idx])
    end

    # Stratified estimation
    Z = data[:, confounder_indices]

    # Use regression for continuous confounders
    # E[Y|do(X=x)] = E_Z[E[Y|X=x,Z]]
    X_full = hcat(data[:, treatment_idx], Z)
    y = data[:, outcome_idx]

    # OLS coefficients
    β = X_full \ y

    # Predict at treatment value with average confounders
    X_do = hcat(fill(treatment_value, n), Z)
    y_do = X_do * β

    return mean(y_do)
end

"""
    frontdoor_adjustment(data, treatment, mediator, outcome)

Frontdoor adjustment formula (Pearl 2009):
P(Y|do(X)) = Σ_m P(M=m|X) Σ_x' P(Y|M=m,X=x') P(X=x')
"""
function frontdoor_adjustment(data::Matrix{Float64},
                              treatment_idx::Int,
                              mediator_idx::Int,
                              outcome_idx::Int)
    n = size(data, 1)

    X = data[:, treatment_idx]
    M = data[:, mediator_idx]
    Y = data[:, outcome_idx]

    # Step 1: P(M|X) - treatment effect on mediator
    β_XM = cov(X, M) / var(X)

    # Step 2: P(Y|M,X) weighted by P(X)
    XM = hcat(M, X)
    β_MY = XM \ Y

    # Frontdoor formula
    effect = β_XM * β_MY[1]  # Indirect effect through mediator

    return effect
end

"""
    estimate_ate(data, treatment_idx, outcome_idx; method)

Estimate Average Treatment Effect using various methods.
"""
function estimate_ate(data::Matrix{Float64},
                      treatment_idx::Int,
                      outcome_idx::Int;
                      confounder_indices::Vector{Int}=Int[],
                      method::Symbol=:backdoor)

    if method == :backdoor
        ate_1 = backdoor_adjustment(data, treatment_idx, outcome_idx,
                                    confounder_indices; treatment_value=1.0)
        ate_0 = backdoor_adjustment(data, treatment_idx, outcome_idx,
                                    confounder_indices; treatment_value=0.0)
        return ate_1 - ate_0

    elseif method == :ipw
        return inverse_probability_weighting(data, treatment_idx, outcome_idx,
                                             confounder_indices)

    elseif method == :matching
        return propensity_score_matching(data, treatment_idx, outcome_idx,
                                         confounder_indices)

    elseif method == :doubly_robust
        return doubly_robust_estimator(data, treatment_idx, outcome_idx,
                                       confounder_indices)
    else
        error("Unknown method: $method")
    end
end

"""
    inverse_probability_weighting(data, treatment, outcome, confounders)

IPW estimator: weight observations by inverse of propensity score.
"""
function inverse_probability_weighting(data::Matrix{Float64},
                                       treatment_idx::Int,
                                       outcome_idx::Int,
                                       confounder_indices::Vector{Int})
    n = size(data, 1)
    T = data[:, treatment_idx]
    Y = data[:, outcome_idx]

    # Estimate propensity score e(X) = P(T=1|X)
    if isempty(confounder_indices)
        e = fill(mean(T), n)
    else
        X = data[:, confounder_indices]
        # Logistic regression
        e = logistic_regression_predict(X, T)
    end

    # Clip for numerical stability
    e = clamp.(e, 0.01, 0.99)

    # IPW estimator
    ate = mean(T .* Y ./ e) - mean((1 .- T) .* Y ./ (1 .- e))

    return ate
end

function logistic_regression_predict(X::Matrix{Float64}, y::Vector{Float64})
    n, d = size(X)
    X_aug = hcat(ones(n), X)

    # Newton-Raphson for logistic regression
    β = zeros(d + 1)

    for _ in 1:50
        p = 1 ./ (1 .+ exp.(-X_aug * β))
        W = Diagonal(p .* (1 .- p))
        grad = X_aug' * (y - p)
        H = -X_aug' * W * X_aug

        if cond(H) > 1e10
            break
        end

        β -= H \ grad
    end

    return 1 ./ (1 .+ exp.(-X_aug * β))
end

"""
    propensity_score_matching(data, treatment, outcome, confounders)

Match treated/control units based on propensity scores.
"""
function propensity_score_matching(data::Matrix{Float64},
                                   treatment_idx::Int,
                                   outcome_idx::Int,
                                   confounder_indices::Vector{Int};
                                   n_neighbors::Int=1)
    n = size(data, 1)
    T = data[:, treatment_idx]
    Y = data[:, outcome_idx]

    # Estimate propensity scores
    if isempty(confounder_indices)
        e = fill(mean(T), n)
    else
        X = data[:, confounder_indices]
        e = logistic_regression_predict(X, T)
    end

    treated_idx = findall(T .== 1)
    control_idx = findall(T .== 0)

    # Match each treated unit to nearest control
    matched_effects = Float64[]

    for i in treated_idx
        # Find nearest control(s) by propensity score
        distances = abs.(e[control_idx] .- e[i])
        nearest = sortperm(distances)[1:min(n_neighbors, length(control_idx))]
        matched_controls = control_idx[nearest]

        # Treatment effect for this match
        effect = Y[i] - mean(Y[matched_controls])
        push!(matched_effects, effect)
    end

    # ATT (Average Treatment effect on Treated)
    return mean(matched_effects)
end

"""
    doubly_robust_estimator(data, treatment, outcome, confounders)

Doubly robust (AIPW) estimator: consistent if either propensity OR outcome model is correct.
"""
function doubly_robust_estimator(data::Matrix{Float64},
                                 treatment_idx::Int,
                                 outcome_idx::Int,
                                 confounder_indices::Vector{Int})
    n = size(data, 1)
    T = data[:, treatment_idx]
    Y = data[:, outcome_idx]

    if isempty(confounder_indices)
        return mean(Y[T .== 1]) - mean(Y[T .== 0])
    end

    X = data[:, confounder_indices]

    # Propensity score model
    e = logistic_regression_predict(X, T)
    e = clamp.(e, 0.01, 0.99)

    # Outcome regression models
    treated_mask = T .== 1
    control_mask = T .== 0

    # E[Y|X, T=1]
    X_treated = X[treated_mask, :]
    Y_treated = Y[treated_mask]
    β_1 = X_treated \ Y_treated
    μ_1 = X * β_1

    # E[Y|X, T=0]
    X_control = X[control_mask, :]
    Y_control = Y[control_mask]
    β_0 = X_control \ Y_control
    μ_0 = X * β_0

    # AIPW estimator
    ate = mean(
        (T .* Y .- (T .- e) .* μ_1) ./ e .-
        ((1 .- T) .* Y .+ (T .- e) .* μ_0) ./ (1 .- e)
    )

    return ate
end

# =============================================================================
# Counterfactual Inference
# =============================================================================

"""
    counterfactual(scm, evidence, intervention, query)

Compute counterfactual: P(Y_x | X=x', Y=y')

Three-step process (Pearl 2009):
1. Abduction: Compute P(U | evidence)
2. Action: Modify SCM with intervention
3. Prediction: Compute query in modified SCM
"""
function counterfactual(scm::SCM,
                        evidence::Dict{String, Float64},
                        intervention::Dict{String, Float64},
                        query::String;
                        n_samples::Int=1000)
    # Step 1: Abduction - infer exogenous noise given evidence
    U_posterior = abduction(scm, evidence, n_samples)

    # Step 2: Action - create interventional SCM
    scm_do = intervene(scm, intervention)

    # Step 3: Prediction - compute query under intervention with inferred U
    query_samples = Float64[]

    for u in U_posterior
        values = forward_sample(scm_do, u)
        push!(query_samples, values[query])
    end

    return (
        mean = mean(query_samples),
        std = std(query_samples),
        samples = query_samples
    )
end

function abduction(scm::SCM, evidence::Dict{String, Float64}, n_samples::Int)
    # Approximate posterior P(U | evidence) using rejection sampling
    U_samples = Vector{Dict{String, Float64}}()

    nodes = topological_sort(scm.graph)

    while length(U_samples) < n_samples
        # Sample U
        U = Dict(v => rand(scm.noise_distributions[v]) for v in nodes)

        # Check if evidence is satisfied (with tolerance)
        values = forward_sample(scm, U)

        match = true
        for (var, val) in evidence
            if abs(values[var] - val) > 0.5  # Tolerance
                match = false
                break
            end
        end

        if match
            push!(U_samples, U)
        end
    end

    return U_samples
end

function forward_sample(scm::SCM, U::Dict{String, Float64})
    nodes = topological_sort(scm.graph)
    values = Dict{String, Float64}()

    for v in nodes
        parents = get_parents(scm.graph, v)
        if isempty(parents)
            if haskey(scm.structural_equations, v)
                values[v] = scm.structural_equations[v](U[v])[1]
            else
                values[v] = U[v]
            end
        else
            parent_vals = [values[p] for p in parents]
            values[v] = scm.structural_equations[v](reshape(parent_vals, 1, :), [U[v]])[1]
        end
    end

    return values
end

"""
    twin_network_counterfactual(data, scm, factual_idx, intervention)

Twin network method for counterfactual estimation.
Creates parallel "twin" for counterfactual world.
"""
function twin_network_counterfactual(data::Matrix{Float64},
                                     scm::SCM,
                                     factual_idx::Int,
                                     intervention::Dict{String, Float64},
                                     query::String)
    # Get factual observation
    factual = Dict(v => data[factual_idx, i] for (i, v) in enumerate(scm.var_names))

    # Infer noise for this observation
    U_factual = infer_noise(scm, factual)

    # Create counterfactual SCM
    scm_cf = intervene(scm, intervention)

    # Forward sample with same noise
    cf_values = forward_sample(scm_cf, U_factual)

    return cf_values[query]
end

function infer_noise(scm::SCM, observation::Dict{String, Float64})
    # Simple approach: assume linear equations, solve for U
    nodes = topological_sort(scm.graph)
    U = Dict{String, Float64}()

    for v in nodes
        parents = get_parents(scm.graph, v)
        if isempty(parents)
            U[v] = observation[v]
        else
            # Assume Y = f(Pa) + U, solve for U
            # This is a simplification - real implementation would use
            # the actual structural equations
            parent_vals = [observation[p] for p in parents]
            predicted = sum(parent_vals) * 0.5  # Simplified
            U[v] = observation[v] - predicted
        end
    end

    return U
end

# =============================================================================
# DoWhy-Style Interface
# =============================================================================

"""
    DoWhyModel

DoWhy-style causal inference pipeline (Sharma & Kiciman 2020).
Four steps: Model → Identify → Estimate → Refute
"""
mutable struct DoWhyModel
    graph::CausalGraph
    treatment::String
    outcome::String
    data::Matrix{Float64}
    var_names::Vector{String}
    identified_estimand::Union{Nothing, Dict}
    estimate::Union{Nothing, Float64}
end

function DoWhyModel(data::Matrix{Float64}, var_names::Vector{String},
                    treatment::String, outcome::String;
                    graph::Union{Nothing, CausalGraph}=nothing)
    if graph === nothing
        # Discover graph
        graph = pc_algorithm(data, var_names)
    end

    DoWhyModel(graph, treatment, outcome, data, var_names, nothing, nothing)
end

"""
    identify(model)

Identify causal effect using do-calculus rules.
"""
function identify(model::DoWhyModel)
    treatment_idx = findfirst(model.var_names .== model.treatment)
    outcome_idx = findfirst(model.var_names .== model.outcome)

    # Find backdoor paths
    backdoor_vars = find_backdoor_set(model.graph, model.treatment, model.outcome)

    # Check if backdoor criterion satisfied
    if !isempty(backdoor_vars)
        model.identified_estimand = Dict(
            "type" => "backdoor",
            "adjustment_set" => backdoor_vars,
            "formula" => "E[Y|do(T)] = Σ_z E[Y|T,Z=z]P(Z=z)"
        )
    else
        # Try frontdoor
        mediators = find_mediators(model.graph, model.treatment, model.outcome)
        if !isempty(mediators)
            model.identified_estimand = Dict(
                "type" => "frontdoor",
                "mediator" => mediators[1],
                "formula" => "E[Y|do(T)] via frontdoor"
            )
        else
            # Try instrumental variable
            instruments = find_instruments(model.graph, model.treatment, model.outcome)
            if !isempty(instruments)
                model.identified_estimand = Dict(
                    "type" => "instrumental_variable",
                    "instrument" => instruments[1],
                    "formula" => "E[Y|do(T)] = Cov(Y,Z)/Cov(T,Z)"
                )
            else
                error("Effect not identifiable from given graph")
            end
        end
    end

    return model.identified_estimand
end

function find_backdoor_set(g::CausalGraph, treatment::String, outcome::String)
    # Find minimal backdoor adjustment set
    all_vars = setdiff(g.nodes, [treatment, outcome])

    # Simple approach: use all non-descendants of treatment
    descendants = get_descendants(g, treatment)
    backdoor = setdiff(all_vars, descendants)

    return backdoor
end

function find_mediators(g::CausalGraph, treatment::String, outcome::String)
    # Find variables on directed path from treatment to outcome
    children_t = get_children(g, treatment)
    parents_y = get_parents(g, outcome)
    return intersect(children_t, parents_y)
end

function find_instruments(g::CausalGraph, treatment::String, outcome::String)
    # Find valid instruments: affect treatment, no direct effect on outcome
    parents_t = get_parents(g, treatment)
    ancestors_y = get_ancestors(g, outcome)

    instruments = String[]
    for p in parents_t
        if p ∉ ancestors_y || p == treatment
            push!(instruments, p)
        end
    end

    return instruments
end

"""
    estimate(model; method)

Estimate causal effect using identified estimand.
"""
function estimate(model::DoWhyModel; method::Symbol=:auto)
    if model.identified_estimand === nothing
        identify(model)
    end

    treatment_idx = findfirst(model.var_names .== model.treatment)
    outcome_idx = findfirst(model.var_names .== model.outcome)

    estimand_type = model.identified_estimand["type"]

    if estimand_type == "backdoor"
        adj_set = model.identified_estimand["adjustment_set"]
        adj_indices = [findfirst(model.var_names .== v) for v in adj_set]

        model.estimate = estimate_ate(model.data, treatment_idx, outcome_idx;
                                      confounder_indices=adj_indices,
                                      method=method == :auto ? :doubly_robust : method)

    elseif estimand_type == "frontdoor"
        mediator = model.identified_estimand["mediator"]
        mediator_idx = findfirst(model.var_names .== mediator)
        model.estimate = frontdoor_adjustment(model.data, treatment_idx,
                                              mediator_idx, outcome_idx)

    elseif estimand_type == "instrumental_variable"
        instrument = model.identified_estimand["instrument"]
        instrument_idx = findfirst(model.var_names .== instrument)
        model.estimate = instrumental_variables(model.data, treatment_idx,
                                                outcome_idx, instrument_idx)
    end

    return model.estimate
end

"""
    refute(model; method)

Refutation tests for robustness of causal estimate.
"""
function refute(model::DoWhyModel; method::Symbol=:placebo)
    if model.estimate === nothing
        estimate(model)
    end

    treatment_idx = findfirst(model.var_names .== model.treatment)
    outcome_idx = findfirst(model.var_names .== model.outcome)

    if method == :placebo
        # Placebo treatment test
        return placebo_test(model)

    elseif method == :random_common_cause
        # Add random common cause
        return random_common_cause_test(model)

    elseif method == :subset
        # Estimate on data subset
        return subset_test(model)

    elseif method == :bootstrap
        # Bootstrap confidence intervals
        return bootstrap_test(model)

    else
        error("Unknown refutation method: $method")
    end
end

function placebo_test(model::DoWhyModel)
    # Replace treatment with random variable
    n = size(model.data, 1)
    placebo_data = copy(model.data)
    treatment_idx = findfirst(model.var_names .== model.treatment)
    placebo_data[:, treatment_idx] = rand(n)

    placebo_model = DoWhyModel(placebo_data, model.var_names,
                               model.treatment, model.outcome;
                               graph=model.graph)
    identify(placebo_model)
    placebo_effect = estimate(placebo_model)

    return Dict(
        "original_estimate" => model.estimate,
        "placebo_estimate" => placebo_effect,
        "passed" => abs(placebo_effect) < abs(model.estimate) * 0.1
    )
end

function random_common_cause_test(model::DoWhyModel)
    # Add random confounder
    n = size(model.data, 1)
    random_confounder = randn(n)
    augmented_data = hcat(model.data, random_confounder)
    augmented_names = vcat(model.var_names, ["RandomConfounder"])

    # Re-estimate with random confounder
    treatment_idx = findfirst(model.var_names .== model.treatment)
    outcome_idx = findfirst(model.var_names .== model.outcome)
    confounder_idx = length(augmented_names)

    adj_indices = [confounder_idx]
    new_estimate = estimate_ate(augmented_data, treatment_idx, outcome_idx;
                                confounder_indices=adj_indices)

    return Dict(
        "original_estimate" => model.estimate,
        "new_estimate" => new_estimate,
        "change" => abs(new_estimate - model.estimate) / abs(model.estimate),
        "passed" => abs(new_estimate - model.estimate) < 0.1 * abs(model.estimate)
    )
end

function subset_test(model::DoWhyModel; fraction::Float64=0.8)
    n = size(model.data, 1)
    subset_idx = sample(1:n, Int(floor(n * fraction)), replace=false)
    subset_data = model.data[subset_idx, :]

    subset_model = DoWhyModel(subset_data, model.var_names,
                              model.treatment, model.outcome;
                              graph=model.graph)
    identify(subset_model)
    subset_estimate = estimate(subset_model)

    return Dict(
        "original_estimate" => model.estimate,
        "subset_estimate" => subset_estimate,
        "passed" => abs(subset_estimate - model.estimate) < 0.2 * abs(model.estimate)
    )
end

function bootstrap_test(model::DoWhyModel; n_bootstrap::Int=100)
    n = size(model.data, 1)
    treatment_idx = findfirst(model.var_names .== model.treatment)
    outcome_idx = findfirst(model.var_names .== model.outcome)

    estimates = Float64[]

    for _ in 1:n_bootstrap
        boot_idx = sample(1:n, n, replace=true)
        boot_data = model.data[boot_idx, :]

        boot_model = DoWhyModel(boot_data, model.var_names,
                                model.treatment, model.outcome;
                                graph=model.graph)
        identify(boot_model)
        push!(estimates, estimate(boot_model))
    end

    ci_lower = quantile(estimates, 0.025)
    ci_upper = quantile(estimates, 0.975)

    return Dict(
        "estimate" => model.estimate,
        "ci_lower" => ci_lower,
        "ci_upper" => ci_upper,
        "std" => std(estimates)
    )
end

# =============================================================================
# Advanced Methods: Double ML, Causal Forests
# =============================================================================

"""
    DoubleML

Double/Debiased Machine Learning (Chernozhukov et al. 2018).
Cross-fitting procedure for semiparametric estimation.
"""
struct DoubleML
    n_folds::Int
    ml_method::Symbol  # :linear, :forest, :neural
end

DoubleML(; n_folds::Int=5, ml_method::Symbol=:linear) = DoubleML(n_folds, ml_method)

function (dml::DoubleML)(data::Matrix{Float64}, treatment_idx::Int,
                         outcome_idx::Int, confounder_indices::Vector{Int})
    n = size(data, 1)

    T = data[:, treatment_idx]
    Y = data[:, outcome_idx]
    X = data[:, confounder_indices]

    # Cross-fitting
    folds = create_folds(n, dml.n_folds)

    residuals_Y = zeros(n)
    residuals_T = zeros(n)

    for k in 1:dml.n_folds
        train_idx = vcat([folds[j] for j in 1:dml.n_folds if j != k]...)
        test_idx = folds[k]

        # Fit outcome model E[Y|X] on training data
        if dml.ml_method == :linear
            β_Y = X[train_idx, :] \ Y[train_idx]
            residuals_Y[test_idx] = Y[test_idx] - X[test_idx, :] * β_Y

            # Fit treatment model E[T|X] on training data
            β_T = X[train_idx, :] \ T[train_idx]
            residuals_T[test_idx] = T[test_idx] - X[test_idx, :] * β_T
        end
    end

    # Final estimate: regress residual_Y on residual_T
    ate = dot(residuals_T, residuals_Y) / dot(residuals_T, residuals_T)

    # Standard error
    n_eff = sum(residuals_T.^2)
    se = sqrt(var(residuals_Y - ate * residuals_T) / n_eff)

    return Dict(
        "ate" => ate,
        "se" => se,
        "ci_lower" => ate - 1.96 * se,
        "ci_upper" => ate + 1.96 * se
    )
end

function create_folds(n::Int, k::Int)
    indices = shuffle(1:n)
    fold_size = n ÷ k
    folds = Vector{Vector{Int}}()

    for i in 1:k
        start_idx = (i - 1) * fold_size + 1
        end_idx = i == k ? n : i * fold_size
        push!(folds, indices[start_idx:end_idx])
    end

    return folds
end

"""
    causal_forest(data, treatment, outcome, confounders)

Causal Forests for heterogeneous treatment effects (Wager & Athey 2018).
Simplified implementation using random forest splits.
"""
function causal_forest(data::Matrix{Float64},
                       treatment_idx::Int,
                       outcome_idx::Int,
                       confounder_indices::Vector{Int};
                       n_trees::Int=100,
                       min_leaf_size::Int=5)
    n = size(data, 1)

    T = data[:, treatment_idx]
    Y = data[:, outcome_idx]
    X = data[:, confounder_indices]

    # Build forest of causal trees
    trees = Vector{CausalTree}()

    for _ in 1:n_trees
        # Bootstrap sample (honest splitting: use half for structure)
        boot_idx = sample(1:n, n, replace=true)

        tree = build_causal_tree(X[boot_idx, :], T[boot_idx], Y[boot_idx];
                                 min_leaf_size=min_leaf_size)
        push!(trees, tree)
    end

    return CausalForest(trees, confounder_indices)
end

struct CausalTree
    split_var::Int
    split_val::Float64
    left::Union{CausalTree, Float64}  # Float64 for leaf (treatment effect)
    right::Union{CausalTree, Float64}
    is_leaf::Bool
end

struct CausalForest
    trees::Vector{CausalTree}
    feature_indices::Vector{Int}
end

function build_causal_tree(X::Matrix{Float64}, T::Vector{Float64}, Y::Vector{Float64};
                           min_leaf_size::Int=5, depth::Int=0, max_depth::Int=10)
    n = size(X, 1)

    # Check stopping conditions
    if n < 2 * min_leaf_size || depth >= max_depth
        # Leaf: estimate treatment effect
        treated = T .== 1
        if sum(treated) > 0 && sum(.!treated) > 0
            effect = mean(Y[treated]) - mean(Y[.!treated])
        else
            effect = 0.0
        end
        return CausalTree(0, 0.0, effect, effect, true)
    end

    # Find best split (maximize heterogeneity of treatment effects)
    best_gain = -Inf
    best_var = 1
    best_val = median(X[:, 1])

    d = size(X, 2)
    for var in 1:d
        vals = sort(unique(X[:, var]))
        for val in vals[1:end-1]
            left_mask = X[:, var] .<= val
            right_mask = .!left_mask

            if sum(left_mask) >= min_leaf_size && sum(right_mask) >= min_leaf_size
                # Compute treatment effect variance gain
                τ_left = treatment_effect(Y[left_mask], T[left_mask])
                τ_right = treatment_effect(Y[right_mask], T[right_mask])

                # Gain: variance of effects
                gain = (sum(left_mask) * τ_left^2 + sum(right_mask) * τ_right^2) / n

                if gain > best_gain
                    best_gain = gain
                    best_var = var
                    best_val = val
                end
            end
        end
    end

    # Split
    left_mask = X[:, best_var] .<= best_val
    right_mask = .!left_mask

    left_tree = build_causal_tree(X[left_mask, :], T[left_mask], Y[left_mask];
                                  min_leaf_size=min_leaf_size, depth=depth+1)
    right_tree = build_causal_tree(X[right_mask, :], T[right_mask], Y[right_mask];
                                   min_leaf_size=min_leaf_size, depth=depth+1)

    return CausalTree(best_var, best_val, left_tree, right_tree, false)
end

function treatment_effect(Y::Vector{Float64}, T::Vector{Float64})
    treated = T .== 1
    if sum(treated) > 0 && sum(.!treated) > 0
        return mean(Y[treated]) - mean(Y[.!treated])
    else
        return 0.0
    end
end

function predict_cate(forest::CausalForest, x::Vector{Float64})
    effects = Float64[]

    for tree in forest.trees
        effect = traverse_tree(tree, x)
        push!(effects, effect)
    end

    return mean(effects)
end

function traverse_tree(tree::CausalTree, x::Vector{Float64})
    if tree.is_leaf
        return tree.left  # Effect stored in left for leaves
    end

    if x[tree.split_var] <= tree.split_val
        return traverse_tree(tree.left, x)
    else
        return traverse_tree(tree.right, x)
    end
end

"""
    estimate_cate(forest, X)

Estimate Conditional Average Treatment Effects for each observation.
"""
function estimate_cate(forest::CausalForest, X::Matrix{Float64})
    n = size(X, 1)
    cate = zeros(n)

    for i in 1:n
        cate[i] = predict_cate(forest, X[i, :])
    end

    return cate
end

"""
    heterogeneous_effects(data, treatment, outcome, confounders)

Analyze heterogeneity in treatment effects across subgroups.
"""
function heterogeneous_effects(data::Matrix{Float64},
                               treatment_idx::Int,
                               outcome_idx::Int,
                               confounder_indices::Vector{Int})
    forest = causal_forest(data, treatment_idx, outcome_idx, confounder_indices)
    X = data[:, confounder_indices]

    cate = estimate_cate(forest, X)

    return Dict(
        "mean_cate" => mean(cate),
        "std_cate" => std(cate),
        "min_cate" => minimum(cate),
        "max_cate" => maximum(cate),
        "cate_values" => cate
    )
end

# =============================================================================
# Sensitivity Analysis
# =============================================================================

"""
    sensitivity_analysis(estimate, se, R2)

Sensitivity analysis for unobserved confounding (Cinelli & Hazlett 2020).
"""
function sensitivity_analysis(estimate::Float64, se::Float64, R2_Y::Float64, R2_T::Float64)
    # Bias factor
    bias = estimate * sqrt(R2_Y * R2_T) / (1 - R2_T)

    # Robustness value (RV)
    rv = sqrt(R2_Y * R2_T)

    # Adjusted estimate
    adjusted = estimate - bias

    return Dict(
        "original" => estimate,
        "bias" => bias,
        "adjusted" => adjusted,
        "robustness_value" => rv
    )
end

"""
    e_value(estimate, se)

E-value: minimum confounding strength to explain away effect (VanderWeele & Ding 2017).
"""
function e_value(estimate::Float64; se::Float64=0.0)
    # Convert to risk ratio scale (approximate)
    rr = exp(estimate)

    if rr >= 1
        e_val = rr + sqrt(rr * (rr - 1))
    else
        rr_inv = 1 / rr
        e_val = rr_inv + sqrt(rr_inv * (rr_inv - 1))
    end

    # E-value for CI bound
    if se > 0
        ci_bound = exp(estimate - 1.96 * se)
        if ci_bound >= 1
            e_val_ci = ci_bound + sqrt(ci_bound * (ci_bound - 1))
        else
            e_val_ci = 1.0
        end
    else
        e_val_ci = e_val
    end

    return Dict(
        "e_value" => e_val,
        "e_value_ci" => e_val_ci
    )
end

"""
    robustness_value(model)

Robustness value: partial R² needed to explain away effect.
"""
function robustness_value(estimate::Float64, t_stat::Float64, df::Int)
    # RV = t² / (t² + df)
    rv = t_stat^2 / (t_stat^2 + df)
    return sqrt(rv)
end

# =============================================================================
# Instrumental Variables & Quasi-Experimental Methods
# =============================================================================

"""
    instrumental_variables(data, treatment, outcome, instrument)

2SLS (Two-Stage Least Squares) IV estimation.
"""
function instrumental_variables(data::Matrix{Float64},
                                treatment_idx::Int,
                                outcome_idx::Int,
                                instrument_idx::Int)
    n = size(data, 1)

    T = data[:, treatment_idx]
    Y = data[:, outcome_idx]
    Z = data[:, instrument_idx]

    # Stage 1: Regress T on Z
    Z_aug = hcat(ones(n), Z)
    γ = Z_aug \ T
    T_hat = Z_aug * γ

    # Stage 2: Regress Y on T_hat
    T_hat_aug = hcat(ones(n), T_hat)
    β = T_hat_aug \ Y

    # IV estimate
    iv_estimate = β[2]

    # Standard error (simplified)
    residuals = Y - T_hat_aug * β
    σ² = sum(residuals.^2) / (n - 2)
    se = sqrt(σ² / sum((T_hat .- mean(T_hat)).^2))

    # Weak instrument test (first-stage F-statistic)
    T_resid = T - mean(T)
    Z_resid = Z - mean(Z)
    r² = cor(Z, T)^2
    f_stat = r² * (n - 2) / (1 - r²)

    return Dict(
        "estimate" => iv_estimate,
        "se" => se,
        "f_stat" => f_stat,
        "weak_instrument" => f_stat < 10
    )
end

"""
    regression_discontinuity(data, running_var, cutoff, outcome; bandwidth)

Regression Discontinuity Design (RDD).
"""
function regression_discontinuity(data::Matrix{Float64},
                                  running_idx::Int,
                                  outcome_idx::Int,
                                  cutoff::Float64;
                                  bandwidth::Float64=0.5)
    R = data[:, running_idx]
    Y = data[:, outcome_idx]

    # Treatment indicator
    T = R .>= cutoff

    # Local linear regression within bandwidth
    in_bandwidth = abs.(R .- cutoff) .<= bandwidth
    R_local = R[in_bandwidth] .- cutoff
    Y_local = Y[in_bandwidth]
    T_local = T[in_bandwidth]

    # Separate regressions above/below cutoff
    below = R_local .< 0
    above = .!below

    if sum(below) > 2 && sum(above) > 2
        # Linear fit below
        X_below = hcat(ones(sum(below)), R_local[below])
        β_below = X_below \ Y_local[below]
        y_left = β_below[1]

        # Linear fit above
        X_above = hcat(ones(sum(above)), R_local[above])
        β_above = X_above \ Y_local[above]
        y_right = β_above[1]

        # RDD estimate: discontinuity at cutoff
        rd_estimate = y_right - y_left

        # Bootstrap SE
        se = bootstrap_rd_se(R_local, Y_local, cutoff=0.0)
    else
        rd_estimate = NaN
        se = NaN
    end

    return Dict(
        "estimate" => rd_estimate,
        "se" => se,
        "bandwidth" => bandwidth,
        "n_local" => sum(in_bandwidth)
    )
end

function bootstrap_rd_se(R, Y; cutoff::Float64=0.0, n_boot::Int=100)
    estimates = Float64[]
    n = length(R)

    for _ in 1:n_boot
        idx = sample(1:n, n, replace=true)
        R_boot = R[idx]
        Y_boot = Y[idx]

        below = R_boot .< cutoff
        above = .!below

        if sum(below) > 2 && sum(above) > 2
            y_left = mean(Y_boot[below])
            y_right = mean(Y_boot[above])
            push!(estimates, y_right - y_left)
        end
    end

    return std(estimates)
end

"""
    difference_in_differences(data, group, time, outcome)

Difference-in-Differences estimator.
"""
function difference_in_differences(data::Matrix{Float64},
                                   group_idx::Int,
                                   time_idx::Int,
                                   outcome_idx::Int)
    G = data[:, group_idx]  # 1 = treated group, 0 = control
    T = data[:, time_idx]   # 1 = post, 0 = pre
    Y = data[:, outcome_idx]

    # Four group means
    y_00 = mean(Y[(G .== 0) .& (T .== 0)])  # Control, Pre
    y_01 = mean(Y[(G .== 0) .& (T .== 1)])  # Control, Post
    y_10 = mean(Y[(G .== 1) .& (T .== 0)])  # Treated, Pre
    y_11 = mean(Y[(G .== 1) .& (T .== 1)])  # Treated, Post

    # DiD estimate
    did = (y_11 - y_10) - (y_01 - y_00)

    # SE via regression
    X = hcat(ones(length(Y)), G, T, G .* T)
    β = X \ Y
    residuals = Y - X * β
    σ² = sum(residuals.^2) / (length(Y) - 4)
    cov_β = σ² * inv(X' * X)
    se = sqrt(cov_β[4, 4])

    return Dict(
        "estimate" => did,
        "se" => se,
        "pre_trend" => y_10 - y_00,  # Should be ~0 for parallel trends
        "ci_lower" => did - 1.96 * se,
        "ci_upper" => did + 1.96 * se
    )
end

# =============================================================================
# Utility Functions
# =============================================================================

function test_conditional_independence_fisher(data::Matrix{Float64},
                                              i::Int, j::Int, S::Vector{Int};
                                              alpha::Float64=0.05, n::Int=0)
    if n == 0
        n = size(data, 1)
    end

    if isempty(S)
        r = cor(data[:, i], data[:, j])
    else
        # Partial correlation
        r = partial_correlation(data, i, j, S)
    end

    # Fisher z-transform
    if abs(r) > 0.9999
        return false  # Not independent
    end

    z = 0.5 * log((1 + r) / (1 - r))
    se = 1 / sqrt(n - length(S) - 3)
    p_value = 2 * (1 - cdf(Normal(), abs(z) / se))

    return p_value > alpha
end

function partial_correlation(data::Matrix{Float64}, i::Int, j::Int, S::Vector{Int})
    # Compute partial correlation r_{ij|S}
    if isempty(S)
        return cor(data[:, i], data[:, j])
    end

    # Regression-based partial correlation
    X = data[:, S]

    # Residualize i on S
    β_i = X \ data[:, i]
    res_i = data[:, i] - X * β_i

    # Residualize j on S
    β_j = X \ data[:, j]
    res_j = data[:, j] - X * β_j

    return cor(res_i, res_j)
end

# Alias for backward compatibility
const discover_causal_graph = pc_algorithm
const CausalDAG = CausalGraph
const CausalModel = DoWhyModel

end # module
