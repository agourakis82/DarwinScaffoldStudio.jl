# AblationFramework.jl - Systematic Ablation Studies
# Framework for evaluating component importance in models and pipelines
# Essential for understanding model behavior and scientific rigor

module AblationFramework

using LinearAlgebra
using Statistics
using Random

export AblationStudy, AblationResult, ComponentAblation
export feature_ablation, architecture_ablation, physics_constraint_ablation
export layer_ablation, module_ablation, hyperparameter_ablation
export generate_ablation_report, ablation_importance_ranking
export PermutationImportance, DropoutAblation, SubstitutionAblation

# ============================================================================
# Ablation Types
# ============================================================================

"""
Abstract type for ablation methods.
"""
abstract type AblationMethod end

"""
Remove component entirely.
"""
struct DropoutAblation <: AblationMethod end

"""
Replace component with random permutation.
"""
struct PermutationImportance <: AblationMethod
    n_permutations::Int

    PermutationImportance(; n_permutations::Int = 10) = new(n_permutations)
end

"""
Replace component with constant or baseline.
"""
struct SubstitutionAblation <: AblationMethod
    substitute::Union{Float64, Vector{Float64}, Nothing}

    SubstitutionAblation(; substitute = nothing) = new(substitute)
end

"""
    ComponentAblation

Definition of a component to ablate.
"""
struct ComponentAblation
    name::String
    component_type::Symbol  # :feature, :layer, :module, :constraint, :hyperparameter
    indices::Union{Vector{Int}, Nothing}
    description::String

    function ComponentAblation(name::String; component_type::Symbol = :feature,
                               indices::Union{Vector{Int}, Nothing} = nothing,
                               description::String = "")
        new(name, component_type, indices, description)
    end
end

# ============================================================================
# Ablation Result
# ============================================================================

"""
    AblationResult

Result from a single ablation experiment.
"""
struct AblationResult
    component::ComponentAblation
    baseline_score::Float64
    ablated_score::Float64
    delta::Float64
    relative_delta::Float64
    method::AblationMethod
    additional_metrics::Dict{Symbol, Any}

    function AblationResult(component::ComponentAblation, baseline::Float64,
                           ablated::Float64, method::AblationMethod;
                           additional_metrics::Dict{Symbol, Any} = Dict{Symbol, Any}())
        delta = baseline - ablated
        rel_delta = baseline != 0 ? delta / abs(baseline) : 0.0
        new(component, baseline, ablated, delta, rel_delta, method, additional_metrics)
    end
end

"""
    AblationStudy

Complete ablation study results.
"""
struct AblationStudy
    name::String
    results::Vector{AblationResult}
    baseline_score::Float64
    timestamp::String
    config::Dict{Symbol, Any}
end

function AblationStudy(name::String, results::Vector{AblationResult}, baseline::Float64;
                       config::Dict{Symbol, Any} = Dict{Symbol, Any}())
    AblationStudy(name, results, baseline, string(now_string()), config)
end

function now_string()
    # Simple timestamp without Dates dependency
    time_ns = time_ns()
    return "ablation_$(time_ns)"
end

# ============================================================================
# Feature Ablation
# ============================================================================

"""
    feature_ablation(X, y, model_fn, score_fn, feature_names; method, cv_folds)

Perform systematic feature ablation study.

# Arguments
- `X::Matrix`: Feature matrix (n_samples × n_features)
- `y::Vector`: Target vector
- `model_fn::Function`: (X, y) -> trained model
- `score_fn::Function`: (model, X, y) -> score (higher is better)
- `feature_names::Vector{String}`: Names of features
- `method::AblationMethod`: How to ablate features
- `cv_folds::Int`: Number of CV folds

# Returns
AblationStudy with results for each feature.
"""
function feature_ablation(X::Matrix{Float64}, y::Vector,
                          model_fn::Function, score_fn::Function,
                          feature_names::Vector{String};
                          method::AblationMethod = PermutationImportance(),
                          cv_folds::Int = 5, verbose::Bool = false)
    n_samples, n_features = size(X)
    @assert length(feature_names) == n_features

    # Compute baseline score
    baseline_score = cv_score(X, y, model_fn, score_fn, cv_folds)

    if verbose
        println("Baseline score: $baseline_score")
        println("Starting feature ablation for $n_features features...")
    end

    results = AblationResult[]

    for (j, fname) in enumerate(feature_names)
        if verbose
            println("  Ablating feature $j/$n_features: $fname")
        end

        component = ComponentAblation(fname, component_type = :feature,
                                       indices = [j])

        if method isa PermutationImportance
            # Permutation importance
            perm_scores = Float64[]
            for _ in 1:method.n_permutations
                X_perm = copy(X)
                X_perm[:, j] = shuffle(X[:, j])
                push!(perm_scores, cv_score(X_perm, y, model_fn, score_fn, cv_folds))
            end
            ablated_score = mean(perm_scores)
            additional = Dict{Symbol, Any}(
                :perm_scores => perm_scores,
                :perm_std => std(perm_scores)
            )

        elseif method isa SubstitutionAblation
            # Replace with constant
            X_sub = copy(X)
            if method.substitute !== nothing
                X_sub[:, j] .= method.substitute
            else
                X_sub[:, j] .= mean(X[:, j])  # Default to mean
            end
            ablated_score = cv_score(X_sub, y, model_fn, score_fn, cv_folds)
            additional = Dict{Symbol, Any}()

        elseif method isa DropoutAblation
            # Remove feature entirely
            keep_cols = setdiff(1:n_features, j)
            X_drop = X[:, keep_cols]
            ablated_score = cv_score(X_drop, y, model_fn, score_fn, cv_folds)
            additional = Dict{Symbol, Any}()

        else
            error("Unknown ablation method: $method")
        end

        push!(results, AblationResult(component, baseline_score, ablated_score,
                                      method, additional_metrics = additional))
    end

    return AblationStudy("Feature Ablation", results, baseline_score)
end

"""
Helper function for cross-validated scoring.
"""
function cv_score(X::Matrix{Float64}, y::Vector,
                  model_fn::Function, score_fn::Function, n_folds::Int)
    n = size(X, 1)
    fold_size = n ÷ n_folds
    scores = Float64[]

    indices = shuffle(1:n)

    for k in 1:n_folds
        test_start = (k - 1) * fold_size + 1
        test_end = k == n_folds ? n : k * fold_size
        test_idx = indices[test_start:test_end]
        train_idx = setdiff(indices, test_idx)

        model = model_fn(X[train_idx, :], y[train_idx])
        push!(scores, score_fn(model, X[test_idx, :], y[test_idx]))
    end

    return mean(scores)
end

# ============================================================================
# Architecture Ablation
# ============================================================================

"""
    ArchitectureComponent

Component of a neural network architecture.
"""
struct ArchitectureComponent
    name::String
    layer_indices::Vector{Int}
    enabled::Bool
end

"""
    architecture_ablation(create_model_fn, train_fn, eval_fn, components; kwargs...)

Ablate architectural components of a neural network.

# Arguments
- `create_model_fn::Function`: (components) -> model (creates model with given components)
- `train_fn::Function`: (model, data) -> trained_model
- `eval_fn::Function`: (model, data) -> score
- `components::Vector{ArchitectureComponent}`: Components to potentially ablate
- `data`: Training/evaluation data
"""
function architecture_ablation(create_model_fn::Function,
                               train_fn::Function,
                               eval_fn::Function,
                               components::Vector{String},
                               data; verbose::Bool = false)
    # Baseline with all components
    all_enabled = Dict(c => true for c in components)
    baseline_model = create_model_fn(all_enabled)
    trained_baseline = train_fn(baseline_model, data)
    baseline_score = eval_fn(trained_baseline, data)

    if verbose
        println("Baseline score (all components): $baseline_score")
    end

    results = AblationResult[]

    for component_name in components
        if verbose
            println("  Ablating: $component_name")
        end

        # Disable this component
        config = Dict(c => (c != component_name) for c in components)
        ablated_model = create_model_fn(config)
        trained_ablated = train_fn(ablated_model, data)
        ablated_score = eval_fn(trained_ablated, data)

        comp = ComponentAblation(component_name, component_type = :module)
        push!(results, AblationResult(comp, baseline_score, ablated_score, DropoutAblation()))
    end

    return AblationStudy("Architecture Ablation", results, baseline_score)
end

"""
    layer_ablation(model_layers, train_fn, eval_fn, data; kwargs...)

Ablate individual layers of a neural network.
"""
function layer_ablation(layer_names::Vector{String},
                        create_model_fn::Function,
                        train_fn::Function,
                        eval_fn::Function,
                        data; skip_patterns::Vector{String} = String[],
                        verbose::Bool = false)
    # Baseline
    all_layers = Dict(l => true for l in layer_names)
    baseline_model = create_model_fn(all_layers)
    trained_baseline = train_fn(baseline_model, data)
    baseline_score = eval_fn(trained_baseline, data)

    if verbose
        println("Baseline score: $baseline_score")
    end

    results = AblationResult[]

    for layer_name in layer_names
        # Skip certain layers
        skip = false
        for pattern in skip_patterns
            if occursin(pattern, layer_name)
                skip = true
                break
            end
        end
        if skip
            continue
        end

        if verbose
            println("  Ablating layer: $layer_name")
        end

        config = Dict(l => (l != layer_name) for l in layer_names)
        ablated_model = create_model_fn(config)
        trained_ablated = train_fn(ablated_model, data)
        ablated_score = eval_fn(trained_ablated, data)

        comp = ComponentAblation(layer_name, component_type = :layer)
        push!(results, AblationResult(comp, baseline_score, ablated_score, DropoutAblation()))
    end

    return AblationStudy("Layer Ablation", results, baseline_score)
end

# ============================================================================
# Physics Constraint Ablation
# ============================================================================

"""
    physics_constraint_ablation(constraints, train_fn, eval_fn, data; kwargs...)

Ablate physics-informed constraints (for PINNs).

# Arguments
- `constraints::Vector{String}`: Names of physics constraints
- `train_fn::Function`: (enabled_constraints::Dict) -> trained_model
- `eval_fn::Function`: (model, constraint_name::String) -> Dict with metrics
"""
function physics_constraint_ablation(constraints::Vector{String},
                                     train_fn::Function,
                                     eval_fn::Function,
                                     data; verbose::Bool = false)
    # Baseline with all constraints
    all_enabled = Dict(c => true for c in constraints)
    baseline_model = train_fn(all_enabled, data)

    # Evaluate each constraint satisfaction for baseline
    baseline_metrics = Dict{String, Float64}()
    total_baseline = 0.0
    for c in constraints
        metrics = eval_fn(baseline_model, c)
        baseline_metrics[c] = get(metrics, :constraint_error, 0.0)
        total_baseline += get(metrics, :prediction_score, 0.0)
    end
    baseline_score = total_baseline / length(constraints)

    if verbose
        println("Baseline prediction score: $baseline_score")
        println("Constraint errors: $baseline_metrics")
    end

    results = AblationResult[]

    for constraint in constraints
        if verbose
            println("  Ablating constraint: $constraint")
        end

        # Train without this constraint
        config = Dict(c => (c != constraint) for c in constraints)
        ablated_model = train_fn(config, data)

        # Evaluate
        ablated_total = 0.0
        ablated_constraint_error = 0.0
        for c in constraints
            metrics = eval_fn(ablated_model, c)
            ablated_total += get(metrics, :prediction_score, 0.0)
            if c == constraint
                ablated_constraint_error = get(metrics, :constraint_error, 0.0)
            end
        end
        ablated_score = ablated_total / length(constraints)

        comp = ComponentAblation(constraint, component_type = :constraint,
                                 description = "Physics constraint: $constraint")
        additional = Dict{Symbol, Any}(
            :constraint_error_with => baseline_metrics[constraint],
            :constraint_error_without => ablated_constraint_error
        )
        push!(results, AblationResult(comp, baseline_score, ablated_score,
                                      DropoutAblation(), additional_metrics = additional))
    end

    return AblationStudy("Physics Constraint Ablation", results, baseline_score)
end

# ============================================================================
# Hyperparameter Ablation
# ============================================================================

"""
    hyperparameter_ablation(base_config, hyperparam_variants, train_fn, eval_fn, data)

Study sensitivity to hyperparameter choices.

# Arguments
- `base_config::Dict`: Baseline hyperparameter configuration
- `hyperparam_variants::Dict{String, Vector}`: Variants to try for each hyperparam
- `train_fn::Function`: (config, data) -> model
- `eval_fn::Function`: (model, data) -> score
"""
function hyperparameter_ablation(base_config::Dict{String, Any},
                                 hyperparam_variants::Dict{String, Vector},
                                 train_fn::Function,
                                 eval_fn::Function,
                                 data; verbose::Bool = false)
    # Baseline
    baseline_model = train_fn(base_config, data)
    baseline_score = eval_fn(baseline_model, data)

    if verbose
        println("Baseline config: $base_config")
        println("Baseline score: $baseline_score")
    end

    results = AblationResult[]

    for (param_name, variants) in hyperparam_variants
        base_value = base_config[param_name]

        for variant_value in variants
            if variant_value == base_value
                continue
            end

            if verbose
                println("  Testing $param_name = $variant_value (base: $base_value)")
            end

            # Create variant config
            variant_config = copy(base_config)
            variant_config[param_name] = variant_value

            variant_model = train_fn(variant_config, data)
            variant_score = eval_fn(variant_model, data)

            comp = ComponentAblation(
                "$param_name: $base_value → $variant_value",
                component_type = :hyperparameter,
                description = "Changed $param_name from $base_value to $variant_value"
            )
            additional = Dict{Symbol, Any}(
                :param_name => param_name,
                :base_value => base_value,
                :variant_value => variant_value
            )
            push!(results, AblationResult(comp, baseline_score, variant_score,
                                          SubstitutionAblation(),
                                          additional_metrics = additional))
        end
    end

    return AblationStudy("Hyperparameter Ablation", results, baseline_score)
end

# ============================================================================
# Analysis and Reporting
# ============================================================================

"""
    ablation_importance_ranking(study::AblationStudy; metric)

Rank components by importance based on ablation results.
"""
function ablation_importance_ranking(study::AblationStudy; metric::Symbol = :delta)
    rankings = Dict{String, Float64}()

    for result in study.results
        if metric == :delta
            rankings[result.component.name] = result.delta
        elseif metric == :relative_delta
            rankings[result.component.name] = result.relative_delta
        elseif metric == :ablated_score
            rankings[result.component.name] = result.ablated_score
        end
    end

    # Sort by importance (larger delta = more important)
    sorted = sort(collect(rankings), by = x -> -x.second)

    return [(name = p.first, importance = p.second) for p in sorted]
end

"""
    generate_ablation_report(study::AblationStudy; format)

Generate publication-ready ablation report.
"""
function generate_ablation_report(study::AblationStudy; format::Symbol = :markdown)
    rankings = ablation_importance_ranking(study)

    if format == :markdown
        return _generate_markdown_report(study, rankings)
    elseif format == :latex
        return _generate_latex_report(study, rankings)
    elseif format == :dict
        return _generate_dict_report(study, rankings)
    else
        error("Unknown format: $format")
    end
end

function _generate_markdown_report(study::AblationStudy, rankings)
    lines = String[]

    push!(lines, "# Ablation Study: $(study.name)")
    push!(lines, "")
    push!(lines, "**Baseline Score:** $(round(study.baseline_score, digits=4))")
    push!(lines, "")
    push!(lines, "## Component Importance Ranking")
    push!(lines, "")
    push!(lines, "| Rank | Component | Δ Score | Relative Δ (%) |")
    push!(lines, "|------|-----------|---------|----------------|")

    for (rank, item) in enumerate(rankings)
        result = first(filter(r -> r.component.name == item.name, study.results))
        rel_pct = round(result.relative_delta * 100, digits=2)
        delta_str = round(result.delta, digits=4)
        push!(lines, "| $rank | $(item.name) | $delta_str | $rel_pct% |")
    end

    push!(lines, "")
    push!(lines, "## Key Findings")
    push!(lines, "")

    # Top 3 most important
    if length(rankings) >= 3
        push!(lines, "**Most Critical Components:**")
        for i in 1:min(3, length(rankings))
            push!(lines, "- $(rankings[i].name): Δ = $(round(rankings[i].importance, digits=4))")
        end
    end

    push!(lines, "")
    push!(lines, "---")
    push!(lines, "*Generated by DarwinScaffoldStudio AblationFramework*")

    return join(lines, "\n")
end

function _generate_latex_report(study::AblationStudy, rankings)
    lines = String[]

    push!(lines, "\\begin{table}[htbp]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{$(study.name) (Baseline: $(round(study.baseline_score, digits=4)))}")
    push!(lines, "\\begin{tabular}{clcc}")
    push!(lines, "\\hline")
    push!(lines, "Rank & Component & \\Delta Score & Relative \\Delta (\\%) \\\\")
    push!(lines, "\\hline")

    for (rank, item) in enumerate(rankings)
        result = first(filter(r -> r.component.name == item.name, study.results))
        rel_pct = round(result.relative_delta * 100, digits=2)
        delta_str = round(result.delta, digits=4)
        name_escaped = replace(item.name, "_" => "\\_")
        push!(lines, "$rank & $name_escaped & $delta_str & $rel_pct \\\\")
    end

    push!(lines, "\\hline")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n")
end

function _generate_dict_report(study::AblationStudy, rankings)
    return Dict(
        :study_name => study.name,
        :baseline_score => study.baseline_score,
        :n_components => length(study.results),
        :rankings => rankings,
        :results => [(
            component = r.component.name,
            baseline = r.baseline_score,
            ablated = r.ablated_score,
            delta = r.delta,
            relative_delta = r.relative_delta
        ) for r in study.results],
        :most_important => length(rankings) > 0 ? rankings[1].name : nothing,
        :least_important => length(rankings) > 0 ? rankings[end].name : nothing
    )
end

# ============================================================================
# Scaffold-Specific Ablation
# ============================================================================

"""
    scaffold_feature_groups()

Standard feature groups for scaffold analysis.
"""
function scaffold_feature_groups()
    Dict(
        "Structural" => [:porosity, :pore_size, :interconnectivity, :strut_thickness],
        "Mechanical" => [:tensile_modulus, :compressive_modulus, :yield_strength],
        "Surface" => [:surface_area, :roughness, :hydrophilicity],
        "Chemical" => [:molecular_weight, :crystallinity, :crosslinking],
        "Biological" => [:cell_attachment, :proliferation, :differentiation]
    )
end

"""
    grouped_feature_ablation(X, y, model_fn, score_fn, feature_names, groups; kwargs...)

Ablate feature groups rather than individual features.
"""
function grouped_feature_ablation(X::Matrix{Float64}, y::Vector,
                                  model_fn::Function, score_fn::Function,
                                  feature_names::Vector{String},
                                  groups::Dict{String, Vector{Symbol}};
                                  method::AblationMethod = PermutationImportance(),
                                  cv_folds::Int = 5, verbose::Bool = false)
    n_features = size(X, 2)

    # Map feature names to indices
    name_to_idx = Dict(name => i for (i, name) in enumerate(feature_names))

    # Baseline score
    baseline_score = cv_score(X, y, model_fn, score_fn, cv_folds)

    if verbose
        println("Baseline score: $baseline_score")
    end

    results = AblationResult[]

    for (group_name, group_features) in groups
        # Get indices for this group
        group_indices = Int[]
        for feat in group_features
            feat_str = string(feat)
            if haskey(name_to_idx, feat_str)
                push!(group_indices, name_to_idx[feat_str])
            end
        end

        if isempty(group_indices)
            continue
        end

        if verbose
            println("  Ablating group: $group_name ($(length(group_indices)) features)")
        end

        if method isa PermutationImportance
            perm_scores = Float64[]
            for _ in 1:method.n_permutations
                X_perm = copy(X)
                for idx in group_indices
                    X_perm[:, idx] = shuffle(X[:, idx])
                end
                push!(perm_scores, cv_score(X_perm, y, model_fn, score_fn, cv_folds))
            end
            ablated_score = mean(perm_scores)
            additional = Dict{Symbol, Any}(:perm_std => std(perm_scores))

        elseif method isa DropoutAblation
            keep_cols = setdiff(1:n_features, group_indices)
            X_drop = X[:, keep_cols]
            ablated_score = cv_score(X_drop, y, model_fn, score_fn, cv_folds)
            additional = Dict{Symbol, Any}()

        else
            error("Unknown method for grouped ablation")
        end

        comp = ComponentAblation(group_name, component_type = :feature,
                                 indices = group_indices,
                                 description = "Feature group: $group_name")
        push!(results, AblationResult(comp, baseline_score, ablated_score,
                                      method, additional_metrics = additional))
    end

    return AblationStudy("Grouped Feature Ablation", results, baseline_score)
end

# ============================================================================
# Statistical Significance
# ============================================================================

"""
    ablation_significance_test(study, n_bootstrap; confidence_level)

Bootstrap confidence intervals for ablation importance.
"""
function ablation_significance_test(deltas::Vector{Float64};
                                    n_bootstrap::Int = 1000,
                                    confidence_level::Float64 = 0.95)
    n = length(deltas)
    bootstrap_means = Float64[]

    for _ in 1:n_bootstrap
        resample = rand(deltas, n)
        push!(bootstrap_means, mean(resample))
    end

    sort!(bootstrap_means)

    alpha = 1 - confidence_level
    lower_idx = max(1, round(Int, alpha / 2 * n_bootstrap))
    upper_idx = min(n_bootstrap, round(Int, (1 - alpha / 2) * n_bootstrap))

    return (
        mean = mean(deltas),
        std = std(deltas),
        ci_lower = bootstrap_means[lower_idx],
        ci_upper = bootstrap_means[upper_idx],
        significant = bootstrap_means[lower_idx] > 0 || bootstrap_means[upper_idx] < 0
    )
end

end # module
