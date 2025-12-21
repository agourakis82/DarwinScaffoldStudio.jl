# CrossValidation.jl - Cross-Validation Framework
# Comprehensive CV strategies for robust model evaluation
# Includes stratified, nested, and time-series aware cross-validation

module CrossValidation

using LinearAlgebra
using Statistics
using Random

export CVStrategy, KFoldCV, LeaveOneOut, NestedCV, TimeSeriesCV
export GroupKFold, StratifiedKFold, RepeatedKFold
export create_splits, get_fold, n_splits
export cross_validate, nested_cross_validate
export distribution_shift_detection, compute_mmd, ks_test_2sample
export CVResult, aggregate_cv_results

# ============================================================================
# CV Strategy Types
# ============================================================================

"""
Abstract type for cross-validation strategies.
"""
abstract type CVStrategy end

"""
    KFoldCV

Standard K-Fold cross-validation.

# Fields
- `n_splits::Int`: Number of folds
- `shuffle::Bool`: Whether to shuffle before splitting
- `seed::Union{Int, Nothing}`: Random seed for reproducibility
"""
struct KFoldCV <: CVStrategy
    n_splits::Int
    shuffle::Bool
    seed::Union{Int, Nothing}

    function KFoldCV(; n_splits::Int = 5, shuffle::Bool = true, seed::Union{Int, Nothing} = nothing)
        @assert n_splits >= 2 "n_splits must be at least 2"
        new(n_splits, shuffle, seed)
    end
end

"""
    StratifiedKFold

Stratified K-Fold maintaining class proportions.

# Fields
- `n_splits::Int`: Number of folds
- `shuffle::Bool`: Whether to shuffle within strata
- `seed::Union{Int, Nothing}`: Random seed
"""
struct StratifiedKFold <: CVStrategy
    n_splits::Int
    shuffle::Bool
    seed::Union{Int, Nothing}

    function StratifiedKFold(; n_splits::Int = 5, shuffle::Bool = true, seed::Union{Int, Nothing} = nothing)
        @assert n_splits >= 2 "n_splits must be at least 2"
        new(n_splits, shuffle, seed)
    end
end

"""
    LeaveOneOut

Leave-One-Out cross-validation.
"""
struct LeaveOneOut <: CVStrategy end

"""
    GroupKFold

K-Fold with group constraints (no group split across folds).

# Fields
- `n_splits::Int`: Number of folds
"""
struct GroupKFold <: CVStrategy
    n_splits::Int

    function GroupKFold(; n_splits::Int = 5)
        @assert n_splits >= 2 "n_splits must be at least 2"
        new(n_splits)
    end
end

"""
    RepeatedKFold

Repeated K-Fold cross-validation.

# Fields
- `n_splits::Int`: Number of folds per repeat
- `n_repeats::Int`: Number of repeats
- `seed::Union{Int, Nothing}`: Base random seed
"""
struct RepeatedKFold <: CVStrategy
    n_splits::Int
    n_repeats::Int
    seed::Union{Int, Nothing}

    function RepeatedKFold(; n_splits::Int = 5, n_repeats::Int = 10,
                           seed::Union{Int, Nothing} = nothing)
        new(n_splits, n_repeats, seed)
    end
end

"""
    TimeSeriesCV

Time-series aware cross-validation (expanding or sliding window).

# Fields
- `n_splits::Int`: Number of splits
- `max_train_size::Union{Int, Nothing}`: Maximum training size (sliding window)
- `gap::Int`: Gap between train and test
- `test_size::Int`: Size of test set
"""
struct TimeSeriesCV <: CVStrategy
    n_splits::Int
    max_train_size::Union{Int, Nothing}
    gap::Int
    test_size::Int

    function TimeSeriesCV(; n_splits::Int = 5, max_train_size::Union{Int, Nothing} = nothing,
                          gap::Int = 0, test_size::Int = 1)
        new(n_splits, max_train_size, gap, test_size)
    end
end

"""
    NestedCV

Nested cross-validation for unbiased hyperparameter tuning.

# Fields
- `outer_cv::CVStrategy`: Outer CV for evaluation
- `inner_cv::CVStrategy`: Inner CV for hyperparameter tuning
"""
struct NestedCV <: CVStrategy
    outer_cv::CVStrategy
    inner_cv::CVStrategy

    function NestedCV(; outer_cv::CVStrategy = KFoldCV(n_splits = 5),
                      inner_cv::CVStrategy = KFoldCV(n_splits = 3))
        new(outer_cv, inner_cv)
    end
end

# ============================================================================
# Split Generation
# ============================================================================

n_splits(cv::KFoldCV) = cv.n_splits
n_splits(cv::StratifiedKFold) = cv.n_splits
n_splits(cv::LeaveOneOut) = -1  # Determined by data size
n_splits(cv::GroupKFold) = cv.n_splits
n_splits(cv::RepeatedKFold) = cv.n_splits * cv.n_repeats
n_splits(cv::TimeSeriesCV) = cv.n_splits
n_splits(cv::NestedCV) = n_splits(cv.outer_cv)

"""
    create_splits(cv::KFoldCV, n::Int)

Create K-Fold split indices.
Returns vector of (train_indices, test_indices) tuples.
"""
function create_splits(cv::KFoldCV, n::Int)
    indices = collect(1:n)

    if cv.shuffle
        rng = cv.seed !== nothing ? Random.MersenneTwister(cv.seed) : Random.GLOBAL_RNG
        Random.shuffle!(rng, indices)
    end

    fold_sizes = fill(n ÷ cv.n_splits, cv.n_splits)
    for i in 1:(n % cv.n_splits)
        fold_sizes[i] += 1
    end

    splits = Tuple{Vector{Int}, Vector{Int}}[]
    current = 1

    for k in 1:cv.n_splits
        test_end = current + fold_sizes[k] - 1
        test_idx = indices[current:test_end]
        train_idx = vcat(indices[1:current-1], indices[test_end+1:end])
        push!(splits, (train_idx, test_idx))
        current = test_end + 1
    end

    return splits
end

"""
    create_splits(cv::StratifiedKFold, n::Int, y::Vector)

Create stratified K-Fold splits maintaining class proportions.
"""
function create_splits(cv::StratifiedKFold, n::Int, y::Vector)
    @assert length(y) == n

    rng = cv.seed !== nothing ? Random.MersenneTwister(cv.seed) : Random.GLOBAL_RNG

    # Group indices by class
    classes = unique(y)
    class_indices = Dict{Any, Vector{Int}}()
    for c in classes
        class_indices[c] = findall(==(c), y)
        if cv.shuffle
            Random.shuffle!(rng, class_indices[c])
        end
    end

    # Distribute each class across folds
    fold_indices = [Int[] for _ in 1:cv.n_splits]

    for c in classes
        indices = class_indices[c]
        n_c = length(indices)
        fold_sizes = fill(n_c ÷ cv.n_splits, cv.n_splits)
        for i in 1:(n_c % cv.n_splits)
            fold_sizes[i] += 1
        end

        current = 1
        for k in 1:cv.n_splits
            append!(fold_indices[k], indices[current:current+fold_sizes[k]-1])
            current += fold_sizes[k]
        end
    end

    # Create splits
    splits = Tuple{Vector{Int}, Vector{Int}}[]
    all_indices = collect(1:n)

    for k in 1:cv.n_splits
        test_idx = fold_indices[k]
        train_idx = setdiff(all_indices, test_idx)
        push!(splits, (train_idx, test_idx))
    end

    return splits
end

"""
    create_splits(cv::LeaveOneOut, n::Int)

Create Leave-One-Out splits.
"""
function create_splits(cv::LeaveOneOut, n::Int)
    splits = Tuple{Vector{Int}, Vector{Int}}[]
    all_indices = collect(1:n)

    for i in 1:n
        test_idx = [i]
        train_idx = setdiff(all_indices, test_idx)
        push!(splits, (train_idx, test_idx))
    end

    return splits
end

"""
    create_splits(cv::GroupKFold, n::Int, groups::Vector)

Create Group K-Fold splits (no group spans folds).
"""
function create_splits(cv::GroupKFold, n::Int, groups::Vector)
    @assert length(groups) == n

    unique_groups = unique(groups)
    n_groups = length(unique_groups)
    @assert n_groups >= cv.n_splits "Not enough groups for $(cv.n_splits) folds"

    # Assign groups to folds
    group_to_fold = Dict{Any, Int}()
    for (i, g) in enumerate(unique_groups)
        group_to_fold[g] = mod1(i, cv.n_splits)
    end

    # Create splits
    splits = Tuple{Vector{Int}, Vector{Int}}[]
    all_indices = collect(1:n)

    for k in 1:cv.n_splits
        test_idx = [i for i in 1:n if group_to_fold[groups[i]] == k]
        train_idx = setdiff(all_indices, test_idx)
        push!(splits, (train_idx, test_idx))
    end

    return splits
end

"""
    create_splits(cv::RepeatedKFold, n::Int)

Create Repeated K-Fold splits.
"""
function create_splits(cv::RepeatedKFold, n::Int)
    splits = Tuple{Vector{Int}, Vector{Int}}[]

    for r in 1:cv.n_repeats
        seed = cv.seed !== nothing ? cv.seed + r - 1 : nothing
        inner_cv = KFoldCV(n_splits = cv.n_splits, shuffle = true, seed = seed)
        append!(splits, create_splits(inner_cv, n))
    end

    return splits
end

"""
    create_splits(cv::TimeSeriesCV, n::Int)

Create Time-Series cross-validation splits.
"""
function create_splits(cv::TimeSeriesCV, n::Int)
    splits = Tuple{Vector{Int}, Vector{Int}}[]

    # Calculate minimum training size
    min_train = n - cv.n_splits * cv.test_size - (cv.n_splits - 1) * cv.gap

    if min_train < 1
        error("Not enough samples for $(cv.n_splits) time series splits")
    end

    for k in 1:cv.n_splits
        # Test indices
        test_start = n - (cv.n_splits - k + 1) * cv.test_size - (cv.n_splits - k) * cv.gap + 1
        test_end = test_start + cv.test_size - 1
        test_idx = collect(test_start:test_end)

        # Training indices (all before gap)
        train_end = test_start - cv.gap - 1
        if cv.max_train_size !== nothing
            train_start = max(1, train_end - cv.max_train_size + 1)
        else
            train_start = 1
        end
        train_idx = collect(train_start:train_end)

        push!(splits, (train_idx, test_idx))
    end

    return splits
end

"""
    get_fold(cv::CVStrategy, n::Int, fold::Int; kwargs...)

Get specific fold indices.
"""
function get_fold(cv::CVStrategy, n::Int, fold::Int; kwargs...)
    splits = create_splits(cv, n; kwargs...)
    @assert 1 <= fold <= length(splits) "Fold $fold out of range"
    return splits[fold]
end

# ============================================================================
# CV Result
# ============================================================================

"""
    CVResult

Result from cross-validation.
"""
struct CVResult
    scores::Vector{Float64}
    mean_score::Float64
    std_score::Float64
    fold_results::Vector{Dict{Symbol, Any}}
    cv_strategy::CVStrategy
end

function CVResult(scores::Vector{Float64}, fold_results::Vector{Dict{Symbol, Any}},
                  cv_strategy::CVStrategy)
    CVResult(scores, mean(scores), std(scores), fold_results, cv_strategy)
end

"""
    aggregate_cv_results(results::Vector{CVResult})

Aggregate multiple CV results (e.g., from different metrics).
"""
function aggregate_cv_results(results::Vector{CVResult})
    all_scores = vcat([r.scores for r in results]...)
    return Dict(
        :mean => mean(all_scores),
        :std => std(all_scores),
        :median => median(all_scores),
        :min => minimum(all_scores),
        :max => maximum(all_scores),
        :n_folds => length(all_scores)
    )
end

# ============================================================================
# Cross-Validation Execution
# ============================================================================

"""
    cross_validate(cv, X, y, model_fn, score_fn; kwargs...)

Perform cross-validation.

# Arguments
- `cv::CVStrategy`: Cross-validation strategy
- `X::Matrix`: Features (samples × features)
- `y::Vector`: Targets
- `model_fn::Function`: Model training function (X_train, y_train) -> model
- `score_fn::Function`: Scoring function (model, X_test, y_test) -> score
- `verbose::Bool`: Print progress
"""
function cross_validate(cv::CVStrategy, X::Matrix{Float64}, y::Vector,
                        model_fn::Function, score_fn::Function;
                        verbose::Bool = false, return_models::Bool = false)
    n = size(X, 1)

    # Create splits based on CV type
    if cv isa StratifiedKFold
        splits = create_splits(cv, n, y)
    elseif cv isa GroupKFold
        error("GroupKFold requires groups argument")
    else
        splits = create_splits(cv, n)
    end

    scores = Float64[]
    fold_results = Dict{Symbol, Any}[]
    models = []

    for (fold, (train_idx, test_idx)) in enumerate(splits)
        if verbose
            println("Fold $fold/$(length(splits))...")
        end

        # Split data
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_test = X[test_idx, :]
        y_test = y[test_idx]

        # Train and evaluate
        model = model_fn(X_train, y_train)
        score = score_fn(model, X_test, y_test)

        push!(scores, score)
        push!(fold_results, Dict(
            :fold => fold,
            :train_size => length(train_idx),
            :test_size => length(test_idx),
            :score => score
        ))

        if return_models
            push!(models, model)
        end
    end

    result = CVResult(scores, fold_results, cv)

    if return_models
        return (result = result, models = models)
    end

    return result
end

"""
    cross_validate_with_groups(cv::GroupKFold, X, y, groups, model_fn, score_fn)

Cross-validate with group constraints.
"""
function cross_validate_with_groups(cv::GroupKFold, X::Matrix{Float64}, y::Vector,
                                    groups::Vector, model_fn::Function, score_fn::Function;
                                    verbose::Bool = false)
    n = size(X, 1)
    splits = create_splits(cv, n, groups)

    scores = Float64[]
    fold_results = Dict{Symbol, Any}[]

    for (fold, (train_idx, test_idx)) in enumerate(splits)
        if verbose
            train_groups = unique(groups[train_idx])
            test_groups = unique(groups[test_idx])
            println("Fold $fold: train groups = $train_groups, test groups = $test_groups")
        end

        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_test = X[test_idx, :]
        y_test = y[test_idx]

        model = model_fn(X_train, y_train)
        score = score_fn(model, X_test, y_test)

        push!(scores, score)
        push!(fold_results, Dict(
            :fold => fold,
            :train_size => length(train_idx),
            :test_size => length(test_idx),
            :train_groups => unique(groups[train_idx]),
            :test_groups => unique(groups[test_idx]),
            :score => score
        ))
    end

    return CVResult(scores, fold_results, cv)
end

# ============================================================================
# Nested Cross-Validation
# ============================================================================

"""
    nested_cross_validate(ncv, X, y, model_fn, score_fn, param_grid; kwargs...)

Perform nested cross-validation for unbiased hyperparameter tuning.

# Arguments
- `ncv::NestedCV`: Nested CV strategy
- `X::Matrix`: Features
- `y::Vector`: Targets
- `model_fn::Function`: (X, y, params) -> model
- `score_fn::Function`: (model, X, y) -> score
- `param_grid::Vector{Dict}`: Hyperparameter combinations to try
"""
function nested_cross_validate(ncv::NestedCV, X::Matrix{Float64}, y::Vector,
                               model_fn::Function, score_fn::Function,
                               param_grid::Vector{Dict}; verbose::Bool = false)
    n = size(X, 1)

    # Outer splits
    if ncv.outer_cv isa StratifiedKFold
        outer_splits = create_splits(ncv.outer_cv, n, y)
    else
        outer_splits = create_splits(ncv.outer_cv, n)
    end

    outer_scores = Float64[]
    fold_results = Dict{Symbol, Any}[]
    best_params_per_fold = Dict[]

    for (outer_fold, (outer_train_idx, outer_test_idx)) in enumerate(outer_splits)
        if verbose
            println("Outer fold $outer_fold/$(length(outer_splits))...")
        end

        X_outer_train = X[outer_train_idx, :]
        y_outer_train = y[outer_train_idx]
        X_test = X[outer_test_idx, :]
        y_test = y[outer_test_idx]

        # Inner CV for hyperparameter tuning
        inner_n = length(outer_train_idx)

        if ncv.inner_cv isa StratifiedKFold
            inner_splits = create_splits(ncv.inner_cv, inner_n, y_outer_train)
        else
            inner_splits = create_splits(ncv.inner_cv, inner_n)
        end

        # Evaluate each parameter combination
        param_scores = Float64[]

        for params in param_grid
            inner_scores = Float64[]

            for (inner_train_idx, inner_val_idx) in inner_splits
                X_inner_train = X_outer_train[inner_train_idx, :]
                y_inner_train = y_outer_train[inner_train_idx]
                X_val = X_outer_train[inner_val_idx, :]
                y_val = y_outer_train[inner_val_idx]

                model = model_fn(X_inner_train, y_inner_train, params)
                score = score_fn(model, X_val, y_val)
                push!(inner_scores, score)
            end

            push!(param_scores, mean(inner_scores))
        end

        # Select best parameters
        best_param_idx = argmax(param_scores)
        best_params = param_grid[best_param_idx]

        # Train final model on full outer training set
        final_model = model_fn(X_outer_train, y_outer_train, best_params)
        outer_score = score_fn(final_model, X_test, y_test)

        push!(outer_scores, outer_score)
        push!(best_params_per_fold, best_params)
        push!(fold_results, Dict(
            :outer_fold => outer_fold,
            :best_params => best_params,
            :best_inner_score => param_scores[best_param_idx],
            :outer_score => outer_score,
            :all_param_scores => param_scores
        ))
    end

    return Dict(
        :outer_scores => outer_scores,
        :mean_score => mean(outer_scores),
        :std_score => std(outer_scores),
        :best_params_per_fold => best_params_per_fold,
        :fold_results => fold_results
    )
end

# ============================================================================
# Distribution Shift Detection
# ============================================================================

"""
    compute_mmd(X, Y; kernel, gamma)

Compute Maximum Mean Discrepancy between two distributions.
"""
function compute_mmd(X::Matrix{Float64}, Y::Matrix{Float64};
                     kernel::Symbol = :rbf, gamma::Float64 = 1.0)
    m = size(X, 1)
    n = size(Y, 1)

    # Kernel function
    if kernel == :rbf
        k(a, b) = exp(-gamma * sum((a .- b).^2))
    elseif kernel == :linear
        k(a, b) = dot(a, b)
    else
        error("Unknown kernel: $kernel")
    end

    # Compute MMD
    xx_sum = 0.0
    for i in 1:m
        for j in 1:m
            if i != j
                xx_sum += k(X[i, :], X[j, :])
            end
        end
    end
    xx_sum /= m * (m - 1)

    yy_sum = 0.0
    for i in 1:n
        for j in 1:n
            if i != j
                yy_sum += k(Y[i, :], Y[j, :])
            end
        end
    end
    yy_sum /= n * (n - 1)

    xy_sum = 0.0
    for i in 1:m
        for j in 1:n
            xy_sum += k(X[i, :], Y[j, :])
        end
    end
    xy_sum /= m * n

    mmd_squared = xx_sum + yy_sum - 2 * xy_sum
    return sqrt(max(0.0, mmd_squared))
end

"""
    ks_test_2sample(x, y)

Two-sample Kolmogorov-Smirnov test.
Returns (D statistic, approximate p-value).
"""
function ks_test_2sample(x::Vector{Float64}, y::Vector{Float64})
    # Sort combined and compute ECDFs
    x_sorted = sort(x)
    y_sorted = sort(y)

    n1 = length(x)
    n2 = length(y)

    # Compute maximum difference between ECDFs
    all_vals = sort(vcat(x, y))
    max_diff = 0.0

    for v in all_vals
        ecdf_x = count(xi -> xi <= v, x) / n1
        ecdf_y = count(yi -> yi <= v, y) / n2
        max_diff = max(max_diff, abs(ecdf_x - ecdf_y))
    end

    # Approximate p-value using asymptotic distribution
    en = sqrt(n1 * n2 / (n1 + n2))
    p_value = 2 * exp(-2 * (en * max_diff)^2)
    p_value = clamp(p_value, 0.0, 1.0)

    return (D = max_diff, p_value = p_value)
end

"""
    distribution_shift_detection(X_train, X_test; method, feature_wise)

Detect distribution shift between training and test data.

# Arguments
- `X_train::Matrix`: Training features
- `X_test::Matrix`: Test features
- `method::Symbol`: Detection method (:mmd, :ks, :both)
- `feature_wise::Bool`: Test each feature separately

# Returns
Dict with detection results and recommendations.
"""
function distribution_shift_detection(X_train::Matrix{Float64}, X_test::Matrix{Float64};
                                      method::Symbol = :both, feature_wise::Bool = true,
                                      alpha::Float64 = 0.05)
    results = Dict{Symbol, Any}()
    n_features = size(X_train, 2)

    if method in (:mmd, :both)
        # Global MMD test
        mmd = compute_mmd(X_train, X_test, gamma = 1.0 / n_features)
        results[:global_mmd] = mmd
        results[:mmd_significant] = mmd > 0.1  # Threshold heuristic
    end

    if method in (:ks, :both) && feature_wise
        # Per-feature KS tests
        ks_results = Dict{Int, NamedTuple}()
        significant_features = Int[]

        for j in 1:n_features
            ks_result = ks_test_2sample(X_train[:, j], X_test[:, j])
            ks_results[j] = ks_result
            if ks_result.p_value < alpha
                push!(significant_features, j)
            end
        end

        results[:ks_per_feature] = ks_results
        results[:significant_features] = significant_features
        results[:n_shifted_features] = length(significant_features)
    end

    # Overall assessment
    shift_detected = false
    if haskey(results, :mmd_significant) && results[:mmd_significant]
        shift_detected = true
    end
    if haskey(results, :n_shifted_features) && results[:n_shifted_features] > n_features * 0.2
        shift_detected = true
    end

    results[:shift_detected] = shift_detected
    results[:recommendation] = if shift_detected
        "Significant distribution shift detected. Consider domain adaptation or recalibration."
    else
        "No significant distribution shift detected."
    end

    return results
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    learning_curve(cv, X, y, model_fn, score_fn, train_sizes; kwargs...)

Compute learning curve using cross-validation.
"""
function learning_curve(cv::CVStrategy, X::Matrix{Float64}, y::Vector,
                        model_fn::Function, score_fn::Function,
                        train_sizes::Vector{Float64}; verbose::Bool = false)
    n = size(X, 1)

    results = Dict{Float64, NamedTuple}()

    for frac in train_sizes
        actual_n = round(Int, n * frac)
        subset_idx = randperm(n)[1:actual_n]

        X_sub = X[subset_idx, :]
        y_sub = y[subset_idx]

        if verbose
            println("Training size: $actual_n ($frac of $n)")
        end

        cv_result = cross_validate(cv, X_sub, y_sub, model_fn, score_fn)

        results[frac] = (
            train_size = actual_n,
            mean_score = cv_result.mean_score,
            std_score = cv_result.std_score
        )
    end

    return results
end

"""
    permutation_test_cv(cv, X, y, model_fn, score_fn; n_permutations)

Perform permutation test to assess model significance.
"""
function permutation_test_cv(cv::CVStrategy, X::Matrix{Float64}, y::Vector,
                            model_fn::Function, score_fn::Function;
                            n_permutations::Int = 100, verbose::Bool = false)
    # True score
    true_result = cross_validate(cv, X, y, model_fn, score_fn)
    true_score = true_result.mean_score

    # Permutation scores
    perm_scores = Float64[]

    for p in 1:n_permutations
        if verbose && p % 10 == 0
            println("Permutation $p/$n_permutations")
        end

        y_perm = shuffle(y)
        perm_result = cross_validate(cv, X, y_perm, model_fn, score_fn)
        push!(perm_scores, perm_result.mean_score)
    end

    # P-value
    p_value = count(s -> s >= true_score, perm_scores) / n_permutations

    return Dict(
        :true_score => true_score,
        :perm_scores => perm_scores,
        :p_value => p_value,
        :significant => p_value < 0.05
    )
end

end # module
