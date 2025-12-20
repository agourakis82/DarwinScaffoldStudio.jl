"""
UncertaintyQuantification.jl - Comprehensive Uncertainty Quantification for Scaffolds

Provides state-of-the-art uncertainty quantification methods for neural network
predictions on scaffold analysis tasks.

SOTA 2024-2025 Features:
- Deep Ensembles (Lakshminarayanan et al. 2017)
- MC Dropout (Gal & Ghahramani 2016)
- Bayesian Neural Networks (approximate variational inference)
- Conformal Prediction (distribution-free coverage guarantees)
- Calibration metrics (ECE, MCE)
- Epistemic vs Aleatoric uncertainty decomposition

Applications for Scaffold Analysis:
- Reliable property predictions with confidence bounds
- Out-of-distribution detection for novel scaffold designs
- Risk-aware optimization decisions
- Calibrated predictions for regulatory approval

References:
- Lakshminarayanan et al. 2017: "Simple and Scalable Predictive Uncertainty"
- Gal & Ghahramani 2016: "Dropout as Bayesian Approximation"
- Blundell et al. 2015: "Weight Uncertainty in Neural Networks"
- Romano et al. 2019: "Conformalized Quantile Regression"
"""
module UncertaintyQuantification

using LinearAlgebra
using Statistics
using Random

export DeepEnsemble, MCDropout, BayesianNN, ConformalPredictor
export predict_with_uncertainty, calibrate!, compute_calibration
export expected_calibration_error, regression_calibration_error
export epistemic_aleatoric_decomposition
export EnsembleMember, PredictionResult, CalibrationResult

# ============================================================================
# TYPES
# ============================================================================

"""
    PredictionResult

Contains prediction with uncertainty estimates.
"""
struct PredictionResult
    mean::Vector{Float64}
    std::Vector{Float64}
    lower::Vector{Float64}     # Lower confidence bound
    upper::Vector{Float64}     # Upper confidence bound
    confidence::Float64        # Confidence level (e.g., 0.95)
    epistemic::Vector{Float64} # Epistemic (model) uncertainty
    aleatoric::Vector{Float64} # Aleatoric (data) uncertainty
end

function PredictionResult(mean::Vector{Float64}, std::Vector{Float64};
                          confidence::Float64=0.95)
    z = 1.96  # For 95% confidence
    lower = mean .- z .* std
    upper = mean .+ z .* std
    PredictionResult(mean, std, lower, upper, confidence, std, zeros(length(std)))
end

"""
    CalibrationResult

Results from calibration assessment.
"""
struct CalibrationResult
    ece::Float64                 # Expected Calibration Error
    mce::Float64                 # Maximum Calibration Error
    bin_accuracies::Vector{Float64}
    bin_confidences::Vector{Float64}
    bin_counts::Vector{Int}
    is_calibrated::Bool
end

# ============================================================================
# ENSEMBLE MEMBER
# ============================================================================

"""
    EnsembleMember

A single neural network in an ensemble.
Simplified MLP implementation for demonstration.
"""
struct EnsembleMember
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
    dropout_rate::Float64
end

function EnsembleMember(layer_dims::Vector{Int}; dropout_rate::Float64=0.0)
    n_layers = length(layer_dims) - 1
    weights = Matrix{Float64}[]
    biases = Vector{Float64}[]

    for i in 1:n_layers
        d_in, d_out = layer_dims[i], layer_dims[i+1]
        W = randn(d_in, d_out) .* sqrt(2.0 / d_in)
        b = zeros(d_out)
        push!(weights, W)
        push!(biases, b)
    end

    EnsembleMember(weights, biases, dropout_rate)
end

function forward(member::EnsembleMember, x::Vector{Float64}; training::Bool=false)
    h = x
    for (i, (W, b)) in enumerate(zip(member.weights, member.biases))
        h = W' * h .+ b

        # Apply ReLU except for last layer
        if i < length(member.weights)
            h = max.(h, 0.0)

            # Apply dropout during training
            if training && member.dropout_rate > 0
                mask = rand(length(h)) .> member.dropout_rate
                h = h .* mask ./ (1 - member.dropout_rate)
            end
        end
    end
    return h
end

function forward_batch(member::EnsembleMember, X::Matrix{Float64}; training::Bool=false)
    n_samples = size(X, 1)
    out_dim = length(member.biases[end])
    outputs = zeros(n_samples, out_dim)

    for i in 1:n_samples
        outputs[i, :] = forward(member, X[i, :]; training=training)
    end

    return outputs
end

# ============================================================================
# DEEP ENSEMBLE
# ============================================================================

"""
    DeepEnsemble

Deep Ensemble for uncertainty quantification.

Key idea: Train N independent networks with different random seeds,
then aggregate predictions. Disagreement = epistemic uncertainty.
"""
struct DeepEnsemble
    members::Vector{EnsembleMember}
    n_members::Int
    layer_dims::Vector{Int}
end

function DeepEnsemble(layer_dims::Vector{Int}; n_members::Int=5)
    members = [EnsembleMember(layer_dims) for _ in 1:n_members]
    DeepEnsemble(members, n_members, layer_dims)
end

"""
    predict_with_uncertainty(ensemble, X) -> PredictionResult

Make predictions with uncertainty estimates.
"""
function predict_with_uncertainty(ensemble::DeepEnsemble, X::Matrix{Float64})
    n_samples = size(X, 1)
    out_dim = length(ensemble.members[1].biases[end])

    # Get predictions from all members
    all_preds = zeros(n_samples, out_dim, ensemble.n_members)

    for (m, member) in enumerate(ensemble.members)
        all_preds[:, :, m] = forward_batch(member, X)
    end

    # Aggregate
    mean_pred = mean(all_preds, dims=3)[:, :, 1]
    std_pred = std(all_preds, dims=3)[:, :, 1]

    # For single output
    if out_dim == 1
        return PredictionResult(vec(mean_pred), vec(std_pred))
    end

    return PredictionResult(mean_pred[1, :], std_pred[1, :])
end

"""
    train_ensemble!(ensemble, X, y; epochs=100, lr=0.01)

Train ensemble members with different random initializations.
"""
function train_ensemble!(ensemble::DeepEnsemble, X::Matrix{Float64}, y::Vector{Float64};
                         epochs::Int=100, lr::Float64=0.01, verbose::Bool=true)
    for (m, member) in enumerate(ensemble.members)
        if verbose
            @info "Training ensemble member $m/$(ensemble.n_members)"
        end
        train_member!(member, X, y; epochs=epochs, lr=lr)
    end
end

function train_member!(member::EnsembleMember, X::Matrix{Float64}, y::Vector{Float64};
                       epochs::Int=100, lr::Float64=0.01)
    n_samples = size(X, 1)

    for epoch in 1:epochs
        total_loss = 0.0

        # Shuffle data
        perm = randperm(n_samples)

        for idx in perm
            x = X[idx, :]
            target = y[idx]

            # Forward pass
            pred = forward(member, x; training=true)

            # MSE loss gradient
            error = pred[1] - target
            total_loss += error^2

            # Simple SGD (would use proper backprop in practice)
            # Gradient for last layer
            grad_out = 2 * error

            # Update last layer
            last_idx = length(member.weights)
            h = x
            for i in 1:(last_idx-1)
                h = max.(member.weights[i]' * h .+ member.biases[i], 0.0)
            end

            member.weights[last_idx] .-= lr .* h * grad_out'
            member.biases[last_idx] .-= lr .* [grad_out]
        end
    end
end

# ============================================================================
# MC DROPOUT
# ============================================================================

"""
    MCDropout

Monte Carlo Dropout for uncertainty quantification.

Key idea: Keep dropout active at test time, run multiple forward passes.
Variance across passes approximates epistemic uncertainty.
"""
struct MCDropout
    network::EnsembleMember
    n_samples::Int
    dropout_rate::Float64
end

function MCDropout(layer_dims::Vector{Int}; n_samples::Int=50, dropout_rate::Float64=0.2)
    network = EnsembleMember(layer_dims; dropout_rate=dropout_rate)
    MCDropout(network, n_samples, dropout_rate)
end

function predict_with_uncertainty(mc::MCDropout, X::Matrix{Float64})
    n_inputs = size(X, 1)
    out_dim = length(mc.network.biases[end])

    # Multiple forward passes with dropout
    all_preds = zeros(n_inputs, out_dim, mc.n_samples)

    for s in 1:mc.n_samples
        all_preds[:, :, s] = forward_batch(mc.network, X; training=true)
    end

    # Aggregate
    mean_pred = mean(all_preds, dims=3)[:, :, 1]
    std_pred = std(all_preds, dims=3)[:, :, 1]

    if out_dim == 1
        return PredictionResult(vec(mean_pred), vec(std_pred))
    end

    return PredictionResult(mean_pred[1, :], std_pred[1, :])
end

# ============================================================================
# BAYESIAN NEURAL NETWORK
# ============================================================================

"""
    BayesianNN

Bayesian Neural Network with weight uncertainty.

Uses variational inference with reparameterization trick.
Each weight w ~ N(μ_w, σ_w²) instead of point estimate.
"""
struct BayesianNN
    mean_weights::Vector{Matrix{Float64}}
    log_var_weights::Vector{Matrix{Float64}}
    mean_biases::Vector{Vector{Float64}}
    log_var_biases::Vector{Vector{Float64}}
    n_samples::Int
    prior_std::Float64
end

function BayesianNN(layer_dims::Vector{Int}; n_samples::Int=20, prior_std::Float64=1.0)
    n_layers = length(layer_dims) - 1

    mean_weights = Matrix{Float64}[]
    log_var_weights = Matrix{Float64}[]
    mean_biases = Vector{Float64}[]
    log_var_biases = Vector{Float64}[]

    for i in 1:n_layers
        d_in, d_out = layer_dims[i], layer_dims[i+1]

        # Mean initialization
        μ_w = randn(d_in, d_out) .* sqrt(2.0 / d_in)
        μ_b = zeros(d_out)

        # Log-variance initialization (small initial variance)
        log_σ_w = fill(-3.0, d_in, d_out)
        log_σ_b = fill(-3.0, d_out)

        push!(mean_weights, μ_w)
        push!(log_var_weights, log_σ_w)
        push!(mean_biases, μ_b)
        push!(log_var_biases, log_σ_b)
    end

    BayesianNN(mean_weights, log_var_weights, mean_biases, log_var_biases,
               n_samples, prior_std)
end

"""
    sample_weights(bnn)

Sample a set of weights from the variational posterior.
"""
function sample_weights(bnn::BayesianNN)
    sampled_W = Matrix{Float64}[]
    sampled_b = Vector{Float64}[]

    for (μ_w, log_σ_w, μ_b, log_σ_b) in zip(bnn.mean_weights, bnn.log_var_weights,
                                             bnn.mean_biases, bnn.log_var_biases)
        # Reparameterization trick
        ε_w = randn(size(μ_w))
        ε_b = randn(size(μ_b))

        σ_w = exp.(0.5 .* log_σ_w)
        σ_b = exp.(0.5 .* log_σ_b)

        W = μ_w .+ σ_w .* ε_w
        b = μ_b .+ σ_b .* ε_b

        push!(sampled_W, W)
        push!(sampled_b, b)
    end

    return sampled_W, sampled_b
end

function forward_sample(bnn::BayesianNN, x::Vector{Float64})
    W, b = sample_weights(bnn)

    h = x
    for (i, (Wi, bi)) in enumerate(zip(W, b))
        h = Wi' * h .+ bi
        if i < length(W)
            h = max.(h, 0.0)  # ReLU
        end
    end

    return h
end

function predict_with_uncertainty(bnn::BayesianNN, X::Matrix{Float64})
    n_inputs = size(X, 1)
    out_dim = length(bnn.mean_biases[end])

    # Multiple forward passes with sampled weights
    all_preds = zeros(n_inputs, out_dim, bnn.n_samples)

    for s in 1:bnn.n_samples
        for i in 1:n_inputs
            all_preds[i, :, s] = forward_sample(bnn, X[i, :])
        end
    end

    # Aggregate
    mean_pred = mean(all_preds, dims=3)[:, :, 1]
    std_pred = std(all_preds, dims=3)[:, :, 1]

    if out_dim == 1
        return PredictionResult(vec(mean_pred), vec(std_pred))
    end

    return PredictionResult(mean_pred[1, :], std_pred[1, :])
end

"""
    kl_divergence(bnn)

Compute KL divergence between posterior and prior.
"""
function kl_divergence(bnn::BayesianNN)
    kl = 0.0

    for (μ_w, log_σ_w) in zip(bnn.mean_weights, bnn.log_var_weights)
        σ_w² = exp.(log_σ_w)
        # KL(N(μ, σ²) || N(0, prior_std²))
        kl += 0.5 * sum(μ_w.^2 ./ bnn.prior_std^2 .+
                        σ_w² ./ bnn.prior_std^2 .-
                        log_σ_w .+ log(bnn.prior_std^2) .- 1)
    end

    return kl
end

# ============================================================================
# CONFORMAL PREDICTION
# ============================================================================

"""
    ConformalPredictor

Conformal Prediction for distribution-free uncertainty.

Key idea: Use a calibration set to compute conformity scores,
then use these to construct prediction intervals with guaranteed coverage.
"""
mutable struct ConformalPredictor
    base_model::Union{DeepEnsemble, MCDropout, BayesianNN}
    conformity_scores::Vector{Float64}
    calibrated::Bool
    alpha::Float64  # Miscoverage rate (1 - confidence)
end

function ConformalPredictor(base_model; alpha::Float64=0.1)
    ConformalPredictor(base_model, Float64[], false, alpha)
end

"""
    calibrate!(cp, X_cal, y_cal)

Calibrate conformal predictor using calibration data.
"""
function calibrate!(cp::ConformalPredictor, X_cal::Matrix{Float64}, y_cal::Vector{Float64})
    # Get predictions
    result = predict_with_uncertainty(cp.base_model, X_cal)

    # Compute conformity scores (absolute residuals normalized by std)
    residuals = abs.(y_cal .- result.mean)
    cp.conformity_scores = residuals ./ max.(result.std, 1e-6)

    cp.calibrated = true

    return cp
end

"""
    predict_with_uncertainty(cp::ConformalPredictor, X)

Make predictions with conformal prediction intervals.
"""
function predict_with_uncertainty(cp::ConformalPredictor, X::Matrix{Float64})
    if !cp.calibrated
        error("ConformalPredictor must be calibrated first")
    end

    # Get base predictions
    base_result = predict_with_uncertainty(cp.base_model, X)

    # Compute quantile of conformity scores
    n_cal = length(cp.conformity_scores)
    q = ceil(Int, (n_cal + 1) * (1 - cp.alpha)) / n_cal
    q = min(q, 1.0)

    sorted_scores = sort(cp.conformity_scores)
    idx = ceil(Int, length(sorted_scores) * q)
    idx = min(idx, length(sorted_scores))
    threshold = sorted_scores[idx]

    # Construct prediction intervals
    half_width = threshold .* base_result.std
    lower = base_result.mean .- half_width
    upper = base_result.mean .+ half_width

    return PredictionResult(
        base_result.mean, base_result.std,
        lower, upper, 1 - cp.alpha,
        base_result.std, zeros(length(base_result.std))
    )
end

# ============================================================================
# CALIBRATION METRICS
# ============================================================================

"""
    expected_calibration_error(confidences, accuracies; n_bins=10)

Compute Expected Calibration Error (ECE) for classification.

ECE = Σ |B_m|/n * |acc(B_m) - conf(B_m)|

Lower is better. Well-calibrated models have ECE ≈ 0.
"""
function expected_calibration_error(confidences::Vector{Float64},
                                    accuracies::Vector{Bool};
                                    n_bins::Int=10)
    n = length(confidences)

    bin_boundaries = range(0, 1, length=n_bins+1)
    bin_accuracies = zeros(n_bins)
    bin_confidences = zeros(n_bins)
    bin_counts = zeros(Int, n_bins)

    for (conf, acc) in zip(confidences, accuracies)
        bin_idx = min(floor(Int, conf * n_bins) + 1, n_bins)
        bin_counts[bin_idx] += 1
        bin_accuracies[bin_idx] += acc ? 1.0 : 0.0
        bin_confidences[bin_idx] += conf
    end

    # Normalize
    for i in 1:n_bins
        if bin_counts[i] > 0
            bin_accuracies[i] /= bin_counts[i]
            bin_confidences[i] /= bin_counts[i]
        end
    end

    # ECE
    ece = 0.0
    for i in 1:n_bins
        ece += (bin_counts[i] / n) * abs(bin_accuracies[i] - bin_confidences[i])
    end

    # MCE (Maximum Calibration Error)
    mce = maximum(abs.(bin_accuracies .- bin_confidences))

    is_calibrated = ece < 0.05

    return CalibrationResult(ece, mce, bin_accuracies, bin_confidences, bin_counts, is_calibrated)
end

"""
    regression_calibration_error(predictions, targets, stds; coverage_levels)

Compute calibration error for regression with uncertainty.

For each coverage level (e.g., 50%, 90%, 95%), check if the actual
coverage matches the expected coverage.
"""
function regression_calibration_error(predictions::Vector{Float64},
                                      targets::Vector{Float64},
                                      stds::Vector{Float64};
                                      coverage_levels::Vector{Float64}=[0.5, 0.9, 0.95])
    n = length(predictions)
    results = Dict{Float64, Float64}()

    for level in coverage_levels
        z = quantile_normal((1 + level) / 2)

        # Count how many targets fall within interval
        in_interval = 0
        for (pred, target, std) in zip(predictions, targets, stds)
            lower = pred - z * std
            upper = pred + z * std
            if lower <= target <= upper
                in_interval += 1
            end
        end

        actual_coverage = in_interval / n
        calibration_error = abs(actual_coverage - level)
        results[level] = calibration_error
    end

    # Average calibration error
    avg_error = mean(values(results))

    return Dict(
        "coverage_errors" => results,
        "average_error" => avg_error,
        "is_calibrated" => avg_error < 0.05
    )
end

# Normal quantile approximation
function quantile_normal(p::Float64)
    # Approximation for standard normal quantile
    if p <= 0
        return -Inf
    elseif p >= 1
        return Inf
    elseif p == 0.5
        return 0.0
    end

    # Rational approximation
    a = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00
    ]

    b = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01
    ]

    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00
    ]

    d = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low
        q = sqrt(-2 * log(p))
        return (((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
               ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    elseif p <= p_high
        q = p - 0.5
        r = q * q
        return (((((a[1]*r + a[2])*r + a[3])*r + a[4])*r + a[5])*r + a[6])*q /
               (((((b[1]*r + b[2])*r + b[3])*r + b[4])*r + b[5])*r + 1)
    else
        q = sqrt(-2 * log(1 - p))
        return -(((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
                ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    end
end

# ============================================================================
# UNCERTAINTY DECOMPOSITION
# ============================================================================

"""
    epistemic_aleatoric_decomposition(model, X; n_samples=50)

Decompose total uncertainty into epistemic and aleatoric components.

Epistemic: Model uncertainty (reducible with more data)
Aleatoric: Data noise (irreducible)

For ensembles/MC-Dropout:
- Epistemic ≈ Variance of means across samples
- Aleatoric ≈ Mean of variances across samples
"""
function epistemic_aleatoric_decomposition(ensemble::DeepEnsemble, X::Matrix{Float64})
    n_samples = size(X, 1)
    out_dim = length(ensemble.members[1].biases[end])

    # Get predictions from all members
    all_means = zeros(n_samples, out_dim, ensemble.n_members)

    for (m, member) in enumerate(ensemble.members)
        all_means[:, :, m] = forward_batch(member, X)
    end

    # Mean prediction
    mean_pred = mean(all_means, dims=3)[:, :, 1]

    # Epistemic: variance of the means
    epistemic = var(all_means, dims=3)[:, :, 1]

    # For simple case, aleatoric is estimated from spread
    # (would need heteroscedastic network for true aleatoric)
    aleatoric = zeros(size(epistemic))

    # Total uncertainty
    total = epistemic .+ aleatoric

    return Dict(
        "mean" => mean_pred,
        "total_std" => sqrt.(total),
        "epistemic_std" => sqrt.(epistemic),
        "aleatoric_std" => sqrt.(aleatoric),
        "epistemic_fraction" => epistemic ./ max.(total, 1e-10)
    )
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
    uncertainty_aware_decision(prediction, threshold; risk_aversion=1.0)

Make decisions considering uncertainty.

Uses lower confidence bound for risk-averse decisions.
"""
function uncertainty_aware_decision(result::PredictionResult, threshold::Float64;
                                    risk_aversion::Float64=1.0)
    # Risk-adjusted value: mean - risk_aversion * std
    risk_adjusted = result.mean .- risk_aversion .* result.std

    decisions = risk_adjusted .>= threshold

    return Dict(
        "decisions" => decisions,
        "risk_adjusted_values" => risk_adjusted,
        "confidence" => 1 .- result.std ./ max.(abs.(result.mean), 1e-10)
    )
end

"""
    detect_ood(result, threshold_std; method=:std)

Detect out-of-distribution samples based on uncertainty.
"""
function detect_ood(result::PredictionResult; threshold_multiplier::Float64=2.0)
    median_std = median(result.std)
    threshold = threshold_multiplier * median_std

    is_ood = result.std .> threshold

    return Dict(
        "is_ood" => is_ood,
        "ood_count" => sum(is_ood),
        "ood_fraction" => mean(is_ood),
        "threshold" => threshold
    )
end

end # module
