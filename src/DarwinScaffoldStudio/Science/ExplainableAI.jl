"""
ExplainableAI.jl - SOTA+++ Explainable AI for Scaffold Design

Interpretability methods for understanding model predictions:
- SHAP (SHapley Additive exPlanations) values
- Attention visualization for transformers
- Feature importance analysis
- Counterfactual explanations

Makes AI-driven scaffold design transparent and trustworthy.

Created: 2025-12-21
Author: Darwin Scaffold Studio Team
Version: 3.4.0
"""

module ExplainableAI

using Statistics
using Random
using LinearAlgebra

export compute_shap_values, visualize_attention, feature_importance
export counterfactual_explanation, explain_prediction

# ============================================================================
# SHAP (SHapley Additive exPlanations)
# ============================================================================

"""
    compute_shap_values(model, x, X_background; n_samples=100)

Compute SHAP values for a prediction using Kernel SHAP.

SHAP values explain the contribution of each feature to the prediction
relative to a baseline (background dataset).

# Arguments
- `model::Function`: Prediction model (x -> y)
- `x::Vector`: Instance to explain
- `X_background::Matrix`: Background dataset for baseline (features × N)
- `n_samples::Int`: Number of samples for approximation

# Returns
- `shap_values::Vector{Float64}`: SHAP value for each feature
- `base_value::Float64`: Baseline prediction (expected value)
"""
function compute_shap_values(model::Function, x::AbstractVector,
                            X_background::AbstractMatrix; n_samples::Int=100)
    
    n_features = length(x)
    n_background = size(X_background, 2)
    
    # Baseline prediction (average over background)
    base_predictions = [model(X_background[:, i:i]) for i in 1:n_background]
    base_value = mean(base_predictions)
    
    # Initialize SHAP values
    shap_values = zeros(Float64, n_features)
    
    # Kernel SHAP: sample feature coalitions
    for _ in 1:n_samples
        # Random coalition (subset of features)
        coalition = rand(Bool, n_features)
        
        # Create masked instance
        x_masked = copy(x)
        for i in 1:n_features
            if !coalition[i]
                # Replace with random background value
                x_masked[i] = X_background[i, rand(1:n_background)]
            end
        end
        
        # Prediction with coalition
        pred_with = model(reshape(x_masked, :, 1))[1]
        
        # Compute marginal contribution for each feature
        for i in 1:n_features
            if coalition[i]
                # Remove feature i from coalition
                x_without = copy(x_masked)
                x_without[i] = X_background[i, rand(1:n_background)]
                pred_without = model(reshape(x_without, :, 1))[1]
                
                # Marginal contribution
                contribution = pred_with - pred_without
                
                # Weight by coalition size (Shapley kernel)
                coalition_size = sum(coalition)
                weight = 1.0 / (n_features * binomial(n_features - 1, coalition_size - 1))
                
                shap_values[i] += weight * contribution
            end
        end
    end
    
    # Normalize
    shap_values ./= n_samples
    
    return shap_values, base_value
end

"""
    explain_prediction(model, x, X_background, feature_names)

Generate human-readable explanation of prediction.

# Arguments
- `model::Function`: Prediction model
- `x::Vector`: Instance to explain
- `X_background::Matrix`: Background dataset
- `feature_names::Vector{String}`: Names of features

# Returns
- `explanation::Dict`: Explanation with SHAP values and interpretation
"""
function explain_prediction(model::Function, x::AbstractVector,
                           X_background::AbstractMatrix,
                           feature_names::Vector{String})
    
    # Compute SHAP values
    shap_values, base_value = compute_shap_values(model, x, X_background)
    
    # Prediction
    prediction = model(reshape(x, :, 1))[1]
    
    # Sort features by absolute SHAP value
    sorted_indices = sortperm(abs.(shap_values), rev=true)
    
    explanation = Dict(
        "prediction" => prediction,
        "base_value" => base_value,
        "shap_values" => shap_values,
        "feature_names" => feature_names,
        "top_features" => []
    )
    
    println("\n" * "="^60)
    println("Prediction Explanation (SHAP)")
    println("="^60)
    println("Prediction: $(round(prediction, digits=4))")
    println("Base value: $(round(base_value, digits=4))")
    println("\nTop Contributing Features:")
    println("-"^60)
    
    for (rank, idx) in enumerate(sorted_indices[1:min(10, length(sorted_indices))])
        feature_name = feature_names[idx]
        shap_val = shap_values[idx]
        feature_val = x[idx]
        
        direction = shap_val > 0 ? "increases" : "decreases"
        
        println("$rank. $feature_name = $(round(feature_val, digits=3))")
        println("   SHAP: $(round(shap_val, digits=4)) ($direction prediction)")
        
        push!(explanation["top_features"], Dict(
            "name" => feature_name,
            "value" => feature_val,
            "shap" => shap_val,
            "rank" => rank
        ))
    end
    
    println("="^60)
    
    return explanation
end

# ============================================================================
# Attention Visualization (for Transformers)
# ============================================================================

"""
    visualize_attention(attention_weights, patch_indices)

Visualize attention weights from transformer model.

# Arguments
- `attention_weights::Matrix`: Attention weights (num_patches × num_patches)
- `patch_indices::Vector{Tuple}`: 3D indices of patches

# Returns
- `attention_map::Dict`: Attention visualization data
"""
function visualize_attention(attention_weights::AbstractMatrix,
                            patch_indices::Vector{<:Tuple})
    
    n_patches = size(attention_weights, 1)
    
    # Compute attention statistics
    attention_map = Dict(
        "weights" => attention_weights,
        "patch_indices" => patch_indices,
        "max_attention" => maximum(attention_weights),
        "mean_attention" => mean(attention_weights),
        "attention_entropy" => compute_attention_entropy(attention_weights)
    )
    
    # Find most attended patches
    avg_attention_received = vec(mean(attention_weights, dims=1))
    top_patches = sortperm(avg_attention_received, rev=true)[1:min(10, n_patches)]
    
    println("\n" * "="^60)
    println("Attention Visualization")
    println("="^60)
    println("Number of patches: $n_patches")
    println("Max attention: $(round(attention_map["max_attention"], digits=4))")
    println("Mean attention: $(round(attention_map["mean_attention"], digits=4))")
    println("Attention entropy: $(round(attention_map["attention_entropy"], digits=4))")
    println("\nMost Attended Patches:")
    
    for (rank, idx) in enumerate(top_patches)
        patch_idx = patch_indices[idx]
        attention_score = avg_attention_received[idx]
        println("$rank. Patch $patch_idx: $(round(attention_score, digits=4))")
    end
    
    println("="^60)
    
    return attention_map
end

"""
    compute_attention_entropy(attention_weights)

Compute entropy of attention distribution (measure of focus).

High entropy = diffuse attention
Low entropy = focused attention
"""
function compute_attention_entropy(attention_weights::AbstractMatrix)
    n_patches = size(attention_weights, 1)
    
    total_entropy = 0.0
    for i in 1:n_patches
        # Attention distribution for patch i
        attn_dist = attention_weights[i, :]
        attn_dist = attn_dist ./ sum(attn_dist)  # Normalize
        
        # Entropy: -Σ p log(p)
        entropy = -sum(attn_dist .* log.(attn_dist .+ 1e-10))
        total_entropy += entropy
    end
    
    return total_entropy / n_patches
end

# ============================================================================
# Feature Importance (Permutation Importance)
# ============================================================================

"""
    feature_importance(model, X_test, y_test; n_repeats=10)

Compute feature importance using permutation importance.

Measures how much model performance decreases when a feature is randomly shuffled.

# Arguments
- `model::Function`: Prediction model
- `X_test::Matrix`: Test inputs (features × samples)
- `y_test::Vector`: Test targets
- `n_repeats::Int`: Number of permutation repeats

# Returns
- `importances::Vector{Float64}`: Importance score for each feature
- `importances_std::Vector{Float64}`: Standard deviation of importance
"""
function feature_importance(model::Function, X_test::AbstractMatrix,
                           y_test::AbstractVector; n_repeats::Int=10)
    
    n_features, n_samples = size(X_test)
    
    # Baseline performance (MSE)
    y_pred_baseline = [model(X_test[:, i:i])[1] for i in 1:n_samples]
    baseline_mse = mean((y_test .- y_pred_baseline).^2)
    
    # Compute importance for each feature
    importances = zeros(Float64, n_features)
    importances_std = zeros(Float64, n_features)
    
    for feat_idx in 1:n_features
        mse_increases = Float64[]
        
        for _ in 1:n_repeats
            # Permute feature
            X_permuted = copy(X_test)
            X_permuted[feat_idx, :] = shuffle(X_permuted[feat_idx, :])
            
            # Predict with permuted feature
            y_pred_permuted = [model(X_permuted[:, i:i])[1] for i in 1:n_samples]
            permuted_mse = mean((y_test .- y_pred_permuted).^2)
            
            # Importance = increase in MSE
            push!(mse_increases, permuted_mse - baseline_mse)
        end
        
        importances[feat_idx] = mean(mse_increases)
        importances_std[feat_idx] = std(mse_increases)
    end
    
    return importances, importances_std
end

"""
    plot_feature_importance(importances, feature_names)

Print feature importance ranking.
"""
function plot_feature_importance(importances::AbstractVector,
                                feature_names::Vector{String})
    
    sorted_indices = sortperm(importances, rev=true)
    
    println("\n" * "="^60)
    println("Feature Importance (Permutation)")
    println("="^60)
    
    for (rank, idx) in enumerate(sorted_indices)
        feature_name = feature_names[idx]
        importance = importances[idx]
        
        # Visual bar
        bar_length = Int(round(importance * 50 / maximum(importances)))
        bar = "█" ^ bar_length
        
        println("$rank. $feature_name")
        println("   $(round(importance, digits=4)) $bar")
    end
    
    println("="^60)
end

# ============================================================================
# Counterfactual Explanations
# ============================================================================

"""
    counterfactual_explanation(model, x, target_value, feature_names; 
                              max_changes=3, lr=0.1, max_iter=100)

Generate counterfactual explanation: minimal changes to achieve target prediction.

"What would need to change for the prediction to be X?"

# Arguments
- `model::Function`: Prediction model
- `x::Vector`: Original instance
- `target_value::Float64`: Desired prediction
- `feature_names::Vector{String}`: Feature names
- `max_changes::Int`: Maximum number of features to change
- `lr::Float64`: Learning rate for optimization
- `max_iter::Int`: Maximum iterations

# Returns
- `counterfactual::Vector{Float64}`: Modified instance
- `changes::Dict`: Description of changes
"""
function counterfactual_explanation(model::Function, x::AbstractVector,
                                   target_value::Float64,
                                   feature_names::Vector{String};
                                   max_changes::Int=3, lr::Float64=0.1,
                                   max_iter::Int=100)
    
    n_features = length(x)
    
    # Initialize counterfactual
    x_cf = copy(x)
    
    # Sparsity mask (which features to change)
    change_mask = zeros(Bool, n_features)
    
    # Select features to change (greedy)
    for _ in 1:max_changes
        best_feature = 0
        best_improvement = -Inf
        
        for feat_idx in 1:n_features
            if change_mask[feat_idx]
                continue  # Already selected
            end
            
            # Try changing this feature
            x_test = copy(x_cf)
            x_test[feat_idx] += lr * sign(target_value - model(reshape(x_cf, :, 1))[1])
            
            pred_test = model(reshape(x_test, :, 1))[1]
            improvement = -abs(pred_test - target_value)
            
            if improvement > best_improvement
                best_improvement = improvement
                best_feature = feat_idx
            end
        end
        
        if best_feature > 0
            change_mask[best_feature] = true
        end
    end
    
    # Optimize selected features
    for iter in 1:max_iter
        pred_current = model(reshape(x_cf, :, 1))[1]
        
        if abs(pred_current - target_value) < 0.01
            break  # Close enough
        end
        
        # Gradient-free optimization (finite differences)
        for feat_idx in 1:n_features
            if !change_mask[feat_idx]
                continue
            end
            
            # Finite difference gradient
            x_plus = copy(x_cf)
            x_plus[feat_idx] += 0.01
            pred_plus = model(reshape(x_plus, :, 1))[1]
            
            gradient = (pred_plus - pred_current) / 0.01
            
            # Update toward target
            x_cf[feat_idx] += lr * gradient * sign(target_value - pred_current)
        end
    end
    
    # Describe changes
    changes = Dict{String, Any}()
    
    println("\n" * "="^60)
    println("Counterfactual Explanation")
    println("="^60)
    println("Original prediction: $(round(model(reshape(x, :, 1))[1], digits=4))")
    println("Target prediction: $(round(target_value, digits=4))")
    println("Counterfactual prediction: $(round(model(reshape(x_cf, :, 1))[1], digits=4))")
    println("\nRequired Changes:")
    println("-"^60)
    
    for feat_idx in 1:n_features
        if change_mask[feat_idx]
            feature_name = feature_names[feat_idx]
            original_val = x[feat_idx]
            new_val = x_cf[feat_idx]
            change = new_val - original_val
            
            println("• $feature_name: $(round(original_val, digits=3)) → $(round(new_val, digits=3))")
            println("  (change: $(round(change, digits=3)))")
            
            changes[feature_name] = Dict(
                "original" => original_val,
                "counterfactual" => new_val,
                "change" => change
            )
        end
    end
    
    println("="^60)
    
    return x_cf, changes
end

# ============================================================================
# Integrated Gradients (for Neural Networks)
# ============================================================================

"""
    integrated_gradients(model, x, baseline; n_steps=50)

Compute integrated gradients for attribution.

Integrates gradients along path from baseline to input.

# Arguments
- `model::Function`: Differentiable model
- `x::Vector`: Input instance
- `baseline::Vector`: Baseline instance (e.g., zeros)
- `n_steps::Int`: Number of integration steps

# Returns
- `attributions::Vector{Float64}`: Attribution for each feature
"""
function integrated_gradients(model::Function, x::AbstractVector,
                             baseline::AbstractVector; n_steps::Int=50)
    
    n_features = length(x)
    
    # Path from baseline to input
    alphas = range(0, 1, length=n_steps)
    
    # Accumulate gradients
    integrated_grads = zeros(Float64, n_features)
    
    for α in alphas
        # Interpolated input
        x_interp = baseline .+ α .* (x .- baseline)
        
        # Compute gradient (finite differences)
        for feat_idx in 1:n_features
            x_plus = copy(x_interp)
            x_plus[feat_idx] += 0.01
            
            pred_plus = model(reshape(x_plus, :, 1))[1]
            pred_current = model(reshape(x_interp, :, 1))[1]
            
            gradient = (pred_plus - pred_current) / 0.01
            integrated_grads[feat_idx] += gradient
        end
    end
    
    # Average and scale by (x - baseline)
    integrated_grads ./= n_steps
    attributions = integrated_grads .* (x .- baseline)
    
    return attributions
end

end # module
