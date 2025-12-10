"""
    UniversalModelV2

Improved Universal PLA degradation model.

IMPROVEMENTS:
1. Ratio-specific embeddings (like material embeddings)
2. Better feature normalization
3. Multi-task learning approach
4. Deeper network with better initialization

Author: Darwin Scaffold Studio
Date: December 2025
"""

# Load database
include(joinpath(@__DIR__, "..", "..", "data", "literature_degradation_database.jl"))

# Ratio categories
const RATIO_CATEGORIES = [
    (50, 50),   # PDLLA
    (70, 30),   # PLDLA standard
    (85, 15),   # PLDLA high L
    (96, 4),    # PLA96
    (100, 0),   # PLLA pure
]

const N_RATIO_CATS = 5
const RATIO_EMBED_DIM = 6

# =============================================================================
# MODEL STRUCTURE
# =============================================================================

mutable struct UniversalModelV2 <: AbstractDegradationModel
    # Ratio embeddings
    ratio_embeddings::Matrix{Float64}

    # Network layers
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
    W3::Matrix{Float64}
    b3::Vector{Float64}

    trained::Bool
end

function UniversalModelV2(; n_hidden::Int=64)
    # 8 base features + 6 ratio embedding = 14 input
    n_input = 8 + RATIO_EMBED_DIM

    ratio_embeddings = randn(N_RATIO_CATS, RATIO_EMBED_DIM) * 0.3

    W1 = randn(n_hidden, n_input) * sqrt(2.0 / n_input)
    b1 = zeros(n_hidden)
    W2 = randn(n_hidden, n_hidden) * sqrt(2.0 / n_hidden)
    b2 = zeros(n_hidden)
    W3 = randn(1, n_hidden) * 0.1
    b3 = [0.5]

    return UniversalModelV2(ratio_embeddings, W1, b1, W2, b2, W3, b3, false)
end

# =============================================================================
# RATIO EMBEDDING
# =============================================================================

function get_ratio_category(ratio_L::Real)
    if ratio_L <= 55
        return 1  # PDLLA
    elseif ratio_L <= 77
        return 2  # PLDLA 70:30
    elseif ratio_L <= 90
        return 3  # PLDLA 85:15
    elseif ratio_L <= 98
        return 4  # PLA96
    else
        return 5  # PLLA pure
    end
end

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

function extract_features_v2(ratio_L::Float64, Mn0::Float64, t::Float64,
                             T::Float64, Xc0::Float64, condition::Symbol)
    # Normalize all features to roughly [0, 1]

    # Time features (multiple scales)
    t_norm = min(t / 100.0, 3.0)  # Saturate at 300 days
    t_sqrt = sqrt(t) / 15.0

    # Molecular weight (log scale)
    Mn0_norm = log10(max(Mn0, 1.0)) / 3.0  # log10(1000) = 3

    # Crystallinity strongly affects degradation
    # More crystalline = slower degradation
    Xc_factor = 1.0 - 0.8 * (Xc0 / 100.0)

    # D-content: more D = faster degradation (amorphous)
    D_content = (100.0 - ratio_L) / 100.0
    D_factor = 1.0 + 2.0 * D_content  # PDLLA degrades ~3x faster

    # Temperature factor (Arrhenius-like)
    T_factor = exp(0.08 * (T - 37.0))  # 8% per degree

    # Condition: in vivo often has enzymatic contribution
    cond_factor = condition == :in_vivo ? 1.5 : 1.0

    # Combined degradation potential
    degrad_potential = D_factor * Xc_factor * T_factor * cond_factor
    degrad_potential = min(degrad_potential / 5.0, 1.0)  # Normalize

    return Float64[
        t_norm,           # 1. Time (normalized)
        t_sqrt,           # 2. sqrt(time)
        Mn0_norm,         # 3. Initial MW (log)
        Xc_factor,        # 4. Crystallinity factor
        D_content,        # 5. D-lactide content
        T_factor / 2.0,   # 6. Temperature factor
        condition == :in_vivo ? 1.0 : 0.0,  # 7. In vivo flag
        degrad_potential, # 8. Combined degradation potential
    ]
end

# =============================================================================
# FORWARD PASS
# =============================================================================

function forward_v2(model::UniversalModelV2, ratio_L::Float64, Mn0::Float64,
                    t::Float64, T::Float64, Xc0::Float64, condition::Symbol)
    # Get ratio embedding
    cat = get_ratio_category(ratio_L)
    ratio_embed = model.ratio_embeddings[cat, :]

    # Get base features
    base_features = extract_features_v2(ratio_L, Mn0, t, T, Xc0, condition)

    # Concatenate
    x = vcat(base_features, ratio_embed)

    # Forward pass with swish activation
    h1 = model.W1 * x .+ model.b1
    a1 = h1 ./ (1.0 .+ exp.(-h1))  # swish

    h2 = model.W2 * a1 .+ model.b2
    a2 = h2 ./ (1.0 .+ exp.(-h2)) .+ a1  # swish + residual

    out = model.W3 * a2 .+ model.b3

    # Sigmoid output
    fraction = 1.0 / (1.0 + exp(-out[1]))

    return fraction
end

# =============================================================================
# PREDICT
# =============================================================================

function predict(model::UniversalModelV2, ratio_L::Float64, Mn0::Float64, t::Float64;
                 T::Float64=37.0, Xc0::Float64=-1.0, condition::Symbol=:in_vitro,
                 kwargs...)
    if t == 0.0
        return Mn0
    end

    # Estimate crystallinity if not provided
    if Xc0 < 0
        # PLLA ~ 50%, PDLLA ~ 0%
        Xc0 = max(0.0, 60.0 * (ratio_L - 50) / 50.0)
    end

    fraction = forward_v2(model, ratio_L, Mn0, t, T, Xc0, condition)
    return max(fraction * Mn0, 0.5)
end

function predict(model::UniversalModelV2, material::String, Mn0::Float64, t::Float64; kwargs...)
    ratio_L = if occursin("100", material) || (occursin("PLLA", uppercase(material)) && !occursin("PD", uppercase(material)))
        100.0
    elseif occursin("96", material)
        96.0
    elseif occursin("85", material)
        85.0
    elseif occursin("70", material) || occursin("PLDLA", uppercase(material))
        70.0
    elseif occursin("50", material) || occursin("PDLLA", uppercase(material))
        50.0
    else
        70.0
    end
    return predict(model, ratio_L, Mn0, t; kwargs...)
end

# =============================================================================
# TRAINING
# =============================================================================

function flatten_params_v2(model::UniversalModelV2)
    return vcat(
        vec(model.ratio_embeddings),
        vec(model.W1), model.b1,
        vec(model.W2), model.b2,
        vec(model.W3), model.b3
    )
end

function set_params_v2!(model::UniversalModelV2, params::Vector{Float64})
    idx = 1

    n_embed = N_RATIO_CATS * RATIO_EMBED_DIM
    model.ratio_embeddings[:] = reshape(params[idx:idx+n_embed-1], N_RATIO_CATS, RATIO_EMBED_DIM)
    idx += n_embed

    n1, m1 = size(model.W1)
    model.W1[:] = reshape(params[idx:idx+n1*m1-1], n1, m1)
    idx += n1*m1
    model.b1[:] = params[idx:idx+n1-1]
    idx += n1

    n2, m2 = size(model.W2)
    model.W2[:] = reshape(params[idx:idx+n2*m2-1], n2, m2)
    idx += n2*m2
    model.b2[:] = params[idx:idx+n2-1]
    idx += n2

    n3, m3 = size(model.W3)
    model.W3[:] = reshape(params[idx:idx+n3*m3-1], n3, m3)
    idx += n3*m3
    model.b3[:] = params[idx:idx+1-1]
end

function compute_loss_v2(model::UniversalModelV2)
    L = 0.0
    n = 0

    for d in DEGRADATION_DATABASE
        for (i, t) in enumerate(d.times)
            if t == 0.0
                continue
            end

            Mn_pred = predict(model, Float64(d.ratio_L), d.Mn0, t,
                             T=d.T, Xc0=d.Xc0, condition=d.condition)
            Mn_exp = d.Mn[i]

            # Relative squared error
            rel_err = (Mn_pred - Mn_exp) / Mn_exp
            L += rel_err^2
            n += 1
        end
    end

    return L / max(n, 1)
end

function train(::Type{UniversalModelV2}; epochs::Int=3000,
               population_size::Int=40, σ::Float64=0.04,
               lr::Float64=0.002, verbose::Bool=true)
    Random.seed!(42)

    if verbose
        println("\n  Training UniversalModelV2...")
        println("  With ratio embeddings for 5 categories")
    end

    model = UniversalModelV2(n_hidden=64)
    θ = flatten_params_v2(model)
    n_params = length(θ)

    if verbose
        println("  Parameters: $n_params")
        println("  Training on $(n_datasets()) datasets, $(sum(length(d.times)-1 for d in DEGRADATION_DATABASE)) data points")
    end

    # Adam
    m = zeros(n_params)
    v = zeros(n_params)
    β1, β2 = 0.9, 0.999
    ϵ = 1e-8

    best_loss = Inf
    best_θ = copy(θ)

    for epoch in 1:epochs
        noise = randn(n_params, population_size)

        losses_pos = Float64[]
        losses_neg = Float64[]

        for i in 1:population_size
            set_params_v2!(model, θ .+ σ .* noise[:, i])
            push!(losses_pos, compute_loss_v2(model))

            set_params_v2!(model, θ .- σ .* noise[:, i])
            push!(losses_neg, compute_loss_v2(model))
        end

        gradient = zeros(n_params)
        for i in 1:population_size
            gradient .+= (losses_pos[i] - losses_neg[i]) .* noise[:, i]
        end
        gradient ./= (2 * population_size * σ)

        m .= β1 .* m .+ (1 - β1) .* gradient
        v .= β2 .* v .+ (1 - β2) .* gradient.^2
        m_hat = m ./ (1 - β1^epoch)
        v_hat = v ./ (1 - β2^epoch)
        θ .-= lr .* m_hat ./ (sqrt.(v_hat) .+ ϵ)

        set_params_v2!(model, θ)
        loss = compute_loss_v2(model)

        if loss < best_loss
            best_loss = loss
            best_θ = copy(θ)
        end

        if verbose && (epoch % 500 == 0 || epoch == 1)
            rmse = sqrt(loss) * 100
            @printf("    Epoch %4d: RMSE = %.1f%%\n", epoch, rmse)
        end
    end

    set_params_v2!(model, best_θ)
    model.trained = true

    if verbose
        final_rmse = sqrt(compute_loss_v2(model)) * 100
        @printf("  Training complete! Final RMSE: %.1f%%\n", final_rmse)
    end

    return model
end

# =============================================================================
# VALIDATION
# =============================================================================

function validate(model::UniversalModelV2; verbose::Bool=true)
    results = Dict{String, NamedTuple}()

    if verbose
        println("\n" * "="^70)
        println("  UNIVERSAL MODEL V2 VALIDATION")
        println("="^70)
    end

    for d in DEGRADATION_DATABASE
        errors = Float64[]

        for (i, t) in enumerate(d.times)
            if t == 0.0
                continue
            end

            Mn_pred = predict(model, Float64(d.ratio_L), d.Mn0, t,
                             T=d.T, Xc0=d.Xc0, condition=d.condition)
            Mn_exp = d.Mn[i]

            err = abs(Mn_pred - Mn_exp) / Mn_exp * 100
            push!(errors, err)
        end

        mape = isempty(errors) ? 0.0 : mean(errors)

        results[d.id] = (
            material = d.material,
            ratio_L = d.ratio_L,
            condition = d.condition,
            mape = mape,
            accuracy = 100 - mape
        )
    end

    if verbose
        # Summary by ratio category
        for (cat_idx, (L, D)) in enumerate(RATIO_CATEGORIES)
            subset = [(k, v) for (k, v) in results if get_ratio_category(v.ratio_L) == cat_idx]
            if isempty(subset)
                continue
            end

            avg_acc = mean([v.accuracy for (_, v) in subset])
            status = avg_acc >= 85 ? "✓" : avg_acc >= 70 ? "~" : "✗"

            println("\n  L:DL ≈ $L:$D ($(length(subset)) datasets)")
            for (id, r) in sort(subset, by=x->-x[2].accuracy)
                @printf("    %-25s: %.1f%% accuracy\n", id, r.accuracy)
            end
            @printf("    → Category average: %.1f%% %s\n", avg_acc, status)
        end

        # Global
        global_mape = mean([r.mape for (_, r) in results])
        global_acc = 100 - global_mape

        println("\n" * "="^70)
        @printf("  GLOBAL: %.1f%% accuracy (MAPE = %.1f%%)\n", global_acc, global_mape)

        if global_acc >= 80
            println("  ✓ Universal model ready for use!")
        elseif global_acc >= 70
            println("  ~ Acceptable for most applications")
        else
            println("  → Needs improvement")
        end
        println("="^70)
    end

    return results
end

# Convenience
train_universal_v2(; kwargs...) = train(UniversalModelV2; kwargs...)
