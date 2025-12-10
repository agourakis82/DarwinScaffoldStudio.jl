"""
    UniversalModelV4

Crystallinity-Aware Universal PLA Degradation Model.

KEY INSIGHT FROM DATA ANALYSIS:
================================
The degradation rate varies 10x WITHIN the same L:DL ratio category.
The dominant factor is CRYSTALLINITY (Xc), not just L:DL ratio.

Within PLLA/PLA96:
- Xc = 0-10%:  k ≈ 0.02-0.03 day⁻¹ (t½ ≈ 24-35 days)
- Xc = 30-50%: k ≈ 0.006-0.01 day⁻¹ (t½ ≈ 70-120 days)
- Xc = 50-65%: k ≈ 0.003-0.004 day⁻¹ (t½ ≈ 170-230 days)

APPROACH:
1. Crystallinity as primary kinetic driver
2. L:DL ratio determines baseline + crystallinity potential
3. Morphology factor (scaffold/film/fiber)
4. Condition factor (in vivo vs in vitro)

Author: Darwin Scaffold Studio
Date: December 2025
"""

using Statistics, Random, Printf

# Load database only if not already loaded
if !@isdefined(DEGRADATION_DATABASE)
    include(joinpath(@__DIR__, "..", "..", "data", "literature_degradation_database.jl"))
end

# =============================================================================
# CRYSTALLINITY-BASED KINETICS
# =============================================================================

"""
    compute_k_effective(ratio_L, Xc, T, condition, morphology)

Compute effective degradation rate constant.

The model is based on:
    k_eff = k_base(ratio_L) × f(Xc) × f(T) × f(condition) × f(morphology)

Where:
- k_base increases with D-content (amorphous regions)
- f(Xc) = exp(-α × Xc) - crystallinity slows degradation exponentially
- f(T) = Arrhenius temperature dependence
- f(condition) = 1.3 for in vivo (enzymatic contribution)
- f(morphology) = surface area effects
"""
function compute_k_base(ratio_L::Real)
    # D-content determines base rate
    D_content = (100.0 - ratio_L) / 100.0

    # Empirical base rates (from literature regression)
    # PDLLA (50:50): k ≈ 0.037 day⁻¹
    # PLDLA (70:30): k ≈ 0.021 day⁻¹
    # PLA96 (96:4):  k ≈ 0.008 day⁻¹ (at low Xc)
    # PLLA (100:0):  k ≈ 0.005 day⁻¹ (at low Xc)

    k_base = 0.005 + 0.06 * D_content^1.5

    return k_base
end

function crystallinity_factor(Xc::Real)
    # Exponential slowdown with crystallinity
    # At Xc=0: factor = 1.0
    # At Xc=50: factor ≈ 0.15
    # At Xc=65: factor ≈ 0.05
    α = 0.038  # Fitted from data
    return exp(-α * Xc)
end

function temperature_factor(T::Real)
    # Arrhenius: k(T) = k(37) × exp(Ea/R × (1/310 - 1/T))
    # Ea/R ≈ 7000 K for PLA hydrolysis
    T_ref = 37.0
    Ea_R = 7000.0

    T_K = T + 273.15
    T_ref_K = T_ref + 273.15

    return exp(Ea_R * (1 / T_ref_K - 1 / T_K))
end

function condition_factor(condition::Symbol)
    # In vivo has enzymatic contribution
    return condition == :in_vivo ? 1.4 : 1.0
end

function morphology_factor(morphology::Symbol)
    # Surface area affects water ingress
    return if morphology == :scaffold
        1.3  # Porous = faster
    elseif morphology == :fiber
        1.2
    elseif morphology == :film
        1.0
    else
        1.0
    end
end

# =============================================================================
# MODEL STRUCTURE
# =============================================================================

mutable struct UniversalModelV4 <: AbstractDegradationModel
    # Learnable parameters
    k_scale::Float64          # Global scale factor
    α_crystallinity::Float64  # Crystallinity sensitivity
    n_autocatalysis::Float64  # Autocatalytic exponent

    # Neural correction network (small)
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}

    trained::Bool
end

function UniversalModelV4()
    n_input = 6
    n_hidden = 24

    return UniversalModelV4(
        1.0,    # k_scale
        0.038,  # α_crystallinity (literature value)
        0.7,    # n_autocatalysis
        randn(n_hidden, n_input) * 0.1,
        zeros(n_hidden),
        randn(1, n_hidden) * 0.05,
        [0.0],
        false
    )
end

# =============================================================================
# FORWARD PASS
# =============================================================================

function forward_v4(model::UniversalModelV4, ratio_L::Float64, Mn0::Float64,
    t::Float64, T::Float64, Xc::Float64, condition::Symbol,
    morphology::Symbol)
    if t <= 0.0
        return 1.0
    end

    # Physics-based rate constant
    k_base = compute_k_base(ratio_L)
    k_xc = exp(-model.α_crystallinity * Xc)
    k_T = temperature_factor(T)
    k_cond = condition_factor(condition)
    k_morph = morphology_factor(morphology)

    k_eff = model.k_scale * k_base * k_xc * k_T * k_cond * k_morph

    # Autocatalytic acceleration (increases with time)
    # Simple model: rate increases as degradation proceeds
    autocatalysis = 1.0 + 0.5 * (1.0 - exp(-t / 50.0))

    # Degradation (modified first-order)
    fraction_physics = exp(-k_eff * autocatalysis * t^model.n_autocatalysis)

    # Neural correction (small adjustment)
    D_content = (100.0 - ratio_L) / 100.0
    x = Float64[
        t/200.0,                         # Normalized time
        log10(max(Mn0, 10.0))/2.5,       # Log MW
        D_content,                          # D-lactide content
        Xc/60.0,                          # Normalized crystallinity
        (T-37.0)/20.0,                 # Temperature deviation
        fraction_physics,                   # Physics prediction
    ]

    h1 = model.W1 * x .+ model.b1
    a1 = tanh.(h1)
    correction = (model.W2*a1.+model.b2)[1]

    # Small bounded correction
    correction = tanh(correction) * 0.2

    final_frac = clamp(fraction_physics * (1.0 + correction), 0.01, 1.0)

    return final_frac
end

# =============================================================================
# PREDICT
# =============================================================================

function predict(model::UniversalModelV4, ratio_L::Float64, Mn0::Float64, t::Float64;
    T::Float64=37.0, Xc0::Float64=-1.0, condition::Symbol=:in_vitro,
    morphology::Symbol=:film, kwargs...)
    if t == 0.0
        return Mn0
    end

    # Estimate crystallinity if not provided
    if Xc0 < 0.0
        # Default estimate based on L-content
        # PLLA can crystallize (40-60%), PDLLA cannot (0%)
        if ratio_L >= 96
            Xc0 = 35.0  # Typical semi-crystalline PLLA
        elseif ratio_L >= 85
            Xc0 = 15.0
        elseif ratio_L >= 70
            Xc0 = 5.0
        else
            Xc0 = 0.0  # PDLLA is amorphous
        end
    end

    fraction = forward_v4(model, ratio_L, Mn0, t, T, Xc0, condition, morphology)
    return fraction * Mn0
end

function predict(model::UniversalModelV4, material::String, Mn0::Float64, t::Float64; kwargs...)
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
# PARAMETER HANDLING
# =============================================================================

function flatten_params_v4(model::UniversalModelV4)
    return vcat(
        [model.k_scale, model.α_crystallinity, model.n_autocatalysis],
        vec(model.W1),
        model.b1,
        vec(model.W2),
        model.b2
    )
end

function set_params_v4!(model::UniversalModelV4, params::Vector{Float64})
    model.k_scale = abs(params[1])
    model.α_crystallinity = abs(params[2])
    model.n_autocatalysis = clamp(params[3], 0.3, 1.5)

    idx = 4
    n1, m1 = size(model.W1)
    model.W1[:] = reshape(params[idx:idx+n1*m1-1], n1, m1)
    idx += n1 * m1

    model.b1[:] = params[idx:idx+n1-1]
    idx += n1

    n2, m2 = size(model.W2)
    model.W2[:] = reshape(params[idx:idx+n2*m2-1], n2, m2)
    idx += n2 * m2

    model.b2[:] = params[idx:idx+length(model.b2)-1]
end

# =============================================================================
# LOSS COMPUTATION
# =============================================================================

function compute_loss_v4(model::UniversalModelV4)
    L = 0.0
    n = 0

    for d in DEGRADATION_DATABASE
        for (i, t) in enumerate(d.times)
            if t == 0.0
                continue
            end

            morphology = if occursin("scaffold", lowercase(string(get(d, :source, ""))))
                :scaffold
            elseif occursin("fiber", lowercase(string(get(d, :source, ""))))
                :fiber
            else
                :film
            end

            Mn_pred = predict(model, Float64(d.ratio_L), d.Mn0, t,
                T=d.T, Xc0=d.Xc0, condition=d.condition,
                morphology=morphology)
            Mn_exp = d.Mn[i]

            # Relative error
            rel_err = (Mn_pred - Mn_exp) / Mn_exp
            L += rel_err^2
            n += 1
        end
    end

    # Regularization to keep parameters reasonable
    reg = 0.001 * (model.k_scale - 1.0)^2
    reg += 0.01 * (model.α_crystallinity - 0.038)^2

    return L / max(n, 1) + reg
end

# =============================================================================
# TRAINING
# =============================================================================

function train(::Type{UniversalModelV4}; epochs::Int=4000,
    population_size::Int=60, σ::Float64=0.025,
    lr::Float64=0.004, verbose::Bool=true)
    Random.seed!(42)

    if verbose
        println("\n  Training UniversalModelV4 (Crystallinity-Aware)...")
        println("  Key insight: Xc explains 10x variation in k within same L:DL ratio")
    end

    model = UniversalModelV4()
    θ = flatten_params_v4(model)
    np = length(θ)

    if verbose
        println("  Parameters: $np (3 physics + $(np-3) neural)")
        println("  Training on $(n_datasets()) datasets")
    end

    # Adam optimizer
    m = zeros(np)
    v = zeros(np)
    β1, β2 = 0.9, 0.999
    ϵ = 1e-8

    best_loss = Inf
    best_θ = copy(θ)
    patience = 0
    max_patience = 500

    for epoch in 1:epochs
        noise = randn(np, population_size)

        losses_pos = Float64[]
        losses_neg = Float64[]

        for i in 1:population_size
            set_params_v4!(model, θ .+ σ .* noise[:, i])
            push!(losses_pos, compute_loss_v4(model))

            set_params_v4!(model, θ .- σ .* noise[:, i])
            push!(losses_neg, compute_loss_v4(model))
        end

        gradient = zeros(np)
        for i in 1:population_size
            gradient .+= (losses_pos[i] - losses_neg[i]) .* noise[:, i]
        end
        gradient ./= (2 * population_size * σ)

        m .= β1 .* m .+ (1 - β1) .* gradient
        v .= β2 .* v .+ (1 - β2) .* gradient .^ 2
        m_hat = m ./ (1 - β1^epoch)
        v_hat = v ./ (1 - β2^epoch)
        θ .-= lr .* m_hat ./ (sqrt.(v_hat) .+ ϵ)

        set_params_v4!(model, θ)
        loss = compute_loss_v4(model)

        if loss < best_loss - 1e-6
            best_loss = loss
            best_θ = copy(θ)
            patience = 0
        else
            patience += 1
        end

        if verbose && (epoch % 500 == 0 || epoch == 1)
            rmse = sqrt(loss) * 100
            @printf("    Epoch %4d: RMSE = %.1f%%\n", epoch, rmse)
        end

        if patience >= max_patience
            if verbose
                println("    Early stopping at epoch $epoch")
            end
            break
        end
    end

    set_params_v4!(model, best_θ)
    model.trained = true

    if verbose
        final_rmse = sqrt(compute_loss_v4(model)) * 100
        @printf("  Training complete! Final RMSE: %.1f%%\n", final_rmse)

        println("\n  Learned physics parameters:")
        @printf("    k_scale = %.3f\n", model.k_scale)
        @printf("    α_crystallinity = %.4f\n", model.α_crystallinity)
        @printf("    n_autocatalysis = %.3f\n", model.n_autocatalysis)
    end

    return model
end

# =============================================================================
# VALIDATION
# =============================================================================

function validate(model::UniversalModelV4; verbose::Bool=true)
    results = Dict{String,Any}()
    all_errors = Float64[]

    if verbose
        println("\n" * "="^70)
        println("  UNIVERSAL MODEL V4 VALIDATION (Crystallinity-Aware)")
        println("="^70)
    end

    per_dataset = Dict{String,NamedTuple}()

    # Group by crystallinity ranges instead of just ratio
    xc_groups = [
        ("Low Xc (0-15%)", d -> d.Xc0 <= 15),
        ("Medium Xc (15-40%)", d -> 15 < d.Xc0 <= 40),
        ("High Xc (40-70%)", d -> d.Xc0 > 40),
    ]

    for d in DEGRADATION_DATABASE
        errors = Float64[]
        predictions = Float64[]
        actuals = Float64[]

        morphology = :film  # Default

        for (i, t) in enumerate(d.times)
            if t == 0.0
                continue
            end

            Mn_pred = predict(model, Float64(d.ratio_L), d.Mn0, t,
                T=d.T, Xc0=d.Xc0, condition=d.condition,
                morphology=morphology)
            Mn_exp = d.Mn[i]

            err = abs(Mn_pred - Mn_exp) / Mn_exp * 100
            push!(errors, err)
            push!(predictions, Mn_pred)
            push!(actuals, Mn_exp)
            push!(all_errors, err)
        end

        mape = isempty(errors) ? 0.0 : mean(errors)

        per_dataset[d.id] = (
            material=d.material,
            ratio_L=d.ratio_L,
            Xc0=d.Xc0,
            condition=d.condition,
            mape=mape,
            accuracy=100 - mape,
            predictions=predictions,
            actuals=actuals
        )
    end

    if verbose
        # Show by crystallinity group
        for (group_name, filter_fn) in xc_groups
            subset_ids = [d.id for d in DEGRADATION_DATABASE if filter_fn(d)]
            subset = [(id, per_dataset[id]) for id in subset_ids if haskey(per_dataset, id)]

            if isempty(subset)
                continue
            end

            avg_acc = mean([v.accuracy for (_, v) in subset])
            status = avg_acc >= 80 ? "✓" : avg_acc >= 65 ? "~" : "✗"

            println("\n  $group_name ($(length(subset)) datasets)")
            for (id, r) in sort(subset, by=x -> -x[2].accuracy)
                ratio_str = "$(round(Int, r.ratio_L)):$(round(Int, 100-r.ratio_L))"
                @printf("    %-22s [%s, Xc=%2.0f%%]: %5.1f%% accuracy\n",
                    id, ratio_str, r.Xc0, r.accuracy)
            end
            @printf("    → Group average: %.1f%% %s\n", avg_acc, status)
        end

        # Global statistics
        global_mape = mean(all_errors)
        global_acc = 100 - global_mape

        println("\n" * "="^70)
        @printf("  GLOBAL: %.1f%% accuracy (MAPE = %.1f%%)\n", global_acc, global_mape)

        if global_acc >= 75
            println("  ✓ Universal model suitable for scaffold design!")
        elseif global_acc >= 65
            println("  ~ Acceptable with known limitations")
        else
            println("  → Consider per-dataset calibration")
        end
        println("="^70)
    end

    results["per_dataset"] = per_dataset
    results["overall_accuracy"] = 100 - mean(all_errors)
    results["mean_mape"] = mean(all_errors)

    return results
end

# Convenience
train_universal_v4(; kwargs...) = train(UniversalModelV4; kwargs...)
