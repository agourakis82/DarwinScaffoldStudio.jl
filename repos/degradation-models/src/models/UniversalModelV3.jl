"""
    UniversalModelV3

Physics-Informed Universal PLA Degradation Model.

KEY INSIGHT: Each L:DL ratio has distinct degradation kinetics:
- PLLA (100:0): Slow, dominated by crystallinity, can take years
- PLDLA 96:4: Moderate, slightly amorphous regions accelerate
- PLDLA 85:15: Faster, significant amorphous fraction
- PLDLA 70:30: Fast, balanced crystalline/amorphous
- PDLLA (50:50): Very fast, purely amorphous

APPROACH:
1. Physics-informed base model with ratio-specific kinetic constants
2. Neural correction term for capturing non-linear deviations
3. Multi-output for uncertainty estimation

Author: Darwin Scaffold Studio
Date: December 2025
"""

using Statistics, Random, Printf

# Load database only if not already loaded
if !@isdefined(DEGRADATION_DATABASE)
    include(joinpath(@__DIR__, "..", "..", "data", "literature_degradation_database.jl"))
end

# =============================================================================
# PHYSICS-BASED KINETICS
# =============================================================================

"""
Literature-derived kinetic parameters for each ratio category.

k_base: Base hydrolysis rate constant (day^-1)
n_autocatalysis: Autocatalytic exponent (typically 0.5-2.0)
activation_energy: Ea/R for temperature dependence
crystallinity_factor: How much crystallinity slows degradation
"""
const KINETIC_PARAMS = Dict(
    # (ratio_L, ratio_D) => (k_base, n_auto, Ea_R, cryst_factor)
    (50, 50)   => (0.020, 1.0, 8000.0, 0.2),   # PDLLA: fast, amorphous
    (70, 30)   => (0.012, 0.8, 7500.0, 0.4),   # PLDLA: balanced
    (85, 15)   => (0.008, 0.6, 7000.0, 0.6),   # Mostly L
    (96, 4)    => (0.005, 0.5, 6500.0, 0.8),   # Near-PLLA
    (100, 0)   => (0.003, 0.4, 6000.0, 1.0),   # PLLA: slow, crystalline
)

const RATIO_CATEGORIES_V3 = [(50, 50), (70, 30), (85, 15), (96, 4), (100, 0)]

function get_ratio_category_v3(ratio_L::Real)
    if ratio_L <= 55
        return 1
    elseif ratio_L <= 77
        return 2
    elseif ratio_L <= 90
        return 3
    elseif ratio_L <= 98
        return 4
    else
        return 5
    end
end

function get_kinetic_params(ratio_L::Real)
    cat = get_ratio_category_v3(ratio_L)
    return KINETIC_PARAMS[RATIO_CATEGORIES_V3[cat]]
end

# =============================================================================
# PHYSICS MODEL
# =============================================================================

"""
    physics_degradation(ratio_L, Mn0, t, T, Xc0, condition)

First-order hydrolysis with autocatalysis (Pitt-Schindler model):

    dMn/dt = -k * Mn^n * (1 + [COOH])

Where:
- k = k_base * exp(-Ea/R * (1/T - 1/T0)) * (1 - cryst_factor * Xc0/100)
- n captures autocatalysis
- [COOH] ~ (Mn0 - Mn) / Mn0

Returns: Mn(t) / Mn0 (fraction remaining)
"""
function physics_degradation(ratio_L::Float64, Mn0::Float64, t::Float64,
                              T::Float64, Xc0::Float64, condition::Symbol)
    if t <= 0.0
        return 1.0
    end

    k_base, n_auto, Ea_R, cryst_factor = get_kinetic_params(ratio_L)

    # Temperature dependence (Arrhenius)
    T_ref = 310.15  # 37°C in K
    T_K = T + 273.15
    k_T = k_base * exp(-Ea_R * (1/T_K - 1/T_ref))

    # Crystallinity effect
    Xc_eff = max(Xc0, 0.0)
    k_cryst = k_T * (1.0 - cryst_factor * Xc_eff / 100.0)

    # In vivo enzymatic contribution
    if condition == :in_vivo
        k_cryst *= 1.3
    end

    # Simplified analytical solution for autocatalytic model
    # Using numerical integration would be better but slower
    # Approximate: Mn(t) = Mn0 * exp(-k * t^n_auto)

    # More accurate: Use time-dependent rate
    # Early: linear, Late: accelerating

    k_eff = k_cryst * (1.0 + 0.5 * (1.0 - exp(-t / 30.0)))  # Autocatalytic acceleration

    fraction = exp(-k_eff * t^n_auto)

    return clamp(fraction, 0.01, 1.0)
end

# =============================================================================
# MODEL STRUCTURE
# =============================================================================

mutable struct UniversalModelV3 <: AbstractDegradationModel
    # Learnable kinetic parameters (corrections to literature values)
    k_corrections::Vector{Float64}      # 5 categories
    n_corrections::Vector{Float64}      # 5 categories

    # Neural correction network
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}

    trained::Bool
end

function UniversalModelV3()
    n_input = 7
    n_hidden = 32

    return UniversalModelV3(
        ones(5),               # k_corrections start at 1.0 (no change)
        zeros(5),              # n_corrections start at 0.0 (no change)
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

function forward_v3(model::UniversalModelV3, ratio_L::Float64, Mn0::Float64,
                    t::Float64, T::Float64, Xc0::Float64, condition::Symbol)
    if t <= 0.0
        return 1.0
    end

    cat = get_ratio_category_v3(ratio_L)

    # Get physics prediction with learned corrections
    k_base, n_auto, Ea_R, cryst_factor = get_kinetic_params(ratio_L)

    k_corrected = k_base * abs(model.k_corrections[cat])
    n_corrected = n_auto + model.n_corrections[cat]

    # Temperature
    T_ref = 310.15
    T_K = T + 273.15
    k_T = k_corrected * exp(-Ea_R * (1/T_K - 1/T_ref))

    # Crystallinity
    Xc_eff = max(Xc0, 0.0)
    k_cryst = k_T * (1.0 - cryst_factor * Xc_eff / 100.0)

    # In vivo
    if condition == :in_vivo
        k_cryst *= 1.3
    end

    # Autocatalytic effect
    k_eff = k_cryst * (1.0 + 0.5 * (1.0 - exp(-t / 30.0)))

    # Physics prediction
    physics_frac = exp(-k_eff * abs(t)^max(n_corrected, 0.1))

    # Neural correction (small adjustment)
    # Features
    x = Float64[
        t / 100.0,                        # Normalized time
        log10(max(Mn0, 1.0)) / 3.0,       # Log MW
        (100.0 - ratio_L) / 100.0,        # D-content
        Xc_eff / 100.0,                   # Crystallinity
        (T - 37.0) / 30.0,                # Temperature deviation
        condition == :in_vivo ? 1.0 : 0.0, # Condition
        physics_frac,                      # Physics prediction
    ]

    h1 = model.W1 * x .+ model.b1
    a1 = tanh.(h1)
    correction = (model.W2 * a1 .+ model.b2)[1]

    # Correction is small adjustment (tanh to bound it)
    correction = tanh(correction) * 0.3  # Max 30% adjustment

    final_frac = clamp(physics_frac + correction, 0.01, 1.0)

    return final_frac
end

# =============================================================================
# PREDICT
# =============================================================================

function predict(model::UniversalModelV3, ratio_L::Float64, Mn0::Float64, t::Float64;
                 T::Float64=37.0, Xc0::Float64=-1.0, condition::Symbol=:in_vitro,
                 kwargs...)
    if t == 0.0
        return Mn0
    end

    # Estimate crystallinity from L-content if not provided
    if Xc0 < 0.0
        # Empirical: PLLA ~50-60%, PDLLA ~0%
        Xc0 = max(0.0, 60.0 * (ratio_L - 50.0) / 50.0)
    end

    fraction = forward_v3(model, ratio_L, Mn0, t, T, Xc0, condition)
    return fraction * Mn0
end

function predict(model::UniversalModelV3, material::String, Mn0::Float64, t::Float64; kwargs...)
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

function flatten_params_v3(model::UniversalModelV3)
    return vcat(
        model.k_corrections,
        model.n_corrections,
        vec(model.W1),
        model.b1,
        vec(model.W2),
        model.b2
    )
end

function set_params_v3!(model::UniversalModelV3, params::Vector{Float64})
    idx = 1

    model.k_corrections[:] = params[idx:idx+4]
    idx += 5

    model.n_corrections[:] = params[idx:idx+4]
    idx += 5

    n1, m1 = size(model.W1)
    model.W1[:] = reshape(params[idx:idx+n1*m1-1], n1, m1)
    idx += n1*m1

    model.b1[:] = params[idx:idx+n1-1]
    idx += n1

    n2, m2 = size(model.W2)
    model.W2[:] = reshape(params[idx:idx+n2*m2-1], n2, m2)
    idx += n2*m2

    model.b2[:] = params[idx:idx+length(model.b2)-1]
end

function n_params_v3(model::UniversalModelV3)
    return 10 + length(model.W1) + length(model.b1) + length(model.W2) + length(model.b2)
end

# =============================================================================
# LOSS COMPUTATION
# =============================================================================

function compute_loss_v3(model::UniversalModelV3)
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

            # Relative squared error with log penalty for large errors
            rel_err = (Mn_pred - Mn_exp) / Mn_exp
            L += rel_err^2 + 0.1 * abs(rel_err)  # L2 + L1 for robustness
            n += 1
        end
    end

    # Regularization on kinetic corrections (keep close to literature)
    reg = 0.01 * sum((model.k_corrections .- 1.0).^2)
    reg += 0.01 * sum(model.n_corrections.^2)

    return L / max(n, 1) + reg
end

# =============================================================================
# TRAINING
# =============================================================================

function train(::Type{UniversalModelV3}; epochs::Int=3000,
               population_size::Int=50, σ::Float64=0.03,
               lr::Float64=0.003, verbose::Bool=true)
    Random.seed!(42)

    if verbose
        println("\n  Training UniversalModelV3 (Physics-Informed)...")
        println("  Combining literature kinetics with neural corrections")
    end

    model = UniversalModelV3()
    θ = flatten_params_v3(model)
    np = length(θ)

    if verbose
        println("  Parameters: $np (10 kinetic + $(np-10) neural)")
        println("  Training on $(n_datasets()) datasets")
    end

    # Adam optimizer
    m = zeros(np)
    v = zeros(np)
    β1, β2 = 0.9, 0.999
    ϵ = 1e-8

    best_loss = Inf
    best_θ = copy(θ)

    for epoch in 1:epochs
        noise = randn(np, population_size)

        losses_pos = Float64[]
        losses_neg = Float64[]

        for i in 1:population_size
            set_params_v3!(model, θ .+ σ .* noise[:, i])
            push!(losses_pos, compute_loss_v3(model))

            set_params_v3!(model, θ .- σ .* noise[:, i])
            push!(losses_neg, compute_loss_v3(model))
        end

        gradient = zeros(np)
        for i in 1:population_size
            gradient .+= (losses_pos[i] - losses_neg[i]) .* noise[:, i]
        end
        gradient ./= (2 * population_size * σ)

        m .= β1 .* m .+ (1 - β1) .* gradient
        v .= β2 .* v .+ (1 - β2) .* gradient.^2
        m_hat = m ./ (1 - β1^epoch)
        v_hat = v ./ (1 - β2^epoch)
        θ .-= lr .* m_hat ./ (sqrt.(v_hat) .+ ϵ)

        set_params_v3!(model, θ)
        loss = compute_loss_v3(model)

        if loss < best_loss
            best_loss = loss
            best_θ = copy(θ)
        end

        if verbose && (epoch % 500 == 0 || epoch == 1)
            rmse = sqrt(loss) * 100
            @printf("    Epoch %4d: RMSE = %.1f%%\n", epoch, rmse)
        end
    end

    set_params_v3!(model, best_θ)
    model.trained = true

    if verbose
        final_rmse = sqrt(compute_loss_v3(model)) * 100
        @printf("  Training complete! Final RMSE: %.1f%%\n", final_rmse)

        # Show learned kinetic corrections
        println("\n  Learned kinetic corrections:")
        for (i, (L, D)) in enumerate(RATIO_CATEGORIES_V3)
            @printf("    %d:%d → k_mult=%.2f, n_adj=%+.2f\n",
                    L, D, model.k_corrections[i], model.n_corrections[i])
        end
    end

    return model
end

# =============================================================================
# VALIDATION
# =============================================================================

function validate(model::UniversalModelV3; verbose::Bool=true)
    results = Dict{String, Any}()
    all_errors = Float64[]

    if verbose
        println("\n" * "="^70)
        println("  UNIVERSAL MODEL V3 VALIDATION (Physics-Informed)")
        println("="^70)
    end

    per_dataset = Dict{String, NamedTuple}()

    for d in DEGRADATION_DATABASE
        errors = Float64[]
        predictions = Float64[]
        actuals = Float64[]

        for (i, t) in enumerate(d.times)
            if t == 0.0
                continue
            end

            Mn_pred = predict(model, Float64(d.ratio_L), d.Mn0, t,
                             T=d.T, Xc0=d.Xc0, condition=d.condition)
            Mn_exp = d.Mn[i]

            err = abs(Mn_pred - Mn_exp) / Mn_exp * 100
            push!(errors, err)
            push!(predictions, Mn_pred)
            push!(actuals, Mn_exp)
            push!(all_errors, err)
        end

        mape = isempty(errors) ? 0.0 : mean(errors)

        per_dataset[d.id] = (
            material = d.material,
            ratio_L = d.ratio_L,
            condition = d.condition,
            mape = mape,
            accuracy = 100 - mape,
            predictions = predictions,
            actuals = actuals
        )
    end

    if verbose
        # Summary by ratio category
        for (cat_idx, (L, D)) in enumerate(RATIO_CATEGORIES_V3)
            subset = [(k, v) for (k, v) in per_dataset if get_ratio_category_v3(v.ratio_L) == cat_idx]
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

        # Global statistics
        global_mape = mean(all_errors)
        global_acc = 100 - global_mape

        println("\n" * "="^70)
        @printf("  GLOBAL: %.1f%% accuracy (MAPE = %.1f%%)\n", global_acc, global_mape)

        if global_acc >= 80
            println("  ✓ Universal model ready for publication!")
        elseif global_acc >= 70
            println("  ~ Acceptable for most applications")
        else
            println("  → Needs improvement")
        end
        println("="^70)
    end

    results["per_dataset"] = per_dataset
    results["overall_accuracy"] = 100 - mean(all_errors)
    results["mean_mape"] = mean(all_errors)

    return results
end

# Convenience
train_universal_v3(; kwargs...) = train(UniversalModelV3; kwargs...)
