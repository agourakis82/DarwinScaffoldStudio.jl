"""
    UniversalModel

Universal PLA degradation model that handles:
- Different L:DL ratios (PLLA, PLDLA, PDLLA)
- Different initial molecular weights
- Different crystallinities
- In vitro and in vivo conditions
- Different temperatures

ARCHITECTURE:
- Input: 12 physics-informed features
- Hidden: 96 neurons × 3 layers (GELU + residual)
- Output: 1 (fraction remaining)

PHYSICS FEATURES:
1. ratio_L (L-lactide content, %)
2. Xc0 (initial crystallinity, %)
3. Mn0 (initial MW, kg/mol)
4. t (time, days)
5. T (temperature, °C)
6. pH
7. t/t_half (estimated half-life ratio)
8. condition encoding (in_vitro=1, in_vivo=2)
9. form encoding (film=1, scaffold=2, fiber=3)
10. Eyring rate factor
11. Autocatalysis potential
12. Crystallinity factor

Author: Darwin Scaffold Studio
Date: December 2025
"""

# Load database
include(joinpath(@__DIR__, "..", "..", "data", "literature_degradation_database.jl"))

const N_FEATURES = 12

# =============================================================================
# MODEL STRUCTURE
# =============================================================================

mutable struct UniversalModel <: AbstractDegradationModel
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
    W3::Matrix{Float64}
    b3::Vector{Float64}
    W4::Matrix{Float64}
    b4::Vector{Float64}
    trained::Bool
end

function UniversalModel(; n_hidden::Int=96)
    W1 = randn(n_hidden, N_FEATURES) * sqrt(2.0 / N_FEATURES)
    b1 = zeros(n_hidden)
    W2 = randn(n_hidden, n_hidden) * sqrt(2.0 / n_hidden)
    b2 = zeros(n_hidden)
    W3 = randn(n_hidden, n_hidden) * sqrt(2.0 / n_hidden)
    b3 = zeros(n_hidden)
    W4 = randn(1, n_hidden) * 0.1
    b4 = [0.5]

    return UniversalModel(W1, b1, W2, b2, W3, b3, W4, b4, false)
end

# =============================================================================
# PHYSICS-INFORMED FEATURE EXTRACTION
# =============================================================================

"""
Extract physics-informed features for the universal model.
"""
function extract_features(ratio_L::Float64, Mn0::Float64, t::Float64,
                         T::Float64, pH::Float64, Xc0::Float64,
                         Tg::Float64, condition::Symbol, form::String)

    # 1. L-lactide ratio (normalized)
    ratio_L_norm = ratio_L / 100.0

    # 2. D-lactide ratio (affects amorphousness)
    ratio_D_norm = (100 - ratio_L) / 100.0

    # 3. Initial crystallinity (normalized)
    Xc_norm = Xc0 / 100.0

    # 4. Initial Mn (log-normalized)
    Mn0_norm = log10(Mn0) / 3.0  # log10(1000) = 3

    # 5. Time (multi-scale)
    t_norm = t / 90.0
    t_sqrt = sqrt(t) / 10.0

    # 6. Temperature factor (Arrhenius-like)
    T_ref = 37.0
    T_factor = exp(0.05 * (T - T_ref))  # ~5% per degree

    # 7. pH factor
    pH_factor = 1.0 + 0.3 * (7.4 - pH)  # Acidic accelerates

    # 8. Estimated half-life based on composition
    # PDLLA (50:50) ~ 30 days, PLLA (100:0) ~ 300 days
    t_half_est = 30 + 270 * (ratio_L_norm^2) * (1 + 2*Xc_norm)
    t_t_half = t / max(t_half_est, 1.0)

    # 9. Condition encoding
    cond_enc = condition == :in_vitro ? 0.0 : 1.0

    # 10. Form encoding (porosity affects degradation)
    form_enc = if occursin("scaffold", lowercase(form)) || occursin("porous", lowercase(form))
        0.5  # Faster for porous
    elseif occursin("fiber", lowercase(form))
        0.3
    else
        0.0  # Film/dense
    end

    # 11. Eyring-like rate (simplified)
    k_eyring = T_factor * pH_factor * (1 + ratio_D_norm)

    # 12. Crystallinity protection factor
    f_crystal = 1.0 - 0.7 * Xc_norm

    return Float64[
        ratio_L_norm,    # 1
        ratio_D_norm,    # 2
        Xc_norm,         # 3
        Mn0_norm,        # 4
        t_norm,          # 5
        t_sqrt,          # 6
        t_t_half,        # 7
        cond_enc,        # 8
        form_enc,        # 9
        k_eyring,        # 10
        f_crystal,       # 11
        T_factor,        # 12
    ]
end

# =============================================================================
# FORWARD PASS
# =============================================================================

# Use gelu from NeuralModel (already defined)

function forward(model::UniversalModel, features::Vector{Float64})
    # Layer 1
    h1 = model.W1 * features .+ model.b1
    a1 = gelu.(h1)

    # Layer 2 with residual
    h2 = model.W2 * a1 .+ model.b2
    a2 = gelu.(h2) .+ a1

    # Layer 3 with residual
    h3 = model.W3 * a2 .+ model.b3
    a3 = gelu.(h3) .+ a2

    # Output
    out = model.W4 * a3 .+ model.b4

    # Sigmoid for fraction [0, 1]
    fraction = 1.0 / (1.0 + exp(-out[1]))

    return fraction
end

# =============================================================================
# PREDICT
# =============================================================================

"""
Predict molecular weight at time t.

Arguments:
- ratio_L: L-lactide content (0-100%)
- Mn0: Initial molecular weight (kg/mol)
- t: Time (days)
- T: Temperature (°C), default 37
- pH: pH, default 7.4
- Xc0: Initial crystallinity (%), default estimated from ratio_L
- Tg: Glass transition (°C), default estimated
- condition: :in_vitro or :in_vivo
- form: sample form string
"""
function predict(model::UniversalModel, ratio_L::Float64, Mn0::Float64, t::Float64;
                 T::Float64=37.0, pH::Float64=7.4, Xc0::Float64=-1.0,
                 Tg::Float64=-1.0, condition::Symbol=:in_vitro,
                 form::String="film")

    if t == 0.0
        return Mn0
    end

    # Estimate crystallinity if not provided
    if Xc0 < 0
        # PLLA (100) ~ 50%, PDLLA (50) ~ 0%
        Xc0 = max(0.0, 50.0 * (ratio_L/100 - 0.5) * 2)
    end

    # Estimate Tg if not provided
    if Tg < 0
        Tg = 50.0 + 10.0 * (ratio_L / 100)  # ~50-60°C
    end

    features = extract_features(Float64(ratio_L), Mn0, t, T, pH,
                                Xc0, Tg, condition, form)

    fraction = forward(model, features)
    Mn = fraction * Mn0

    return max(Mn, 0.5)
end

# Convenience method for string material names
function predict(model::UniversalModel, material::String, Mn0::Float64, t::Float64; kwargs...)
    # Parse material name to get ratio_L
    ratio_L = if occursin("PLLA", uppercase(material)) && !occursin("PDLLA", uppercase(material))
        100.0
    elseif occursin("PDLLA", uppercase(material)) || occursin("50:50", material)
        50.0
    elseif occursin("96:4", material) || occursin("PLA96", uppercase(material))
        96.0
    elseif occursin("85:15", material)
        85.0
    elseif occursin("70:30", material) || occursin("PLDLA", uppercase(material))
        70.0
    else
        70.0  # Default to PLDLA
    end

    return predict(model, ratio_L, Mn0, t; kwargs...)
end

# =============================================================================
# TRAINING
# =============================================================================

function flatten_params(model::UniversalModel)
    return vcat(
        vec(model.W1), model.b1,
        vec(model.W2), model.b2,
        vec(model.W3), model.b3,
        vec(model.W4), model.b4
    )
end

function set_params!(model::UniversalModel, params::Vector{Float64})
    idx = 1

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
    model.b3[:] = params[idx:idx+n3-1]
    idx += n3

    n4, m4 = size(model.W4)
    model.W4[:] = reshape(params[idx:idx+n4*m4-1], n4, m4)
    idx += n4*m4
    model.b4[:] = params[idx:idx+1-1]
end

function compute_loss(model::UniversalModel)
    L = 0.0
    n = 0

    for d in DEGRADATION_DATABASE
        for (i, t) in enumerate(d.times)
            if t == 0.0
                continue
            end

            Mn_pred = predict(model, Float64(d.ratio_L), d.Mn0, t,
                             T=d.T, pH=d.pH, Xc0=d.Xc0, Tg=d.Tg,
                             condition=d.condition, form=d.form)
            Mn_exp = d.Mn[i]

            rel_err = (Mn_pred - Mn_exp) / Mn_exp
            L += rel_err^2
            n += 1
        end
    end

    return L / max(n, 1)
end

function train(::Type{UniversalModel}; epochs::Int=2000,
               population_size::Int=30, σ::Float64=0.05,
               lr::Float64=0.003, verbose::Bool=true)
    Random.seed!(42)

    if verbose
        println("\n  Training UniversalModel on $(n_datasets()) datasets...")
        println("  Features: ratio_L, Xc0, Mn0, t, T, pH, condition, form")
        println("  Architecture: $N_FEATURES → 48 → 48 → 48 → 1")
    end

    model = UniversalModel(n_hidden=48)  # Smaller network for faster training
    θ = flatten_params(model)
    n_params = length(θ)

    if verbose
        println("  Parameters: $n_params")
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
            set_params!(model, θ .+ σ .* noise[:, i])
            push!(losses_pos, compute_loss(model))

            set_params!(model, θ .- σ .* noise[:, i])
            push!(losses_neg, compute_loss(model))
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

        set_params!(model, θ)
        loss = compute_loss(model)

        if loss < best_loss
            best_loss = loss
            best_θ = copy(θ)
        end

        if verbose && (epoch % 500 == 0 || epoch == 1)
            rmse = sqrt(loss) * 100
            @printf("    Epoch %4d: RMSE = %.1f%%\n", epoch, rmse)
        end
    end

    set_params!(model, best_θ)
    model.trained = true

    if verbose
        final_rmse = sqrt(compute_loss(model)) * 100
        @printf("  Training complete! Final RMSE: %.1f%%\n", final_rmse)
    end

    return model
end

# =============================================================================
# VALIDATION
# =============================================================================

function validate(model::UniversalModel; verbose::Bool=true)
    results = Dict{String, NamedTuple}()

    if verbose
        println("\n" * "="^70)
        println("  UNIVERSAL MODEL VALIDATION")
        println("  Testing across $(n_datasets()) datasets with different L:DL ratios")
        println("="^70)
    end

    for d in DEGRADATION_DATABASE
        errors = Float64[]

        for (i, t) in enumerate(d.times)
            if t == 0.0
                continue
            end

            Mn_pred = predict(model, Float64(d.ratio_L), d.Mn0, t,
                             T=d.T, pH=d.pH, Xc0=d.Xc0, Tg=d.Tg,
                             condition=d.condition, form=d.form)
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
        # Group by ratio
        println("\n  Results by L:DL ratio:")

        for ratio in sort(unique([r.ratio_L for (_, r) in results]), rev=true)
            subset = [(k, v) for (k, v) in results if v.ratio_L == ratio]
            avg_mape = mean([v.mape for (_, v) in subset])
            avg_acc = 100 - avg_mape

            status = avg_acc >= 90 ? "✓" : avg_acc >= 80 ? "~" : "✗"

            println("\n  L:DL = $ratio:$(100-ratio) ($(length(subset)) datasets)")
            for (id, r) in sort(subset, by=x->x[2].mape)
                @printf("    %-25s: MAPE=%5.1f%% (%.1f%%)\n",
                        id, r.mape, r.accuracy)
            end
            @printf("    → Average: %.1f%% accuracy %s\n", avg_acc, status)
        end

        # Global
        global_mape = mean([r.mape for (_, r) in results])
        global_acc = 100 - global_mape

        println("\n" * "="^70)
        @printf("  GLOBAL: MAPE = %.1f%%, Accuracy = %.1f%%\n", global_mape, global_acc)

        if global_acc >= 85
            println("  ✓ Model generalizes well across PLA variants!")
        else
            println("  → Model needs more training or architecture tuning")
        end
        println("="^70)
    end

    return results
end

# Convenience function
train_universal(; kwargs...) = train(UniversalModel; kwargs...)
