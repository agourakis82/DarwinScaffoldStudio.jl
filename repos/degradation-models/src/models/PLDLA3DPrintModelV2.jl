"""
    PLDLA3DPrintModelV2

Improved model for 3D-printed PLDLA 70:30 scaffolds.

KEY INSIGHT FROM DATA:
======================
Looking at the experimental data more carefully:

PLDLA (Mn0=51.3):  51.3 → 25.4 → 18.3 → 7.9  (fractions: 1.00 → 0.50 → 0.36 → 0.15)
TEC1  (Mn0=45.0):  45.0 → 19.3 → 11.7 → 8.1  (fractions: 1.00 → 0.43 → 0.26 → 0.18)
TEC2  (Mn0=32.7):  32.7 → 15.0 → 12.6 → 6.6  (fractions: 1.00 → 0.46 → 0.38 → 0.20)

Observations:
1. TEC reduces initial Mn (processing effect)
2. Relative degradation rates are SIMILAR across formulations
3. All reach ~15-20% of initial Mn by 90 days

APPROACH:
Simple physics model with per-dataset calibration of Mn0 effect.

Author: Darwin Scaffold Studio  
Date: December 2025
"""

using Statistics, Random, Printf

# =============================================================================
# EXPERIMENTAL DATA
# =============================================================================

const PLDLA_DATA_V2 = [
    (id="PLDLA", Mn0=51.285, times=[0.0, 30.0, 60.0, 90.0],
        Mn=[51.285, 25.447, 18.313, 7.904], TEC=0.0),
    (id="TEC1", Mn0=44.998, times=[0.0, 30.0, 60.0, 90.0],
        Mn=[44.998, 19.261, 11.676, 8.128], TEC=1.0),
    (id="TEC2", Mn0=32.733, times=[0.0, 30.0, 60.0, 90.0],
        Mn=[32.733, 14.953, 12.574, 6.643], TEC=2.0),
]

# =============================================================================
# MODEL
# =============================================================================

mutable struct PLDLA3DPrintModelV2 <: AbstractDegradationModel
    # Kinetic parameters (fitted)
    k::Float64            # Hydrolysis rate constant
    n::Float64            # Time exponent  
    a::Float64            # Autocatalysis strength
    τ::Float64            # Autocatalysis time constant

    # Per-formulation adjustments
    δ_TEC1::Float64       # TEC1 rate adjustment
    δ_TEC2::Float64       # TEC2 rate adjustment

    trained::Bool
end

function PLDLA3DPrintModelV2()
    return PLDLA3DPrintModelV2(
        0.025,   # k
        0.65,    # n
        0.8,     # a
        25.0,    # τ
        1.0,     # δ_TEC1 (multiplicative)
        1.0,     # δ_TEC2
        false
    )
end

# =============================================================================
# PREDICTION
# =============================================================================

"""
    predict(model, Mn0, t; TEC=0.0)

Predict Mn at time t.

Model: Mn(t) = Mn0 × exp(-k_eff × t^n)

where k_eff = k × δ_TEC × (1 + a×(1 - exp(-t/τ)))
"""
function predict(model::PLDLA3DPrintModelV2, Mn0::Float64, t::Float64;
    TEC::Float64=0.0, kwargs...)
    if t <= 0.0
        return Mn0
    end

    # TEC adjustment factor
    δ = if TEC >= 1.5
        model.δ_TEC2
    elseif TEC >= 0.5
        model.δ_TEC1
    else
        1.0
    end

    # Autocatalysis: rate increases as degradation proceeds
    autocatalysis = 1.0 + model.a * (1.0 - exp(-t / model.τ))

    # Effective rate
    k_eff = model.k * δ * autocatalysis

    # Degradation
    fraction = exp(-k_eff * t^model.n)

    return max(fraction * Mn0, 0.5)
end

function predict(model::PLDLA3DPrintModelV2, material::String, Mn0::Float64, t::Float64; kwargs...)
    TEC = if occursin("TEC2", material)
        2.0
    elseif occursin("TEC1", material)
        1.0
    else
        0.0
    end
    return predict(model, Mn0, t; TEC=TEC, kwargs...)
end

# =============================================================================
# TRAINING
# =============================================================================

function flatten_params_v2(model::PLDLA3DPrintModelV2)
    return [model.k, model.n, model.a, model.τ, model.δ_TEC1, model.δ_TEC2]
end

function set_params_v2!(model::PLDLA3DPrintModelV2, p::Vector{Float64})
    model.k = abs(p[1])
    model.n = clamp(p[2], 0.3, 1.0)
    model.a = abs(p[3])
    model.τ = abs(p[4]) + 1.0
    model.δ_TEC1 = abs(p[5])
    model.δ_TEC2 = abs(p[6])
end

function compute_loss_v2(model::PLDLA3DPrintModelV2)
    L = 0.0
    n = 0

    for d in PLDLA_DATA_V2
        for (i, t) in enumerate(d.times)
            if t == 0.0
                continue
            end

            Mn_pred = predict(model, d.Mn0, t, TEC=d.TEC)
            Mn_exp = d.Mn[i]

            rel_err = (Mn_pred - Mn_exp) / Mn_exp
            L += rel_err^2
            n += 1
        end
    end

    return L / max(n, 1)
end

function train(::Type{PLDLA3DPrintModelV2}; epochs::Int=3000,
    population_size::Int=30, σ::Float64=0.05,
    lr::Float64=0.01, verbose::Bool=true)
    Random.seed!(42)

    if verbose
        println("\n" * "="^60)
        println("  PLDLA 3D-Print Model V2 - Training")
        println("="^60)
        println("  Simple physics with per-formulation calibration")
    end

    model = PLDLA3DPrintModelV2()
    θ = flatten_params_v2(model)
    np = length(θ)

    if verbose
        println("  Parameters: $np (k, n, a, τ, δ_TEC1, δ_TEC2)")
    end

    # Adam
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
            set_params_v2!(model, θ .+ σ .* noise[:, i])
            push!(losses_pos, compute_loss_v2(model))

            set_params_v2!(model, θ .- σ .* noise[:, i])
            push!(losses_neg, compute_loss_v2(model))
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

        set_params_v2!(model, θ)
        loss = compute_loss_v2(model)

        if loss < best_loss
            best_loss = loss
            best_θ = copy(θ)
        end

        if verbose && (epoch % 500 == 0 || epoch == 1)
            rmse = sqrt(loss) * 100
            @printf("    Epoch %4d: RMSE = %.1f%% (Accuracy: %.1f%%)\n",
                epoch, rmse, 100 - rmse)
        end
    end

    set_params_v2!(model, best_θ)
    model.trained = true

    if verbose
        final_rmse = sqrt(compute_loss_v2(model)) * 100

        println("\n  Training complete!")
        @printf("  Final: RMSE = %.1f%%, Accuracy = %.1f%%\n", final_rmse, 100 - final_rmse)

        println("\n  Learned parameters:")
        @printf("    k = %.4f (hydrolysis rate)\n", model.k)
        @printf("    n = %.3f (time exponent)\n", model.n)
        @printf("    a = %.3f (autocatalysis strength)\n", model.a)
        @printf("    τ = %.1f days (autocatalysis time constant)\n", model.τ)
        @printf("    δ_TEC1 = %.3f (TEC1 rate multiplier)\n", model.δ_TEC1)
        @printf("    δ_TEC2 = %.3f (TEC2 rate multiplier)\n", model.δ_TEC2)
    end

    return model
end

# =============================================================================
# VALIDATION
# =============================================================================

function validate(model::PLDLA3DPrintModelV2; verbose::Bool=true)
    results = Dict{String,Any}()
    all_errors = Float64[]

    if verbose
        println("\n" * "="^60)
        println("  PLDLA 3D-Print Model V2 - Validation")
        println("="^60)
    end

    for d in PLDLA_DATA_V2
        if verbose
            tec_str = d.TEC > 0 ? " + $(Int(d.TEC))% TEC" : " (pure)"
            println("\n  $(d.id)$tec_str, Mn0 = $(d.Mn0) kg/mol")
            println("  " * "-"^50)
            @printf("  %8s  %12s  %12s  %8s\n", "Time(d)", "Experimental", "Predicted", "Error")
        end

        errors = Float64[]

        for (i, t) in enumerate(d.times)
            Mn_pred = predict(model, d.Mn0, t, TEC=d.TEC)
            Mn_exp = d.Mn[i]

            err = t > 0 ? abs(Mn_pred - Mn_exp) / Mn_exp * 100 : 0.0

            if t > 0
                push!(errors, err)
                push!(all_errors, err)
            end

            if verbose
                @printf("  %8.0f  %12.2f  %12.2f  %7.1f%%\n", t, Mn_exp, Mn_pred, err)
            end
        end

        acc = 100 - mean(errors)
        if verbose
            @printf("  → Accuracy: %.1f%%\n", acc)
        end

        results[d.id] = (accuracy=acc, mape=mean(errors))
    end

    global_acc = 100 - mean(all_errors)

    if verbose
        println("\n" * "="^60)
        @printf("  GLOBAL ACCURACY: %.1f%%\n", global_acc)

        if global_acc >= 90
            println("  ✓ Excellent! Ready for scaffold design optimization.")
        elseif global_acc >= 85
            println("  ✓ Very good for engineering applications.")
        elseif global_acc >= 80
            println("  ✓ Good accuracy.")
        else
            println("  ~ Consider model improvements.")
        end
        println("="^60)
    end

    results["overall_accuracy"] = global_acc
    return results
end

# =============================================================================
# UTILITIES
# =============================================================================

function estimate_halflife(model::PLDLA3DPrintModelV2, Mn0::Float64; TEC::Float64=0.0)
    target = Mn0 / 2
    t_low, t_high = 0.0, 200.0

    for _ in 1:50
        t_mid = (t_low + t_high) / 2
        Mn_mid = predict(model, Mn0, t_mid, TEC=TEC)

        if Mn_mid > target
            t_low = t_mid
        else
            t_high = t_mid
        end

        if abs(t_high - t_low) < 0.1
            break
        end
    end

    return (t_low + t_high) / 2
end

function predict_curve(model::PLDLA3DPrintModelV2, Mn0::Float64;
    TEC::Float64=0.0, t_max::Float64=90.0, n_points::Int=50)
    times = collect(range(0, t_max, length=n_points))
    Mn = [predict(model, Mn0, t, TEC=TEC) for t in times]
    return (times=times, Mn=Mn)
end
