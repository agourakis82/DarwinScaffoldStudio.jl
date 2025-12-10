"""
    PLDLANeuralODEFast

Fast Physics-Informed Neural ODE for PLDLA degradation.

Optimized version using analytical approximation instead of RK4.

PHYSICS MODEL (Wang-Han simplified):
====================================
For autocatalytic hydrolysis with Cacid ∝ (1 - Mn/Mn0):

    dMn/dt = -k·Mn·(1 + α·(1 - Mn/Mn0))

Approximate solution (valid for moderate autocatalysis):
    Mn(t) ≈ Mn0 · exp(-k·(1 + α/2)·t) · correction_factor

Author: Darwin Scaffold Studio
Date: December 2025
"""

using Statistics, Random, Printf, LinearAlgebra

# =============================================================================
# FAST PHYSICS MODEL
# =============================================================================

"""
Analytical approximation to Wang-Han ODE.

For dMn/dt = -k·Mn·(1 + α·(1-Mn/Mn0)), use perturbation solution:
    Mn(t) ≈ Mn0 · exp(-k_eff·t^n)

where k_eff incorporates autocatalysis through time-varying rate.
"""
mutable struct FastPhysics
    k0::Float64      # Pre-exponential factor (day⁻¹)
    Ea::Float64      # Activation energy (kJ/mol)
    α::Float64       # Autocatalysis strength
    n::Float64       # Time exponent
    τ::Float64       # Autocatalysis time constant (days)
end

function FastPhysics()
    return FastPhysics(0.025, 80.0, 0.8, 0.65, 25.0)
end

function rate_constant_fast(p::FastPhysics, T::Float64)
    R = 8.314e-3  # kJ/mol·K
    T_K = T + 273.15
    T_ref = 310.15  # 37°C
    return p.k0 * exp(-p.Ea / R * (1/T_K - 1/T_ref))
end

function solve_fast(p::FastPhysics, Mn0::Float64, t::Float64, T::Float64;
                    condition::Symbol=:in_vitro)
    if t <= 0.0
        return Mn0
    end

    k = rate_constant_fast(p, T)

    # In vivo enzymatic acceleration
    if condition == :in_vivo
        k *= 1.35
    end

    # Time-varying autocatalysis (accelerates as degradation proceeds)
    # Models COOH accumulation: starts slow, accelerates
    autocatalysis = 1.0 + p.α * (1.0 - exp(-t / p.τ))

    # Effective rate with time exponent for sub-diffusion
    k_eff = k * autocatalysis

    # Analytical solution
    fraction = exp(-k_eff * t^p.n)

    return max(fraction * Mn0, 0.5)
end

# =============================================================================
# NEURAL CORRECTION (Smaller network for speed)
# =============================================================================

mutable struct FastNeural
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
    scale::Float64
end

function FastNeural(; n_input=5, n_hidden=16, scale=0.25)
    W1 = randn(n_hidden, n_input) * sqrt(2.0 / n_input)
    b1 = zeros(n_hidden)
    W2 = randn(1, n_hidden) * 0.1
    b2 = zeros(1)
    return FastNeural(W1, b1, W2, b2, scale)
end

function forward_fast(nn::FastNeural, x::Vector{Float64})
    h1 = nn.W1 * x .+ nn.b1
    a1 = max.(h1, 0.0)  # ReLU
    out = (nn.W2 * a1 .+ nn.b2)[1]
    return tanh(out) * nn.scale
end

# =============================================================================
# MAIN MODEL
# =============================================================================

mutable struct PLDLANeuralODEFast <: AbstractDegradationModel
    physics::FastPhysics
    neural::FastNeural
    σ_base::Float64    # Base uncertainty
    trained::Bool
end

function PLDLANeuralODEFast()
    return PLDLANeuralODEFast(
        FastPhysics(),
        FastNeural(),
        0.08,
        false
    )
end

# Body temperatures
const BODY_TEMPS_FAST = Dict{Symbol, Float64}(
    :skin_surface => 33.0,
    :subcutaneous => 35.5,
    :muscle => 37.0,
    :bone => 37.0,
    :cartilage => 35.0,
    :liver => 37.5,
    :inflammation => 39.0,
    :standard => 37.0,
)

# =============================================================================
# PREDICTION
# =============================================================================

function predict(model::PLDLANeuralODEFast, Mn0::Float64, t::Float64;
                 T::Float64=37.0, condition::Symbol=:in_vitro,
                 region::Union{Symbol,Nothing}=nothing, TEC::Float64=0.0,
                 with_uncertainty::Bool=false, kwargs...)
    if t <= 0.0
        if with_uncertainty
            return (Mn=Mn0, σ=0.0, lower=Mn0, upper=Mn0)
        else
            return Mn0
        end
    end

    # Get temperature from region
    if region !== nothing && haskey(BODY_TEMPS_FAST, region)
        T = BODY_TEMPS_FAST[region]
    end

    # Physics prediction
    Mn_physics = solve_fast(model.physics, Mn0, t, T, condition=condition)

    # Neural correction
    x = Float64[
        t / 100.0,
        (Mn0 - 40.0) / 20.0,
        (T - 37.0) / 10.0,
        Mn_physics / Mn0,
        condition == :in_vivo ? 1.0 : 0.0,
    ]
    correction = forward_fast(model.neural, x)

    # Combined
    Mn_pred = Mn_physics * (1.0 + correction)
    Mn_pred = max(Mn_pred, 0.5)

    if with_uncertainty
        # Uncertainty grows with time and extrapolation
        σ = model.σ_base * Mn_pred * (1.0 + t/100.0)
        if condition == :in_vivo
            σ *= 1.5
        end
        return (
            Mn = Mn_pred,
            σ = σ,
            lower = max(Mn_pred - 1.96*σ, 0.5),
            upper = Mn_pred + 1.96*σ
        )
    else
        return Mn_pred
    end
end

function predict(model::PLDLANeuralODEFast, material::String, Mn0::Float64, t::Float64; kwargs...)
    TEC = occursin("TEC2", material) ? 2.0 : occursin("TEC1", material) ? 1.0 : 0.0
    return predict(model, Mn0, t; TEC=TEC, kwargs...)
end

# =============================================================================
# TRAINING DATA
# =============================================================================

const TRAINING_DATA_FAST = [
    (id="PLDLA", Mn0=51.285, times=[0.0, 30.0, 60.0, 90.0],
     Mn=[51.285, 25.447, 18.313, 7.904], T=37.0, condition=:in_vitro),
    (id="TEC1", Mn0=44.998, times=[0.0, 30.0, 60.0, 90.0],
     Mn=[44.998, 19.261, 11.676, 8.128], T=37.0, condition=:in_vitro),
    (id="TEC2", Mn0=32.733, times=[0.0, 30.0, 60.0, 90.0],
     Mn=[32.733, 14.953, 12.574, 6.643], T=37.0, condition=:in_vitro),
]

# =============================================================================
# PARAMETER HANDLING
# =============================================================================

function flatten_params_fast(model::PLDLANeuralODEFast)
    p = model.physics
    nn = model.neural
    return vcat(
        [log(p.k0), p.Ea, p.α, p.n, p.τ],
        vec(nn.W1), nn.b1,
        vec(nn.W2), nn.b2
    )
end

function set_params_fast!(model::PLDLANeuralODEFast, params::Vector{Float64})
    idx = 1

    k0 = exp(params[idx]); idx += 1
    Ea = clamp(params[idx], 50.0, 120.0); idx += 1
    α = abs(params[idx]); idx += 1
    n = clamp(params[idx], 0.3, 1.0); idx += 1
    τ = abs(params[idx]) + 5.0; idx += 1

    model.physics = FastPhysics(k0, Ea, α, n, τ)

    nn = model.neural
    n1, m1 = size(nn.W1)
    nn.W1[:] = reshape(params[idx:idx+n1*m1-1], n1, m1)
    idx += n1*m1
    nn.b1[:] = params[idx:idx+n1-1]
    idx += n1

    n2, m2 = size(nn.W2)
    nn.W2[:] = reshape(params[idx:idx+n2*m2-1], n2, m2)
    idx += n2*m2
    nn.b2[:] = params[idx:idx+length(nn.b2)-1]
end

# =============================================================================
# LOSS
# =============================================================================

function compute_loss_fast(model::PLDLANeuralODEFast)
    L = 0.0
    n = 0

    for d in TRAINING_DATA_FAST
        for (i, t) in enumerate(d.times)
            if t == 0.0
                continue
            end

            Mn_pred = predict(model, d.Mn0, t, T=d.T, condition=d.condition)
            Mn_exp = d.Mn[i]

            L += ((Mn_pred - Mn_exp) / Mn_exp)^2
            n += 1
        end
    end

    # Regularization
    L += 0.001 * (model.physics.Ea - 80.0)^2

    return L / max(n, 1)
end

# =============================================================================
# TRAINING
# =============================================================================

function train(::Type{PLDLANeuralODEFast}; epochs::Int=2000,
               population_size::Int=30, σ::Float64=0.04,
               lr::Float64=0.008, verbose::Bool=true)
    Random.seed!(42)

    if verbose
        println("\n" * "="^70)
        println("  PLDLA Neural ODE (Fast) - Physics-Informed Training")
        println("="^70)
        println("  Wang-Han physics + Neural correction (optimized)")
    end

    model = PLDLANeuralODEFast()
    θ = flatten_params_fast(model)
    np = length(θ)

    if verbose
        println("  Parameters: $np (5 physics + $(np-5) neural)")
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
            set_params_fast!(model, θ .+ σ .* noise[:, i])
            push!(losses_pos, compute_loss_fast(model))

            set_params_fast!(model, θ .- σ .* noise[:, i])
            push!(losses_neg, compute_loss_fast(model))
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

        set_params_fast!(model, θ)
        loss = compute_loss_fast(model)

        if loss < best_loss
            best_loss = loss
            best_θ = copy(θ)
        end

        if verbose && (epoch % 500 == 0 || epoch == 1)
            rmse = sqrt(loss) * 100
            @printf("    Epoch %4d: RMSE = %.1f%% (Accuracy ≈ %.1f%%)\n",
                    epoch, rmse, 100-rmse)
        end
    end

    set_params_fast!(model, best_θ)
    model.trained = true

    if verbose
        final_rmse = sqrt(compute_loss_fast(model)) * 100
        println("\n  Training complete!")
        @printf("  Final accuracy: %.1f%%\n", 100-final_rmse)

        p = model.physics
        println("\n  Physics parameters (Wang-Han):")
        @printf("    k₀ = %.4f day⁻¹\n", p.k0)
        @printf("    Ea = %.1f kJ/mol\n", p.Ea)
        @printf("    α  = %.3f (autocatalysis)\n", p.α)
        @printf("    n  = %.3f (time exponent)\n", p.n)
        @printf("    τ  = %.1f days (autocatalysis timescale)\n", p.τ)
    end

    return model
end

# =============================================================================
# VALIDATION
# =============================================================================

function validate(model::PLDLANeuralODEFast; verbose::Bool=true)
    results = Dict{String, Any}()
    all_errors = Float64[]

    if verbose
        println("\n" * "="^70)
        println("  PLDLA Neural ODE (Fast) - Validation")
        println("="^70)
    end

    for d in TRAINING_DATA_FAST
        if verbose
            println("\n  $(d.id) (Mn0=$(d.Mn0), T=$(d.T)°C)")
            println("  " * "-"^55)
            @printf("  %8s  %10s  %10s  %10s  %8s\n",
                    "Time", "Exp", "Physics", "Neural+P", "Error")
        end

        errors = Float64[]

        for (i, t) in enumerate(d.times)
            Mn_exp = d.Mn[i]
            Mn_physics = solve_fast(model.physics, d.Mn0, t, d.T, condition=d.condition)
            Mn_neural = predict(model, d.Mn0, t, T=d.T, condition=d.condition)

            err = t > 0 ? abs(Mn_neural - Mn_exp) / Mn_exp * 100 : 0.0

            if t > 0
                push!(errors, err)
                push!(all_errors, err)
            end

            if verbose
                @printf("  %8.0f  %10.2f  %10.2f  %10.2f  %7.1f%%\n",
                        t, Mn_exp, Mn_physics, Mn_neural, err)
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
        println("\n" * "="^70)
        @printf("  GLOBAL ACCURACY: %.1f%%\n", global_acc)
        println("="^70)
    end

    results["overall_accuracy"] = global_acc
    return results
end

# =============================================================================
# UTILITIES
# =============================================================================

function estimate_halflife(model::PLDLANeuralODEFast, Mn0::Float64;
                           T::Float64=37.0, condition::Symbol=:in_vitro,
                           region::Union{Symbol,Nothing}=nothing)
    target = Mn0 / 2
    t_low, t_high = 0.0, 200.0

    for _ in 1:30
        t_mid = (t_low + t_high) / 2
        Mn_mid = predict(model, Mn0, t_mid, T=T, condition=condition, region=region)

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

function compare_conditions(model::PLDLANeuralODEFast, Mn0::Float64)
    println("\n  Comparison across conditions (Mn0=$Mn0 kg/mol):")
    println("  " * "-"^50)
    @printf("  %-22s  %8s  %10s  %10s\n", "Condition", "T (°C)", "t½ (d)", "Mn(60d)")
    println("  " * "-"^50)

    for (cond, region, label) in [
        (:in_vitro, :standard, "In vitro 37°C"),
        (:in_vivo, :bone, "In vivo bone"),
        (:in_vivo, :cartilage, "In vivo cartilage"),
        (:in_vivo, :subcutaneous, "In vivo subcutâneo"),
        (:in_vivo, :inflammation, "In vivo inflamação"),
    ]
        T = BODY_TEMPS_FAST[region]
        t_half = estimate_halflife(model, Mn0, condition=cond, region=region)
        Mn_60 = predict(model, Mn0, 60.0, condition=cond, region=region)
        @printf("  %-22s  %8.1f  %10.1f  %10.1f\n", label, T, t_half, Mn_60)
    end
end
