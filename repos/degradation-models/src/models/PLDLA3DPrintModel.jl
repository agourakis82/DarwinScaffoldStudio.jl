"""
    PLDLA3DPrintModel

Specialized degradation model for 3D-printed PLDLA 70:30 scaffolds.

SCOPE:
- Material: PLDLA 70:30 (poly-L/DL-lactide)
- Form: 3D-printed scaffolds
- Conditions: In vitro PBS pH 7.4, 37°C
- Time range: 0-90 days (scaffold degradation window)

DATA SOURCES:
- Kaique Hergesel PhD thesis (PLDLA, TEC1, TEC2)
- Literature PLDLA 70:30 studies

KEY PHYSICS:
- Hydrolysis with autocatalysis
- Surface erosion + bulk degradation
- Plasticizer (TEC) accelerates degradation

Author: Darwin Scaffold Studio
Date: December 2025
"""

using Statistics, Random, Printf

# =============================================================================
# EXPERIMENTAL DATA - 3D-PRINTED PLDLA
# =============================================================================

const PLDLA_3DPRINT_DATA = [
    # Kaique's experimental data - PLDLA 70:30 scaffolds
    (
        id="Kaique_PLDLA",
        description="Pure PLDLA 70:30, 3D-printed scaffold",
        Mn0=51.285,  # kg/mol
        times=[0.0, 30.0, 60.0, 90.0],
        Mn=[51.285, 25.447, 18.313, 7.904],
        T=37.0,
        TEC=0.0,  # No plasticizer
        porosity=0.5,  # Estimated
    ),
    (
        id="Kaique_TEC1",
        description="PLDLA + 1% TEC plasticizer",
        Mn0=44.998,
        times=[0.0, 30.0, 60.0, 90.0],
        Mn=[44.998, 19.261, 11.676, 8.128],
        T=37.0,
        TEC=1.0,
        porosity=0.5,
    ),
    (
        id="Kaique_TEC2",
        description="PLDLA + 2% TEC plasticizer",
        Mn0=32.733,
        times=[0.0, 30.0, 60.0, 90.0],
        Mn=[32.733, 14.953, 12.574, 6.643],
        T=37.0,
        TEC=2.0,
        porosity=0.5,
    ),
]

# =============================================================================
# MODEL STRUCTURE
# =============================================================================

"""
    PLDLA3DPrintModel

Physics-informed model for 3D-printed PLDLA 70:30 degradation.

The model uses:
- First-order hydrolysis with autocatalysis
- TEC plasticizer effect (increases water uptake)
- Molecular weight dependent degradation
"""
mutable struct PLDLA3DPrintModel <: AbstractDegradationModel
    # Kinetic parameters
    k0::Float64           # Base hydrolysis rate (day^-1)
    n::Float64            # Autocatalytic exponent
    α_TEC::Float64        # TEC acceleration factor
    β_Mn::Float64         # MW dependence

    # Neural correction
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}

    trained::Bool
end

function PLDLA3DPrintModel()
    n_input = 4
    n_hidden = 16

    return PLDLA3DPrintModel(
        0.020,   # k0: ~20 per 1000 days (from data analysis)
        0.75,    # n: slightly sublinear
        0.15,    # α_TEC: 15% increase per %TEC
        0.005,   # β_Mn: higher MW degrades slightly slower
        randn(n_hidden, n_input) * 0.1,
        zeros(n_hidden),
        randn(1, n_hidden) * 0.05,
        [0.0],
        false
    )
end

# =============================================================================
# PHYSICS MODEL
# =============================================================================

"""
    physics_degradation(model, Mn0, t, TEC)

Compute Mn(t)/Mn0 using physics-based model.

Kinetics:
    dMn/dt = -k_eff * Mn
    
where:
    k_eff = k0 * (1 + α_TEC * TEC) * (1 - β_Mn * Mn0) * f_autocatalysis(t)
    f_autocatalysis = 1 + 0.5 * (1 - exp(-t/30))
"""
function physics_degradation(model::PLDLA3DPrintModel, Mn0::Float64,
    t::Float64, TEC::Float64)
    if t <= 0.0
        return 1.0
    end

    # TEC effect: plasticizer increases water uptake → faster degradation
    k_TEC = 1.0 + model.α_TEC * TEC

    # MW effect: very high MW slightly slower (less end groups)
    k_Mn = 1.0 - model.β_Mn * (Mn0 - 40.0) / 40.0
    k_Mn = clamp(k_Mn, 0.7, 1.3)

    # Effective rate constant
    k_eff = model.k0 * k_TEC * k_Mn

    # Autocatalysis: acidic products accelerate degradation
    # Gradual increase over ~30 days as COOH accumulates
    autocatalysis = 1.0 + 0.8 * (1.0 - exp(-t / 25.0))

    # Modified first-order decay
    fraction = exp(-k_eff * autocatalysis * t^model.n)

    return clamp(fraction, 0.01, 1.0)
end

# =============================================================================
# FORWARD PASS
# =============================================================================

function forward_pldla(model::PLDLA3DPrintModel, Mn0::Float64,
    t::Float64, TEC::Float64)
    # Physics prediction
    frac_physics = physics_degradation(model, Mn0, t, TEC)

    # Neural correction (small adjustment)
    x = Float64[
        t/90.0,                    # Normalized time (0-90 days)
        (Mn0-40.0)/20.0,         # Centered MW
        TEC/2.0,                   # Normalized TEC
        frac_physics,                # Physics prediction
    ]

    h1 = model.W1 * x .+ model.b1
    a1 = tanh.(h1)
    correction = (model.W2*a1.+model.b2)[1]

    # Small bounded correction (±15%)
    correction = tanh(correction) * 0.15

    final_frac = clamp(frac_physics * (1.0 + correction), 0.01, 1.0)

    return final_frac
end

# =============================================================================
# PREDICT
# =============================================================================

"""
    predict(model, Mn0, t; TEC=0.0, T=37.0)

Predict molecular weight at time t for 3D-printed PLDLA scaffold.

# Arguments
- `Mn0`: Initial molecular weight (kg/mol)
- `t`: Time (days)
- `TEC`: Plasticizer content (%), default 0.0
- `T`: Temperature (°C), default 37.0

# Returns
- Predicted Mn (kg/mol)

# Example
```julia
model = train(PLDLA3DPrintModel)
Mn_30d = predict(model, 51.0, 30.0)  # Pure PLDLA at 30 days
Mn_60d = predict(model, 45.0, 60.0, TEC=1.0)  # With 1% TEC
```
"""
function predict(model::PLDLA3DPrintModel, Mn0::Float64, t::Float64;
    TEC::Float64=0.0, T::Float64=37.0, kwargs...)
    if t <= 0.0
        return Mn0
    end

    # Temperature correction (Arrhenius)
    # Reference is 37°C
    if T != 37.0
        Ea_R = 7000.0  # K
        T_factor = exp(Ea_R * (1 / 310.15 - 1 / (T + 273.15)))
        # Apply to time (equivalent to rate scaling)
        t = t * T_factor
    end

    fraction = forward_pldla(model, Mn0, t, TEC)
    return fraction * Mn0
end

# Convenience method for material string
function predict(model::PLDLA3DPrintModel, material::String, Mn0::Float64, t::Float64; kwargs...)
    # Extract TEC from material name if present
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
# PARAMETER HANDLING
# =============================================================================

function flatten_params_pldla(model::PLDLA3DPrintModel)
    return vcat(
        [model.k0, model.n, model.α_TEC, model.β_Mn],
        vec(model.W1),
        model.b1,
        vec(model.W2),
        model.b2
    )
end

function set_params_pldla!(model::PLDLA3DPrintModel, params::Vector{Float64})
    model.k0 = abs(params[1])
    model.n = clamp(params[2], 0.3, 1.2)
    model.α_TEC = abs(params[3])
    model.β_Mn = params[4]

    idx = 5
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

function compute_loss_pldla(model::PLDLA3DPrintModel)
    L = 0.0
    n = 0

    for d in PLDLA_3DPRINT_DATA
        for (i, t) in enumerate(d.times)
            if t == 0.0
                continue
            end

            Mn_pred = predict(model, d.Mn0, t, TEC=d.TEC)
            Mn_exp = d.Mn[i]

            # Relative squared error
            rel_err = (Mn_pred - Mn_exp) / Mn_exp
            L += rel_err^2
            n += 1
        end
    end

    return L / max(n, 1)
end

# =============================================================================
# TRAINING
# =============================================================================

function train(::Type{PLDLA3DPrintModel}; epochs::Int=2000,
    population_size::Int=40, σ::Float64=0.03,
    lr::Float64=0.005, verbose::Bool=true)
    Random.seed!(42)

    if verbose
        println("\n" * "="^60)
        println("  PLDLA 3D-Print Model Training")
        println("="^60)
        println("  Specialized for: PLDLA 70:30 3D-printed scaffolds")
        println("  Data: Kaique Hergesel experimental (PLDLA, TEC1, TEC2)")
    end

    model = PLDLA3DPrintModel()
    θ = flatten_params_pldla(model)
    np = length(θ)

    if verbose
        n_points = sum(length(d.times) - 1 for d in PLDLA_3DPRINT_DATA)
        println("  Parameters: $np (4 physics + $(np-4) neural)")
        println("  Training points: $n_points")
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
            set_params_pldla!(model, θ .+ σ .* noise[:, i])
            push!(losses_pos, compute_loss_pldla(model))

            set_params_pldla!(model, θ .- σ .* noise[:, i])
            push!(losses_neg, compute_loss_pldla(model))
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

        set_params_pldla!(model, θ)
        loss = compute_loss_pldla(model)

        if loss < best_loss
            best_loss = loss
            best_θ = copy(θ)
        end

        if verbose && (epoch % 500 == 0 || epoch == 1)
            rmse = sqrt(loss) * 100
            acc = 100 - rmse
            @printf("    Epoch %4d: RMSE = %5.1f%% (Accuracy: %5.1f%%)\n", epoch, rmse, acc)
        end
    end

    set_params_pldla!(model, best_θ)
    model.trained = true

    if verbose
        final_rmse = sqrt(compute_loss_pldla(model)) * 100
        final_acc = 100 - final_rmse

        println("\n  Training complete!")
        @printf("  Final: RMSE = %.1f%%, Accuracy = %.1f%%\n", final_rmse, final_acc)

        println("\n  Learned parameters:")
        @printf("    k0 = %.4f day⁻¹ (base hydrolysis rate)\n", model.k0)
        @printf("    n  = %.3f (autocatalytic exponent)\n", model.n)
        @printf("    α_TEC = %.3f (TEC acceleration: +%.0f%% per %%TEC)\n",
            model.α_TEC, model.α_TEC * 100)
        @printf("    β_Mn = %.4f (MW dependence)\n", model.β_Mn)
    end

    return model
end

# =============================================================================
# VALIDATION
# =============================================================================

function validate(model::PLDLA3DPrintModel; verbose::Bool=true)
    results = Dict{String,Any}()
    all_errors = Float64[]

    if verbose
        println("\n" * "="^60)
        println("  PLDLA 3D-Print Model Validation")
        println("="^60)
    end

    for d in PLDLA_3DPRINT_DATA
        errors = Float64[]

        if verbose
            println("\n  $(d.id): $(d.description)")
            println("  Mn0 = $(d.Mn0) kg/mol, TEC = $(d.TEC)%")
            println("  " * "-"^50)
            @printf("  %8s  %10s  %10s  %8s\n", "Time(d)", "Exp(kg/mol)", "Pred", "Error")
        end

        for (i, t) in enumerate(d.times)
            Mn_pred = predict(model, d.Mn0, t, TEC=d.TEC)
            Mn_exp = d.Mn[i]

            if t > 0.0
                err = abs(Mn_pred - Mn_exp) / Mn_exp * 100
                push!(errors, err)
                push!(all_errors, err)
            else
                err = 0.0
            end

            if verbose
                @printf("  %8.0f  %10.2f  %10.2f  %7.1f%%\n", t, Mn_exp, Mn_pred, err)
            end
        end

        mape = mean(errors)
        acc = 100 - mape

        if verbose
            @printf("  → Dataset accuracy: %.1f%%\n", acc)
        end

        results[d.id] = (mape=mape, accuracy=acc, n_points=length(errors))
    end

    global_mape = mean(all_errors)
    global_acc = 100 - global_mape

    if verbose
        println("\n" * "="^60)
        @printf("  GLOBAL: %.1f%% accuracy (MAPE = %.1f%%)\n", global_acc, global_mape)

        if global_acc >= 90
            println("  ✓ Excellent! Model ready for scaffold design optimization.")
        elseif global_acc >= 80
            println("  ✓ Good accuracy for engineering applications.")
        else
            println("  ~ Consider collecting more data or adjusting model.")
        end
        println("="^60)
    end

    results["overall_accuracy"] = global_acc
    results["mean_mape"] = global_mape

    return results
end

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

"""
    predict_degradation_curve(model, Mn0; TEC=0.0, t_max=90, n_points=50)

Generate a full degradation curve for visualization.
"""
function predict_degradation_curve(model::PLDLA3DPrintModel, Mn0::Float64;
    TEC::Float64=0.0, t_max::Float64=90.0,
    n_points::Int=50)
    times = range(0, t_max, length=n_points)
    Mn_values = [predict(model, Mn0, t, TEC=TEC) for t in times]
    return (times=collect(times), Mn=Mn_values)
end

"""
    estimate_halflife(model, Mn0; TEC=0.0)

Estimate time to 50% molecular weight loss.
"""
function estimate_halflife(model::PLDLA3DPrintModel, Mn0::Float64; TEC::Float64=0.0)
    target = Mn0 / 2

    # Binary search
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

# Convenience
train_pldla_3dprint(; kwargs...) = train(PLDLA3DPrintModel; kwargs...)
