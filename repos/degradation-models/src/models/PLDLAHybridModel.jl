"""
    PLDLAHybridModel

Hybrid model for PLDLA 70:30 degradation combining:
- Physics from PLDLA3DPrintModelV2 (k, n, autocatalysis)
- Temperature dependence (Arrhenius)
- In vivo enzymatic factor
- Body region temperature mapping
- Uncertainty quantification

SCOPE:
- Material: PLDLA 70:30 (3D-printed scaffolds)
- Conditions: In vitro AND in vivo
- Temperature: 32-40°C (body regions)
- Time: 0-180 days

Author: Darwin Scaffold Studio
Date: December 2025
"""

using Statistics, Random, Printf

# =============================================================================
# BODY REGION TEMPERATURES
# =============================================================================

"""
Body region temperature mapping (°C).
Based on clinical literature for implant sites.
"""
const BODY_TEMPERATURES = Dict{Symbol, NamedTuple{(:T, :range, :description), Tuple{Float64, Tuple{Float64,Float64}, String}}}(
    :skin_surface    => (T=33.0, range=(32.0, 34.0), description="Superficial skin"),
    :subcutaneous    => (T=35.5, range=(35.0, 36.0), description="Subcutaneous tissue"),
    :muscle          => (T=37.0, range=(36.5, 37.5), description="Muscle tissue"),
    :bone            => (T=37.0, range=(36.5, 37.5), description="Bone/periosteum"),
    :cartilage       => (T=35.0, range=(34.0, 36.0), description="Articular cartilage"),
    :liver           => (T=37.5, range=(37.0, 38.0), description="Liver/visceral"),
    :inflammation    => (T=39.0, range=(38.0, 40.0), description="Inflamed tissue"),
    :standard        => (T=37.0, range=(37.0, 37.0), description="Standard in vitro"),
)

# =============================================================================
# EXPERIMENTAL DATA
# =============================================================================

const HYBRID_TRAINING_DATA = [
    # Kaique in vitro data (PLDLA 70:30)
    (id="PLDLA_invitro", Mn0=51.285, times=[0.0, 30.0, 60.0, 90.0],
     Mn=[51.285, 25.447, 18.313, 7.904], T=37.0, TEC=0.0, condition=:in_vitro),
    (id="TEC1_invitro", Mn0=44.998, times=[0.0, 30.0, 60.0, 90.0],
     Mn=[44.998, 19.261, 11.676, 8.128], T=37.0, TEC=1.0, condition=:in_vitro),
    (id="TEC2_invitro", Mn0=32.733, times=[0.0, 30.0, 60.0, 90.0],
     Mn=[32.733, 14.953, 12.574, 6.643], T=37.0, TEC=2.0, condition=:in_vitro),

    # Literature in vivo data (estimated for PLDLA based on PLLA studies)
    # In vivo typically 1.3-1.5x faster due to enzymatic activity
    # Bergsma PLA96 showed ~1.4x acceleration in vivo
]

# =============================================================================
# MODEL STRUCTURE
# =============================================================================

"""
    PLDLAHybridModel

Combines physics-based degradation with environmental factors.

Parameters:
- k0: Base hydrolysis rate at 37°C in vitro
- n: Time exponent
- a: Autocatalysis strength
- τ: Autocatalysis time constant
- Ea: Activation energy (kJ/mol)
- f_invivo: In vivo acceleration factor
"""
mutable struct PLDLAHybridModel <: AbstractDegradationModel
    # Kinetic parameters (from PLDLA3DPrintModelV2)
    k0::Float64           # Base rate at 37°C in vitro (day^-1)
    n::Float64            # Time exponent
    a::Float64            # Autocatalysis strength
    τ::Float64            # Autocatalysis time constant (days)

    # Temperature dependence
    Ea::Float64           # Activation energy (kJ/mol)

    # In vivo factor
    f_invivo::Float64     # In vivo acceleration (typically 1.3-1.5)

    # Uncertainty
    σ_k::Float64          # Relative uncertainty in k

    trained::Bool
end

function PLDLAHybridModel()
    return PLDLAHybridModel(
        0.045,    # k0: from V2 model
        0.67,     # n: from V2 model
        0.82,     # a: from V2 model
        26.0,     # τ: from V2 model
        80.0,     # Ea: ~80 kJ/mol for PLA hydrolysis (literature)
        1.35,     # f_invivo: based on Bergsma comparison
        0.15,     # σ_k: 15% uncertainty
        false
    )
end

# =============================================================================
# PHYSICS
# =============================================================================

"""
    arrhenius_factor(T, T_ref, Ea)

Calculate temperature-dependent rate factor using Arrhenius equation.

k(T) = k(T_ref) × exp(-Ea/R × (1/T - 1/T_ref))

Where:
- Ea: Activation energy (kJ/mol)
- R: Gas constant (8.314 J/mol·K)
- T, T_ref: Temperatures (°C, converted to K internally)
"""
function arrhenius_factor(T::Float64, T_ref::Float64, Ea::Float64)
    R = 8.314e-3  # kJ/mol·K
    T_K = T + 273.15
    T_ref_K = T_ref + 273.15

    return exp(-Ea / R * (1/T_K - 1/T_ref_K))
end

"""
    condition_factor(condition, f_invivo)

Get acceleration factor based on condition (in_vitro or in_vivo).
"""
function condition_factor(condition::Symbol, f_invivo::Float64)
    return condition == :in_vivo ? f_invivo : 1.0
end

# =============================================================================
# PREDICTION
# =============================================================================

"""
    predict(model, Mn0, t; T=37.0, condition=:in_vitro, region=nothing, TEC=0.0)

Predict molecular weight at time t.

# Arguments
- `Mn0`: Initial molecular weight (kg/mol)
- `t`: Time (days)
- `T`: Temperature (°C), default 37.0
- `condition`: :in_vitro or :in_vivo
- `region`: Body region symbol (overrides T if provided)
- `TEC`: Plasticizer content (%)

# Returns
- Predicted Mn (kg/mol)

# Example
```julia
# Standard in vitro
Mn = predict(model, 50.0, 30.0)

# In vivo subcutaneous implant
Mn = predict(model, 50.0, 30.0, condition=:in_vivo, region=:subcutaneous)

# In vivo bone scaffold at inflammation site
Mn = predict(model, 50.0, 30.0, condition=:in_vivo, region=:inflammation)
```
"""
function predict(model::PLDLAHybridModel, Mn0::Float64, t::Float64;
                 T::Float64=37.0, condition::Symbol=:in_vitro,
                 region::Union{Symbol,Nothing}=nothing, TEC::Float64=0.0,
                 with_uncertainty::Bool=false, kwargs...)
    if t <= 0.0
        return with_uncertainty ? (Mn=Mn0, σ=0.0) : Mn0
    end

    # Get temperature from region if specified
    if region !== nothing && haskey(BODY_TEMPERATURES, region)
        T = BODY_TEMPERATURES[region].T
    end

    # Temperature factor (Arrhenius)
    f_T = arrhenius_factor(T, 37.0, model.Ea)

    # Condition factor (in vivo acceleration)
    f_cond = condition_factor(condition, model.f_invivo)

    # Autocatalysis
    autocatalysis = 1.0 + model.a * (1.0 - exp(-t / model.τ))

    # Effective rate constant
    k_eff = model.k0 * f_T * f_cond * autocatalysis

    # Degradation
    fraction = exp(-k_eff * t^model.n)
    Mn_pred = max(fraction * Mn0, 0.5)

    if with_uncertainty
        # Propagate uncertainty
        σ_Mn = Mn_pred * model.σ_k * sqrt(t / 30.0)  # Uncertainty grows with time
        return (Mn=Mn_pred, σ=σ_Mn)
    else
        return Mn_pred
    end
end

function predict(model::PLDLAHybridModel, material::String, Mn0::Float64, t::Float64; kwargs...)
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

function flatten_params(model::PLDLAHybridModel)
    return [model.k0, model.n, model.a, model.τ, model.Ea, model.f_invivo]
end

function set_params!(model::PLDLAHybridModel, p::Vector{Float64})
    model.k0 = abs(p[1])
    model.n = clamp(p[2], 0.3, 1.0)
    model.a = abs(p[3])
    model.τ = abs(p[4]) + 1.0
    model.Ea = clamp(p[5], 50.0, 120.0)  # Reasonable range for Ea
    model.f_invivo = clamp(p[6], 1.0, 2.0)  # 1-2x acceleration
end

function compute_loss(model::PLDLAHybridModel)
    L = 0.0
    n = 0

    for d in HYBRID_TRAINING_DATA
        for (i, t) in enumerate(d.times)
            if t == 0.0
                continue
            end

            Mn_pred = predict(model, d.Mn0, t, T=d.T, condition=d.condition, TEC=d.TEC)
            Mn_exp = d.Mn[i]

            rel_err = (Mn_pred - Mn_exp) / Mn_exp
            L += rel_err^2
            n += 1
        end
    end

    # Regularization: keep Ea near literature value (~80 kJ/mol)
    L += 0.001 * (model.Ea - 80.0)^2

    return L / max(n, 1)
end

function train(::Type{PLDLAHybridModel}; epochs::Int=2000,
               population_size::Int=30, σ::Float64=0.05,
               lr::Float64=0.01, verbose::Bool=true)
    Random.seed!(42)

    if verbose
        println("\n" * "="^65)
        println("  PLDLA HYBRID MODEL - Training")
        println("="^65)
        println("  Combines: Physics + Temperature + In vivo factors")
        println("  Base: Kaique experimental data (in vitro)")
        println("  Extension: Arrhenius temperature, enzymatic acceleration")
    end

    model = PLDLAHybridModel()
    θ = flatten_params(model)
    np = length(θ)

    if verbose
        println("  Parameters: k0, n, a, τ, Ea, f_invivo")
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
            set_params!(model, θ .+ σ .* noise[:, i])
            push!(losses_pos, compute_loss(model))

            set_params!(model, θ .- σ .* noise[:, i])
            push!(losses_neg, compute_loss(model))
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

        println("\n  Training complete!")
        @printf("  Accuracy: %.1f%%\n", 100 - final_rmse)

        println("\n  Learned parameters:")
        @printf("    k0 = %.4f day⁻¹ (base rate at 37°C in vitro)\n", model.k0)
        @printf("    n  = %.3f (time exponent)\n", model.n)
        @printf("    a  = %.3f (autocatalysis strength)\n", model.a)
        @printf("    τ  = %.1f days (autocatalysis time constant)\n", model.τ)
        @printf("    Ea = %.1f kJ/mol (activation energy)\n", model.Ea)
        @printf("    f_invivo = %.2f (in vivo acceleration)\n", model.f_invivo)
    end

    return model
end

# =============================================================================
# VALIDATION
# =============================================================================

function validate(model::PLDLAHybridModel; verbose::Bool=true)
    results = Dict{String, Any}()
    all_errors = Float64[]

    if verbose
        println("\n" * "="^65)
        println("  PLDLA HYBRID MODEL - Validation")
        println("="^65)
    end

    for d in HYBRID_TRAINING_DATA
        if verbose
            cond_str = d.condition == :in_vivo ? "IN VIVO" : "in vitro"
            println("\n  $(d.id) [$cond_str, T=$(d.T)°C]")
            println("  " * "-"^55)
            @printf("  %8s  %12s  %12s  %8s\n", "Time(d)", "Experimental", "Predicted", "Error")
        end

        errors = Float64[]

        for (i, t) in enumerate(d.times)
            Mn_pred = predict(model, d.Mn0, t, T=d.T, condition=d.condition, TEC=d.TEC)
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
        println("\n" * "="^65)
        @printf("  GLOBAL ACCURACY: %.1f%%\n", global_acc)
        println("="^65)
    end

    results["overall_accuracy"] = global_acc
    return results
end

# =============================================================================
# UTILITIES
# =============================================================================

"""
    estimate_halflife(model, Mn0; T=37.0, condition=:in_vitro, region=nothing)

Estimate time to 50% Mn loss.
"""
function estimate_halflife(model::PLDLAHybridModel, Mn0::Float64;
                           T::Float64=37.0, condition::Symbol=:in_vitro,
                           region::Union{Symbol,Nothing}=nothing, TEC::Float64=0.0)
    target = Mn0 / 2
    t_low, t_high = 0.0, 300.0

    for _ in 1:50
        t_mid = (t_low + t_high) / 2
        Mn_mid = predict(model, Mn0, t_mid, T=T, condition=condition, region=region, TEC=TEC)

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

"""
    compare_conditions(model, Mn0; t_max=90)

Compare degradation across different conditions.
"""
function compare_conditions(model::PLDLAHybridModel, Mn0::Float64; t_max::Float64=90.0)
    conditions = [
        (:in_vitro, :standard, "In vitro 37°C"),
        (:in_vivo, :subcutaneous, "In vivo subcutâneo"),
        (:in_vivo, :bone, "In vivo osso"),
        (:in_vivo, :muscle, "In vivo músculo"),
        (:in_vivo, :cartilage, "In vivo cartilagem"),
        (:in_vivo, :inflammation, "In vivo inflamação"),
    ]

    println("\n  Comparação de condições (Mn0 = $Mn0 kg/mol)")
    println("  " * "-"^60)
    @printf("  %-25s  %8s  %10s  %10s\n", "Condição", "T (°C)", "t½ (dias)", "Mn(60d)")
    println("  " * "-"^60)

    for (cond, region, label) in conditions
        T = BODY_TEMPERATURES[region].T
        t_half = estimate_halflife(model, Mn0, condition=cond, region=region)
        Mn_60 = predict(model, Mn0, 60.0, condition=cond, region=region)

        @printf("  %-25s  %8.1f  %10.1f  %10.1f\n", label, T, t_half, Mn_60)
    end
end

"""
    predict_with_uncertainty(model, Mn0, t; kwargs...)

Predict with uncertainty bounds (mean ± 2σ for 95% CI).
"""
function predict_with_uncertainty(model::PLDLAHybridModel, Mn0::Float64, t::Float64; kwargs...)
    result = predict(model, Mn0, t; with_uncertainty=true, kwargs...)
    return (
        mean = result.Mn,
        lower = max(result.Mn - 2*result.σ, 0.5),
        upper = result.Mn + 2*result.σ,
        σ = result.σ
    )
end

"""
    list_body_regions()

List available body regions and their temperatures.
"""
function list_body_regions()
    println("\n  Regiões do corpo disponíveis:")
    println("  " * "-"^50)
    @printf("  %-18s  %8s  %15s\n", "Região", "T (°C)", "Range")
    println("  " * "-"^50)

    for (sym, data) in sort(collect(BODY_TEMPERATURES), by=x->x[2].T)
        @printf("  :%-17s  %8.1f  %6.1f - %.1f°C\n",
                sym, data.T, data.range[1], data.range[2])
    end
end
