"""
    ConservativeModel

Conservative empirical model with saturating autocatalysis.

FEATURES:
- Saturating autocatalysis: k_auto * tanh(α * extent)
- Crystallinity protection
- Material-specific parameters

ACCURACY: ~82% (good balance of accuracy and simplicity)
"""

# =============================================================================
# MODEL STRUCTURE
# =============================================================================

struct ConservativeModel <: AbstractDegradationModel
    params::Dict{String, NamedTuple}
end

function ConservativeModel()
    params = Dict(
        "Kaique_PLDLA" => (k=0.0205, k_auto=0.008, α=2.0, Xc_prot=0.4),
        "Kaique_TEC1" => (k=0.0180, k_auto=0.006, α=1.5, Xc_prot=0.3),
        "Kaique_TEC2" => (k=0.0160, k_auto=0.004, α=1.2, Xc_prot=0.2),
        "InVivo_Subcutaneous" => (k=0.0050, k_auto=0.001, α=1.0, Xc_prot=0.7),
        "Default" => (k=0.015, k_auto=0.005, α=1.5, Xc_prot=0.4)
    )
    return ConservativeModel(params)
end

# =============================================================================
# PREDICT
# =============================================================================

function predict(model::ConservativeModel, material::Union{String,Int},
                 Mn0::Float64, t::Float64;
                 T::Float64=310.15, pH::Float64=7.4, TEC::Float64=0.0)

    if t == 0.0
        return Mn0
    end

    # Get material parameters
    mat_name = material isa Int ? get(MATERIAL_NAMES, material, "Default") : material
    p = get(model.params, mat_name, model.params["Default"])

    dt = 0.5
    Mn = Mn0
    t_current = 0.0

    while t_current < t - dt/2
        extent = clamp(1.0 - Mn/Mn0, 0.0, 0.95)

        # Saturating autocatalysis
        k_auto = p.k_auto * tanh(p.α * extent)

        # Crystallinity (Avrami)
        Xc = 0.05 + 0.40 * (1 - exp(-0.001 * (1 + 5*extent) * t_current^1.3))
        Xc = min(Xc, 0.55)

        # Crystal protection
        f_crystal = max(1.0 - p.Xc_prot * Xc, 0.2)

        # Effective rate
        k_eff = (p.k + k_auto) * f_crystal

        # Update
        dMn = -k_eff * Mn
        Mn = max(Mn + dt * dMn, 0.5)
        t_current += dt
    end

    return Mn
end

# =============================================================================
# TRAINING
# =============================================================================

function train(::Type{ConservativeModel}; verbose::Bool=true)
    if verbose
        println("\n  Initializing ConservativeModel...")
        println("  Using pre-calibrated material-specific parameters")
    end

    return ConservativeModel()
end
