"""
    ThermodynamicModel

First-principles thermodynamic model for PLDLA degradation.

THEORY:
- Eyring: k = (kB·T/h)·exp(-ΔG‡/RT)
- Transition State: ΔG‡ = ΔH‡ - T·ΔS‡
- Fick's Law: Diffusion-limited water access
- Fox-Flory: Tg = Tg∞ - K/Mn

THERMODYNAMICS:
- ΔH‡ = 78 kJ/mol (activation enthalpy)
- ΔS‡ = -80 J/(mol·K) (activation entropy)
- ΔG‡ = 103 kJ/mol at 37°C

ACCURACY: ~50% (theoretically rigorous but needs empirical correction)
"""

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

const THERMO_CONSTANTS = (
    R = 8.314,           # J/(mol·K)
    kB = 1.381e-23,      # J/K
    h = 6.626e-34,       # J·s

    # Activation parameters
    ΔH_act = 78.0e3,     # J/mol
    ΔS_act = -80.0,      # J/(mol·K)

    # Reaction thermodynamics
    ΔH_rxn = -12.0e3,    # J/mol (exothermic)
    ΔS_rxn = 65.0,       # J/(mol·K) (entropy increase)

    # Polymer
    Tg_inf = 330.15,     # K
    K_ff = 55.0,         # kg/mol
    Mc = 9.0,            # kg/mol (entanglement)

    # Brønsted
    pKa_lactic = 3.86
)

# =============================================================================
# MODEL STRUCTURE
# =============================================================================

struct ThermodynamicModel <: AbstractDegradationModel
    accessibility::Float64  # Base accessibility factor
    scale::Float64          # Rate scaling
end

function ThermodynamicModel(; accessibility::Float64=5e-3, scale::Float64=1.0)
    return ThermodynamicModel(accessibility, scale)
end

# =============================================================================
# EYRING RATE CONSTANT
# =============================================================================

function eyring_rate(T::Float64)
    ν = THERMO_CONSTANTS.kB * T / THERMO_CONSTANTS.h
    ΔG = THERMO_CONSTANTS.ΔH_act - T * THERMO_CONSTANTS.ΔS_act
    k = ν * exp(-ΔG / (THERMO_CONSTANTS.R * T))
    return k * 86400  # Convert to per day
end

# =============================================================================
# PREDICT
# =============================================================================

function predict(model::ThermodynamicModel, material::Union{String,Int},
                 Mn0::Float64, t::Float64;
                 T::Float64=310.15, pH::Float64=7.4, TEC::Float64=0.0)

    if t == 0.0
        return Mn0
    end

    dt = 0.5
    Mn = Mn0
    t_current = 0.0

    while t_current < t - dt/2
        extent = clamp(1.0 - Mn/Mn0, 0.0, 0.95)

        # Eyring rate
        k_eyring = eyring_rate(T)

        # Accessibility (solid-state polymer)
        f_access = model.accessibility * (1 + 5*extent) * (50.0 / Mn0)

        # Crystallinity barrier
        Xc = 0.05 + 0.35 * (1 - exp(-0.001 * (1 + 3*extent) * t_current^1.5))
        f_crystal = 1.0 - 0.8 * Xc

        # Entanglement
        if Mn > 3 * THERMO_CONSTANTS.Mc
            f_ent = 0.4
        elseif Mn > THERMO_CONSTANTS.Mc
            f_ent = 0.4 + 0.6 * (3*THERMO_CONSTANTS.Mc - Mn) / (2*THERMO_CONSTANTS.Mc)
        else
            f_ent = 1.0
        end

        # Autocatalysis (from Brønsted)
        n_scissions = max(0.0, Mn0/max(Mn, 0.5) - 1.0)
        C_COOH = 0.01 * n_scissions
        Ka = 10^(-THERMO_CONSTANTS.pKa_lactic)
        if C_COOH > 1e-10
            H_acid = (-Ka + sqrt(Ka^2 + 4*Ka*C_COOH)) / 2
            pH_local = -log10(10^(-pH) + H_acid)
            pH_local = max(pH_local, 3.4)
        else
            pH_local = pH
        end
        f_auto = 1.0 + 2.0 * (7.4 - pH_local)

        # Tg effect
        Tg = THERMO_CONSTANTS.Tg_inf - THERMO_CONSTANTS.K_ff * 1000 / max(Mn, 1.0)
        Tg -= 10.0 * TEC
        if T > Tg + 10
            f_Tg = 1.0
        elseif T > Tg - 20
            f_Tg = 0.3 + 0.7 * (T - Tg + 20) / 30
        else
            f_Tg = 0.1
        end

        # Combined rate
        k_eff = k_eyring * f_access * f_crystal * f_ent * f_auto * f_Tg * model.scale

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

function train(::Type{ThermodynamicModel}; verbose::Bool=true)
    if verbose
        println("\n  Initializing ThermodynamicModel...")
        println("  Using first-principles Eyring theory")

        # Calculate ΔG‡ at 37°C
        T = 310.15
        ΔG = THERMO_CONSTANTS.ΔH_act - T * THERMO_CONSTANTS.ΔS_act
        @printf("  ΔG‡(37°C) = %.1f kJ/mol\n", ΔG/1000)
        @printf("  ΔH‡ = %.1f kJ/mol, ΔS‡ = %.1f J/(mol·K)\n",
                THERMO_CONSTANTS.ΔH_act/1000, THERMO_CONSTANTS.ΔS_act)
    end

    return ThermodynamicModel(accessibility=5e-3, scale=1.0)
end
