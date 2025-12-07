"""
    PolymerBlendPredictor

Predict properties of polymer blends and composites using established mixing rules.

Supported Models:
- Rule of Mixtures (ROM): Linear interpolation for miscible blends
- Fox Equation: Glass transition temperature of blends
- Gordon-Taylor: Tg with interaction parameter
- Logarithmic Rule: Viscosity blending
- Kerner Model: Modulus of particle-filled composites
- Halpin-Tsai: Modulus of fiber-reinforced composites
- Modified Rule of Mixtures: With efficiency factors
- Flory-Huggins: Miscibility prediction

References:
- Fox TG (1956) Bull Am Phys Soc 1:123
- Gordon M, Taylor JS (1952) J Appl Chem 2:493
- Kerner EH (1956) Proc Phys Soc B 69:808
- Halpin JC, Kardos JL (1976) Polym Eng Sci 16:344

Author: Dr. Demetrios Agourakis
"""
module PolymerBlendPredictor

using ..PhysicalPropertiesLibrary

export BlendComponent, BlendPrediction, CompositeType
export predict_blend_properties, predict_Tg, predict_modulus, predict_viscosity
export predict_density, predict_degradation_rate, check_miscibility
export optimize_blend_ratio, BlendOptimizationResult

# =============================================================================
# Types
# =============================================================================

"""Component in a polymer blend."""
struct BlendComponent
    material_id::String
    weight_fraction::Float64  # 0-1
    volume_fraction::Float64  # 0-1 (calculated from density)
end

"""Type of composite/blend system."""
@enum CompositeType begin
    MISCIBLE_BLEND      # Homogeneous single-phase
    IMMISCIBLE_BLEND    # Two-phase with interface
    PARTICLE_COMPOSITE  # Particles in matrix
    FIBER_COMPOSITE     # Fibers in matrix
    LAMINATE           # Layered structure
end

"""Predicted properties of a blend."""
struct BlendPrediction
    components::Vector{BlendComponent}
    composite_type::CompositeType

    # Predicted properties
    density_g_cm3::Float64
    elastic_modulus_mpa::Float64
    tensile_strength_mpa::Float64
    glass_transition_c::Float64
    melting_point_c::Float64
    viscosity_pa_s::Float64
    degradation_rate_relative::Float64  # Relative to slowest component

    # Confidence and notes
    prediction_confidence::Float64  # 0-1
    model_used::Dict{Symbol,String}
    warnings::Vector{String}
    recommendations::Vector{String}
end

"""Result of blend optimization."""
struct BlendOptimizationResult
    optimal_ratios::Dict{String,Float64}
    predicted_properties::BlendPrediction
    objective_value::Float64
    constraints_satisfied::Bool
end

# =============================================================================
# Main Prediction Functions
# =============================================================================

"""
    predict_blend_properties(materials::Vector{Tuple{String,Float64}}; kwargs...)

Predict properties of a polymer blend.

# Arguments
- `materials`: Vector of (material_id, weight_fraction) tuples
- `composite_type`: Type of blend (default: auto-detect)
- `interaction_param`: Polymer-polymer interaction parameter χ (default: 0)

# Returns
BlendPrediction with all predicted properties

# Example
```julia
# 70/30 PCL/PLA blend
pred = predict_blend_properties([("PCL", 0.7), ("PLA", 0.3)])

# With ceramic filler
pred = predict_blend_properties([("PCL", 0.8), ("Hydroxyapatite", 0.2)];
                                composite_type=PARTICLE_COMPOSITE)
```
"""
function predict_blend_properties(materials::Vector{Tuple{String,Float64}};
    composite_type::Union{CompositeType,Nothing}=nothing,
    interaction_param::Float64=0.0)

    # Validate weight fractions sum to 1
    total_weight = sum(m[2] for m in materials)
    if abs(total_weight - 1.0) > 0.01
        error("Weight fractions must sum to 1.0 (got $total_weight)")
    end

    # Get properties for each component
    props = Dict{String,Any}()
    for (id, wf) in materials
        p = PhysicalPropertiesLibrary.get_physical_properties(id)
        if isnothing(p)
            error("Material $id not found in database")
        end
        props[id] = p
    end

    # Auto-detect composite type if not specified
    if isnothing(composite_type)
        composite_type = detect_composite_type(props, materials)
    end

    # Calculate volume fractions from weight fractions and densities
    components = calculate_volume_fractions(materials, props)

    # Initialize tracking
    models_used = Dict{Symbol,String}()
    warnings = String[]
    recommendations = String[]
    confidence = 1.0

    # Predict each property
    density = predict_density(components, props)
    models_used[:density] = "Rule of Mixtures"

    modulus, mod_model, mod_conf = predict_modulus(components, props, composite_type)
    models_used[:modulus] = mod_model
    confidence = min(confidence, mod_conf)

    strength = predict_strength(components, props, composite_type)
    models_used[:strength] = "Modified ROM"

    Tg, tg_model = predict_Tg(components, props; χ=interaction_param)
    models_used[:Tg] = tg_model

    Tm = predict_melting_point(components, props)
    models_used[:Tm] = "Weight-averaged (crystalline phases)"

    viscosity = predict_viscosity(components, props)
    models_used[:viscosity] = "Logarithmic mixing rule"

    deg_rate = predict_degradation_rate(components, props)
    models_used[:degradation] = "Empirical blend model"

    # Check for potential issues
    check_blend_compatibility!(warnings, recommendations, props, materials, composite_type)

    # Adjust confidence based on warnings
    confidence *= (1.0 - 0.1 * length(warnings))
    confidence = max(confidence, 0.3)

    return BlendPrediction(
        components, composite_type,
        density, modulus, strength, Tg, Tm, viscosity, deg_rate,
        confidence, models_used, warnings, recommendations
    )
end

# =============================================================================
# Individual Property Predictions
# =============================================================================

"""
    predict_Tg(components, props; χ=0.0, model=:auto)

Predict glass transition temperature using Fox or Gordon-Taylor equation.

Fox equation (χ ≈ 0):
    1/Tg_blend = Σ(wi/Tg_i)

Gordon-Taylor (with interaction):
    Tg_blend = (w1*Tg1 + k*w2*Tg2) / (w1 + k*w2)
    where k ≈ Δα2/Δα1 (ratio of thermal expansion coefficients)
"""
function predict_Tg(components::Vector{BlendComponent}, props::Dict;
    χ::Float64=0.0, model::Symbol=:auto)

    # Collect Tg values (skip if NaN)
    tg_data = Tuple{Float64,Float64}[]  # (weight_fraction, Tg_K)

    for comp in components
        p = props[comp.material_id]
        Tg_c = p.thermal.glass_transition_c
        if !isnan(Tg_c)
            Tg_K = Tg_c + 273.15
            push!(tg_data, (comp.weight_fraction, Tg_K))
        end
    end

    if isempty(tg_data)
        return (NaN, "No Tg data available")
    end

    if length(tg_data) == 1
        return (tg_data[1][2] - 273.15, "Single component")
    end

    # Choose model
    use_model = model
    if model == :auto
        use_model = abs(χ) < 0.1 ? :fox : :gordon_taylor
    end

    if use_model == :fox
        # Fox equation: 1/Tg = Σ(wi/Tg_i)
        inv_Tg = sum(w/Tg for (w, Tg) in tg_data)
        Tg_blend_K = 1.0 / inv_Tg
        return (Tg_blend_K - 273.15, "Fox equation")
    else
        # Gordon-Taylor (simplified, k ≈ 1 for similar polymers)
        k = 1.0 + χ  # Approximate interaction effect
        w1, Tg1 = tg_data[1]
        w2, Tg2 = tg_data[2]
        Tg_blend_K = (w1 * Tg1 + k * w2 * Tg2) / (w1 + k * w2)
        return (Tg_blend_K - 273.15, "Gordon-Taylor")
    end
end

"""
    predict_modulus(components, props, composite_type)

Predict elastic modulus using appropriate model for composite type.

Models:
- Rule of Mixtures (upper bound): E = Σ(Vi * Ei) - parallel loading
- Inverse ROM (lower bound): 1/E = Σ(Vi/Ei) - series loading
- Halpin-Tsai: For fiber composites
- Kerner: For particle composites
"""
function predict_modulus(components::Vector{BlendComponent}, props::Dict,
                         composite_type::CompositeType)

    # Collect modulus data
    E_data = Tuple{Float64,Float64,Float64}[]  # (weight_frac, vol_frac, E_mpa)

    for comp in components
        p = props[comp.material_id]
        E = p.mechanical.elastic_modulus_mpa
        if !isnan(E)
            push!(E_data, (comp.weight_fraction, comp.volume_fraction, E))
        end
    end

    if isempty(E_data)
        return (NaN, "No modulus data", 0.0)
    end

    if length(E_data) == 1
        return (E_data[1][3], "Single component", 1.0)
    end

    if composite_type == MISCIBLE_BLEND
        # Use geometric mean for miscible blends (empirically good)
        log_E = sum(vf * log(E) for (_, vf, E) in E_data)
        E_blend = exp(log_E)
        return (E_blend, "Geometric mean (miscible)", 0.8)

    elseif composite_type == PARTICLE_COMPOSITE
        # Kerner model for particle-filled systems
        # Simplified: matrix + filler
        E_matrix = E_data[1][3]
        vf_filler = E_data[2][2]
        E_filler = E_data[2][3]

        # Kerner equation (assuming ν ≈ 0.35)
        ν = 0.35
        A = (7 - 5*ν) / (8 - 10*ν)
        B = (E_filler/E_matrix - 1) / (E_filler/E_matrix + A)
        E_blend = E_matrix * (1 + A*B*vf_filler) / (1 - B*vf_filler)

        return (E_blend, "Kerner model", 0.85)

    elseif composite_type == FIBER_COMPOSITE
        # Halpin-Tsai for aligned fibers
        E_matrix = E_data[1][3]
        vf_fiber = E_data[2][2]
        E_fiber = E_data[2][3]

        # Halpin-Tsai with ξ = 2 (typical for fibers)
        ξ = 2.0
        η = (E_fiber/E_matrix - 1) / (E_fiber/E_matrix + ξ)
        E_blend = E_matrix * (1 + ξ*η*vf_fiber) / (1 - η*vf_fiber)

        return (E_blend, "Halpin-Tsai", 0.8)

    else
        # Default: Voigt-Reuss bounds average
        E_upper = sum(vf * E for (_, vf, E) in E_data)  # Voigt (parallel)
        E_lower = 1.0 / sum(vf/E for (_, vf, E) in E_data)  # Reuss (series)
        E_blend = sqrt(E_upper * E_lower)  # Geometric mean of bounds

        return (E_blend, "Voigt-Reuss average", 0.7)
    end
end

"""
    predict_viscosity(components, props)

Predict melt viscosity using logarithmic mixing rule.

log(η_blend) = Σ(wi * log(ηi))
"""
function predict_viscosity(components::Vector{BlendComponent}, props::Dict)
    η_data = Tuple{Float64,Float64}[]

    for comp in components
        p = props[comp.material_id]
        η = p.rheological.viscosity_pa_s
        if !isnan(η) && η > 0
            push!(η_data, (comp.weight_fraction, η))
        end
    end

    if isempty(η_data)
        return NaN
    end

    # Logarithmic mixing rule (Arrhenius-type)
    log_η = sum(w * log(η) for (w, η) in η_data)
    return exp(log_η)
end

"""
    predict_density(components, props)

Predict density using rule of mixtures (exact for ideal mixing).
"""
function predict_density(components::Vector{BlendComponent}, props::Dict)
    ρ_blend = 0.0
    total_vol = 0.0

    for comp in components
        p = props[comp.material_id]
        ρ = p.structural.density_g_cm3
        if !isnan(ρ)
            # ρ_blend = Σ(Vi * ρi) where Vi is volume fraction
            ρ_blend += comp.volume_fraction * ρ
            total_vol += comp.volume_fraction
        end
    end

    return total_vol > 0 ? ρ_blend / total_vol : NaN
end

"""
    predict_degradation_rate(components, props)

Predict relative degradation rate of blend.
Returns value relative to 1.0 (slowest degrading component).
"""
function predict_degradation_rate(components::Vector{BlendComponent}, props::Dict)
    # Estimate based on category and known degradation behavior
    # This is simplified - actual degradation is complex

    rates = Float64[]
    weights = Float64[]

    for comp in components
        p = props[comp.material_id]

        # Estimate relative degradation rate by material type
        rate = if comp.material_id in ["PCL"]
            0.3  # Slow (2-3 years)
        elseif comp.material_id in ["PLLA", "PLA"]
            0.7  # Medium (1-2 years)
        elseif comp.material_id in ["PLGA", "PGA"]
            1.5  # Fast (weeks-months)
        elseif comp.material_id in ["Collagen", "GelMA", "Fibrin"]
            2.0  # Very fast (days-weeks)
        elseif p.category == :ceramic
            0.05  # Very slow
        elseif p.category == :metal
            0.01  # Extremely slow (unless Mg/Zn)
        else
            1.0  # Default medium
        end

        push!(rates, rate)
        push!(weights, comp.weight_fraction)
    end

    # Weighted average of degradation rates
    return isempty(rates) ? 1.0 : sum(rates .* weights)
end

"""
    predict_strength(components, props, composite_type)

Predict tensile strength using modified rule of mixtures.
"""
function predict_strength(components::Vector{BlendComponent}, props::Dict,
                          composite_type::CompositeType)

    σ_data = Tuple{Float64,Float64}[]  # (vol_frac, strength)

    for comp in components
        p = props[comp.material_id]
        σ = p.mechanical.tensile_strength_mpa
        if !isnan(σ)
            push!(σ_data, (comp.volume_fraction, σ))
        end
    end

    if isempty(σ_data)
        return NaN
    end

    # Efficiency factor based on composite type
    η_eff = if composite_type == FIBER_COMPOSITE
        0.9  # Good load transfer
    elseif composite_type == PARTICLE_COMPOSITE
        0.7  # Moderate load transfer
    elseif composite_type == MISCIBLE_BLEND
        0.95  # Homogeneous
    else
        0.8  # Default
    end

    # Modified ROM with efficiency
    σ_blend = η_eff * sum(vf * σ for (vf, σ) in σ_data)

    return σ_blend
end

"""
    predict_melting_point(components, props)

Predict melting point (for semi-crystalline blends).
"""
function predict_melting_point(components::Vector{BlendComponent}, props::Dict)
    Tm_data = Tuple{Float64,Float64}[]

    for comp in components
        p = props[comp.material_id]
        Tm = p.thermal.melting_point_c
        if !isnan(Tm)
            push!(Tm_data, (comp.weight_fraction, Tm))
        end
    end

    if isempty(Tm_data)
        return NaN
    end

    # Weighted average (simplified - actual behavior depends on miscibility)
    return sum(w * Tm for (w, Tm) in Tm_data)
end

# =============================================================================
# Miscibility and Compatibility
# =============================================================================

"""
    check_miscibility(material1::String, material2::String)

Check miscibility of two polymers using solubility parameter approach.

Returns: (is_miscible::Bool, Δδ::Float64, notes::String)
"""
function check_miscibility(material1::String, material2::String)
    # Solubility parameters (MPa^0.5) - literature values
    δ_params = Dict(
        "PCL" => 17.4,
        "PLA" => 19.5,
        "PLLA" => 19.5,
        "PLGA" => 19.0,
        "PEG" => 17.6,
        "PVA" => 25.8,
        "PEEK" => 20.0,
        "PU" => 20.5,
        "Collagen" => 24.0,
        "GelMA" => 23.0,
        "Chitosan" => 26.0,
        "Alginate" => 28.0
    )

    δ1 = get(δ_params, material1, nothing)
    δ2 = get(δ_params, material2, nothing)

    if isnothing(δ1) || isnothing(δ2)
        return (false, NaN, "Solubility parameter not available")
    end

    Δδ = abs(δ1 - δ2)

    # Rule of thumb: Δδ < 2 MPa^0.5 suggests miscibility
    if Δδ < 2.0
        return (true, Δδ, "Likely miscible (Δδ < 2)")
    elseif Δδ < 4.0
        return (false, Δδ, "Partially miscible (2 < Δδ < 4)")
    else
        return (false, Δδ, "Immiscible (Δδ > 4)")
    end
end

# =============================================================================
# Blend Optimization
# =============================================================================

"""
    optimize_blend_ratio(materials::Vector{String}, target_properties::Dict; constraints=Dict())

Optimize blend ratio to achieve target properties.

# Arguments
- `materials`: List of materials to blend
- `target_properties`: Dict of :property => target_value
- `constraints`: Dict of :property => (min, max)

# Example
```julia
result = optimize_blend_ratio(
    ["PCL", "PLA"],
    Dict(:elastic_modulus_mpa => 1000.0, :glass_transition_c => 40.0),
    constraints=Dict(:degradation_rate_relative => (0.5, 1.5))
)
```
"""
function optimize_blend_ratio(materials::Vector{String},
                              target_properties::Dict{Symbol,Float64};
                              constraints::Dict{Symbol,Tuple{Float64,Float64}}=Dict{Symbol,Tuple{Float64,Float64}}())

    n = length(materials)
    if n < 2
        error("Need at least 2 materials for blending")
    end

    best_ratios = nothing
    best_error = Inf
    best_prediction = nothing

    # Grid search (for simplicity - could use proper optimization)
    step = 0.1

    if n == 2
        for w1 in 0.1:step:0.9
            w2 = 1.0 - w1

            blend = [(materials[1], w1), (materials[2], w2)]
            pred = predict_blend_properties(blend)

            # Check constraints
            valid = true
            for (prop, (min_val, max_val)) in constraints
                val = getfield(pred, prop)
                if !isnan(val) && (val < min_val || val > max_val)
                    valid = false
                    break
                end
            end

            if !valid
                continue
            end

            # Calculate error from targets
            error = 0.0
            for (prop, target) in target_properties
                val = getfield(pred, prop)
                if !isnan(val)
                    rel_error = ((val - target) / target)^2
                    error += rel_error
                end
            end

            if error < best_error
                best_error = error
                best_ratios = Dict(materials[1] => w1, materials[2] => w2)
                best_prediction = pred
            end
        end
    elseif n == 3
        for w1 in 0.1:step:0.8
            for w2 in 0.1:step:(0.9-w1)
                w3 = 1.0 - w1 - w2
                if w3 < 0.05
                    continue
                end

                blend = [(materials[1], w1), (materials[2], w2), (materials[3], w3)]
                pred = predict_blend_properties(blend)

                # Check constraints
                valid = true
                for (prop, (min_val, max_val)) in constraints
                    val = getfield(pred, prop)
                    if !isnan(val) && (val < min_val || val > max_val)
                        valid = false
                        break
                    end
                end

                if !valid
                    continue
                end

                # Calculate error
                error = 0.0
                for (prop, target) in target_properties
                    val = getfield(pred, prop)
                    if !isnan(val)
                        rel_error = ((val - target) / target)^2
                        error += rel_error
                    end
                end

                if error < best_error
                    best_error = error
                    best_ratios = Dict(materials[1] => w1, materials[2] => w2, materials[3] => w3)
                    best_prediction = pred
                end
            end
        end
    else
        error("Optimization for >3 components not yet implemented")
    end

    if isnothing(best_ratios)
        error("No valid blend found satisfying constraints")
    end

    return BlendOptimizationResult(
        best_ratios,
        best_prediction,
        sqrt(best_error),
        true
    )
end

# =============================================================================
# Helper Functions
# =============================================================================

function calculate_volume_fractions(materials::Vector{Tuple{String,Float64}}, props::Dict)
    components = BlendComponent[]

    # First pass: get densities and calculate volumes
    volumes = Float64[]
    for (id, wf) in materials
        ρ = props[id].structural.density_g_cm3
        if isnan(ρ)
            ρ = 1.0  # Default assumption
        end
        vol = wf / ρ  # Relative volume (mass/density)
        push!(volumes, vol)
    end

    total_vol = sum(volumes)

    # Second pass: create components with volume fractions
    for (i, (id, wf)) in enumerate(materials)
        vf = volumes[i] / total_vol
        push!(components, BlendComponent(id, wf, vf))
    end

    return components
end

function detect_composite_type(props::Dict, materials::Vector{Tuple{String,Float64}})
    categories = [props[m[1]].category for m in materials]

    # Check for filler materials
    has_ceramic = :ceramic in categories
    has_metal = :metal in categories
    has_polymer = :polymer in categories || :hydrogel in categories

    if has_ceramic || has_metal
        if has_polymer
            return PARTICLE_COMPOSITE
        else
            return IMMISCIBLE_BLEND
        end
    end

    # Check polymer miscibility
    if length(materials) == 2
        m1, m2 = materials[1][1], materials[2][1]
        is_misc, _, _ = check_miscibility(m1, m2)
        return is_misc ? MISCIBLE_BLEND : IMMISCIBLE_BLEND
    end

    return IMMISCIBLE_BLEND  # Default conservative assumption
end

function check_blend_compatibility!(warnings, recommendations, props, materials, composite_type)
    ids = [m[1] for m in materials]

    # Check for known incompatibilities
    if "PCL" in ids && "PVA" in ids
        push!(warnings, "PCL and PVA are immiscible - consider compatibilizer")
        push!(recommendations, "Add PCL-g-PVA copolymer as compatibilizer")
    end

    if "Collagen" in ids && any(id -> props[id].category == :metal, ids)
        push!(warnings, "Metal ions may denature collagen")
    end

    # Check processing compatibility
    temps = [props[id].thermal.melting_point_c for id in ids]
    valid_temps = filter(!isnan, temps)
    if length(valid_temps) >= 2
        if maximum(valid_temps) - minimum(valid_temps) > 50
            push!(warnings, "Large difference in processing temperatures")
            push!(recommendations, "Consider solution blending instead of melt blending")
        end
    end

    # Check for ceramic/polymer blends
    if composite_type == PARTICLE_COMPOSITE
        push!(recommendations, "Surface treat ceramic particles to improve interface")
        push!(recommendations, "Limit filler content to <30 vol% for processability")
    end
end

end # module
