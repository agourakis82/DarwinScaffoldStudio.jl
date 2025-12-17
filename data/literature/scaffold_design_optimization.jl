"""
Scaffold Design Optimization
============================

Integrates polymer science, structure-property relationships, and tissue
requirements into a practical scaffold design tool.

This module:
1. Takes tissue healing requirements as input
2. Selects appropriate polymer based on degradation timeline
3. Calculates optimal scaffold geometry
4. Predicts properties over time
5. Validates against Q1 literature targets

All models and targets sourced from peer-reviewed literature.
"""

using Statistics
using Printf

# ============================================================================
# TISSUE HEALING REQUIREMENTS
# ============================================================================

"""
Tissue healing phases and requirements from literature.
Sources: PMC3347861, Frontiers Bioeng 2024
"""
struct TissueRequirements
    name::String
    healing_time_weeks::Tuple{Float64, Float64}  # Min-max healing time
    target_porosity::Tuple{Float64, Float64}     # Min-max porosity
    target_pore_size_um::Tuple{Float64, Float64} # Min-max pore size
    target_modulus_mpa::Tuple{Float64, Float64}  # Min-max modulus
    target_strength_mpa::Tuple{Float64, Float64} # Min-max strength
    cell_type::Symbol
    vascularization_critical::Bool
    load_bearing::Bool
    source::String
end

const TISSUE_REQUIREMENTS = Dict(
    :trabecular_bone => TissueRequirements(
        "Trabecular Bone",
        (12.0, 24.0),      # 3-6 months healing
        (0.50, 0.90),      # 50-90% porosity
        (300.0, 600.0),    # 300-600 um pores
        (100.0, 500.0),    # E match trabecular (100-500 MPa)
        (2.0, 12.0),       # Strength 2-12 MPa
        :osteoblast,
        true,
        true,
        "Karageorgiou 2005"
    ),
    :cortical_bone => TissueRequirements(
        "Cortical Bone",
        (24.0, 52.0),      # 6-12 months
        (0.30, 0.60),      # Lower porosity for strength
        (150.0, 400.0),    # Smaller pores OK
        (5000.0, 20000.0), # E match cortical
        (50.0, 200.0),     # High strength
        :osteoblast,
        false,
        true,
        "Murphy 2010"
    ),
    :articular_cartilage => TissueRequirements(
        "Articular Cartilage",
        (16.0, 52.0),      # Long healing
        (0.80, 0.95),      # High porosity
        (100.0, 300.0),    # Smaller pores
        (0.5, 10.0),       # Low modulus
        (0.5, 5.0),        # Low strength
        :chondrocyte,
        false,
        false,
        "PMID:17961371"
    ),
    :skin => TissueRequirements(
        "Skin/Dermis",
        (2.0, 8.0),        # Fast healing
        (0.70, 0.90),
        (50.0, 200.0),
        (0.1, 1.0),        # Very soft
        (0.1, 1.0),
        :fibroblast,
        true,
        false,
        "PMC6682490"
    ),
    :liver => TissueRequirements(
        "Liver",
        (4.0, 12.0),
        (0.85, 0.95),
        (100.0, 250.0),
        (0.1, 2.0),
        (0.1, 1.0),
        :hepatocyte,
        true,
        false,
        "PMID:23201040"
    ),
    :cardiac => TissueRequirements(
        "Cardiac Muscle",
        (8.0, 24.0),
        (0.70, 0.90),
        (100.0, 200.0),
        (0.05, 0.5),       # Very soft, elastic
        (0.01, 0.1),
        :cardiomyocyte,
        true,
        false,
        "PMC5449418"
    ),
    :nerve => TissueRequirements(
        "Peripheral Nerve",
        (12.0, 52.0),      # Very slow
        (0.60, 0.85),
        (20.0, 100.0),     # Small pores for guidance
        (0.01, 0.5),
        (0.01, 0.5),
        :neuron,
        true,
        false,
        "PMC4424662"
    ),
)

# ============================================================================
# POLYMER DATABASE (from polymer_degradation_science.jl)
# ============================================================================

struct PolymerCandidate
    name::String
    abbrev::String
    E_solid_mpa::Float64        # Solid modulus
    sigma_solid_mpa::Float64    # Solid strength
    degradation_weeks::Tuple{Float64, Float64}  # Degradation time range
    mechanism::Symbol           # :chain_end or :random
    biocompatibility::Float64   # 0-1 score
    cost_relative::Float64      # 1.0 = standard
    fda_approved::Bool
    source::String
end

const POLYMER_CANDIDATES = [
    # Stiff polyesters (for bone, cartilage)
    PolymerCandidate("Poly(lactic acid)", "PLA", 3500.0, 50.0, (48.0, 104.0), :chain_end, 0.95, 1.0, true, "PMC6682490"),
    PolymerCandidate("Poly(glycolic acid)", "PGA", 7000.0, 60.0, (4.0, 12.0), :chain_end, 0.90, 1.2, true, "PMC6682490"),
    PolymerCandidate("PLGA 50:50", "PLGA50", 2000.0, 40.0, (4.0, 8.0), :random, 0.92, 1.5, true, "PMC3347861"),
    PolymerCandidate("PLGA 85:15", "PLGA85", 2500.0, 45.0, (20.0, 40.0), :random, 0.93, 1.4, true, "PMC3347861"),
    PolymerCandidate("Polycaprolactone", "PCL", 400.0, 25.0, (104.0, 208.0), :chain_end, 0.98, 0.8, true, "PMC5449418"),
    PolymerCandidate("Poly(3-hydroxybutyrate)", "PHB", 3500.0, 40.0, (52.0, 156.0), :chain_end, 0.85, 2.0, false, "PMC4424662"),
    PolymerCandidate("Poly(propylene fumarate)", "PPF", 2000.0, 30.0, (26.0, 78.0), :random, 0.88, 3.0, false, "PMID:17961371"),
    PolymerCandidate("Polyurethane (degradable)", "PU", 50.0, 10.0, (26.0, 104.0), :random, 0.80, 2.5, false, "PMC5923535"),

    # Hydrogels (for soft tissues: skin, nerve, cardiac, liver)
    # E values for hydrogels are much lower (kPa range, expressed as MPa)
    PolymerCandidate("Collagen Type I", "COL1", 0.05, 0.01, (2.0, 8.0), :enzymatic, 0.99, 3.0, true, "PMC4082975"),
    PolymerCandidate("Gelatin Methacrylate", "GelMA", 0.02, 0.005, (2.0, 6.0), :enzymatic, 0.95, 2.0, false, "PMC5449418"),
    PolymerCandidate("Alginate", "ALG", 0.1, 0.02, (4.0, 12.0), :dissolution, 0.92, 0.5, true, "PMC3347861"),
    PolymerCandidate("Hyaluronic Acid", "HA", 0.01, 0.002, (1.0, 4.0), :enzymatic, 0.98, 4.0, true, "PMC6682490"),
    PolymerCandidate("Fibrin", "FIB", 0.005, 0.001, (1.0, 3.0), :enzymatic, 0.99, 5.0, true, "PMC4424662"),
    PolymerCandidate("Chitosan", "CHI", 0.15, 0.03, (4.0, 16.0), :enzymatic, 0.88, 1.5, false, "PMC5923535"),
    PolymerCandidate("PEG Diacrylate", "PEGDA", 0.5, 0.1, (4.0, 52.0), :hydrolytic, 0.90, 2.0, true, "PMID:23201040"),
    PolymerCandidate("Silk Fibroin", "SILK", 1.0, 0.2, (12.0, 52.0), :enzymatic, 0.95, 3.5, true, "PMC4082975"),
    PolymerCandidate("Matrigel", "MAT", 0.001, 0.0005, (0.5, 2.0), :enzymatic, 0.97, 10.0, false, "PMC5449418"),
]

# ============================================================================
# CROSSLINKING DATABASE
# ============================================================================

"""
Crosslinking method for hydrogels.
- E_multiplier: factor increase in modulus (1.0 = no change)
- degradation_factor: multiplier for degradation time (>1 = slower)
- biocompat_penalty: reduction in biocompatibility score (0 = none)
- compatible_polymers: which hydrogels work with this method
"""
struct CrosslinkingMethod
    name::String
    abbrev::String
    E_multiplier::Float64           # Modulus increase factor
    degradation_factor::Float64     # Degradation slowdown factor
    biocompat_penalty::Float64      # Biocompatibility reduction (0-1)
    compatible_polymers::Vector{String}  # Polymer abbreviations
    source::String
end

const CROSSLINKING_METHODS = [
    # Chemical crosslinking
    CrosslinkingMethod(
        "EDC/NHS", "EDC",
        5.0, 2.0, 0.02,
        ["COL1", "GelMA", "HA", "CHI"],
        "PMC4082975"
    ),
    CrosslinkingMethod(
        "Glutaraldehyde", "GTA",
        20.0, 3.0, 0.15,  # High modulus but cytotoxic
        ["COL1", "GelMA", "CHI"],
        "PMC3347861"
    ),
    CrosslinkingMethod(
        "Genipin", "GEN",
        8.0, 2.5, 0.03,   # Natural crosslinker, low toxicity
        ["COL1", "GelMA", "CHI", "SILK"],
        "PMC5449418"
    ),
    # Physical crosslinking
    CrosslinkingMethod(
        "Calcium chloride (ionic)", "CaCl2",
        3.0, 1.5, 0.0,    # Very biocompatible
        ["ALG"],
        "PMC3347861"
    ),
    CrosslinkingMethod(
        "Thermal gelation", "THERM",
        2.0, 1.2, 0.0,
        ["COL1", "GelMA", "MAT"],
        "PMC4082975"
    ),
    # Photo-crosslinking
    CrosslinkingMethod(
        "UV + LAP", "UV-LAP",
        15.0, 2.0, 0.05,
        ["GelMA", "PEGDA", "HA"],
        "PMC5449418"
    ),
    CrosslinkingMethod(
        "UV + Irgacure", "UV-IRG",
        12.0, 2.0, 0.08,
        ["GelMA", "PEGDA"],
        "PMID:23201040"
    ),
    # Enzymatic crosslinking
    CrosslinkingMethod(
        "Transglutaminase", "TG",
        4.0, 1.8, 0.01,   # Very biocompatible, natural enzyme
        ["COL1", "GelMA", "FIB"],
        "PMC4424662"
    ),
    CrosslinkingMethod(
        "Thrombin", "THR",
        3.0, 1.5, 0.0,    # Natural fibrin crosslinker
        ["FIB"],
        "PMC4424662"
    ),
    # No crosslinking (baseline)
    CrosslinkingMethod(
        "None (uncrosslinked)", "NONE",
        1.0, 1.0, 0.0,
        ["COL1", "GelMA", "ALG", "HA", "FIB", "CHI", "PEGDA", "SILK", "MAT"],
        "N/A"
    ),
]

"""
Select optimal crosslinking method for a hydrogel to achieve target modulus.
Returns the method that gets closest to target E while maximizing biocompatibility.
"""
function select_crosslinking(polymer::PolymerCandidate, target_E_mpa::Float64)
    if polymer.E_solid_mpa > 1.0
        # Not a hydrogel, no crosslinking needed
        return (method=CROSSLINKING_METHODS[end], E_achieved=polymer.E_solid_mpa, needed=false)
    end

    compatible = filter(m -> polymer.abbrev in m.compatible_polymers, CROSSLINKING_METHODS)

    if isempty(compatible)
        return (method=CROSSLINKING_METHODS[end], E_achieved=polymer.E_solid_mpa, needed=false)
    end

    # Score each method: prioritize achieving target E, then biocompatibility
    best_method = compatible[1]
    best_score = -Inf
    best_E = polymer.E_solid_mpa

    for method in compatible
        E_crosslinked = polymer.E_solid_mpa * method.E_multiplier

        # Score: how close to target, penalized by toxicity
        if E_crosslinked >= target_E_mpa
            # Can achieve target - prefer lowest crosslinking that works
            score = 100 - method.E_multiplier - method.biocompat_penalty * 50
        else
            # Can't achieve target - maximize E
            score = E_crosslinked / target_E_mpa * 50 - method.biocompat_penalty * 50
        end

        if score > best_score
            best_score = score
            best_method = method
            best_E = E_crosslinked
        end
    end

    return (method=best_method, E_achieved=best_E, needed=best_method.abbrev != "NONE")
end

# ============================================================================
# DESIGN ALGORITHM
# ============================================================================

"""
Score a polymer for a given tissue application.
"""
function score_polymer(polymer::PolymerCandidate, tissue::TissueRequirements)
    score = 0.0
    reasons = String[]

    # 1. Degradation time match (40% weight)
    deg_min, deg_max = polymer.degradation_weeks
    heal_min, heal_max = tissue.healing_time_weeks

    if deg_min <= heal_max && deg_max >= heal_min
        # Overlap exists
        overlap = min(deg_max, heal_max) - max(deg_min, heal_min)
        total_range = max(deg_max, heal_max) - min(deg_min, heal_min)
        deg_score = 40 * overlap / total_range
        score += deg_score
        push!(reasons, @sprintf("Degradation match: %.0f%%", deg_score/40*100))
    else
        push!(reasons, "Degradation time mismatch")
    end

    # 2. Mechanical properties potential (30% weight)
    # Can we achieve target modulus with reasonable porosity?
    target_E_min, target_E_max = tissue.target_modulus_mpa
    target_phi_min, target_phi_max = tissue.target_porosity

    # Gibson-Ashby: E_scaffold = E_solid * (1-phi)^2
    E_at_max_porosity = polymer.E_solid_mpa * (1 - target_phi_max)^2
    E_at_min_porosity = polymer.E_solid_mpa * (1 - target_phi_min)^2

    # Determine if this is a soft tissue (target E <= 1 MPa)
    is_soft_tissue = target_E_max <= 1.0
    is_hydrogel = polymer.E_solid_mpa <= 1.0

    if is_soft_tissue && is_hydrogel
        # For soft tissues, hydrogels are ideal
        # Check if hydrogel E is in right range (within order of magnitude)
        if polymer.E_solid_mpa <= target_E_max * 10 && polymer.E_solid_mpa >= target_E_min * 0.1
            score += 30
            push!(reasons, "Hydrogel ideal for soft tissue: E_solid = $(round(polymer.E_solid_mpa * 1000, digits=0)) kPa")
        else
            score += 20
            push!(reasons, "Hydrogel suitable: E_solid = $(round(polymer.E_solid_mpa * 1000, digits=0)) kPa")
        end
    elseif is_soft_tissue && !is_hydrogel
        # Stiff polymer for soft tissue - problematic
        if E_at_max_porosity <= target_E_max
            score += 15
            push!(reasons, "Stiff polymer, may work at high porosity")
        else
            score += 0
            push!(reasons, "Too stiff for soft tissue ($(round(E_at_max_porosity, digits=1)) MPa > $(target_E_max) MPa)")
        end
    elseif E_at_max_porosity <= target_E_max && E_at_min_porosity >= target_E_min
        # Perfect match for stiff tissues
        score += 30
        push!(reasons, "Modulus achievable: E = $(round(E_at_max_porosity, digits=1))-$(round(E_at_min_porosity, digits=1)) MPa")
    elseif E_at_min_porosity >= target_E_min
        score += 15
        push!(reasons, "Modulus achievable at low porosity only")
    else
        push!(reasons, "Cannot achieve target modulus")
    end

    # 3. Biocompatibility (20% weight)
    score += 20 * polymer.biocompatibility
    push!(reasons, @sprintf("Biocompatibility: %.0f%%", polymer.biocompatibility * 100))

    # 4. FDA approval (10% weight)
    if polymer.fda_approved
        score += 10
        push!(reasons, "FDA approved")
    else
        push!(reasons, "Not FDA approved (research use)")
    end

    return (score=score, reasons=reasons, polymer=polymer)
end

"""
Select best polymer for tissue application.
"""
function select_polymer(tissue::TissueRequirements;
                        require_fda::Bool=true,
                        max_cost::Float64=3.0)
    candidates = filter(p -> p.cost_relative <= max_cost, POLYMER_CANDIDATES)
    if require_fda
        candidates = filter(p -> p.fda_approved, candidates)
    end

    if isempty(candidates)
        error("No polymers meet requirements")
    end

    scored = [score_polymer(p, tissue) for p in candidates]
    sort!(scored, by=x -> x.score, rev=true)

    return scored
end

"""
Calculate optimal scaffold geometry for given polymer and tissue.
Optionally applies crosslinking for hydrogels.
"""
function optimize_geometry(polymer::PolymerCandidate, tissue::TissueRequirements;
                           crosslinking::Union{Nothing, CrosslinkingMethod}=nothing)
    # Target values
    target_E_min, target_E_max = tissue.target_modulus_mpa
    target_phi_min, target_phi_max = tissue.target_porosity
    target_pore_min, target_pore_max = tissue.target_pore_size_um

    # Check if this is a hydrogel (soft material)
    is_hydrogel = polymer.E_solid_mpa <= 1.0

    # Effective modulus with crosslinking
    E_effective = polymer.E_solid_mpa
    if is_hydrogel && crosslinking !== nothing
        E_effective = polymer.E_solid_mpa * crosslinking.E_multiplier
    end

    if is_hydrogel
        # For hydrogels, use porosity optimized for cell infiltration
        # (not mechanical properties, which are inherently soft)
        optimal_porosity = (target_phi_min + target_phi_max) / 2
        # Hydrogel modulus with crosslinking, slight reduction for porosity
        actual_E = E_effective * (1 - optimal_porosity * 0.3)
    else
        # Stiff polymer: Calculate porosity for target modulus (Gibson-Ashby inverse)
        # E_target = E_solid * (1-phi)^2
        # phi = 1 - sqrt(E_target / E_solid)

        target_E = (target_E_min + target_E_max) / 2
        ratio = target_E / polymer.E_solid_mpa

        if ratio <= 0 || ratio > 1
            # Can't achieve target E, use tissue-preferred porosity
            phi_for_E = (target_phi_min + target_phi_max) / 2
        else
            phi_for_E = 1 - sqrt(ratio)
        end
        phi_for_E = clamp(phi_for_E, 0.0, 0.99)

        # Reconcile with tissue porosity requirements
        optimal_porosity = clamp(phi_for_E, target_phi_min, target_phi_max)

        # Recalculate achievable E
        actual_E = polymer.E_solid_mpa * (1 - optimal_porosity)^2
    end

    # Optimal pore size: middle of range, biased toward cell requirements
    optimal_pore = (target_pore_min + target_pore_max) / 2

    # Window size for interconnectivity (target: 30-40% of pore size)
    window_size = 0.35 * optimal_pore

    # Wall thickness from porosity and pore size
    # For cubic unit cells: wall ~ pore * (1-phi)^(1/3) / phi^(1/3)
    wall_thickness = optimal_pore * (1 - optimal_porosity)^(1/3) / optimal_porosity^(1/3)

    # Strength also scales with crosslinking
    sigma_multiplier = crosslinking !== nothing ? sqrt(crosslinking.E_multiplier) : 1.0
    sigma_scaffold = polymer.sigma_solid_mpa * sigma_multiplier * 0.3 * (1 - optimal_porosity)^1.5

    return (
        porosity = optimal_porosity,
        pore_size_um = optimal_pore,
        window_size_um = window_size,
        wall_thickness_um = wall_thickness,
        E_scaffold_mpa = actual_E,
        sigma_scaffold_mpa = sigma_scaffold,
        crosslinking = crosslinking,
    )
end

"""
Predict scaffold properties over time as it degrades.
Crosslinking slows degradation by the crosslinking.degradation_factor.
"""
function predict_degradation_profile(polymer::PolymerCandidate,
                                      geometry::NamedTuple;
                                      time_weeks::Vector{Float64}=Float64.(collect(0:4:52)))
    # Degradation rate (first-order approximation)
    # Half-life ~ (deg_min + deg_max) / 2
    half_life = (polymer.degradation_weeks[1] + polymer.degradation_weeks[2]) / 2

    # Crosslinking slows degradation
    if haskey(geometry, :crosslinking) && geometry.crosslinking !== nothing
        half_life *= geometry.crosslinking.degradation_factor
    end

    k = log(2) / half_life  # per week

    results = []

    E0 = geometry.E_scaffold_mpa
    phi0 = geometry.porosity

    for t in time_weeks
        # MW decay (exponential)
        mw_fraction = exp(-k * t)

        # Mass loss depends on degradation mechanism
        if polymer.mechanism == :random
            # Autocatalytic bulk erosion (PLGA-like)
            mass_remaining = 1.0 / (1.0 + exp(0.2 * (t - half_life)))
        elseif polymer.mechanism == :chain_end
            # Surface erosion (PLA, PCL-like)
            mass_remaining = max(0, 1.0 - (1 - exp(-k * t/2)))
        elseif polymer.mechanism == :enzymatic
            # Enzymatic degradation (collagen, gelatin, HA, fibrin)
            # Faster initial degradation, then plateau
            mass_remaining = exp(-k * t) * (1 + 0.5 * exp(-2*k * t))
            mass_remaining = clamp(mass_remaining, 0.0, 1.0)
        elseif polymer.mechanism == :dissolution
            # Dissolution (alginate - ion exchange)
            # More linear degradation profile
            mass_remaining = max(0.0, 1.0 - t / (2 * half_life))
        elseif polymer.mechanism == :hydrolytic
            # Hydrolytic (PEG-based)
            # Similar to chain-end but with swelling
            mass_remaining = exp(-k * t * 0.8)
        else
            # Default: exponential
            mass_remaining = exp(-k * t)
        end
        mass_remaining = clamp(mass_remaining, 0.0, 1.0)

        # Porosity increases as mass is lost
        phi_t = 1.0 - (1.0 - phi0) * mass_remaining
        phi_t = clamp(phi_t, phi0, 0.999)

        # Modulus decay (combined MW and porosity effects)
        E_t = E0 * mw_fraction^0.7 * mass_remaining^2

        # Mechanical integrity
        if mw_fraction < 0.15
            E_t = 0.0  # Below critical MW
        end

        push!(results, (
            time_weeks = t,
            mw_fraction = mw_fraction,
            mass_remaining = mass_remaining,
            porosity = phi_t,
            E_mpa = E_t,
            integrity = mw_fraction >= 0.15 ? "Intact" : "Fragmented"
        ))
    end

    return results
end

"""
Validate design against Q1 literature targets.
"""
function validate_design(geometry::NamedTuple, tissue::TissueRequirements)
    validations = []
    all_pass = true

    # Porosity check
    phi_min, phi_max = tissue.target_porosity
    if phi_min <= geometry.porosity <= phi_max
        push!(validations, (param="Porosity", value=geometry.porosity,
                           target="$(phi_min*100)-$(phi_max*100)%", status="PASS"))
    else
        push!(validations, (param="Porosity", value=geometry.porosity,
                           target="$(phi_min*100)-$(phi_max*100)%", status="FAIL"))
        all_pass = false
    end

    # Pore size check
    pore_min, pore_max = tissue.target_pore_size_um
    if pore_min <= geometry.pore_size_um <= pore_max
        push!(validations, (param="Pore size", value=geometry.pore_size_um,
                           target="$(pore_min)-$(pore_max) um", status="PASS"))
    else
        push!(validations, (param="Pore size", value=geometry.pore_size_um,
                           target="$(pore_min)-$(pore_max) um", status="FAIL"))
        all_pass = false
    end

    # Modulus check
    E_min, E_max = tissue.target_modulus_mpa
    if E_min <= geometry.E_scaffold_mpa <= E_max
        push!(validations, (param="Modulus", value=geometry.E_scaffold_mpa,
                           target="$(E_min)-$(E_max) MPa", status="PASS"))
    else
        push!(validations, (param="Modulus", value=geometry.E_scaffold_mpa,
                           target="$(E_min)-$(E_max) MPa", status="FAIL"))
        all_pass = false
    end

    # Interconnectivity estimate (based on window/pore ratio)
    intercon = geometry.window_size_um / geometry.pore_size_um > 0.3 ? 0.9 : 0.7
    if intercon >= 0.9
        push!(validations, (param="Interconnectivity", value=intercon,
                           target=">90%", status="PASS"))
    else
        push!(validations, (param="Interconnectivity", value=intercon,
                           target=">90%", status="WARN"))
    end

    return (validations=validations, all_pass=all_pass, source=tissue.source)
end

# ============================================================================
# MAIN DESIGN WORKFLOW
# ============================================================================

"""
Complete scaffold design workflow.
"""
function design_scaffold(tissue_type::Symbol;
                         require_fda::Bool=true,
                         verbose::Bool=true)
    if !haskey(TISSUE_REQUIREMENTS, tissue_type)
        error("Unknown tissue type: $tissue_type. Available: $(keys(TISSUE_REQUIREMENTS))")
    end

    tissue = TISSUE_REQUIREMENTS[tissue_type]

    if verbose
        println("=" ^ 70)
        println("SCAFFOLD DESIGN FOR: $(tissue.name)")
        println("=" ^ 70)
        println()
        println("TISSUE REQUIREMENTS ($(tissue.source))")
        println("-" ^ 70)
        println(@sprintf("  Healing time: %.0f-%.0f weeks", tissue.healing_time_weeks...))
        println(@sprintf("  Target porosity: %.0f-%.0f%%", tissue.target_porosity[1]*100, tissue.target_porosity[2]*100))
        println(@sprintf("  Target pore size: %.0f-%.0f um", tissue.target_pore_size_um...))
        println(@sprintf("  Target modulus: %.1f-%.1f MPa", tissue.target_modulus_mpa...))
        println(@sprintf("  Cell type: %s", tissue.cell_type))
        println(@sprintf("  Vascularization critical: %s", tissue.vascularization_critical))
        println(@sprintf("  Load bearing: %s", tissue.load_bearing))
        println()
    end

    # Step 1: Select polymer
    if verbose
        println("POLYMER SELECTION")
        println("-" ^ 70)
    end

    rankings = select_polymer(tissue; require_fda=require_fda)

    if verbose
        for (i, r) in enumerate(rankings[1:min(3, length(rankings))])
            println(@sprintf("%d. %s (score: %.0f)", i, r.polymer.name, r.score))
            for reason in r.reasons
                println("   - $reason")
            end
            println()
        end
    end

    best_polymer = rankings[1].polymer

    # Step 1.5: Select crosslinking for hydrogels
    target_E = (tissue.target_modulus_mpa[1] + tissue.target_modulus_mpa[2]) / 2
    crosslink_info = select_crosslinking(best_polymer, target_E)
    crosslinking = crosslink_info.needed ? crosslink_info.method : nothing

    if verbose && crosslink_info.needed
        println("CROSSLINKING SELECTION")
        println("-" ^ 70)
        println(@sprintf("  Method: %s", crosslink_info.method.name))
        println(@sprintf("  Modulus increase: %.1fx", crosslink_info.method.E_multiplier))
        println(@sprintf("  Degradation slowdown: %.1fx", crosslink_info.method.degradation_factor))
        biocompat_adj = best_polymer.biocompatibility * (1 - crosslink_info.method.biocompat_penalty)
        println(@sprintf("  Adjusted biocompatibility: %.0f%%", biocompat_adj * 100))
        println(@sprintf("  Source: %s", crosslink_info.method.source))
        println()
    end

    # Step 2: Optimize geometry
    if verbose
        println("OPTIMAL GEOMETRY")
        println("-" ^ 70)
    end

    geometry = optimize_geometry(best_polymer, tissue; crosslinking=crosslinking)

    if verbose
        println(@sprintf("  Porosity: %.1f%%", geometry.porosity * 100))
        println(@sprintf("  Pore size: %.0f um", geometry.pore_size_um))
        println(@sprintf("  Window size: %.0f um", geometry.window_size_um))
        println(@sprintf("  Wall thickness: %.0f um", geometry.wall_thickness_um))
        println(@sprintf("  Scaffold E: %.2f MPa", geometry.E_scaffold_mpa))
        println(@sprintf("  Scaffold strength: %.3f MPa", geometry.sigma_scaffold_mpa))
        if crosslink_info.needed
            println(@sprintf("  (with %s crosslinking)", crosslink_info.method.abbrev))
        end
        println()
    end

    # Step 3: Validate design
    if verbose
        println("VALIDATION (Q1 Literature)")
        println("-" ^ 70)
    end

    validation = validate_design(geometry, tissue)

    if verbose
        for v in validation.validations
            status_str = v.status == "PASS" ? "[PASS]" : v.status == "FAIL" ? "[FAIL]" : "[WARN]"
            if typeof(v.value) <: Number
                println(@sprintf("  %s %s: %.2f (target: %s)", status_str, v.param, v.value, v.target))
            else
                println(@sprintf("  %s %s: %s (target: %s)", status_str, v.param, v.value, v.target))
            end
        end
        println(@sprintf("\n  Source: %s", validation.source))
        println()
    end

    # Step 4: Degradation profile
    if verbose
        println("DEGRADATION PROFILE")
        println("-" ^ 70)
    end

    profile = predict_degradation_profile(best_polymer, geometry)

    if verbose
        println(@sprintf("  %-8s %-10s %-10s %-10s %-10s %s",
                        "Week", "MW%", "Mass%", "Porosity%", "E(MPa)", "Status"))
        for p in profile[1:4:end]  # Every 4th entry
            println(@sprintf("  %-8.0f %-10.1f %-10.1f %-10.1f %-10.2f %s",
                            p.time_weeks, p.mw_fraction*100, p.mass_remaining*100,
                            p.porosity*100, p.E_mpa, p.integrity))
        end
        println()
    end

    # Summary
    if verbose
        println("=" ^ 70)
        println("DESIGN SUMMARY")
        println("=" ^ 70)
        println(@sprintf("  Polymer: %s (%s)", best_polymer.name, best_polymer.abbrev))
        if crosslink_info.needed
            println(@sprintf("  Crosslinking: %s (%.1fx modulus)", crosslink_info.method.name, crosslink_info.method.E_multiplier))
        end
        println(@sprintf("  Tissue: %s", tissue.name))
        println(@sprintf("  Validation: %s", validation.all_pass ? "ALL PASS" : "ISSUES FOUND"))
        # Adjusted scaffold life with crosslinking
        deg_min = best_polymer.degradation_weeks[1]
        deg_max = best_polymer.degradation_weeks[2]
        if crosslink_info.needed
            deg_min *= crosslink_info.method.degradation_factor
            deg_max *= crosslink_info.method.degradation_factor
        end
        println(@sprintf("  Estimated scaffold life: %.0f-%.0f weeks", deg_min, deg_max))
        println()
    end

    return (
        tissue = tissue,
        polymer = best_polymer,
        crosslinking = crosslinking,
        geometry = geometry,
        validation = validation,
        profile = profile
    )
end

# ============================================================================
# MAIN
# ============================================================================

function main()
    println("\n" * "=" ^ 70)
    println("SCAFFOLD DESIGN OPTIMIZATION EXAMPLES")
    println("=" ^ 70 * "\n")

    # Example 1: Trabecular bone scaffold
    design_scaffold(:trabecular_bone)

    println("\n" * "-" ^ 70 * "\n")

    # Example 2: Cartilage scaffold
    design_scaffold(:articular_cartilage)

    println("\n" * "-" ^ 70 * "\n")

    # Example 3: Skin scaffold (fast healing)
    design_scaffold(:skin)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
