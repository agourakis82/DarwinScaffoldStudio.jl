"""
    IntegratedOntology

Unified ontology system that combines:
1. Local knowledge base (MaterialsOntology.jl) - curated parameters
2. External ontologies (OntologyLoader.jl) - DEBBIE, EMMO, PubChem

WORKFLOW:
1. Query local KB first (fast, curated)
2. If not found, search external ontologies
3. Merge data from multiple sources
4. Track provenance of all values

Author: Darwin Scaffold Studio
Date: December 2025
"""
module IntegratedOntology

export query_material, MaterialInfo, get_degradation_model_params
export list_all_materials, enrich_from_external
export show_material, show_all_materials

# Include sub-modules
include("MaterialsOntology.jl")
include("OntologyLoader.jl")

using .MaterialsOntology
using .OntologyLoader

using Printf

# =============================================================================
# TYPES
# =============================================================================

"""
Complete material information from all sources.
"""
struct MaterialInfo
    # Identity
    name::String
    aliases::Vector{String}

    # External IDs
    pubchem_cid::Int
    chebi_id::String
    debbie_uri::String
    emmo_uri::String

    # Structural (from PubChem)
    molecular_formula::String
    molecular_weight::Float64
    smiles::String

    # Physical properties (from MaterialsOntology)
    density::Float64
    Tg_infinity::Float64
    Tm::Float64

    # Degradation kinetics
    k_hydrolysis_37C::Float64
    k_source::String
    Ea_hydrolysis::Float64

    # Fox-Flory
    fox_flory_K::Float64

    # Sources
    sources::Vector{String}
end

# =============================================================================
# QUERY FUNCTIONS
# =============================================================================

"""
    query_material(name)

Query material from all available ontologies.
Priority: Local KB → DEBBIE → EMMO → PubChem
"""
function query_material(name::String)
    sources = String[]

    # 1. Try local MaterialsOntology first
    local_polymer = MaterialsOntology.query_polymer(name)

    if local_polymer !== nothing
        push!(sources, "MaterialsOntology (local)")

        # Build MaterialInfo from local data
        return MaterialInfo(
            local_polymer.name,
            local_polymer.aliases,
            0,  # pubchem_cid - enrich later
            local_polymer.chebi_id,
            "",  # debbie_uri
            local_polymer.id,
            "",  # formula - from monomer
            local_polymer.repeat_unit_mw,
            "",  # smiles
            local_polymer.density,
            local_polymer.Tg_infinity.value,
            local_polymer.Tm.value,
            local_polymer.k_hydrolysis_37C.value,
            local_polymer.k_hydrolysis_37C.source,
            local_polymer.Ea_hydrolysis.value,
            local_polymer.fox_flory_K.value,
            sources
        )
    end

    # 2. Search external ontologies
    external = OntologyLoader.search_ontology(name)

    # Initialize with defaults
    info = MaterialInfo(
        name, String[], 0, "", "", "", "", 0.0, "",
        0.0, 0.0, 0.0, 0.0, "", 0.0, 0.0, sources
    )

    # Merge from DEBBIE
    if haskey(external, "debbie") && !isempty(external["debbie"])
        deb = external["debbie"][1]
        push!(sources, "DEBBIE")
        info = MaterialInfo(
            deb.name,
            deb.aliases,
            info.pubchem_cid,
            info.chebi_id,
            deb.uri,
            info.emmo_uri,
            info.molecular_formula,
            info.molecular_weight,
            info.smiles,
            info.density,
            info.Tg_infinity,
            info.Tm,
            info.k_hydrolysis_37C,
            info.k_source,
            info.Ea_hydrolysis,
            info.fox_flory_K,
            sources
        )
    end

    # Merge from PubChem (for monomers)
    if haskey(external, "pubchem") && !isempty(external["pubchem"])
        pub = external["pubchem"][1]
        push!(sources, "PubChem")
        info = MaterialInfo(
            info.name,
            info.aliases,
            pub.cid,
            info.chebi_id,
            info.debbie_uri,
            info.emmo_uri,
            pub.molecular_formula,
            pub.molecular_weight,
            pub.smiles,
            info.density,
            info.Tg_infinity,
            info.Tm,
            info.k_hydrolysis_37C,
            info.k_source,
            info.Ea_hydrolysis,
            info.fox_flory_K,
            sources
        )
    end

    return info
end

"""
    get_degradation_model_params(material_name)

Get parameters needed for degradation modeling.
Returns named tuple with all physics parameters.
"""
function get_degradation_model_params(material_name::String)
    # Try local KB first
    polymer = MaterialsOntology.query_polymer(material_name)

    if polymer !== nothing
        return (
            name = polymer.name,
            k_hydrolysis = polymer.k_hydrolysis_37C.value,
            k_source = polymer.k_hydrolysis_37C.source,
            k_uncertainty = polymer.k_hydrolysis_37C.uncertainty,
            Ea = polymer.Ea_hydrolysis.value,
            Ea_source = polymer.Ea_hydrolysis.source,
            Tg_infinity = polymer.Tg_infinity.value,
            Tg_source = polymer.Tg_infinity.source,
            fox_flory_K = polymer.fox_flory_K.value,
            density = polymer.density,
            repeat_unit_mw = polymer.repeat_unit_mw,
            found_in = "MaterialsOntology"
        )
    end

    # Search external
    external = OntologyLoader.search_ontology(material_name)

    if haskey(external, "debbie")
        # Found in DEBBIE but no kinetic data
        deb = external["debbie"][1]
        return (
            name = deb.name,
            k_hydrolysis = 0.0,  # Not available in DEBBIE
            k_source = "Not available - use literature",
            k_uncertainty = 1.0,
            Ea = 0.0,
            Ea_source = "Not available",
            Tg_infinity = 0.0,
            Tg_source = "Not available",
            fox_flory_K = 0.0,
            density = 0.0,
            repeat_unit_mw = 0.0,
            found_in = "DEBBIE (structural only)"
        )
    end

    error("Material '$material_name' not found in any ontology")
end

"""
    list_all_materials()

List all materials available from all sources.
"""
function list_all_materials()
    materials = Dict{String, Vector{String}}()

    # Local KB
    materials["MaterialsOntology"] = String[]
    for (id, polymer) in MaterialsOntology.POLYMER_ONTOLOGY
        push!(materials["MaterialsOntology"], polymer.name)
    end

    # DEBBIE biomaterials
    materials["DEBBIE"] = String[]
    try
        for bm in OntologyLoader.get_all_biomaterials()
            push!(materials["DEBBIE"], bm.name)
        end
    catch e
        @warn "Could not load DEBBIE: $e"
    end

    # PubChem compounds
    materials["PubChem"] = String[]
    try
        for c in OntologyLoader.list_pubchem_compounds()
            push!(materials["PubChem"], replace(c, "_" => " "))
        end
    catch e
        @warn "Could not load PubChem: $e"
    end

    return materials
end

# =============================================================================
# DISPLAY
# =============================================================================

function show_material(name::String)
    info = query_material(name)

    println("\n" * "="^70)
    println("MATERIAL: $(info.name)")
    println("="^70)

    println("\nIdentifiers:")
    if !isempty(info.aliases)
        println("  Aliases: $(join(info.aliases, ", "))")
    end
    if info.pubchem_cid > 0
        println("  PubChem CID: $(info.pubchem_cid)")
    end
    if !isempty(info.chebi_id)
        println("  ChEBI: $(info.chebi_id)")
    end
    if !isempty(info.debbie_uri)
        println("  DEBBIE: $(info.debbie_uri)")
    end

    if !isempty(info.molecular_formula)
        println("\nStructure:")
        println("  Formula: $(info.molecular_formula)")
        println("  MW: $(info.molecular_weight) g/mol")
        if !isempty(info.smiles)
            println("  SMILES: $(info.smiles)")
        end
    end

    if info.density > 0
        println("\nPhysical Properties:")
        @printf("  Density: %.2f g/cm³\n", info.density)
        if info.Tg_infinity > 0
            @printf("  Tg∞: %.1f °C\n", info.Tg_infinity)
        end
        if info.Tm > 0
            @printf("  Tm: %.1f °C\n", info.Tm)
        end
    end

    if info.k_hydrolysis_37C > 0
        println("\nDegradation Kinetics:")
        @printf("  k (37°C): %.4f day⁻¹ [%s]\n", info.k_hydrolysis_37C, info.k_source)
        @printf("  Ea: %.1f kJ/mol\n", info.Ea_hydrolysis)
        @printf("  Fox-Flory K: %.1f kg/mol\n", info.fox_flory_K)
    end

    println("\nSources: $(join(info.sources, ", "))")
    println("="^70)
end

function show_all_materials()
    materials = list_all_materials()

    println("\n" * "="^70)
    println("AVAILABLE MATERIALS BY SOURCE")
    println("="^70)

    for (source, names) in materials
        println("\n--- $source ($(length(names)) materials) ---")
        for name in sort(names)[1:min(15, length(names))]
            println("  • $name")
        end
        if length(names) > 15
            println("  ... and $(length(names) - 15) more")
        end
    end

    println("\n" * "="^70)
end

end # module
