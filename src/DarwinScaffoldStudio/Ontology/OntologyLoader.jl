"""
    OntologyLoader

Load and parse local ontology files (OWL, TTL, JSON) from disk E.

SUPPORTED SOURCES:
- PubChem JSON exports
- DEBBIE biomaterials ontology (OWL)
- EMMO materials ontology (TTL)

ONTOLOGY PATHS:
- /mnt/e/darwin-ontologies/pubchem/
- /mnt/e/darwin-ontologies/debbie/
- /mnt/e/darwin-ontologies/emmo/

Author: Darwin Scaffold Studio
Date: December 2025
"""
module OntologyLoader

export load_pubchem_compound, load_debbie_classes, load_emmo_materials
export OntologyPaths, get_all_biomaterials, search_ontology
export list_pubchem_compounds, search_debbie, show_search_results

using JSON

# =============================================================================
# PATHS
# =============================================================================

const ONTOLOGY_BASE = "/mnt/e/darwin-ontologies"

struct OntologyPaths
    pubchem::String
    debbie::String
    emmo::String
end

const PATHS = OntologyPaths(
    joinpath(ONTOLOGY_BASE, "pubchem"),
    joinpath(ONTOLOGY_BASE, "debbie"),
    joinpath(ONTOLOGY_BASE, "emmo")
)

# =============================================================================
# PUBCHEM LOADER
# =============================================================================

"""
Compound data from PubChem.
"""
struct PubChemCompound
    cid::Int
    name::String
    molecular_formula::String
    molecular_weight::Float64
    iupac_name::String
    smiles::String
    xlogp::Float64
    tpsa::Float64
end

"""
    load_pubchem_compound(name)

Load compound data from local PubChem JSON file.
"""
function load_pubchem_compound(name::String)
    filename = lowercase(replace(name, " " => "_")) * ".json"
    filepath = joinpath(PATHS.pubchem, filename)

    if !isfile(filepath)
        error("PubChem file not found: $filepath. Available: $(readdir(PATHS.pubchem))")
    end

    data = JSON.parsefile(filepath)
    props = data["PropertyTable"]["Properties"][1]

    return PubChemCompound(
        get(props, "CID", 0),
        name,
        get(props, "MolecularFormula", ""),
        parse(Float64, get(props, "MolecularWeight", "0")),
        get(props, "IUPACName", ""),
        get(props, "ConnectivitySMILES", get(props, "CanonicalSMILES", "")),
        get(props, "XLogP", 0.0),
        get(props, "TPSA", 0.0)
    )
end

"""
List available PubChem compounds.
"""
function list_pubchem_compounds()
    files = filter(f -> endswith(f, ".json"), readdir(PATHS.pubchem))
    return [replace(f, ".json" => "") for f in files]
end

# =============================================================================
# DEBBIE OWL LOADER
# =============================================================================

"""
Biomaterial class from DEBBIE ontology.
"""
struct DEBBIEClass
    uri::String
    name::String
    aliases::Vector{String}
    parent_class::String
    definition::String
end

"""
    load_debbie_classes()

Parse DEBBIE OWL file and extract biomaterial classes.
"""
function load_debbie_classes()
    owl_path = joinpath(PATHS.debbie, "ontology", "DEB_ont.owl")

    if !isfile(owl_path)
        error("DEBBIE ontology not found: $owl_path")
    end

    content = read(owl_path, String)
    classes = DEBBIEClass[]

    # Simple regex-based parsing (not full OWL parser)
    # Pattern: <owl:Class rdf:about="...#ClassName">
    class_pattern = r"<owl:Class rdf:about=\"([^\"]+)\">"
    subclass_pattern = r"<rdfs:subClassOf rdf:resource=\"([^\"]+)\"/>"
    altlabel_pattern = r"<skos:altLabel>([^<]+)</skos:altLabel>"
    definition_pattern = r"<dc:definition>([^<]+)</dc:definition>"

    # Split by class definitions
    class_blocks = split(content, "<!-- http://")

    for block in class_blocks
        # Find class URI
        m = match(class_pattern, block)
        if m === nothing
            continue
        end

        uri = m.captures[1]
        name = split(uri, "#")[end]

        # Find parent class
        parent = ""
        pm = match(subclass_pattern, block)
        if pm !== nothing
            parent = split(pm.captures[1], "#")[end]
        end

        # Find aliases
        aliases = String[]
        for am in eachmatch(altlabel_pattern, block)
            push!(aliases, am.captures[1])
        end

        # Find definition
        definition = ""
        dm = match(definition_pattern, block)
        if dm !== nothing
            definition = dm.captures[1]
        end

        push!(classes, DEBBIEClass(uri, name, aliases, parent, definition))
    end

    return classes
end

"""
Get all biomaterial classes (materials that inherit from Biomaterial).
"""
function get_all_biomaterials()
    classes = load_debbie_classes()
    return filter(c -> c.parent_class == "Biomaterial", classes)
end

"""
Search for a specific material in DEBBIE.
"""
function search_debbie(query::String)
    classes = load_debbie_classes()
    query_lower = lowercase(query)

    results = DEBBIEClass[]

    for c in classes
        # Check name
        if occursin(query_lower, lowercase(c.name))
            push!(results, c)
            continue
        end

        # Check aliases
        for alias in c.aliases
            if occursin(query_lower, lowercase(alias))
                push!(results, c)
                break
            end
        end
    end

    return results
end

# =============================================================================
# EMMO TTL LOADER
# =============================================================================

"""
EMMO material class.
"""
struct EMMOClass
    uri::String
    name::String
    definition::String
    alt_names::Vector{String}
end

"""
    load_emmo_materials()

Parse EMMO materials.ttl file.
"""
function load_emmo_materials()
    ttl_path = joinpath(PATHS.emmo, "core", "disciplines", "materials.ttl")

    if !isfile(ttl_path)
        error("EMMO materials not found: $ttl_path")
    end

    content = read(ttl_path, String)
    classes = EMMOClass[]

    # Parse TTL format (simplified)
    # Pattern: skos:prefLabel "Name"@en
    prefLabel_pattern = r"skos:prefLabel \"([^\"]+)\"@en"
    definition_pattern = r":EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9 \"([^\"]+)\""

    # Split by class definitions (###)
    blocks = split(content, "###")

    for block in blocks
        # Find URI
        uri_match = match(r"(https://w3id\.org/emmo#EMMO_[a-f0-9_]+)", block)
        if uri_match === nothing
            continue
        end
        uri = uri_match.captures[1]

        # Find name
        name = ""
        name_match = match(prefLabel_pattern, block)
        if name_match !== nothing
            name = name_match.captures[1]
        end

        if isempty(name)
            continue
        end

        # Find definition
        definition = ""
        def_match = match(definition_pattern, block)
        if def_match !== nothing
            definition = def_match.captures[1]
        end

        push!(classes, EMMOClass(uri, name, definition, String[]))
    end

    return classes
end

# =============================================================================
# UNIFIED SEARCH
# =============================================================================

"""
    search_ontology(query)

Search across all ontologies for a material/compound.
"""
function search_ontology(query::String)
    results = Dict{String, Any}()

    # Search PubChem
    pubchem_files = list_pubchem_compounds()
    query_lower = lowercase(query)
    pubchem_matches = filter(f -> occursin(query_lower, lowercase(f)), pubchem_files)
    if !isempty(pubchem_matches)
        results["pubchem"] = [load_pubchem_compound(replace(m, "_" => " ")) for m in pubchem_matches]
    end

    # Search DEBBIE
    debbie_matches = search_debbie(query)
    if !isempty(debbie_matches)
        results["debbie"] = debbie_matches
    end

    # Search EMMO
    emmo_classes = load_emmo_materials()
    emmo_matches = filter(c -> occursin(query_lower, lowercase(c.name)), emmo_classes)
    if !isempty(emmo_matches)
        results["emmo"] = emmo_matches
    end

    return results
end

# =============================================================================
# DISPLAY
# =============================================================================

function show_search_results(query::String)
    results = search_ontology(query)

    println("\n" * "="^60)
    println("ONTOLOGY SEARCH: \"$query\"")
    println("="^60)

    if haskey(results, "pubchem")
        println("\n--- PubChem ---")
        for c in results["pubchem"]
            println("  CID $(c.cid): $(c.name)")
            println("    Formula: $(c.molecular_formula)")
            println("    MW: $(c.molecular_weight) g/mol")
            println("    SMILES: $(c.smiles)")
        end
    end

    if haskey(results, "debbie")
        println("\n--- DEBBIE (Biomaterials) ---")
        for c in results["debbie"]
            println("  $(c.name)")
            if !isempty(c.aliases)
                println("    Aliases: $(join(c.aliases, ", "))")
            end
            println("    Parent: $(c.parent_class)")
        end
    end

    if haskey(results, "emmo")
        println("\n--- EMMO (Materials) ---")
        for c in results["emmo"]
            println("  $(c.name)")
            if !isempty(c.definition)
                println("    Definition: $(first(c.definition, 100))...")
            end
        end
    end

    if isempty(results)
        println("\nNo results found.")
    end

    println("="^60)
end

end # module
