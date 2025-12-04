"""
    OntologyManager

3-Tier Ontology Lookup System for Darwin Scaffold Studio.

## Architecture
- TIER 1: Hardcoded core terms (~150) - instant, offline
- TIER 2: Extended OWL subsets (~5000) - lazy load, local
- TIER 3: Online API lookup (OLS/BioPortal) - cached in SQLite

## APIs Used
- EBI OLS: https://www.ebi.ac.uk/ols4/api
- NCBO BioPortal: https://data.bioontology.org

## FAIR Compliance
- RDF/Turtle export
- JSON-LD with Schema.org
- Persistent URIs (PURLs)

# Author: Dr. Demetrios Agourakis
# Master's Thesis: Tissue Engineering Scaffold Optimization
"""
module OntologyManager

using HTTP
using JSON3
using SQLite
using Dates

# Import core OBO terms from parent module's OBOFoundry
using ...OBOFoundry: OBOTerm, UBERON, CL, CHEBI, NCIT, GO, BTO, DOID
using ...OBOFoundry: lookup_term as core_lookup, get_iri

export OntologyConfig, init_ontology_system, shutdown_ontology_system
export smart_lookup, batch_lookup, search_terms
export get_ancestors, get_descendants, get_related_terms
export export_rdf, export_jsonld, export_annotation
export OntologyStats, get_stats, clear_cache
export ONTOLOGY_PREFIXES

#=============================================================================
  CONFIGURATION
=============================================================================#

"""Supported ontology prefixes and their metadata."""
const ONTOLOGY_PREFIXES = Dict{String,NamedTuple}(
    "UBERON" => (name="Uber Anatomy Ontology", ols_id="uberon", category=:anatomy),
    "CL" => (name="Cell Ontology", ols_id="cl", category=:cell),
    "CHEBI" => (name="Chemical Entities of Biological Interest", ols_id="chebi", category=:chemical),
    "GO" => (name="Gene Ontology", ols_id="go", category=:process),
    "NCIT" => (name="NCI Thesaurus", ols_id="ncit", category=:disease),
    "BTO" => (name="BRENDA Tissue Ontology", ols_id="bto", category=:cell_line),
    "DOID" => (name="Disease Ontology", ols_id="doid", category=:disease),
    "HP" => (name="Human Phenotype Ontology", ols_id="hp", category=:phenotype),
    "MP" => (name="Mammalian Phenotype Ontology", ols_id="mp", category=:phenotype),
    "PATO" => (name="Phenotype And Trait Ontology", ols_id="pato", category=:quality),
    "OBI" => (name="Ontology for Biomedical Investigations", ols_id="obi", category=:investigation),
    "BAO" => (name="BioAssay Ontology", ols_id="bao", category=:assay),
    "EFO" => (name="Experimental Factor Ontology", ols_id="efo", category=:experimental),
    "MONDO" => (name="Mondo Disease Ontology", ols_id="mondo", category=:disease),
    "PR" => (name="Protein Ontology", ols_id="pr", category=:protein),
    "SO" => (name="Sequence Ontology", ols_id="so", category=:sequence),
    "ECO" => (name="Evidence & Conclusion Ontology", ols_id="eco", category=:evidence),
    "IAO" => (name="Information Artifact Ontology", ols_id="iao", category=:information),
)

"""
    OntologyConfig

Configuration for ontology manager.
"""
Base.@kwdef mutable struct OntologyConfig
    # API endpoints
    ols_base_url::String = "https://www.ebi.ac.uk/ols4/api"
    bioportal_base_url::String = "https://data.bioontology.org"
    bioportal_api_key::String = ""  # Optional, for higher rate limits

    # Cache settings
    cache_db_path::String = joinpath(homedir(), ".darwin_scaffold_studio", "ontology_cache.db")
    cache_ttl_days::Int = 30
    max_cache_size_mb::Int = 100

    # Performance
    api_timeout_seconds::Int = 10
    max_retries::Int = 3
    batch_size::Int = 50
    enable_tier3::Bool = true  # Can disable API calls for offline mode

    # Logging
    verbose::Bool = false
end

# Global state
const CONFIG = Ref{OntologyConfig}(OntologyConfig())
const DB_CONNECTION = Ref{Union{SQLite.DB,Nothing}}(nothing)
const STATS = Ref{Dict{String,Int}}(Dict(
    "tier1_hits" => 0,
    "tier2_hits" => 0,
    "tier3_hits" => 0,
    "cache_hits" => 0,
    "api_calls" => 0,
    "api_errors" => 0
))

#=============================================================================
  INITIALIZATION
=============================================================================#

"""
    init_ontology_system(; config_overrides...)

Initialize the ontology management system.
Creates cache database and loads extended terms.
"""
function init_ontology_system(; kwargs...)
    # Apply config overrides
    config = OntologyConfig(; kwargs...)
    CONFIG[] = config

    # Ensure cache directory exists
    cache_dir = dirname(config.cache_db_path)
    isdir(cache_dir) || mkpath(cache_dir)

    # Initialize SQLite cache
    DB_CONNECTION[] = SQLite.DB(config.cache_db_path)
    init_cache_schema()

    # Reset stats
    STATS[] = Dict(
        "tier1_hits" => 0,
        "tier2_hits" => 0,
        "tier3_hits" => 0,
        "cache_hits" => 0,
        "api_calls" => 0,
        "api_errors" => 0
    )

    config.verbose && @info "Ontology system initialized" cache_path = config.cache_db_path

    return nothing
end

"""
    shutdown_ontology_system()

Clean shutdown of ontology system.
"""
function shutdown_ontology_system()
    if !isnothing(DB_CONNECTION[])
        SQLite.close(DB_CONNECTION[])
        DB_CONNECTION[] = nothing
    end
end

"""Initialize SQLite cache schema."""
function init_cache_schema()
    db = DB_CONNECTION[]
    isnothing(db) && return

    # Terms cache table
    SQLite.execute(
        db,
        """
    CREATE TABLE IF NOT EXISTS terms (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        definition TEXT,
        synonyms TEXT,
        parents TEXT,
        part_of TEXT,
        xrefs TEXT,
        ontology TEXT,
        iri TEXT,
        cached_at TEXT,
        source TEXT
    )
"""
    )

    # Search cache table
    SQLite.execute(
        db,
        """
    CREATE TABLE IF NOT EXISTS searches (
        query TEXT,
        ontology TEXT,
        results TEXT,
        cached_at TEXT,
        PRIMARY KEY (query, ontology)
    )
"""
    )

    # Relationships cache
    SQLite.execute(
        db,
        """
    CREATE TABLE IF NOT EXISTS relationships (
        term_id TEXT,
        relation_type TEXT,
        related_ids TEXT,
        cached_at TEXT,
        PRIMARY KEY (term_id, relation_type)
    )
"""
    )

    # Create indexes
    SQLite.execute(db, "CREATE INDEX IF NOT EXISTS idx_terms_ontology ON terms(ontology)")
    SQLite.execute(db, "CREATE INDEX IF NOT EXISTS idx_terms_cached ON terms(cached_at)")
end

#=============================================================================
  3-TIER LOOKUP SYSTEM
=============================================================================#

"""
    smart_lookup(id::String) -> Union{OBOTerm, Nothing}

Intelligent lookup across all tiers.
Returns cached result or fetches from API if needed.
"""
function smart_lookup(id::String)
    # Validate ID format
    if !contains(id, ":")
        CONFIG[].verbose && @warn "Invalid term ID format" id
        return nothing
    end

    # TIER 1: Check hardcoded core terms (instant)
    term = core_lookup(id)
    if !isnothing(term)
        STATS[]["tier1_hits"] += 1
        return term
    end

    # TIER 2: Check SQLite cache
    term = lookup_cache(id)
    if !isnothing(term)
        STATS[]["cache_hits"] += 1
        return term
    end

    # TIER 3: API lookup (if enabled)
    if CONFIG[].enable_tier3
        term = lookup_ols_api(id)
        if !isnothing(term)
            STATS[]["tier3_hits"] += 1
            cache_term(term)
            return term
        end
    end

    return nothing
end

"""
    batch_lookup(ids::Vector{String}) -> Dict{String, Union{OBOTerm, Nothing}}

Batch lookup for multiple terms. Optimizes API calls.
"""
function batch_lookup(ids::Vector{String})
    results = Dict{String,Union{OBOTerm,Nothing}}()
    missing_ids = String[]

    # First pass: check local sources
    for id in ids
        term = core_lookup(id)
        if !isnothing(term)
            results[id] = term
            STATS[]["tier1_hits"] += 1
        else
            term = lookup_cache(id)
            if !isnothing(term)
                results[id] = term
                STATS[]["cache_hits"] += 1
            else
                push!(missing_ids, id)
            end
        end
    end

    # Batch API lookup for missing
    if CONFIG[].enable_tier3 && !isempty(missing_ids)
        for chunk in Iterators.partition(missing_ids, CONFIG[].batch_size)
            for id in chunk
                term = lookup_ols_api(id)
                results[id] = term
                if !isnothing(term)
                    STATS[]["tier3_hits"] += 1
                    cache_term(term)
                end
            end
        end
    end

    results
end

#=============================================================================
  CACHE OPERATIONS
=============================================================================#

"""Lookup term in SQLite cache."""
function lookup_cache(id::String)
    db = DB_CONNECTION[]
    isnothing(db) && return nothing

    result = SQLite.DBInterface.execute(db,
        "SELECT * FROM terms WHERE id = ? AND datetime(cached_at) > datetime('now', '-' || ? || ' days')",
        [id, CONFIG[].cache_ttl_days]
    ) |> SQLite.Tables.rowtable

    isempty(result) && return nothing

    row = first(result)
    OBOTerm(
        row.id,
        row.name,
        something(row.definition, ""),
        isnothing(row.synonyms) ? String[] : split(row.synonyms, "|"),
        isnothing(row.parents) ? String[] : split(row.parents, "|"),
        isnothing(row.part_of) ? String[] : split(row.part_of, "|"),
        isnothing(row.xrefs) ? String[] : split(row.xrefs, "|"),
        Symbol(row.ontology)
    )
end

"""Cache term in SQLite."""
function cache_term(term::OBOTerm)
    db = DB_CONNECTION[]
    isnothing(db) && return

    SQLite.DBInterface.execute(db, """
        INSERT OR REPLACE INTO terms
        (id, name, definition, synonyms, parents, part_of, xrefs, ontology, iri, cached_at, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'api')
    """, [
            term.id,
            term.name,
            term.definition,
            join(term.synonyms, "|"),
            join(term.parents, "|"),
            join(term.part_of, "|"),
            join(term.xrefs, "|"),
            String(term.ontology),
            get_iri(term)
        ])
end

"""Clear expired cache entries."""
function clear_cache(; all::Bool=false)
    db = DB_CONNECTION[]
    isnothing(db) && return 0

    if all
        SQLite.execute(db, "DELETE FROM terms WHERE source = 'api'")
        SQLite.execute(db, "DELETE FROM searches")
        SQLite.execute(db, "DELETE FROM relationships")
    else
        SQLite.execute(db,
            "DELETE FROM terms WHERE datetime(cached_at) < datetime('now', '-' || ? || ' days')",
            [CONFIG[].cache_ttl_days]
        )
    end

    # Vacuum to reclaim space
    SQLite.execute(db, "VACUUM")

    return SQLite.changes(db)
end

#=============================================================================
  OLS API CLIENT
=============================================================================#

"""
    lookup_ols_api(id::String) -> Union{OBOTerm, Nothing}

Fetch term from EBI Ontology Lookup Service.
"""
function lookup_ols_api(id::String)
    prefix, local_id = split(id, ":")

    # Check if ontology is supported
    if !haskey(ONTOLOGY_PREFIXES, prefix)
        CONFIG[].verbose && @warn "Unsupported ontology prefix" prefix
        return nothing
    end

    ontology_id = ONTOLOGY_PREFIXES[prefix].ols_id

    # OLS4 API uses double-encoded IRI
    iri = "http://purl.obolibrary.org/obo/$(prefix)_$(local_id)"
    encoded_iri = HTTP.URIs.escapeuri(HTTP.URIs.escapeuri(iri))

    url = "$(CONFIG[].ols_base_url)/ontologies/$(ontology_id)/terms/$(encoded_iri)"

    try
        STATS[]["api_calls"] += 1

        response = HTTP.get(url;
            headers=["Accept" => "application/json"],
            readtimeout=CONFIG[].api_timeout_seconds,
            retry=false
        )

        if response.status == 200
            data = JSON3.read(response.body)
            return parse_ols_term(data, prefix)
        end
    catch e
        STATS[]["api_errors"] += 1
        if CONFIG[].verbose
            @warn "OLS API error" id exception = e
        end
    end

    return nothing
end

"""Parse OLS API response into OBOTerm."""
function parse_ols_term(data, prefix::AbstractString)
    # Extract ID from IRI
    iri = get(data, :iri, "")
    id = if contains(iri, "_")
        parts = split(basename(iri), "_")
        "$(parts[1]):$(parts[2])"
    else
        "$(prefix):unknown"
    end

    # Get basic fields
    name = get(data, :label, "Unknown")
    definition = ""
    if haskey(data, :description) && !isempty(data.description)
        definition = first(data.description)
    end

    # Get synonyms
    synonyms = String[]
    if haskey(data, :synonyms) && !isnothing(data.synonyms)
        synonyms = String.(data.synonyms)
    end

    # Get parents (is_a relationships)
    parents = String[]
    if haskey(data, :_links) && haskey(data._links, :parents)
        # Would need additional API call - skip for now
    end

    # Get cross-references
    xrefs = String[]
    if haskey(data, :obo_xref) && !isnothing(data.obo_xref)
        for xref in data.obo_xref
            if haskey(xref, :id)
                push!(xrefs, xref.id)
            end
        end
    end

    OBOTerm(id, name;
        definition=definition,
        synonyms=synonyms,
        parents=parents,
        xrefs=xrefs
    )
end

#=============================================================================
  SEARCH FUNCTIONALITY
=============================================================================#

"""
    search_terms(query::String; ontology::String="", limit::Int=20) -> Vector{OBOTerm}

Search for terms matching query string.
"""
function search_terms(query::String; ontology::String="", limit::Int=20)
    results = OBOTerm[]

    # First search local terms
    local_results = search_local(query, ontology)
    append!(results, local_results)

    # If not enough results, search API
    if length(results) < limit && CONFIG[].enable_tier3
        api_results = search_ols_api(query, ontology, limit - length(results))
        append!(results, api_results)
    end

    # Deduplicate by ID
    seen = Set{String}()
    unique_results = OBOTerm[]
    for term in results
        if !(term.id in seen)
            push!(seen, term.id)
            push!(unique_results, term)
        end
    end

    first(unique_results, limit)
end

"""Search local hardcoded terms."""
function search_local(query::String, ontology::String)
    results = OBOTerm[]
    query_lower = lowercase(query)

    # Search all ontology dictionaries
    for (dict_name, dict) in [("UBERON", UBERON), ("CL", CL), ("CHEBI", CHEBI),
        ("NCIT", NCIT), ("GO", GO), ("BTO", BTO), ("DOID", DOID)]
        if !isempty(ontology) && ontology != dict_name
            continue
        end

        for (id, term) in dict
            # Match name, synonyms, or definition
            if contains(lowercase(term.name), query_lower) ||
               any(s -> contains(lowercase(s), query_lower), term.synonyms) ||
               contains(lowercase(term.definition), query_lower)
                push!(results, term)
            end
        end
    end

    results
end

"""Search OLS API."""
function search_ols_api(query::String, ontology::String, limit::Int)
    results = OBOTerm[]

    params = ["q" => query, "rows" => string(limit)]
    if !isempty(ontology) && haskey(ONTOLOGY_PREFIXES, ontology)
        push!(params, "ontology" => ONTOLOGY_PREFIXES[ontology].ols_id)
    end

    url = "$(CONFIG[].ols_base_url)/search?" * join(["$(k)=$(HTTP.URIs.escapeuri(v))" for (k, v) in params], "&")

    try
        STATS[]["api_calls"] += 1
        response = HTTP.get(url;
            headers=["Accept" => "application/json"],
            readtimeout=CONFIG[].api_timeout_seconds
        )

        if response.status == 200
            data = JSON3.read(response.body)
            if haskey(data, :response) && haskey(data.response, :docs)
                for doc in data.response.docs
                    term = parse_ols_search_result(doc)
                    if !isnothing(term)
                        push!(results, term)
                        cache_term(term)
                    end
                end
            end
        end
    catch e
        STATS[]["api_errors"] += 1
        CONFIG[].verbose && @warn "OLS search error" query exception = e
    end

    results
end

"""Parse OLS search result."""
function parse_ols_search_result(doc)
    id = get(doc, :obo_id, nothing)
    isnothing(id) && return nothing

    name = get(doc, :label, "Unknown")
    definition = ""
    if haskey(doc, :description) && !isnothing(doc.description) && !isempty(doc.description)
        definition = first(doc.description)
    end

    ontology = String(split(id, ":")[1])

    OBOTerm(id, name;
        definition=definition,
        synonyms=String[]
    )
end

#=============================================================================
  RELATIONSHIP QUERIES
=============================================================================#

"""
    get_ancestors(id::String; max_depth::Int=10) -> Vector{OBOTerm}

Get all ancestor terms (parent, grandparent, etc.)
"""
function get_ancestors(id::String; max_depth::Int=10)
    ancestors = OBOTerm[]
    visited = Set{String}()
    queue = [(id, 0)]

    while !isempty(queue)
        current_id, depth = popfirst!(queue)

        if current_id in visited || depth > max_depth
            continue
        end
        push!(visited, current_id)

        term = smart_lookup(current_id)
        if !isnothing(term) && current_id != id
            push!(ancestors, term)
        end

        if !isnothing(term)
            for parent_id in term.parents
                push!(queue, (parent_id, depth + 1))
            end
        end
    end

    ancestors
end

"""
    get_descendants(id::String; max_depth::Int=3) -> Vector{OBOTerm}

Get descendant terms (children, grandchildren, etc.)
Note: Limited depth to avoid explosion.
"""
function get_descendants(id::String; max_depth::Int=3)
    # This would require reverse index or API calls
    # For now, search local terms only
    descendants = OBOTerm[]

    for (dict_name, dict) in [("UBERON", UBERON), ("CL", CL), ("CHEBI", CHEBI),
        ("NCIT", NCIT), ("GO", GO), ("BTO", BTO)]
        for (term_id, term) in dict
            if id in term.parents
                push!(descendants, term)
            end
        end
    end

    descendants
end

"""
    get_related_terms(id::String) -> Dict{String, Vector{OBOTerm}}

Get all related terms organized by relationship type.
"""
function get_related_terms(id::String)
    term = smart_lookup(id)
    isnothing(term) && return Dict{String,Vector{OBOTerm}}()

    Dict(
        "parents" => [t for t in [smart_lookup(p) for p in term.parents] if !isnothing(t)],
        "children" => get_descendants(id, max_depth=1),
        "part_of" => [t for t in [smart_lookup(p) for p in term.part_of] if !isnothing(t)],
        "ancestors" => get_ancestors(id, max_depth=5)
    )
end

#=============================================================================
  EXPORT FUNCTIONS (FAIR Compliance)
=============================================================================#

"""
    export_rdf(terms::Vector{OBOTerm}, filepath::String)

Export terms to RDF/Turtle format for linked data.
"""
function export_rdf(terms::Vector{OBOTerm}, filepath::String)
    open(filepath, "w") do io
        # Prefixes
        println(io, "@prefix obo: <http://purl.obolibrary.org/obo/> .")
        println(io, "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
        println(io, "@prefix oboInOwl: <http://www.geneontology.org/formats/oboInOwl#> .")
        println(io, "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .")
        println(io, "@prefix dss: <http://darwin-scaffold-studio.org/ontology#> .")
        println(io)

        for term in terms
            prefix, local_id = split(term.id, ":")
            uri = "obo:$(prefix)_$(local_id)"

            println(io, "$(uri) a owl:Class ;")
            println(io, "    rdfs:label \"$(escape_turtle(term.name))\" ;")

            if !isempty(term.definition)
                println(io, "    obo:IAO_0000115 \"$(escape_turtle(term.definition))\" ;")
            end

            for syn in term.synonyms
                println(io, "    oboInOwl:hasExactSynonym \"$(escape_turtle(syn))\" ;")
            end

            for parent in term.parents
                p_prefix, p_local = split(parent, ":")
                println(io, "    rdfs:subClassOf obo:$(p_prefix)_$(p_local) ;")
            end

            println(io, "    .")
            println(io)
        end
    end

    filepath
end

"""Escape string for Turtle format."""
function escape_turtle(s::String)
    replace(replace(replace(s, "\\" => "\\\\"), "\"" => "\\\""), "\n" => "\\n")
end

"""
    export_jsonld(annotation::Dict, filepath::String)

Export scaffold annotation to JSON-LD with Schema.org.
"""
function export_jsonld(annotation::Dict, filepath::String)
    # Build JSON-LD structure
    jsonld = Dict{String,Any}(
        "@context" => Dict(
            "@vocab" => "http://schema.org/",
            "obo" => "http://purl.obolibrary.org/obo/",
            "dss" => "http://darwin-scaffold-studio.org/ontology#",
            "tissue" => Dict("@id" => "dss:targetTissue", "@type" => "@id"),
            "material" => Dict("@id" => "dss:scaffoldMaterial", "@type" => "@id"),
            "cells" => Dict("@id" => "dss:seededCells", "@type" => "@id", "@container" => "@set"),
            "processes" => Dict("@id" => "dss:biologicalProcesses", "@type" => "@id", "@container" => "@set")
        ),
        "@type" => "dss:Scaffold",
        "dateCreated" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        "creator" => Dict(
            "@type" => "Organization",
            "name" => "Darwin Scaffold Studio"
        )
    )

    # Add tissue
    if haskey(annotation, "tissue") && !isnothing(annotation["tissue"])
        jsonld["tissue"] = get_iri(annotation["tissue"])
        jsonld["tissueLabel"] = annotation["tissue"].name
    end

    # Add material
    if haskey(annotation, "material") && !isnothing(annotation["material"])
        jsonld["material"] = get_iri(annotation["material"])
        jsonld["materialLabel"] = annotation["material"].name
    end

    # Add cells
    if haskey(annotation, "cells")
        cells = [c for c in annotation["cells"] if !isnothing(c)]
        if !isempty(cells)
            jsonld["cells"] = [get_iri(c) for c in cells]
            jsonld["cellLabels"] = [c.name for c in cells]
        end
    end

    # Add processes
    if haskey(annotation, "biological_processes")
        procs = [p for p in annotation["biological_processes"] if !isnothing(p)]
        if !isempty(procs)
            jsonld["processes"] = [get_iri(p) for p in procs]
            jsonld["processLabels"] = [p.name for p in procs]
        end
    end

    # Write to file
    open(filepath, "w") do io
        JSON3.pretty(io, jsonld)
    end

    filepath
end

"""
    export_annotation(workspace, filepath::String; format::Symbol=:jsonld)

Export scaffold workspace with full ontological annotation.
"""
function export_annotation(workspace, filepath::String; format::Symbol=:jsonld)
    # Build annotation from workspace
    annotation = Dict{String,Any}()

    # Get tissue term
    if isdefined(workspace, :tissue_target)
        tissue_id = get(TISSUE_OBO_MAP, workspace.tissue_target, nothing)
        if !isnothing(tissue_id)
            annotation["tissue"] = smart_lookup(tissue_id.uberon_id)
        end
    end

    # Get material term
    if isdefined(workspace, :material)
        mat = get(MATERIAL_DATABASE, workspace.material, nothing)
        if !isnothing(mat) && haskey(mat, :chebi_id)
            annotation["material"] = smart_lookup(mat.chebi_id)
        end
    end

    # Add metrics as additional data
    if isdefined(workspace, :metrics)
        annotation["metrics"] = Dict(
            "porosity" => workspace.metrics.porosity,
            "poreSize_um" => workspace.metrics.mean_pore_size_um,
            "interconnectivity" => workspace.metrics.interconnectivity,
            "elasticModulus_MPa" => workspace.metrics.elastic_modulus
        )
    end

    # Export in requested format
    if format == :jsonld
        export_jsonld(annotation, filepath)
    elseif format == :rdf
        terms = [v for v in values(annotation) if v isa OBOTerm]
        export_rdf(terms, filepath)
    else
        error("Unsupported export format: $format")
    end
end

# Reference to ScaffoldEditor types (will be available at runtime)
const TISSUE_OBO_MAP = Dict{Symbol,Any}()
const MATERIAL_DATABASE = Dict{String,Any}()

#=============================================================================
  STATISTICS
=============================================================================#

"""Statistics structure."""
struct OntologyStats
    tier1_hits::Int
    tier2_hits::Int
    tier3_hits::Int
    cache_hits::Int
    api_calls::Int
    api_errors::Int
    cache_size_mb::Float64
    cached_terms::Int
end

"""
    get_stats() -> OntologyStats

Get current ontology system statistics.
"""
function get_stats()
    cache_size = 0.0
    cached_terms = 0

    db = DB_CONNECTION[]
    if !isnothing(db)
        # Get cache file size
        cache_path = CONFIG[].cache_db_path
        if isfile(cache_path)
            cache_size = filesize(cache_path) / (1024 * 1024)
        end

        # Count cached terms
        result = SQLite.DBInterface.execute(db, "SELECT COUNT(*) as cnt FROM terms") |>
                 SQLite.Tables.rowtable
        if !isempty(result)
            cached_terms = first(result).cnt
        end
    end

    OntologyStats(
        STATS[]["tier1_hits"],
        STATS[]["tier2_hits"],
        STATS[]["tier3_hits"],
        STATS[]["cache_hits"],
        STATS[]["api_calls"],
        STATS[]["api_errors"],
        cache_size,
        cached_terms
    )
end

"""Pretty print stats."""
function Base.show(io::IO, stats::OntologyStats)
    total_lookups = stats.tier1_hits + stats.cache_hits + stats.tier3_hits
    hit_rate = total_lookups > 0 ? (stats.tier1_hits + stats.cache_hits) / total_lookups * 100 : 0.0

    println(io, "OntologyStats:")
    println(io, "  Tier 1 (hardcoded): $(stats.tier1_hits) hits")
    println(io, "  Cache hits: $(stats.cache_hits)")
    println(io, "  Tier 3 (API): $(stats.tier3_hits) hits")
    println(io, "  API calls: $(stats.api_calls) (errors: $(stats.api_errors))")
    println(io, "  Local hit rate: $(round(hit_rate, digits=1))%")
    println(io, "  Cache: $(stats.cached_terms) terms ($(round(stats.cache_size_mb, digits=2)) MB)")
end

end # module
