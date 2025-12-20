"""
Epistemic.jl - Demetrios-Style Epistemic Computing for Julia

Provides Knowledge types that carry metadata about confidence, provenance,
temporal context, and ontological bindings. Every scientific measurement
or computed value knows its own uncertainty and origin.

The Four Epistemic Dimensions (τ, ε, δ, Φ):
- τ (temporal): When was this known? What was the context?
- ε (epistemic): How confident are we? What's the evidence?
- δ (ontological): What ontologies does this map to?
- Φ (provenance): How was this derived? What's the chain?

This is the core of epistemic computing: values that know themselves.

References:
- Demetrios Language: github.com/chiuratto-AI/demetrios
- W3C PROV-O ontology for provenance
- Schema.org for ontological bindings
"""
module Epistemic

using Dates
using UUIDs
using Statistics

export Knowledge, knowledge
export Temporal, Confidence, OntologyBinding, Provenance
export τ, ε, δ, Φ  # Greek letter accessors
export with_confidence, with_provenance, with_temporal, with_binding
export propagate_confidence, combine_knowledge
export EpistemicValue, @epistemic
export ConfidenceLevel, HIGH, MEDIUM, LOW, UNCERTAIN

# ============================================================================
# CONFIDENCE LEVELS
# ============================================================================

"""
    ConfidenceLevel

Qualitative confidence levels for epistemic values.
"""
@enum ConfidenceLevel begin
    HIGH = 4       # Strong evidence, replicated, peer-reviewed
    MEDIUM = 3     # Good evidence, some uncertainty
    LOW = 2        # Limited evidence, high uncertainty
    UNCERTAIN = 1  # Unknown or untrusted
end

# ============================================================================
# TEMPORAL CONTEXT (τ)
# ============================================================================

"""
    Temporal

Temporal context for knowledge: when was it known, in what context?

# Fields
- `timestamp::DateTime`: When the value was measured/computed
- `valid_from::Union{DateTime, Nothing}`: Start of validity period
- `valid_until::Union{DateTime, Nothing}`: End of validity period
- `context::String`: Experimental or computational context
- `version::String`: Version of method/protocol used
"""
struct Temporal
    timestamp::DateTime
    valid_from::Union{DateTime, Nothing}
    valid_until::Union{DateTime, Nothing}
    context::String
    version::String

    function Temporal(;
        timestamp::DateTime=now(),
        valid_from::Union{DateTime, Nothing}=nothing,
        valid_until::Union{DateTime, Nothing}=nothing,
        context::String="",
        version::String="1.0"
    )
        new(timestamp, valid_from, valid_until, context, version)
    end
end

# Simple constructor
Temporal(ts::DateTime) = Temporal(timestamp=ts)
Temporal() = Temporal(timestamp=now())

function Base.show(io::IO, t::Temporal)
    print(io, "τ[", Dates.format(t.timestamp, "yyyy-mm-dd HH:MM"))
    if !isempty(t.context)
        print(io, ", ", t.context)
    end
    print(io, "]")
end

"""
    is_valid(t::Temporal, at::DateTime=now()) -> Bool

Check if the temporal context is still valid.
"""
function is_valid(t::Temporal, at::DateTime=now())
    from_ok = isnothing(t.valid_from) || at >= t.valid_from
    until_ok = isnothing(t.valid_until) || at <= t.valid_until
    return from_ok && until_ok
end

# ============================================================================
# CONFIDENCE/EPISTEMIC STATUS (ε)
# ============================================================================

"""
    Confidence

Epistemic status: how confident are we in this value?

# Fields
- `level::ConfidenceLevel`: Qualitative confidence level
- `value::Float64`: Quantitative confidence (0.0 to 1.0)
- `method::String`: Method used to determine confidence
- `evidence_count::Int`: Number of supporting evidence sources
- `uncertainty::Float64`: Standard error or uncertainty estimate
- `sources::Vector{String}`: References/sources for confidence
"""
struct Confidence
    level::ConfidenceLevel
    value::Float64
    method::String
    evidence_count::Int
    uncertainty::Float64
    sources::Vector{String}

    function Confidence(;
        level::ConfidenceLevel=MEDIUM,
        value::Float64=0.5,
        method::String="default",
        evidence_count::Int=0,
        uncertainty::Float64=0.0,
        sources::Vector{String}=String[]
    )
        @assert 0.0 <= value <= 1.0 "Confidence value must be between 0 and 1"
        new(level, value, method, evidence_count, uncertainty, sources)
    end
end

# Convenience constructors
Confidence(level::ConfidenceLevel) = Confidence(level=level, value=Float64(Int(level))/4.0)
Confidence(value::Float64) = Confidence(value=value, level=value_to_level(value))

function value_to_level(v::Float64)
    if v >= 0.85
        return HIGH
    elseif v >= 0.60
        return MEDIUM
    elseif v >= 0.35
        return LOW
    else
        return UNCERTAIN
    end
end

function Base.show(io::IO, c::Confidence)
    print(io, "ε[", c.level, " (", round(c.value * 100, digits=1), "%)")
    if c.uncertainty > 0
        print(io, " ±", round(c.uncertainty * 100, digits=1), "%")
    end
    print(io, "]")
end

"""
    combine_confidence(c1::Confidence, c2::Confidence) -> Confidence

Combine two confidence values (e.g., for derived values).
Uses minimum confidence principle for conservative estimation.
"""
function combine_confidence(c1::Confidence, c2::Confidence)
    # Combined confidence is minimum (conservative)
    new_value = min(c1.value, c2.value)

    # Propagate uncertainty (quadrature)
    new_uncertainty = sqrt(c1.uncertainty^2 + c2.uncertainty^2)

    # Combine evidence
    new_count = c1.evidence_count + c2.evidence_count
    new_sources = unique(vcat(c1.sources, c2.sources))

    Confidence(
        level=value_to_level(new_value),
        value=new_value,
        method="combined",
        evidence_count=new_count,
        uncertainty=new_uncertainty,
        sources=new_sources
    )
end

# ============================================================================
# ONTOLOGY BINDING (δ)
# ============================================================================

"""
    OntologyBinding

Mapping to standard ontologies for semantic interoperability.

# Fields
- `primary::String`: Primary ontology term (e.g., "obo:UBERON_0002481")
- `label::String`: Human-readable label
- `ontology::String`: Source ontology (e.g., "UBERON", "Schema.org")
- `uri::String`: Full URI if available
- `mappings::Dict{String, String}`: Mappings to other ontologies
"""
struct OntologyBinding
    primary::String
    label::String
    ontology::String
    uri::String
    mappings::Dict{String, String}

    function OntologyBinding(;
        primary::String="",
        label::String="",
        ontology::String="",
        uri::String="",
        mappings::Dict{String, String}=Dict{String, String}()
    )
        new(primary, label, ontology, uri, mappings)
    end
end

# Simple constructors
OntologyBinding(primary::String) = OntologyBinding(primary=primary)
OntologyBinding(primary::String, label::String) = OntologyBinding(primary=primary, label=label)

function Base.show(io::IO, b::OntologyBinding)
    if isempty(b.primary)
        print(io, "δ[unbound]")
    else
        print(io, "δ[", b.primary)
        if !isempty(b.label)
            print(io, ": ", b.label)
        end
        print(io, "]")
    end
end

# Common bindings for scaffold analysis
const POROSITY_BINDING = OntologyBinding(
    primary="PATO:0000973",
    label="porosity",
    ontology="PATO",
    uri="http://purl.obolibrary.org/obo/PATO_0000973"
)

const PORE_SIZE_BINDING = OntologyBinding(
    primary="PATO:0000117",
    label="size",
    ontology="PATO",
    mappings=Dict("Schema.org" => "https://schema.org/size")
)

const SCAFFOLD_BINDING = OntologyBinding(
    primary="UBERON:0000479",
    label="tissue scaffold",
    ontology="UBERON"
)

# ============================================================================
# PROVENANCE (Φ)
# ============================================================================

"""
    Provenance

Derivation chain: how was this value computed/obtained?

# Fields
- `id::UUID`: Unique identifier for this provenance record
- `source::String`: Original data source or measurement device
- `agent::String`: Who/what computed this (person, algorithm, instrument)
- `activity::String`: What process was used
- `generated_at::DateTime`: When this was generated
- `derived_from::Vector{UUID}`: Parent provenance IDs
- `transformations::Vector{String}`: List of transformations applied
- `parameters::Dict{String, Any}`: Parameters used in derivation
"""
struct Provenance
    id::UUID
    source::String
    agent::String
    activity::String
    generated_at::DateTime
    derived_from::Vector{UUID}
    transformations::Vector{String}
    parameters::Dict{String, Any}

    function Provenance(;
        id::UUID=uuid4(),
        source::String="",
        agent::String="DarwinScaffoldStudio",
        activity::String="computation",
        generated_at::DateTime=now(),
        derived_from::Vector{UUID}=UUID[],
        transformations::Vector{String}=String[],
        parameters::Dict{String, Any}=Dict{String, Any}()
    )
        new(id, source, agent, activity, generated_at, derived_from, transformations, parameters)
    end
end

# Simple constructors
Provenance(source::String) = Provenance(source=source)
Provenance(source::String, agent::String) = Provenance(source=source, agent=agent)

function Base.show(io::IO, p::Provenance)
    print(io, "Φ[", string(p.id)[1:8], "...")
    if !isempty(p.source)
        print(io, " from ", p.source)
    end
    if !isempty(p.derived_from)
        print(io, " ← ", length(p.derived_from), " parents")
    end
    print(io, "]")
end

"""
    derive_provenance(parents::Vector{Provenance}, activity::String; kwargs...) -> Provenance

Create provenance for a derived value.
"""
function derive_provenance(parents::Vector{Provenance}, activity::String;
                           agent::String="computation",
                           transformation::String="",
                           parameters::Dict{String, Any}=Dict{String, Any}())
    parent_ids = [p.id for p in parents]
    transformations = isempty(transformation) ? String[] : [transformation]

    Provenance(
        source="derived",
        agent=agent,
        activity=activity,
        derived_from=parent_ids,
        transformations=transformations,
        parameters=parameters
    )
end

# ============================================================================
# KNOWLEDGE TYPE
# ============================================================================

"""
    Knowledge{T}

A value that knows itself: carries the value along with epistemic metadata.

Every scientific measurement or computed result should be wrapped in Knowledge
to track its confidence, origin, temporal validity, and semantic meaning.

# Type Parameters
- `T`: The underlying value type

# Fields
- `value::T`: The actual value
- `τ::Temporal`: Temporal context
- `ε::Confidence`: Epistemic status / confidence
- `δ::OntologyBinding`: Ontological binding
- `Φ::Provenance`: Derivation provenance

# Example
```julia
porosity = Knowledge(
    0.92,
    τ=Temporal(context="MicroCT scan"),
    ε=Confidence(level=HIGH, uncertainty=0.02),
    δ=POROSITY_BINDING,
    Φ=Provenance(source="Bruker SkyScan 1275")
)
```
"""
struct Knowledge{T}
    value::T
    τ::Temporal
    ε::Confidence
    δ::OntologyBinding
    Φ::Provenance

    function Knowledge{T}(
        value::T;
        τ::Temporal=Temporal(),
        ε::Confidence=Confidence(),
        δ::OntologyBinding=OntologyBinding(),
        Φ::Provenance=Provenance()
    ) where T
        new{T}(value, τ, ε, δ, Φ)
    end
end

# Constructor without type parameter
function Knowledge(value::T; kwargs...) where T
    Knowledge{T}(value; kwargs...)
end

# Convenience constructor with just confidence
function knowledge(value::T, confidence::Float64; kwargs...) where T
    Knowledge{T}(value; ε=Confidence(confidence), kwargs...)
end

# Accessors using Greek letters
τ(k::Knowledge) = k.τ
ε(k::Knowledge) = k.ε
δ(k::Knowledge) = k.δ
Φ(k::Knowledge) = k.Φ

# Get the underlying value
Base.getindex(k::Knowledge) = k.value
unwrap(k::Knowledge) = k.value

function Base.show(io::IO, k::Knowledge{T}) where T
    print(io, "Knowledge{", T, "}(", k.value, ")\n")
    print(io, "  ", k.τ, "\n")
    print(io, "  ", k.ε, "\n")
    print(io, "  ", k.δ, "\n")
    print(io, "  ", k.Φ)
end

function Base.show(io::IO, ::MIME"text/plain", k::Knowledge)
    show(io, k)
end

# Compact display
function compact_show(io::IO, k::Knowledge)
    print(io, k.value, " @ ", round(k.ε.value * 100, digits=0), "%")
end

# ============================================================================
# KNOWLEDGE BUILDERS
# ============================================================================

"""
    with_confidence(value, confidence; kwargs...) -> Knowledge

Create knowledge with specified confidence.
"""
function with_confidence(value::T, confidence::Float64; kwargs...) where T
    Knowledge{T}(value; ε=Confidence(confidence), kwargs...)
end

function with_confidence(value::T, level::ConfidenceLevel; kwargs...) where T
    Knowledge{T}(value; ε=Confidence(level), kwargs...)
end

"""
    with_provenance(value, source::String; kwargs...) -> Knowledge

Create knowledge with provenance source.
"""
function with_provenance(value::T, source::String; kwargs...) where T
    Knowledge{T}(value; Φ=Provenance(source), kwargs...)
end

"""
    with_temporal(value, context::String; kwargs...) -> Knowledge

Create knowledge with temporal context.
"""
function with_temporal(value::T, context::String; kwargs...) where T
    Knowledge{T}(value; τ=Temporal(context=context), kwargs...)
end

"""
    with_binding(value, binding::OntologyBinding; kwargs...) -> Knowledge

Create knowledge with ontology binding.
"""
function with_binding(value::T, binding::OntologyBinding; kwargs...) where T
    Knowledge{T}(value; δ=binding, kwargs...)
end

# ============================================================================
# KNOWLEDGE ARITHMETIC
# ============================================================================

# Arithmetic operations that propagate epistemic metadata

function Base.:+(k1::Knowledge{T}, k2::Knowledge{T}) where T <: Number
    new_value = k1.value + k2.value
    new_conf = combine_confidence(k1.ε, k2.ε)
    new_prov = derive_provenance([k1.Φ, k2.Φ], "addition")

    Knowledge{T}(new_value;
        τ=k1.τ,  # Use first temporal context
        ε=new_conf,
        δ=k1.δ,  # Use first binding
        Φ=new_prov
    )
end

function Base.:-(k1::Knowledge{T}, k2::Knowledge{T}) where T <: Number
    new_value = k1.value - k2.value
    new_conf = combine_confidence(k1.ε, k2.ε)
    new_prov = derive_provenance([k1.Φ, k2.Φ], "subtraction")

    Knowledge{T}(new_value; τ=k1.τ, ε=new_conf, δ=k1.δ, Φ=new_prov)
end

function Base.:*(k1::Knowledge{T1}, k2::Knowledge{T2}) where {T1 <: Number, T2 <: Number}
    T = promote_type(T1, T2)
    new_value = k1.value * k2.value
    new_conf = combine_confidence(k1.ε, k2.ε)
    new_prov = derive_provenance([k1.Φ, k2.Φ], "multiplication")

    Knowledge{T}(new_value; τ=k1.τ, ε=new_conf, Φ=new_prov)
end

function Base.:/(k1::Knowledge{T1}, k2::Knowledge{T2}) where {T1 <: Number, T2 <: Number}
    T = promote_type(T1, T2)
    new_value = k1.value / k2.value
    new_conf = combine_confidence(k1.ε, k2.ε)
    new_prov = derive_provenance([k1.Φ, k2.Φ], "division")

    Knowledge{T}(new_value; τ=k1.τ, ε=new_conf, Φ=new_prov)
end

# Scalar operations
Base.:*(k::Knowledge{T}, s::Number) where T <: Number =
    Knowledge{T}(k.value * s; τ=k.τ, ε=k.ε, δ=k.δ, Φ=k.Φ)
Base.:*(s::Number, k::Knowledge) = k * s
Base.:/(k::Knowledge{T}, s::Number) where T <: Number =
    Knowledge{T}(k.value / s; τ=k.τ, ε=k.ε, δ=k.δ, Φ=k.Φ)

# ============================================================================
# KNOWLEDGE AGGREGATION
# ============================================================================

"""
    combine_knowledge(knowledges::Vector{Knowledge{T}}, agg=mean) -> Knowledge{T}

Combine multiple knowledge values into one.
Uses confidence-weighted aggregation.
"""
function combine_knowledge(knowledges::Vector{Knowledge{T}};
                           aggregation::Function=mean) where T <: Number
    if isempty(knowledges)
        error("Cannot combine empty knowledge vector")
    end

    # Confidence-weighted aggregation
    weights = [k.ε.value for k in knowledges]
    total_weight = sum(weights)

    if total_weight > 0
        weighted_values = [k.value * k.ε.value for k in knowledges]
        new_value = sum(weighted_values) / total_weight
    else
        new_value = aggregation([k.value for k in knowledges])
    end

    # Combined confidence (geometric mean)
    new_conf_value = exp(mean(log.(max.(weights, 1e-10))))

    # Combined provenance
    all_prov = [k.Φ for k in knowledges]
    new_prov = derive_provenance(all_prov, "aggregation",
        parameters=Dict{String, Any}("n" => length(knowledges)))

    # Combined uncertainty
    new_uncertainty = std([k.value for k in knowledges]) / sqrt(length(knowledges))

    new_conf = Confidence(
        value=new_conf_value,
        method="combined",
        evidence_count=sum(k.ε.evidence_count for k in knowledges),
        uncertainty=new_uncertainty
    )

    Knowledge{T}(new_value; τ=knowledges[1].τ, ε=new_conf, δ=knowledges[1].δ, Φ=new_prov)
end

# ============================================================================
# EPISTEMIC VALUE MACRO
# ============================================================================

"""
    @epistemic value confidence source

Create an epistemic value with inline syntax.

# Example
```julia
porosity = @epistemic 0.92 0.95 "MicroCT"
```
"""
macro epistemic(value, confidence, source)
    quote
        Knowledge($(esc(value));
            ε=Confidence($(esc(confidence))),
            Φ=Provenance($(esc(source)))
        )
    end
end

# ============================================================================
# TYPE ALIASES FOR COMMON USE
# ============================================================================

"""
    EpistemicValue{T}

Alias for Knowledge{T} - a value with epistemic metadata.
"""
const EpistemicValue{T} = Knowledge{T}

"""
    EpistemicFloat

Common type for float values with epistemic tracking.
"""
const EpistemicFloat = Knowledge{Float64}

"""
    EpistemicInt

Common type for integer values with epistemic tracking.
"""
const EpistemicInt = Knowledge{Int}

end # module
