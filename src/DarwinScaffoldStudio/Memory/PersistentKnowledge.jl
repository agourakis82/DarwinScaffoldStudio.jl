"""
    PersistentKnowledge

Persistent knowledge storage for Darwin Scaffold Studio.

Implements:
- SQLite-backed knowledge base
- Vector embeddings for semantic search (ChromaDB-style)
- Session memory with context windows
- Long-term scaffold design history

# Architecture
- KnowledgeStore: Main interface
- EmbeddingIndex: Vector similarity search
- SessionMemory: Conversation context
- DesignHistory: Scaffold versioning

# Author: Dr. Demetrios Agourakis
# Darwin Scaffold Studio v0.5.0
"""
module PersistentKnowledge

using SQLite
using JSON3
using Dates
using UUIDs
using LinearAlgebra
using Statistics

export KnowledgeStore, init_knowledge_store!, close_knowledge_store!
export store_scaffold!, retrieve_scaffold, search_scaffolds
export store_embedding!, search_similar, EmbeddingRecord
export SessionMemory, add_message!, get_context, clear_session!
export DesignHistory, save_design_version!, get_design_history, restore_version
export export_knowledge_base, import_knowledge_base

#=============================================================================
  CONFIGURATION
=============================================================================#

const DEFAULT_DB_PATH = joinpath(homedir(), ".darwin_scaffold_studio", "knowledge.db")
const EMBEDDING_DIM = 384  # Sentence-transformers default

#=============================================================================
  KNOWLEDGE STORE
=============================================================================#

"""
    KnowledgeStore

Main persistent storage interface.
"""
mutable struct KnowledgeStore
    db::SQLite.DB
    db_path::String
    embedding_cache::Dict{String, Vector{Float32}}
    is_open::Bool

    function KnowledgeStore(db_path::String=DEFAULT_DB_PATH)
        # Ensure directory exists
        mkpath(dirname(db_path))

        db = SQLite.DB(db_path)
        store = new(db, db_path, Dict{String, Vector{Float32}}(), true)
        init_schema!(store)
        return store
    end
end

# Global instance
const STORE = Ref{Union{KnowledgeStore, Nothing}}(nothing)

"""
    init_knowledge_store!(; db_path=DEFAULT_DB_PATH)

Initialize the global knowledge store.
"""
function init_knowledge_store!(; db_path::String=DEFAULT_DB_PATH)
    if !isnothing(STORE[]) && STORE[].is_open
        close_knowledge_store!()
    end
    STORE[] = KnowledgeStore(db_path)
    @info "Knowledge store initialized" path=db_path
    return STORE[]
end

"""
    close_knowledge_store!()

Close the knowledge store connection.
"""
function close_knowledge_store!()
    if !isnothing(STORE[]) && STORE[].is_open
        SQLite.close(STORE[].db)
        STORE[].is_open = false
        @info "Knowledge store closed"
    end
end

"""
    get_store()

Get the current knowledge store, initializing if needed.
"""
function get_store()
    if isnothing(STORE[]) || !STORE[].is_open
        init_knowledge_store!()
    end
    return STORE[]
end

"""Initialize database schema."""
function init_schema!(store::KnowledgeStore)
    # Scaffolds table
    SQLite.execute(store.db, """
        CREATE TABLE IF NOT EXISTS scaffolds (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            porosity REAL,
            pore_size_um REAL,
            interconnectivity REAL,
            material TEXT,
            target_tissue TEXT,
            volume_data BLOB,
            metrics_json TEXT,
            created_at TEXT,
            updated_at TEXT,
            tags TEXT
        )
    """)

    # Embeddings table for vector search
    SQLite.execute(store.db, """
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            source_type TEXT,
            source_id TEXT,
            content TEXT,
            embedding BLOB,
            created_at TEXT
        )
    """)

    # Session memory table
    SQLite.execute(store.db, """
        CREATE TABLE IF NOT EXISTS session_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            metadata TEXT
        )
    """)

    # Design history table
    SQLite.execute(store.db, """
        CREATE TABLE IF NOT EXISTS design_history (
            id TEXT PRIMARY KEY,
            scaffold_id TEXT,
            version INTEGER,
            changes TEXT,
            volume_data BLOB,
            metrics_json TEXT,
            created_at TEXT,
            FOREIGN KEY (scaffold_id) REFERENCES scaffolds(id)
        )
    """)

    # Create indexes
    SQLite.execute(store.db, "CREATE INDEX IF NOT EXISTS idx_scaffolds_material ON scaffolds(material)")
    SQLite.execute(store.db, "CREATE INDEX IF NOT EXISTS idx_scaffolds_tissue ON scaffolds(target_tissue)")
    SQLite.execute(store.db, "CREATE INDEX IF NOT EXISTS idx_embeddings_source ON embeddings(source_type, source_id)")
    SQLite.execute(store.db, "CREATE INDEX IF NOT EXISTS idx_session_id ON session_memory(session_id)")
    SQLite.execute(store.db, "CREATE INDEX IF NOT EXISTS idx_design_scaffold ON design_history(scaffold_id)")
end

#=============================================================================
  SCAFFOLD STORAGE
=============================================================================#

"""
    store_scaffold!(; kwargs...) -> String

Store a scaffold design in the knowledge base.

Returns the scaffold ID.
"""
function store_scaffold!(;
    name::String,
    description::String="",
    porosity::Float64=0.0,
    pore_size_um::Float64=0.0,
    interconnectivity::Float64=0.0,
    material::String="",
    target_tissue::String="",
    volume_data::Union{Array{Bool,3}, Nothing}=nothing,
    metrics::Union{Dict, Nothing}=nothing,
    tags::Vector{String}=String[]
)
    store = get_store()

    id = string(uuid4())
    now_str = string(now())

    # Serialize volume data if provided
    volume_blob = isnothing(volume_data) ? nothing : serialize_volume(volume_data)

    # Serialize metrics
    metrics_json = isnothing(metrics) ? "{}" : JSON3.write(metrics)

    # Tags as JSON array
    tags_json = JSON3.write(tags)

    SQLite.execute(store.db, """
        INSERT INTO scaffolds
        (id, name, description, porosity, pore_size_um, interconnectivity,
         material, target_tissue, volume_data, metrics_json, created_at, updated_at, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [id, name, description, porosity, pore_size_um, interconnectivity,
          material, target_tissue, volume_blob, metrics_json, now_str, now_str, tags_json])

    # Generate and store embedding for semantic search
    content = "$name $description $material $target_tissue $(join(tags, " "))"
    embedding = generate_embedding(content)
    store_embedding!(source_type="scaffold", source_id=id, content=content, embedding=embedding)

    @info "Scaffold stored" id=id name=name
    return id
end

"""
    retrieve_scaffold(id::String) -> Union{Dict, Nothing}

Retrieve a scaffold by ID.
"""
function retrieve_scaffold(id::String)
    store = get_store()

    result = SQLite.DBInterface.execute(store.db,
        "SELECT * FROM scaffolds WHERE id = ?", [id])

    rows = collect(result)
    if isempty(rows)
        return nothing
    end

    row = rows[1]
    return Dict(
        "id" => row.id,
        "name" => row.name,
        "description" => row.description,
        "porosity" => row.porosity,
        "pore_size_um" => row.pore_size_um,
        "interconnectivity" => row.interconnectivity,
        "material" => row.material,
        "target_tissue" => row.target_tissue,
        "volume_data" => isnothing(row.volume_data) ? nothing : deserialize_volume(row.volume_data),
        "metrics" => JSON3.read(row.metrics_json, Dict),
        "created_at" => row.created_at,
        "updated_at" => row.updated_at,
        "tags" => JSON3.read(row.tags, Vector{String})
    )
end

"""
    search_scaffolds(; kwargs...) -> Vector{Dict}

Search scaffolds by various criteria.
"""
function search_scaffolds(;
    material::Union{String, Nothing}=nothing,
    target_tissue::Union{String, Nothing}=nothing,
    min_porosity::Union{Float64, Nothing}=nothing,
    max_porosity::Union{Float64, Nothing}=nothing,
    tags::Union{Vector{String}, Nothing}=nothing,
    limit::Int=100
)
    store = get_store()

    conditions = String[]
    params = Any[]

    if !isnothing(material)
        push!(conditions, "material = ?")
        push!(params, material)
    end

    if !isnothing(target_tissue)
        push!(conditions, "target_tissue = ?")
        push!(params, target_tissue)
    end

    if !isnothing(min_porosity)
        push!(conditions, "porosity >= ?")
        push!(params, min_porosity)
    end

    if !isnothing(max_porosity)
        push!(conditions, "porosity <= ?")
        push!(params, max_porosity)
    end

    where_clause = isempty(conditions) ? "" : "WHERE " * join(conditions, " AND ")

    query = "SELECT id, name, material, target_tissue, porosity, pore_size_um, created_at
             FROM scaffolds $where_clause ORDER BY created_at DESC LIMIT ?"
    push!(params, limit)

    result = SQLite.DBInterface.execute(store.db, query, params)

    return [Dict(
        "id" => row.id,
        "name" => row.name,
        "material" => row.material,
        "target_tissue" => row.target_tissue,
        "porosity" => row.porosity,
        "pore_size_um" => row.pore_size_um,
        "created_at" => row.created_at
    ) for row in result]
end

#=============================================================================
  VECTOR EMBEDDINGS
=============================================================================#

"""
    EmbeddingRecord

A stored embedding with metadata.
"""
struct EmbeddingRecord
    id::String
    source_type::String
    source_id::String
    content::String
    embedding::Vector{Float32}
end

"""
    generate_embedding(text::String) -> Vector{Float32}

Generate embedding vector for text.
Uses simple TF-IDF style embedding (production would use sentence-transformers).
"""
function generate_embedding(text::String)
    # Simple character n-gram embedding (placeholder for real embedding model)
    # Production: Use Transformers.jl or PyCall to sentence-transformers

    text_lower = lowercase(text)
    embedding = zeros(Float32, EMBEDDING_DIM)

    # Character trigram hashing
    for i in 1:length(text_lower)-2
        trigram = text_lower[i:i+2]
        hash_idx = mod(hash(trigram), EMBEDDING_DIM) + 1
        embedding[hash_idx] += 1.0f0
    end

    # Word-level features
    words = split(text_lower)
    for word in words
        hash_idx = mod(hash(word), EMBEDDING_DIM) + 1
        embedding[hash_idx] += 2.0f0
    end

    # L2 normalize
    norm_val = norm(embedding)
    if norm_val > 0
        embedding ./= norm_val
    end

    return embedding
end

"""
    store_embedding!(; source_type, source_id, content, embedding)

Store an embedding in the database.
"""
function store_embedding!(;
    source_type::String,
    source_id::String,
    content::String,
    embedding::Vector{Float32}
)
    store = get_store()

    id = string(uuid4())
    embedding_blob = reinterpret(UInt8, embedding)

    SQLite.execute(store.db, """
        INSERT OR REPLACE INTO embeddings (id, source_type, source_id, content, embedding, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [id, source_type, source_id, content, embedding_blob, string(now())])

    # Cache in memory
    store.embedding_cache["$source_type:$source_id"] = embedding

    return id
end

"""
    search_similar(query::String; top_k=10, source_type=nothing) -> Vector{Dict}

Semantic similarity search using vector embeddings.
"""
function search_similar(query::String; top_k::Int=10, source_type::Union{String, Nothing}=nothing)
    store = get_store()

    # Generate query embedding
    query_embedding = generate_embedding(query)

    # Get all embeddings (in production, use ANN index like FAISS)
    type_filter = isnothing(source_type) ? "" : "WHERE source_type = '$source_type'"
    result = SQLite.DBInterface.execute(store.db,
        "SELECT id, source_type, source_id, content, embedding FROM embeddings $type_filter")

    # Compute similarities
    similarities = Tuple{Float64, String, String, String, String}[]

    for row in result
        embedding_blob = row.embedding
        if isnothing(embedding_blob) || isempty(embedding_blob)
            continue
        end

        embedding = reinterpret(Float32, Vector{UInt8}(embedding_blob))

        # Cosine similarity
        sim = dot(query_embedding, embedding)

        push!(similarities, (sim, row.id, row.source_type, row.source_id, row.content))
    end

    # Sort by similarity (descending)
    sort!(similarities, by=x->x[1], rev=true)

    # Return top_k
    return [Dict(
        "similarity" => sim,
        "id" => id,
        "source_type" => src_type,
        "source_id" => src_id,
        "content" => content
    ) for (sim, id, src_type, src_id, content) in similarities[1:min(top_k, length(similarities))]]
end

#=============================================================================
  SESSION MEMORY
=============================================================================#

"""
    SessionMemory

Manages conversation context for agent interactions.
"""
mutable struct SessionMemory
    session_id::String
    max_messages::Int

    function SessionMemory(; session_id::String=string(uuid4()), max_messages::Int=50)
        new(session_id, max_messages)
    end
end

"""
    add_message!(memory::SessionMemory, role::String, content::String; metadata=nothing)

Add a message to session memory.
"""
function add_message!(memory::SessionMemory, role::String, content::String;
                      metadata::Union{Dict, Nothing}=nothing)
    store = get_store()

    metadata_json = isnothing(metadata) ? "{}" : JSON3.write(metadata)

    SQLite.execute(store.db, """
        INSERT INTO session_memory (session_id, role, content, timestamp, metadata)
        VALUES (?, ?, ?, ?, ?)
    """, [memory.session_id, role, content, string(now()), metadata_json])

    # Trim old messages if exceeding max
    SQLite.execute(store.db, """
        DELETE FROM session_memory
        WHERE session_id = ? AND id NOT IN (
            SELECT id FROM session_memory WHERE session_id = ?
            ORDER BY timestamp DESC LIMIT ?
        )
    """, [memory.session_id, memory.session_id, memory.max_messages])
end

"""
    get_context(memory::SessionMemory; last_n=10) -> Vector{Dict}

Get recent conversation context.
"""
function get_context(memory::SessionMemory; last_n::Int=10)
    store = get_store()

    result = SQLite.DBInterface.execute(store.db, """
        SELECT role, content, timestamp, metadata
        FROM session_memory
        WHERE session_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, [memory.session_id, last_n])

    messages = [Dict(
        "role" => row.role,
        "content" => row.content,
        "timestamp" => row.timestamp,
        "metadata" => JSON3.read(row.metadata, Dict)
    ) for row in result]

    return reverse(messages)  # Chronological order
end

"""
    clear_session!(memory::SessionMemory)

Clear all messages for this session.
"""
function clear_session!(memory::SessionMemory)
    store = get_store()
    SQLite.execute(store.db, "DELETE FROM session_memory WHERE session_id = ?", [memory.session_id])
end

#=============================================================================
  DESIGN HISTORY
=============================================================================#

"""
    DesignHistory

Version control for scaffold designs.
"""
struct DesignHistory
    scaffold_id::String
end

"""
    save_design_version!(scaffold_id::String, changes::String, volume_data, metrics) -> String

Save a new version of a scaffold design.
"""
function save_design_version!(scaffold_id::String, changes::String;
                              volume_data::Union{Array{Bool,3}, Nothing}=nothing,
                              metrics::Union{Dict, Nothing}=nothing)
    store = get_store()

    # Get current max version
    result = SQLite.DBInterface.execute(store.db,
        "SELECT MAX(version) as max_ver FROM design_history WHERE scaffold_id = ?", [scaffold_id])
    rows = collect(result)
    current_version = isnothing(rows[1].max_ver) ? 0 : rows[1].max_ver
    new_version = current_version + 1

    id = string(uuid4())
    volume_blob = isnothing(volume_data) ? nothing : serialize_volume(volume_data)
    metrics_json = isnothing(metrics) ? "{}" : JSON3.write(metrics)

    SQLite.execute(store.db, """
        INSERT INTO design_history (id, scaffold_id, version, changes, volume_data, metrics_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [id, scaffold_id, new_version, changes, volume_blob, metrics_json, string(now())])

    @info "Design version saved" scaffold_id=scaffold_id version=new_version
    return id
end

"""
    get_design_history(scaffold_id::String) -> Vector{Dict}

Get version history for a scaffold.
"""
function get_design_history(scaffold_id::String)
    store = get_store()

    result = SQLite.DBInterface.execute(store.db, """
        SELECT id, version, changes, metrics_json, created_at
        FROM design_history
        WHERE scaffold_id = ?
        ORDER BY version DESC
    """, [scaffold_id])

    return [Dict(
        "id" => row.id,
        "version" => row.version,
        "changes" => row.changes,
        "metrics" => JSON3.read(row.metrics_json, Dict),
        "created_at" => row.created_at
    ) for row in result]
end

"""
    restore_version(history_id::String) -> Union{Array{Bool,3}, Nothing}

Restore a specific version's volume data.
"""
function restore_version(history_id::String)
    store = get_store()

    result = SQLite.DBInterface.execute(store.db,
        "SELECT volume_data FROM design_history WHERE id = ?", [history_id])

    rows = collect(result)
    if isempty(rows) || isnothing(rows[1].volume_data)
        return nothing
    end

    return deserialize_volume(rows[1].volume_data)
end

#=============================================================================
  IMPORT/EXPORT
=============================================================================#

"""
    export_knowledge_base(path::String)

Export entire knowledge base to JSON file.
"""
function export_knowledge_base(path::String)
    store = get_store()

    # Export scaffolds
    scaffolds = SQLite.DBInterface.execute(store.db, "SELECT * FROM scaffolds")
    scaffolds_data = [Dict(pairs(row)) for row in scaffolds]

    # Export embeddings metadata (not vectors)
    embeddings = SQLite.DBInterface.execute(store.db,
        "SELECT id, source_type, source_id, content, created_at FROM embeddings")
    embeddings_data = [Dict(pairs(row)) for row in embeddings]

    export_data = Dict(
        "version" => "0.5.0",
        "exported_at" => string(now()),
        "scaffolds" => scaffolds_data,
        "embeddings_count" => length(embeddings_data),
        "embeddings_metadata" => embeddings_data
    )

    open(path, "w") do f
        JSON3.write(f, export_data)
    end

    @info "Knowledge base exported" path=path scaffolds=length(scaffolds_data)
end

"""
    import_knowledge_base(path::String)

Import knowledge base from JSON file.
"""
function import_knowledge_base(path::String)
    data = open(path, "r") do f
        JSON3.read(f, Dict)
    end

    store = get_store()

    imported = 0
    for scaffold in data["scaffolds"]
        try
            SQLite.execute(store.db, """
                INSERT OR IGNORE INTO scaffolds
                (id, name, description, porosity, pore_size_um, interconnectivity,
                 material, target_tissue, metrics_json, created_at, updated_at, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [scaffold["id"], scaffold["name"], get(scaffold, "description", ""),
                  get(scaffold, "porosity", 0), get(scaffold, "pore_size_um", 0),
                  get(scaffold, "interconnectivity", 0), get(scaffold, "material", ""),
                  get(scaffold, "target_tissue", ""), get(scaffold, "metrics_json", "{}"),
                  scaffold["created_at"], scaffold["updated_at"], get(scaffold, "tags", "[]")])
            imported += 1
        catch e
            @warn "Failed to import scaffold" id=scaffold["id"] error=e
        end
    end

    @info "Knowledge base imported" path=path imported=imported
end

#=============================================================================
  HELPERS
=============================================================================#

"""Serialize 3D boolean volume to compressed bytes."""
function serialize_volume(volume::Array{Bool,3})
    dims = size(volume)
    # Pack bools into bytes
    flat = vec(volume)
    n_bytes = cld(length(flat), 8)
    bytes = zeros(UInt8, n_bytes + 12)  # 12 bytes for dimensions

    # Store dimensions (3 x Int32)
    bytes[1:4] = reinterpret(UInt8, [Int32(dims[1])])
    bytes[5:8] = reinterpret(UInt8, [Int32(dims[2])])
    bytes[9:12] = reinterpret(UInt8, [Int32(dims[3])])

    # Pack bits
    for (i, b) in enumerate(flat)
        if b
            byte_idx = 12 + div(i - 1, 8) + 1
            bit_idx = mod(i - 1, 8)
            bytes[byte_idx] |= (0x01 << bit_idx)
        end
    end

    return bytes
end

"""Deserialize bytes to 3D boolean volume."""
function deserialize_volume(bytes::Vector{UInt8})
    # Read dimensions
    dims = (
        reinterpret(Int32, bytes[1:4])[1],
        reinterpret(Int32, bytes[5:8])[1],
        reinterpret(Int32, bytes[9:12])[1]
    )

    total = dims[1] * dims[2] * dims[3]
    flat = falses(total)

    # Unpack bits
    for i in 1:total
        byte_idx = 12 + div(i - 1, 8) + 1
        bit_idx = mod(i - 1, 8)
        if byte_idx <= length(bytes)
            flat[i] = (bytes[byte_idx] >> bit_idx) & 0x01 == 0x01
        end
    end

    return reshape(flat, dims)
end

end # module
