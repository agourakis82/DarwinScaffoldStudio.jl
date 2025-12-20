"""
ReproducibilityTracker.jl - Scientific Reproducibility Infrastructure

Provides comprehensive experiment tracking, seed management, and result caching
for dissertation-quality reproducible research.

SOTA 2024-2025 Features:
- Deterministic seed management with purpose-based forking
- Experiment tracking with parameter/metric/artifact logging
- Configuration versioning with content hashing
- Model checkpointing and recovery
- Results caching for expensive computations

References:
- MLflow (Zaharia et al. 2018) - Experiment tracking patterns
- Weights & Biases - Modern ML experiment management
- Sacred (Greff et al. 2017) - Reproducible experiment framework
"""
module ReproducibilityTracker

using Dates
using Random
using SHA
using JSON3
using Serialization  # Standard library for binary serialization
using UUIDs

export SeedManager, set_global_seed, fork_seed, get_seed_log
export ExperimentConfig, validate_config, config_hash
export ExperimentRun, ExperimentTracker
export create_experiment, log_params, log_metrics, log_artifact
export save_checkpoint, load_checkpoint, finish_experiment
export cached_result, clear_cache

# ============================================================================
# SEED MANAGEMENT
# ============================================================================

"""
    SeedManager

Centralized seed management for reproducible experiments.

Provides deterministic seed forking based on purpose strings,
ensuring different parts of the experiment get different but
reproducible random sequences.

# Fields
- `base_seed::Int`: Original seed for the experiment
- `current_seed::Int`: Currently active seed
- `seed_log::Vector{Tuple{String, Int}}`: Log of all forked seeds

# Example
```julia
sm = SeedManager(42)
fork_seed(sm, "data_split")      # Deterministic seed for data splitting
fork_seed(sm, "weight_init")     # Different seed for network initialization
fork_seed(sm, "dropout")         # Different seed for dropout sampling
```
"""
mutable struct SeedManager
    base_seed::Int
    current_seed::Int
    seed_log::Vector{Tuple{String, Int, DateTime}}

    function SeedManager(seed::Int=42)
        Random.seed!(seed)
        log = [("initialization", seed, now())]
        return new(seed, seed, log)
    end
end

"""
    set_global_seed(seed::Int) -> Int

Set global random seed for reproducibility.
Affects Random module and attempts to set CUDA seed if available.
"""
function set_global_seed(seed::Int)
    Random.seed!(seed)

    # Try to set CUDA seed if available
    try
        @eval begin
            if @isdefined(CUDA) && CUDA.functional()
                CUDA.seed!(seed)
            end
        end
    catch
        # CUDA not available - silently continue
    end

    return seed
end

"""
    fork_seed(manager::SeedManager, purpose::String) -> Int

Create a new deterministic seed from base seed for a specific purpose.
Uses SHA-256 hashing to ensure different purposes get different seeds.

# Arguments
- `manager`: SeedManager instance
- `purpose`: Descriptive string for the seed's purpose

# Returns
- New seed value (also sets global Random seed)
"""
function fork_seed(manager::SeedManager, purpose::String)
    # Hash the purpose string with base seed for determinism
    hash_input = string(manager.base_seed) * "|" * purpose
    hash_bytes = sha256(hash_input)

    # Convert first 8 bytes to Int, ensure positive
    new_seed = abs(reinterpret(Int64, hash_bytes[1:8])[1]) % 1_000_000_000

    # Update manager state
    manager.current_seed = new_seed
    push!(manager.seed_log, (purpose, new_seed, now()))

    # Set global seed
    Random.seed!(new_seed)

    return new_seed
end

"""
    get_seed_log(manager::SeedManager) -> Vector

Get the full log of all seeded operations.
"""
function get_seed_log(manager::SeedManager)
    return manager.seed_log
end

"""
    reset_seeds(manager::SeedManager)

Reset to base seed and clear log (except initialization).
"""
function reset_seeds(manager::SeedManager)
    manager.current_seed = manager.base_seed
    manager.seed_log = [("initialization", manager.base_seed, now())]
    Random.seed!(manager.base_seed)
end

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

"""
    ExperimentConfig

Versioned configuration for reproducible experiments.

Configuration is hashed to create a unique version identifier,
allowing detection of configuration changes between runs.

# Fields
- `name::String`: Experiment name
- `version::String`: Auto-generated version (v1.0-<hash>)
- `parameters::Dict{String, Any}`: Configuration parameters
- `hash::String`: SHA-256 hash of parameters (first 8 chars)
- `created_at::DateTime`: Creation timestamp
"""
struct ExperimentConfig
    name::String
    version::String
    parameters::Dict{String, Any}
    hash::String
    created_at::DateTime

    function ExperimentConfig(name::String, parameters::Dict)
        # Compute hash of parameters for versioning
        param_str = JSON3.write(parameters)
        hash = bytes2hex(sha256(param_str))[1:8]
        version = "v1.0-" * hash

        return new(name, version, parameters, hash, now())
    end
end

"""
    ExperimentConfig(name::String; kwargs...)

Create config from keyword arguments.
"""
function ExperimentConfig(name::String; kwargs...)
    params = Dict{String, Any}(string(k) => v for (k, v) in kwargs)
    return ExperimentConfig(name, params)
end

"""
    config_hash(config::ExperimentConfig) -> String

Get the configuration hash.
"""
config_hash(config::ExperimentConfig) = config.hash

"""
    validate_config(config::ExperimentConfig, expected_hash::String) -> Bool

Verify configuration matches expected hash (for reproducibility check).
"""
function validate_config(config::ExperimentConfig, expected_hash::String)
    param_str = JSON3.write(config.parameters)
    actual_hash = bytes2hex(sha256(param_str))[1:8]

    if actual_hash != expected_hash
        @warn "Configuration hash mismatch!" expected=expected_hash actual=actual_hash
        return false
    end
    return true
end

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

"""
    ExperimentRun

Single run of an experiment with full tracking.

# Fields
- `run_id::String`: Unique identifier (timestamp + random)
- `experiment_name::String`: Parent experiment name
- `config::ExperimentConfig`: Configuration used
- `parameters::Dict`: All logged parameters
- `metrics::Dict{String, Vector{Float64}}`: Time-series metrics
- `artifacts::Vector{String}`: Paths to saved artifacts
- `start_time::DateTime`: Run start time
- `end_time::Union{DateTime, Nothing}`: Run end time
- `status::Symbol`: :running, :completed, :failed
- `git_commit::String`: Git commit hash at run time
- `seed_manager::SeedManager`: Seed tracking
"""
mutable struct ExperimentRun
    run_id::String
    experiment_name::String
    config::ExperimentConfig
    parameters::Dict{String, Any}
    metrics::Dict{String, Vector{Float64}}
    artifacts::Vector{String}
    start_time::DateTime
    end_time::Union{DateTime, Nothing}
    status::Symbol
    git_commit::String
    seed_manager::SeedManager
end

"""
    ExperimentTracker

Central tracker for experiments.

Manages experiment runs, logging, and persistence.
"""
mutable struct ExperimentTracker
    base_dir::String
    experiments::Dict{String, Vector{ExperimentRun}}
    current_run::Union{ExperimentRun, Nothing}

    function ExperimentTracker(; base_dir::String="experiments/")
        mkpath(base_dir)
        return new(base_dir, Dict{String, Vector{ExperimentRun}}(), nothing)
    end
end

"""
    create_experiment(tracker, name, config; seed=42) -> ExperimentRun

Start a new experiment run.
"""
function create_experiment(tracker::ExperimentTracker, name::String,
                           config::ExperimentConfig; seed::Int=42)
    # Generate unique run ID
    run_id = Dates.format(now(), "yyyymmdd_HHMMSS") * "_" * string(uuid4())[1:8]

    # Get git commit if available
    git_commit = try
        strip(read(`git rev-parse HEAD`, String))
    catch
        "unknown"
    end

    # Create seed manager
    seed_manager = SeedManager(seed)

    run = ExperimentRun(
        run_id, name, config,
        Dict{String, Any}(config.parameters),
        Dict{String, Vector{Float64}}(),
        String[], now(), nothing, :running,
        git_commit, seed_manager
    )

    tracker.current_run = run

    if !haskey(tracker.experiments, name)
        tracker.experiments[name] = ExperimentRun[]
    end
    push!(tracker.experiments[name], run)

    # Create run directory
    run_dir = joinpath(tracker.base_dir, name, run_id)
    mkpath(run_dir)

    # Save initial config
    open(joinpath(run_dir, "config.json"), "w") do f
        JSON3.pretty(f, config.parameters)
    end

    # Save metadata
    metadata = Dict(
        "run_id" => run_id,
        "experiment" => name,
        "config_version" => config.version,
        "config_hash" => config.hash,
        "git_commit" => git_commit,
        "start_time" => string(now()),
        "seed" => seed
    )
    open(joinpath(run_dir, "metadata.json"), "w") do f
        JSON3.pretty(f, metadata)
    end

    @info "Started experiment run" run_id=run_id name=name git=git_commit[1:min(8, length(git_commit))] seed=seed

    return run
end

"""
    log_params(tracker, params::Dict)

Log hyperparameters to current run.
"""
function log_params(tracker::ExperimentTracker, params::Dict)
    if isnothing(tracker.current_run)
        error("No active experiment run. Call create_experiment first.")
    end

    merge!(tracker.current_run.parameters, params)

    # Persist
    run_dir = joinpath(tracker.base_dir, tracker.current_run.experiment_name,
                       tracker.current_run.run_id)
    open(joinpath(run_dir, "params.json"), "w") do f
        JSON3.pretty(f, tracker.current_run.parameters)
    end
end

"""
    log_params(tracker; kwargs...)

Log parameters from keyword arguments.
"""
function log_params(tracker::ExperimentTracker; kwargs...)
    params = Dict{String, Any}(string(k) => v for (k, v) in kwargs)
    log_params(tracker, params)
end

"""
    log_metrics(tracker, metrics::Dict; step=nothing)

Log metrics (loss, accuracy, etc.) to current run.
"""
function log_metrics(tracker::ExperimentTracker, metrics::Dict;
                     step::Union{Int, Nothing}=nothing)
    if isnothing(tracker.current_run)
        error("No active experiment run. Call create_experiment first.")
    end

    for (key, value) in metrics
        key_str = string(key)
        if !haskey(tracker.current_run.metrics, key_str)
            tracker.current_run.metrics[key_str] = Float64[]
        end
        push!(tracker.current_run.metrics[key_str], Float64(value))
    end

    # Append to metrics file (JSONL format)
    run_dir = joinpath(tracker.base_dir, tracker.current_run.experiment_name,
                       tracker.current_run.run_id)
    open(joinpath(run_dir, "metrics.jsonl"), "a") do f
        entry = merge(
            Dict(string(k) => v for (k, v) in metrics),
            Dict("_step" => step, "_timestamp" => string(now()))
        )
        JSON3.write(f, entry)
        println(f)
    end
end

"""
    log_metrics(tracker; step=nothing, kwargs...)

Log metrics from keyword arguments.
"""
function log_metrics(tracker::ExperimentTracker; step::Union{Int, Nothing}=nothing, kwargs...)
    metrics = Dict{String, Any}(string(k) => v for (k, v) in kwargs)
    log_metrics(tracker, metrics; step=step)
end

"""
    log_artifact(tracker, name::String, data; format=:auto)

Save artifact (model, results, plots) to current run.

# Arguments
- `name`: Artifact filename
- `data`: Data to save
- `format`: :bson, :json, or :auto (determined by extension)
"""
function log_artifact(tracker::ExperimentTracker, name::String, data;
                      format::Symbol=:auto)
    if isnothing(tracker.current_run)
        error("No active experiment run. Call create_experiment first.")
    end

    run_dir = joinpath(tracker.base_dir, tracker.current_run.experiment_name,
                       tracker.current_run.run_id, "artifacts")
    mkpath(run_dir)

    artifact_path = joinpath(run_dir, name)

    # Determine format
    if format == :auto
        if endswith(name, ".bson")
            format = :bson
        elseif endswith(name, ".json")
            format = :json
        else
            format = :bson
        end
    end

    if format == :bson
        # Use Julia's built-in serialization instead of BSON
        open(artifact_path, "w") do f
            serialize(f, Dict(:data => data))
        end
    elseif format == :json
        open(artifact_path, "w") do f
            JSON3.pretty(f, data)
        end
    else
        # Binary
        open(artifact_path, "w") do f
            write(f, data)
        end
    end

    push!(tracker.current_run.artifacts, artifact_path)
    @info "Saved artifact" name=name path=artifact_path

    return artifact_path
end

"""
    save_checkpoint(tracker, model_state, epoch; optimizer_state=nothing)

Save model checkpoint.
"""
function save_checkpoint(tracker::ExperimentTracker, model_state, epoch::Int;
                         optimizer_state=nothing)
    checkpoint_name = "checkpoint_epoch$(lpad(epoch, 4, '0')).bson"

    checkpoint_data = Dict(
        "model_state" => model_state,
        "epoch" => epoch,
        "metrics" => tracker.current_run.metrics,
        "timestamp" => string(now()),
        "seed_log" => get_seed_log(tracker.current_run.seed_manager)
    )

    if !isnothing(optimizer_state)
        checkpoint_data["optimizer_state"] = optimizer_state
    end

    return log_artifact(tracker, checkpoint_name, checkpoint_data)
end

"""
    load_checkpoint(path::String) -> Dict

Load model checkpoint.
"""
function load_checkpoint(path::String)
    data = open(path, "r") do f
        deserialize(f)
    end
    return data[:data]
end

"""
    finish_experiment(tracker; status=:completed)

Finalize experiment run.
"""
function finish_experiment(tracker::ExperimentTracker; status::Symbol=:completed)
    if isnothing(tracker.current_run)
        @warn "No active experiment run to finish"
        return nothing
    end

    tracker.current_run.end_time = now()
    tracker.current_run.status = status

    # Compute duration
    duration = Dates.value(tracker.current_run.end_time -
                          tracker.current_run.start_time) / 1000  # seconds

    # Save final summary
    run_dir = joinpath(tracker.base_dir, tracker.current_run.experiment_name,
                       tracker.current_run.run_id)

    # Compute final metrics (last value of each)
    final_metrics = Dict{String, Float64}()
    for (k, v) in tracker.current_run.metrics
        if !isempty(v)
            final_metrics[k] = v[end]
        end
    end

    summary = Dict(
        "run_id" => tracker.current_run.run_id,
        "experiment" => tracker.current_run.experiment_name,
        "status" => string(status),
        "start_time" => string(tracker.current_run.start_time),
        "end_time" => string(tracker.current_run.end_time),
        "duration_seconds" => duration,
        "git_commit" => tracker.current_run.git_commit,
        "config_hash" => tracker.current_run.config.hash,
        "final_metrics" => final_metrics,
        "n_artifacts" => length(tracker.current_run.artifacts),
        "seed_log" => [(p, s) for (p, s, _) in get_seed_log(tracker.current_run.seed_manager)]
    )

    open(joinpath(run_dir, "summary.json"), "w") do f
        JSON3.pretty(f, summary)
    end

    @info "Finished experiment" run_id=tracker.current_run.run_id status=status duration="$(round(duration, digits=1))s"

    run = tracker.current_run
    tracker.current_run = nothing

    return run
end

# ============================================================================
# RESULTS CACHING
# ============================================================================

"""
    cached_result(fn, cache_key::String; cache_dir="cache/", force_recompute=false)

Cache expensive computations for reproducibility and efficiency.

# Arguments
- `fn`: Zero-argument function to compute result
- `cache_key`: Unique key for this computation
- `cache_dir`: Directory for cache files
- `force_recompute`: If true, recompute even if cached

# Returns
- Cached or freshly computed result
"""
function cached_result(fn::Function, cache_key::String;
                       cache_dir::String="cache/", force_recompute::Bool=false)
    mkpath(cache_dir)

    # Sanitize cache key for filename
    safe_key = replace(cache_key, r"[^a-zA-Z0-9_-]" => "_")
    cache_path = joinpath(cache_dir, safe_key * ".jls")  # Julia serialization format

    if !force_recompute && isfile(cache_path)
        @info "Loading cached result" key=cache_key
        data = open(cache_path, "r") do f
            deserialize(f)
        end
        return data[:result]
    end

    @info "Computing result (will cache)" key=cache_key
    result = fn()

    open(cache_path, "w") do f
        serialize(f, Dict(:result => result, :timestamp => string(now()), :key => cache_key))
    end
    @info "Cached result" key=cache_key path=cache_path

    return result
end

"""
    clear_cache(; cache_dir="cache/", pattern=nothing)

Clear cached results.

# Arguments
- `cache_dir`: Cache directory
- `pattern`: Optional regex pattern to match keys (nothing = clear all)
"""
function clear_cache(; cache_dir::String="cache/", pattern=nothing)
    if !isdir(cache_dir)
        return 0
    end

    count = 0
    for file in readdir(cache_dir)
        if endswith(file, ".jls") || endswith(file, ".bson")  # Support both formats
            if isnothing(pattern) || occursin(pattern, file)
                rm(joinpath(cache_dir, file))
                count += 1
            end
        end
    end

    @info "Cleared cache" removed=count
    return count
end

# ============================================================================
# UTILITIES
# ============================================================================

"""
    list_experiments(tracker) -> Vector{String}

List all experiment names.
"""
function list_experiments(tracker::ExperimentTracker)
    if isdir(tracker.base_dir)
        return [d for d in readdir(tracker.base_dir) if isdir(joinpath(tracker.base_dir, d))]
    end
    return String[]
end

"""
    list_runs(tracker, experiment_name) -> Vector{String}

List all run IDs for an experiment.
"""
function list_runs(tracker::ExperimentTracker, experiment_name::String)
    exp_dir = joinpath(tracker.base_dir, experiment_name)
    if isdir(exp_dir)
        return [d for d in readdir(exp_dir) if isdir(joinpath(exp_dir, d))]
    end
    return String[]
end

"""
    load_run_summary(tracker, experiment_name, run_id) -> Dict

Load summary for a specific run.
"""
function load_run_summary(tracker::ExperimentTracker, experiment_name::String, run_id::String)
    summary_path = joinpath(tracker.base_dir, experiment_name, run_id, "summary.json")
    if isfile(summary_path)
        return JSON3.read(read(summary_path, String))
    end
    return nothing
end

end # module
