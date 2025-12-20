"""
CompilerBridge.jl - Interface to Demetrios Compiler

Provides Julia bindings to the Demetrios (D) language compiler for
running scientific models with epistemic computing, units of measure,
and GPU support.

Features:
- Detect and locate Demetrios compiler (dc)
- Compile .d source files
- Run compiled models with Julia data
- Capture output and errors
- GPU kernel compilation support

References:
- github.com/chiuratto-AI/demetrios
"""
module CompilerBridge

export DemetriosCompiler, find_compiler, is_available
export compile_demetrios, run_demetrios, check_syntax
export DemetriosError, CompilationError, RuntimeError
export get_compiler_version, get_compiler_info

# ============================================================================
# EXCEPTIONS
# ============================================================================

"""
    DemetriosError

Base exception for Demetrios-related errors.
"""
abstract type DemetriosError <: Exception end

"""
    CompilationError

Error during Demetrios compilation.
"""
struct CompilationError <: DemetriosError
    message::String
    source_file::String
    line::Union{Int, Nothing}
    column::Union{Int, Nothing}
    stderr_output::String
end

function Base.showerror(io::IO, e::CompilationError)
    print(io, "CompilationError: ", e.message)
    if !isnothing(e.line)
        print(io, " at line ", e.line)
        if !isnothing(e.column)
            print(io, ":", e.column)
        end
    end
    if !isempty(e.source_file)
        print(io, " in ", e.source_file)
    end
end

"""
    RuntimeError

Error during Demetrios model execution.
"""
struct RuntimeError <: DemetriosError
    message::String
    stderr_output::String
end

function Base.showerror(io::IO, e::RuntimeError)
    print(io, "RuntimeError: ", e.message)
end

# ============================================================================
# COMPILER CONFIGURATION
# ============================================================================

"""
    DemetriosCompiler

Configuration for the Demetrios compiler.

# Fields
- `path::String`: Path to dc executable
- `version::String`: Compiler version
- `features::Vector{Symbol}`: Available features (:gpu, :prob, :units, etc.)
- `stdlib_path::String`: Path to standard library
"""
struct DemetriosCompiler
    path::String
    version::String
    features::Vector{Symbol}
    stdlib_path::String
end

# Global compiler instance (lazy initialized)
const _compiler = Ref{Union{DemetriosCompiler, Nothing}}(nothing)

"""
    find_compiler(; paths=nothing) -> Union{DemetriosCompiler, Nothing}

Locate the Demetrios compiler.

Searches in:
1. Custom paths if provided
2. PATH environment variable
3. Common installation locations
4. Project-local compiler directory

Returns nothing if not found.
"""
function find_compiler(; paths::Union{Vector{String}, Nothing}=nothing)
    # Build search paths
    search_paths = String[]

    # Custom paths first
    if !isnothing(paths)
        append!(search_paths, paths)
    end

    # Environment variable
    if haskey(ENV, "DEMETRIOS_HOME")
        push!(search_paths, joinpath(ENV["DEMETRIOS_HOME"], "target", "release"))
        push!(search_paths, joinpath(ENV["DEMETRIOS_HOME"], "target", "debug"))
    end

    # Common locations
    common_paths = [
        # Local project
        joinpath(dirname(dirname(dirname(@__DIR__))), "compiler", "target", "release"),
        joinpath(dirname(dirname(dirname(@__DIR__))), "compiler", "target", "debug"),
        # User home
        joinpath(homedir(), ".demetrios", "bin"),
        joinpath(homedir(), "demetrios", "target", "release"),
        # System-wide
        "/usr/local/bin",
        "/usr/bin",
        # Workspace locations (common development setups)
        joinpath(dirname(dirname(dirname(dirname(@__DIR__)))), "demetrios", "compiler", "target", "release"),
        joinpath(dirname(dirname(dirname(dirname(@__DIR__)))), "Darwin-demetrios", "compiler", "target", "release"),
    ]
    append!(search_paths, common_paths)

    # Search for dc executable
    for dir in search_paths
        dc_path = joinpath(dir, "dc")
        if Sys.iswindows()
            dc_path *= ".exe"
        end

        if isfile(dc_path)
            try
                # Try to get version
                version_output = read(`$dc_path --version`, String)
                version = strip(split(version_output, "\n")[1])

                # Detect features by checking help
                features = Symbol[]
                try
                    help_output = read(`$dc_path --help`, String)
                    if occursin("--gpu", help_output) || occursin("gpu", lowercase(help_output))
                        push!(features, :gpu)
                    end
                    if occursin("prob", lowercase(help_output))
                        push!(features, :prob)
                    end
                    push!(features, :units)  # Units always available
                    push!(features, :epistemic)  # Core feature
                catch
                    # Minimal feature set
                    features = [:units, :epistemic]
                end

                # Find stdlib
                stdlib_path = ""
                for stdlib_candidate in [
                    joinpath(dirname(dir), "stdlib"),
                    joinpath(dirname(dirname(dir)), "stdlib"),
                    joinpath(dirname(dir), "..", "stdlib"),
                ]
                    if isdir(stdlib_candidate)
                        stdlib_path = stdlib_candidate
                        break
                    end
                end

                compiler = DemetriosCompiler(dc_path, version, features, stdlib_path)
                _compiler[] = compiler

                @info "Found Demetrios compiler" path=dc_path version=version features=features

                return compiler
            catch e
                @debug "Found dc at $dc_path but couldn't verify" exception=e
                continue
            end
        end
    end

    @warn "Demetrios compiler (dc) not found. Some features will be unavailable."
    return nothing
end

"""
    is_available() -> Bool

Check if Demetrios compiler is available.
"""
function is_available()
    if isnothing(_compiler[])
        find_compiler()
    end
    return !isnothing(_compiler[])
end

"""
    get_compiler() -> DemetriosCompiler

Get the compiler instance, finding it if necessary.
"""
function get_compiler()
    if isnothing(_compiler[])
        compiler = find_compiler()
        if isnothing(compiler)
            error("Demetrios compiler not found. Install from github.com/chiuratto-AI/demetrios")
        end
    end
    return _compiler[]
end

"""
    get_compiler_version() -> String

Get compiler version string.
"""
function get_compiler_version()
    compiler = get_compiler()
    return compiler.version
end

"""
    get_compiler_info() -> Dict

Get comprehensive compiler information.
"""
function get_compiler_info()
    compiler = get_compiler()
    return Dict(
        "path" => compiler.path,
        "version" => compiler.version,
        "features" => compiler.features,
        "stdlib_path" => compiler.stdlib_path,
        "has_gpu" => :gpu in compiler.features,
        "has_prob" => :prob in compiler.features
    )
end

# ============================================================================
# COMPILATION
# ============================================================================

"""
    check_syntax(source::String; source_file="<inline>") -> Bool

Check Demetrios source code syntax without full compilation.
"""
function check_syntax(source::String; source_file::String="<inline>")
    compiler = get_compiler()

    # Write to temp file
    temp_file = tempname() * ".d"
    try
        write(temp_file, source)

        # Run syntax check
        result = run(pipeline(`$(compiler.path) check $temp_file`, stderr=stderr), wait=false)
        wait(result)

        return result.exitcode == 0
    finally
        rm(temp_file, force=true)
    end
end

"""
    compile_demetrios(source_file::String; output=nothing, optimize=true, gpu=false)

Compile a Demetrios source file.

# Arguments
- `source_file`: Path to .d source file
- `output`: Output path (default: same name with different extension)
- `optimize`: Enable optimizations
- `gpu`: Enable GPU code generation

# Returns
- Path to compiled output
"""
function compile_demetrios(source_file::String;
                           output::Union{String, Nothing}=nothing,
                           optimize::Bool=true,
                           gpu::Bool=false)
    compiler = get_compiler()

    if !isfile(source_file)
        error("Source file not found: $source_file")
    end

    # Build command
    cmd_parts = [compiler.path]

    # Add stdlib path if available
    if !isempty(compiler.stdlib_path)
        push!(cmd_parts, "--stdlib")
        push!(cmd_parts, compiler.stdlib_path)
    end

    # Optimization
    if optimize
        push!(cmd_parts, "--release")
    end

    # GPU
    if gpu
        if !(:gpu in compiler.features)
            @warn "GPU not available in this Demetrios build"
        else
            push!(cmd_parts, "--gpu")
        end
    end

    # Output path
    if isnothing(output)
        output = replace(source_file, r"\.d$" => ".out")
    end
    push!(cmd_parts, "-o")
    push!(cmd_parts, output)

    # Source file
    push!(cmd_parts, source_file)

    # Run compiler
    cmd = Cmd(cmd_parts)
    stdout_buf = IOBuffer()
    stderr_buf = IOBuffer()

    try
        run(pipeline(cmd, stdout=stdout_buf, stderr=stderr_buf))
        return output
    catch e
        stderr_output = String(take!(stderr_buf))

        # Parse error location if possible
        line, col = nothing, nothing
        m = match(r"line\s+(\d+)", stderr_output)
        if !isnothing(m)
            line = parse(Int, m.captures[1])
        end
        m = match(r"column\s+(\d+)", stderr_output)
        if !isnothing(m)
            col = parse(Int, m.captures[1])
        end

        throw(CompilationError(
            "Compilation failed",
            source_file, line, col,
            stderr_output
        ))
    end
end

"""
    run_demetrios(source_file::String, inputs::Dict; timeout=60) -> Dict

Run a Demetrios model with given inputs.

# Arguments
- `source_file`: Path to .d source file
- `inputs`: Dict of input parameters
- `timeout`: Maximum execution time in seconds

# Returns
- Dict with output values
"""
function run_demetrios(source_file::String, inputs::Dict=Dict();
                       timeout::Int=60)
    compiler = get_compiler()

    if !isfile(source_file)
        error("Source file not found: $source_file")
    end

    # Build command
    cmd_parts = [compiler.path, "run"]

    # Add stdlib path
    if !isempty(compiler.stdlib_path)
        push!(cmd_parts, "--stdlib")
        push!(cmd_parts, compiler.stdlib_path)
    end

    # Add input parameters
    for (k, v) in inputs
        push!(cmd_parts, "--param")
        push!(cmd_parts, "$k=$v")
    end

    # Source file
    push!(cmd_parts, source_file)

    # Run with timeout
    cmd = Cmd(cmd_parts)
    stdout_buf = IOBuffer()
    stderr_buf = IOBuffer()

    try
        proc = run(pipeline(cmd, stdout=stdout_buf, stderr=stderr_buf), wait=false)

        # Wait with timeout
        start_time = time()
        while process_running(proc)
            if time() - start_time > timeout
                kill(proc)
                throw(RuntimeError("Execution timeout after $(timeout)s", ""))
            end
            sleep(0.1)
        end

        if proc.exitcode != 0
            stderr_output = String(take!(stderr_buf))
            throw(RuntimeError("Execution failed with code $(proc.exitcode)", stderr_output))
        end

        # Parse output
        stdout_output = String(take!(stdout_buf))
        return parse_demetrios_output(stdout_output)

    catch e
        if e isa DemetriosError
            rethrow()
        end
        stderr_output = String(take!(stderr_buf))
        throw(RuntimeError("Execution error: $e", stderr_output))
    end
end

"""
    parse_demetrios_output(output::String) -> Dict

Parse Demetrios program output into Julia Dict.
"""
function parse_demetrios_output(output::String)
    result = Dict{String, Any}()

    for line in split(output, "\n")
        line = strip(line)
        if isempty(line) || startswith(line, "#")
            continue
        end

        # Try to parse key=value or key: value
        m = match(r"^(\w+)\s*[=:]\s*(.+)$", line)
        if !isnothing(m)
            key = m.captures[1]
            value_str = strip(m.captures[2])

            # Parse value
            value = try
                # Try as number
                if occursin(".", value_str)
                    parse(Float64, value_str)
                else
                    parse(Int, value_str)
                end
            catch
                # Keep as string
                value_str
            end

            result[key] = value
        end
    end

    return result
end

# ============================================================================
# GPU KERNEL SUPPORT
# ============================================================================

"""
    compile_demetrios_kernel(source::String; name="kernel") -> CompiledKernel

Compile a Demetrios GPU kernel for use from Julia.

# Example
```julia
kernel = compile_demetrios_kernel(\"""
    @gpu fn diffusion_step(C: &mut [f32], D: f32, dt: f32) {
        let idx = thread_idx();
        C[idx] += D * dt * laplacian(C, idx);
    }
\""")
```
"""
function compile_demetrios_kernel(source::String; name::String="kernel")
    compiler = get_compiler()

    if !(:gpu in compiler.features)
        error("GPU support not available in this Demetrios build")
    end

    # Write kernel source
    temp_file = tempname() * ".d"
    try
        write(temp_file, source)

        # Compile to PTX/GPU IR
        output_file = tempname() * ".ptx"
        cmd = `$(compiler.path) compile --gpu --emit-ptx -o $output_file $temp_file`

        run(cmd)

        # Read compiled kernel
        ptx_code = read(output_file, String)

        return CompiledKernel(name, ptx_code, source)
    finally
        rm(temp_file, force=true)
    end
end

"""
    CompiledKernel

A compiled Demetrios GPU kernel.
"""
struct CompiledKernel
    name::String
    ptx_code::String
    source::String
end

# ============================================================================
# INITIALIZATION
# ============================================================================

function __init__()
    # Try to find compiler on module load
    find_compiler()
end

end # module
