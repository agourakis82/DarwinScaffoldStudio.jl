"""
Configuration for DarwinScaffoldStudio
"""

module Config

export ScaffoldConfig, get_config, set_config, GlobalConfig, get_global_config, VERSION

"""
    VERSION::String

Current version of Darwin Scaffold Studio, read from Project.toml.
"""
const VERSION = let
    project_file = joinpath(dirname(dirname(dirname(@__DIR__))), "Project.toml")
    version = "0.0.0"
    if isfile(project_file)
        for line in eachline(project_file)
            m = match(r"^version\s*=\s*\"([^\"]+)\"", line)
            if m !== nothing
                version = m.captures[1]
                break
            end
        end
    end
    version
end

"""
    GlobalConfig

System-wide configuration for DARWIN Scaffold Studio.

# Fields
- `data_directory::String`: Root directory for data files (default: "data/")
- `results_directory::String`: Output directory for results (default: "results/")
- `ollama_base_url::String`: Ollama API endpoint (default: "http://localhost:11434")
- `enable_advanced_modules::Bool`: Load advanced features (Quantum, Blockchain, etc.)
- `enable_frontier_ai::Bool`: Load FRONTIER AI modules (PINNs, TDA, GNN)
- `enable_visualization::Bool`: Load visualization modules (NeRF, Gaussian Splatting)
- `debug_mode::Bool`: Enable verbose logging and error traces
"""
struct GlobalConfig
    data_directory::String
    results_directory::String
    ollama_base_url::String
    enable_advanced_modules::Bool
    enable_frontier_ai::Bool
    enable_visualization::Bool
    debug_mode::Bool
    
    function GlobalConfig(;
        data_directory::String = "data/",
        results_directory::String = "results/",
        ollama_base_url::String = "http://localhost:11434",
        enable_advanced_modules::Bool = false,  # Off by default for stability
        enable_frontier_ai::Bool = true,         # Core FRONTIER features
        enable_visualization::Bool = true,
        debug_mode::Bool = false
    )
        # Create directories if they don't exist
        for dir in [data_directory, results_directory]
            if !isdir(dir)
                mkpath(dir)
            end
        end
        
        new(data_directory, results_directory, ollama_base_url,
            enable_advanced_modules, enable_frontier_ai, enable_visualization,
            debug_mode)
    end
end

const _system_config = Ref{GlobalConfig}()

"""
    get_global_config() -> GlobalConfig

Get system-wide configuration.
"""
function get_global_config()
    if !isassigned(_system_config)
        _system_config[] = GlobalConfig()
    end
    return _system_config[]
end

"""
    ScaffoldConfig

Configuration for scaffold analysis and optimization.

# Fields
- `voxel_size_um::Float64`: Voxel size in micrometers
- `porosity_target::Float64`: Target porosity (0.90-0.95, Murphy 2010)
- `pore_size_target_um::Float64`: Target pore size in μm (100-200, Murphy 2010)
- `interconnectivity_target::Float64`: Target interconnectivity (≥0.90, Karageorgiou 2005)
- `tortuosity_target::Float64`: Target tortuosity (<1.2)
"""
struct ScaffoldConfig
    voxel_size_um::Float64
    porosity_target::Float64
    pore_size_target_um::Float64
    interconnectivity_target::Float64
    tortuosity_target::Float64
    
    function ScaffoldConfig(;
        voxel_size_um::Float64 = 10.0,
        porosity_target::Float64 = 0.92,
        pore_size_target_um::Float64 = 150.0,
        interconnectivity_target::Float64 = 0.95,
        tortuosity_target::Float64 = 1.1
    )
        new(voxel_size_um, porosity_target, pore_size_target_um, 
            interconnectivity_target, tortuosity_target)
    end
end

const _global_config = Ref{ScaffoldConfig}()

"""
    get_config() -> ScaffoldConfig

Get global scaffold configuration.
"""
function get_config()
    if !isassigned(_global_config)
        _global_config[] = ScaffoldConfig()
    end
    return _global_config[]
end

"""
    set_config(config::ScaffoldConfig)

Set global scaffold configuration.
"""
function set_config(config::ScaffoldConfig)
    _global_config[] = config
end

end # module Config

