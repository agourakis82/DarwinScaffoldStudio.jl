"""
Scaffold Optimizer Module

Parametric scaffold optimization based on Murphy 2010, Karageorgiou 2005.
"""

module ScaffoldOptimizer

using ..Types
using ..Config
using Statistics
using Random

# Import shared fabrication constants
include("FabricationConstants.jl")
using .FabricationConstants

export Optimizer, optimize_scaffold, detect_problems

# Note: Metrics module will be imported when needed
# import ..Metrics

"""
    Optimizer

Main scaffold optimizer struct.
"""
mutable struct Optimizer
    voxel_size_um::Float64
    config::ScaffoldConfig
    
    function Optimizer(; voxel_size_um::Float64=10.0, config::ScaffoldConfig=Config.get_config())
        new(voxel_size_um, config)
    end
end

"""
    detect_problems(metrics::ScaffoldMetrics) -> Dict{String, String}

Detect scaffold design problems based on Q1 literature.

# Returns
- Dictionary of problem descriptions
"""
function detect_problems(metrics::ScaffoldMetrics)::Dict{String, String}
    problems = Dict{String, String}()

    # Murphy 2010: pore size should be 100-200 μm
    if !(PORE_SIZE_MIN_UM <= metrics.mean_pore_size_um <= PORE_SIZE_MAX_UM)
        if metrics.mean_pore_size_um < PORE_SIZE_MIN_UM
            problems["pore_size"] = "Pore size too small ($(round(metrics.mean_pore_size_um, digits=1)) μm < $(Int(PORE_SIZE_MIN_UM)) μm). Target: $(Int(PORE_SIZE_MIN_UM))-$(Int(PORE_SIZE_MAX_UM)) μm (Murphy 2010)."
        else
            problems["pore_size"] = "Pore size too large ($(round(metrics.mean_pore_size_um, digits=1)) μm > $(Int(PORE_SIZE_MAX_UM)) μm). Target: $(Int(PORE_SIZE_MIN_UM))-$(Int(PORE_SIZE_MAX_UM)) μm (Murphy 2010)."
        end
    end

    # Karageorgiou 2005: porosity should be 0.90-0.95
    if !(POROSITY_MIN <= metrics.porosity <= POROSITY_MAX)
        if metrics.porosity < POROSITY_MIN
            problems["porosity"] = "Porosity too low ($(round(metrics.porosity * 100, digits=1))% < $(Int(POROSITY_MIN * 100))%). Target: $(Int(POROSITY_MIN * 100))-$(Int(POROSITY_MAX * 100))% (Karageorgiou 2005)."
        else
            problems["porosity"] = "Porosity too high ($(round(metrics.porosity * 100, digits=1))% > $(Int(POROSITY_MAX * 100))%). Target: $(Int(POROSITY_MIN * 100))-$(Int(POROSITY_MAX * 100))% (Karageorgiou 2005)."
        end
    end

    # Karageorgiou 2005: interconnectivity should be ≥90%
    if metrics.interconnectivity < INTERCONNECTIVITY_MIN
        problems["interconnectivity"] = "Interconnectivity too low ($(round(metrics.interconnectivity * 100, digits=1))% < $(Int(INTERCONNECTIVITY_MIN * 100))%). Target: ≥$(Int(INTERCONNECTIVITY_MIN * 100))% (Karageorgiou 2005)."
    end

    # Tortuosity should be <1.2
    if metrics.tortuosity >= TORTUOSITY_MAX
        problems["tortuosity"] = "Tortuosity too high ($(round(metrics.tortuosity, digits=2)) ≥ $(TORTUOSITY_MAX)). Target: <$(TORTUOSITY_MAX) for straight paths."
    end

    return problems
end

"""
    optimize_scaffold(optimizer::Optimizer, 
                      original_volume::Array{Bool, 3},
                      target_params::ScaffoldParameters) -> OptimizationResults

Optimize scaffold to meet target parameters.

# Arguments
- `optimizer`: Optimizer instance
- `original_volume`: Original scaffold binary volume
- `target_params`: Target scaffold parameters

# Returns
- OptimizationResults with optimized scaffold
"""
function optimize_scaffold(
    optimizer::Optimizer,
    original_volume::Array{Bool, 3},
    target_params::ScaffoldParameters
)::OptimizationResults
    # Import Metrics here to avoid circular dependency
    Metrics = Base.parentmodule(@__MODULE__).Metrics
    
    # Analyze original
    original_metrics = Metrics.compute_metrics(original_volume, optimizer.voxel_size_um)
    
    # Generate optimized volume
    optimized_volume = generate_optimized_volume(
        original_volume,
        target_params,
        optimizer.voxel_size_um
    )
    
    # Analyze optimized
    optimized_metrics = Metrics.compute_metrics(optimized_volume, optimizer.voxel_size_um)
    
    # Compute improvements (percentage change)
    improvement = Dict{String, Float64}(
        "porosity" => ((optimized_metrics.porosity - original_metrics.porosity) / max(original_metrics.porosity, MIN_POROSITY_DIVISOR)) * 100.0,
        "pore_size" => ((optimized_metrics.mean_pore_size_um - original_metrics.mean_pore_size_um) / max(original_metrics.mean_pore_size_um, MIN_PORE_SIZE_DIVISOR)) * 100.0,
        "interconnectivity" => ((optimized_metrics.interconnectivity - original_metrics.interconnectivity) / max(original_metrics.interconnectivity, MIN_INTERCONNECTIVITY_DIVISOR)) * 100.0,
        "tortuosity" => ((original_metrics.tortuosity - optimized_metrics.tortuosity) / max(original_metrics.tortuosity, MIN_TORTUOSITY_DIVISOR)) * 100.0
    )
    
    # Determine fabrication method
    fabrication_method, fabrication_params = recommend_fabrication_method(optimized_metrics)
    
    return OptimizationResults(
        optimized_volume,
        original_metrics,
        optimized_metrics,
        improvement,
        fabrication_method,
        fabrication_params
    )
end

"""
    generate_optimized_volume(original::Array{Bool, 3},
                             target_params::ScaffoldParameters,
                             voxel_size_um::Float64) -> Array{Bool, 3}

Generate optimized scaffold volume.

Uses parametric approach: adjust porosity and pore size distribution.
"""
function generate_optimized_volume(
    original::Array{Bool, 3},
    target_params::ScaffoldParameters,
    voxel_size_um::Float64
)::Array{Bool, 3}
    dims = size(original)
    optimized = Array{Bool, 3}(undef, dims)
    
    # Current porosity
    current_porosity = 1.0 - (sum(original) / length(original))
    target_porosity = target_params.porosity_target
    
    # Adjust porosity
    if target_porosity > current_porosity
        # Need more pores - remove some solid
        threshold = 1.0 - target_porosity
        n_solid_to_remove = Int(ceil((target_porosity - current_porosity) * length(original)))
    else
        # Need less pores - add some solid
        threshold = 1.0 - target_porosity
        n_pores_to_fill = Int(ceil((current_porosity - target_porosity) * length(original)))
    end
    
    # Simple approach: threshold-based adjustment
    # In production, would use more sophisticated parametric generation
    optimized = copy(original)
    
    # Adjust to target porosity
    if target_porosity != current_porosity
        # Randomly adjust voxels to meet target
        all_indices = collect(CartesianIndices(optimized))
        Random.shuffle!(all_indices)
        
        if target_porosity > current_porosity
            # Remove solid voxels
            n_to_remove = Int(ceil((target_porosity - current_porosity) * length(optimized)))
            for idx in all_indices[1:min(n_to_remove, length(all_indices))]
                if optimized[idx]
                    optimized[idx] = false
                end
            end
        else
            # Add solid voxels
            n_to_add = Int(ceil((current_porosity - target_porosity) * length(optimized)))
            for idx in all_indices[1:min(n_to_add, length(all_indices))]
                if !optimized[idx]
                    optimized[idx] = true
                end
            end
        end
    end
    
    return optimized
end

"""
    recommend_fabrication_method(metrics::ScaffoldMetrics) -> Tuple{String, Dict{String, Any}}

Recommend fabrication method based on scaffold metrics.

# Returns
- (method_name, parameters)
"""
function recommend_fabrication_method(
    metrics::ScaffoldMetrics
)::Tuple{String, Dict{String, Any}}
    # Decision tree based on metrics
    if metrics.porosity > HIGH_POROSITY_THRESHOLD && metrics.mean_pore_size_um > LARGE_PORE_THRESHOLD_UM
        return ("freeze-casting", Dict(
            "temperature" => FREEZE_CASTING_TEMP_C,
            "freezing_rate" => FREEZE_CASTING_RATE,
            "solute_concentration" => FREEZE_CASTING_SOLUTE_CONC
        ))
    elseif metrics.mean_pore_size_um < SMALL_PORE_THRESHOLD_UM
        return ("3D-bioprinting", Dict(
            "nozzle_diameter_um" => BIOPRINTING_NOZZLE_UM,
            "layer_height_um" => BIOPRINTING_LAYER_UM,
            "print_speed" => BIOPRINTING_SPEED
        ))
    else
        return ("salt-leaching", Dict(
            "salt_particle_size_um" => metrics.mean_pore_size_um,
            "salt_volume_fraction" => metrics.porosity,
            "leaching_time_h" => SALT_LEACHING_TIME_H
        ))
    end
end

end # module ScaffoldOptimizer

