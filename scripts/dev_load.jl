"""
dev_load.jl - Quick development loader for DARWIN Scaffold Studio

Use this for development instead of `using DarwinScaffoldStudio` until
full module integration is complete.

Usage:
    include("dev_load.jl")
    
Then call functions directly:
    volume, meta = generate_synthetic_scaffold(size_voxels=(100,100,100))
    metrics = compute_metrics(volume, 10.0)
"""

println("🔧 Loading DARWIN Scaffold Studio (Development Mode)...")

# Core modules first
include("src/DarwinScaffoldStudio/Core/Config.jl")
include("src/DarwinScaffoldStudio/Core/Types.jl")
include("src/DarwinScaffoldStudio/Core/Utils.jl")
include("src/DarwinScaffoldStudio/Core/ErrorHandling.jl")
include("src/DarwinScaffoldStudio/Core/DataIngestion.jl")

# MicroCT modules
include("src/DarwinScaffoldStudio/MicroCT/ImageLoader.jl")
include("src/DarwinScaffoldStudio/MicroCT/Preprocessing.jl")
include("src/DarwinScaffoldStudio/MicroCT/Segmentation.jl")
include("src/DarwinScaffoldStudio/MicroCT/Metrics.jl")

# Optimization modules
include("src/DarwinScaffoldStudio/Optimization/Parametric.jl")
# Note: ScaffoldOptimizer has dependency issues, skip for now

# Visualization modules
include("src/DarwinScaffoldStudio/Visualization/Mesh3D.jl")
include("src/DarwinScaffoldStudio/Visualization/Export.jl")

# Scientific modules (Core)
include("src/DarwinScaffoldStudio/Science/Topology.jl")
include("src/DarwinScaffoldStudio/Science/Percolation.jl")
include("src/DarwinScaffoldStudio/Science/ML.jl")

# Agent Framework  
include("src/DarwinScaffoldStudio/LLM/OllamaClient.jl")
include("src/DarwinScaffoldStudio/Agents/Core.jl")
include("src/DarwinScaffoldStudio/Agents/DesignAgent.jl")
include("src/DarwinScaffoldStudio/Agents/AnalysisAgent.jl")
include("src/DarwinScaffoldStudio/Agents/SynthesisAgent.jl")

# Make key modules available
using .Config
using .Types
using .DataIngestion
using .ImageLoader
using .Metrics
using .Mesh3D
using .Export
using .Topology
using .Percolation
using .ML
using .OllamaClient
using .Core: Agent, AgentTool, AgentWorkspace, run_agent
using .DesignAgent
using .AnalysisAgent
using .SynthesisAgent

println("✅ Core modules loaded successfully!")
println()
println("📦 Available modules:")
println("  • Config (get_config, get_global_config)")
println("  • DataIngestion (generate_synthetic_scaffold, load_scaffold_data)")
println("  • Metrics (compute_metrics)")
println("  • Topology (compute_kec_metrics)")
println("  • Percolation (compute_percolation_metrics)")
println("  • ML (predict_viability)")
println("  • Agents (Agent, create_design_agent, create_analysis_agent, create_synthesis_agent)")
println("  • OllamaClient (OllamaModel, chat)")
println()
println("💡 Quick start:")
println("  volume, meta = DataIngestion.generate_synthetic_scaffold()")
println("  metrics = Metrics.compute_metrics(volume, 10.0)")
println()
