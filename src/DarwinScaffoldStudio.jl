"""
DarwinScaffoldStudio.jl - Q1-Level MicroCT & SEM Analysis

Julia 1.10 implementation for tissue engineering scaffold analysis.

Created: 2025-11-13
Author: Dr. Demetrios Agourakis + AI Assistant
Version: 1.0.0
"""

module DarwinScaffoldStudio

# Core modules (MUST load successfully)
include("DarwinScaffoldStudio/Core/Config.jl")
include("DarwinScaffoldStudio/Core/ErrorHandling.jl")
include("DarwinScaffoldStudio/Core/Types.jl")
include("DarwinScaffoldStudio/Core/Utils.jl")

using .Config: get_global_config
using .ErrorHandling: @safe_include

# Get configuration to determine what to load
const SYSTEM_CONFIG = get_global_config()

# MicroCT modules (Core functionality)
include("DarwinScaffoldStudio/MicroCT/ImageLoader.jl")
include("DarwinScaffoldStudio/MicroCT/Preprocessing.jl")
include("DarwinScaffoldStudio/MicroCT/Segmentation.jl")
include("DarwinScaffoldStudio/MicroCT/Metrics.jl")

# Optimization modules (Core functionality)
include("DarwinScaffoldStudio/Optimization/Parametric.jl")
include("DarwinScaffoldStudio/Optimization/ScaffoldOptimizer.jl")

# Visualization modules (Core functionality)
include("DarwinScaffoldStudio/Visualization/Mesh3D.jl")
include("DarwinScaffoldStudio/Visualization/Export.jl")

# Scientific modules (Thesis - Core functionality)
include("DarwinScaffoldStudio/Science/Topology.jl")
include("DarwinScaffoldStudio/Science/Percolation.jl")
include("DarwinScaffoldStudio/Science/ML.jl")
include("DarwinScaffoldStudio/Science/Optimization.jl")

# SOTA 2025: Agent Framework (Core functionality)
include("DarwinScaffoldStudio/LLM/OllamaClient.jl")
include("DarwinScaffoldStudio/Agents/Core.jl")
include("DarwinScaffoldStudio/Agents/DesignAgent.jl")
include("DarwinScaffoldStudio/Agents/AnalysisAgent.jl")
include("DarwinScaffoldStudio/Agents/SynthesisAgent.jl")

# SEMANTIC LAYER: Epistemic Ontology (Demetrios-inspired)
# Knowledge[τ, ε, δ, Φ] types + Schema.org + Hyperbolic embeddings
@info "Loading Semantic layer (Ontology, KnowledgeGraph, EpistemicAgents)..."
@safe_include "DarwinScaffoldStudio/Semantic/Ontology.jl" "Ontology"
@safe_include "DarwinScaffoldStudio/Semantic/KnowledgeGraph.jl" "KnowledgeGraph"
@safe_include "DarwinScaffoldStudio/Semantic/EpistemicAgents.jl" "EpistemicAgents"

# OBO FOUNDRY INTEGRATION: Standard biomedical ontologies
# UBERON (anatomy), CL (cells), CHEBI (chemicals), GO (processes), NCIT (diseases)
@info "Loading OBO Foundry integration (UBERON, CL, CHEBI, GO, NCIT, BTO)..."
include("DarwinScaffoldStudio/Ontology/OBOFoundry.jl")

# ONTOLOGY MANAGER: 3-Tier lookup system with caching
# Tier 1: Hardcoded (~150 terms), Tier 2: SQLite cache, Tier 3: EBI OLS API
@info "Loading OntologyManager (3-tier lookup with FAIR export)..."
@safe_include "DarwinScaffoldStudio/Ontology/OntologyManager.jl" "OntologyManager"

# ONTOLOGY LIBRARIES: Comprehensive tissue, cell, and material databases
# 150+ tissues, 100+ cells, 80+ materials with properties and relationships
@info "Loading Ontology Libraries (TissueLibrary, CellLibrary, MaterialLibrary)..."
@safe_include "DarwinScaffoldStudio/Ontology/TissueLibrary.jl" "TissueLibrary"
@safe_include "DarwinScaffoldStudio/Ontology/CellLibrary.jl" "CellLibrary"
@safe_include "DarwinScaffoldStudio/Ontology/MaterialLibrary.jl" "MaterialLibrary"

# INTERACTIVE: ScaffoldEditor with Q1 Literature Validation
@info "Loading Interactive layer (ScaffoldEditor)..."
module Interactive
# ScaffoldEditor will import Types and Config directly via ... prefix
include("DarwinScaffoldStudio/Interactive/ScaffoldEditor.jl")
end

# FRONTIER: Advanced AI Modules (Optional but recommended)
if SYSTEM_CONFIG.enable_frontier_ai
    @info "Loading FRONTIER AI modules (PINNs, TDA, GNN)..."
    @safe_include "DarwinScaffoldStudio/Science/PINNs.jl" "PINNs"
    @safe_include "DarwinScaffoldStudio/Science/TDA.jl" "TDA"
    @safe_include "DarwinScaffoldStudio/Science/GraphNeuralNetworks.jl" "GNN"
else
    @warn "FRONTIER AI modules disabled. Enable with GlobalConfig(enable_frontier_ai=true)"
end

# SOTA Visualization & Preprocessing (Optional)
if SYSTEM_CONFIG.enable_visualization
    @safe_include "DarwinScaffoldStudio/Visualization/AdvancedPreprocessing.jl" "AdvancedPreprocessing"
    @safe_include "DarwinScaffoldStudio/Visualization/GaussianSplatting.jl" "GaussianSplatting"
    @safe_include "DarwinScaffoldStudio/Visualization/NeRF.jl" "NeRF"
    @safe_include "DarwinScaffoldStudio/Visualization/SAM2Integration.jl" "SAM2"
end

# TRUE 2025 SOTA: Tissue Engineering + Drug Delivery (Optional)
if SYSTEM_CONFIG.enable_frontier_ai
    @safe_include "DarwinScaffoldStudio/Science/DrugDeliveryModeling.jl" "DrugDelivery"
    @safe_include "DarwinScaffoldStudio/Science/AlphaFold3Integration.jl" "AlphaFold3"
end

# ULTRA 2025: Fractal + Biomimetic + Vision (Optional)
if SYSTEM_CONFIG.enable_visualization
    @safe_include "DarwinScaffoldStudio/Science/FractalVascularization.jl" "FractalVascularization"
    @safe_include "DarwinScaffoldStudio/Science/BiomimeticPatterns.jl" "BiomimeticPatterns"
    @safe_include "DarwinScaffoldStudio/Vision/SEMCellIdentification.jl" "SEMCellIdentification"
end

# FRONTIER BEYOND: Advanced Technologies (Optional - disabled by default)
if SYSTEM_CONFIG.enable_advanced_modules
    @info "Loading ADVANCED modules (Quantum, Blockchain, etc.)..."
    @safe_include "DarwinScaffoldStudio/Advanced/QuantumOptimization.jl" "QuantumOptimization"
    @safe_include "DarwinScaffoldStudio/Advanced/OrganOnChip.jl" "OrganOnChip"
    @safe_include "DarwinScaffoldStudio/Advanced/DigitalTwin.jl" "DigitalTwin"
    @safe_include "DarwinScaffoldStudio/Advanced/BlockchainProvenance.jl" "BlockchainProvenance"
    @safe_include "DarwinScaffoldStudio/Advanced/SupercomputingBridge.jl" "SupercomputingBridge"
    @safe_include "DarwinScaffoldStudio/Advanced/EdgeDeployment.jl" "EdgeDeployment"
end

# THEORETICAL DEEP: Category Theory, Information Theory (Optional - disabled by default)
if SYSTEM_CONFIG.enable_advanced_modules
    @safe_include "DarwinScaffoldStudio/Theory/CategoryTheoreticScaffolds.jl" "CategoryTheory"
    @safe_include "DarwinScaffoldStudio/Theory/InformationTheoreticDesign.jl" "InformationTheory"
    @safe_include "DarwinScaffoldStudio/Theory/CausalScaffoldDiscovery.jl" "CausalInference"
    @safe_include "DarwinScaffoldStudio/Theory/SymbolicRegression.jl" "SymbolicRegression"
end

# HAUSEN SPECIAL EDITION: Specialized Research Modules (Optional)
if SYSTEM_CONFIG.enable_advanced_modules
    @safe_include "DarwinScaffoldStudio/Hausen/BioactiveGlassOptimization.jl" "BioactiveGlass"
    @safe_include "DarwinScaffoldStudio/Hausen/AntimicrobialMicrospheres.jl" "AntimicrobialMicrospheres"
    @safe_include "DarwinScaffoldStudio/Hausen/PhytochemicalScaffold.jl" "PhytochemicalScaffold"
end

# UNIFIED PIPELINE: End-to-End Workflow
@safe_include "DarwinScaffoldStudio/Pipeline/DarwinPipeline.jl" "Pipeline"

# TISSUE GROWTH SIMULATION: SOTA 2024 Methods
@safe_include "DarwinScaffoldStudio/Simulation/TissueGrowthSimulator.jl" "TissueGrowthSimulator"

# FOUNDATION MODELS: 2025 SOTA+ (ESM-3, Diffusion, Neural Operators)
if SYSTEM_CONFIG.enable_advanced_modules
    @safe_include "DarwinScaffoldStudio/Foundation/FoundationModels.jl" "FoundationModels"
    @safe_include "DarwinScaffoldStudio/Foundation/DiffusionScaffoldGenerator.jl" "DiffusionGenerator"
    @safe_include "DarwinScaffoldStudio/Foundation/NeuralOperators.jl" "NeuralOperators"
end

# Log module loading summary
ErrorHandling.log_module_status()

# Re-export modules
using .Config: ScaffoldConfig, get_config, GlobalConfig, get_global_config
using .Types: ScaffoldMetrics, OptimizationResults, ScaffoldParameters
using .ImageLoader: load_image
using .Preprocessing: preprocess_image
using .Segmentation: segment_scaffold
using .Metrics: compute_metrics
using .ScaffoldOptimizer: Optimizer, optimize_scaffold, detect_problems
using .Mesh3D: create_mesh, create_mesh_simple
using .Export: export_stl, export_stl_from_binary
using .Topology: compute_kec_metrics
using .Percolation: compute_percolation_metrics
using .ML: predict_viability, predict_failure_load
using .Optimization: optimize_scaffold_thesis

# Export public API
export
    # MicroCT
    load_image,
    preprocess_image,
    compute_metrics,
    segment_scaffold,
    ScaffoldMetrics,
    # Optimization
    Optimizer,
    optimize_scaffold,
    detect_problems,
    ScaffoldParameters,
    OptimizationResults,
    # Visualization
    create_mesh,
    create_mesh_simple,
    export_stl,
    export_stl_from_binary,
    # Core
    ScaffoldConfig,
    get_config,
    GlobalConfig,
    get_global_config,
    # Science (Thesis)
    compute_kec_metrics,
    compute_percolation_metrics,
    predict_viability,
    predict_failure_load,
    optimize_scaffold_thesis

end # module
