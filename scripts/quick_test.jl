# Quick Test - Verify dev_load.jl works

include("dev_load.jl")

println("="^60)
println("Testing DARWIN Scaffold Studio (Dev Mode)")
println("="^60)

# Test 1: Generate synthetic scaffold
println("\n📊 Test 1: Synthetic Scaffold Generation")
volume, metadata = DataIngestion.generate_synthetic_scaffold(
    size_voxels=(50, 50, 50),
    porosity=0.75
)
println("  ✓ Generated $(metadata["dimensions"]) scaffold")
println("  ✓ Porosity: $(round(metadata["actual_porosity"], digits=3))")

# Test 2: Compute metrics
println("\n📊 Test 2: Metrics Computation")
metrics = Metrics.compute_metrics(volume, 10.0)
println("  ✓ Porosity: $(round(metrics.porosity, digits=3))")
println("  ✓ Mean pore size: $(round(metrics.mean_pore_size_um, digits=1)) μm")
println("  ✓ Interconnectivity: $(round(metrics.interconnectivity, digits=3))")
println("  ✓ Tortuosity: $(round(metrics.tortuosity, digits=2))")
println("  ✓ Permeability: $(round(metrics.permeability, sigdigits=3)) m²")

# Test 3: ML predictions 
println("\n📊 Test 3: ML Viability Prediction")
viability = ML.predict_viability(metrics)
println("  ✓ Predicted viability: $(round(viability, digits=3))")

# Test 4: Configuration
println("\n📊 Test 4: Configuration System")
global_config = Config.get_global_config()
println("  ✓ Data directory: $(global_config.data_directory)")
println("  ✓ Results directory: $(global_config.results_directory)")
println("  ✓ FRONTIER AI enabled: $(global_config.enable_frontier_ai)")
println("  ✓ Debug mode: $(global_config.debug_mode)")

# Test 5: Agent creation
println("\n📊 Test 5: Agent Framework")
design_agent = DesignAgent.create_design_agent()
analysis_agent = AnalysisAgent.create_analysis_agent()
synthesis_agent = SynthesisAgent.create_synthesis_agent()
println("  ✓ Design Agent created: $(design_agent.name)")  
println("  ✓ Analysis Agent created: $(analysis_agent.name)")
println("  ✓ Synthesis Agent created: $(synthesis_agent.name)")
println("  ✓ Total tools available: $(length(design_agent.tools) + length(analysis_agent.tools) + length(synthesis_agent.tools))")

println("\n" * "="^60)
println("✅ All core tests passed! System ready for development.")
println("="^60)
println("\n💡 Next: Implement real Ollama integration for agents")
