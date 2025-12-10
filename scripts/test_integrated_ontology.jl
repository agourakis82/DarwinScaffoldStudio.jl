"""
Test the integrated ontology system.
"""

include("../src/DarwinScaffoldStudio/Ontology/IntegratedOntology.jl")
using .IntegratedOntology

println("="^70)
println("     DARWIN SCAFFOLD STUDIO - INTEGRATED ONTOLOGY SYSTEM")
println("="^70)

# Show all available materials
show_all_materials()

# Query specific materials
println("\n\n" * "="^70)
println("QUERYING SPECIFIC MATERIALS")
println("="^70)

# PLDLA from local KB
show_material("PLDLA 70/30")

# PLLA from local + DEBBIE
show_material("PLLA")

# Get model parameters
println("\n\n" * "="^70)
println("DEGRADATION MODEL PARAMETERS")
println("="^70)

for mat in ["PLDLA", "PLLA", "PLA"]
    println("\n--- $mat ---")
    try
        params = get_degradation_model_params(mat)
        println("  k = $(params.k_hydrolysis) day⁻¹ [$(params.k_source)]")
        println("  Ea = $(params.Ea) kJ/mol")
        println("  Tg∞ = $(params.Tg_infinity) °C")
        println("  Found in: $(params.found_in)")
    catch e
        println("  Error: $e")
    end
end

# Query external-only material
println("\n\n--- PCL (external only) ---")
try
    params = get_degradation_model_params("PCL")
    println("  Found in: $(params.found_in)")
    println("  Note: $(params.k_source)")
catch e
    println("  $e")
end

println("\n" * "="^70)
println("ONTOLOGY SYSTEM READY")
println("="^70)
