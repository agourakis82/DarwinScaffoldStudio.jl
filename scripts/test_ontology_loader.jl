"""
Test script for OntologyLoader
"""

include("../src/DarwinScaffoldStudio/Ontology/OntologyLoader.jl")
using .OntologyLoader

println("=== Testando OntologyLoader ===")

# Listar compostos PubChem disponíveis
println("\nCompostos PubChem disponíveis:")
for c in list_pubchem_compounds()
    println("  - ", c)
end

# Carregar ácido lático
println("\n--- Lactic Acid ---")
lac = load_pubchem_compound("lactic acid")
println("  CID: ", lac.cid)
println("  Formula: ", lac.molecular_formula)
println("  MW: ", lac.molecular_weight, " g/mol")
println("  SMILES: ", lac.smiles)

# Carregar TEC
println("\n--- Triethyl Citrate (TEC) ---")
tec = load_pubchem_compound("triethyl citrate")
println("  CID: ", tec.cid)
println("  Formula: ", tec.molecular_formula)
println("  MW: ", tec.molecular_weight, " g/mol")

# Buscar PLLA no DEBBIE
println("\n--- Buscando PLLA no DEBBIE ---")
show_search_results("PLLA")

# Buscar biomateriais
println("\n--- Listando todos os Biomateriais (DEBBIE) ---")
biomats = get_all_biomaterials()
println("Total: ", length(biomats), " biomateriais")
for b in biomats[1:min(20, length(biomats))]
    aliases = isempty(b.aliases) ? "" : " ($(join(b.aliases, ", ")))"
    println("  - ", b.name, aliases)
end
if length(biomats) > 20
    println("  ... e mais ", length(biomats) - 20, " outros")
end
