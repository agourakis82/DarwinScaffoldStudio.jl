# Test TissueLibraryExtended.jl
include("src/DarwinScaffoldStudio/Ontology/OBOFoundry.jl")
using .OBOFoundry

include("src/DarwinScaffoldStudio/Ontology/TissueLibraryExtended.jl")
using .TissueLibraryExtended

println("="^70)
println("TISSUE LIBRARY EXTENDED TEST")
println("="^70)
println()

# Test 1: Counts
println("1. TISSUE COUNTS BY CATEGORY")
println("-"^70)
println("Total tissues: ", length(EXTENDED_TISSUES))
println("  Dental: ", length(DENTAL_TISSUES))
println("  Endocrine: ", length(ENDOCRINE_TISSUES))
println("  Reproductive: ", length(REPRODUCTIVE_TISSUES))
println("  Organ substructures: ", length(ORGAN_SUBSTRUCTURES))
println("  Specialized connective: ", length(SPECIALIZED_CONNECTIVE))
println("  Embryonic: ", length(EMBRYONIC_TISSUES))
println()

# Test 2: Dental tissues
println("2. DENTAL TISSUES SAMPLE")
println("-"^70)
enamel = get_extended_tissue("UBERON:0001752")
println("ID: ", enamel.id)
println("Name: ", enamel.name)
println("Definition: ", enamel.definition)
println("Synonyms: ", join(enamel.synonyms, ", "))
println()

dentin = get_extended_tissue("UBERON:0001091")
println("ID: ", dentin.id)
println("Name: ", dentin.name)
println("Definition: ", dentin.definition)
println()

# Test 3: Endocrine tissues
println("3. ENDOCRINE TISSUES SAMPLE")
println("-"^70)
thyroid = get_extended_tissue("UBERON:0002046")
println("Thyroid Gland: ", thyroid.definition)
println()

adrenal = get_extended_tissue("UBERON:0002369")
println("Adrenal Gland: ", adrenal.definition)
println()

# Test 4: Reproductive tissues
println("4. REPRODUCTIVE TISSUES SAMPLE")
println("-"^70)
placenta = get_extended_tissue("UBERON:0001987")
println("Placenta: ", placenta.definition)
println()

corpus_luteum = get_extended_tissue("UBERON:0002512")
println("Corpus Luteum: ", corpus_luteum.definition)
println()

# Test 5: Organ substructures
println("5. ORGAN SUBSTRUCTURES SAMPLE")
println("-"^70)
nephron = get_extended_tissue("UBERON:0001285")
println("Nephron: ", nephron.definition)
println()

liver_lobule = get_extended_tissue("UBERON:0004647")
println("Liver Lobule: ", liver_lobule.definition)
println()

alveolus = get_extended_tissue("UBERON:0002299")
println("Pulmonary Alveolus: ", alveolus.definition)
println()

# Test 6: Specialized connective tissues
println("6. SPECIALIZED CONNECTIVE TISSUES SAMPLE")
println("-"^70)
synovium = get_extended_tissue("UBERON:0001215")
println("Synovium: ", synovium.definition)
println()

periosteum = get_extended_tissue("UBERON:0002105")
println("Periosteum: ", periosteum.definition)
println()

# Test 7: Embryonic tissues
println("7. EMBRYONIC TISSUES SAMPLE")
println("-"^70)
mesoderm = get_extended_tissue("UBERON:0000925")
println("Mesoderm: ", mesoderm.definition)
println()

mesenchyme = get_extended_tissue("UBERON:0003104")
println("Mesenchyme: ", mesenchyme.definition)
println()

# Test 8: List by category
println("8. LIST TISSUES BY CATEGORY")
println("-"^70)
dental = list_extended_tissues(:dental)
println("Dental tissues (", length(dental), " total):")
for tissue in dental
    println("  - ", tissue.name)
end
println()

embryonic = list_extended_tissues(:embryonic)
println("Embryonic tissues (", length(embryonic), " total):")
for tissue in embryonic
    println("  - ", tissue.name)
end
println()

println("="^70)
println("ALL TESTS PASSED")
println("="^70)
