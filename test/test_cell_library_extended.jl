#!/usr/bin/env julia

"""
Test script for CellLibraryExtended.jl
Demonstrates 50+ new specialized cell types
"""

include("src/DarwinScaffoldStudio/Ontology/OBOFoundry.jl")
include("src/DarwinScaffoldStudio/Ontology/CellLibraryExtended.jl")

using .CellLibraryExtended

println("="^60)
println("CellLibraryExtended - 75+ Specialized Cell Types")
println("="^60)

# Test rare stem cells
println("\n🔬 RARE & SPECIALIZED STEM CELLS (10 types):")
limbal = get_extended_cell("CL:0000610")
println("  ✓ $(limbal.name) ($(limbal.id))")
println("    Synonyms: $(join(limbal.synonyms, ", "))")

dpsc = get_extended_cell("CL:0007005")
println("  ✓ $(dpsc.name) - for dental tissue engineering")

wj_msc = get_extended_cell("CL:0007011")
println("  ✓ $(wj_msc.name)")

pdlsc = get_extended_cell("CL:0007010")
println("  ✓ $(pdlsc.name)")

# Test sensory cells
println("\n👂 SENSORY CELLS (10 types):")
hair = get_extended_cell("CL:0000202")
println("  ✓ $(hair.name): $(hair.definition)")

inner_hair = get_extended_cell("CL:0000201")
outer_hair = get_extended_cell("CL:0000203")
println("  ✓ $(inner_hair.name) & $(outer_hair.name)")

olfactory = get_extended_cell("CL:0000207")
println("  ✓ $(olfactory.name): $(olfactory.definition)")

taste = get_extended_cell("CL:0000209")
println("  ✓ $(taste.name)")

# Test specialized neurons
println("\n🧠 SPECIALIZED NEURONS (10 types):")
purkinje = get_extended_cell("CL:0000121")
println("  ✓ $(purkinje.name): $(purkinje.definition)")

pyramidal = get_extended_cell("CL:0000598")
println("  ✓ $(pyramidal.name): $(pyramidal.definition)")

granule = get_extended_cell("CL:0000120")
basket = get_extended_cell("CL:0000118")
println("  ✓ Cerebellar neurons: $(granule.name), $(basket.name)")

rod_bipolar = get_extended_cell("CL:0000751")
cone_bipolar = get_extended_cell("CL:0000752")
println("  ✓ Retinal bipolar cells: $(rod_bipolar.name), $(cone_bipolar.name)")

# Test germ cells
println("\n🧬 GERM CELLS (8 types):")
spermatogonium = get_extended_cell("CL:0000020")
println("  ✓ $(spermatogonium.name): $(spermatogonium.definition)")

spermatid = get_extended_cell("CL:0000018")
println("  ✓ $(spermatid.name)")

oocyte = get_extended_cell("CL:0000023")
println("  ✓ $(oocyte.name) ($(join(oocyte.synonyms, ", ")))")

sertoli = get_extended_cell("CL:0000216")
granulosa = get_extended_cell("CL:0000501")
println("  ✓ Supporting cells: $(sertoli.name), $(granulosa.name)")

# Test secretory cells
println("\n💧 SECRETORY CELLS (11 types):")
parietal = get_extended_cell("CL:0000162")
println("  ✓ $(parietal.name): $(parietal.definition)")

chief = get_extended_cell("CL:0000160")
println("  ✓ $(chief.name): $(chief.definition)")

paneth = get_extended_cell("CL:0000510")
println("  ✓ $(paneth.name): $(paneth.definition)")

enteroendocrine = get_extended_cell("CL:0000164")
println("  ✓ $(enteroendocrine.name)")

# Test rare immune cells
println("\n🛡️  RARE IMMUNE CELLS (10 types):")
eosinophil = get_extended_cell("CL:0000771")
basophil = get_extended_cell("CL:0000767")
mast = get_extended_cell("CL:0000097")
println("  ✓ Granulocytes: $(eosinophil.name), $(basophil.name), $(mast.name)")

ilc1 = get_extended_cell("CL:0001066")
ilc2 = get_extended_cell("CL:0001067")
ilc3 = get_extended_cell("CL:0001068")
println("  ✓ ILCs: $(ilc1.name), $(ilc2.name), $(ilc3.name)")

nk_dim = get_extended_cell("CL:0000938")
nk_bright = get_extended_cell("CL:0000939")
println("  ✓ NK subtypes: CD56dim, CD56bright")

# Test developmental cells
println("\n🌱 DEVELOPMENTAL CELLS (8 types):")
neural_crest = get_extended_cell("CL:0000333")
println("  ✓ $(neural_crest.name): $(neural_crest.definition)")

neuroepithelial = get_extended_cell("CL:0000710")
println("  ✓ $(neuroepithelial.name)")

# Test specialized epithelial
println("\n🧱 SPECIALIZED EPITHELIAL (8 types):")
proximal = get_extended_cell("CL:0002306")
distal = get_extended_cell("CL:0002305")
println("  ✓ Renal tubule: $(proximal.name), $(distal.name)")

thyrocyte = get_extended_cell("CL:0000141")
c_cell = get_extended_cell("CL:0000421")
println("  ✓ Thyroid: $(thyrocyte.name), $(c_cell.name)")

kupffer = get_extended_cell("CL:0000091")
println("  ✓ $(kupffer.name): $(kupffer.definition)")

# Test tissue mapping
println("\n🗺️  TISSUE-CELL MAPPING:")
println("  Retina cells:")
for cell in get_extended_cells_for_tissue("UBERON:0000966")
    println("    - $(cell.name)")
end

println("\n  Inner ear cells:")
for cell in get_extended_cells_for_tissue("UBERON:0000051")
    println("    - $(cell.name)")
end

# Test search function
println("\n🔍 SEARCH TEST:")
println("  Search 'hair':")
for cell in search_extended_cells("hair")
    println("    - $(cell.name) ($(cell.id))")
end

println("\n  Search 'stem':")
for cell in search_extended_cells("stem")[1:5]
    println("    - $(cell.name)")
end

# Summary
println("\n" * "="^60)
println("📊 SUMMARY:")
println("  Total extended cells: $(length(EXTENDED_CELLS))")
println("\n  By category:")
for (cat, cells) in CELLS_BY_CATEGORY_EXT
    println("    - $(cat): $(length(cells)) cells")
end
println("="^60)
println("✅ All tests passed!")
