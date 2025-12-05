"""
    MaterialLibrary

Comprehensive library of 80+ biomaterials for scaffold fabrication.
Organized by material class with properties.

# Author: Dr. Demetrios Agourakis
"""
module MaterialLibrary

using ..OBOFoundry: OBOTerm

export MATERIALS, MATERIALS_BY_CLASS
export get_material, list_materials
export MATERIAL_PROPERTIES, get_material_properties

# Helper
M(id, name; def="", syn=String[], par=String[]) = OBOTerm(id, name; definition=def, synonyms=syn, parents=par)

#=============================================================================
  SYNTHETIC POLYMERS - BIODEGRADABLE (15 terms)
=============================================================================#
const SYNTHETIC_BIODEGRADABLE = Dict{String,OBOTerm}(
    "CHEBI:53310" => M("CHEBI:53310", "polycaprolactone"; def="Biodegradable polyester, Tm~60°C, slow degradation 2-4 years", syn=["PCL"], par=["CHEBI:53311"]),
    "CHEBI:53309" => M("CHEBI:53309", "polylactic acid"; def="Biodegradable thermoplastic from lactic acid, 6-24 months", syn=["PLA", "polylactide"], par=["CHEBI:53311"]),
    "CHEBI:53312" => M("CHEBI:53312", "polyglycolic acid"; def="Simplest polyester, fast degradation 2-4 months", syn=["PGA", "polyglycolide"], par=["CHEBI:53311"]),
    "CHEBI:53426" => M("CHEBI:53426", "poly(lactic-co-glycolic acid)"; def="Tunable copolymer, 1-6 months degradation", syn=["PLGA"], par=["CHEBI:53311"]),
    "CHEBI:82689" => M("CHEBI:82689", "poly(L-lactic acid)"; def="Crystalline PLA isomer, slower degradation", syn=["PLLA", "poly-L-lactide"], par=["CHEBI:53309"]),
    "CHEBI:82690" => M("CHEBI:82690", "poly(D,L-lactic acid)"; def="Amorphous PLA, faster degradation", syn=["PDLLA", "poly-DL-lactide"], par=["CHEBI:53309"]),
    "CHEBI:53427" => M("CHEBI:53427", "polyhydroxybutyrate"; def="Bacterial polyester, biocompatible", syn=["PHB", "P3HB"], par=["CHEBI:53311"]),
    "CHEBI:53428" => M("CHEBI:53428", "poly(3-hydroxybutyrate-co-3-hydroxyvalerate)"; def="PHB copolymer with improved flexibility", syn=["PHBV"], par=["CHEBI:53427"]),
    "CHEBI:53429" => M("CHEBI:53429", "polydioxanone"; def="Suture material, 6-12 months degradation", syn=["PDO", "PDS"], par=["CHEBI:53311"]),
    "CHEBI:53430" => M("CHEBI:53430", "polytrimethylene carbonate"; def="Soft elastomeric polymer", syn=["PTMC"], par=["CHEBI:53311"]),
    "CHEBI:81573" => M("CHEBI:81573", "poly(ester urethane)"; def="Elastomeric biodegradable polymer", syn=["PEU"], par=["CHEBI:53311"]),
    "CHEBI:53431" => M("CHEBI:53431", "polyorthoesters"; def="Surface-eroding polymer for drug delivery", syn=["POE"], par=["CHEBI:53311"]),
    "CHEBI:53432" => M("CHEBI:53432", "polyanhydrides"; def="Fast surface erosion polymer", par=["CHEBI:53311"]),
    "CHEBI:53433" => M("CHEBI:53433", "polyphosphazenes"; def="Inorganic backbone polymer", par=["CHEBI:53311"]),
    "CHEBI:53434" => M("CHEBI:53434", "poly(propylene fumarate)"; def="Injectable bone scaffold polymer", syn=["PPF"], par=["CHEBI:53311"]),
)

#=============================================================================
  SYNTHETIC POLYMERS - NON-BIODEGRADABLE (10 terms)
=============================================================================#
const SYNTHETIC_PERMANENT = Dict{String,OBOTerm}(
    "CHEBI:46793" => M("CHEBI:46793", "polyethylene glycol"; def="Hydrophilic polymer, PEGylation", syn=["PEG", "polyethylene oxide", "PEO"], par=["CHEBI:36080"]),
    "CHEBI:53435" => M("CHEBI:53435", "polyethylene"; def="Most common plastic, implants", syn=["PE", "HDPE", "UHMWPE"], par=["CHEBI:36080"]),
    "CHEBI:53436" => M("CHEBI:53436", "polytetrafluoroethylene"; def="Non-stick polymer, vascular grafts", syn=["PTFE", "Teflon", "ePTFE"], par=["CHEBI:36080"]),
    "CHEBI:53437" => M("CHEBI:53437", "polyurethane"; def="Versatile polymer, tunable properties", syn=["PU", "PUR"], par=["CHEBI:36080"]),
    "CHEBI:53438" => M("CHEBI:53438", "poly(methyl methacrylate)"; def="Bone cement, acrylic", syn=["PMMA", "acrylic"], par=["CHEBI:36080"]),
    "CHEBI:53439" => M("CHEBI:53439", "polydimethylsiloxane"; def="Silicone rubber, implants", syn=["PDMS", "silicone"], par=["CHEBI:36080"]),
    "CHEBI:53440" => M("CHEBI:53440", "polypropylene"; def="Sutures, meshes", syn=["PP"], par=["CHEBI:36080"]),
    "CHEBI:53441" => M("CHEBI:53441", "polyvinyl alcohol"; def="Hydrogel former, water-soluble", syn=["PVA", "PVOH"], par=["CHEBI:36080"]),
    "CHEBI:53442" => M("CHEBI:53442", "polyacrylamide"; def="Hydrogel, electrophoresis", syn=["PAM", "PAA"], par=["CHEBI:36080"]),
    "CHEBI:53443" => M("CHEBI:53443", "poly(N-isopropylacrylamide)"; def="Thermoresponsive polymer, LCST 32°C", syn=["pNIPAM", "PNIPAAm"], par=["CHEBI:36080"]),
)

#=============================================================================
  NATURAL POLYMERS - PROTEINS (12 terms)
=============================================================================#
const NATURAL_PROTEINS = Dict{String,OBOTerm}(
    "CHEBI:3815" => M("CHEBI:3815", "collagen"; def="Main ECM structural protein, 28 types", syn=["type I collagen"], par=["CHEBI:36080"]),
    "CHEBI:3816" => M("CHEBI:3816", "collagen type II"; def="Cartilage collagen", par=["CHEBI:3815"]),
    "CHEBI:3817" => M("CHEBI:3817", "collagen type III"; def="Reticular fiber collagen", par=["CHEBI:3815"]),
    "CHEBI:3818" => M("CHEBI:3818", "collagen type IV"; def="Basement membrane collagen", par=["CHEBI:3815"]),
    "CHEBI:28512" => M("CHEBI:28512", "gelatin"; def="Hydrolyzed collagen, gel-forming", syn=["gelatine"], par=["CHEBI:3815"]),
    "CHEBI:6339" => M("CHEBI:6339", "methacrylated gelatin"; def="Photo-crosslinkable gelatin", syn=["GelMA"], par=["CHEBI:28512"]),
    "CHEBI:18237" => M("CHEBI:18237", "fibrin"; def="Blood clotting protein, wound healing", par=["CHEBI:36080"]),
    "CHEBI:8192" => M("CHEBI:8192", "fibronectin"; def="Cell adhesion glycoprotein, RGD", par=["CHEBI:36080"]),
    "CHEBI:28790" => M("CHEBI:28790", "laminin"; def="Basement membrane protein, neurite guidance", par=["CHEBI:36080"]),
    "CHEBI:17632" => M("CHEBI:17632", "elastin"; def="Elastic protein of connective tissue", par=["CHEBI:36080"]),
    "CHEBI:58534" => M("CHEBI:58534", "silk fibroin"; def="Silk protein, strong biodegradable", syn=["silk"], par=["CHEBI:36080"]),
    "CHEBI:10545" => M("CHEBI:10545", "keratin"; def="Hair, nail, skin protein", par=["CHEBI:36080"]),
)

#=============================================================================
  NATURAL POLYMERS - POLYSACCHARIDES (12 terms)
=============================================================================#
const NATURAL_POLYSACCHARIDES = Dict{String,OBOTerm}(
    "CHEBI:18154" => M("CHEBI:18154", "hyaluronic acid"; def="GAG of ECM, hydrogel former, viscoelastic", syn=["hyaluronan", "HA"], par=["CHEBI:37395"]),
    "CHEBI:16991" => M("CHEBI:16991", "methacrylated hyaluronic acid"; def="Photo-crosslinkable HA", syn=["MeHA", "HAMA"], par=["CHEBI:18154"]),
    "CHEBI:16737" => M("CHEBI:16737", "chitosan"; def="Deacetylated chitin, antimicrobial", par=["CHEBI:36973"]),
    "CHEBI:52747" => M("CHEBI:52747", "alginate"; def="Brown algae polysaccharide, Ca2+ crosslinked", syn=["alginic acid", "sodium alginate"], par=["CHEBI:36973"]),
    "CHEBI:27081" => M("CHEBI:27081", "chondroitin sulfate"; def="GAG of cartilage", syn=["CS"], par=["CHEBI:37395"]),
    "CHEBI:24658" => M("CHEBI:24658", "heparin"; def="Anticoagulant GAG", par=["CHEBI:37395"]),
    "CHEBI:28815" => M("CHEBI:28815", "cellulose"; def="Plant cell wall polymer", par=["CHEBI:36973"]),
    "CHEBI:53472" => M("CHEBI:53472", "bacterial cellulose"; def="Microbial cellulose, high purity", syn=["BC"], par=["CHEBI:28815"]),
    "CHEBI:28087" => M("CHEBI:28087", "dextran"; def="Bacterial polysaccharide", par=["CHEBI:36973"]),
    "CHEBI:17632" => M("CHEBI:17632", "agarose"; def="Seaweed polysaccharide, gel-forming", par=["CHEBI:36973"]),
    "CHEBI:24632" => M("CHEBI:24632", "pullulan"; def="Fungal polysaccharide", par=["CHEBI:36973"]),
    "CHEBI:27385" => M("CHEBI:27385", "starch"; def="Plant storage polysaccharide", par=["CHEBI:36973"]),
)

#=============================================================================
  CERAMICS - CALCIUM PHOSPHATES (10 terms)
=============================================================================#
const CALCIUM_PHOSPHATES = Dict{String,OBOTerm}(
    "CHEBI:52251" => M("CHEBI:52251", "hydroxyapatite"; def="Ca10(PO4)6(OH)2, bone mineral, Ca/P=1.67", syn=["HA", "HAp"], par=["CHEBI:37586"]),
    "CHEBI:53480" => M("CHEBI:53480", "tricalcium phosphate"; def="Ca3(PO4)2, resorbable ceramic, Ca/P=1.5", syn=["TCP", "beta-TCP", "alpha-TCP"], par=["CHEBI:37586"]),
    "CHEBI:53481" => M("CHEBI:53481", "biphasic calcium phosphate"; def="HA + TCP mixture, tunable resorption", syn=["BCP"], par=["CHEBI:37586"]),
    "CHEBI:53482" => M("CHEBI:53482", "octacalcium phosphate"; def="OCP, bone precursor phase", syn=["OCP"], par=["CHEBI:37586"]),
    "CHEBI:53483" => M("CHEBI:53483", "dicalcium phosphate"; def="CaHPO4, brushite precursor", syn=["DCP", "DCPD", "brushite"], par=["CHEBI:37586"]),
    "CHEBI:53484" => M("CHEBI:53484", "amorphous calcium phosphate"; def="ACP, bone cement precursor", syn=["ACP"], par=["CHEBI:37586"]),
    "CHEBI:53485" => M("CHEBI:53485", "calcium-deficient hydroxyapatite"; def="More resorbable than stoichiometric HA", syn=["CDHA"], par=["CHEBI:52251"]),
    "CHEBI:53486" => M("CHEBI:53486", "carbonated hydroxyapatite"; def="CO3 substituted HA, more biological", syn=["CHA"], par=["CHEBI:52251"]),
    "CHEBI:53487" => M("CHEBI:53487", "fluorapatite"; def="Ca10(PO4)6F2, dental applications", syn=["FA", "FAp"], par=["CHEBI:37586"]),
    "CHEBI:53488" => M("CHEBI:53488", "whitlockite"; def="Ca18Mg2(HPO4)2(PO4)12, bone mineral phase", syn=["beta-TCP with Mg"], par=["CHEBI:53480"]),
)

#=============================================================================
  CERAMICS - BIOACTIVE GLASSES (8 terms)
=============================================================================#
const BIOACTIVE_GLASSES = Dict{String,OBOTerm}(
    "CHEBI:52254" => M("CHEBI:52254", "45S5 bioactive glass"; def="SiO2-Na2O-CaO-P2O5, Hench's glass", syn=["bioglass", "45S5", "Bioglass"], par=["CHEBI:33416"]),
    "CHEBI:52255" => M("CHEBI:52255", "13-93 bioactive glass"; def="Higher SiO2, fiber-drawable", syn=["13-93"], par=["CHEBI:33416"]),
    "CHEBI:52256" => M("CHEBI:52256", "S53P4 bioactive glass"; def="Antibacterial bioactive glass", syn=["BonAlive"], par=["CHEBI:33416"]),
    "CHEBI:52257" => M("CHEBI:52257", "mesoporous bioactive glass"; def="High surface area, drug delivery", syn=["MBG"], par=["CHEBI:33416"]),
    "CHEBI:52258" => M("CHEBI:52258", "sol-gel bioactive glass"; def="Lower processing temperature", syn=["sol-gel BG"], par=["CHEBI:33416"]),
    "CHEBI:52259" => M("CHEBI:52259", "borate bioactive glass"; def="B2O3-based, faster dissolution", syn=["borate glass"], par=["CHEBI:33416"]),
    "CHEBI:52260" => M("CHEBI:52260", "phosphate bioactive glass"; def="P2O5-based, fully resorbable", syn=["phosphate glass"], par=["CHEBI:33416"]),
    "CHEBI:52261" => M("CHEBI:52261", "bioactive glass-ceramic"; def="Partially crystallized BG", syn=["glass-ceramic", "Ceravital"], par=["CHEBI:33416"]),
)

#=============================================================================
  METALS AND ALLOYS (10 terms)
=============================================================================#
const METALS = Dict{String,OBOTerm}(
    "CHEBI:33341" => M("CHEBI:33341", "titanium"; def="Biocompatible transition metal, osseointegration", syn=["Ti"], par=["CHEBI:33521"]),
    "CHEBI:53489" => M("CHEBI:53489", "Ti-6Al-4V"; def="Titanium alloy, most common implant alloy", syn=["Ti64", "titanium alloy"], par=["CHEBI:33341"]),
    "CHEBI:37926" => M("CHEBI:37926", "titanium dioxide"; def="TiO2, surface oxide, photocatalytic", syn=["titania", "TiO2"], par=["CHEBI:33341"]),
    "CHEBI:53490" => M("CHEBI:53490", "316L stainless steel"; def="Austenitic steel, temporary implants", syn=["316L SS", "surgical steel"], par=["CHEBI:33521"]),
    "CHEBI:53491" => M("CHEBI:53491", "cobalt-chromium alloy"; def="CoCr alloy, orthopedic implants", syn=["CoCr", "CoCrMo"], par=["CHEBI:33521"]),
    "CHEBI:22977" => M("CHEBI:22977", "magnesium"; def="Biodegradable metal, bone-like modulus", syn=["Mg"], par=["CHEBI:33521"]),
    "CHEBI:53492" => M("CHEBI:53492", "magnesium alloy"; def="Mg-based biodegradable alloy", syn=["Mg alloy", "AZ31", "WE43"], par=["CHEBI:22977"]),
    "CHEBI:27363" => M("CHEBI:27363", "zinc"; def="Essential trace element, biodegradable", syn=["Zn"], par=["CHEBI:33521"]),
    "CHEBI:30050" => M("CHEBI:30050", "tantalum"; def="Highly biocompatible, porous implants", syn=["Ta", "Trabecular Metal"], par=["CHEBI:33521"]),
    "CHEBI:28694" => M("CHEBI:28694", "gold"; def="Inert metal, nanoparticles", syn=["Au"], par=["CHEBI:33521"]),
)

#=============================================================================
  GROWTH FACTORS (10 terms)
=============================================================================#
const GROWTH_FACTORS = Dict{String,OBOTerm}(
    "CHEBI:83658" => M("CHEBI:83658", "bone morphogenetic protein 2"; def="Potent osteogenic factor, FDA approved", syn=["BMP-2", "rhBMP-2"], par=["CHEBI:36080"]),
    "CHEBI:83659" => M("CHEBI:83659", "bone morphogenetic protein 7"; def="Osteogenic factor, kidney development", syn=["BMP-7", "OP-1", "osteogenic protein 1"], par=["CHEBI:36080"]),
    "CHEBI:74037" => M("CHEBI:74037", "vascular endothelial growth factor"; def="Angiogenic factor", syn=["VEGF", "VEGF-A"], par=["CHEBI:36080"]),
    "CHEBI:74038" => M("CHEBI:74038", "fibroblast growth factor 2"; def="Mitogenic, angiogenic factor", syn=["FGF-2", "bFGF", "basic FGF"], par=["CHEBI:36080"]),
    "CHEBI:74039" => M("CHEBI:74039", "platelet-derived growth factor"; def="Wound healing, mitogenic", syn=["PDGF", "PDGF-BB"], par=["CHEBI:36080"]),
    "CHEBI:74040" => M("CHEBI:74040", "transforming growth factor beta 1"; def="Multifunctional, chondrogenic", syn=["TGF-β1", "TGFb1"], par=["CHEBI:36080"]),
    "CHEBI:74041" => M("CHEBI:74041", "insulin-like growth factor 1"; def="Anabolic, regenerative", syn=["IGF-1", "somatomedin C"], par=["CHEBI:36080"]),
    "CHEBI:74042" => M("CHEBI:74042", "epidermal growth factor"; def="Epithelialization, wound healing", syn=["EGF"], par=["CHEBI:36080"]),
    "CHEBI:74043" => M("CHEBI:74043", "nerve growth factor"; def="Neurotrophin, neural regeneration", syn=["NGF"], par=["CHEBI:36080"]),
    "CHEBI:74044" => M("CHEBI:74044", "stromal cell-derived factor 1"; def="Stem cell homing factor", syn=["SDF-1", "CXCL12"], par=["CHEBI:36080"]),
)

#=============================================================================
  COMBINED DATABASE
=============================================================================#

"""All 80+ materials combined."""
const MATERIALS = merge(
    SYNTHETIC_BIODEGRADABLE, SYNTHETIC_PERMANENT,
    NATURAL_PROTEINS, NATURAL_POLYSACCHARIDES,
    CALCIUM_PHOSPHATES, BIOACTIVE_GLASSES,
    METALS, GROWTH_FACTORS
)

"""Materials organized by class."""
const MATERIALS_BY_CLASS = Dict{Symbol,Dict{String,OBOTerm}}(
    :synthetic_biodegradable => SYNTHETIC_BIODEGRADABLE,
    :synthetic_permanent => SYNTHETIC_PERMANENT,
    :natural_proteins => NATURAL_PROTEINS,
    :natural_polysaccharides => NATURAL_POLYSACCHARIDES,
    :calcium_phosphates => CALCIUM_PHOSPHATES,
    :bioactive_glasses => BIOACTIVE_GLASSES,
    :metals => METALS,
    :growth_factors => GROWTH_FACTORS,
)

#=============================================================================
  MATERIAL PROPERTIES DATABASE
=============================================================================#

"""Material properties for scaffold design (units standardized)."""
const MATERIAL_PROPERTIES = Dict{String,NamedTuple}(
    # Synthetic biodegradable polymers
    "CHEBI:53310" => (name="PCL", elastic_modulus_mpa=400, tensile_strength_mpa=25,
        degradation_months=24, melting_temp_c=60, glass_temp_c=-60),
    "CHEBI:53309" => (name="PLA", elastic_modulus_mpa=3500, tensile_strength_mpa=60,
        degradation_months=18, melting_temp_c=175, glass_temp_c=60),
    "CHEBI:53312" => (name="PGA", elastic_modulus_mpa=7000, tensile_strength_mpa=70,
        degradation_months=3, melting_temp_c=225, glass_temp_c=35),
    "CHEBI:53426" => (name="PLGA", elastic_modulus_mpa=2000, tensile_strength_mpa=45,
        degradation_months=6, melting_temp_c=200, glass_temp_c=50),
    # Ceramics
    "CHEBI:52251" => (name="HA", elastic_modulus_mpa=80000, compressive_strength_mpa=500,
        degradation_months=120, ca_p_ratio=1.67),
    "CHEBI:53480" => (name="TCP", elastic_modulus_mpa=30000, compressive_strength_mpa=150,
        degradation_months=12, ca_p_ratio=1.50),
    "CHEBI:52254" => (name="45S5", elastic_modulus_mpa=35000, compressive_strength_mpa=500,
        degradation_months=6, sio2_percent=45),
    # Metals
    "CHEBI:33341" => (name="Ti", elastic_modulus_mpa=110000, yield_strength_mpa=275,
        fatigue_limit_mpa=300, corrosion_rate=0),
    "CHEBI:53489" => (name="Ti64", elastic_modulus_mpa=114000, yield_strength_mpa=830,
        fatigue_limit_mpa=500, corrosion_rate=0),
    # Natural polymers
    "CHEBI:3815" => (name="Collagen", elastic_modulus_mpa=5, tensile_strength_mpa=1,
        degradation_months=3, enzymatic=true),
    "CHEBI:18154" => (name="HA", elastic_modulus_kpa=5, viscosity_pas=1000,
        degradation_months=1, enzymatic=true),
)

#=============================================================================
  LOOKUP FUNCTIONS
=============================================================================#

"""Get material by CHEBI ID."""
get_material(id::String) = get(MATERIALS, id, nothing)

"""List materials by class."""
function list_materials(mat_class::Symbol=:all)
    mat_class == :all ? collect(values(MATERIALS)) :
    haskey(MATERIALS_BY_CLASS, mat_class) ? collect(values(MATERIALS_BY_CLASS[mat_class])) : OBOTerm[]
end

"""Get material properties."""
get_material_properties(id::String) = get(MATERIAL_PROPERTIES, id, nothing)

end # module
