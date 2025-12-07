"""
    CompatibilityMatrix

Comprehensive compatibility database for tissue engineering.

Contains:
- Material-Material compatibility (composites, coatings)
- Material-Cell compatibility (biocompatibility, adhesion)
- Material-Drug compatibility (loading, stability, release)
- Drug-Drug interactions
- Cell-Cell interactions (co-culture compatibility)
- Fabrication process compatibility
- Sterilization compatibility

Data sources:
- ISO 10993 biocompatibility standards
- Published Q1 literature
- FDA/CE device approval data

Author: Dr. Demetrios Agourakis
"""
module CompatibilityMatrix

export CompatibilityScore, CompatibilityEntry
export MATERIAL_CELL_COMPATIBILITY, MATERIAL_DRUG_COMPATIBILITY
export MATERIAL_MATERIAL_COMPATIBILITY, FABRICATION_COMPATIBILITY
export STERILIZATION_COMPATIBILITY
export check_compatibility, get_compatibility_score
export find_compatible_materials, find_optimal_combination
export generate_compatibility_report

# =============================================================================
# Compatibility Scoring
# =============================================================================

"""Compatibility score with evidence level."""
struct CompatibilityScore
    score::Float64          # 0-1 (0=incompatible, 1=excellent)
    rating::Symbol          # :excellent, :good, :moderate, :poor, :incompatible
    evidence_level::Symbol  # :clinical, :in_vivo, :in_vitro, :theoretical
    notes::String
    references::Vector{String}
end

function CompatibilityScore(score::Float64;
    notes::String="",
    evidence_level::Symbol=:in_vitro,
    references::Vector{String}=String[])

    rating = if score >= 0.9
        :excellent
    elseif score >= 0.7
        :good
    elseif score >= 0.5
        :moderate
    elseif score >= 0.3
        :poor
    else
        :incompatible
    end

    CompatibilityScore(score, rating, evidence_level, notes, references)
end

# =============================================================================
# Material-Cell Compatibility Matrix
# =============================================================================

"""
Material-Cell compatibility database.
Key format: (material_id, cell_type) => CompatibilityScore
"""
const MATERIAL_CELL_COMPATIBILITY = Dict{Tuple{String,Symbol},CompatibilityScore}(
    # =========================================================================
    # PCL with various cell types
    # =========================================================================
    ("PCL", :osteoblast) => CompatibilityScore(0.85;
        notes="Good adhesion, proliferation; surface modification improves outcomes",
        evidence_level=:in_vivo,
        references=["Woodruff 2010 Prog Polym Sci", "Cipitria 2011 Biomaterials"]
    ),
    ("PCL", :chondrocyte) => CompatibilityScore(0.75;
        notes="Adequate for cartilage; benefits from surface treatment",
        evidence_level=:in_vitro,
        references=["Li 2005 Biomaterials"]
    ),
    ("PCL", :fibroblast) => CompatibilityScore(0.90;
        notes="Excellent compatibility, widely used",
        evidence_level=:clinical,
        references=["Guarino 2007 Biomaterials"]
    ),
    ("PCL", :msc) => CompatibilityScore(0.85;
        notes="Supports stemness and differentiation",
        evidence_level=:in_vivo,
        references=["Dash 2011 Mol Pharm"]
    ),
    ("PCL", :endothelial) => CompatibilityScore(0.70;
        notes="Moderate; surface modification recommended",
        evidence_level=:in_vitro
    ),
    ("PCL", :neuron) => CompatibilityScore(0.65;
        notes="Requires surface functionalization for neural applications",
        evidence_level=:in_vitro
    ),

    # =========================================================================
    # PLA with various cell types
    # =========================================================================
    ("PLA", :osteoblast) => CompatibilityScore(0.80;
        notes="Good compatibility; acidic degradation products concern",
        evidence_level=:in_vivo,
        references=["Athanasiou 1996 Biomaterials"]
    ),
    ("PLA", :chondrocyte) => CompatibilityScore(0.70;
        notes="Acidic degradation may affect chondrocyte phenotype",
        evidence_level=:in_vitro
    ),
    ("PLA", :fibroblast) => CompatibilityScore(0.85;
        notes="Well-tolerated",
        evidence_level=:in_vivo
    ),
    ("PLA", :msc) => CompatibilityScore(0.80;
        notes="Supports differentiation",
        evidence_level=:in_vitro
    ),

    # =========================================================================
    # PLGA with various cell types
    # =========================================================================
    ("PLGA", :osteoblast) => CompatibilityScore(0.85;
        notes="Widely used for bone; tunable degradation",
        evidence_level=:clinical,
        references=["Lanao 2013 Biomaterials"]
    ),
    ("PLGA", :chondrocyte) => CompatibilityScore(0.75;
        notes="Acceptable for cartilage applications",
        evidence_level=:in_vivo
    ),
    ("PLGA", :fibroblast) => CompatibilityScore(0.85;
        notes="Excellent biocompatibility",
        evidence_level=:in_vivo
    ),
    ("PLGA", :msc) => CompatibilityScore(0.85;
        notes="Standard scaffold material for MSC culture",
        evidence_level=:in_vivo
    ),
    ("PLGA", :neuron) => CompatibilityScore(0.70;
        notes="Used in nerve conduits",
        evidence_level=:in_vivo
    ),

    # =========================================================================
    # Collagen with various cell types
    # =========================================================================
    ("Collagen", :osteoblast) => CompatibilityScore(0.95;
        notes="Native ECM component, excellent compatibility",
        evidence_level=:clinical,
        references=["Glowacki 2008 Biopolymers"]
    ),
    ("Collagen", :chondrocyte) => CompatibilityScore(0.95;
        notes="Optimal for cartilage regeneration",
        evidence_level=:clinical
    ),
    ("Collagen", :fibroblast) => CompatibilityScore(0.98;
        notes="Native environment, highest compatibility",
        evidence_level=:clinical
    ),
    ("Collagen", :msc) => CompatibilityScore(0.95;
        notes="Excellent for stem cell culture and differentiation",
        evidence_level=:in_vivo
    ),
    ("Collagen", :keratinocyte) => CompatibilityScore(0.95;
        notes="Standard for skin substitutes",
        evidence_level=:clinical
    ),
    ("Collagen", :endothelial) => CompatibilityScore(0.90;
        notes="Supports vascularization",
        evidence_level=:in_vivo
    ),
    ("Collagen", :neuron) => CompatibilityScore(0.85;
        notes="Supports neurite extension",
        evidence_level=:in_vivo
    ),
    ("Collagen", :cardiomyocyte) => CompatibilityScore(0.90;
        notes="Native cardiac ECM component",
        evidence_level=:in_vivo
    ),

    # =========================================================================
    # GelMA with various cell types
    # =========================================================================
    ("GelMA", :osteoblast) => CompatibilityScore(0.90;
        notes="Excellent encapsulation, supports mineralization",
        evidence_level=:in_vivo,
        references=["Nichol 2010 Biomaterials"]
    ),
    ("GelMA", :chondrocyte) => CompatibilityScore(0.95;
        notes="Optimal for cartilage bioprinting",
        evidence_level=:in_vivo,
        references=["Levett 2014 PLoS One"]
    ),
    ("GelMA", :msc) => CompatibilityScore(0.90;
        notes="Supports encapsulation and 3D culture",
        evidence_level=:in_vivo
    ),
    ("GelMA", :endothelial) => CompatibilityScore(0.90;
        notes="Supports vascular network formation",
        evidence_level=:in_vivo
    ),

    # =========================================================================
    # Alginate with various cell types
    # =========================================================================
    ("Alginate", :chondrocyte) => CompatibilityScore(0.90;
        notes="Maintains chondrogenic phenotype",
        evidence_level=:clinical,
        references=["Bonaventure 1994 Exp Cell Res"]
    ),
    ("Alginate", :osteoblast) => CompatibilityScore(0.60;
        notes="Poor cell adhesion without modification",
        evidence_level=:in_vitro
    ),
    ("Alginate", :msc) => CompatibilityScore(0.75;
        notes="RGD modification improves adhesion",
        evidence_level=:in_vitro
    ),
    ("Alginate", :islet) => CompatibilityScore(0.95;
        notes="Standard for islet encapsulation",
        evidence_level=:clinical
    ),

    # =========================================================================
    # Hydroxyapatite with various cell types
    # =========================================================================
    ("Hydroxyapatite", :osteoblast) => CompatibilityScore(0.95;
        notes="Bone mineral composition, optimal for bone",
        evidence_level=:clinical,
        references=["LeGeros 2008 Chem Rev"]
    ),
    ("Hydroxyapatite", :msc) => CompatibilityScore(0.90;
        notes="Promotes osteogenic differentiation",
        evidence_level=:in_vivo
    ),
    ("Hydroxyapatite", :osteoclast) => CompatibilityScore(0.85;
        notes="Supports normal bone remodeling",
        evidence_level=:in_vivo
    ),

    # =========================================================================
    # TCP with various cell types
    # =========================================================================
    ("TCP", :osteoblast) => CompatibilityScore(0.90;
        notes="Resorbable, good for bone regeneration",
        evidence_level=:clinical
    ),
    ("TCP", :msc) => CompatibilityScore(0.85;
        notes="Supports osteogenic differentiation",
        evidence_level=:in_vivo
    ),

    # =========================================================================
    # Bioactive Glass with various cell types
    # =========================================================================
    ("Bioglass_45S5", :osteoblast) => CompatibilityScore(0.95;
        notes="Stimulates bone gene expression",
        evidence_level=:clinical,
        references=["Hench 2006 Science"]
    ),
    ("Bioglass_45S5", :msc) => CompatibilityScore(0.90;
        notes="Ion release promotes osteogenesis",
        evidence_level=:in_vivo
    ),
    ("Bioglass_45S5", :fibroblast) => CompatibilityScore(0.80;
        notes="May stimulate soft tissue as well",
        evidence_level=:in_vitro
    ),

    # =========================================================================
    # Titanium with various cell types
    # =========================================================================
    ("Titanium", :osteoblast) => CompatibilityScore(0.95;
        notes="Gold standard for osseointegration",
        evidence_level=:clinical,
        references=["Branemark 1977"]
    ),
    ("Titanium", :fibroblast) => CompatibilityScore(0.85;
        notes="Good soft tissue integration",
        evidence_level=:clinical
    ),
    ("Ti6Al4V", :osteoblast) => CompatibilityScore(0.90;
        notes="Standard orthopedic implant alloy",
        evidence_level=:clinical
    ),

    # =========================================================================
    # Chitosan with various cell types
    # =========================================================================
    ("Chitosan", :osteoblast) => CompatibilityScore(0.80;
        notes="Antimicrobial, good for bone",
        evidence_level=:in_vivo
    ),
    ("Chitosan", :chondrocyte) => CompatibilityScore(0.85;
        notes="GAG-like structure beneficial",
        evidence_level=:in_vivo
    ),
    ("Chitosan", :keratinocyte) => CompatibilityScore(0.85;
        notes="Good for wound healing",
        evidence_level=:clinical
    ),
    ("Chitosan", :fibroblast) => CompatibilityScore(0.85;
        notes="Promotes wound healing",
        evidence_level=:in_vivo
    ),

    # =========================================================================
    # Hyaluronic Acid with various cell types
    # =========================================================================
    ("Hyaluronic_Acid", :chondrocyte) => CompatibilityScore(0.95;
        notes="Native cartilage ECM component",
        evidence_level=:clinical
    ),
    ("Hyaluronic_Acid", :msc) => CompatibilityScore(0.90;
        notes="Promotes chondrogenic differentiation",
        evidence_level=:in_vivo
    ),
    ("Hyaluronic_Acid", :fibroblast) => CompatibilityScore(0.90;
        notes="Wound healing applications",
        evidence_level=:clinical
    ),
    ("Hyaluronic_Acid", :keratinocyte) => CompatibilityScore(0.90;
        notes="Skin regeneration",
        evidence_level=:clinical
    )
)

# =============================================================================
# Material-Drug Compatibility Matrix
# =============================================================================

const MATERIAL_DRUG_COMPATIBILITY = Dict{Tuple{String,String},CompatibilityScore}(
    # =========================================================================
    # PCL - Drug Combinations
    # =========================================================================
    ("PCL", "vancomycin") => CompatibilityScore(0.85;
        notes="Good release profile, stable in PCL matrix",
        evidence_level=:in_vivo,
        references=["Adams 2009 J Biomed Mater Res"]
    ),
    ("PCL", "gentamicin") => CompatibilityScore(0.90;
        notes="Well-established combination",
        evidence_level=:clinical
    ),
    ("PCL", "dexamethasone") => CompatibilityScore(0.85;
        notes="Sustained release achievable",
        evidence_level=:in_vivo
    ),
    ("PCL", "paclitaxel") => CompatibilityScore(0.90;
        notes="Drug-eluting stents application",
        evidence_level=:clinical
    ),
    ("PCL", "rhBMP2") => CompatibilityScore(0.70;
        notes="Protein may denature during processing",
        evidence_level=:in_vitro
    ),

    # =========================================================================
    # PLGA - Drug Combinations
    # =========================================================================
    ("PLGA", "vancomycin") => CompatibilityScore(0.90;
        notes="Excellent sustained release",
        evidence_level=:clinical
    ),
    ("PLGA", "gentamicin") => CompatibilityScore(0.85;
        notes="Standard antibiotic delivery system",
        evidence_level=:clinical
    ),
    ("PLGA", "dexamethasone") => CompatibilityScore(0.90;
        notes="Widely used combination",
        evidence_level=:clinical
    ),
    ("PLGA", "paclitaxel") => CompatibilityScore(0.90;
        notes="Microspheres for chemoembolization",
        evidence_level=:clinical
    ),
    ("PLGA", "rhBMP2") => CompatibilityScore(0.80;
        notes="Acidic degradation may affect protein",
        evidence_level=:in_vivo
    ),
    ("PLGA", "ciprofloxacin") => CompatibilityScore(0.85;
        notes="Good encapsulation efficiency",
        evidence_level=:in_vivo
    ),

    # =========================================================================
    # Collagen - Drug Combinations
    # =========================================================================
    ("Collagen", "vancomycin") => CompatibilityScore(0.90;
        notes="Fleece/sponge combinations in clinical use",
        evidence_level=:clinical
    ),
    ("Collagen", "gentamicin") => CompatibilityScore(0.95;
        notes="Gentamicin-collagen sponge (Collatamp)",
        evidence_level=:clinical
    ),
    ("Collagen", "rhBMP2") => CompatibilityScore(0.95;
        notes="INFUSE Bone Graft - FDA approved",
        evidence_level=:clinical,
        references=["Burkus 2002 Spine"]
    ),
    ("Collagen", "dexamethasone") => CompatibilityScore(0.80;
        notes="Rapid release without modification",
        evidence_level=:in_vitro
    ),
    ("Collagen", "VEGF") => CompatibilityScore(0.85;
        notes="Heparin binding helps retention",
        evidence_level=:in_vivo
    ),

    # =========================================================================
    # GelMA - Drug Combinations
    # =========================================================================
    ("GelMA", "dexamethasone") => CompatibilityScore(0.85;
        notes="Good encapsulation and release",
        evidence_level=:in_vitro
    ),
    ("GelMA", "rhBMP2") => CompatibilityScore(0.85;
        notes="Gentle UV crosslinking preserves activity",
        evidence_level=:in_vitro
    ),
    ("GelMA", "VEGF") => CompatibilityScore(0.85;
        notes="Supports angiogenic factor delivery",
        evidence_level=:in_vivo
    ),

    # =========================================================================
    # Alginate - Drug Combinations
    # =========================================================================
    ("Alginate", "vancomycin") => CompatibilityScore(0.75;
        notes="Fast release, burst effect common",
        evidence_level=:in_vitro
    ),
    ("Alginate", "rhBMP2") => CompatibilityScore(0.80;
        notes="Gentle conditions preserve protein",
        evidence_level=:in_vivo
    ),
    ("Alginate", "VEGF") => CompatibilityScore(0.80;
        notes="Requires heparin for sustained release",
        evidence_level=:in_vivo
    ),

    # =========================================================================
    # Hydroxyapatite - Drug Combinations
    # =========================================================================
    ("Hydroxyapatite", "vancomycin") => CompatibilityScore(0.85;
        notes="Adsorption-based loading",
        evidence_level=:clinical
    ),
    ("Hydroxyapatite", "gentamicin") => CompatibilityScore(0.90;
        notes="Standard bone cement antibiotic",
        evidence_level=:clinical
    ),
    ("Hydroxyapatite", "alendronate") => CompatibilityScore(0.95;
        notes="Bisphosphonate has affinity for HA",
        evidence_level=:in_vivo
    ),
    ("Hydroxyapatite", "rhBMP2") => CompatibilityScore(0.80;
        notes="Adsorption possible but fast release",
        evidence_level=:in_vivo
    ),

    # =========================================================================
    # TCP - Drug Combinations
    # =========================================================================
    ("TCP", "vancomycin") => CompatibilityScore(0.80;
        notes="Surface adsorption loading",
        evidence_level=:in_vivo
    ),
    ("TCP", "gentamicin") => CompatibilityScore(0.85;
        notes="Good bone antibiotic delivery",
        evidence_level=:clinical
    ),
    ("TCP", "alendronate") => CompatibilityScore(0.90;
        notes="Phosphonate affinity for calcium phosphate",
        evidence_level=:in_vivo
    )
)

# =============================================================================
# Material-Material Compatibility (Composites/Coatings)
# =============================================================================

const MATERIAL_MATERIAL_COMPATIBILITY = Dict{Tuple{String,String},CompatibilityScore}(
    # PCL composites
    ("PCL", "Hydroxyapatite") => CompatibilityScore(0.95;
        notes="Common bone scaffold composite",
        evidence_level=:clinical
    ),
    ("PCL", "TCP") => CompatibilityScore(0.90;
        notes="Good mechanical and biological properties",
        evidence_level=:in_vivo
    ),
    ("PCL", "Bioglass_45S5") => CompatibilityScore(0.85;
        notes="Bioactive composite",
        evidence_level=:in_vivo
    ),
    ("PCL", "Collagen") => CompatibilityScore(0.85;
        notes="Coating or blend",
        evidence_level=:in_vivo
    ),
    ("PCL", "Chitosan") => CompatibilityScore(0.80;
        notes="Antimicrobial composite",
        evidence_level=:in_vitro
    ),

    # PLA composites
    ("PLA", "Hydroxyapatite") => CompatibilityScore(0.90;
        notes="Bone scaffold composite",
        evidence_level=:in_vivo
    ),
    ("PLA", "TCP") => CompatibilityScore(0.90;
        notes="Resorbable bone composite",
        evidence_level=:in_vivo
    ),
    ("PLA", "Bioglass_45S5") => CompatibilityScore(0.80;
        notes="Processing temperature concerns",
        evidence_level=:in_vitro
    ),

    # PLGA composites
    ("PLGA", "Hydroxyapatite") => CompatibilityScore(0.90;
        notes="Widely used bone composite",
        evidence_level=:clinical
    ),
    ("PLGA", "TCP") => CompatibilityScore(0.90;
        notes="Tunable degradation",
        evidence_level=:in_vivo
    ),
    ("PLGA", "Collagen") => CompatibilityScore(0.80;
        notes="Coating application",
        evidence_level=:in_vitro
    ),

    # Collagen composites
    ("Collagen", "Hydroxyapatite") => CompatibilityScore(0.95;
        notes="Biomimetic bone composite",
        evidence_level=:clinical,
        references=["Wang 2011 Biomaterials"]
    ),
    ("Collagen", "Chitosan") => CompatibilityScore(0.90;
        notes="Skin and wound healing",
        evidence_level=:clinical
    ),
    ("Collagen", "Alginate") => CompatibilityScore(0.85;
        notes="Injectable combination",
        evidence_level=:in_vivo
    ),
    ("Collagen", "GelMA") => CompatibilityScore(0.90;
        notes="Bioprinting combination",
        evidence_level=:in_vivo
    ),

    # Chitosan composites
    ("Chitosan", "Hydroxyapatite") => CompatibilityScore(0.90;
        notes="Bone scaffold application",
        evidence_level=:in_vivo
    ),
    ("Chitosan", "Alginate") => CompatibilityScore(0.90;
        notes="Polyelectrolyte complex",
        evidence_level=:in_vivo
    ),

    # Titanium coatings
    ("Titanium", "Hydroxyapatite") => CompatibilityScore(0.95;
        notes="HA-coated implants standard of care",
        evidence_level=:clinical
    ),
    ("Ti6Al4V", "Hydroxyapatite") => CompatibilityScore(0.95;
        notes="Commercial product",
        evidence_level=:clinical
    )
)

# =============================================================================
# Fabrication Process Compatibility
# =============================================================================

const FABRICATION_COMPATIBILITY = Dict{Tuple{String,Symbol},CompatibilityScore}(
    # 3D Printing (FDM/FFF)
    ("PCL", :fdm) => CompatibilityScore(0.95; notes="Optimal for FDM, low Tm"),
    ("PLA", :fdm) => CompatibilityScore(0.90; notes="Standard FDM material"),
    ("PLGA", :fdm) => CompatibilityScore(0.80; notes="Possible but degradation risk"),
    ("Collagen", :fdm) => CompatibilityScore(0.20; notes="Thermal degradation"),

    # Bioprinting (extrusion)
    ("GelMA", :bioprinting) => CompatibilityScore(0.95; notes="Optimal bioink"),
    ("Alginate", :bioprinting) => CompatibilityScore(0.90; notes="Standard bioink"),
    ("Collagen", :bioprinting) => CompatibilityScore(0.85; notes="Neutralized collagen bioink"),
    ("Hyaluronic_Acid", :bioprinting) => CompatibilityScore(0.85; notes="Viscosity modifier"),
    ("PCL", :bioprinting) => CompatibilityScore(0.70; notes="Support material"),

    # Electrospinning
    ("PCL", :electrospinning) => CompatibilityScore(0.95; notes="Excellent fiber formation"),
    ("PLA", :electrospinning) => CompatibilityScore(0.90; notes="Good fibers"),
    ("PLGA", :electrospinning) => CompatibilityScore(0.90; notes="Standard for nanofibers"),
    ("Collagen", :electrospinning) => CompatibilityScore(0.75; notes="HFIP solvent required"),
    ("Chitosan", :electrospinning) => CompatibilityScore(0.70; notes="Blend with PEO helps"),

    # Freeze-drying/lyophilization
    ("Collagen", :freeze_drying) => CompatibilityScore(0.95; notes="Standard method"),
    ("Chitosan", :freeze_drying) => CompatibilityScore(0.90; notes="Good pore formation"),
    ("Alginate", :freeze_drying) => CompatibilityScore(0.85; notes="Crosslink first"),
    ("GelMA", :freeze_drying) => CompatibilityScore(0.85; notes="Requires crosslinking"),

    # Salt leaching
    ("PCL", :salt_leaching) => CompatibilityScore(0.90; notes="Standard method"),
    ("PLA", :salt_leaching) => CompatibilityScore(0.90; notes="Common technique"),
    ("PLGA", :salt_leaching) => CompatibilityScore(0.85; notes="Solvent choice important"),

    # Melt molding
    ("PCL", :melt_molding) => CompatibilityScore(0.95; notes="Low Tm ideal"),
    ("PLA", :melt_molding) => CompatibilityScore(0.85; notes="Higher temperature needed"),
    ("Collagen", :melt_molding) => CompatibilityScore(0.10; notes="Thermal denaturation"),

    # Injectable systems
    ("Alginate", :injectable) => CompatibilityScore(0.95; notes="In situ gelation"),
    ("GelMA", :injectable) => CompatibilityScore(0.90; notes="UV curing in situ"),
    ("Collagen", :injectable) => CompatibilityScore(0.80; notes="Temperature-triggered"),
    ("Hyaluronic_Acid", :injectable) => CompatibilityScore(0.90; notes="Viscosupplementation"),
    ("Chitosan", :injectable) => CompatibilityScore(0.85; notes="pH-triggered gelation")
)

# =============================================================================
# Sterilization Compatibility
# =============================================================================

const STERILIZATION_COMPATIBILITY = Dict{Tuple{String,Symbol},CompatibilityScore}(
    # Gamma irradiation (25 kGy)
    ("PCL", :gamma) => CompatibilityScore(0.85; notes="Some MW reduction, acceptable"),
    ("PLA", :gamma) => CompatibilityScore(0.70; notes="Chain scission, MW loss"),
    ("PLGA", :gamma) => CompatibilityScore(0.65; notes="Significant degradation"),
    ("Collagen", :gamma) => CompatibilityScore(0.75; notes="Crosslinking occurs"),
    ("Chitosan", :gamma) => CompatibilityScore(0.80; notes="Chain scission"),
    ("Hydroxyapatite", :gamma) => CompatibilityScore(0.95; notes="No effect"),
    ("TCP", :gamma) => CompatibilityScore(0.95; notes="No effect"),
    ("Titanium", :gamma) => CompatibilityScore(0.95; notes="No effect"),

    # Ethylene oxide (EtO)
    ("PCL", :eto) => CompatibilityScore(0.90; notes="Good compatibility"),
    ("PLA", :eto) => CompatibilityScore(0.85; notes="Residual EtO concern"),
    ("PLGA", :eto) => CompatibilityScore(0.85; notes="Aeration important"),
    ("Collagen", :eto) => CompatibilityScore(0.80; notes="Residual EtO may affect cells"),
    ("GelMA", :eto) => CompatibilityScore(0.75; notes="May affect methacrylate groups"),
    ("Alginate", :eto) => CompatibilityScore(0.80; notes="Requires full aeration"),
    ("Hydroxyapatite", :eto) => CompatibilityScore(0.90; notes="Good"),
    ("Titanium", :eto) => CompatibilityScore(0.95; notes="Excellent"),

    # Autoclave (steam, 121°C)
    ("PCL", :autoclave) => CompatibilityScore(0.30; notes="Melting, deformation"),
    ("PLA", :autoclave) => CompatibilityScore(0.20; notes="Severe degradation"),
    ("PLGA", :autoclave) => CompatibilityScore(0.20; notes="Complete degradation"),
    ("Collagen", :autoclave) => CompatibilityScore(0.10; notes="Denaturation"),
    ("GelMA", :autoclave) => CompatibilityScore(0.10; notes="Denaturation"),
    ("Alginate", :autoclave) => CompatibilityScore(0.40; notes="Viscosity loss"),
    ("Hydroxyapatite", :autoclave) => CompatibilityScore(0.95; notes="Excellent"),
    ("TCP", :autoclave) => CompatibilityScore(0.95; notes="Excellent"),
    ("Titanium", :autoclave) => CompatibilityScore(0.95; notes="Excellent"),
    ("316L_SS", :autoclave) => CompatibilityScore(0.95; notes="Excellent"),

    # UV sterilization
    ("PCL", :uv) => CompatibilityScore(0.75; notes="Surface only, UV degradation"),
    ("PLA", :uv) => CompatibilityScore(0.70; notes="UV degradation"),
    ("PLGA", :uv) => CompatibilityScore(0.70; notes="Limited penetration"),
    ("Collagen", :uv) => CompatibilityScore(0.65; notes="Crosslinking, limited depth"),
    ("GelMA", :uv) => CompatibilityScore(0.50; notes="May initiate crosslinking"),
    ("Alginate", :uv) => CompatibilityScore(0.80; notes="Limited effect"),
    ("Hydroxyapatite", :uv) => CompatibilityScore(0.90; notes="Good"),

    # E-beam
    ("PCL", :ebeam) => CompatibilityScore(0.85; notes="Similar to gamma"),
    ("PLA", :ebeam) => CompatibilityScore(0.75; notes="Less degradation than gamma"),
    ("PLGA", :ebeam) => CompatibilityScore(0.70; notes="Better than gamma"),
    ("Collagen", :ebeam) => CompatibilityScore(0.75; notes="Crosslinking"),
    ("Hydroxyapatite", :ebeam) => CompatibilityScore(0.95; notes="No effect"),
    ("Titanium", :ebeam) => CompatibilityScore(0.95; notes="No effect")
)

# =============================================================================
# Lookup and Analysis Functions
# =============================================================================

"""
    check_compatibility(type, item1, item2)

Check compatibility between two items.
Type: :material_cell, :material_drug, :material_material, :fabrication, :sterilization
"""
function check_compatibility(type::Symbol, item1, item2)
    db = if type == :material_cell
        MATERIAL_CELL_COMPATIBILITY
    elseif type == :material_drug
        MATERIAL_DRUG_COMPATIBILITY
    elseif type == :material_material
        MATERIAL_MATERIAL_COMPATIBILITY
    elseif type == :fabrication
        FABRICATION_COMPATIBILITY
    elseif type == :sterilization
        STERILIZATION_COMPATIBILITY
    else
        error("Unknown compatibility type: $type")
    end

    key = (item1, item2)
    score = get(db, key, nothing)

    if isnothing(score)
        # Try reverse for symmetric relationships
        if type == :material_material
            score = get(db, (item2, item1), nothing)
        end
    end

    if isnothing(score)
        return CompatibilityScore(0.5;
            notes="No data available",
            evidence_level=:theoretical)
    end

    return score
end

"""
    get_compatibility_score(material, cell_type)

Quick lookup for material-cell compatibility.
"""
function get_compatibility_score(material::String, cell_type::Symbol)
    return check_compatibility(:material_cell, material, cell_type)
end

"""
    find_compatible_materials(cell_type; min_score=0.7)

Find all materials compatible with a cell type.
"""
function find_compatible_materials(cell_type::Symbol; min_score::Float64=0.7)
    compatible = Tuple{String,CompatibilityScore}[]

    for ((mat, ct), score) in MATERIAL_CELL_COMPATIBILITY
        if ct == cell_type && score.score >= min_score
            push!(compatible, (mat, score))
        end
    end

    # Sort by score descending
    sort!(compatible, by=x -> x[2].score, rev=true)
    return compatible
end

"""
    find_optimal_combination(target_tissue, drug, fabrication_method)

Find optimal material for given application requirements.
"""
function find_optimal_combination(target_tissue::Symbol,
                                  drug::String,
                                  fabrication_method::Symbol;
                                  cell_type::Symbol=:msc)
    candidates = Dict{String,Float64}()

    # Score each material
    materials = unique([k[1] for k in keys(MATERIAL_CELL_COMPATIBILITY)])

    for mat in materials
        total_score = 0.0
        weights = 0.0

        # Cell compatibility (weight 3)
        cell_compat = get(MATERIAL_CELL_COMPATIBILITY, (mat, cell_type), nothing)
        if !isnothing(cell_compat)
            total_score += 3 * cell_compat.score
            weights += 3
        end

        # Drug compatibility (weight 2)
        drug_compat = get(MATERIAL_DRUG_COMPATIBILITY, (mat, drug), nothing)
        if !isnothing(drug_compat)
            total_score += 2 * drug_compat.score
            weights += 2
        end

        # Fabrication compatibility (weight 2)
        fab_compat = get(FABRICATION_COMPATIBILITY, (mat, fabrication_method), nothing)
        if !isnothing(fab_compat)
            total_score += 2 * fab_compat.score
            weights += 2
        end

        if weights > 0
            candidates[mat] = total_score / weights
        end
    end

    # Sort by score
    sorted = sort(collect(candidates), by=x->x[2], rev=true)

    return sorted
end

"""
    generate_compatibility_report(material, cell_type, drug, fabrication, sterilization)

Generate comprehensive compatibility report for scaffold design.
"""
function generate_compatibility_report(material::String;
                                       cell_type::Symbol=:msc,
                                       drug::String="",
                                       fabrication::Symbol=:bioprinting,
                                       sterilization::Symbol=:gamma)
    report = Dict{Symbol,Any}()

    # Material-Cell
    cell_score = check_compatibility(:material_cell, material, cell_type)
    report[:cell_compatibility] = (
        cell_type = cell_type,
        score = cell_score.score,
        rating = cell_score.rating,
        notes = cell_score.notes
    )

    # Material-Drug
    if !isempty(drug)
        drug_score = check_compatibility(:material_drug, material, drug)
        report[:drug_compatibility] = (
            drug = drug,
            score = drug_score.score,
            rating = drug_score.rating,
            notes = drug_score.notes
        )
    end

    # Fabrication
    fab_score = check_compatibility(:fabrication, material, fabrication)
    report[:fabrication_compatibility] = (
        method = fabrication,
        score = fab_score.score,
        rating = fab_score.rating,
        notes = fab_score.notes
    )

    # Sterilization
    ster_score = check_compatibility(:sterilization, material, sterilization)
    report[:sterilization_compatibility] = (
        method = sterilization,
        score = ster_score.score,
        rating = ster_score.rating,
        notes = ster_score.notes
    )

    # Overall score
    scores = [cell_score.score, fab_score.score, ster_score.score]
    if haskey(report, :drug_compatibility)
        push!(scores, report[:drug_compatibility].score)
    end

    report[:overall_score] = mean(scores)
    report[:overall_rating] = if report[:overall_score] >= 0.9
        :excellent
    elseif report[:overall_score] >= 0.7
        :good
    elseif report[:overall_score] >= 0.5
        :moderate
    else
        :poor
    end

    # Recommendations
    recommendations = String[]
    if cell_score.score < 0.7
        push!(recommendations, "Consider surface modification to improve cell adhesion")
    end
    if fab_score.score < 0.7
        push!(recommendations, "Consider alternative fabrication method")
    end
    if ster_score.score < 0.7
        push!(recommendations, "Use alternative sterilization (EtO or e-beam recommended)")
    end
    report[:recommendations] = recommendations

    return report
end

# Helper function
function mean(x)
    return sum(x) / length(x)
end

end # module
