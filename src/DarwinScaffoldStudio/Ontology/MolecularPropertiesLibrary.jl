"""
    MolecularPropertiesLibrary

Comprehensive molecular properties database for biomaterials and drugs.

Contains:
- SMILES notation for chemical structure
- Molecular weight (MW)
- LogP (octanol-water partition coefficient)
- pKa (acid dissociation constant)
- Hydrogen bond donors/acceptors
- Rotatable bonds
- Polar surface area (PSA)
- Lipinski's Rule of Five compliance
- Drug-likeness scores

Data sources:
- PubChem (https://pubchem.ncbi.nlm.nih.gov/)
- ChEMBL (https://www.ebi.ac.uk/chembl/)
- DrugBank (https://go.drugbank.com/)
- ChEBI (https://www.ebi.ac.uk/chebi/)

Author: Dr. Demetrios Agourakis
"""
module MolecularPropertiesLibrary

export MolecularProperties, get_molecular_properties
export MOLECULAR_DB, POLYMER_MONOMERS, DRUG_MOLECULES, GROWTH_FACTORS_MOL
export calculate_druglikeness, check_lipinski, check_veber
export get_smiles, get_molecular_weight, get_logp
export search_by_property, filter_druglike
export FUNCTIONAL_GROUPS, identify_functional_groups

# =============================================================================
# Molecular Properties Structure
# =============================================================================

"""
    MolecularProperties

Comprehensive molecular property data.
"""
struct MolecularProperties
    id::String                    # ChEBI/PubChem/DrugBank ID
    name::String                  # Common name
    smiles::String               # SMILES notation
    inchi::String                # InChI identifier
    inchikey::String             # InChIKey (hashed)
    molecular_formula::String    # e.g., "C6H12O6"
    molecular_weight::Float64    # g/mol
    exact_mass::Float64          # Monoisotopic mass
    logp::Float64                # Octanol-water partition
    logd_7_4::Float64            # LogD at pH 7.4
    pka_acidic::Vector{Float64}  # Acidic pKa values
    pka_basic::Vector{Float64}   # Basic pKa values
    hbd::Int                     # H-bond donors
    hba::Int                     # H-bond acceptors
    rotatable_bonds::Int         # Rotatable bonds
    tpsa::Float64                # Topological polar surface area (Å²)
    heavy_atoms::Int             # Non-hydrogen atoms
    aromatic_rings::Int          # Number of aromatic rings
    formal_charge::Int           # Net formal charge
    solubility_mg_ml::Float64    # Aqueous solubility
    melting_point_c::Float64     # Melting point
    boiling_point_c::Float64     # Boiling point
    density_g_cm3::Float64       # Density
    refraction_index::Float64    # Refractive index
    functional_groups::Vector{String}  # Identified functional groups
    xrefs::Dict{String,String}   # Cross-references
end

# Convenience constructor with defaults
function MolecularProperties(id::String, name::String, smiles::String;
    inchi::String="",
    inchikey::String="",
    molecular_formula::String="",
    molecular_weight::Float64=0.0,
    exact_mass::Float64=0.0,
    logp::Float64=0.0,
    logd_7_4::Float64=0.0,
    pka_acidic::Vector{Float64}=Float64[],
    pka_basic::Vector{Float64}=Float64[],
    hbd::Int=0,
    hba::Int=0,
    rotatable_bonds::Int=0,
    tpsa::Float64=0.0,
    heavy_atoms::Int=0,
    aromatic_rings::Int=0,
    formal_charge::Int=0,
    solubility_mg_ml::Float64=NaN,
    melting_point_c::Float64=NaN,
    boiling_point_c::Float64=NaN,
    density_g_cm3::Float64=NaN,
    refraction_index::Float64=NaN,
    functional_groups::Vector{String}=String[],
    xrefs::Dict{String,String}=Dict{String,String}())

    MolecularProperties(id, name, smiles, inchi, inchikey, molecular_formula,
        molecular_weight, exact_mass, logp, logd_7_4, pka_acidic, pka_basic,
        hbd, hba, rotatable_bonds, tpsa, heavy_atoms, aromatic_rings,
        formal_charge, solubility_mg_ml, melting_point_c, boiling_point_c,
        density_g_cm3, refraction_index, functional_groups, xrefs)
end

# =============================================================================
# Polymer Monomers Database
# =============================================================================

const POLYMER_MONOMERS = Dict{String,MolecularProperties}(
    # Biodegradable polymer monomers
    "lactic_acid" => MolecularProperties(
        "CHEBI:422", "lactic acid", "CC(O)C(=O)O";
        inchi="InChI=1S/C3H6O3/c1-2(4)3(5)6/h2,4H,1H3,(H,5,6)",
        inchikey="JVTAAEKCZFNVCJ-UHFFFAOYSA-N",
        molecular_formula="C3H6O3",
        molecular_weight=90.08,
        exact_mass=90.0317,
        logp=-0.72,
        pka_acidic=[3.86],
        hbd=2, hba=3, rotatable_bonds=1, tpsa=57.53, heavy_atoms=6,
        melting_point_c=18.0, boiling_point_c=122.0, density_g_cm3=1.206,
        functional_groups=["carboxylic_acid", "hydroxyl", "chiral_center"],
        xrefs=Dict("CAS"=>"50-21-5", "PubChem"=>"612")
    ),

    "glycolic_acid" => MolecularProperties(
        "CHEBI:17497", "glycolic acid", "OCC(=O)O";
        inchi="InChI=1S/C2H4O3/c3-1-2(4)5/h3H,1H2,(H,4,5)",
        inchikey="AEMRFAOFKBGASW-UHFFFAOYSA-N",
        molecular_formula="C2H4O3",
        molecular_weight=76.05,
        exact_mass=76.0160,
        logp=-1.11,
        pka_acidic=[3.83],
        hbd=2, hba=3, rotatable_bonds=1, tpsa=57.53, heavy_atoms=5,
        melting_point_c=80.0, density_g_cm3=1.49,
        functional_groups=["carboxylic_acid", "hydroxyl"],
        xrefs=Dict("CAS"=>"79-14-1", "PubChem"=>"757")
    ),

    "caprolactone" => MolecularProperties(
        "CHEBI:36606", "epsilon-caprolactone", "O=C1CCCCCO1";
        inchi="InChI=1S/C6H10O2/c7-6-4-2-1-3-5-8-6/h1-5H2",
        inchikey="JHWNWJKBPDFINM-UHFFFAOYSA-N",
        molecular_formula="C6H10O2",
        molecular_weight=114.14,
        exact_mass=114.0681,
        logp=1.02,
        hbd=0, hba=2, rotatable_bonds=0, tpsa=26.30, heavy_atoms=8,
        melting_point_c=-1.0, boiling_point_c=253.0, density_g_cm3=1.03,
        functional_groups=["lactone", "ester"],
        xrefs=Dict("CAS"=>"502-44-3", "PubChem"=>"10401")
    ),

    "ethylene_glycol" => MolecularProperties(
        "CHEBI:30742", "ethylene glycol", "OCCO";
        inchi="InChI=1S/C2H6O2/c3-1-2-4/h3-4H,1-2H2",
        inchikey="LYCAIKOWRPUZTN-UHFFFAOYSA-N",
        molecular_formula="C2H6O2",
        molecular_weight=62.07,
        exact_mass=62.0368,
        logp=-1.36,
        hbd=2, hba=2, rotatable_bonds=1, tpsa=40.46, heavy_atoms=4,
        melting_point_c=-13.0, boiling_point_c=197.3, density_g_cm3=1.113,
        functional_groups=["diol", "hydroxyl"],
        xrefs=Dict("CAS"=>"107-21-1", "PubChem"=>"174")
    ),

    "hydroxybutyrate" => MolecularProperties(
        "CHEBI:37054", "3-hydroxybutyric acid", "CC(O)CC(=O)O";
        inchi="InChI=1S/C4H8O3/c1-3(5)2-4(6)7/h3,5H,2H2,1H3,(H,6,7)",
        inchikey="WHBMMWSBFZVSSE-UHFFFAOYSA-N",
        molecular_formula="C4H8O3",
        molecular_weight=104.10,
        exact_mass=104.0473,
        logp=-0.64,
        pka_acidic=[4.70],
        hbd=2, hba=3, rotatable_bonds=2, tpsa=57.53, heavy_atoms=7,
        functional_groups=["carboxylic_acid", "secondary_alcohol", "chiral_center"],
        xrefs=Dict("CAS"=>"300-85-6", "PubChem"=>"441")
    ),

    # Natural polymer building blocks
    "glucosamine" => MolecularProperties(
        "CHEBI:5417", "D-glucosamine", "NC1C(O)OC(CO)C(O)C1O";
        inchi="InChI=1S/C6H13NO5/c7-3-5(10)4(9)2(1-8)12-6(3)11/h2-6,8-11H,1,7H2",
        inchikey="MSWZFWKMSRAUBD-GASJEMHNSA-N",
        molecular_formula="C6H13NO5",
        molecular_weight=179.17,
        exact_mass=179.0794,
        logp=-2.04,
        pka_basic=[7.75],
        hbd=5, hba=6, rotatable_bonds=1, tpsa=119.25, heavy_atoms=12,
        melting_point_c=88.0,
        functional_groups=["amine", "pyranose", "polyol"],
        xrefs=Dict("CAS"=>"3416-24-8", "PubChem"=>"439213")
    ),

    "n_acetylglucosamine" => MolecularProperties(
        "CHEBI:506227", "N-acetyl-D-glucosamine", "CC(=O)NC1C(O)OC(CO)C(O)C1O";
        inchi="InChI=1S/C8H15NO6/c1-3(11)9-5-7(13)6(12)4(2-10)15-8(5)14/h4-8,10,12-14H,2H2,1H3,(H,9,11)",
        inchikey="OVRNDRQMDRJTHS-FMDGEEDCSA-N",
        molecular_formula="C8H15NO6",
        molecular_weight=221.21,
        exact_mass=221.0899,
        logp=-1.78,
        hbd=4, hba=6, rotatable_bonds=2, tpsa=119.25, heavy_atoms=15,
        melting_point_c=211.0,
        functional_groups=["amide", "acetyl", "pyranose", "polyol"],
        xrefs=Dict("CAS"=>"7512-17-6", "PubChem"=>"439174")
    ),

    "mannuronic_acid" => MolecularProperties(
        "CHEBI:28022", "D-mannuronic acid", "OCC1OC(O)C(O)C(O)C1O";
        molecular_formula="C6H10O7",
        molecular_weight=194.14,
        logp=-2.5,
        pka_acidic=[3.38],
        hbd=5, hba=7, rotatable_bonds=1, tpsa=127.45, heavy_atoms=13,
        functional_groups=["uronic_acid", "pyranose", "polyol"],
        xrefs=Dict("CAS"=>"6814-36-4")
    ),

    "guluronic_acid" => MolecularProperties(
        "CHEBI:28661", "L-guluronic acid", "OC1C(O)C(O)OC(C(=O)O)C1O";
        molecular_formula="C6H10O7",
        molecular_weight=194.14,
        logp=-2.5,
        pka_acidic=[3.65],
        hbd=5, hba=7, rotatable_bonds=1, tpsa=127.45, heavy_atoms=13,
        functional_groups=["uronic_acid", "pyranose", "polyol"],
        xrefs=Dict("CAS"=>"15769-56-9")
    ),

    # Crosslinker molecules
    "genipin" => MolecularProperties(
        "CHEBI:5367", "genipin", "COC(=O)C1=COC(O)C2C(CO)=CCC12";
        molecular_formula="C11H14O5",
        molecular_weight=226.23,
        exact_mass=226.0841,
        logp=0.18,
        hbd=2, hba=5, rotatable_bonds=2, tpsa=76.74, heavy_atoms=16,
        melting_point_c=121.0,
        functional_groups=["ester", "iridoid", "hemiacetal", "hydroxyl"],
        xrefs=Dict("CAS"=>"6902-77-8", "PubChem"=>"442424")
    ),

    "glutaraldehyde" => MolecularProperties(
        "CHEBI:17588", "glutaraldehyde", "O=CCCCC=O";
        inchi="InChI=1S/C5H8O2/c6-4-2-1-3-5-7/h4-5H,1-3H2",
        inchikey="SXRSQZLOMIGNAQ-UHFFFAOYSA-N",
        molecular_formula="C5H8O2",
        molecular_weight=100.12,
        exact_mass=100.0524,
        logp=0.18,
        hbd=0, hba=2, rotatable_bonds=4, tpsa=34.14, heavy_atoms=7,
        boiling_point_c=187.0, density_g_cm3=1.06,
        functional_groups=["dialdehyde", "aldehyde"],
        xrefs=Dict("CAS"=>"111-30-8", "PubChem"=>"3485")
    ),

    "carbodiimide_edc" => MolecularProperties(
        "CHEBI:53023", "EDC", "CCN=C=NCCCN(C)C";
        molecular_formula="C8H17N3",
        molecular_weight=155.24,
        exact_mass=155.1422,
        logp=0.95,
        hbd=0, hba=3, rotatable_bonds=6, tpsa=32.34, heavy_atoms=11,
        functional_groups=["carbodiimide", "tertiary_amine"],
        xrefs=Dict("CAS"=>"25952-53-8", "PubChem"=>"15908")
    ),

    # Methacrylate functionalization
    "methacrylic_acid" => MolecularProperties(
        "CHEBI:25219", "methacrylic acid", "CC(=C)C(=O)O";
        inchi="InChI=1S/C4H6O2/c1-3(2)4(5)6/h1H2,2H3,(H,5,6)",
        inchikey="CERQOIWHTDAKMF-UHFFFAOYSA-N",
        molecular_formula="C4H6O2",
        molecular_weight=86.09,
        exact_mass=86.0368,
        logp=0.93,
        pka_acidic=[4.66],
        hbd=1, hba=2, rotatable_bonds=1, tpsa=37.30, heavy_atoms=6,
        melting_point_c=15.0, boiling_point_c=161.0, density_g_cm3=1.015,
        functional_groups=["carboxylic_acid", "vinyl", "methyl"],
        xrefs=Dict("CAS"=>"79-41-4", "PubChem"=>"4093")
    ),

    "2_hydroxyethyl_methacrylate" => MolecularProperties(
        "CHEBI:53116", "HEMA", "CC(=C)C(=O)OCCO";
        molecular_formula="C6H10O3",
        molecular_weight=130.14,
        exact_mass=130.0630,
        logp=0.47,
        hbd=1, hba=3, rotatable_bonds=4, tpsa=46.53, heavy_atoms=9,
        boiling_point_c=205.0, density_g_cm3=1.073,
        functional_groups=["ester", "vinyl", "hydroxyl"],
        xrefs=Dict("CAS"=>"868-77-9", "PubChem"=>"13360")
    )
)

# =============================================================================
# Drug Molecules Database
# =============================================================================

const DRUG_MOLECULES = Dict{String,MolecularProperties}(
    # Anti-inflammatory drugs
    "dexamethasone" => MolecularProperties(
        "CHEBI:41879", "dexamethasone",
        "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO";
        inchi="InChI=1S/C22H29FO5/c1-12-8-16-15-5-4-13-9-14(25)6-7-19(13,2)21(15,23)17(26)10-20(16,3)22(12,28)18(27)11-24/h6-7,9,12,15-17,24,26,28H,4-5,8,10-11H2,1-3H3",
        inchikey="UREBDLICKHMUKA-CXSFZGCWSA-N",
        molecular_formula="C22H29FO5",
        molecular_weight=392.46,
        exact_mass=392.2000,
        logp=1.83,
        logd_7_4=1.83,
        pka_acidic=[12.42],
        hbd=3, hba=6, rotatable_bonds=2, tpsa=94.83, heavy_atoms=28,
        aromatic_rings=0,
        solubility_mg_ml=0.089,
        melting_point_c=262.0,
        functional_groups=["ketone", "fluorine", "steroid", "hydroxyl", "hemiketal"],
        xrefs=Dict("CAS"=>"50-02-2", "DrugBank"=>"DB01234", "PubChem"=>"5743")
    ),

    "ibuprofen" => MolecularProperties(
        "CHEBI:5855", "ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O";
        inchi="InChI=1S/C13H18O2/c1-9(2)8-11-4-6-12(7-5-11)10(3)13(14)15/h4-7,9-10H,8H2,1-3H3,(H,14,15)",
        inchikey="HEFNNWSXXWATRW-UHFFFAOYSA-N",
        molecular_formula="C13H18O2",
        molecular_weight=206.28,
        exact_mass=206.1307,
        logp=3.97,
        logd_7_4=0.84,
        pka_acidic=[4.91],
        hbd=1, hba=2, rotatable_bonds=4, tpsa=37.30, heavy_atoms=15,
        aromatic_rings=1,
        solubility_mg_ml=0.021,
        melting_point_c=76.0,
        functional_groups=["carboxylic_acid", "isobutyl", "phenyl"],
        xrefs=Dict("CAS"=>"15687-27-1", "DrugBank"=>"DB01050", "PubChem"=>"3672")
    ),

    "indomethacin" => MolecularProperties(
        "CHEBI:49662", "indomethacin",
        "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c3ccc(Cl)cc3";
        molecular_formula="C19H16ClNO4",
        molecular_weight=357.79,
        exact_mass=357.0768,
        logp=4.27,
        logd_7_4=0.96,
        pka_acidic=[4.50],
        hbd=1, hba=4, rotatable_bonds=4, tpsa=68.53, heavy_atoms=25,
        aromatic_rings=2,
        solubility_mg_ml=0.0025,
        melting_point_c=160.0,
        functional_groups=["carboxylic_acid", "indole", "amide", "chloro", "methoxy"],
        xrefs=Dict("CAS"=>"53-86-1", "DrugBank"=>"DB00328", "PubChem"=>"3715")
    ),

    # Antibiotics
    "vancomycin" => MolecularProperties(
        "CHEBI:28001", "vancomycin", "[Complex glycopeptide SMILES]";
        molecular_formula="C66H75Cl2N9O24",
        molecular_weight=1449.25,
        exact_mass=1447.4303,
        logp=-3.1,
        logd_7_4=-3.1,
        pka_acidic=[2.18, 7.75, 8.89],
        pka_basic=[9.59],
        hbd=19, hba=26, rotatable_bonds=10, tpsa=530.49, heavy_atoms=101,
        aromatic_rings=5,
        solubility_mg_ml=100.0,
        melting_point_c=190.0,
        functional_groups=["glycopeptide", "chloro", "phenol", "amide", "carboxylic_acid"],
        xrefs=Dict("CAS"=>"1404-90-6", "DrugBank"=>"DB00512", "PubChem"=>"14969")
    ),

    "gentamicin" => MolecularProperties(
        "CHEBI:27412", "gentamicin", "[Aminoglycoside mixture]";
        molecular_formula="C21H43N5O7",
        molecular_weight=477.60,
        exact_mass=477.3162,
        logp=-3.1,
        pka_basic=[6.2, 8.2, 8.6],
        hbd=8, hba=12, rotatable_bonds=8, tpsa=199.73, heavy_atoms=33,
        solubility_mg_ml=100.0,
        functional_groups=["aminoglycoside", "amine", "hydroxyl"],
        xrefs=Dict("CAS"=>"1403-66-3", "DrugBank"=>"DB00798", "PubChem"=>"3467")
    ),

    "ciprofloxacin" => MolecularProperties(
        "CHEBI:100241", "ciprofloxacin",
        "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O";
        molecular_formula="C17H18FN3O3",
        molecular_weight=331.34,
        exact_mass=331.1332,
        logp=0.28,
        logd_7_4=-0.81,
        pka_acidic=[6.09],
        pka_basic=[8.74],
        hbd=2, hba=6, rotatable_bonds=3, tpsa=72.88, heavy_atoms=24,
        aromatic_rings=2,
        solubility_mg_ml=30.0,
        melting_point_c=255.0,
        functional_groups=["quinolone", "carboxylic_acid", "piperazine", "fluorine", "cyclopropyl"],
        xrefs=Dict("CAS"=>"85721-33-1", "DrugBank"=>"DB00537", "PubChem"=>"2764")
    ),

    "rifampicin" => MolecularProperties(
        "CHEBI:28077", "rifampicin", "[Complex ansamycin SMILES]";
        molecular_formula="C43H58N4O12",
        molecular_weight=822.94,
        exact_mass=822.4051,
        logp=4.24,
        logd_7_4=2.77,
        pka_acidic=[1.7, 7.9],
        hbd=6, hba=14, rotatable_bonds=5, tpsa=220.15, heavy_atoms=59,
        aromatic_rings=2,
        solubility_mg_ml=2.5,
        melting_point_c=183.0,
        functional_groups=["ansamycin", "macrocycle", "hydroxyl", "amide", "phenol"],
        xrefs=Dict("CAS"=>"13292-46-1", "DrugBank"=>"DB01045", "PubChem"=>"135398735")
    ),

    # Chemotherapeutics
    "doxorubicin" => MolecularProperties(
        "CHEBI:28748", "doxorubicin",
        "COc1cccc2C(=O)c3c(O)c4CC(O)(CC(OC5CC(N)C(O)C(C)O5)c4c(O)c3C(=O)c12)C(=O)CO";
        molecular_formula="C27H29NO11",
        molecular_weight=543.52,
        exact_mass=543.1741,
        logp=1.27,
        logd_7_4=0.53,
        pka_acidic=[8.22, 10.16],
        pka_basic=[8.15],
        hbd=6, hba=12, rotatable_bonds=5, tpsa=206.07, heavy_atoms=39,
        aromatic_rings=3,
        solubility_mg_ml=10.0,
        melting_point_c=205.0,
        functional_groups=["anthracycline", "glycoside", "quinone", "hydroxyl", "ketone"],
        xrefs=Dict("CAS"=>"23214-92-8", "DrugBank"=>"DB00997", "PubChem"=>"31703")
    ),

    "methotrexate" => MolecularProperties(
        "CHEBI:44185", "methotrexate",
        "CN(Cc1cnc2nc(N)nc(N)c2n1)c3ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc3";
        molecular_formula="C20H22N8O5",
        molecular_weight=454.44,
        exact_mass=454.1713,
        logp=-1.85,
        logd_7_4=-5.67,
        pka_acidic=[3.36, 4.70, 5.71],
        hbd=5, hba=12, rotatable_bonds=9, tpsa=210.54, heavy_atoms=33,
        aromatic_rings=2,
        solubility_mg_ml=0.26,
        melting_point_c=185.0,
        functional_groups=["pteridine", "diaminopyrimidine", "carboxylic_acid", "amide", "amino"],
        xrefs=Dict("CAS"=>"59-05-2", "DrugBank"=>"DB00563", "PubChem"=>"126941")
    ),

    "paclitaxel" => MolecularProperties(
        "CHEBI:45863", "paclitaxel", "[Complex taxane SMILES]";
        molecular_formula="C47H51NO14",
        molecular_weight=853.91,
        exact_mass=853.3310,
        logp=3.96,
        logd_7_4=3.96,
        hbd=4, hba=15, rotatable_bonds=14, tpsa=221.29, heavy_atoms=62,
        aromatic_rings=3,
        solubility_mg_ml=0.0003,
        melting_point_c=216.0,
        functional_groups=["taxane", "ester", "amide", "hydroxyl", "epoxide"],
        xrefs=Dict("CAS"=>"33069-62-4", "DrugBank"=>"DB01229", "PubChem"=>"36314")
    ),

    # Osteogenic drugs
    "alendronate" => MolecularProperties(
        "CHEBI:2567", "alendronic acid", "NCCCC(O)(P(=O)(O)O)P(=O)(O)O";
        molecular_formula="C4H13NO7P2",
        molecular_weight=249.10,
        exact_mass=249.0168,
        logp=-4.40,
        logd_7_4=-5.75,
        pka_acidic=[0.8, 2.2, 6.3, 10.9],
        pka_basic=[12.0],
        hbd=6, hba=8, rotatable_bonds=5, tpsa=177.89, heavy_atoms=14,
        solubility_mg_ml=10.0,
        melting_point_c=243.0,
        functional_groups=["bisphosphonate", "phosphonic_acid", "amine", "hydroxyl"],
        xrefs=Dict("CAS"=>"66376-36-1", "DrugBank"=>"DB00630", "PubChem"=>"2088")
    ),

    "simvastatin" => MolecularProperties(
        "CHEBI:9150", "simvastatin",
        "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12";
        molecular_formula="C25H38O5",
        molecular_weight=418.57,
        exact_mass=418.2719,
        logp=4.68,
        logd_7_4=4.68,
        hbd=1, hba=5, rotatable_bonds=7, tpsa=72.83, heavy_atoms=30,
        aromatic_rings=0,
        solubility_mg_ml=0.0003,
        melting_point_c=138.0,
        functional_groups=["lactone", "ester", "hydroxyl", "decalin"],
        xrefs=Dict("CAS"=>"79902-63-9", "DrugBank"=>"DB00641", "PubChem"=>"54454")
    ),

    # Local anesthetics
    "lidocaine" => MolecularProperties(
        "CHEBI:6456", "lidocaine", "CCN(CC)CC(=O)Nc1c(C)cccc1C";
        molecular_formula="C14H22N2O",
        molecular_weight=234.34,
        exact_mass=234.1732,
        logp=2.44,
        logd_7_4=1.63,
        pka_basic=[7.96],
        hbd=1, hba=2, rotatable_bonds=5, tpsa=32.34, heavy_atoms=17,
        aromatic_rings=1,
        solubility_mg_ml=4.1,
        melting_point_c=69.0,
        functional_groups=["amide", "aniline", "tertiary_amine", "xylyl"],
        xrefs=Dict("CAS"=>"137-58-6", "DrugBank"=>"DB00281", "PubChem"=>"3676")
    ),

    "bupivacaine" => MolecularProperties(
        "CHEBI:3215", "bupivacaine", "CCCCN1CCCCC1C(=O)Nc2c(C)cccc2C";
        molecular_formula="C18H28N2O",
        molecular_weight=288.43,
        exact_mass=288.2202,
        logp=3.41,
        logd_7_4=2.54,
        pka_basic=[8.10],
        hbd=1, hba=2, rotatable_bonds=5, tpsa=32.34, heavy_atoms=21,
        aromatic_rings=1,
        solubility_mg_ml=0.05,
        melting_point_c=107.0,
        functional_groups=["amide", "piperidine", "aniline", "xylyl"],
        xrefs=Dict("CAS"=>"2180-92-9", "DrugBank"=>"DB00297", "PubChem"=>"2474")
    )
)

# =============================================================================
# Growth Factors (Molecular Properties)
# =============================================================================

const GROWTH_FACTORS_MOL = Dict{String,MolecularProperties}(
    "bmp2" => MolecularProperties(
        "UNIPROT:P12643", "BMP-2", "PROTEIN";
        molecular_formula="C584H918N166O173S9",
        molecular_weight=12894.0,  # Mature dimer ~26 kDa
        hbd=200, hba=250,
        functional_groups=["TGF-beta_superfamily", "cystine_knot", "homodimer"],
        xrefs=Dict("UniProt"=>"P12643", "PDB"=>"3BMP")
    ),

    "bmp7" => MolecularProperties(
        "UNIPROT:P18075", "BMP-7", "PROTEIN";
        molecular_formula="C596H932N168O175S11",
        molecular_weight=13100.0,
        functional_groups=["TGF-beta_superfamily", "cystine_knot", "homodimer"],
        xrefs=Dict("UniProt"=>"P18075", "PDB"=>"1BMP")
    ),

    "vegf_a" => MolecularProperties(
        "UNIPROT:P15692", "VEGF-A", "PROTEIN";
        molecular_formula="C934H1468N262O282S14",
        molecular_weight=21000.0,  # 165 isoform, homodimer
        functional_groups=["VEGF_family", "cystine_knot", "homodimer", "heparin_binding"],
        xrefs=Dict("UniProt"=>"P15692", "PDB"=>"1VPF")
    ),

    "fgf2" => MolecularProperties(
        "UNIPROT:P09038", "FGF-2", "PROTEIN";
        molecular_formula="C756H1202N218O226S8",
        molecular_weight=17200.0,  # 155 aa form
        functional_groups=["FGF_family", "heparin_binding", "beta_trefoil"],
        xrefs=Dict("UniProt"=>"P09038", "PDB"=>"1BFG")
    ),

    "tgfb1" => MolecularProperties(
        "UNIPROT:P01137", "TGF-beta1", "PROTEIN";
        molecular_formula="C560H876N152O168S8",
        molecular_weight=12700.0,  # Mature homodimer ~25 kDa
        functional_groups=["TGF-beta_superfamily", "cystine_knot", "homodimer"],
        xrefs=Dict("UniProt"=>"P01137", "PDB"=>"1KLC")
    ),

    "pdgf_bb" => MolecularProperties(
        "UNIPROT:P01127", "PDGF-BB", "PROTEIN";
        molecular_formula="C548H856N156O158S12",
        molecular_weight=12300.0,  # B-chain homodimer
        functional_groups=["PDGF_family", "cystine_knot", "homodimer"],
        xrefs=Dict("UniProt"=>"P01127", "PDB"=>"1PDG")
    ),

    "igf1" => MolecularProperties(
        "UNIPROT:P05019", "IGF-1", "PROTEIN";
        molecular_formula="C331H512N94O101S6",
        molecular_weight=7649.0,  # 70 aa
        functional_groups=["IGF_family", "insulin_fold"],
        xrefs=Dict("UniProt"=>"P05019", "PDB"=>"1GZR")
    ),

    "egf" => MolecularProperties(
        "UNIPROT:P01133", "EGF", "PROTEIN";
        molecular_formula="C257H381N73O83S7",
        molecular_weight=6045.0,  # 53 aa
        functional_groups=["EGF_family", "EGF_domain"],
        xrefs=Dict("UniProt"=>"P01133", "PDB"=>"1EGF")
    ),

    "ngf" => MolecularProperties(
        "UNIPROT:P01138", "NGF", "PROTEIN";
        molecular_formula="C580H908N164O178S10",
        molecular_weight=13000.0,  # Homodimer
        functional_groups=["neurotrophin_family", "cystine_knot", "homodimer"],
        xrefs=Dict("UniProt"=>"P01138", "PDB"=>"1BET")
    ),

    "sdf1" => MolecularProperties(
        "UNIPROT:P48061", "SDF-1/CXCL12", "PROTEIN";
        molecular_formula="C340H528N98O102S4",
        molecular_weight=7800.0,  # Alpha isoform
        functional_groups=["CXC_chemokine", "heparin_binding", "homodimer"],
        xrefs=Dict("UniProt"=>"P48061", "PDB"=>"1SDF")
    )
)

# =============================================================================
# Combined Database
# =============================================================================

"""All molecular properties combined."""
const MOLECULAR_DB = merge(POLYMER_MONOMERS, DRUG_MOLECULES, GROWTH_FACTORS_MOL)

# =============================================================================
# Functional Groups Dictionary
# =============================================================================

const FUNCTIONAL_GROUPS = Dict{String,String}(
    # Organic functional groups
    "hydroxyl" => "O attached to carbon (-OH)",
    "carboxylic_acid" => "Carbon with =O and -OH (COOH)",
    "ester" => "Carbon with =O bonded to -O-C",
    "amide" => "Carbon with =O bonded to -N",
    "amine" => "Nitrogen with hydrogen(s) attached",
    "tertiary_amine" => "Nitrogen bonded to three carbons",
    "aldehyde" => "Terminal C=O group",
    "ketone" => "C=O between two carbons",
    "ether" => "Oxygen between two carbons",
    "vinyl" => "C=C double bond (terminal)",
    "phenyl" => "Benzene ring",
    "lactone" => "Cyclic ester",

    # Biomolecule-specific
    "pyranose" => "Six-membered sugar ring",
    "polyol" => "Multiple hydroxyl groups",
    "glycoside" => "Sugar attached to non-sugar",
    "peptide_bond" => "Amide bond in proteins",
    "phosphate" => "PO4 group",
    "sulfate" => "SO4 group",

    # Polymer functional groups
    "methacrylate" => "Methacrylic ester (photo-crosslinkable)",
    "acrylate" => "Acrylic ester (photo-crosslinkable)",
    "thiol" => "Sulfhydryl group (-SH)",
    "maleimide" => "Michael acceptor for thiols",
    "nhs_ester" => "N-hydroxysuccinimide ester (amine reactive)",
    "azide" => "N3 group (click chemistry)",
    "alkyne" => "C≡C triple bond (click chemistry)",

    # Drug-specific
    "steroid" => "Four-ring steroid backbone",
    "quinolone" => "Quinolone antibiotic core",
    "beta_lactam" => "Four-membered lactam ring",
    "taxane" => "Taxol-like backbone",
    "anthracycline" => "Anthracycline antibiotic core",
    "bisphosphonate" => "Two phosphonate groups on carbon"
)

# =============================================================================
# Drug-likeness Rules
# =============================================================================

"""
    check_lipinski(mol)

Check Lipinski's Rule of Five (Lipinski et al. 2001):
- MW ≤ 500
- LogP ≤ 5
- HBD ≤ 5
- HBA ≤ 10
"""
function check_lipinski(mol::MolecularProperties)
    violations = 0
    details = String[]

    if mol.molecular_weight > 500
        violations += 1
        push!(details, "MW $(mol.molecular_weight) > 500")
    end
    if mol.logp > 5
        violations += 1
        push!(details, "LogP $(mol.logp) > 5")
    end
    if mol.hbd > 5
        violations += 1
        push!(details, "HBD $(mol.hbd) > 5")
    end
    if mol.hba > 10
        violations += 1
        push!(details, "HBA $(mol.hba) > 10")
    end

    return (
        passed = violations <= 1,
        violations = violations,
        details = details
    )
end

"""
    check_veber(mol)

Check Veber's rules for oral bioavailability (Veber et al. 2002):
- Rotatable bonds ≤ 10
- TPSA ≤ 140 Å²
"""
function check_veber(mol::MolecularProperties)
    violations = 0
    details = String[]

    if mol.rotatable_bonds > 10
        violations += 1
        push!(details, "Rotatable bonds $(mol.rotatable_bonds) > 10")
    end
    if mol.tpsa > 140
        violations += 1
        push!(details, "TPSA $(mol.tpsa) > 140 Å²")
    end

    return (
        passed = violations == 0,
        violations = violations,
        details = details
    )
end

"""
    calculate_druglikeness(mol)

Calculate composite drug-likeness score.
"""
function calculate_druglikeness(mol::MolecularProperties)
    lipinski = check_lipinski(mol)
    veber = check_veber(mol)

    # Calculate QED-like score (simplified)
    score = 1.0

    # MW penalty
    if mol.molecular_weight > 500
        score *= exp(-(mol.molecular_weight - 500) / 200)
    end

    # LogP penalty (optimal range 1-3)
    if mol.logp < 0 || mol.logp > 5
        score *= exp(-abs(mol.logp - 2) / 3)
    end

    # HBD penalty
    if mol.hbd > 5
        score *= exp(-(mol.hbd - 5) / 3)
    end

    # TPSA penalty
    if mol.tpsa > 140
        score *= exp(-(mol.tpsa - 140) / 50)
    end

    return (
        score = score,
        lipinski = lipinski,
        veber = veber,
        oral_bioavailability = lipinski.passed && veber.passed
    )
end

# =============================================================================
# Lookup Functions
# =============================================================================

"""Get molecular properties by ID or name."""
function get_molecular_properties(id::String)
    # Try direct lookup
    mol = get(MOLECULAR_DB, id, nothing)
    if !isnothing(mol)
        return mol
    end

    # Try case-insensitive name match
    id_lower = lowercase(id)
    for (key, mol) in MOLECULAR_DB
        if lowercase(mol.name) == id_lower || lowercase(key) == id_lower
            return mol
        end
    end

    return nothing
end

"""Get SMILES for compound."""
get_smiles(id::String) = (mol = get_molecular_properties(id); isnothing(mol) ? nothing : mol.smiles)

"""Get molecular weight for compound."""
get_molecular_weight(id::String) = (mol = get_molecular_properties(id); isnothing(mol) ? nothing : mol.molecular_weight)

"""Get LogP for compound."""
get_logp(id::String) = (mol = get_molecular_properties(id); isnothing(mol) ? nothing : mol.logp)

"""
    search_by_property(property, min_val, max_val)

Search molecules by property range.
"""
function search_by_property(property::Symbol, min_val::Real, max_val::Real)
    results = MolecularProperties[]

    for (_, mol) in MOLECULAR_DB
        val = getfield(mol, property)
        if val isa Number && !isnan(val) && min_val <= val <= max_val
            push!(results, mol)
        end
    end

    return results
end

"""
    filter_druglike(molecules; strict=false)

Filter molecules that pass drug-likeness rules.
"""
function filter_druglike(molecules::Vector{MolecularProperties}; strict::Bool=false)
    druglike = MolecularProperties[]

    for mol in molecules
        result = calculate_druglikeness(mol)
        if strict
            if result.lipinski.passed && result.veber.passed
                push!(druglike, mol)
            end
        else
            if result.lipinski.violations <= 1
                push!(druglike, mol)
            end
        end
    end

    return druglike
end

"""
    identify_functional_groups(smiles)

Identify functional groups in SMILES string (basic pattern matching).
"""
function identify_functional_groups(smiles::String)
    groups = String[]

    # Pattern matching for common groups
    patterns = Dict(
        "carboxylic_acid" => r"C\(=O\)O[H]?(?![A-Z])",
        "ester" => r"C\(=O\)O[A-Z]",
        "amide" => r"C\(=O\)N",
        "aldehyde" => r"C=O$|C=O(?=[^O])",
        "ketone" => r"CC\(=O\)C",
        "hydroxyl" => r"[^=]O[H]?(?![A-Z])",
        "amine" => r"N(?!\(=O\))",
        "phenyl" => r"c1ccccc1",
        "fluorine" => r"F",
        "chlorine" => r"Cl",
        "bromine" => r"Br",
        "nitro" => r"N\(=O\)=O",
        "sulfone" => r"S\(=O\)\(=O\)",
        "phosphate" => r"P\(=O\)",
    )

    for (name, pattern) in patterns
        if occursin(pattern, smiles)
            push!(groups, name)
        end
    end

    return groups
end

end # module
