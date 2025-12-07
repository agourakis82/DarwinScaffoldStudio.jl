"""
    BiomarkersLibrary

Comprehensive biomarkers database for tissue engineering.

Contains:
- Gene expression markers (osteogenic, chondrogenic, adipogenic, etc.)
- Protein markers (ECM, growth factors, cytokines)
- Surface markers (CD markers for cell identification)
- Functional assays and endpoints
- Histological stains and scoring systems
- Imaging biomarkers
- Clinical outcome measures

Data sources:
- NCBI Gene database
- UniProt protein database
- HUGO Gene Nomenclature Committee
- Published Q1 literature

Author: Dr. Demetrios Agourakis
"""
module BiomarkersLibrary

export Biomarker, GeneMarker, ProteinMarker, SurfaceMarker
export FunctionalAssay, HistologicalScore
export OSTEOGENIC_MARKERS, CHONDROGENIC_MARKERS, ADIPOGENIC_MARKERS
export ANGIOGENIC_MARKERS, INFLAMMATORY_MARKERS, NEURAL_MARKERS
export STEM_CELL_MARKERS, CARDIAC_MARKERS, SKIN_MARKERS
export FUNCTIONAL_ASSAYS, HISTOLOGICAL_SCORES
export get_markers_for_tissue, get_differentiation_panel
export design_qpcr_panel, suggest_assay_timepoints
export REFERENCE_GENES, ASSAY_PROTOCOLS

# =============================================================================
# Biomarker Structures
# =============================================================================

"""Gene expression biomarker."""
struct GeneMarker
    symbol::String              # Gene symbol (e.g., "RUNX2")
    name::String               # Full name
    ncbi_id::String            # NCBI Gene ID
    ensembl_id::String         # Ensembl ID
    function_::String          # Biological function
    expression_pattern::String # When/where expressed
    fold_change_threshold::Float64  # Typical significant change
    timepoint_days::Vector{Int}     # Optimal measurement times
    primer_forward::String     # Example qPCR primer
    primer_reverse::String
end

function GeneMarker(symbol::String, name::String;
    ncbi_id::String="",
    ensembl_id::String="",
    function_::String="",
    expression_pattern::String="",
    fold_change_threshold::Float64=2.0,
    timepoint_days::Vector{Int}=Int[],
    primer_forward::String="",
    primer_reverse::String="")

    GeneMarker(symbol, name, ncbi_id, ensembl_id, function_,
        expression_pattern, fold_change_threshold, timepoint_days,
        primer_forward, primer_reverse)
end

"""Protein biomarker."""
struct ProteinMarker
    name::String               # Protein name
    gene_symbol::String        # Encoding gene
    uniprot_id::String         # UniProt ID
    molecular_weight_kda::Float64
    function_::String
    detection_method::Vector{Symbol}  # :elisa, :western, :ihc, :icc
    normal_range::String       # Reference range
    unit::String               # Measurement unit
end

function ProteinMarker(name::String, gene_symbol::String;
    uniprot_id::String="",
    molecular_weight_kda::Float64=NaN,
    function_::String="",
    detection_method::Vector{Symbol}=Symbol[],
    normal_range::String="",
    unit::String="")

    ProteinMarker(name, gene_symbol, uniprot_id, molecular_weight_kda,
        function_, detection_method, normal_range, unit)
end

"""Surface marker (for FACS)."""
struct SurfaceMarker
    cd_number::String          # CD designation
    name::String               # Full name
    gene_symbol::String
    cell_types::Vector{Symbol} # Cells expressing this marker
    expression_level::Symbol   # :high, :low, :negative
    function_::String
end

"""Functional assay."""
struct FunctionalAssay
    name::String
    target::Symbol             # What it measures
    method::String             # How it's performed
    readout::String            # What's measured
    unit::String
    positive_threshold::Float64
    timepoints_days::Vector{Int}
    controls::Vector{String}
    references::Vector{String}
end

"""Histological scoring system."""
struct HistologicalScore
    name::String
    tissue_type::Symbol
    parameters::Vector{String}
    score_range::Tuple{Int,Int}
    description::String
    references::Vector{String}
end

# =============================================================================
# Osteogenic Differentiation Markers
# =============================================================================

const OSTEOGENIC_MARKERS = Dict{String,GeneMarker}(
    # Early markers (Days 3-7)
    "RUNX2" => GeneMarker("RUNX2", "Runt-related transcription factor 2";
        ncbi_id="860", ensembl_id="ENSG00000124813",
        function_="Master regulator of osteogenesis, activates osteoblast genes",
        expression_pattern="Early osteogenic (day 3-14), peaks day 7",
        fold_change_threshold=3.0,
        timepoint_days=[3, 7, 14]
    ),

    "OSTERIX" => GeneMarker("SP7", "Osterix/Sp7 transcription factor";
        ncbi_id="121340", ensembl_id="ENSG00000170374",
        function_="Downstream of RUNX2, essential for osteoblast maturation",
        expression_pattern="Early-mid osteogenic (day 7-14)",
        fold_change_threshold=3.0,
        timepoint_days=[7, 14]
    ),

    "ALP" => GeneMarker("ALPL", "Alkaline phosphatase";
        ncbi_id="249", ensembl_id="ENSG00000162551",
        function_="Hydrolysis of phosphate esters, mineralization",
        expression_pattern="Early marker (day 7-14), decreases with maturation",
        fold_change_threshold=5.0,
        timepoint_days=[7, 14, 21]
    ),

    "COL1A1" => GeneMarker("COL1A1", "Collagen type I alpha 1";
        ncbi_id="1277", ensembl_id="ENSG00000108821",
        function_="Main structural protein of bone matrix",
        expression_pattern="Constitutive, increases with osteogenesis",
        fold_change_threshold=2.0,
        timepoint_days=[7, 14, 21]
    ),

    # Mid-stage markers (Days 14-21)
    "IBSP" => GeneMarker("IBSP", "Bone sialoprotein";
        ncbi_id="3381", ensembl_id="ENSG00000029559",
        function_="Nucleation of hydroxyapatite, cell adhesion",
        expression_pattern="Mid-late osteogenic (day 14-28)",
        fold_change_threshold=5.0,
        timepoint_days=[14, 21, 28]
    ),

    "SPP1" => GeneMarker("SPP1", "Osteopontin";
        ncbi_id="6696", ensembl_id="ENSG00000118785",
        function_="Cell adhesion, mineralization regulation",
        expression_pattern="Mid-late osteogenic",
        fold_change_threshold=3.0,
        timepoint_days=[14, 21]
    ),

    # Late markers (Days 21-28)
    "BGLAP" => GeneMarker("BGLAP", "Osteocalcin";
        ncbi_id="632", ensembl_id="ENSG00000176725",
        function_="Calcium binding, marker of mature osteoblasts",
        expression_pattern="Late osteogenic, mature osteoblasts only",
        fold_change_threshold=10.0,
        timepoint_days=[21, 28]
    ),

    "DMP1" => GeneMarker("DMP1", "Dentin matrix protein 1";
        ncbi_id="1758", ensembl_id="ENSG00000152592",
        function_="Osteocyte marker, mineralization regulation",
        expression_pattern="Very late, osteocyte marker",
        fold_change_threshold=5.0,
        timepoint_days=[28, 42]
    ),

    "SOST" => GeneMarker("SOST", "Sclerostin";
        ncbi_id="50964", ensembl_id="ENSG00000167941",
        function_="Osteocyte marker, Wnt inhibitor",
        expression_pattern="Mature osteocyte only",
        fold_change_threshold=5.0,
        timepoint_days=[42, 56]
    )
)

# =============================================================================
# Chondrogenic Differentiation Markers
# =============================================================================

const CHONDROGENIC_MARKERS = Dict{String,GeneMarker}(
    # Early markers
    "SOX9" => GeneMarker("SOX9", "SRY-box transcription factor 9";
        ncbi_id="6662", ensembl_id="ENSG00000125398",
        function_="Master regulator of chondrogenesis",
        expression_pattern="Early chondrogenic, maintained in mature cartilage",
        fold_change_threshold=3.0,
        timepoint_days=[3, 7, 14, 21]
    ),

    "SOX5" => GeneMarker("SOX5", "SRY-box transcription factor 5";
        ncbi_id="6660", ensembl_id="ENSG00000134532",
        function_="Works with SOX9 in chondrogenesis",
        expression_pattern="Early-mid chondrogenic",
        fold_change_threshold=2.0,
        timepoint_days=[7, 14]
    ),

    "SOX6" => GeneMarker("SOX6", "SRY-box transcription factor 6";
        ncbi_id="55553", ensembl_id="ENSG00000110693",
        function_="Works with SOX9 in chondrogenesis",
        expression_pattern="Early-mid chondrogenic",
        fold_change_threshold=2.0,
        timepoint_days=[7, 14]
    ),

    # Matrix markers
    "COL2A1" => GeneMarker("COL2A1", "Collagen type II alpha 1";
        ncbi_id="1280", ensembl_id="ENSG00000139219",
        function_="Main collagen of hyaline cartilage",
        expression_pattern="Specific to chondrocytes",
        fold_change_threshold=10.0,
        timepoint_days=[14, 21, 28]
    ),

    "ACAN" => GeneMarker("ACAN", "Aggrecan";
        ncbi_id="176", ensembl_id="ENSG00000157766",
        function_="Major proteoglycan of cartilage",
        expression_pattern="Chondrocyte-specific",
        fold_change_threshold=5.0,
        timepoint_days=[14, 21, 28]
    ),

    "COMP" => GeneMarker("COMP", "Cartilage oligomeric matrix protein";
        ncbi_id="1311", ensembl_id="ENSG00000105664",
        function_="ECM organization, collagen assembly",
        expression_pattern="Cartilage-specific",
        fold_change_threshold=5.0,
        timepoint_days=[14, 21]
    ),

    # Hypertrophic markers (negative for healthy cartilage)
    "COL10A1" => GeneMarker("COL10A1", "Collagen type X alpha 1";
        ncbi_id="1300", ensembl_id="ENSG00000123610",
        function_="Hypertrophic chondrocyte marker",
        expression_pattern="Hypertrophic zone, endochondral ossification",
        fold_change_threshold=2.0,
        timepoint_days=[21, 28]
    ),

    "MMP13" => GeneMarker("MMP13", "Matrix metalloproteinase 13";
        ncbi_id="4322", ensembl_id="ENSG00000137745",
        function_="Collagen degradation, cartilage breakdown",
        expression_pattern="Hypertrophy, OA marker",
        fold_change_threshold=2.0,
        timepoint_days=[21, 28]
    ),

    "RUNX2" => GeneMarker("RUNX2", "Runt-related transcription factor 2";
        ncbi_id="860",
        function_="Hypertrophy marker (negative in healthy cartilage)",
        expression_pattern="Should be low for healthy cartilage",
        fold_change_threshold=2.0,
        timepoint_days=[14, 21, 28]
    )
)

# =============================================================================
# Adipogenic Differentiation Markers
# =============================================================================

const ADIPOGENIC_MARKERS = Dict{String,GeneMarker}(
    "PPARG" => GeneMarker("PPARG", "Peroxisome proliferator-activated receptor gamma";
        ncbi_id="5468", ensembl_id="ENSG00000132170",
        function_="Master regulator of adipogenesis",
        expression_pattern="Early adipogenic, maintained",
        fold_change_threshold=5.0,
        timepoint_days=[3, 7, 14]
    ),

    "CEBPA" => GeneMarker("CEBPA", "CCAAT enhancer binding protein alpha";
        ncbi_id="1050", ensembl_id="ENSG00000245848",
        function_="Adipogenic transcription factor",
        expression_pattern="Early-mid adipogenic",
        fold_change_threshold=3.0,
        timepoint_days=[3, 7]
    ),

    "FABP4" => GeneMarker("FABP4", "Fatty acid binding protein 4";
        ncbi_id="2167", ensembl_id="ENSG00000170323",
        function_="Fatty acid transport in adipocytes",
        expression_pattern="Mature adipocyte marker",
        fold_change_threshold=10.0,
        timepoint_days=[7, 14, 21]
    ),

    "ADIPOQ" => GeneMarker("ADIPOQ", "Adiponectin";
        ncbi_id="9370", ensembl_id="ENSG00000181092",
        function_="Adipokine, metabolic regulation",
        expression_pattern="Mature adipocyte marker",
        fold_change_threshold=10.0,
        timepoint_days=[14, 21]
    ),

    "LPL" => GeneMarker("LPL", "Lipoprotein lipase";
        ncbi_id="4023", ensembl_id="ENSG00000175445",
        function_="Lipid metabolism",
        expression_pattern="Mid-late adipogenic",
        fold_change_threshold=5.0,
        timepoint_days=[7, 14]
    ),

    "LEP" => GeneMarker("LEP", "Leptin";
        ncbi_id="3952", ensembl_id="ENSG00000174697",
        function_="Adipokine, energy homeostasis",
        expression_pattern="Mature adipocytes",
        fold_change_threshold=5.0,
        timepoint_days=[14, 21]
    )
)

# =============================================================================
# Angiogenic Markers
# =============================================================================

const ANGIOGENIC_MARKERS = Dict{String,GeneMarker}(
    "VEGFA" => GeneMarker("VEGFA", "Vascular endothelial growth factor A";
        ncbi_id="7422", ensembl_id="ENSG00000112715",
        function_="Major angiogenic factor",
        expression_pattern="Hypoxia-induced, pro-angiogenic",
        fold_change_threshold=3.0,
        timepoint_days=[3, 7, 14]
    ),

    "KDR" => GeneMarker("KDR", "VEGF receptor 2";
        ncbi_id="3791", ensembl_id="ENSG00000128052",
        function_="Main VEGF signaling receptor",
        expression_pattern="Endothelial cells",
        fold_change_threshold=2.0,
        timepoint_days=[7, 14]
    ),

    "PECAM1" => GeneMarker("PECAM1", "CD31/Platelet endothelial cell adhesion molecule";
        ncbi_id="5175", ensembl_id="ENSG00000261371",
        function_="Endothelial cell marker, adhesion",
        expression_pattern="Endothelial-specific",
        fold_change_threshold=2.0,
        timepoint_days=[7, 14, 21]
    ),

    "CDH5" => GeneMarker("CDH5", "VE-Cadherin";
        ncbi_id="1003", ensembl_id="ENSG00000179776",
        function_="Endothelial junction protein",
        expression_pattern="Endothelial-specific",
        fold_change_threshold=2.0,
        timepoint_days=[7, 14]
    ),

    "ANGPT1" => GeneMarker("ANGPT1", "Angiopoietin 1";
        ncbi_id="284", ensembl_id="ENSG00000154188",
        function_="Vessel maturation and stabilization",
        expression_pattern="Vessel maturation phase",
        fold_change_threshold=2.0,
        timepoint_days=[14, 21]
    ),

    "ANGPT2" => GeneMarker("ANGPT2", "Angiopoietin 2";
        ncbi_id="285", ensembl_id="ENSG00000091879",
        function_="Vessel destabilization for sprouting",
        expression_pattern="Active angiogenesis",
        fold_change_threshold=2.0,
        timepoint_days=[7, 14]
    ),

    "TEK" => GeneMarker("TEK", "Tie2 receptor";
        ncbi_id="7010", ensembl_id="ENSG00000120156",
        function_="Angiopoietin receptor",
        expression_pattern="Endothelial cells",
        fold_change_threshold=2.0,
        timepoint_days=[7, 14]
    ),

    "vWF" => GeneMarker("VWF", "von Willebrand factor";
        ncbi_id="7450", ensembl_id="ENSG00000110799",
        function_="Endothelial marker, hemostasis",
        expression_pattern="Mature endothelial cells",
        fold_change_threshold=3.0,
        timepoint_days=[14, 21]
    )
)

# =============================================================================
# Inflammatory Markers
# =============================================================================

const INFLAMMATORY_MARKERS = Dict{String,GeneMarker}(
    # Pro-inflammatory
    "IL1B" => GeneMarker("IL1B", "Interleukin 1 beta";
        ncbi_id="3553", ensembl_id="ENSG00000125538",
        function_="Pro-inflammatory cytokine",
        expression_pattern="Acute inflammation",
        fold_change_threshold=5.0,
        timepoint_days=[1, 3, 7]
    ),

    "IL6" => GeneMarker("IL6", "Interleukin 6";
        ncbi_id="3569", ensembl_id="ENSG00000136244",
        function_="Pro-inflammatory, acute phase response",
        expression_pattern="Acute inflammation",
        fold_change_threshold=5.0,
        timepoint_days=[1, 3, 7]
    ),

    "TNF" => GeneMarker("TNF", "Tumor necrosis factor";
        ncbi_id="7124", ensembl_id="ENSG00000232810",
        function_="Pro-inflammatory cytokine",
        expression_pattern="Acute inflammation",
        fold_change_threshold=3.0,
        timepoint_days=[1, 3, 7]
    ),

    "CXCL8" => GeneMarker("CXCL8", "Interleukin 8";
        ncbi_id="3576", ensembl_id="ENSG00000169429",
        function_="Neutrophil chemotaxis",
        expression_pattern="Acute inflammation",
        fold_change_threshold=5.0,
        timepoint_days=[1, 3]
    ),

    # Anti-inflammatory / Resolution
    "IL10" => GeneMarker("IL10", "Interleukin 10";
        ncbi_id="3586", ensembl_id="ENSG00000136634",
        function_="Anti-inflammatory cytokine",
        expression_pattern="Resolution phase",
        fold_change_threshold=3.0,
        timepoint_days=[3, 7, 14]
    ),

    "TGFB1" => GeneMarker("TGFB1", "Transforming growth factor beta 1";
        ncbi_id="7040", ensembl_id="ENSG00000105329",
        function_="Anti-inflammatory, fibrotic",
        expression_pattern="Resolution and fibrosis",
        fold_change_threshold=2.0,
        timepoint_days=[7, 14, 21]
    ),

    # Macrophage polarization
    "ARG1" => GeneMarker("ARG1", "Arginase 1";
        ncbi_id="383", ensembl_id="ENSG00000118520",
        function_="M2 macrophage marker",
        expression_pattern="Anti-inflammatory macrophages",
        fold_change_threshold=3.0,
        timepoint_days=[3, 7, 14]
    ),

    "NOS2" => GeneMarker("NOS2", "Inducible nitric oxide synthase";
        ncbi_id="4843", ensembl_id="ENSG00000007171",
        function_="M1 macrophage marker",
        expression_pattern="Pro-inflammatory macrophages",
        fold_change_threshold=5.0,
        timepoint_days=[1, 3, 7]
    ),

    "CD163" => GeneMarker("CD163", "Hemoglobin scavenger receptor";
        ncbi_id="9332", ensembl_id="ENSG00000177575",
        function_="M2 macrophage marker",
        expression_pattern="Anti-inflammatory macrophages",
        fold_change_threshold=2.0,
        timepoint_days=[7, 14, 21]
    ),

    "CD206" => GeneMarker("MRC1", "Mannose receptor C-type 1";
        ncbi_id="4360", ensembl_id="ENSG00000260314",
        function_="M2 macrophage marker",
        expression_pattern="Anti-inflammatory macrophages",
        fold_change_threshold=2.0,
        timepoint_days=[7, 14, 21]
    )
)

# =============================================================================
# Neural Markers
# =============================================================================

const NEURAL_MARKERS = Dict{String,GeneMarker}(
    # Neural stem/progenitor
    "NES" => GeneMarker("NES", "Nestin";
        ncbi_id="10763", ensembl_id="ENSG00000132688",
        function_="Neural stem/progenitor marker",
        expression_pattern="Neural progenitors",
        fold_change_threshold=3.0,
        timepoint_days=[3, 7]
    ),

    "SOX2" => GeneMarker("SOX2", "SRY-box transcription factor 2";
        ncbi_id="6657", ensembl_id="ENSG00000181449",
        function_="Neural stem cell marker, pluripotency",
        expression_pattern="Neural progenitors",
        fold_change_threshold=2.0,
        timepoint_days=[3, 7]
    ),

    # Neuronal markers
    "TUBB3" => GeneMarker("TUBB3", "Beta-III tubulin";
        ncbi_id="10381", ensembl_id="ENSG00000258947",
        function_="Neuronal cytoskeleton marker",
        expression_pattern="Early neurons",
        fold_change_threshold=3.0,
        timepoint_days=[7, 14]
    ),

    "MAP2" => GeneMarker("MAP2", "Microtubule-associated protein 2";
        ncbi_id="4133", ensembl_id="ENSG00000078018",
        function_="Mature neuron marker, dendrites",
        expression_pattern="Mature neurons",
        fold_change_threshold=3.0,
        timepoint_days=[14, 21]
    ),

    "RBFOX3" => GeneMarker("RBFOX3", "NeuN";
        ncbi_id="146713", ensembl_id="ENSG00000167281",
        function_="Mature neuron marker",
        expression_pattern="Post-mitotic neurons",
        fold_change_threshold=5.0,
        timepoint_days=[14, 21, 28]
    ),

    "SYP" => GeneMarker("SYP", "Synaptophysin";
        ncbi_id="6855", ensembl_id="ENSG00000102003",
        function_="Synaptic marker",
        expression_pattern="Synaptic maturation",
        fold_change_threshold=3.0,
        timepoint_days=[21, 28]
    ),

    # Glial markers
    "GFAP" => GeneMarker("GFAP", "Glial fibrillary acidic protein";
        ncbi_id="2670", ensembl_id="ENSG00000131095",
        function_="Astrocyte marker",
        expression_pattern="Astrocytes",
        fold_change_threshold=5.0,
        timepoint_days=[14, 21]
    ),

    "MBP" => GeneMarker("MBP", "Myelin basic protein";
        ncbi_id="4155", ensembl_id="ENSG00000197971",
        function_="Oligodendrocyte/Schwann cell marker",
        expression_pattern="Myelinating glia",
        fold_change_threshold=10.0,
        timepoint_days=[21, 28]
    ),

    "S100B" => GeneMarker("S100B", "S100 calcium-binding protein B";
        ncbi_id="6285", ensembl_id="ENSG00000160307",
        function_="Schwann cell marker",
        expression_pattern="Schwann cells, astrocytes",
        fold_change_threshold=3.0,
        timepoint_days=[7, 14, 21]
    )
)

# =============================================================================
# Stem Cell Markers (for MSC characterization)
# =============================================================================

const STEM_CELL_MARKERS = Dict{String,SurfaceMarker}(
    # Positive markers (ISCT criteria)
    "CD73" => SurfaceMarker("CD73", "Ecto-5'-nucleotidase", "NT5E",
        [:msc], :high, "Adenosine production, immunomodulation"),

    "CD90" => SurfaceMarker("CD90", "Thy-1", "THY1",
        [:msc, :fibroblast], :high, "Cell adhesion, signaling"),

    "CD105" => SurfaceMarker("CD105", "Endoglin", "ENG",
        [:msc, :endothelial], :high, "TGF-beta co-receptor"),

    # Negative markers (ISCT criteria)
    "CD34" => SurfaceMarker("CD34", "Hematopoietic progenitor antigen", "CD34",
        [:hsc, :endothelial], :negative, "Should be negative on MSC"),

    "CD45" => SurfaceMarker("CD45", "Leukocyte common antigen", "PTPRC",
        [:hematopoietic], :negative, "Should be negative on MSC"),

    "CD14" => SurfaceMarker("CD14", "Monocyte differentiation antigen", "CD14",
        [:monocyte, :macrophage], :negative, "Should be negative on MSC"),

    "CD19" => SurfaceMarker("CD19", "B-lymphocyte antigen", "CD19",
        [:b_cell], :negative, "Should be negative on MSC"),

    "HLA-DR" => SurfaceMarker("HLA-DR", "MHC class II", "HLA-DRA",
        [:antigen_presenting], :negative, "Should be negative on MSC"),

    # Pluripotency markers
    "SSEA-4" => SurfaceMarker("SSEA-4", "Stage-specific embryonic antigen 4", "",
        [:esc, :ipsc], :high, "Pluripotency marker"),

    "TRA-1-60" => SurfaceMarker("TRA-1-60", "Podocalyxin", "PODXL",
        [:esc, :ipsc], :high, "Pluripotency marker"),

    "TRA-1-81" => SurfaceMarker("TRA-1-81", "Podocalyxin", "PODXL",
        [:esc, :ipsc], :high, "Pluripotency marker")
)

# =============================================================================
# Cardiac Markers
# =============================================================================

const CARDIAC_MARKERS = Dict{String,GeneMarker}(
    "TNNT2" => GeneMarker("TNNT2", "Cardiac troponin T";
        ncbi_id="7139", ensembl_id="ENSG00000118194",
        function_="Cardiomyocyte-specific, contractile protein",
        expression_pattern="Cardiomyocytes only",
        fold_change_threshold=10.0,
        timepoint_days=[7, 14, 21]
    ),

    "TNNI3" => GeneMarker("TNNI3", "Cardiac troponin I";
        ncbi_id="7137", ensembl_id="ENSG00000129991",
        function_="Cardiomyocyte marker",
        expression_pattern="Cardiomyocytes",
        fold_change_threshold=10.0,
        timepoint_days=[14, 21]
    ),

    "MYH6" => GeneMarker("MYH6", "Alpha myosin heavy chain";
        ncbi_id="4624", ensembl_id="ENSG00000197616",
        function_="Atrial/fetal cardiomyocyte marker",
        expression_pattern="Atrial, early cardiac",
        fold_change_threshold=5.0,
        timepoint_days=[7, 14]
    ),

    "MYH7" => GeneMarker("MYH7", "Beta myosin heavy chain";
        ncbi_id="4625", ensembl_id="ENSG00000092054",
        function_="Ventricular cardiomyocyte marker",
        expression_pattern="Ventricular, mature",
        fold_change_threshold=5.0,
        timepoint_days=[14, 21, 28]
    ),

    "NKX2-5" => GeneMarker("NKX2-5", "NK2 homeobox 5";
        ncbi_id="1482", ensembl_id="ENSG00000183072",
        function_="Cardiac transcription factor",
        expression_pattern="Early cardiac progenitor",
        fold_change_threshold=5.0,
        timepoint_days=[3, 7, 14]
    ),

    "GATA4" => GeneMarker("GATA4", "GATA binding protein 4";
        ncbi_id="2626", ensembl_id="ENSG00000136574",
        function_="Cardiac transcription factor",
        expression_pattern="Cardiac development",
        fold_change_threshold=3.0,
        timepoint_days=[3, 7, 14]
    ),

    "GJA1" => GeneMarker("GJA1", "Connexin 43";
        ncbi_id="2697", ensembl_id="ENSG00000152661",
        function_="Gap junction, electrical coupling",
        expression_pattern="Mature cardiomyocytes",
        fold_change_threshold=2.0,
        timepoint_days=[14, 21, 28]
    )
)

# =============================================================================
# Skin/Wound Healing Markers
# =============================================================================

const SKIN_MARKERS = Dict{String,GeneMarker}(
    "KRT14" => GeneMarker("KRT14", "Keratin 14";
        ncbi_id="3861", ensembl_id="ENSG00000186847",
        function_="Basal keratinocyte marker",
        expression_pattern="Epidermal stem cells, basal layer",
        fold_change_threshold=2.0,
        timepoint_days=[7, 14]
    ),

    "KRT10" => GeneMarker("KRT10", "Keratin 10";
        ncbi_id="3858", ensembl_id="ENSG00000186395",
        function_="Suprabasal keratinocyte marker",
        expression_pattern="Differentiating keratinocytes",
        fold_change_threshold=3.0,
        timepoint_days=[14, 21]
    ),

    "IVL" => GeneMarker("IVL", "Involucrin";
        ncbi_id="3713", ensembl_id="ENSG00000163207",
        function_="Terminal differentiation marker",
        expression_pattern="Granular layer",
        fold_change_threshold=5.0,
        timepoint_days=[14, 21]
    ),

    "FLG" => GeneMarker("FLG", "Filaggrin";
        ncbi_id="2312", ensembl_id="ENSG00000143631",
        function_="Stratum corneum formation",
        expression_pattern="Terminal differentiation",
        fold_change_threshold=10.0,
        timepoint_days=[21, 28]
    ),

    "COL3A1" => GeneMarker("COL3A1", "Collagen type III alpha 1";
        ncbi_id="1281", ensembl_id="ENSG00000168542",
        function_="Early wound healing, scar",
        expression_pattern="Fibroblasts, granulation tissue",
        fold_change_threshold=3.0,
        timepoint_days=[7, 14, 21]
    ),

    "ACTA2" => GeneMarker("ACTA2", "Alpha smooth muscle actin";
        ncbi_id="59", ensembl_id="ENSG00000107796",
        function_="Myofibroblast marker",
        expression_pattern="Wound contraction phase",
        fold_change_threshold=5.0,
        timepoint_days=[7, 14]
    )
)

# =============================================================================
# Reference/Housekeeping Genes
# =============================================================================

const REFERENCE_GENES = Dict{String,GeneMarker}(
    "GAPDH" => GeneMarker("GAPDH", "Glyceraldehyde-3-phosphate dehydrogenase";
        ncbi_id="2597", ensembl_id="ENSG00000111640",
        function_="Housekeeping, glycolysis",
        expression_pattern="Ubiquitous (but variable with hypoxia)"
    ),

    "ACTB" => GeneMarker("ACTB", "Beta-actin";
        ncbi_id="60", ensembl_id="ENSG00000075624",
        function_="Housekeeping, cytoskeleton",
        expression_pattern="Ubiquitous (but affected by cytoskeletal changes)"
    ),

    "B2M" => GeneMarker("B2M", "Beta-2-microglobulin";
        ncbi_id="567", ensembl_id="ENSG00000166710",
        function_="Housekeeping, MHC component",
        expression_pattern="Ubiquitous"
    ),

    "RPLP0" => GeneMarker("RPLP0", "Ribosomal protein lateral stalk subunit P0";
        ncbi_id="6175", ensembl_id="ENSG00000089157",
        function_="Housekeeping, ribosomal protein",
        expression_pattern="Ubiquitous, stable"
    ),

    "HPRT1" => GeneMarker("HPRT1", "Hypoxanthine phosphoribosyltransferase 1";
        ncbi_id="3251", ensembl_id="ENSG00000165704",
        function_="Housekeeping, purine salvage",
        expression_pattern="Ubiquitous, very stable"
    ),

    "TBP" => GeneMarker("TBP", "TATA-box binding protein";
        ncbi_id="6908", ensembl_id="ENSG00000112592",
        function_="Housekeeping, transcription",
        expression_pattern="Ubiquitous, most stable reference"
    )
)

# =============================================================================
# Functional Assays
# =============================================================================

const FUNCTIONAL_ASSAYS = Dict{Symbol,FunctionalAssay}(
    :alizarin_red => FunctionalAssay(
        "Alizarin Red S Staining",
        :mineralization,
        "Stain calcium deposits with Alizarin Red S dye, quantify by extraction",
        "Optical density at 405 nm",
        "OD/mg protein",
        0.5,
        [14, 21, 28],
        ["Undifferentiated control", "Osteogenic positive control"],
        ["Gregory 2004 Analytical Biochemistry"]
    ),

    :alp_activity => FunctionalAssay(
        "Alkaline Phosphatase Activity",
        :osteogenic_differentiation,
        "Incubate with p-nitrophenyl phosphate substrate",
        "p-Nitrophenol production",
        "nmol pNP/min/mg protein",
        10.0,
        [7, 14],
        ["Undifferentiated control"],
        ["Golub 1992 Connect Tissue Res"]
    ),

    :oil_red_o => FunctionalAssay(
        "Oil Red O Staining",
        :lipid_accumulation,
        "Stain neutral lipids with Oil Red O",
        "Optical density at 510 nm",
        "OD/well",
        0.3,
        [14, 21],
        ["Undifferentiated control", "Adipogenic positive control"],
        ["Ramírez-Zacarías 1992 Histochemistry"]
    ),

    :alcian_blue => FunctionalAssay(
        "Alcian Blue Staining",
        :gag_production,
        "Stain sulfated GAGs with Alcian Blue at pH 2.5",
        "Optical density at 620 nm",
        "OD/pellet",
        0.5,
        [14, 21, 28],
        ["Undifferentiated control"],
        ["Lev 1964 J Histochem Cytochem"]
    ),

    :dmmb_gag => FunctionalAssay(
        "DMMB GAG Assay",
        :gag_quantification,
        "React with dimethylmethylene blue dye",
        "Absorbance at 525 nm",
        "μg GAG/μg DNA",
        10.0,
        [14, 21, 28],
        ["Chondroitin sulfate standard curve"],
        ["Farndale 1986 Biochim Biophys Acta"]
    ),

    :hydroxyproline => FunctionalAssay(
        "Hydroxyproline Assay",
        :collagen_content,
        "Acid hydrolysis followed by colorimetric detection",
        "Absorbance at 560 nm",
        "μg hydroxyproline/mg dry weight",
        5.0,
        [21, 28, 42],
        ["Hydroxyproline standard curve"],
        ["Reddy 1996 Clinical Biochemistry"]
    ),

    :live_dead => FunctionalAssay(
        "Live/Dead Assay",
        :viability,
        "Calcein-AM (live) and EthD-1 (dead) staining",
        "Green/Red fluorescence ratio",
        "% viability",
        80.0,
        [1, 3, 7, 14],
        ["Positive control (100% live)", "Negative control (methanol-fixed)"],
        ["Molecular Probes protocol"]
    ),

    :mtt => FunctionalAssay(
        "MTT Assay",
        :metabolic_activity,
        "Reduction of MTT to formazan by mitochondria",
        "Absorbance at 570 nm",
        "OD",
        0.5,
        [1, 3, 7, 14, 21],
        ["Cells only (100%)", "Media blank"],
        ["Mosmann 1983 J Immunol Methods"]
    ),

    :tube_formation => FunctionalAssay(
        "Tube Formation Assay",
        :angiogenesis,
        "Endothelial cells on Matrigel form tube networks",
        "Total tube length, branch points",
        "mm/field",
        5.0,
        [4, 8, 16, 24],  # Hours
        ["VEGF positive control", "Suramin negative control"],
        ["Arnaoutova 2009 Nat Protoc"]
    ),

    :scratch_wound => FunctionalAssay(
        "Scratch Wound Assay",
        :migration,
        "Create scratch in monolayer, measure closure",
        "Wound closure percentage",
        "% closure",
        50.0,
        [6, 12, 24, 48],  # Hours
        ["Mitomycin C-treated (migration only)"],
        ["Liang 2007 Nat Protoc"]
    )
)

# =============================================================================
# Histological Scoring Systems
# =============================================================================

const HISTOLOGICAL_SCORES = Dict{Symbol,HistologicalScore}(
    :bern_score => HistologicalScore(
        "Bern Score",
        :cartilage,
        ["Uniformity", "Matrix staining", "Cell morphology", "Surface"],
        (0, 9),
        "Histological grading for cartilage tissue engineering",
        ["Grogan 2006 Tissue Eng"]
    ),

    :icrs_score => HistologicalScore(
        "ICRS Visual Histological Assessment",
        :cartilage,
        ["Surface", "Matrix", "Cell distribution", "Cell viability",
         "Subchondral bone", "Cartilage mineralization"],
        (0, 18),
        "International Cartilage Repair Society scoring",
        ["Mainil-Varlet 2003 Osteoarthritis Cartilage"]
    ),

    :oarsi_score => HistologicalScore(
        "OARSI Histopathology",
        :osteoarthritis,
        ["Structural damage", "Proteoglycan loss", "Cellularity",
         "Tidemark integrity", "Subchondral changes"],
        (0, 24),
        "Osteoarthritis Research Society International grading",
        ["Pritzker 2006 Osteoarthritis Cartilage"]
    ),

    :lane_sandhu => HistologicalScore(
        "Lane-Sandhu Score",
        :bone,
        ["Union", "Spongiosa", "Cortex", "Bone marrow"],
        (0, 12),
        "Bone healing assessment",
        ["Lane 1987 CORR"]
    ),

    :emery_score => HistologicalScore(
        "Emery Score",
        :bone,
        ["Bone formation", "Cartilage formation", "Fibrous tissue",
         "Marrow formation"],
        (0, 8),
        "Bone defect healing score",
        ["Emery 1999 Calcif Tissue Int"]
    )
)

# =============================================================================
# Helper Functions
# =============================================================================

"""Get markers appropriate for a tissue type."""
function get_markers_for_tissue(tissue::Symbol)
    markers = if tissue == :bone
        OSTEOGENIC_MARKERS
    elseif tissue == :cartilage
        CHONDROGENIC_MARKERS
    elseif tissue == :fat
        ADIPOGENIC_MARKERS
    elseif tissue == :vessel || tissue == :vascular
        ANGIOGENIC_MARKERS
    elseif tissue == :nerve || tissue == :neural
        NEURAL_MARKERS
    elseif tissue == :heart || tissue == :cardiac
        CARDIAC_MARKERS
    elseif tissue == :skin
        SKIN_MARKERS
    elseif tissue == :inflammation
        INFLAMMATORY_MARKERS
    else
        Dict{String,GeneMarker}()
    end

    return markers
end

"""Design qPCR panel for differentiation assessment."""
function design_qpcr_panel(tissue::Symbol;
                           include_reference::Bool=true,
                           max_genes::Int=10)
    tissue_markers = get_markers_for_tissue(tissue)

    # Select key markers
    selected = collect(values(tissue_markers))

    # Sort by importance (fold change threshold as proxy)
    sort!(selected, by=m -> m.fold_change_threshold, rev=true)

    # Limit to max_genes
    if length(selected) > max_genes
        selected = selected[1:max_genes]
    end

    # Add reference genes
    if include_reference
        refs = [REFERENCE_GENES["TBP"], REFERENCE_GENES["HPRT1"]]
        append!(selected, refs)
    end

    return selected
end

"""Suggest optimal timepoints for marker assessment."""
function suggest_assay_timepoints(tissue::Symbol)
    markers = get_markers_for_tissue(tissue)

    # Collect all timepoints
    all_timepoints = Set{Int}()
    for marker in values(markers)
        union!(all_timepoints, Set(marker.timepoint_days))
    end

    return sort(collect(all_timepoints))
end

"""Get standard differentiation panel for MSC."""
function get_differentiation_panel(lineage::Symbol)
    if lineage == :osteogenic
        return (
            early = ["RUNX2", "ALP", "COL1A1"],
            mid = ["IBSP", "SPP1"],
            late = ["BGLAP", "DMP1"],
            functional = [:alizarin_red, :alp_activity]
        )
    elseif lineage == :chondrogenic
        return (
            early = ["SOX9", "SOX5", "SOX6"],
            mid = ["COL2A1", "ACAN", "COMP"],
            negative = ["COL10A1", "MMP13", "RUNX2"],
            functional = [:alcian_blue, :dmmb_gag]
        )
    elseif lineage == :adipogenic
        return (
            early = ["PPARG", "CEBPA"],
            mid = ["FABP4", "LPL"],
            late = ["ADIPOQ", "LEP"],
            functional = [:oil_red_o]
        )
    else
        error("Unknown lineage: $lineage")
    end
end

end # module
