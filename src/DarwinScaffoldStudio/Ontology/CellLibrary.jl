"""
    CellLibrary

Comprehensive library of 100+ cell types for tissue engineering.
Organized by lineage with differentiation relationships.

# Author: Dr. Demetrios Agourakis
"""
module CellLibrary

using ..OBOFoundry: OBOTerm

export CELLS, CELLS_BY_LINEAGE, CELL_LINES
export get_cell, list_cells, get_derived_cells
export TISSUE_CELL_MAP, get_cells_for_tissue

# Helper
C(id, name; def="", syn=String[], par=String[]) = OBOTerm(id, name; definition=def, synonyms=syn, parents=par)

#=============================================================================
  STEM CELLS (15 terms)
=============================================================================#
const STEM_CELLS = Dict{String,OBOTerm}(
    "CL:0000034" => C("CL:0000034", "stem cell"; def="Self-renewing undifferentiated cell", par=["CL:0000000"]),
    "CL:0002248" => C("CL:0002248", "pluripotent stem cell"; def="Can form all cell types", syn=["PSC"], par=["CL:0000034"]),
    "CL:0002322" => C("CL:0002322", "embryonic stem cell"; def="From inner cell mass", syn=["ESC", "ES cell"], par=["CL:0002248"]),
    "CL:0000222" => C("CL:0000222", "induced pluripotent stem cell"; def="Reprogrammed somatic cell", syn=["iPSC", "iPS cell"], par=["CL:0002248"]),
    "CL:0000134" => C("CL:0000134", "mesenchymal stem cell"; def="Multipotent stromal cell", syn=["MSC", "mesenchymal stromal cell"], par=["CL:0000034"]),
    "CL:0002371" => C("CL:0002371", "hematopoietic stem cell"; def="Gives rise to blood cells", syn=["HSC"], par=["CL:0000034"]),
    "CL:0000055" => C("CL:0000055", "non-terminally differentiated cell"; def="Can still divide and differentiate", par=["CL:0000034"]),
    "CL:0002618" => C("CL:0002618", "adipose-derived stem cell"; def="MSC from adipose tissue", syn=["ADSC", "ASC"], par=["CL:0000134"]),
    "CL:0002328" => C("CL:0002328", "neural stem cell"; def="Gives rise to neural lineage", syn=["NSC"], par=["CL:0000034"]),
    "CL:0002338" => C("CL:0002338", "epithelial stem cell"; def="Gives rise to epithelia", par=["CL:0000034"]),
    "CL:0008001" => C("CL:0008001", "muscle stem cell"; def="Gives rise to muscle", syn=["satellite cell"], par=["CL:0000034"]),
    "CL:0000048" => C("CL:0000048", "multi fate stem cell"; def="Limited differentiation potential", par=["CL:0000034"]),
    "CL:0000037" => C("CL:0000037", "hematopoietic progenitor"; def="Committed blood progenitor", par=["CL:0002371"]),
    "CL:0000557" => C("CL:0000557", "granulocyte monocyte progenitor"; def="GMP progenitor", syn=["GMP"], par=["CL:0000037"]),
    "CL:0000049" => C("CL:0000049", "common myeloid progenitor"; def="CMP cell", syn=["CMP"], par=["CL:0000037"]),
)

#=============================================================================
  BONE CELLS (8 terms)
=============================================================================#
const BONE_CELLS = Dict{String,OBOTerm}(
    "CL:0000062" => C("CL:0000062", "osteoblast"; def="Bone-forming cell, produces osteoid", syn=["bone forming cell"], par=["CL:0000055"]),
    "CL:0000137" => C("CL:0000137", "osteocyte"; def="Mature bone cell in lacunae", syn=["bone cell"], par=["CL:0000062"]),
    "CL:0000092" => C("CL:0000092", "osteoclast"; def="Multinucleated bone-resorbing cell", syn=["bone resorbing cell"], par=["CL:0000094"]),
    "CL:0001035" => C("CL:0001035", "bone cell"; def="Any cell of bone tissue", par=["CL:0000000"]),
    "CL:0000093" => C("CL:0000093", "osteoprogenitor cell"; def="Bone progenitor cell", syn=["preosteoblast"], par=["CL:0000134"]),
    "CL:0000778" => C("CL:0000778", "mononuclear osteoclast"; def="Single-nucleated osteoclast", par=["CL:0000092"]),
    "CL:0000779" => C("CL:0000779", "multinuclear osteoclast"; def="Mature multinucleated osteoclast", par=["CL:0000092"]),
    "CL:0000058" => C("CL:0000058", "chondroblast"; def="Immature cartilage cell", syn=["cartilage forming cell"], par=["CL:0000055"]),
)

#=============================================================================
  CARTILAGE CELLS (6 terms)
=============================================================================#
const CARTILAGE_CELLS = Dict{String,OBOTerm}(
    "CL:0000138" => C("CL:0000138", "chondrocyte"; def="Mature cartilage cell producing ECM", syn=["cartilage cell"], par=["CL:0000058"]),
    "CL:0000743" => C("CL:0000743", "hypertrophic chondrocyte"; def="Enlarged chondrocyte undergoing calcification", par=["CL:0000138"]),
    "CL:0000742" => C("CL:0000742", "proliferating chondrocyte"; def="Actively dividing chondrocyte", par=["CL:0000138"]),
    "CL:0000744" => C("CL:0000744", "articular chondrocyte"; def="Chondrocyte of articular cartilage", par=["CL:0000138"]),
    "CL:0000745" => C("CL:0000745", "growth plate chondrocyte"; def="Chondrocyte in epiphyseal plate", par=["CL:0000138"]),
    "CL:0002553" => C("CL:0002553", "fibrocartilage chondrocyte"; def="Chondrocyte in fibrocartilage", par=["CL:0000138"]),
)

#=============================================================================
  CONNECTIVE TISSUE CELLS (10 terms)
=============================================================================#
const CONNECTIVE_CELLS = Dict{String,OBOTerm}(
    "CL:0000057" => C("CL:0000057", "fibroblast"; def="ECM-producing connective tissue cell", syn=["fibrocyte"], par=["CL:0000055"]),
    "CL:0002620" => C("CL:0002620", "skin fibroblast"; def="Dermal fibroblast", syn=["dermal fibroblast"], par=["CL:0000057"]),
    "CL:0002555" => C("CL:0002555", "cardiac fibroblast"; def="Heart fibroblast", par=["CL:0000057"]),
    "CL:0002556" => C("CL:0002556", "lung fibroblast"; def="Pulmonary fibroblast", par=["CL:0000057"]),
    "CL:0002557" => C("CL:0002557", "synovial fibroblast"; def="Joint fibroblast", par=["CL:0000057"]),
    "CL:0000136" => C("CL:0000136", "adipocyte"; def="Fat cell storing lipids", syn=["fat cell", "lipocyte"], par=["CL:0000055"]),
    "CL:0000448" => C("CL:0000448", "white adipocyte"; def="White fat cell", par=["CL:0000136"]),
    "CL:0000449" => C("CL:0000449", "brown adipocyte"; def="Brown fat cell, thermogenic", par=["CL:0000136"]),
    "CL:0000669" => C("CL:0000669", "pericyte"; def="Perivascular cell supporting capillaries", syn=["mural cell"], par=["CL:0000055"]),
    "CL:0000145" => C("CL:0000145", "tenocyte"; def="Tendon cell", syn=["tendon cell"], par=["CL:0000057"]),
)

#=============================================================================
  EPITHELIAL CELLS (15 terms)
=============================================================================#
const EPITHELIAL_CELLS = Dict{String,OBOTerm}(
    "CL:0000066" => C("CL:0000066", "epithelial cell"; def="Cell of epithelial tissue", syn=["epitheliocyte"], par=["CL:0000000"]),
    "CL:0000312" => C("CL:0000312", "keratinocyte"; def="Keratin-producing epidermal cell", syn=["keratocyte"], par=["CL:0000066"]),
    "CL:0002187" => C("CL:0002187", "basal keratinocyte"; def="Keratinocyte in stratum basale", syn=["basal cell"], par=["CL:0000312"]),
    "CL:0002188" => C("CL:0002188", "spinous keratinocyte"; def="Keratinocyte in stratum spinosum", par=["CL:0000312"]),
    "CL:0000361" => C("CL:0000361", "melanocyte"; def="Melanin-producing cell", syn=["pigment cell"], par=["CL:0000066"]),
    "CL:0000453" => C("CL:0000453", "Langerhans cell"; def="Dendritic cell of epidermis", par=["CL:0000066"]),
    "CL:0000362" => C("CL:0000362", "Merkel cell"; def="Mechanoreceptor in skin", par=["CL:0000066"]),
    "CL:0000082" => C("CL:0000082", "hepatocyte"; def="Liver parenchymal cell", syn=["liver cell"], par=["CL:0000066"]),
    "CL:0000584" => C("CL:0000584", "enterocyte"; def="Intestinal absorptive cell", syn=["intestinal epithelial cell"], par=["CL:0000066"]),
    "CL:0000158" => C("CL:0000158", "Clara cell"; def="Bronchiolar secretory cell", syn=["club cell"], par=["CL:0000066"]),
    "CL:0000065" => C("CL:0000065", "ependymal cell"; def="Lines brain ventricles", syn=["ependymocyte"], par=["CL:0000066"]),
    "CL:0000239" => C("CL:0000239", "goblet cell"; def="Mucus-secreting cell", par=["CL:0000066"]),
    "CL:0000322" => C("CL:0000322", "pneumocyte"; def="Lung alveolar epithelial cell", syn=["alveolar cell"], par=["CL:0000066"]),
    "CL:0002062" => C("CL:0002062", "type I pneumocyte"; def="Gas exchange cell", syn=["AT1 cell"], par=["CL:0000322"]),
    "CL:0002063" => C("CL:0002063", "type II pneumocyte"; def="Surfactant-producing cell", syn=["AT2 cell"], par=["CL:0000322"]),
)

#=============================================================================
  VASCULAR CELLS (10 terms)
=============================================================================#
const VASCULAR_CELLS = Dict{String,OBOTerm}(
    "CL:0000115" => C("CL:0000115", "endothelial cell"; def="Lines blood vessels", syn=["EC"], par=["CL:0000066"]),
    "CL:0002139" => C("CL:0002139", "arterial endothelial cell"; def="Endothelium of arteries", par=["CL:0000115"]),
    "CL:0002543" => C("CL:0002543", "venous endothelial cell"; def="Endothelium of veins", par=["CL:0000115"]),
    "CL:0002144" => C("CL:0002144", "capillary endothelial cell"; def="Endothelium of capillaries", par=["CL:0000115"]),
    "CL:0002138" => C("CL:0002138", "lymphatic endothelial cell"; def="Endothelium of lymphatics", syn=["LEC"], par=["CL:0000115"]),
    "CL:0000192" => C("CL:0000192", "smooth muscle cell"; def="Involuntary muscle cell", syn=["SMC"], par=["CL:0000187"]),
    "CL:0002539" => C("CL:0002539", "vascular smooth muscle cell"; def="Smooth muscle of vessel wall", syn=["VSMC"], par=["CL:0000192"]),
    "CL:0000359" => C("CL:0000359", "vascular associated smooth muscle cell"; def="SMC associated with vessels", par=["CL:0002539"]),
    "CL:0002548" => C("CL:0002548", "cardiac endothelial cell"; def="Endothelium of heart", par=["CL:0000115"]),
    "CL:0002546" => C("CL:0002546", "corneal endothelial cell"; def="Endothelium of cornea", par=["CL:0000115"]),
)

#=============================================================================
  CARDIAC CELLS (8 terms)
=============================================================================#
const CARDIAC_CELLS = Dict{String,OBOTerm}(
    "CL:0000746" => C("CL:0000746", "cardiomyocyte"; def="Contractile heart muscle cell", syn=["cardiac myocyte", "cardiac muscle cell"], par=["CL:0000187"]),
    "CL:0002098" => C("CL:0002098", "atrial cardiomyocyte"; def="Cardiomyocyte of atria", syn=["atrial myocyte"], par=["CL:0000746"]),
    "CL:0002131" => C("CL:0002131", "ventricular cardiomyocyte"; def="Cardiomyocyte of ventricles", syn=["ventricular myocyte"], par=["CL:0000746"]),
    "CL:0002072" => C("CL:0002072", "pacemaker cell"; def="Generates electrical impulses", syn=["nodal cell"], par=["CL:0000746"]),
    "CL:0002132" => C("CL:0002132", "Purkinje fiber cell"; def="Conducts impulses in ventricles", syn=["Purkinje cell"], par=["CL:0000746"]),
    "CL:0010020" => C("CL:0010020", "cardiac fibroblast"; def="Fibroblast of heart", par=["CL:0000057"]),
    "CL:0000333" => C("CL:0000333", "cardiac progenitor"; def="Heart progenitor cell", syn=["cardiomyocyte progenitor"], par=["CL:0000055"]),
    "CL:0000747" => C("CL:0000747", "cardiac muscle myoblast"; def="Immature cardiomyocyte", par=["CL:0000333"]),
)

#=============================================================================
  NEURAL CELLS (15 terms)
=============================================================================#
const NEURAL_CELLS = Dict{String,OBOTerm}(
    "CL:0000540" => C("CL:0000540", "neuron"; def="Electrically excitable cell", syn=["nerve cell"], par=["CL:0002319"]),
    "CL:0000104" => C("CL:0000104", "multipolar neuron"; def="Neuron with multiple dendrites", par=["CL:0000540"]),
    "CL:0000106" => C("CL:0000106", "unipolar neuron"; def="Neuron with single process", par=["CL:0000540"]),
    "CL:0000105" => C("CL:0000105", "bipolar neuron"; def="Neuron with two processes", par=["CL:0000540"]),
    "CL:0000679" => C("CL:0000679", "glutamatergic neuron"; def="Excitatory neuron using glutamate", par=["CL:0000540"]),
    "CL:0000617" => C("CL:0000617", "GABAergic neuron"; def="Inhibitory neuron using GABA", par=["CL:0000540"]),
    "CL:0000700" => C("CL:0000700", "dopaminergic neuron"; def="Neuron producing dopamine", par=["CL:0000540"]),
    "CL:0000125" => C("CL:0000125", "glial cell"; def="Support cell of nervous system", syn=["neuroglia", "glia"], par=["CL:0002319"]),
    "CL:0000127" => C("CL:0000127", "astrocyte"; def="Star-shaped CNS glia", syn=["astroglia"], par=["CL:0000125"]),
    "CL:0000128" => C("CL:0000128", "oligodendrocyte"; def="CNS myelinating cell", syn=["oligodendroglia"], par=["CL:0000125"]),
    "CL:0000129" => C("CL:0000129", "microglial cell"; def="CNS immune cell", syn=["microglia"], par=["CL:0000125"]),
    "CL:0002573" => C("CL:0002573", "Schwann cell"; def="PNS myelinating cell", syn=["neurilemma cell"], par=["CL:0000125"]),
    "CL:0000111" => C("CL:0000111", "peripheral sensory neuron"; def="Sensory neuron of PNS", syn=["sensory neuron"], par=["CL:0000540"]),
    "CL:0000100" => C("CL:0000100", "motor neuron"; def="Controls muscle contraction", syn=["motoneuron"], par=["CL:0000540"]),
    "CL:0000099" => C("CL:0000099", "interneuron"; def="Connects neurons", par=["CL:0000540"]),
)

#=============================================================================
  MUSCLE CELLS (10 terms)
=============================================================================#
const MUSCLE_CELLS = Dict{String,OBOTerm}(
    "CL:0000187" => C("CL:0000187", "muscle cell"; def="Contractile cell", syn=["myocyte"], par=["CL:0000211"]),
    "CL:0000188" => C("CL:0000188", "skeletal muscle cell"; def="Multinucleated striated muscle", syn=["myofiber", "skeletal myocyte"], par=["CL:0000187"]),
    "CL:0000189" => C("CL:0000189", "slow muscle cell"; def="Type I muscle fiber", syn=["slow twitch fiber", "type I fiber"], par=["CL:0000188"]),
    "CL:0000190" => C("CL:0000190", "fast muscle cell"; def="Type II muscle fiber", syn=["fast twitch fiber", "type II fiber"], par=["CL:0000188"]),
    "CL:0000515" => C("CL:0000515", "skeletal muscle myoblast"; def="Muscle precursor cell", syn=["myoblast"], par=["CL:0000055"]),
    "CL:0000680" => C("CL:0000680", "muscle satellite cell"; def="Muscle stem cell under basal lamina", syn=["satellite cell"], par=["CL:0008001"]),
    "CL:0000355" => C("CL:0000355", "myotube"; def="Fused myoblasts, immature fiber", par=["CL:0000188"]),
    "CL:0002372" => C("CL:0002372", "myofibroblast"; def="Contractile fibroblast", par=["CL:0000057"]),
    "CL:0000193" => C("CL:0000193", "cardiac muscle cell"; def="Heart muscle cell", syn=["cardiomyocyte"], par=["CL:0000746"]),
    "CL:0008000" => C("CL:0008000", "tendon sheath cell"; def="Cell surrounding tendon", par=["CL:0000057"]),
)

#=============================================================================
  IMMUNE CELLS (15 terms)
=============================================================================#
const IMMUNE_CELLS = Dict{String,OBOTerm}(
    "CL:0000235" => C("CL:0000235", "macrophage"; def="Phagocytic immune cell", syn=["histiocyte"], par=["CL:0000576"]),
    "CL:0000860" => C("CL:0000860", "M1 macrophage"; def="Pro-inflammatory macrophage", syn=["classically activated macrophage"], par=["CL:0000235"]),
    "CL:0000863" => C("CL:0000863", "M2 macrophage"; def="Anti-inflammatory macrophage", syn=["alternatively activated macrophage"], par=["CL:0000235"]),
    "CL:0000576" => C("CL:0000576", "monocyte"; def="Blood precursor to macrophage", par=["CL:0000766"]),
    "CL:0000084" => C("CL:0000084", "T cell"; def="Adaptive immune lymphocyte", syn=["T lymphocyte"], par=["CL:0000945"]),
    "CL:0000624" => C("CL:0000624", "CD4 T cell"; def="Helper T cell", syn=["helper T cell", "Th cell"], par=["CL:0000084"]),
    "CL:0000625" => C("CL:0000625", "CD8 T cell"; def="Cytotoxic T cell", syn=["cytotoxic T cell", "CTL"], par=["CL:0000084"]),
    "CL:0000815" => C("CL:0000815", "regulatory T cell"; def="Suppressive T cell", syn=["Treg", "T regulatory cell"], par=["CL:0000084"]),
    "CL:0000236" => C("CL:0000236", "B cell"; def="Antibody-producing lymphocyte", syn=["B lymphocyte"], par=["CL:0000945"]),
    "CL:0000786" => C("CL:0000786", "plasma cell"; def="Antibody-secreting B cell", syn=["plasmocyte"], par=["CL:0000236"]),
    "CL:0000623" => C("CL:0000623", "natural killer cell"; def="Innate immune killer", syn=["NK cell"], par=["CL:0000945"]),
    "CL:0000451" => C("CL:0000451", "dendritic cell"; def="Antigen-presenting cell", syn=["DC"], par=["CL:0000766"]),
    "CL:0000094" => C("CL:0000094", "granulocyte"; def="Granulated white blood cell", par=["CL:0000766"]),
    "CL:0000775" => C("CL:0000775", "neutrophil"; def="Most common granulocyte", syn=["PMN"], par=["CL:0000094"]),
    "CL:0000097" => C("CL:0000097", "mast cell"; def="Histamine-releasing immune cell", syn=["mastocyte"], par=["CL:0000094"]),
)

#=============================================================================
  SPECIALIZED CELLS (10 terms)
=============================================================================#
const SPECIALIZED_CELLS = Dict{String,OBOTerm}(
    "CL:0000083" => C("CL:0000083", "epithelial cell of pancreas"; def="Pancreatic epithelium", par=["CL:0000066"]),
    "CL:0000169" => C("CL:0000169", "insulin-secreting cell"; def="Beta cell of islets", syn=["beta cell"], par=["CL:0000083"]),
    "CL:0000171" => C("CL:0000171", "glucagon-secreting cell"; def="Alpha cell of islets", syn=["alpha cell"], par=["CL:0000083"]),
    "CL:0000651" => C("CL:0000651", "kidney collecting duct cell"; def="Renal tubule cell", par=["CL:0000066"]),
    "CL:0000653" => C("CL:0000653", "podocyte"; def="Glomerular filtration cell", syn=["glomerular epithelial cell"], par=["CL:0000066"]),
    "CL:0000598" => C("CL:0000598", "renal proximal tubule cell"; def="Proximal tubule epithelium", par=["CL:0000066"]),
    "CL:0002306" => C("CL:0002306", "hair follicle cell"; def="Cell of hair follicle", par=["CL:0000312"]),
    "CL:0000696" => C("CL:0000696", "photoreceptor cell"; def="Light-sensing retinal cell", par=["CL:0000540"]),
    "CL:0000573" => C("CL:0000573", "rod photoreceptor"; def="Low-light vision cell", syn=["rod cell"], par=["CL:0000696"]),
    "CL:0000574" => C("CL:0000574", "cone photoreceptor"; def="Color vision cell", syn=["cone cell"], par=["CL:0000696"]),
)

#=============================================================================
  CELL LINES (15 terms)
=============================================================================#
const CELL_LINES = Dict{String,OBOTerm}(
    "BTO:0000968" => C("BTO:0000968", "MC3T3-E1"; def="Murine osteoblast cell line", syn=["MC3T3"], par=["CL:0000062"]),
    "BTO:0001279" => C("BTO:0001279", "Saos-2"; def="Human osteosarcoma line", par=["CL:0000062"]),
    "BTO:0001426" => C("BTO:0001426", "MG-63"; def="Human osteosarcoma line", par=["CL:0000062"]),
    "BTO:0002922" => C("BTO:0002922", "hFOB 1.19"; def="Human fetal osteoblast line", par=["CL:0000062"]),
    "BTO:0001906" => C("BTO:0001906", "NIH-3T3"; def="Murine fibroblast line", syn=["3T3"], par=["CL:0000057"]),
    "BTO:0002335" => C("BTO:0002335", "L929"; def="Murine fibroblast line", par=["CL:0000057"]),
    "BTO:0002354" => C("BTO:0002354", "hMSC"; def="Human mesenchymal stem cells", syn=["human MSC"], par=["CL:0000134"]),
    "BTO:0001529" => C("BTO:0001529", "HUVEC"; def="Human umbilical vein endothelial cells", par=["CL:0000115"]),
    "BTO:0003566" => C("BTO:0003566", "EA.hy926"; def="Immortalized human endothelial line", par=["CL:0000115"]),
    "BTO:0000178" => C("BTO:0000178", "ATDC5"; def="Murine chondrogenic line", par=["CL:0000138"]),
    "BTO:0000921" => C("BTO:0000921", "PC-12"; def="Rat neuronal line", syn=["PC12"], par=["CL:0000540"]),
    "BTO:0000793" => C("BTO:0000793", "SH-SY5Y"; def="Human neuroblastoma line", par=["CL:0000540"]),
    "BTO:0000520" => C("BTO:0000520", "HeLa"; def="Human cervical cancer line", par=["CL:0000066"]),
    "BTO:0000567" => C("BTO:0000567", "HEK-293"; def="Human embryonic kidney line", syn=["HEK293", "293"], par=["CL:0000066"]),
    "BTO:0000018" => C("BTO:0000018", "A549"; def="Human lung carcinoma line", par=["CL:0000066"]),
)

#=============================================================================
  COMBINED DATABASE
=============================================================================#

"""All 100+ cell types combined."""
const CELLS = merge(
    STEM_CELLS, BONE_CELLS, CARTILAGE_CELLS, CONNECTIVE_CELLS,
    EPITHELIAL_CELLS, VASCULAR_CELLS, CARDIAC_CELLS, NEURAL_CELLS,
    MUSCLE_CELLS, IMMUNE_CELLS, SPECIALIZED_CELLS, CELL_LINES
)

"""Cells organized by lineage."""
const CELLS_BY_LINEAGE = Dict{Symbol,Dict{String,OBOTerm}}(
    :stem => STEM_CELLS,
    :bone => BONE_CELLS,
    :cartilage => CARTILAGE_CELLS,
    :connective => CONNECTIVE_CELLS,
    :epithelial => EPITHELIAL_CELLS,
    :vascular => VASCULAR_CELLS,
    :cardiac => CARDIAC_CELLS,
    :neural => NEURAL_CELLS,
    :muscle => MUSCLE_CELLS,
    :immune => IMMUNE_CELLS,
    :specialized => SPECIALIZED_CELLS,
    :cell_lines => CELL_LINES,
)

#=============================================================================
  TISSUE-CELL MAPPING
=============================================================================#

"""Mapping of tissues to relevant cell types."""
const TISSUE_CELL_MAP = Dict{String,Vector{String}}(
    # Bone
    "UBERON:0002481" => ["CL:0000062", "CL:0000137", "CL:0000092", "CL:0000134", "CL:0000093"],
    "UBERON:0001474" => ["CL:0000062", "CL:0000137", "CL:0000092"],
    "UBERON:0001475" => ["CL:0000062", "CL:0000137"],
    # Cartilage
    "UBERON:0002418" => ["CL:0000138", "CL:0000134", "CL:0000058"],
    "UBERON:0001085" => ["CL:0000744", "CL:0000138"],
    "UBERON:0001563" => ["CL:0002553", "CL:0000138"],
    # Skin
    "UBERON:0002097" => ["CL:0000312", "CL:0000057", "CL:0000361", "CL:0000453"],
    "UBERON:0002067" => ["CL:0000057", "CL:0002620"],
    "UBERON:0001003" => ["CL:0000312", "CL:0002187", "CL:0000361"],
    # Vascular
    "UBERON:0001981" => ["CL:0000115", "CL:0000192", "CL:0000669"],
    "UBERON:0001637" => ["CL:0002139", "CL:0002539"],
    "UBERON:0001638" => ["CL:0002543", "CL:0000192"],
    # Cardiac
    "UBERON:0000948" => ["CL:0000746", "CL:0010020", "CL:0002548"],
    "UBERON:0002349" => ["CL:0000746", "CL:0002131", "CL:0002098"],
    # Neural
    "UBERON:0001017" => ["CL:0000540", "CL:0000127", "CL:0000128", "CL:0000129"],
    "UBERON:0000955" => ["CL:0000540", "CL:0000127", "CL:0000128"],
    "UBERON:0002240" => ["CL:0000540", "CL:0000127", "CL:0000100"],
    "UBERON:0001021" => ["CL:0000540", "CL:0002573", "CL:0000111"],
    # Muscle
    "UBERON:0002385" => ["CL:0000187", "CL:0000680", "CL:0000515"],
    "UBERON:0001134" => ["CL:0000188", "CL:0000680", "CL:0000515"],
    "UBERON:0001133" => ["CL:0000746", "CL:0002072"],
    # Liver
    "UBERON:0002107" => ["CL:0000082", "CL:0000115", "CL:0000235"],
    # Kidney
    "UBERON:0002113" => ["CL:0000653", "CL:0000598", "CL:0000651"],
    # Lung
    "UBERON:0002048" => ["CL:0002062", "CL:0002063", "CL:0000158", "CL:0000115"],
    # Pancreas
    "UBERON:0001264" => ["CL:0000169", "CL:0000171", "CL:0000083"],
    # Eye
    "UBERON:0000966" => ["CL:0000573", "CL:0000574", "CL:0000696"],
    "UBERON:0000964" => ["CL:0002546", "CL:0000066"],
)

#=============================================================================
  LOOKUP FUNCTIONS
=============================================================================#

"""Get cell by CL/BTO ID."""
get_cell(id::String) = get(CELLS, id, nothing)

"""List cells by lineage."""
function list_cells(lineage::Symbol=:all)
    lineage == :all ? collect(values(CELLS)) :
    haskey(CELLS_BY_LINEAGE, lineage) ? collect(values(CELLS_BY_LINEAGE[lineage])) : OBOTerm[]
end

"""Get derived/child cells."""
function get_derived_cells(parent_id::String)
    [c for c in values(CELLS) if parent_id in c.parents]
end

"""Get cells for a tissue."""
function get_cells_for_tissue(tissue_id::String)
    cell_ids = get(TISSUE_CELL_MAP, tissue_id, String[])
    [CELLS[id] for id in cell_ids if haskey(CELLS, id)]
end

end # module
