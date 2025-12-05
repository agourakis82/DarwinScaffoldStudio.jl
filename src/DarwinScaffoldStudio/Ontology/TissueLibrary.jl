"""
    TissueLibrary

Comprehensive library of 150+ tissues for tissue engineering.
Organized by anatomical system with hierarchical relationships.

# Author: Dr. Demetrios Agourakis
"""
module TissueLibrary

using ..OBOFoundry: OBOTerm

export TISSUES, TISSUE_BY_SYSTEM, get_tissue, list_tissues, get_subtissues

# Helper to create terms quickly
T(id, name; def="", syn=String[], par=String[]) = OBOTerm(id, name; definition=def, synonyms=syn, parents=par)

#=============================================================================
  SKELETAL SYSTEM - BONE (25 terms)
=============================================================================#
const BONE = Dict{String,OBOTerm}(
    "UBERON:0002481" => T("UBERON:0002481", "bone tissue"; def="Mineralized connective tissue", syn=["bone", "osseous tissue"], par=["UBERON:0000479"]),
    "UBERON:0001474" => T("UBERON:0001474", "trabecular bone"; def="Spongy bone with trabeculae", syn=["cancellous bone", "spongy bone"], par=["UBERON:0002481"]),
    "UBERON:0001475" => T("UBERON:0001475", "cortical bone"; def="Dense compact outer bone", syn=["compact bone"], par=["UBERON:0002481"]),
    "UBERON:0002513" => T("UBERON:0002513", "endochondral bone"; def="Bone from cartilage template", par=["UBERON:0002481"]),
    "UBERON:0002514" => T("UBERON:0002514", "intramembranous bone"; def="Bone from mesenchyme directly", par=["UBERON:0002481"]),
    "UBERON:0007688" => T("UBERON:0007688", "woven bone"; def="Immature bone, random collagen", syn=["primary bone"], par=["UBERON:0002481"]),
    "UBERON:0001433" => T("UBERON:0001433", "lamellar bone"; def="Mature bone, parallel layers", syn=["secondary bone"], par=["UBERON:0002481"]),
    "UBERON:0001448" => T("UBERON:0001448", "femur"; def="Thigh bone, largest bone", syn=["thigh bone"], par=["UBERON:0002481"]),
    "UBERON:0001442" => T("UBERON:0001442", "tibia"; def="Shin bone", syn=["shinbone"], par=["UBERON:0002481"]),
    "UBERON:0001446" => T("UBERON:0001446", "fibula"; def="Lateral lower leg bone", syn=["calf bone"], par=["UBERON:0002481"]),
    "UBERON:0001450" => T("UBERON:0001450", "humerus"; def="Upper arm bone", par=["UBERON:0002481"]),
    "UBERON:0001423" => T("UBERON:0001423", "radius"; def="Lateral forearm bone", par=["UBERON:0002481"]),
    "UBERON:0001424" => T("UBERON:0001424", "ulna"; def="Medial forearm bone", par=["UBERON:0002481"]),
    "UBERON:0001684" => T("UBERON:0001684", "mandible"; def="Lower jaw bone", syn=["lower jaw"], par=["UBERON:0002481"]),
    "UBERON:0001680" => T("UBERON:0001680", "maxilla"; def="Upper jaw bone", syn=["upper jaw"], par=["UBERON:0002481"]),
    "UBERON:0001363" => T("UBERON:0001363", "cranium"; def="Skull bones", syn=["skull"], par=["UBERON:0002481"]),
    "UBERON:0001436" => T("UBERON:0001436", "vertebra"; def="Spinal bone", syn=["spinal vertebra"], par=["UBERON:0002481"]),
    "UBERON:0002228" => T("UBERON:0002228", "rib"; def="Curved thoracic bone", syn=["costa"], par=["UBERON:0002481"]),
    "UBERON:0000975" => T("UBERON:0000975", "sternum"; def="Breastbone", syn=["breastbone"], par=["UBERON:0002481"]),
    "UBERON:0001272" => T("UBERON:0001272", "pelvis"; def="Hip bone", syn=["hip bone", "innominate"], par=["UBERON:0002481"]),
    "UBERON:0001440" => T("UBERON:0001440", "patella"; def="Kneecap", syn=["kneecap"], par=["UBERON:0002481"]),
    "UBERON:0001292" => T("UBERON:0001292", "scapula"; def="Shoulder blade", syn=["shoulder blade"], par=["UBERON:0002481"]),
    "UBERON:0001105" => T("UBERON:0001105", "clavicle"; def="Collar bone", syn=["collar bone"], par=["UBERON:0002481"]),
    "UBERON:0002428" => T("UBERON:0002428", "calcaneus"; def="Heel bone", syn=["heel bone"], par=["UBERON:0002481"]),
    "UBERON:0002372" => T("UBERON:0002372", "bone marrow"; def="Tissue producing blood cells", syn=["medulla ossium"], par=["UBERON:0002481"]),
)

#=============================================================================
  SKELETAL SYSTEM - CARTILAGE (15 terms)
=============================================================================#
const CARTILAGE = Dict{String,OBOTerm}(
    "UBERON:0002418" => T("UBERON:0002418", "cartilage tissue"; def="Avascular connective tissue with chondrocytes", syn=["cartilage"], par=["UBERON:0000479"]),
    "UBERON:0001994" => T("UBERON:0001994", "hyaline cartilage"; def="Most common type, glassy appearance", par=["UBERON:0002418"]),
    "UBERON:0001995" => T("UBERON:0001995", "fibrocartilage"; def="Dense collagen bundles", syn=["fibrous cartilage"], par=["UBERON:0002418"]),
    "UBERON:0001996" => T("UBERON:0001996", "elastic cartilage"; def="Contains elastic fibers", syn=["yellow cartilage"], par=["UBERON:0002418"]),
    "UBERON:0001085" => T("UBERON:0001085", "articular cartilage"; def="Covers joint surfaces", syn=["joint cartilage"], par=["UBERON:0001994"]),
    "UBERON:0001563" => T("UBERON:0001563", "meniscus"; def="Knee fibrocartilage pad", syn=["knee meniscus"], par=["UBERON:0001995"]),
    "UBERON:0001066" => T("UBERON:0001066", "intervertebral disc"; def="Spine cushion", syn=["IVD", "spinal disc"], par=["UBERON:0001995"]),
    "UBERON:0003359" => T("UBERON:0003359", "costal cartilage"; def="Connects ribs to sternum", syn=["rib cartilage"], par=["UBERON:0001994"]),
    "UBERON:0001739" => T("UBERON:0001739", "nasal cartilage"; def="Nose structure", par=["UBERON:0001994"]),
    "UBERON:0001082" => T("UBERON:0001082", "growth plate"; def="Epiphyseal plate for bone growth", syn=["epiphyseal plate", "physis"], par=["UBERON:0001994"]),
    "UBERON:0001738" => T("UBERON:0001738", "thyroid cartilage"; def="Largest laryngeal cartilage", syn=["Adams apple"], par=["UBERON:0001994"]),
    "UBERON:0001570" => T("UBERON:0001570", "epiglottis"; def="Covers trachea during swallowing", par=["UBERON:0001996"]),
    "UBERON:0001757" => T("UBERON:0001757", "ear cartilage"; def="External ear structure", syn=["auricular cartilage"], par=["UBERON:0001996"]),
    "UBERON:0035943" => T("UBERON:0035943", "TMJ disc"; def="Temporomandibular joint disc", par=["UBERON:0001995"]),
    "UBERON:0001091" => T("UBERON:0001091", "tracheal cartilage"; def="Trachea rings", par=["UBERON:0001994"]),
)

#=============================================================================
  INTEGUMENTARY SYSTEM - SKIN (12 terms)
=============================================================================#
const SKIN = Dict{String,OBOTerm}(
    "UBERON:0002097" => T("UBERON:0002097", "skin"; def="External body covering, largest organ", syn=["cutis", "integument"], par=["UBERON:0000479"]),
    "UBERON:0002067" => T("UBERON:0002067", "dermis"; def="Layer beneath epidermis", syn=["derma", "corium"], par=["UBERON:0002097"]),
    "UBERON:0001003" => T("UBERON:0001003", "epidermis"; def="Outermost skin layer", syn=["cuticle"], par=["UBERON:0002097"]),
    "UBERON:0002072" => T("UBERON:0002072", "hypodermis"; def="Subcutaneous layer", syn=["subcutis"], par=["UBERON:0002097"]),
    "UBERON:0002069" => T("UBERON:0002069", "stratum corneum"; def="Outermost epidermis, dead cells", syn=["horny layer"], par=["UBERON:0001003"]),
    "UBERON:0002025" => T("UBERON:0002025", "stratum basale"; def="Deepest epidermis, stem cells", syn=["basal layer"], par=["UBERON:0001003"]),
    "UBERON:0002026" => T("UBERON:0002026", "stratum spinosum"; def="Spiny keratinocyte layer", syn=["prickle cell layer"], par=["UBERON:0001003"]),
    "UBERON:0002068" => T("UBERON:0002068", "papillary dermis"; def="Upper dermis", syn=["papillary layer"], par=["UBERON:0002067"]),
    "UBERON:0002084" => T("UBERON:0002084", "reticular dermis"; def="Deep dermis, dense collagen", syn=["reticular layer"], par=["UBERON:0002067"]),
    "UBERON:0002371" => T("UBERON:0002371", "basement membrane"; def="Dermal-epidermal junction", syn=["DEJ"], par=["UBERON:0002097"]),
    "UBERON:0001820" => T("UBERON:0001820", "sweat gland"; def="Eccrine gland for thermoregulation", par=["UBERON:0002097"]),
    "UBERON:0001821" => T("UBERON:0001821", "sebaceous gland"; def="Oil-producing gland", par=["UBERON:0002097"]),
)

#=============================================================================
  CARDIOVASCULAR SYSTEM (25 terms)
=============================================================================#
const CARDIOVASCULAR = Dict{String,OBOTerm}(
    "UBERON:0001981" => T("UBERON:0001981", "blood vessel"; def="Tubular structure carrying blood", syn=["vessel"], par=["UBERON:0000479"]),
    "UBERON:0001637" => T("UBERON:0001637", "artery"; def="Carries blood from heart", syn=["arterial vessel"], par=["UBERON:0001981"]),
    "UBERON:0001638" => T("UBERON:0001638", "vein"; def="Carries blood to heart", syn=["venous vessel"], par=["UBERON:0001981"]),
    "UBERON:0001982" => T("UBERON:0001982", "capillary"; def="Smallest vessel for exchange", par=["UBERON:0001981"]),
    "UBERON:0002012" => T("UBERON:0002012", "aorta"; def="Main artery from heart", par=["UBERON:0001637"]),
    "UBERON:0001033" => T("UBERON:0001033", "coronary artery"; def="Supplies heart muscle", par=["UBERON:0001637"]),
    "UBERON:0001515" => T("UBERON:0001515", "carotid artery"; def="Supplies head and neck", par=["UBERON:0001637"]),
    "UBERON:0001395" => T("UBERON:0001395", "femoral artery"; def="Main thigh artery", par=["UBERON:0001637"]),
    "UBERON:0001585" => T("UBERON:0001585", "vena cava"; def="Large vein to heart", par=["UBERON:0001638"]),
    "UBERON:0001546" => T("UBERON:0001546", "saphenous vein"; def="Leg vein, graft source", par=["UBERON:0001638"]),
    "UBERON:0002522" => T("UBERON:0002522", "tunica intima"; def="Inner vessel layer", syn=["intima"], par=["UBERON:0001981"]),
    "UBERON:0002523" => T("UBERON:0002523", "tunica media"; def="Middle vessel layer", syn=["media"], par=["UBERON:0001981"]),
    "UBERON:0002524" => T("UBERON:0002524", "tunica adventitia"; def="Outer vessel layer", syn=["adventitia"], par=["UBERON:0001981"]),
    "UBERON:0000948" => T("UBERON:0000948", "heart"; def="Muscular pump organ", syn=["cardiac organ"], par=["UBERON:0000062"]),
    "UBERON:0002349" => T("UBERON:0002349", "myocardium"; def="Heart muscle wall", syn=["cardiac muscle"], par=["UBERON:0000948"]),
    "UBERON:0002165" => T("UBERON:0002165", "endocardium"; def="Inner heart lining", par=["UBERON:0000948"]),
    "UBERON:0002407" => T("UBERON:0002407", "pericardium"; def="Heart sac", par=["UBERON:0000948"]),
    "UBERON:0002079" => T("UBERON:0002079", "left ventricle"; def="Pumps to systemic circulation", syn=["LV"], par=["UBERON:0000948"]),
    "UBERON:0002080" => T("UBERON:0002080", "right ventricle"; def="Pumps to lungs", syn=["RV"], par=["UBERON:0000948"]),
    "UBERON:0002081" => T("UBERON:0002081", "left atrium"; def="Receives from pulmonary veins", syn=["LA"], par=["UBERON:0000948"]),
    "UBERON:0002082" => T("UBERON:0002082", "right atrium"; def="Receives from venae cavae", syn=["RA"], par=["UBERON:0000948"]),
    "UBERON:0002134" => T("UBERON:0002134", "heart valve"; def="Controls blood flow", par=["UBERON:0000948"]),
    "UBERON:0002135" => T("UBERON:0002135", "mitral valve"; def="Left AV valve", syn=["bicuspid valve"], par=["UBERON:0002134"]),
    "UBERON:0002137" => T("UBERON:0002137", "aortic valve"; def="LV to aorta valve", par=["UBERON:0002134"]),
    "UBERON:0003920" => T("UBERON:0003920", "ventricular septum"; def="Wall between ventricles", syn=["IVS"], par=["UBERON:0000948"]),
)

#=============================================================================
  NERVOUS SYSTEM (25 terms)
=============================================================================#
const NEURAL = Dict{String,OBOTerm}(
    "UBERON:0001017" => T("UBERON:0001017", "central nervous system"; def="Brain and spinal cord", syn=["CNS"], par=["UBERON:0000479"]),
    "UBERON:0000955" => T("UBERON:0000955", "brain"; def="Central organ in skull", syn=["encephalon"], par=["UBERON:0001017"]),
    "UBERON:0002240" => T("UBERON:0002240", "spinal cord"; def="CNS in vertebral canal", syn=["medulla spinalis"], par=["UBERON:0001017"]),
    "UBERON:0001016" => T("UBERON:0001016", "peripheral nervous system"; def="Nerves outside CNS", syn=["PNS"], par=["UBERON:0000479"]),
    "UBERON:0001021" => T("UBERON:0001021", "nerve"; def="Bundle of axons", syn=["peripheral nerve"], par=["UBERON:0001016"]),
    "UBERON:0001759" => T("UBERON:0001759", "sciatic nerve"; def="Largest nerve in body", par=["UBERON:0001021"]),
    "UBERON:0001492" => T("UBERON:0001492", "radial nerve"; def="Posterior arm nerve", par=["UBERON:0001021"]),
    "UBERON:0001493" => T("UBERON:0001493", "median nerve"; def="Anterior forearm nerve", par=["UBERON:0001021"]),
    "UBERON:0001494" => T("UBERON:0001494", "ulnar nerve"; def="Medial forearm nerve", par=["UBERON:0001021"]),
    "UBERON:0001647" => T("UBERON:0001647", "facial nerve"; def="Cranial nerve VII", par=["UBERON:0001021"]),
    "UBERON:0001758" => T("UBERON:0001758", "optic nerve"; def="Cranial nerve II", par=["UBERON:0001021"]),
    "UBERON:0000956" => T("UBERON:0000956", "cerebral cortex"; def="Outer brain layer", syn=["cortex", "neocortex"], par=["UBERON:0000955"]),
    "UBERON:0001898" => T("UBERON:0001898", "hypothalamus"; def="Autonomic control center", par=["UBERON:0000955"]),
    "UBERON:0001882" => T("UBERON:0001882", "hippocampus"; def="Memory structure", par=["UBERON:0000955"]),
    "UBERON:0001876" => T("UBERON:0001876", "amygdala"; def="Emotion processing", par=["UBERON:0000955"]),
    "UBERON:0002037" => T("UBERON:0002037", "cerebellum"; def="Motor coordination", syn=["little brain"], par=["UBERON:0000955"]),
    "UBERON:0002038" => T("UBERON:0002038", "substantia nigra"; def="Dopamine production", par=["UBERON:0000955"]),
    "UBERON:0002792" => T("UBERON:0002792", "cervical spinal cord"; def="Neck region", par=["UBERON:0002240"]),
    "UBERON:0002793" => T("UBERON:0002793", "thoracic spinal cord"; def="Chest region", par=["UBERON:0002240"]),
    "UBERON:0002795" => T("UBERON:0002795", "lumbar spinal cord"; def="Lower back region", par=["UBERON:0002240"]),
    "UBERON:0001870" => T("UBERON:0001870", "gray matter"; def="Neuronal cell bodies", syn=["grey matter"], par=["UBERON:0001017"]),
    "UBERON:0001871" => T("UBERON:0001871", "white matter"; def="Myelinated axons", par=["UBERON:0001017"]),
    "UBERON:0002435" => T("UBERON:0002435", "striatum"; def="Basal ganglia component", par=["UBERON:0000955"]),
    "UBERON:0001896" => T("UBERON:0001896", "thalamus"; def="Sensory relay center", par=["UBERON:0000955"]),
    "UBERON:0002298" => T("UBERON:0002298", "brain stem"; def="Connects brain to spinal cord", syn=["brainstem"], par=["UBERON:0000955"]),
)

#=============================================================================
  MUSCULAR SYSTEM (20 terms)
=============================================================================#
const MUSCLE = Dict{String,OBOTerm}(
    "UBERON:0002385" => T("UBERON:0002385", "muscle tissue"; def="Contractile tissue", syn=["muscle"], par=["UBERON:0000479"]),
    "UBERON:0001134" => T("UBERON:0001134", "skeletal muscle"; def="Voluntary striated muscle", syn=["striated muscle"], par=["UBERON:0002385"]),
    "UBERON:0001133" => T("UBERON:0001133", "cardiac muscle"; def="Heart muscle", syn=["myocardium"], par=["UBERON:0002385"]),
    "UBERON:0001135" => T("UBERON:0001135", "smooth muscle"; def="Involuntary muscle", syn=["visceral muscle"], par=["UBERON:0002385"]),
    "UBERON:0001388" => T("UBERON:0001388", "gastrocnemius"; def="Calf muscle", syn=["calf muscle"], par=["UBERON:0001134"]),
    "UBERON:0001381" => T("UBERON:0001381", "biceps brachii"; def="Upper arm flexor", syn=["biceps"], par=["UBERON:0001134"]),
    "UBERON:0001382" => T("UBERON:0001382", "triceps brachii"; def="Upper arm extensor", syn=["triceps"], par=["UBERON:0001134"]),
    "UBERON:0001377" => T("UBERON:0001377", "quadriceps"; def="Anterior thigh muscle", syn=["quads"], par=["UBERON:0001134"]),
    "UBERON:0011907" => T("UBERON:0011907", "diaphragm"; def="Primary breathing muscle", par=["UBERON:0001134"]),
    "UBERON:0001100" => T("UBERON:0001100", "pectoralis major"; def="Chest muscle", syn=["pec major"], par=["UBERON:0001134"]),
    "UBERON:0002000" => T("UBERON:0002000", "gluteus maximus"; def="Largest gluteal muscle", syn=["glute max"], par=["UBERON:0001134"]),
    "UBERON:0001589" => T("UBERON:0001589", "deltoid"; def="Shoulder muscle", par=["UBERON:0001134"]),
    "UBERON:0008779" => T("UBERON:0008779", "rotator cuff"; def="Shoulder stabilizers", par=["UBERON:0001134"]),
    "UBERON:0002384" => T("UBERON:0002384", "connective tissue"; def="Support tissue with ECM", par=["UBERON:0000479"]),
    "UBERON:0006590" => T("UBERON:0006590", "tendon"; def="Muscle to bone connection", syn=["sinew"], par=["UBERON:0002384"]),
    "UBERON:0000211" => T("UBERON:0000211", "ligament"; def="Bone to bone connection", par=["UBERON:0002384"]),
    "UBERON:0003705" => T("UBERON:0003705", "Achilles tendon"; def="Calf to heel tendon", syn=["calcaneal tendon"], par=["UBERON:0006590"]),
    "UBERON:0004709" => T("UBERON:0004709", "ACL"; def="Anterior cruciate ligament", syn=["anterior cruciate ligament"], par=["UBERON:0000211"]),
    "UBERON:0004710" => T("UBERON:0004710", "PCL"; def="Posterior cruciate ligament", syn=["posterior cruciate ligament"], par=["UBERON:0000211"]),
    "UBERON:0006588" => T("UBERON:0006588", "fascia"; def="Connective tissue sheet", par=["UBERON:0002384"]),
)

#=============================================================================
  VISCERAL ORGANS (25 terms)
=============================================================================#
const VISCERAL = Dict{String,OBOTerm}(
    "UBERON:0002107" => T("UBERON:0002107", "liver"; def="Largest internal organ, metabolism", syn=["hepar"], par=["UBERON:0000062"]),
    "UBERON:0001264" => T("UBERON:0001264", "pancreas"; def="Digestive enzymes and hormones", par=["UBERON:0000062"]),
    "UBERON:0000945" => T("UBERON:0000945", "stomach"; def="Food storage and digestion", syn=["gaster"], par=["UBERON:0000062"]),
    "UBERON:0002108" => T("UBERON:0002108", "small intestine"; def="Nutrient absorption", par=["UBERON:0000062"]),
    "UBERON:0002116" => T("UBERON:0002116", "large intestine"; def="Water absorption", syn=["colon"], par=["UBERON:0000062"]),
    "UBERON:0002113" => T("UBERON:0002113", "kidney"; def="Blood filtration organ", syn=["ren"], par=["UBERON:0000062"]),
    "UBERON:0001285" => T("UBERON:0001285", "nephron"; def="Kidney functional unit", par=["UBERON:0002113"]),
    "UBERON:0001286" => T("UBERON:0001286", "glomerulus"; def="Filtration capillary tuft", par=["UBERON:0001285"]),
    "UBERON:0001254" => T("UBERON:0001254", "urinary bladder"; def="Urine storage", syn=["bladder"], par=["UBERON:0000062"]),
    "UBERON:0002048" => T("UBERON:0002048", "lung"; def="Gas exchange organ", syn=["pulmo"], par=["UBERON:0000062"]),
    "UBERON:0002299" => T("UBERON:0002299", "alveolus"; def="Gas exchange air sac", par=["UBERON:0002048"]),
    "UBERON:0002185" => T("UBERON:0002185", "bronchus"; def="Airway to lungs", par=["UBERON:0002048"]),
    "UBERON:0003126" => T("UBERON:0003126", "trachea"; def="Windpipe", syn=["windpipe"], par=["UBERON:0000062"]),
    "UBERON:0002106" => T("UBERON:0002106", "spleen"; def="Blood filter, immunity", par=["UBERON:0000062"]),
    "UBERON:0002370" => T("UBERON:0002370", "thymus"; def="T cell maturation", par=["UBERON:0000062"]),
    "UBERON:0002046" => T("UBERON:0002046", "gallbladder"; def="Bile storage", par=["UBERON:0000062"]),
    "UBERON:0001153" => T("UBERON:0001153", "intestinal villi"; def="Absorption projections", syn=["villi"], par=["UBERON:0002108"]),
    "UBERON:0001013" => T("UBERON:0001013", "adipose tissue"; def="Fat storage tissue", syn=["fat"], par=["UBERON:0002384"]),
    "UBERON:0014454" => T("UBERON:0014454", "white adipose"; def="Energy storage fat", syn=["WAT"], par=["UBERON:0001013"]),
    "UBERON:0014455" => T("UBERON:0014455", "brown adipose"; def="Thermogenic fat", syn=["BAT"], par=["UBERON:0001013"]),
    "UBERON:0002509" => T("UBERON:0002509", "lymph node"; def="Lymph filter", par=["UBERON:0000062"]),
    "UBERON:0000992" => T("UBERON:0000992", "ovary"; def="Female gonad", par=["UBERON:0000062"]),
    "UBERON:0000473" => T("UBERON:0000473", "testis"; def="Male gonad", syn=["testicle"], par=["UBERON:0000062"]),
    "UBERON:0000995" => T("UBERON:0000995", "uterus"; def="Embryo development organ", syn=["womb"], par=["UBERON:0000062"]),
    "UBERON:0002110" => T("UBERON:0002110", "bile duct"; def="Bile transport", par=["UBERON:0000062"]),
)

#=============================================================================
  SENSORY ORGANS (15 terms)
=============================================================================#
const SENSORY = Dict{String,OBOTerm}(
    "UBERON:0000970" => T("UBERON:0000970", "eye"; def="Vision organ", syn=["eyeball", "oculus"], par=["UBERON:0000062"]),
    "UBERON:0000966" => T("UBERON:0000966", "retina"; def="Light-sensitive layer", par=["UBERON:0000970"]),
    "UBERON:0000965" => T("UBERON:0000965", "lens"; def="Focuses light", syn=["crystalline lens"], par=["UBERON:0000970"]),
    "UBERON:0000964" => T("UBERON:0000964", "cornea"; def="Transparent front of eye", par=["UBERON:0000970"]),
    "UBERON:0001801" => T("UBERON:0001801", "sclera"; def="White of eye", par=["UBERON:0000970"]),
    "UBERON:0001769" => T("UBERON:0001769", "iris"; def="Controls pupil size", par=["UBERON:0000970"]),
    "UBERON:0001766" => T("UBERON:0001766", "choroid"; def="Vascular layer of eye", par=["UBERON:0000970"]),
    "UBERON:0001690" => T("UBERON:0001690", "ear"; def="Hearing and balance", syn=["auris"], par=["UBERON:0000062"]),
    "UBERON:0001756" => T("UBERON:0001756", "middle ear"; def="Contains ossicles", syn=["tympanic cavity"], par=["UBERON:0001690"]),
    "UBERON:0001755" => T("UBERON:0001755", "inner ear"; def="Contains cochlea", syn=["labyrinth"], par=["UBERON:0001690"]),
    "UBERON:0001844" => T("UBERON:0001844", "cochlea"; def="Hearing structure", par=["UBERON:0001755"]),
    "UBERON:0001707" => T("UBERON:0001707", "nasal cavity"; def="Air passage in nose", par=["UBERON:0000062"]),
    "UBERON:0001723" => T("UBERON:0001723", "tongue"; def="Taste and speech organ", par=["UBERON:0000062"]),
    "UBERON:0001716" => T("UBERON:0001716", "taste bud"; def="Taste receptor cluster", par=["UBERON:0001723"]),
    "UBERON:0001850" => T("UBERON:0001850", "lacrimal gland"; def="Tear production", par=["UBERON:0000970"]),
)

#=============================================================================
  COMBINED DATABASE
=============================================================================#

"""All 150+ tissues combined."""
const TISSUES = merge(BONE, CARTILAGE, SKIN, CARDIOVASCULAR, NEURAL, MUSCLE, VISCERAL, SENSORY)

"""Tissues organized by anatomical system."""
const TISSUE_BY_SYSTEM = Dict{Symbol,Dict{String,OBOTerm}}(
    :skeletal => merge(BONE, CARTILAGE),
    :integumentary => SKIN,
    :cardiovascular => CARDIOVASCULAR,
    :nervous => NEURAL,
    :muscular => MUSCLE,
    :visceral => VISCERAL,
    :sensory => SENSORY,
)

#=============================================================================
  LOOKUP FUNCTIONS
=============================================================================#

"""Get tissue by UBERON ID."""
get_tissue(id::String) = get(TISSUES, id, nothing)

"""List tissues by system."""
function list_tissues(system::Symbol=:all)
    system == :all ? collect(values(TISSUES)) :
    haskey(TISSUE_BY_SYSTEM, system) ? collect(values(TISSUE_BY_SYSTEM[system])) : OBOTerm[]
end

"""Get child tissues."""
function get_subtissues(parent_id::String)
    [t for t in values(TISSUES) if parent_id in t.parents]
end

end # module
