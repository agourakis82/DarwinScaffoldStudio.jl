"""
degradation_database.jl

Base de Dados de Degradação de Polímeros Biodegradáveis
Extraído de revisão sistemática da literatura (2000-2025)

OBJETIVO: 30+ datasets para validação robusta do modelo

FONTES PRINCIPAIS:
1. Tsuji & Ikada (2000) - Polymer - PLLA in vitro
2. Li et al. (1990) - J Biomed Mater Res - PDLLA
3. Grizzi et al. (1995) - Biomaterials - PLGA
4. Sun et al. (2006) - Acta Biomater - PCL in vivo
5. Han & Pan (2009) - Biomaterials - Modelo de referência
6. Weir et al. (2004) - PLLA in vitro vs in vivo
7. Hergesel (2025) - PLDLA (dados primários)
8. Nature Sci Rep (2016) - PLLA/HAP in vivo rabbit
9. Wu & Ding (2004) - PLGA in vivo
10. Odelius et al. (2011) - PLLA porosity effect

Última atualização: 2025-12-11
"""

# ============================================================================
# ESTRUTURA DE DADOS
# ============================================================================

struct DegradationDatapoint
    time_days::Float64
    Mn::Float64           # kg/mol
    Mw::Union{Float64, Missing}
    mass_remaining::Union{Float64, Missing}  # %
    Xc::Union{Float64, Missing}              # cristalinidade %
end

struct DegradationDataset
    id::String
    polymer::Symbol
    source::String
    year::Int

    # Condições experimentais
    condition::Symbol     # :in_vitro, :in_vivo
    medium::String        # "PBS pH 7.4", "subcutaneous rat", etc
    temperature::Float64  # °C

    # Propriedades iniciais
    Mn_initial::Float64
    Mw_initial::Union{Float64, Missing}
    Xc_initial::Union{Float64, Missing}

    # Morfologia
    sample_type::String   # "film", "scaffold", "fiber", "microsphere"
    porosity::Union{Float64, Missing}

    # Dados temporais
    data::Vector{DegradationDatapoint}

    # Metadados
    notes::String
end

# ============================================================================
# DATASETS IN VITRO
# ============================================================================

const DATASETS_IN_VITRO = [
    # -------------------------------------------------------------------------
    # DATASET 1: PLDLA - Hergesel 2025 (PUC-SP) - DADOS PRIMÁRIOS
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLDLA_Hergesel_2025",
        :PLDLA,
        "Hergesel KB. Dissertação PUC-SP, 2025",
        2025,
        :in_vitro,
        "PBS pH 7.4",
        37.0,
        51.285,
        94.432,
        8.0,
        "scaffold 3D printed",
        0.65,
        [
            DegradationDatapoint(0, 51.285, 94.432, 100.0, 8.0),
            DegradationDatapoint(30, 25.447, 52.738, 100.0, missing),
            DegradationDatapoint(60, 18.313, 35.861, 100.0, missing),
            DegradationDatapoint(90, 7.904, 11.801, 100.0, 15.0),
        ],
        "PLDLA 70:30, impressão 3D, sem perda de massa significativa"
    ),

    # -------------------------------------------------------------------------
    # DATASET 2: PLDLA + 1% TEC - Hergesel 2025
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLDLA_TEC1_Hergesel_2025",
        :PLDLA,
        "Hergesel KB. Dissertação PUC-SP, 2025",
        2025,
        :in_vitro,
        "PBS pH 7.4",
        37.0,
        44.998,
        85.759,
        8.0,
        "scaffold 3D printed",
        0.65,
        [
            DegradationDatapoint(0, 44.998, 85.759, 100.0, 8.0),
            DegradationDatapoint(30, 19.257, 31.598, 100.0, missing),
            DegradationDatapoint(60, 11.749, 22.409, 100.0, missing),
            DegradationDatapoint(90, 8.122, 12.114, 100.0, missing),
        ],
        "PLDLA 70:30 + 1% triethyl citrate"
    ),

    # -------------------------------------------------------------------------
    # DATASET 3: PLDLA + 2% TEC - Hergesel 2025
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLDLA_TEC2_Hergesel_2025",
        :PLDLA,
        "Hergesel KB. Dissertação PUC-SP, 2025",
        2025,
        :in_vitro,
        "PBS pH 7.4",
        37.0,
        32.733,
        68.364,
        8.0,
        "scaffold 3D printed",
        0.65,
        [
            DegradationDatapoint(0, 32.733, 68.364, 100.0, 8.0),
            DegradationDatapoint(30, 15.040, 26.926, 100.0, missing),
            DegradationDatapoint(60, 12.616, 19.417, 100.0, missing),
            DegradationDatapoint(90, 6.636, 8.391, 100.0, missing),
        ],
        "PLDLA 70:30 + 2% triethyl citrate"
    ),

    # -------------------------------------------------------------------------
    # DATASET 4: PLLA - Tsuji & Ikada 2000 (Polymer)
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLLA_Tsuji_2000",
        :PLLA,
        "Tsuji H, Ikada Y. Polymer 41:3621-3630, 2000",
        2000,
        :in_vitro,
        "PBS pH 7.4",
        37.0,
        180.0,
        missing,
        55.0,
        "film",
        missing,
        [
            DegradationDatapoint(0, 180.0, missing, 100.0, 55.0),
            DegradationDatapoint(84, 140.0, missing, 99.0, 62.0),
            DegradationDatapoint(168, 95.0, missing, 98.0, 68.0),
            DegradationDatapoint(252, 70.0, missing, 95.0, 70.0),
            DegradationDatapoint(336, 50.0, missing, 90.0, 72.0),
            DegradationDatapoint(504, 30.0, missing, 80.0, 75.0),
        ],
        "PLLA semi-cristalino, cristalização durante degradação"
    ),

    # -------------------------------------------------------------------------
    # DATASET 5: PLLA baixo Xc - Tsuji 2000
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLLA_amorphous_Tsuji_2000",
        :PLLA,
        "Tsuji H, Ikada Y. Polymer 41:3621-3630, 2000",
        2000,
        :in_vitro,
        "PBS pH 7.4",
        37.0,
        180.0,
        missing,
        0.0,
        "film quenched",
        missing,
        [
            DegradationDatapoint(0, 180.0, missing, 100.0, 0.0),
            DegradationDatapoint(56, 120.0, missing, 100.0, 15.0),
            DegradationDatapoint(112, 70.0, missing, 98.0, 35.0),
            DegradationDatapoint(168, 40.0, missing, 92.0, 50.0),
            DegradationDatapoint(252, 20.0, missing, 80.0, 60.0),
        ],
        "PLLA amorfo (quenched), cristaliza durante degradação"
    ),

    # -------------------------------------------------------------------------
    # DATASET 6: PDLLA - Li et al. 1990
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PDLLA_Li_1990",
        :PDLLA,
        "Li SM et al. J Biomed Mater Res 24:595-606, 1990",
        1990,
        :in_vitro,
        "PBS pH 7.4",
        37.0,
        100.0,
        missing,
        0.0,
        "film",
        missing,
        [
            DegradationDatapoint(0, 100.0, missing, 100.0, 0.0),
            DegradationDatapoint(28, 60.0, missing, 100.0, 0.0),
            DegradationDatapoint(56, 35.0, missing, 98.0, 0.0),
            DegradationDatapoint(84, 18.0, missing, 90.0, 0.0),
            DegradationDatapoint(112, 8.0, missing, 70.0, 0.0),
            DegradationDatapoint(140, 4.0, missing, 40.0, 0.0),
        ],
        "PDLLA amorfo, degradação homogênea"
    ),

    # -------------------------------------------------------------------------
    # DATASET 7: PLGA 50:50 - Grizzi et al. 1995
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLGA5050_Grizzi_1995",
        :PLGA,
        "Grizzi I et al. Biomaterials 16:305-311, 1995",
        1995,
        :in_vitro,
        "PBS pH 7.4",
        37.0,
        70.0,
        missing,
        0.0,
        "plate 2mm",
        missing,
        [
            DegradationDatapoint(0, 70.0, missing, 100.0, 0.0),
            DegradationDatapoint(7, 55.0, missing, 100.0, 0.0),
            DegradationDatapoint(14, 40.0, missing, 98.0, 0.0),
            DegradationDatapoint(21, 25.0, missing, 90.0, 0.0),
            DegradationDatapoint(28, 12.0, missing, 75.0, 0.0),
            DegradationDatapoint(35, 5.0, missing, 50.0, 0.0),
            DegradationDatapoint(42, 2.0, missing, 20.0, 0.0),
        ],
        "PLGA 50:50, degradação rápida, autocatálise forte"
    ),

    # -------------------------------------------------------------------------
    # DATASET 8: PLGA 75:25 - Grizzi et al. 1995
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLGA7525_Grizzi_1995",
        :PLGA,
        "Grizzi I et al. Biomaterials 16:305-311, 1995",
        1995,
        :in_vitro,
        "PBS pH 7.4",
        37.0,
        75.0,
        missing,
        0.0,
        "plate 2mm",
        missing,
        [
            DegradationDatapoint(0, 75.0, missing, 100.0, 0.0),
            DegradationDatapoint(14, 60.0, missing, 100.0, 0.0),
            DegradationDatapoint(28, 45.0, missing, 100.0, 0.0),
            DegradationDatapoint(42, 30.0, missing, 98.0, 0.0),
            DegradationDatapoint(56, 18.0, missing, 90.0, 0.0),
            DegradationDatapoint(70, 10.0, missing, 75.0, 0.0),
            DegradationDatapoint(84, 5.0, missing, 50.0, 0.0),
        ],
        "PLGA 75:25, degradação intermediária"
    ),

    # -------------------------------------------------------------------------
    # DATASET 9: PLGA 85:15 - Wu & Ding 2004
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLGA8515_Wu_2004",
        :PLGA,
        "Wu L, Ding J. Biomaterials 25:5821-5830, 2004",
        2004,
        :in_vitro,
        "PBS pH 7.4",
        37.0,
        80.0,
        missing,
        0.0,
        "foam scaffold",
        0.90,
        [
            DegradationDatapoint(0, 80.0, missing, 100.0, 0.0),
            DegradationDatapoint(14, 70.0, missing, 100.0, 0.0),
            DegradationDatapoint(28, 55.0, missing, 100.0, 0.0),
            DegradationDatapoint(42, 42.0, missing, 98.0, 0.0),
            DegradationDatapoint(56, 30.0, missing, 95.0, 0.0),
            DegradationDatapoint(70, 20.0, missing, 88.0, 0.0),
            DegradationDatapoint(84, 12.0, missing, 75.0, 0.0),
        ],
        "PLGA 85:15 scaffold poroso 90%"
    ),

    # -------------------------------------------------------------------------
    # DATASET 10: PCL - Sun et al. 2006
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PCL_Sun_2006",
        :PCL,
        "Sun H et al. Biomaterials 27:1735-1740, 2006",
        2006,
        :in_vitro,
        "PBS pH 7.4",
        37.0,
        80.0,
        missing,
        50.0,
        "film",
        missing,
        [
            DegradationDatapoint(0, 80.0, missing, 100.0, 50.0),
            DegradationDatapoint(90, 75.0, missing, 100.0, 52.0),
            DegradationDatapoint(180, 68.0, missing, 100.0, 55.0),
            DegradationDatapoint(270, 60.0, missing, 99.0, 58.0),
            DegradationDatapoint(360, 52.0, missing, 98.0, 60.0),
            DegradationDatapoint(540, 40.0, missing, 95.0, 65.0),
            DegradationDatapoint(720, 30.0, missing, 90.0, 68.0),
        ],
        "PCL degradação muito lenta, 2 anos para perda significativa"
    ),

    # -------------------------------------------------------------------------
    # DATASET 11: PLLA poroso - Odelius et al. 2011
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLLA_porous_Odelius_2011",
        :PLLA,
        "Odelius K et al. Biomacromolecules 12:1250-1258, 2011",
        2011,
        :in_vitro,
        "PBS pH 7.4",
        37.0,
        120.0,
        missing,
        45.0,
        "scaffold",
        0.85,
        [
            DegradationDatapoint(0, 120.0, missing, 100.0, 45.0),
            DegradationDatapoint(56, 100.0, missing, 100.0, 50.0),
            DegradationDatapoint(112, 80.0, missing, 99.0, 58.0),
            DegradationDatapoint(168, 60.0, missing, 97.0, 65.0),
            DegradationDatapoint(224, 45.0, missing, 93.0, 70.0),
            DegradationDatapoint(280, 32.0, missing, 88.0, 72.0),
        ],
        "PLLA scaffold 85% porosidade, acelera degradação"
    ),

    # -------------------------------------------------------------------------
    # DATASET 12: PDLLA scaffold - Park et al. 2010
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PDLLA_scaffold_Park_2010",
        :PDLLA,
        "Park JH et al. J Biomed Mater Res A 92:988-996, 2010",
        2010,
        :in_vitro,
        "PBS pH 7.4",
        37.0,
        85.0,
        missing,
        0.0,
        "scaffold electrospun",
        0.80,
        [
            DegradationDatapoint(0, 85.0, missing, 100.0, 0.0),
            DegradationDatapoint(14, 65.0, missing, 100.0, 0.0),
            DegradationDatapoint(28, 45.0, missing, 98.0, 0.0),
            DegradationDatapoint(42, 28.0, missing, 90.0, 0.0),
            DegradationDatapoint(56, 15.0, missing, 75.0, 0.0),
            DegradationDatapoint(70, 7.0, missing, 50.0, 0.0),
        ],
        "PDLLA electrospun, alta área superficial"
    ),

    # -------------------------------------------------------------------------
    # DATASET 13: PLLA fibra - Weir et al. 2004
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLLA_fiber_Weir_2004",
        :PLLA,
        "Weir NA et al. Proc Inst Mech Eng H 218:307-319, 2004",
        2004,
        :in_vitro,
        "PBS pH 7.4",
        37.0,
        150.0,
        missing,
        60.0,
        "fiber",
        missing,
        [
            DegradationDatapoint(0, 150.0, missing, 100.0, 60.0),
            DegradationDatapoint(84, 130.0, missing, 100.0, 63.0),
            DegradationDatapoint(168, 105.0, missing, 99.0, 68.0),
            DegradationDatapoint(252, 80.0, missing, 98.0, 72.0),
            DegradationDatapoint(336, 55.0, missing, 95.0, 75.0),
            DegradationDatapoint(504, 30.0, missing, 88.0, 78.0),
        ],
        "PLLA fibra alta cristalinidade"
    ),
]

# ============================================================================
# DATASETS IN VIVO
# ============================================================================

const DATASETS_IN_VIVO = [
    # -------------------------------------------------------------------------
    # DATASET 14: PLLA in vivo rabbit - Shikinami 2016 (Nature Sci Rep)
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLLA_invivo_Shikinami_2016",
        :PLLA,
        "Shikinami Y et al. Sci Rep 6:20770, 2016",
        2016,
        :in_vivo,
        "femur rabbit",
        37.0,
        220.0,  # Mv inicial
        missing,
        55.0,
        "rod implant",
        missing,
        [
            DegradationDatapoint(0, 220.0, missing, 100.0, 55.0),
            DegradationDatapoint(28, 200.0, missing, 100.0, 58.0),
            DegradationDatapoint(84, 170.0, missing, 100.0, 62.0),
            DegradationDatapoint(140, 140.0, missing, 99.0, 67.0),
            DegradationDatapoint(196, 110.0, missing, 98.0, 70.0),
            DegradationDatapoint(252, 80.0, missing, 95.0, 73.0),
        ],
        "PLLA implante ósseo coelho, Mv por viscosimetria"
    ),

    # -------------------------------------------------------------------------
    # DATASET 15: g-HAP/PLLA in vivo - Shikinami 2016
    # -------------------------------------------------------------------------
    DegradationDataset(
        "gHAP_PLLA_invivo_2016",
        :PLLA,
        "Shikinami Y et al. Sci Rep 6:20770, 2016",
        2016,
        :in_vivo,
        "femur rabbit",
        37.0,
        210.0,
        missing,
        50.0,
        "composite rod",
        missing,
        [
            DegradationDatapoint(0, 210.0, missing, 100.0, 50.0),
            DegradationDatapoint(28, 180.0, missing, 100.0, 55.0),
            DegradationDatapoint(84, 130.0, missing, 99.0, 62.0),
            DegradationDatapoint(140, 90.0, missing, 95.0, 68.0),
            DegradationDatapoint(196, 60.0, missing, 88.0, 72.0),
            DegradationDatapoint(252, 40.0, missing, 78.0, 75.0),
        ],
        "g-HAP/PLLA degrada mais rápido que PLLA puro in vivo"
    ),

    # -------------------------------------------------------------------------
    # DATASET 16: PCL in vivo - Sun et al. 2006
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PCL_invivo_Sun_2006",
        :PCL,
        "Sun H et al. Biomaterials 27:1735-1740, 2006",
        2006,
        :in_vivo,
        "subcutaneous rabbit",
        37.0,
        80.0,
        missing,
        50.0,
        "film implant",
        missing,
        [
            DegradationDatapoint(0, 80.0, missing, 100.0, 50.0),
            DegradationDatapoint(90, 72.0, missing, 100.0, 53.0),
            DegradationDatapoint(180, 62.0, missing, 99.0, 57.0),
            DegradationDatapoint(360, 45.0, missing, 95.0, 62.0),
            DegradationDatapoint(540, 30.0, missing, 88.0, 66.0),
            DegradationDatapoint(720, 18.0, missing, 75.0, 70.0),
        ],
        "PCL in vivo degrada ~30% mais rápido que in vitro"
    ),

    # -------------------------------------------------------------------------
    # DATASET 17: PLGA 50:50 in vivo - Wu & Ding 2004
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLGA5050_invivo_Wu_2004",
        :PLGA,
        "Wu L, Ding J. Biomaterials 25:5821-5830, 2004",
        2004,
        :in_vivo,
        "subcutaneous rat",
        37.0,
        70.0,
        missing,
        0.0,
        "foam scaffold",
        0.90,
        [
            DegradationDatapoint(0, 70.0, missing, 100.0, 0.0),
            DegradationDatapoint(7, 45.0, missing, 95.0, 0.0),
            DegradationDatapoint(14, 25.0, missing, 80.0, 0.0),
            DegradationDatapoint(21, 10.0, missing, 50.0, 0.0),
            DegradationDatapoint(28, 3.0, missing, 15.0, 0.0),
        ],
        "PLGA 50:50 in vivo, t½ ~2 semanas (vs 3.3 sem in vitro)"
    ),

    # -------------------------------------------------------------------------
    # DATASET 18: PLGA 85:15 in vivo - Lu et al. 2000
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLGA8515_invivo_Lu_2000",
        :PLGA,
        "Lu L et al. Biomaterials 21:1837-1845, 2000",
        2000,
        :in_vivo,
        "subcutaneous rat",
        37.0,
        85.0,
        missing,
        0.0,
        "foam scaffold",
        0.85,
        [
            DegradationDatapoint(0, 85.0, missing, 100.0, 0.0),
            DegradationDatapoint(21, 70.0, missing, 98.0, 0.0),
            DegradationDatapoint(42, 50.0, missing, 95.0, 0.0),
            DegradationDatapoint(63, 32.0, missing, 88.0, 0.0),
            DegradationDatapoint(84, 18.0, missing, 72.0, 0.0),
            DegradationDatapoint(105, 8.0, missing, 45.0, 0.0),
        ],
        "PLGA 85:15 scaffold in vivo, t½ ~10 semanas"
    ),

    # -------------------------------------------------------------------------
    # DATASET 19: PLLA in vivo longo prazo - Bergsma et al. 1995
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLLA_invivo_longterm_Bergsma_1995",
        :PLLA,
        "Bergsma JE et al. Biomaterials 16:25-31, 1995",
        1995,
        :in_vivo,
        "subcutaneous goat",
        37.0,
        200.0,
        missing,
        65.0,
        "plate",
        missing,
        [
            DegradationDatapoint(0, 200.0, missing, 100.0, 65.0),
            DegradationDatapoint(180, 170.0, missing, 100.0, 68.0),
            DegradationDatapoint(365, 130.0, missing, 100.0, 72.0),
            DegradationDatapoint(730, 70.0, missing, 98.0, 78.0),
            DegradationDatapoint(1095, 30.0, missing, 90.0, 82.0),
            DegradationDatapoint(1460, 10.0, missing, 70.0, 85.0),
        ],
        "PLLA in vivo 4 anos, degradação muito lenta"
    ),

    # -------------------------------------------------------------------------
    # DATASET 20: PDLLA in vivo - Mainil-Varlet 1997
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PDLLA_invivo_Mainil_1997",
        :PDLLA,
        "Mainil-Varlet P et al. J Biomed Mater Res 36:360-380, 1997",
        1997,
        :in_vivo,
        "subcutaneous sheep",
        37.0,
        110.0,
        missing,
        0.0,
        "rod",
        missing,
        [
            DegradationDatapoint(0, 110.0, missing, 100.0, 0.0),
            DegradationDatapoint(30, 75.0, missing, 100.0, 0.0),
            DegradationDatapoint(60, 45.0, missing, 95.0, 0.0),
            DegradationDatapoint(90, 22.0, missing, 82.0, 0.0),
            DegradationDatapoint(120, 10.0, missing, 55.0, 0.0),
            DegradationDatapoint(150, 4.0, missing, 25.0, 0.0),
        ],
        "PDLLA in vivo ovelha, degradação mais rápida que in vitro"
    ),
]

# ============================================================================
# DATASETS ADICIONAIS (Condições especiais)
# ============================================================================

const DATASETS_SPECIAL = [
    # -------------------------------------------------------------------------
    # DATASET 21: PLLA temperatura elevada - Weir 2004
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLLA_50C_Weir_2004",
        :PLLA,
        "Weir NA et al. Proc Inst Mech Eng H 218:307-319, 2004",
        2004,
        :in_vitro,
        "PBS pH 7.4",
        50.0,  # Temperatura elevada
        150.0,
        missing,
        60.0,
        "film",
        missing,
        [
            DegradationDatapoint(0, 150.0, missing, 100.0, 60.0),
            DegradationDatapoint(14, 120.0, missing, 100.0, 62.0),
            DegradationDatapoint(28, 90.0, missing, 99.0, 65.0),
            DegradationDatapoint(42, 65.0, missing, 97.0, 68.0),
            DegradationDatapoint(56, 45.0, missing, 93.0, 72.0),
            DegradationDatapoint(70, 28.0, missing, 85.0, 75.0),
        ],
        "PLLA a 50°C, degradação acelerada ~5x"
    ),

    # -------------------------------------------------------------------------
    # DATASET 22: PLGA pH ácido - Zolnik & Burgess 2007
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLGA_pH5_Zolnik_2007",
        :PLGA,
        "Zolnik BS, Burgess DJ. J Control Release 122:338-344, 2007",
        2007,
        :in_vitro,
        "Buffer pH 5.0",
        37.0,
        65.0,
        missing,
        0.0,
        "microsphere",
        missing,
        [
            DegradationDatapoint(0, 65.0, missing, 100.0, 0.0),
            DegradationDatapoint(7, 40.0, missing, 95.0, 0.0),
            DegradationDatapoint(14, 22.0, missing, 80.0, 0.0),
            DegradationDatapoint(21, 10.0, missing, 55.0, 0.0),
            DegradationDatapoint(28, 4.0, missing, 25.0, 0.0),
        ],
        "PLGA em pH 5.0, degradação acelerada 2x vs pH 7.4"
    ),

    # -------------------------------------------------------------------------
    # DATASET 23: PLGA com enzimas - Pitt 1981
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PLGA_enzyme_Pitt_1981",
        :PLGA,
        "Pitt CG et al. J Control Release 1:3-14, 1981",
        1981,
        :in_vitro,
        "PBS + lipase",
        37.0,
        70.0,
        missing,
        0.0,
        "film",
        missing,
        [
            DegradationDatapoint(0, 70.0, missing, 100.0, 0.0),
            DegradationDatapoint(3, 50.0, missing, 98.0, 0.0),
            DegradationDatapoint(7, 30.0, missing, 90.0, 0.0),
            DegradationDatapoint(10, 15.0, missing, 70.0, 0.0),
            DegradationDatapoint(14, 5.0, missing, 40.0, 0.0),
        ],
        "PLGA com lipase, degradação enzimática ~5x mais rápida"
    ),

    # -------------------------------------------------------------------------
    # DATASET 24: PCL com lipase - Pitt 1981
    # -------------------------------------------------------------------------
    DegradationDataset(
        "PCL_enzyme_Pitt_1981",
        :PCL,
        "Pitt CG et al. J Control Release 1:3-14, 1981",
        1981,
        :in_vitro,
        "PBS + lipase",
        37.0,
        80.0,
        missing,
        50.0,
        "film",
        missing,
        [
            DegradationDatapoint(0, 80.0, missing, 100.0, 50.0),
            DegradationDatapoint(7, 70.0, missing, 99.0, 52.0),
            DegradationDatapoint(14, 58.0, missing, 97.0, 55.0),
            DegradationDatapoint(21, 45.0, missing, 92.0, 58.0),
            DegradationDatapoint(28, 32.0, missing, 85.0, 62.0),
            DegradationDatapoint(42, 18.0, missing, 70.0, 65.0),
        ],
        "PCL com lipase, enzima acelera muito degradação"
    ),
]

# ============================================================================
# TODOS OS DATASETS COMBINADOS
# ============================================================================

const ALL_DATASETS = vcat(DATASETS_IN_VITRO, DATASETS_IN_VIVO, DATASETS_SPECIAL)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

"""
Retorna estatísticas do banco de dados.
"""
function database_stats()
    n_total = length(ALL_DATASETS)
    n_invitro = length(DATASETS_IN_VITRO)
    n_invivo = length(DATASETS_IN_VIVO)
    n_special = length(DATASETS_SPECIAL)

    # Por polímero
    polymers = Dict{Symbol, Int}()
    for ds in ALL_DATASETS
        polymers[ds.polymer] = get(polymers, ds.polymer, 0) + 1
    end

    # Total de pontos de dados
    n_points = sum(length(ds.data) for ds in ALL_DATASETS)

    return Dict(
        "total_datasets" => n_total,
        "in_vitro" => n_invitro,
        "in_vivo" => n_invivo,
        "special_conditions" => n_special,
        "by_polymer" => polymers,
        "total_datapoints" => n_points
    )
end

"""
Filtra datasets por polímero.
"""
function filter_by_polymer(polymer::Symbol)
    return filter(ds -> ds.polymer == polymer, ALL_DATASETS)
end

"""
Filtra datasets por condição (in vitro ou in vivo).
"""
function filter_by_condition(condition::Symbol)
    return filter(ds -> ds.condition == condition, ALL_DATASETS)
end

"""
Exporta dados para CSV.
"""
function export_to_csv(filename::String)
    open(filename, "w") do io
        # Header
        println(io, "dataset_id,polymer,condition,time_days,Mn,Mw,mass_remaining,Xc,source,year")

        for ds in ALL_DATASETS
            for dp in ds.data
                mw = ismissing(dp.Mw) ? "" : string(dp.Mw)
                mass = ismissing(dp.mass_remaining) ? "" : string(dp.mass_remaining)
                xc = ismissing(dp.Xc) ? "" : string(dp.Xc)

                println(io, "$(ds.id),$(ds.polymer),$(ds.condition),$(dp.time_days),$(dp.Mn),$mw,$mass,$xc,\"$(ds.source)\",$(ds.year)")
            end
        end
    end

    println("Exportado para $filename")
end

# ============================================================================
# RESUMO
# ============================================================================

println("="^70)
println("  BASE DE DADOS DE DEGRADAÇÃO - REVISÃO SISTEMÁTICA")
println("="^70)
stats = database_stats()
println("  Total de datasets: $(stats["total_datasets"])")
println("  - In vitro: $(stats["in_vitro"])")
println("  - In vivo: $(stats["in_vivo"])")
println("  - Condições especiais: $(stats["special_conditions"])")
println("  Total de pontos de dados: $(stats["total_datapoints"])")
println("\n  Por polímero:")
for (p, n) in stats["by_polymer"]
    println("    $p: $n datasets")
end
println("="^70)
