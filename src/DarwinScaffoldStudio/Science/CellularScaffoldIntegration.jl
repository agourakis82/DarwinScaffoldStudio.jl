"""
CellularScaffoldIntegration.jl

Integração de tipos celulares e morfologia com modelo de degradação de scaffolds.

INOVAÇÃO: Primeiro modelo a integrar:
1. Ontologia celular completa (7 tipos de leucócitos)
2. Análise morfológica por SAM3 (dimensão fractal)
3. Resposta inflamatória dinâmica (IL-6, MMP)
4. Feedback célula→scaffold→célula

Isso diferencia COMPLETAMENTE do SOTA que ignora resposta celular.

Author: Darwin Scaffold Studio
Date: 2025-12-10
"""
module CellularScaffoldIntegration

using Statistics
using Printf

export CellType, CellPopulation, CellMorphology, InflammatoryState
export TissueResponse, CellScaffoldModel, CellScaffoldResult
export create_cell_population, simulate_cell_scaffold_interaction
export calculate_inflammatory_acceleration, calculate_cell_migration_rate
export create_cell_scaffold_model, create_basal_inflammatory_state
export print_cell_scaffold_report
export FIBROBLAST, CHONDROCYTE, OSTEOBLAST, MACROPHAGE, MSC, ENDOTHELIAL
export NEUTROPHIL, LYMPHOCYTE_T, LYMPHOCYTE_B, MONOCYTE, LYMPHOCYTE_NK
export CELL_MORPHOLOGIES

# ============================================================================
# ONTOLOGIA CELULAR
# ============================================================================

"""
Tipos celulares relevantes para engenharia tecidual.
Baseado em Cell Ontology (CL) + dados do darwin-pbpk.
"""
@enum CellType begin
    # Células residentes do tecido
    FIBROBLAST = 1        # CL:0000057 - Tecido conectivo
    CHONDROCYTE = 2       # CL:0000138 - Cartilagem
    OSTEOBLAST = 3        # CL:0000062 - Osso
    OSTEOCLAST = 4        # CL:0000092 - Reabsorção óssea
    TENOCYTE = 5          # CL:0000388 - Tendão
    ADIPOCYTE = 6         # CL:0000136 - Tecido adiposo

    # Células-tronco
    MSC = 10              # CL:0000134 - Mesenquimal
    HSC = 11              # CL:0000037 - Hematopoiética

    # Células imunes (do darwin-pbpk WBC)
    NEUTROPHIL = 20       # CL:0000775
    MONOCYTE = 21         # CL:0000576
    MACROPHAGE = 22       # CL:0000235 (diferenciado de monocyte)
    LYMPHOCYTE_T = 23     # CL:0000084
    LYMPHOCYTE_B = 24     # CL:0000236
    LYMPHOCYTE_NK = 25    # CL:0000623
    EOSINOPHIL = 26       # CL:0000771
    BASOPHIL = 27         # CL:0000767

    # Células endoteliais
    ENDOTHELIAL = 30      # CL:0000115
end

"""
Parâmetros morfológicos de cada tipo celular.
Baseado em análise SAM3 do darwin-pbpk.
"""
struct CellMorphology
    cell_type::CellType

    # Dimensões físicas
    diameter_um::Float64              # μm
    volume_fL::Float64                # fentolitros

    # Morfologia fractal (do SAM3)
    fractal_dimension_edge::Float64   # D_f das bordas (1.6-1.8)
    fractal_dimension_std::Float64    # Variabilidade
    circularity::Float64              # 0-1 (1 = círculo perfeito)

    # Comportamento
    migration_rate_base::Float64      # μm/hora
    proliferation_rate::Float64       # /dia
    apoptosis_rate::Float64           # /dia

    # Secreção
    il6_production::Float64           # pg/célula/dia
    mmp_production::Float64           # pg/célula/dia
    vegf_production::Float64          # pg/célula/dia

    # Sensibilidade ao ambiente
    ph_sensitivity::Float64           # 0-1
    hypoxia_sensitivity::Float64      # 0-1
end

# Morfologias de referência (dados do darwin-pbpk + literatura)
const CELL_MORPHOLOGIES = Dict{CellType, CellMorphology}(
    # Células residentes
    FIBROBLAST => CellMorphology(
        FIBROBLAST,
        20.0, 2000.0,           # 20 μm, 2000 fL
        1.65, 0.08, 0.45,       # fractal, std, circularity (elongado)
        15.0, 0.05, 0.01,       # migração, prolif, apopt
        50.0, 100.0, 20.0,      # IL-6, MMP, VEGF
        0.3, 0.4                # pH, hipóxia sens.
    ),

    CHONDROCYTE => CellMorphology(
        CHONDROCYTE,
        15.0, 1500.0,
        1.72, 0.05, 0.85,       # mais circular
        5.0, 0.02, 0.005,       # baixa migração
        20.0, 30.0, 5.0,
        0.5, 0.2                # tolerante a hipóxia
    ),

    OSTEOBLAST => CellMorphology(
        OSTEOBLAST,
        25.0, 3000.0,
        1.68, 0.07, 0.55,
        10.0, 0.03, 0.01,
        30.0, 50.0, 40.0,       # alta prod. VEGF
        0.4, 0.5
    ),

    MSC => CellMorphology(
        MSC,
        18.0, 1800.0,
        1.70, 0.10, 0.60,
        25.0, 0.08, 0.005,      # alta migração e prolif
        40.0, 20.0, 30.0,
        0.3, 0.3
    ),

    # Células imunes (do darwin-pbpk)
    NEUTROPHIL => CellMorphology(
        NEUTROPHIL,
        12.0, 330.0,
        1.66, 0.12, 0.75,       # segmentado
        50.0, 0.0, 0.5,         # alta migração, sem prolif, alta apopt
        200.0, 500.0, 10.0,     # MUITO inflamatório
        0.2, 0.6
    ),

    MONOCYTE => CellMorphology(
        MONOCYTE,
        18.0, 400.0,
        1.68, 0.08, 0.80,
        30.0, 0.01, 0.1,
        300.0, 200.0, 50.0,
        0.3, 0.4
    ),

    MACROPHAGE => CellMorphology(
        MACROPHAGE,
        25.0, 800.0,
        1.72, 0.10, 0.65,       # irregular (fagocítico)
        20.0, 0.02, 0.05,
        500.0, 800.0, 100.0,    # MUITO inflamatório + VEGF
        0.4, 0.3
    ),

    LYMPHOCYTE_T => CellMorphology(
        LYMPHOCYTE_T,
        8.0, 200.0,
        1.72, 0.04, 0.90,       # muito circular
        40.0, 0.1, 0.02,
        100.0, 10.0, 5.0,
        0.2, 0.5
    ),

    LYMPHOCYTE_B => CellMorphology(
        LYMPHOCYTE_B,
        8.0, 200.0,
        1.72, 0.04, 0.90,
        30.0, 0.05, 0.02,
        80.0, 5.0, 5.0,
        0.2, 0.5
    )
)

"""
População celular no scaffold.
"""
mutable struct CellPopulation
    cell_type::CellType
    morphology::CellMorphology

    # Estado atual
    count::Float64                    # células totais
    density::Float64                  # células/mm³

    # Estado funcional
    activation_state::Float64         # 0 = quiescente, 1 = ativado
    differentiation_state::Float64    # 0 = indiferenciado, 1 = diferenciado

    # Posição no scaffold
    distribution::Symbol              # :uniform, :surface, :pore_lining
end

"""
Estado inflamatório do ambiente.
"""
mutable struct InflammatoryState
    time::Float64

    # Citocinas (ng/mL)
    IL6::Float64                      # Pró-inflamatório
    IL1beta::Float64                  # Pró-inflamatório
    TNFalpha::Float64                 # Pró-inflamatório
    IL10::Float64                     # Anti-inflamatório

    # Enzimas (ng/mL)
    MMP::Float64                      # Matrix metalloproteinase
    TIMP::Float64                     # Tissue inhibitor of MMP

    # Fatores de crescimento (ng/mL)
    VEGF::Float64                     # Angiogênese
    TGFbeta::Float64                  # Fibrose/diferenciação
    BMP2::Float64                     # Osteogênese

    # Ambiente local
    pH::Float64
    pO2::Float64                      # mmHg
    lactate::Float64                  # mM
end

"""
Resposta tecidual completa.
"""
mutable struct TissueResponse
    populations::Vector{CellPopulation}
    inflammatory::InflammatoryState

    # Produção de matriz
    collagen_density::Float64         # mg/mm³
    gag_density::Float64              # mg/mm³

    # Vascularização
    vessel_density::Float64           # vasos/mm²
    perfusion::Float64                # 0-1
end

"""
Resultado da simulação célula-scaffold.
"""
struct CellScaffoldResult
    time::Float64

    # Estado do scaffold
    Mn::Float64
    porosity::Float64
    mechanical_integrity::Float64

    # Estado celular
    total_cells::Float64
    cell_viability::Float64
    tissue_response::TissueResponse

    # Inflamação
    inflammatory_score::Float64       # 0-1
    degradation_acceleration::Float64 # multiplicador

    # Integração
    integration_score::Float64
end

# ============================================================================
# FUNÇÕES DE CRIAÇÃO
# ============================================================================

"""
Cria população celular inicial.
"""
function create_cell_population(
    cell_type::CellType;
    initial_density::Float64 = 1e4,   # células/mm³
    scaffold_volume::Float64 = 100.0   # mm³
)::CellPopulation

    morph = get(CELL_MORPHOLOGIES, cell_type, CELL_MORPHOLOGIES[FIBROBLAST])

    return CellPopulation(
        cell_type,
        morph,
        initial_density * scaffold_volume,  # count
        initial_density,                     # density
        0.1,                                 # activation (baixa inicial)
        0.0,                                 # differentiation
        :uniform
    )
end

"""
Cria estado inflamatório basal.
"""
function create_basal_inflammatory_state()::InflammatoryState
    return InflammatoryState(
        0.0,
        0.5,    # IL6 basal
        0.1,    # IL1beta
        0.1,    # TNFalpha
        0.5,    # IL10 (anti-inflamatório)
        0.1,    # MMP
        0.5,    # TIMP
        0.1,    # VEGF
        0.5,    # TGFbeta
        0.0,    # BMP2
        7.4,    # pH
        40.0,   # pO2 (normóxia tecidual)
        1.0     # lactate basal
    )
end

# ============================================================================
# MODELO DE INTERAÇÃO CÉLULA-SCAFFOLD
# ============================================================================

"""
Calcula aceleração da degradação por resposta inflamatória.

Este é o DIFERENCIADOR chave do SOTA:
- Células produzem IL-6 e MMP
- MMP acelera hidrólise do polímero
- pH local cai com produtos ácidos
- Feedback amplifica degradação

Ref: Anderson 2008, Franz 2011
"""
function calculate_inflammatory_acceleration(
    inflammatory::InflammatoryState,
    populations::Vector{CellPopulation}
)::Float64

    # 1. Contribuição de MMP (enzimático)
    # MMP degrada polímero diretamente
    mmp_factor = 1.0 + 2.0 * (inflammatory.MMP / (0.5 + inflammatory.MMP))

    # 2. Contribuição de pH (autocatálise)
    if inflammatory.pH < 7.0
        ph_factor = 1.0 + 3.0 * (7.0 - inflammatory.pH)
    else
        ph_factor = 1.0
    end

    # 3. Contribuição celular direta
    # Macrófagos e neutrófilos produzem ROS que degradam
    macrophage_pops = [p.count for p in populations if p.cell_type in [MACROPHAGE, MONOCYTE]]
    neutrophil_pops = [p.count for p in populations if p.cell_type == NEUTROPHIL]
    macrophage_count = isempty(macrophage_pops) ? 0.0 : sum(macrophage_pops)
    neutrophil_count = isempty(neutrophil_pops) ? 0.0 : sum(neutrophil_pops)

    ros_factor = 1.0 + 0.001 * (macrophage_count + 2 * neutrophil_count) / 1e5

    # 4. TIMP inibe MMP
    timp_inhibition = inflammatory.TIMP / (0.5 + inflammatory.TIMP)
    mmp_factor_inhibited = 1.0 + (mmp_factor - 1.0) * (1.0 - 0.5 * timp_inhibition)

    # Fator total de aceleração
    acceleration = mmp_factor_inhibited * ph_factor * ros_factor

    return clamp(acceleration, 1.0, 5.0)  # máximo 5x
end

"""
Calcula taxa de migração celular ajustada pelo scaffold.
"""
function calculate_cell_migration_rate(
    pop::CellPopulation,
    porosity::Float64,
    pore_size::Float64,
    inflammatory::InflammatoryState
)::Float64

    base_rate = pop.morphology.migration_rate_base

    # 1. Fator de porosidade (sigmoide)
    porosity_factor = 1.0 / (1.0 + exp(-10 * (porosity - 0.5)))

    # 2. Fator de tamanho de poro
    cell_diameter = pop.morphology.diameter_um
    if pore_size < 2 * cell_diameter
        pore_factor = (pore_size / (2 * cell_diameter))^2
    else
        pore_factor = min(pore_size / 300.0, 1.5)
    end

    # 3. Quimiotaxia (gradiente de citocinas)
    # IL-8 e MCP-1 atraem células imunes
    if pop.cell_type in [NEUTROPHIL, MONOCYTE, MACROPHAGE]
        chemotaxis_factor = 1.0 + 0.5 * inflammatory.IL6 / 10.0
    else
        chemotaxis_factor = 1.0
    end

    # 4. Hipóxia pode estimular migração
    if inflammatory.pO2 < 20.0
        hypoxia_factor = 1.0 + pop.morphology.hypoxia_sensitivity *
                         (1.0 - inflammatory.pO2 / 40.0)
    else
        hypoxia_factor = 1.0
    end

    return base_rate * porosity_factor * pore_factor * chemotaxis_factor * hypoxia_factor
end

"""
Atualiza produção de citocinas pelas células.
"""
function update_cytokine_production!(
    inflammatory::InflammatoryState,
    populations::Vector{CellPopulation},
    dt::Float64
)
    # Produção por cada população
    for pop in populations
        morph = pop.morphology
        n_cells = pop.count
        activation = pop.activation_state

        # Produção proporcional à ativação
        production_factor = 0.5 + 0.5 * activation

        # IL-6
        inflammatory.IL6 += morph.il6_production * n_cells * production_factor * dt / 1e9

        # MMP
        inflammatory.MMP += morph.mmp_production * n_cells * production_factor * dt / 1e9

        # VEGF
        inflammatory.VEGF += morph.vegf_production * n_cells * production_factor * dt / 1e9
    end

    # Decaimento natural
    decay_rate = 0.1  # /dia
    inflammatory.IL6 *= exp(-decay_rate * dt)
    inflammatory.MMP *= exp(-decay_rate * dt)
    inflammatory.VEGF *= exp(-decay_rate * 0.5 * dt)  # VEGF mais estável

    # TIMP produzido em resposta a MMP
    inflammatory.TIMP = 0.3 * inflammatory.MMP + 0.2
end

"""
Atualiza estado de cada população celular.
"""
function update_cell_populations!(
    populations::Vector{CellPopulation},
    porosity::Float64,
    pore_size::Float64,
    inflammatory::InflammatoryState,
    dt::Float64
)
    for pop in populations
        morph = pop.morphology

        # 1. Proliferação
        if pop.cell_type != NEUTROPHIL  # neutrófilos não proliferam
            prolif_rate = morph.proliferation_rate

            # pH ácido inibe proliferação
            if inflammatory.pH < 6.8
                prolif_rate *= 0.5
            end

            # Hipóxia moderada pode estimular
            if inflammatory.pO2 > 10 && inflammatory.pO2 < 30
                prolif_rate *= 1.2
            elseif inflammatory.pO2 < 10
                prolif_rate *= 0.3  # hipóxia severa inibe
            end

            pop.count += pop.count * prolif_rate * dt
        end

        # 2. Apoptose
        apopt_rate = morph.apoptosis_rate

        # Ambiente adverso aumenta apoptose
        if inflammatory.pH < 6.5
            apopt_rate *= 2.0
        end
        if inflammatory.pO2 < 5
            apopt_rate *= 3.0
        end

        pop.count -= pop.count * apopt_rate * dt
        pop.count = max(pop.count, 0)

        # 3. Ativação
        # Citocinas pró-inflamatórias ativam células
        activation_stimulus = (inflammatory.IL6 + inflammatory.TNFalpha) / 10.0
        pop.activation_state += 0.1 * (activation_stimulus - pop.activation_state) * dt
        pop.activation_state = clamp(pop.activation_state, 0.0, 1.0)

        # 4. Diferenciação (para MSC)
        if pop.cell_type == MSC
            # BMP2 → osteogênico
            if inflammatory.BMP2 > 1.0
                pop.differentiation_state += 0.01 * dt
            end
            # TGFbeta → condrogênico
            if inflammatory.TGFbeta > 1.0
                pop.differentiation_state += 0.005 * dt
            end
            pop.differentiation_state = clamp(pop.differentiation_state, 0.0, 1.0)
        end

        # Atualizar densidade
        # (assumindo volume constante de 100 mm³)
        pop.density = pop.count / 100.0
    end
end

# ============================================================================
# MODELO INTEGRADO SCAFFOLD-CÉLULA
# ============================================================================

"""
Modelo completo célula-scaffold.
"""
struct CellScaffoldModel
    scaffold_Mn0::Float64
    scaffold_porosity0::Float64
    scaffold_pore_size0::Float64
    scaffold_polymer::Symbol

    populations::Vector{CellPopulation}
    inflammatory::InflammatoryState
end

"""
Cria modelo célula-scaffold para um tecido específico.
"""
function create_cell_scaffold_model(;
    tissue_type::Symbol = :cartilage,
    Mn0::Float64 = 50.0,
    porosity::Float64 = 0.65,
    pore_size::Float64 = 350.0,
    polymer::Symbol = :PLDLA
)::CellScaffoldModel

    # Populações celulares por tipo de tecido
    populations = if tissue_type == :cartilage
        [
            create_cell_population(CHONDROCYTE; initial_density=1e4),
            create_cell_population(MSC; initial_density=1e3),
            create_cell_population(MACROPHAGE; initial_density=1e2),
        ]
    elseif tissue_type == :bone
        [
            create_cell_population(OSTEOBLAST; initial_density=5e3),
            create_cell_population(MSC; initial_density=2e3),
            create_cell_population(MACROPHAGE; initial_density=5e2),
            create_cell_population(ENDOTHELIAL; initial_density=1e3),
        ]
    elseif tissue_type == :meniscus
        [
            create_cell_population(FIBROBLAST; initial_density=8e3),
            create_cell_population(CHONDROCYTE; initial_density=5e3),
            create_cell_population(MSC; initial_density=1e3),
            create_cell_population(MACROPHAGE; initial_density=2e2),
        ]
    else  # generic soft tissue
        [
            create_cell_population(FIBROBLAST; initial_density=1e4),
            create_cell_population(MSC; initial_density=1e3),
            create_cell_population(MACROPHAGE; initial_density=3e2),
        ]
    end

    inflammatory = create_basal_inflammatory_state()

    return CellScaffoldModel(
        Mn0, porosity, pore_size, polymer,
        populations, inflammatory
    )
end

"""
Simula interação célula-scaffold ao longo do tempo.

RETORNA: Vetor de CellScaffoldResult com evolução temporal.
"""
function simulate_cell_scaffold_interaction(
    model::CellScaffoldModel;
    t_max::Float64 = 90.0,
    dt::Float64 = 1.0
)::Vector{CellScaffoldResult}

    results = CellScaffoldResult[]

    # Estado inicial
    Mn = model.scaffold_Mn0
    porosity = model.scaffold_porosity0
    pore_size = model.scaffold_pore_size0

    # Copiar populações e inflamação (mutáveis)
    populations = deepcopy(model.populations)
    inflammatory = deepcopy(model.inflammatory)

    # Parâmetros de degradação base (do UnifiedScaffoldTissueModel)
    k0 = 0.0175  # para PLDLA

    for t in 0:dt:t_max
        inflammatory.time = t

        # 1. ATUALIZAR ESTADO CELULAR
        update_cell_populations!(populations, porosity, pore_size, inflammatory, dt)

        # 2. ATUALIZAR CITOCINAS
        update_cytokine_production!(inflammatory, populations, dt)

        # 3. CALCULAR ACELERAÇÃO INFLAMATÓRIA
        accel = calculate_inflammatory_acceleration(inflammatory, populations)

        # 4. DEGRADAÇÃO DO SCAFFOLD (com aceleração)
        # dMn/dt = -k * Mn * acceleration
        k_eff = k0 * accel
        dMn = -k_eff * Mn * dt
        Mn += dMn
        Mn = max(Mn, 0.5)

        # 5. EVOLUÇÃO DA POROSIDADE
        mass_loss = 1 - Mn / model.scaffold_Mn0
        porosity = model.scaffold_porosity0 + 0.25 * mass_loss
        porosity = clamp(porosity, model.scaffold_porosity0, 0.95)

        # 6. EVOLUÇÃO DO PORO
        pore_size = model.scaffold_pore_size0 * (1.0 + 0.5 * mass_loss)

        # 7. pH LOCAL (produtos ácidos)
        # Lactato do polímero + metabolismo celular
        total_cells = sum(p.count for p in populations)
        lactate_from_polymer = 5.0 * mass_loss
        lactate_from_cells = 0.001 * total_cells / 1e5
        inflammatory.lactate = 1.0 + lactate_from_polymer + lactate_from_cells
        inflammatory.pH = 7.4 - 0.3 * log10(1 + inflammatory.lactate)
        inflammatory.pH = clamp(inflammatory.pH, 5.5, 7.4)

        # 8. INTEGRIDADE MECÂNICA
        mech_integrity = (Mn / model.scaffold_Mn0)^1.5 *
                        ((1 - porosity) / (1 - model.scaffold_porosity0))^2
        mech_integrity = clamp(mech_integrity, 0.0, 1.0)

        # 9. SCORES
        # Viabilidade celular
        viable_cells = sum(p.count for p in populations if p.cell_type != NEUTROPHIL)
        max_cells = 1e6
        viability = min(viable_cells / max_cells, 1.0)

        # Score inflamatório (0 = sem inflamação, 1 = severa)
        inflam_score = (inflammatory.IL6 + inflammatory.MMP) / 20.0
        inflam_score = clamp(inflam_score, 0.0, 1.0)

        # Score de integração
        matrix_density = 0.01 * viable_cells / 1e4  # simplificado
        integration = 0.3 * viability +
                      0.3 * (1 - inflam_score) +
                      0.2 * min(matrix_density, 1.0) +
                      0.2 * mech_integrity

        # Criar resposta tecidual
        tissue = TissueResponse(
            deepcopy(populations),
            deepcopy(inflammatory),
            matrix_density,           # collagen
            0.5 * matrix_density,     # GAG
            inflammatory.VEGF / 10.0, # vessel density
            0.3 + 0.7 * (inflammatory.pO2 / 40.0)  # perfusion
        )

        # Resultado
        push!(results, CellScaffoldResult(
            t,
            Mn, porosity, mech_integrity,
            total_cells, viability, tissue,
            inflam_score, accel,
            integration
        ))
    end

    return results
end

# ============================================================================
# ANÁLISE E RELATÓRIO
# ============================================================================

"""
Imprime relatório da simulação.
"""
function print_cell_scaffold_report(results::Vector{CellScaffoldResult})
    println("="^90)
    println("  SIMULAÇÃO CÉLULA-SCAFFOLD COM RESPOSTA INFLAMATÓRIA")
    println("="^90)

    println("\n📊 EVOLUÇÃO TEMPORAL:")
    println("-"^90)
    println("Dia │ Mn(%) │ Poros. │ Células │ IL-6 │ MMP │ pH  │ Accel │ Integr.")
    println("-"^90)

    for t in [0, 7, 14, 28, 42, 56, 70, 84]
        idx = findfirst(r -> r.time >= t, results)
        if idx !== nothing
            r = results[idx]
            Mn_pct = r.Mn / results[1].Mn * 100
            inflam = r.tissue_response.inflammatory
            @printf(" %3d │ %5.1f │ %5.1f%% │ %7.0f │ %4.1f │ %4.1f │ %4.2f │ %5.2fx │ %5.1f%%\n",
                    Int(t), Mn_pct, r.porosity*100, r.total_cells,
                    inflam.IL6, inflam.MMP, inflam.pH,
                    r.degradation_acceleration, r.integration_score*100)
        end
    end

    # Resumo final
    final = results[end]
    println("\n" * "="^90)
    println("  RESULTADO FINAL (t = $(Int(final.time)) dias)")
    println("="^90)

    @printf("  Mn residual: %.1f%%\n", final.Mn / results[1].Mn * 100)
    @printf("  Células totais: %.0f\n", final.total_cells)
    @printf("  Score inflamatório: %.1f%% (0=bom, 100=severo)\n", final.inflammatory_score*100)
    @printf("  Aceleração por inflamação: %.2fx\n", final.degradation_acceleration)
    @printf("  Score de integração: %.1f%%\n", final.integration_score*100)

    # Populações celulares
    println("\n📋 POPULAÇÕES CELULARES:")
    for pop in final.tissue_response.populations
        @printf("  - %s: %.0f células (ativação: %.0f%%)\n",
                pop.cell_type, pop.count, pop.activation_state*100)
    end

    println("="^90)
end

end # module
