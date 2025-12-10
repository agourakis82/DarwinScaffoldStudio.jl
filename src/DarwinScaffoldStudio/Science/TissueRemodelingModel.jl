"""
TissueRemodelingModel.jl

Modelo de remodelamento tecidual acoplado à degradação do scaffold PLDLA.

QUESTÕES FUNDAMENTAIS:
1. A degradação AUMENTA a porosidade (não diminui!)
   - Erosão superficial → poros maiores
   - Degradação bulk → novos microporos
   - Coalescência de poros

2. Impacto na integração tecidual:
   - Porosidade inicial muito baixa (<50%) → dificulta invasão celular
   - Porosidade muito alta (>90%) → perda de integridade mecânica
   - Janela ótima: 60-85% dependendo do tecido

3. Remodelamento tecidual:
   - Tecidos MOLES (menisco, cartilagem): 4-12 semanas
   - Tecidos DUROS (osso): 12-52 semanas

Baseado em:
- Murphy et al. 2010 (pore size optimal)
- Karageorgiou 2005 (porosity requirements)
- Hollister 2005 (scaffold design)
"""
module TissueRemodelingModel

using Statistics
using Printf

export TissueParams, ScaffoldState, TissueState, IntegrationModel
export predict_tissue_integration, predict_remodeling_timeline
export simulate_full_integration, print_integration_report
export calculate_scaffold_state, identify_remodeling_phases
export MENISCUS, CARTILAGE, BONE

# ============================================================================
# TIPOS E ESTRUTURAS
# ============================================================================

"""
Parâmetros específicos do tecido alvo
"""
Base.@kwdef struct TissueParams
    name::String = "menisco"
    type::Symbol = :soft              # :soft ou :hard

    # Taxas de invasão celular (células/dia/mm²)
    cell_migration_rate::Float64 = 50.0
    cell_proliferation_rate::Float64 = 0.1  # /dia

    # Produção de matriz extracelular
    ecm_production_rate::Float64 = 0.02     # mg/dia/célula

    # Remodelamento
    remodeling_start::Float64 = 14.0        # dias para iniciar
    remodeling_rate::Float64 = 0.01         # /dia

    # Requisitos de porosidade
    min_porosity::Float64 = 0.5             # mínimo para invasão
    optimal_porosity::Float64 = 0.7         # ótimo
    max_porosity::Float64 = 0.9             # máximo antes de colapso

    # Requisitos de poro
    min_pore_size::Float64 = 100.0          # μm - mínimo para células
    optimal_pore_size::Float64 = 300.0      # μm

    # Tempo de maturação (dias)
    maturation_time::Float64 = 90.0
end

# Tecidos pré-definidos
const MENISCUS = TissueParams(
    name = "menisco",
    type = :soft,
    cell_migration_rate = 40.0,
    cell_proliferation_rate = 0.08,
    ecm_production_rate = 0.015,
    remodeling_start = 14.0,
    remodeling_rate = 0.015,
    min_porosity = 0.55,
    optimal_porosity = 0.70,
    max_porosity = 0.85,
    min_pore_size = 150.0,
    optimal_pore_size = 350.0,
    maturation_time = 84.0  # 12 semanas
)

const CARTILAGE = TissueParams(
    name = "cartilagem",
    type = :soft,
    cell_migration_rate = 30.0,
    cell_proliferation_rate = 0.05,
    ecm_production_rate = 0.025,
    remodeling_start = 21.0,
    remodeling_rate = 0.008,
    min_porosity = 0.60,
    optimal_porosity = 0.75,
    max_porosity = 0.90,
    min_pore_size = 200.0,
    optimal_pore_size = 400.0,
    maturation_time = 112.0  # 16 semanas
)

const BONE = TissueParams(
    name = "osso",
    type = :hard,
    cell_migration_rate = 20.0,
    cell_proliferation_rate = 0.03,
    ecm_production_rate = 0.01,
    remodeling_start = 28.0,
    remodeling_rate = 0.005,
    min_porosity = 0.50,
    optimal_porosity = 0.65,
    max_porosity = 0.80,
    min_pore_size = 100.0,
    optimal_pore_size = 300.0,
    maturation_time = 180.0  # 26 semanas
)

"""
Estado do scaffold em um momento t
"""
struct ScaffoldState
    time::Float64
    Mn::Float64              # kg/mol
    porosity::Float64        # 0-1
    pore_size::Float64       # μm
    surface_area::Float64    # mm²/mm³
    mechanical_integrity::Float64  # 0-1
end

"""
Estado do tecido em formação
"""
struct TissueState
    time::Float64
    cell_density::Float64    # células/mm³
    ecm_volume::Float64      # fração de volume (0-1)
    tissue_maturity::Float64 # 0-1
    vascularization::Float64 # 0-1 (apenas para osso)
    integration_score::Float64 # 0-1
end

"""
Modelo completo de integração scaffold-tecido
"""
struct IntegrationModel
    tissue::TissueParams

    # Parâmetros do scaffold inicial
    initial_porosity::Float64
    initial_pore_size::Float64
    initial_Mn::Float64

    # Parâmetros de degradação (do modelo calibrado)
    k_degradation::Float64
end

function IntegrationModel(tissue::TissueParams;
                          porosity::Float64=0.65,
                          pore_size::Float64=350.0,
                          Mn::Float64=51.0)
    # k calibrado para PLDLA
    IntegrationModel(tissue, porosity, pore_size, Mn, 0.020)
end

# ============================================================================
# MODELO DE EVOLUÇÃO DO SCAFFOLD
# ============================================================================

"""
Calcula estado do scaffold durante degradação.
IMPORTANTE: Porosidade AUMENTA com degradação!
"""
function calculate_scaffold_state(model::IntegrationModel, t::Float64)
    # Degradação de Mn (modelo calibrado)
    k = model.k_degradation
    Mn = model.initial_Mn * exp(-k * t * (1 + 0.01 * t))  # com autocatálise
    Mn = max(Mn, 1.0)

    # Fração de massa perdida
    mass_loss_fraction = 1 - Mn / model.initial_Mn

    # POROSIDADE AUMENTA com degradação
    # Mecanismos:
    # 1. Erosão superficial dos struts
    # 2. Formação de microporos internos (degradação bulk)
    # 3. Coalescência de poros adjacentes

    erosion_rate = 0.003  # /dia
    bulk_degradation_factor = mass_loss_fraction * 0.25

    porosity = model.initial_porosity + erosion_rate * t + bulk_degradation_factor
    porosity = clamp(porosity, model.initial_porosity, 0.95)

    # TAMANHO DE PORO também aumenta (coalescência)
    pore_growth_factor = 1.0 + 0.5 * mass_loss_fraction + 0.002 * t
    pore_size = model.initial_pore_size * pore_growth_factor
    pore_size = min(pore_size, model.initial_pore_size * 3.0)  # máximo 3x

    # Área superficial específica (diminui com erosão)
    # S = 4φ/d para estrutura idealizada
    if pore_size > 0
        surface_area = 4 * porosity / (pore_size / 1000)  # mm²/mm³
    else
        surface_area = 0.0
    end

    # Integridade mecânica (Gibson-Ashby + Mn)
    Mn_factor = (Mn / model.initial_Mn)^2
    porosity_factor = ((1 - porosity) / (1 - model.initial_porosity))^2
    mechanical_integrity = Mn_factor * porosity_factor
    mechanical_integrity = clamp(mechanical_integrity, 0.0, 1.0)

    return ScaffoldState(t, Mn, porosity, pore_size, surface_area, mechanical_integrity)
end

# ============================================================================
# MODELO DE INTEGRAÇÃO TECIDUAL
# ============================================================================

"""
Calcula fator de favorabilidade para invasão celular baseado na morfologia.
"""
function calculate_invasion_factor(scaffold::ScaffoldState, tissue::TissueParams)
    # Fator de porosidade (sigmoidal)
    if scaffold.porosity < tissue.min_porosity
        porosity_factor = exp(-5 * (tissue.min_porosity - scaffold.porosity))
    elseif scaffold.porosity > tissue.max_porosity
        porosity_factor = exp(-3 * (scaffold.porosity - tissue.max_porosity))
    else
        # Ótimo na faixa ideal
        dist_to_optimal = abs(scaffold.porosity - tissue.optimal_porosity)
        porosity_factor = 1.0 - 0.5 * dist_to_optimal / 0.2
    end

    # Fator de tamanho de poro
    if scaffold.pore_size < tissue.min_pore_size
        pore_factor = (scaffold.pore_size / tissue.min_pore_size)^2
    else
        pore_factor = min(scaffold.pore_size / tissue.optimal_pore_size, 1.5)
        pore_factor = min(pore_factor, 1.0) + 0.5 * max(0, pore_factor - 1.0)
    end

    # Fator de integridade mecânica (precisa de suporte)
    mech_factor = 0.5 + 0.5 * scaffold.mechanical_integrity

    return porosity_factor * pore_factor * mech_factor
end

"""
Simula formação de tecido ao longo do tempo.
"""
function calculate_tissue_state(model::IntegrationModel,
                                scaffold::ScaffoldState,
                                prev_tissue::Union{TissueState, Nothing}=nothing)
    t = scaffold.time
    tissue = model.tissue
    dt = 1.0  # dia

    # Estado anterior ou inicial
    if prev_tissue === nothing
        cell_density = 100.0  # células iniciais (adesão)
        ecm_volume = 0.0
        tissue_maturity = 0.0
        vascularization = 0.0
    else
        cell_density = prev_tissue.cell_density
        ecm_volume = prev_tissue.ecm_volume
        tissue_maturity = prev_tissue.tissue_maturity
        vascularization = prev_tissue.vascularization
    end

    # Fator de invasão baseado na morfologia do scaffold
    invasion_factor = calculate_invasion_factor(scaffold, tissue)

    # Espaço disponível para células (poros não preenchidos)
    available_space = scaffold.porosity - ecm_volume
    available_space = max(available_space, 0.0)

    # Migração celular (proporcional à área superficial e espaço)
    if t > 0 && available_space > 0.1
        migration = tissue.cell_migration_rate * invasion_factor * available_space * dt
        cell_density += migration
    end

    # Proliferação celular (logística)
    max_density = 1e6 * scaffold.porosity  # densidade máxima
    proliferation = tissue.cell_proliferation_rate * cell_density * (1 - cell_density/max_density) * dt
    cell_density += max(proliferation, 0)

    # Produção de ECM
    if t > 7  # após adesão inicial
        ecm_production = tissue.ecm_production_rate * cell_density * 1e-6 * invasion_factor * dt
        ecm_volume += ecm_production
        ecm_volume = min(ecm_volume, scaffold.porosity * 0.9)  # máximo 90% dos poros
    end

    # Remodelamento e maturação
    if t > tissue.remodeling_start
        remodeling_progress = tissue.remodeling_rate * (t - tissue.remodeling_start) / tissue.maturation_time
        tissue_maturity = min(remodeling_progress + ecm_volume / scaffold.porosity * 0.5, 1.0)
    end

    # Vascularização (apenas para osso)
    if tissue.type == :hard && t > 21
        vasc_rate = 0.005
        vascularization = min(vasc_rate * (t - 21) / 100, 1.0)
    end

    # Score de integração (média ponderada)
    integration_score = 0.3 * min(cell_density / 1e5, 1.0) +
                        0.3 * ecm_volume / (scaffold.porosity * 0.5) +
                        0.3 * tissue_maturity +
                        0.1 * (tissue.type == :hard ? vascularization : 1.0)
    integration_score = clamp(integration_score, 0.0, 1.0)

    return TissueState(t, cell_density, ecm_volume, tissue_maturity, vascularization, integration_score)
end

# ============================================================================
# SIMULAÇÃO COMPLETA
# ============================================================================

"""
Simula integração completa scaffold-tecido ao longo do tempo.
"""
function simulate_full_integration(model::IntegrationModel;
                                   t_max::Float64=180.0, dt::Float64=1.0)
    scaffold_states = ScaffoldState[]
    tissue_states = TissueState[]

    prev_tissue = nothing

    for t in 0:dt:t_max
        # Estado do scaffold
        scaffold = calculate_scaffold_state(model, t)
        push!(scaffold_states, scaffold)

        # Estado do tecido
        tissue_state = calculate_tissue_state(model, scaffold, prev_tissue)
        push!(tissue_states, tissue_state)

        prev_tissue = tissue_state
    end

    return scaffold_states, tissue_states
end

"""
Identifica fases do remodelamento tecidual.
"""
function identify_remodeling_phases(tissue_states::Vector{TissueState}, tissue::TissueParams)
    phases = Dict{String, Tuple{Float64, Float64}}()

    # Fase 1: Adesão inicial (0-7 dias)
    phases["Adesão"] = (0.0, 7.0)

    # Fase 2: Proliferação (7-21 dias para moles, 7-28 para duros)
    prolif_end = tissue.type == :soft ? 21.0 : 28.0
    phases["Proliferação"] = (7.0, prolif_end)

    # Fase 3: Síntese de ECM
    ecm_start = prolif_end
    ecm_end = tissue.remodeling_start + 14.0
    phases["Síntese ECM"] = (ecm_start, ecm_end)

    # Fase 4: Remodelamento
    phases["Remodelamento"] = (tissue.remodeling_start, tissue.maturation_time)

    # Fase 5: Maturação
    phases["Maturação"] = (tissue.maturation_time, tissue.maturation_time + 60.0)

    return phases
end

"""
Prediz timeline de remodelamento para um tecido específico.
"""
function predict_remodeling_timeline(model::IntegrationModel)
    tissue = model.tissue

    timeline = Dict{String, Any}()

    # Simular
    scaffold_states, tissue_states = simulate_full_integration(model; t_max=300.0)

    # Encontrar marcos importantes
    for (i, ts) in enumerate(tissue_states)
        t = ts.time

        # 50% de integração
        if !haskey(timeline, "integration_50") && ts.integration_score >= 0.5
            timeline["integration_50"] = t
        end

        # 80% de integração
        if !haskey(timeline, "integration_80") && ts.integration_score >= 0.8
            timeline["integration_80"] = t
        end

        # 50% de maturidade
        if !haskey(timeline, "maturity_50") && ts.tissue_maturity >= 0.5
            timeline["maturity_50"] = t
        end

        # Maturidade completa
        if !haskey(timeline, "maturity_complete") && ts.tissue_maturity >= 0.95
            timeline["maturity_complete"] = t
        end
    end

    # Encontrar quando scaffold perde integridade
    for (i, ss) in enumerate(scaffold_states)
        if !haskey(timeline, "scaffold_degraded") && ss.mechanical_integrity < 0.1
            timeline["scaffold_degraded"] = ss.time
            break
        end
    end

    # Verificar se integração ocorre antes da degradação
    if haskey(timeline, "integration_80") && haskey(timeline, "scaffold_degraded")
        timeline["successful_integration"] = timeline["integration_80"] < timeline["scaffold_degraded"]
    else
        timeline["successful_integration"] = false
    end

    return timeline, scaffold_states, tissue_states
end

# ============================================================================
# RELATÓRIO
# ============================================================================

"""
Imprime relatório completo de integração.
"""
function print_integration_report(model::IntegrationModel)
    tissue = model.tissue

    println("="^80)
    println("  RELATÓRIO DE INTEGRAÇÃO SCAFFOLD-TECIDO")
    println("  Tecido: $(tissue.name) ($(tissue.type))")
    println("="^80)

    # Parâmetros iniciais
    println("\n📦 SCAFFOLD INICIAL:")
    println("   Porosidade: $(model.initial_porosity * 100)%")
    println("   Tamanho poro: $(model.initial_pore_size) μm")
    println("   Mn: $(model.initial_Mn) kg/mol")

    # Requisitos do tecido
    println("\n🎯 REQUISITOS DO TECIDO:")
    println("   Porosidade: $(tissue.min_porosity*100)% - $(tissue.max_porosity*100)%")
    println("   Poro mínimo: $(tissue.min_pore_size) μm")
    println("   Tempo maturação: $(tissue.maturation_time) dias")

    # Simular e obter timeline
    timeline, scaffold_states, tissue_states = predict_remodeling_timeline(model)

    # Evolução temporal
    println("\n📈 EVOLUÇÃO TEMPORAL:")
    println("-"^70)
    println("Tempo │ Porosidade │ Poro (μm) │ Integ.Mec │ Células │ ECM  │ Integração")
    println("-"^70)

    for t in [0, 7, 14, 28, 42, 56, 84, 112, 140, 180]
        idx = findfirst(s -> s.time >= t, scaffold_states)
        if idx !== nothing
            ss = scaffold_states[idx]
            ts = tissue_states[idx]

            @printf(" %4d  │   %5.1f%%   │   %5.0f   │   %5.1f%%  │ %5.0fk │ %4.1f%% │   %5.1f%%\n",
                    t, ss.porosity*100, ss.pore_size, ss.mechanical_integrity*100,
                    ts.cell_density/1000, ts.ecm_volume*100, ts.integration_score*100)
        end
    end

    # Timeline de marcos
    println("\n⏱️  MARCOS IMPORTANTES:")
    println("-"^50)

    if haskey(timeline, "integration_50")
        @printf("   50%% integração: %.0f dias\n", timeline["integration_50"])
    end
    if haskey(timeline, "integration_80")
        @printf("   80%% integração: %.0f dias\n", timeline["integration_80"])
    end
    if haskey(timeline, "maturity_50")
        @printf("   50%% maturidade: %.0f dias\n", timeline["maturity_50"])
    end
    if haskey(timeline, "scaffold_degraded")
        @printf("   Scaffold degradado (<10%% integridade): %.0f dias\n", timeline["scaffold_degraded"])
    end

    # Fases do remodelamento
    phases = identify_remodeling_phases(tissue_states, tissue)

    println("\n🔄 FASES DO REMODELAMENTO:")
    println("-"^50)
    for (phase, (t_start, t_end)) in sort(collect(phases), by=x->x[2][1])
        @printf("   %-15s: dias %3.0f - %3.0f\n", phase, t_start, t_end)
    end

    # Avaliação final
    println("\n" * "="^80)
    if get(timeline, "successful_integration", false)
        println("✅ PROGNÓSTICO: Integração bem-sucedida esperada")
        println("   Tecido atinge 80% integração ANTES da degradação do scaffold")
    else
        println("⚠️  PROGNÓSTICO: Risco de falha na integração")
        println("   Scaffold pode degradar antes da integração completa")
    end
    println("="^80)

    return timeline, scaffold_states, tissue_states
end

end # module
