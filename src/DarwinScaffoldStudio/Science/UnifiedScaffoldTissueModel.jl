"""
UnifiedScaffoldTissueModel.jl

Modelo Unificado de Integração Scaffold-Tecido com:
1. Degradação de PLDLA (modelo PINN calibrado)
2. Remodelamento tecidual multi-fase
3. Dimensão fractal D (da rede vascular)
4. Variáveis biológicas PBPK
5. Percolação e conectividade

FUNDAMENTOS CIENTÍFICOS:
========================
- FractalBlood (darwin-pbpk-platform): D_vascular = 2.7 (Lei de Murray)
- Distribuição power-law de tempos de trânsito: α ≈ 1.37
- Teoria de percolação: limiar crítico φ_c ≈ 0.593 (3D)
- Golden ratio (φ): porosidade ótima ≈ 61.8% para muitos sistemas

REFERÊNCIAS:
===========
- Goirand et al. 2021, Nature Comm: Transporte anômalo em redes fractais
- Macheras 1996: Farmacocinética fractal
- Murray 1926: Lei de ramificação vascular
- Murphy et al. 2010: Tamanho de poro ótimo para osso
- Karageorgiou 2005: Requisitos de porosidade

Author: Darwin Scaffold Studio
Date: 2025-12-10
"""
module UnifiedScaffoldTissueModel

using Statistics
using Printf

# Implementação da função gamma (Lanczos approximation)
# Evita dependência de SpecialFunctions
function _gamma(z::Float64)::Float64
    if z < 0.5
        return π / (sin(π * z) * _gamma(1 - z))
    end

    z -= 1
    g = 7
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]

    x = c[1]
    for i in 1:g+1
        x += c[i+1] / (z + i)
    end

    t = z + g + 0.5
    return sqrt(2π) * t^(z + 0.5) * exp(-t) * x
end

export UnifiedModel, BiologicalParams, VascularParams, PercolationParams
export TissueType, ScaffoldDesign, IntegrationResult
export simulate_unified_model, predict_optimal_scaffold
export print_unified_report, calculate_fractal_metrics
export SOFT_TISSUE, HARD_TISSUE, MENISCUS_BIO, CARTILAGE_BIO, BONE_BIO
export MENISCUS_TYPE, CARTILAGE_TYPE, BONE_TYPE, SKIN_TYPE, MUSCLE_TYPE
export percolation_probability, effective_tortuosity
export calculate_Mn, calculate_Mn_advanced, calculate_porosity, calculate_pore_size, mechanical_integrity
export PolymerDegradationParams, POLYMER_PARAMS, get_polymer_params, create_polymer_scaffold

# ============================================================================
# CONSTANTES FÍSICAS E BIOLÓGICAS
# ============================================================================

# Dimensão fractal da rede vascular (Lei de Murray)
const D_VASCULAR = 2.7

# Expoente power-law para tempos de trânsito vascular
const ALPHA_TRANSIT = 1.37

# Golden ratio
const PHI = (1 + sqrt(5)) / 2  # ≈ 1.618

# Limiar de percolação 3D (rede cúbica)
const PHI_C_3D = 0.593

# Expoentes de percolação 3D
const PERCOLATION_BETA = 0.418    # expoente da ordem de parâmetro
const PERCOLATION_NU = 0.875      # expoente de correlação
const PERCOLATION_TAU = 2.189     # expoente de distribuição de clusters

# Constantes de difusão (cm²/s)
const D_OXYGEN = 2.0e-5           # O2 em tecido
const D_GLUCOSE = 6.7e-6          # Glicose em tecido
const D_VEGF = 1.0e-7             # VEGF (growth factor)

# ============================================================================
# ESTRUTURAS DE DADOS
# ============================================================================

"""
Tipo de tecido alvo
"""
@enum TissueType begin
    MENISCUS_TYPE = 1
    CARTILAGE_TYPE = 2
    BONE_TYPE = 3
    SKIN_TYPE = 4
    MUSCLE_TYPE = 5
end

"""
Parâmetros biológicos do tecido (inspirado em PBPK tissue_partition)
"""
Base.@kwdef struct BiologicalParams
    name::String = "generic"
    tissue_type::TissueType = MENISCUS_TYPE

    # Composição tecidual (baseado em Rodgers-Rowland)
    f_water::Float64 = 0.75           # Fração de água
    f_lipid::Float64 = 0.02           # Fração lipídica
    f_protein::Float64 = 0.20         # Fração proteica
    f_collagen::Float64 = 0.15        # Fração de colágeno (para cartilagem/osso)

    # Taxas celulares
    cell_migration_rate::Float64 = 40.0    # μm/dia
    cell_proliferation_rate::Float64 = 0.1 # /dia
    apoptosis_rate::Float64 = 0.01         # /dia

    # Metabolismo
    oxygen_consumption::Float64 = 1.0e-8   # mol/célula/s
    glucose_consumption::Float64 = 5.0e-9  # mol/célula/s
    lactate_production::Float64 = 1.0e-8   # mol/célula/s

    # Fatores de crescimento
    vegf_production::Float64 = 1.0e-12     # mol/célula/s
    bmp_sensitivity::Float64 = 1.0         # (osso)
    tgfb_sensitivity::Float64 = 1.0        # (cartilagem)

    # Requisitos de pH
    ph_optimal::Float64 = 7.4
    ph_tolerance::Float64 = 0.5            # desvio tolerável

    # Tempos característicos (dias)
    adhesion_time::Float64 = 2.0
    proliferation_start::Float64 = 7.0
    ecm_production_start::Float64 = 14.0
    remodeling_start::Float64 = 28.0
    maturation_time::Float64 = 90.0
end

# Parâmetros pré-definidos para cada tecido
const MENISCUS_BIO = BiologicalParams(
    name = "menisco",
    tissue_type = MENISCUS_TYPE,
    f_water = 0.72,
    f_lipid = 0.01,
    f_protein = 0.22,
    f_collagen = 0.20,
    cell_migration_rate = 35.0,
    cell_proliferation_rate = 0.08,
    tgfb_sensitivity = 1.2,
    maturation_time = 84.0
)

const CARTILAGE_BIO = BiologicalParams(
    name = "cartilagem",
    tissue_type = CARTILAGE_TYPE,
    f_water = 0.70,
    f_lipid = 0.01,
    f_protein = 0.25,
    f_collagen = 0.25,
    cell_migration_rate = 25.0,
    cell_proliferation_rate = 0.05,
    oxygen_consumption = 5.0e-9,  # avascular, menor consumo
    tgfb_sensitivity = 1.5,
    maturation_time = 112.0
)

const BONE_BIO = BiologicalParams(
    name = "osso",
    tissue_type = BONE_TYPE,
    f_water = 0.45,
    f_lipid = 0.02,
    f_protein = 0.35,
    f_collagen = 0.30,
    cell_migration_rate = 20.0,
    cell_proliferation_rate = 0.03,
    vegf_production = 2.0e-12,
    bmp_sensitivity = 2.0,
    remodeling_start = 42.0,
    maturation_time = 180.0
)

"""
Parâmetros de vascularização (baseado em FractalBlood)
"""
Base.@kwdef struct VascularParams
    # Geometria fractal
    fractal_dimension::Float64 = D_VASCULAR      # D ≈ 2.7
    branching_ratio::Float64 = 2.0               # número de ramos por bifurcação
    murray_exponent::Float64 = 3.0               # expoente da Lei de Murray

    # Distribuição de tempos de trânsito
    transit_alpha::Float64 = ALPHA_TRANSIT       # expoente power-law
    tau_min::Float64 = 0.1                       # tempo mínimo (s)
    tau_mean::Float64 = 20.0                     # tempo médio (s)

    # Difusão anômala (CTRW)
    beta_anomalous::Float64 = 0.8                # expoente de difusão anômala

    # Angiogênese
    capillary_density_target::Float64 = 500.0    # capilares/mm²
    angiogenesis_rate::Float64 = 5.0             # μm/dia de crescimento
    max_diffusion_distance::Float64 = 200.0      # μm (distância máxima sem vaso)
end

"""
Parâmetros de percolação para conectividade
"""
Base.@kwdef struct PercolationParams
    # Limiar crítico
    phi_c::Float64 = PHI_C_3D                    # 0.593 para 3D

    # Expoentes críticos
    beta::Float64 = PERCOLATION_BETA             # ordem de parâmetro
    nu::Float64 = PERCOLATION_NU                 # comprimento de correlação
    tau::Float64 = PERCOLATION_TAU               # distribuição de clusters

    # Dimensão fractal do cluster percolante
    df_percolating::Float64 = 2.53               # D_f em 3D

    # Tortuosidade
    tortuosity_exponent::Float64 = 0.5           # τ ∝ (φ - φ_c)^(-ν/2)
end

"""
Design do scaffold
"""
Base.@kwdef struct ScaffoldDesign
    # Morfologia inicial
    porosity::Float64 = 0.65
    pore_size::Float64 = 350.0        # μm
    strut_size::Float64 = 100.0       # μm

    # Material (PLDLA)
    Mn_initial::Float64 = 51.3        # kg/mol
    crystallinity::Float64 = 0.35     # fração cristalina (0-1)

    # Tipo de polímero para ajustes específicos
    polymer_type::Symbol = :PLDLA     # :PLLA, :PLDLA, :PLGA, :PCL, :PDLLA

    # Degradação (modelo calibrado)
    k0::Float64 = 0.0175              # /dia (calibrado com GPC)
    Ea::Float64 = 80.0                # kJ/mol
    autocatalysis::Float64 = 0.066

    # Arquitetura
    surface_area::Float64 = 10.0      # mm²/mm³
    interconnectivity::Float64 = 0.95 # fração de poros conectados
end

# ============================================================================
# PARÂMETROS DE DEGRADAÇÃO POR POLÍMERO (literatura)
# ============================================================================

# Referências:
# - Tsuji & Ikada 2000: PLLA cristalino degrada muito mais lento
# - Grizzi et al. 1995: Autocatálise bulk vs surface erosion
# - Li et al. 1990: PDLLA vs PLLA rates
# - Sun et al. 2006: PCL slow degradation

"""
Estrutura com parâmetros de degradação específicos por polímero.
Baseado em revisão extensa da literatura.
"""
struct PolymerDegradationParams
    name::String
    k0_base::Float64              # Taxa base de degradação (/dia)
    Ea::Float64                   # Energia de ativação (kJ/mol)
    autocatalysis_base::Float64   # Fator de autocatálise base
    crystallinity_typical::Float64  # Cristalinidade típica
    crystallinity_effect::Float64   # Quão forte cristalinidade afeta degradação
    water_uptake_rate::Float64      # Taxa de absorção de água (/dia)
    Tg::Float64                     # Temperatura de transição vítrea (°C)
end

# Parâmetros calibrados com dados da literatura
const POLYMER_PARAMS = Dict{Symbol, PolymerDegradationParams}(
    :PLDLA => PolymerDegradationParams(
        "PLDLA (70:30)",
        0.0175,     # k0 calibrado com dados Kaique
        80.0,       # Ea
        0.066,      # autocatálise
        0.10,       # baixa cristalinidade (amorfo)
        0.3,        # efeito moderado da cristalinidade
        0.02,       # absorção de água rápida
        50.0        # Tg baixa
    ),
    :PLLA => PolymerDegradationParams(
        "PLLA (semi-cristalino)",
        0.0075,     # k0 ajustado - degradação bifásica (amorfo rápido, cristalino lento)
        82.0,       # Ea ajustado
        0.045,      # autocatálise moderada (aumenta com degradação amorfa)
        0.55,       # alta cristalinidade típica
        0.6,        # efeito da cristalinidade ajustado para bifásico
        0.008,      # absorção de água lenta mas não mínima
        65.0        # Tg mais alta
    ),
    :PDLLA => PolymerDegradationParams(
        "PDLLA (amorfo)",
        0.022,      # k0 alto - totalmente amorfo
        78.0,       # Ea
        0.080,      # autocatálise alta
        0.0,        # sem cristalinidade
        0.0,        # sem efeito
        0.03,       # absorção de água muito rápida
        45.0        # Tg baixa
    ),
    :PLGA => PolymerDegradationParams(
        "PLGA (75:25)",
        0.030,      # k0 alto - GA acelera
        75.0,       # Ea menor
        0.12,       # autocatálise alta
        0.0,        # amorfo
        0.0,        # sem efeito
        0.04,       # absorção de água rápida
        48.0        # Tg moderada
    ),
    :PCL => PolymerDegradationParams(
        "PCL (semi-cristalino)",
        0.0015,     # k0 muito baixo
        90.0,       # Ea alta
        0.01,       # autocatálise mínima
        0.50,       # alta cristalinidade
        0.7,        # forte efeito
        0.001,      # absorção de água mínima
        -60.0       # Tg muito baixa (borrachoso)
    )
)

"""
Obtém parâmetros de degradação para um polímero.
Se não encontrado, retorna parâmetros genéricos.
"""
function get_polymer_params(polymer::Symbol)::PolymerDegradationParams
    return get(POLYMER_PARAMS, polymer, POLYMER_PARAMS[:PLDLA])
end

"""
Resultado da integração
"""
struct IntegrationResult
    time::Float64

    # Estado do scaffold
    Mn::Float64
    porosity::Float64
    pore_size::Float64
    mechanical_integrity::Float64

    # Estado do tecido
    cell_density::Float64             # células/mm³
    ecm_volume_fraction::Float64      # 0-1
    tissue_maturity::Float64          # 0-1

    # Vascularização
    vascular_density::Float64         # vasos/mm²
    oxygen_availability::Float64      # 0-1

    # Conectividade
    percolation_probability::Float64  # P_∞
    effective_tortuosity::Float64
    fractal_dimension::Float64        # D do tecido/scaffold

    # Scores
    integration_score::Float64        # 0-1
    viability_score::Float64          # 0-1
end

"""
Modelo unificado completo
"""
struct UnifiedModel
    scaffold::ScaffoldDesign
    biology::BiologicalParams
    vascular::VascularParams
    percolation::PercolationParams
end

function UnifiedModel(;
    tissue_type::TissueType = MENISCUS_TYPE,
    porosity::Float64 = 0.65,
    pore_size::Float64 = 350.0
)
    # Selecionar parâmetros biológicos
    bio = if tissue_type == MENISCUS_TYPE
        MENISCUS_BIO
    elseif tissue_type == CARTILAGE_TYPE
        CARTILAGE_BIO
    elseif tissue_type == BONE_TYPE
        BONE_BIO
    else
        BiologicalParams(tissue_type=tissue_type)
    end

    scaffold = ScaffoldDesign(porosity=porosity, pore_size=pore_size)
    vascular = VascularParams()
    percolation = PercolationParams()

    return UnifiedModel(scaffold, bio, vascular, percolation)
end

# ============================================================================
# FUNÇÕES MATEMÁTICAS FRACTAIS
# ============================================================================

"""
Função de Mittag-Leffler para cinética fractal.
E_{α,β}(z) = Σ z^k / Γ(αk + β)
"""
function mittag_leffler(α::Float64, β::Float64, z::Float64; n_terms::Int=100)::Float64
    result = 0.0
    z_power = 1.0

    for k in 0:n_terms
        term = z_power / _gamma(α * k + β)
        result += term

        if abs(term) < 1e-15
            break
        end

        z_power *= z
    end

    return result
end

mittag_leffler(α::Float64, z::Float64) = mittag_leffler(α, 1.0, z)

"""
Distribuição power-law para tempos de trânsito.
p(τ) = (α-1)/τ_min × (τ/τ_min)^(-α)
"""
function power_law_transit(t::Float64, α::Float64, τ_min::Float64)::Float64
    if t < τ_min
        return 0.0
    end
    return (α - 1) / τ_min * (t / τ_min)^(-α)
end

"""
Probabilidade de percolação P_∞(φ).
Comportamento crítico perto de φ_c.
"""
function percolation_probability(φ::Float64, params::PercolationParams)::Float64
    if φ < params.phi_c
        return 0.0
    end

    # P_∞ ∝ (φ - φ_c)^β
    P_inf = ((φ - params.phi_c) / (1.0 - params.phi_c))^params.beta
    return clamp(P_inf, 0.0, 1.0)
end

"""
Tortuosidade efetiva baseada em percolação.
τ diverge em φ_c.
"""
function effective_tortuosity(φ::Float64, params::PercolationParams)::Float64
    if φ <= params.phi_c
        return Inf
    end

    # τ ∝ (φ - φ_c)^(-ν/2)
    ξ = (φ - params.phi_c) / (1.0 - params.phi_c)
    τ = 1.0 + (1.0 / ξ)^params.tortuosity_exponent

    return min(τ, 10.0)  # cap máximo
end

"""
Dimensão fractal do scaffold/tecido.
Transição de D_scaffold para D_tissue durante remodelamento.
"""
function calculate_fractal_dimension(
    scaffold_porosity::Float64,
    tissue_fraction::Float64,
    vascular_fraction::Float64,
    vascular::VascularParams
)::Float64
    # Scaffold: D relacionado à porosidade
    # Para estruturas aleatórias: D ≈ 3 - β*log(1-φ)
    D_scaffold = 3.0 - 0.5 * log(1.0 - scaffold_porosity + 0.01)
    D_scaffold = clamp(D_scaffold, 2.0, 2.9)

    # Tecido: contribuição vascular (D ≈ 2.7)
    D_vascular = vascular.fractal_dimension

    # Interpolação baseada nas frações
    scaffold_weight = (1 - tissue_fraction) * (1 - vascular_fraction)
    tissue_weight = tissue_fraction * (1 - vascular_fraction)
    vascular_weight = vascular_fraction

    D_effective = scaffold_weight * D_scaffold +
                  tissue_weight * 2.5 +  # tecido não-vascular
                  vascular_weight * D_vascular

    return D_effective
end

# ============================================================================
# MODELO DE DEGRADAÇÃO (integrado do PINN calibrado)
# ============================================================================

"""
Degradação de Mn com modelo calibrado.
Versão básica - mantida para compatibilidade.
"""
function calculate_Mn(scaffold::ScaffoldDesign, t::Float64; T::Float64=310.15)::Float64
    return calculate_Mn_advanced(scaffold, t; T=T)
end

"""
Modelo avançado de degradação considerando:
1. Tipo específico de polímero (PLLA, PLDLA, PLGA, PCL, PDLLA)
2. Cristalinidade como barreira à hidrólise
3. Autocatálise heterogênea
4. Absorção de água dinâmica
5. Efeito de temperatura vs Tg
6. NOVO: Degradação bifásica para polímeros semi-cristalinos (PLLA, PCL)

Referências:
- Wang et al. 2019: Multi-physics degradation model
- Han & Pan 2009: Autocatalytic degradation kinetics
- Tsuji & Ikada 2000: Crystallinity effects on PLLA hydrolysis
- Weir et al. 2004: Two-phase degradation in semi-crystalline polymers
"""
function calculate_Mn_advanced(
    scaffold::ScaffoldDesign,
    t::Float64;
    T::Float64=310.15,
    use_polymer_params::Bool=true
)::Float64

    R = 8.314e-3  # kJ/(mol·K)
    T_ref = 310.15  # 37°C

    # Obter parâmetros específicos do polímero
    if use_polymer_params && haskey(POLYMER_PARAMS, scaffold.polymer_type)
        params = POLYMER_PARAMS[scaffold.polymer_type]
        k0 = params.k0_base
        Ea = params.Ea
        α_base = params.autocatalysis_base
        Xc_typical = params.crystallinity_typical
        Xc_effect = params.crystallinity_effect
        water_uptake = params.water_uptake_rate
        Tg = params.Tg
    else
        # Usar parâmetros do scaffold diretamente
        k0 = scaffold.k0
        Ea = scaffold.Ea
        α_base = scaffold.autocatalysis
        Xc_typical = 0.35
        Xc_effect = 0.3
        water_uptake = 0.02
        Tg = 50.0
    end

    # Fator de temperatura (Arrhenius)
    k_temp = k0 * exp(-Ea / R * (1/T - 1/T_ref))

    # =============================================
    # MODELO DE CRISTALINIDADE INICIAL
    # =============================================
    Xc_initial = scaffold.crystallinity

    # =============================================
    # MODELO DE ABSORÇÃO DE ÁGUA BASE
    # =============================================
    t_half_water = 7.0 / (1.0 + water_uptake * 50)

    # =============================================
    # EFEITO DE Tg
    # =============================================
    T_celsius = T - 273.15
    f_tg = T_celsius > Tg ? 1.0 + 0.1 * (T_celsius - Tg) / 10.0 : 1.0

    # =============================================
    # MODELO BIFÁSICO PARA SEMI-CRISTALINOS
    # =============================================
    # Polímeros semi-cristalinos (PLLA, PCL) degradam em duas fases:
    # Fase 1: Degradação rápida da região amorfa
    # Fase 2: Degradação lenta da região cristalina
    # Ref: Weir et al. 2004, Tsuji & Ikada 2000

    is_semicrystalline = scaffold.polymer_type in [:PLLA, :PCL] && Xc_initial > 0.3

    # =============================================
    # INTEGRAÇÃO NUMÉRICA COM MODELO BIFÁSICO
    # =============================================
    Mn = scaffold.Mn_initial
    Mn0 = scaffold.Mn_initial
    dt = 0.5  # dia

    # Estado dinâmico da cristalinidade
    Xc_current = Xc_initial
    amorphous_fraction = 1.0 - Xc_initial

    for ti in 0:dt:t
        # Absorção de água (dinâmica)
        f_water = 1.0 - exp(-0.693 * ti / t_half_water)
        f_water_eff = f_water * (1.0 - 0.4 * Xc_current)

        # Extensão da degradação
        degradation_extent = 1.0 - Mn / Mn0

        if is_semicrystalline
            # ========================================
            # MODELO BIFÁSICO
            # ========================================
            # Fase 1: Degradação preferencial da região amorfa
            # Fase 2: Após ~70% degradação amorfa, cristais começam a degradar

            # Fração amorfa restante (diminui com degradação)
            amorphous_remaining = max(0.0, amorphous_fraction - degradation_extent * 0.8)

            # Cristalinidade aparente aumenta à medida que amorfo degrada
            # (fenômeno real observado em PLLA - Tsuji 2000)
            if amorphous_remaining > 0.1
                Xc_current = Xc_initial + 0.15 * degradation_extent
                Xc_current = min(Xc_current, 0.75)  # máximo ~75%
            end

            # Taxa de degradação bifásica
            if amorphous_remaining > 0.15
                # Fase 1: Degradação da região amorfa (mais rápida)
                k_amorphous = k_temp * 2.0  # amorfo degrada 2x mais rápido
                k_crystalline = k_temp * 0.15  # cristalino muito mais lento

                # Média ponderada pelas frações
                k_eff = k_amorphous * amorphous_remaining + k_crystalline * Xc_current
            else
                # Fase 2: Principalmente degradação cristalina (mais lenta)
                k_eff = k_temp * 0.4 * (1.0 + degradation_extent)
            end

            # Autocatálise (aumenta na fase 2 devido acúmulo de ácido)
            α_eff = α_base * (1.0 + 0.5 * degradation_extent)

        else
            # ========================================
            # MODELO PADRÃO (amorfos e semi-amorfos)
            # ========================================
            f_crystallinity = (1.0 - Xc_current)^(1 + Xc_effect)
            k_eff = k_temp * f_crystallinity * f_water_eff * f_tg
            α_eff = α_base * (1.0 - 0.5 * Xc_current)
        end

        # Termo de autocatálise
        autocatalysis_term = 1.0 + α_eff * degradation_extent

        # Equação diferencial
        dMn = -k_eff * Mn * autocatalysis_term * f_water_eff * f_tg

        # Atualização
        Mn += dMn * dt
        Mn = max(Mn, 0.5)
    end

    return Mn
end

"""
Wrapper para criar scaffold com parâmetros específicos de polímero.
"""
function create_polymer_scaffold(
    polymer::Symbol;
    Mn_initial::Float64 = 50.0,
    porosity::Float64 = 0.65,
    pore_size::Float64 = 350.0,
    crystallinity::Union{Float64, Nothing} = nothing
)::ScaffoldDesign

    params = get_polymer_params(polymer)

    # Usar cristalinidade típica se não especificada
    Xc = isnothing(crystallinity) ? params.crystallinity_typical : crystallinity

    return ScaffoldDesign(
        porosity = porosity,
        pore_size = pore_size,
        Mn_initial = Mn_initial,
        crystallinity = Xc,
        polymer_type = polymer,
        k0 = params.k0_base,
        Ea = params.Ea,
        autocatalysis = params.autocatalysis_base
    )
end

"""
Evolução da porosidade durante degradação.
"""
function calculate_porosity(scaffold::ScaffoldDesign, t::Float64, Mn::Float64)::Float64
    mass_loss = 1 - Mn / scaffold.Mn_initial

    # Erosão superficial + degradação bulk
    erosion = 0.002 * t
    bulk = 0.3 * mass_loss

    porosity = scaffold.porosity + erosion + bulk
    return clamp(porosity, scaffold.porosity, 0.95)
end

"""
Evolução do tamanho de poro.
"""
function calculate_pore_size(scaffold::ScaffoldDesign, t::Float64, porosity::Float64)::Float64
    # Coalescência de poros
    porosity_increase = porosity - scaffold.porosity
    growth_factor = 1.0 + 0.8 * porosity_increase + 0.001 * t

    pore_size = scaffold.pore_size * growth_factor
    return min(pore_size, scaffold.pore_size * 4.0)
end

"""
Integridade mecânica (Gibson-Ashby + Mn).
"""
function mechanical_integrity(scaffold::ScaffoldDesign, porosity::Float64, Mn::Float64)::Float64
    # Gibson-Ashby: E ∝ (1-φ)^2
    ga_factor = ((1 - porosity) / (1 - scaffold.porosity))^2

    # Degradação de Mn
    mn_factor = (Mn / scaffold.Mn_initial)^2

    integrity = ga_factor * mn_factor
    return clamp(integrity, 0.0, 1.0)
end

# ============================================================================
# MODELO DE INTEGRAÇÃO TECIDUAL
# ============================================================================

"""
Taxa de migração celular ajustada pela morfologia.
"""
function adjusted_migration_rate(
    bio::BiologicalParams,
    porosity::Float64,
    pore_size::Float64
)::Float64
    # Fator de porosidade (sigmoide)
    φ_opt = 0.7
    φ_factor = 1.0 / (1.0 + exp(-10 * (porosity - 0.5)))

    # Fator de tamanho de poro
    min_pore = 100.0  # μm
    if pore_size < min_pore
        pore_factor = (pore_size / min_pore)^2
    else
        pore_factor = min(pore_size / 300.0, 1.5)
    end

    return bio.cell_migration_rate * φ_factor * pore_factor
end

"""
Disponibilidade de oxigênio baseada em difusão e vascularização.
"""
function oxygen_availability(
    porosity::Float64,
    vascular_density::Float64,
    cell_density::Float64,
    bio::BiologicalParams,
    vascular::VascularParams
)::Float64
    # Distância máxima de difusão
    if vascular_density > 0
        avg_vessel_spacing = 1000.0 / sqrt(vascular_density)  # μm
    else
        avg_vessel_spacing = 1000.0  # default
    end

    # Fator de difusão (Krogh cylinder model simplificado)
    if avg_vessel_spacing <= vascular.max_diffusion_distance
        diffusion_factor = 1.0
    else
        diffusion_factor = (vascular.max_diffusion_distance / avg_vessel_spacing)^2
    end

    # Consumo pelas células
    max_cells = 1e6  # células/mm³
    consumption_factor = 1.0 - 0.5 * (cell_density / max_cells)

    # Porosidade ajuda difusão
    porosity_factor = porosity / 0.7

    O2 = diffusion_factor * consumption_factor * porosity_factor
    return clamp(O2, 0.1, 1.0)
end

"""
Taxa de angiogênese baseada em VEGF e hipóxia.
"""
function angiogenesis_rate(
    oxygen::Float64,
    cell_density::Float64,
    bio::BiologicalParams,
    vascular::VascularParams
)::Float64
    # Hipóxia estimula VEGF
    hypoxia_factor = max(0, 1.0 - oxygen)

    # Produção de VEGF pelas células
    vegf = bio.vegf_production * cell_density * hypoxia_factor

    # Taxa de crescimento vascular
    rate = vascular.angiogenesis_rate * (1.0 + 10.0 * vegf / 1e-10)

    return min(rate, vascular.angiogenesis_rate * 3.0)
end

# ============================================================================
# SIMULAÇÃO PRINCIPAL
# ============================================================================

"""
Simula modelo unificado ao longo do tempo.
"""
function simulate_unified_model(model::UnifiedModel; t_max::Float64=180.0, dt::Float64=1.0)
    results = IntegrationResult[]

    # Estados iniciais
    cell_density = 100.0       # células/mm³ (adesão inicial)
    ecm_fraction = 0.0
    tissue_maturity = 0.0
    vascular_density = 0.0     # vasos/mm²

    for t in 0:dt:t_max
        # 1. ESTADO DO SCAFFOLD
        Mn = calculate_Mn(model.scaffold, t)
        porosity = calculate_porosity(model.scaffold, t, Mn)
        pore_size = calculate_pore_size(model.scaffold, t, porosity)
        mech_integ = mechanical_integrity(model.scaffold, porosity, Mn)

        # 2. PERCOLAÇÃO E CONECTIVIDADE
        P_perc = percolation_probability(porosity, model.percolation)
        tortuosity = effective_tortuosity(porosity, model.percolation)

        # 3. DISPONIBILIDADE DE OXIGÊNIO
        O2_avail = oxygen_availability(
            porosity, vascular_density, cell_density,
            model.biology, model.vascular
        )

        # 4. DINÂMICA CELULAR
        if t > model.biology.adhesion_time
            # Migração ajustada
            migration = adjusted_migration_rate(model.biology, porosity, pore_size)

            # Proliferação (logística) ajustada por O2
            max_cells = 1e6 * porosity
            prolif = model.biology.cell_proliferation_rate *
                     cell_density * (1 - cell_density/max_cells) * O2_avail

            # Apoptose (aumenta com baixo O2)
            apop = model.biology.apoptosis_rate * cell_density * (1.5 - O2_avail)

            cell_density += (migration + prolif - apop) * dt
            cell_density = max(cell_density, 0)
        end

        # 5. PRODUÇÃO DE ECM
        if t >= model.biology.ecm_production_start
            # Taxa aumentada para refletir produção real
            ecm_rate = 0.001 * (cell_density / 1e4) * O2_avail * (1 + 0.5 * tissue_maturity)
            ecm_fraction += ecm_rate * dt
            ecm_fraction = min(ecm_fraction, porosity * 0.8)
        end

        # 6. VASCULARIZAÇÃO (apenas osso e tecidos que precisam)
        if model.biology.tissue_type == BONE_TYPE && t > 21
            angio_rate = angiogenesis_rate(O2_avail, cell_density, model.biology, model.vascular)
            vascular_density += angio_rate * 0.01 * dt
            vascular_density = min(vascular_density, model.vascular.capillary_density_target)
        end

        # 7. MATURAÇÃO TECIDUAL
        if t >= model.biology.remodeling_start
            # Maturação baseada em ECM e tempo
            ecm_contribution = ecm_fraction > 0.01 ? ecm_fraction / (porosity * 0.5) : 0.0
            time_contribution = (t - model.biology.remodeling_start) / model.biology.maturation_time
            maturation_rate = 0.005 * O2_avail * (0.3 + 0.7 * ecm_contribution + 0.3 * time_contribution)
            tissue_maturity += maturation_rate * dt
            tissue_maturity = min(tissue_maturity, 1.0)
        end

        # 8. DIMENSÃO FRACTAL
        vascular_fraction = vascular_density / model.vascular.capillary_density_target
        D_eff = calculate_fractal_dimension(
            porosity, ecm_fraction, vascular_fraction, model.vascular
        )

        # 9. SCORES DE INTEGRAÇÃO
        # Viabilidade: oxigênio + integridade mecânica
        viability = 0.5 * O2_avail + 0.5 * min(mech_integ, 1.0)

        # Integração: células + ECM + maturidade
        integration = 0.25 * min(cell_density / 1e5, 1.0) +
                      0.25 * (ecm_fraction / (porosity * 0.5)) +
                      0.30 * tissue_maturity +
                      0.20 * P_perc
        integration = clamp(integration, 0.0, 1.0)

        # Resultado
        result = IntegrationResult(
            t,
            Mn, porosity, pore_size, mech_integ,
            cell_density, ecm_fraction, tissue_maturity,
            vascular_density, O2_avail,
            P_perc, tortuosity, D_eff,
            integration, viability
        )
        push!(results, result)
    end

    return results
end

# ============================================================================
# OTIMIZAÇÃO DE DESIGN
# ============================================================================

"""
Encontra design ótimo de scaffold para um tecido específico.
"""
function predict_optimal_scaffold(
    tissue_type::TissueType;
    porosity_range::Tuple{Float64,Float64} = (0.5, 0.85),
    pore_size_range::Tuple{Float64,Float64} = (200.0, 500.0),
    n_samples::Int = 10
)
    best_score = 0.0
    best_design = nothing
    best_results = nothing

    porosities = range(porosity_range[1], porosity_range[2], length=n_samples)
    pore_sizes = range(pore_size_range[1], pore_size_range[2], length=n_samples)

    for φ in porosities
        for d in pore_sizes
            model = UnifiedModel(tissue_type=tissue_type, porosity=φ, pore_size=d)
            results = simulate_unified_model(model; t_max=180.0)

            # Score final
            final = results[end]
            score = final.integration_score * final.viability_score

            if score > best_score
                best_score = score
                best_design = (porosity=φ, pore_size=d)
                best_results = results
            end
        end
    end

    return best_design, best_results, best_score
end

"""
Calcula métricas fractais ao longo do tempo.
"""
function calculate_fractal_metrics(results::Vector{IntegrationResult})
    times = [r.time for r in results]
    D_values = [r.fractal_dimension for r in results]
    P_perc = [r.percolation_probability for r in results]
    tau = [r.effective_tortuosity for r in results]

    return Dict(
        "times" => times,
        "fractal_dimension" => D_values,
        "percolation_probability" => P_perc,
        "tortuosity" => tau,
        "D_mean" => mean(D_values),
        "D_final" => D_values[end],
        "D_vascular_reference" => D_VASCULAR,
        "phi_golden" => 1/PHI  # ≈ 0.618
    )
end

# ============================================================================
# RELATÓRIO
# ============================================================================

"""
Imprime relatório completo do modelo unificado.
"""
function print_unified_report(model::UnifiedModel, results::Vector{IntegrationResult})
    println("="^90)
    println("  MODELO UNIFICADO SCAFFOLD-TECIDO")
    println("  Integrando: Degradação + Remodelamento + PBPK + Dimensão Fractal")
    println("="^90)

    bio = model.biology
    scaffold = model.scaffold
    vascular = model.vascular
    perc = model.percolation

    println("\n📊 PARÂMETROS DO MODELO:")
    println("-"^70)
    println("  Tecido: $(bio.name)")
    @printf("  Composição: %.0f%% água, %.0f%% proteína, %.0f%% colágeno\n",
            bio.f_water*100, bio.f_protein*100, bio.f_collagen*100)
    @printf("  Scaffold: φ=%.1f%%, poro=%.0fμm, Mn=%.1f kg/mol\n",
            scaffold.porosity*100, scaffold.pore_size, scaffold.Mn_initial)

    println("\n🔬 PARÂMETROS FRACTAIS (FractalBlood):")
    println("-"^70)
    @printf("  Dimensão fractal vascular: D = %.2f (Lei de Murray)\n", vascular.fractal_dimension)
    @printf("  Expoente power-law trânsito: α = %.2f\n", vascular.transit_alpha)
    @printf("  Expoente difusão anômala: β = %.2f\n", vascular.beta_anomalous)
    @printf("  Golden ratio φ = %.4f → porosidade ótima ≈ %.1f%%\n", PHI, 100/PHI)

    println("\n🌐 PARÂMETROS DE PERCOLAÇÃO:")
    println("-"^70)
    @printf("  Limiar crítico φ_c = %.3f (3D)\n", perc.phi_c)
    @printf("  Expoente β = %.3f (ordem de parâmetro)\n", perc.beta)
    @printf("  Dimensão fractal cluster: D_f = %.2f\n", perc.df_percolating)

    # Evolução temporal
    println("\n📈 EVOLUÇÃO TEMPORAL:")
    println("-"^90)
    println("Dia │ Mn(kg/mol) │ Porosid. │ Poro(μm) │ Células/mm³ │  ECM  │ D_fract │ Integração")
    println("-"^90)

    for t in [0, 7, 14, 28, 42, 56, 84, 112, 140, 180]
        idx = findfirst(r -> r.time >= t, results)
        if idx !== nothing
            r = results[idx]
            @printf(" %3d │   %5.1f    │  %5.1f%%  │  %5.0f   │   %6.0f    │ %4.1f%% │  %4.2f  │   %5.1f%%\n",
                    Int(t), r.Mn, r.porosity*100, r.pore_size,
                    r.cell_density, r.ecm_volume_fraction*100,
                    r.fractal_dimension, r.integration_score*100)
        end
    end

    # Métricas fractais
    metrics = calculate_fractal_metrics(results)

    println("\n🔷 MÉTRICAS FRACTAIS:")
    println("-"^70)
    @printf("  D inicial: %.2f\n", metrics["fractal_dimension"][1])
    @printf("  D final: %.2f\n", metrics["D_final"])
    @printf("  D médio: %.2f\n", metrics["D_mean"])
    @printf("  D vascular referência: %.2f\n", metrics["D_vascular_reference"])

    # Análise final
    final = results[end]

    println("\n" * "="^90)
    println("  ANÁLISE FINAL (t = $(Int(final.time)) dias)")
    println("="^90)

    @printf("  Score de integração: %.1f%%\n", final.integration_score*100)
    @printf("  Score de viabilidade: %.1f%%\n", final.viability_score*100)
    @printf("  Integridade mecânica: %.1f%%\n", final.mechanical_integrity*100)
    @printf("  Probabilidade de percolação: %.1f%%\n", final.percolation_probability*100)

    if final.integration_score > 0.7 && final.viability_score > 0.5
        println("\n✅ PROGNÓSTICO: Integração bem-sucedida esperada")
    elseif final.integration_score > 0.5
        println("\n⚠️  PROGNÓSTICO: Integração parcial - monitoramento necessário")
    else
        println("\n❌ PROGNÓSTICO: Alto risco de falha na integração")
    end

    println("="^90)

    return metrics
end

end # module
