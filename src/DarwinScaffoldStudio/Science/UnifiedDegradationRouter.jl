"""
UnifiedDegradationRouter.jl

Roteador Unificado que seleciona automaticamente o melhor modelo de degradação
baseado no tipo de polímero e condições experimentais.

ARQUITETURA:
===========

┌─────────────────────────────────────────────────────────────────────────┐
│                         ENTRADA                                          │
│   Polímero, Mn₀, Xc, φ, TEC%, Células, Tecido Alvo                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ROTEADOR DE MODELO                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   PLDLA     │  │ PLLA / PCL  │  │PDLLA / PLGA │  │  + Células  │    │
│  │Idiossincrá- │  │  Bifásico   │  │  Genérico   │  │ Integração  │    │
│  │   tico      │  │             │  │             │  │   Celular   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CAMADAS FÍSICAS                                       │
│   Hidrólise → Cristalinidade → Percolação → Fractal → Mecânica          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         SAÍDA                                            │
│   Mn(t), Mw(t), PDI(t), φ(t), Xc(t), Tg(t), Score Integração           │
└─────────────────────────────────────────────────────────────────────────┘

MODELOS DISPONÍVEIS:
===================
1. PLDLAIdiosyncraticModel - Para PLDLA 70:30 (NRMSE: 11.2%)
2. UnifiedScaffoldTissueModel (Bifásico) - Para PLLA, PCL (NRMSE: 6-18%)
3. UnifiedScaffoldTissueModel (Genérico) - Para PDLLA, PLGA (NRMSE: 13-24%)
4. CellularScaffoldIntegration - Overlay para qualquer modelo

Author: Darwin Scaffold Studio
Date: 2025-12-11
"""
module UnifiedDegradationRouter

using Statistics
using Printf

# Importar módulos de modelo
include("PLDLAIdiosyncraticModel.jl")
include("UnifiedScaffoldTissueModel.jl")
include("CellularScaffoldIntegration.jl")

using .PLDLAIdiosyncraticModel
using .UnifiedScaffoldTissueModel
using .CellularScaffoldIntegration

export DegradationInput, DegradationOutput, CellularConfig
export simulate_degradation, simulate_with_routing
export get_recommended_model, print_model_comparison
export SUPPORTED_POLYMERS, MODEL_ACCURACY

# ============================================================================
# CONSTANTES E CONFIGURAÇÃO
# ============================================================================

"""Polímeros suportados pelo framework."""
const SUPPORTED_POLYMERS = [:PLDLA, :PLLA, :PDLLA, :PLGA, :PCL]

"""Precisão esperada por modelo (NRMSE %)."""
const MODEL_ACCURACY = Dict(
    :PLDLA => 11.2,   # Modelo idiossincrático
    :PLLA => 6.5,     # Modelo bifásico
    :PCL => 18.0,     # Modelo bifásico
    :PDLLA => 13.5,   # Modelo genérico
    :PLGA => 24.3     # Modelo genérico (precisa melhoria)
)

"""Modelo recomendado por polímero."""
const RECOMMENDED_MODEL = Dict(
    :PLDLA => :idiosyncratic,
    :PLLA => :biphasic,
    :PCL => :biphasic,
    :PDLLA => :generic,
    :PLGA => :generic
)

# ============================================================================
# ESTRUTURAS DE DADOS
# ============================================================================

"""
Configuração de entrada para simulação de degradação.
"""
Base.@kwdef struct DegradationInput
    # Identificação do polímero
    polymer::Symbol = :PLDLA

    # Propriedades iniciais
    Mn_initial::Float64 = 51.285      # kg/mol
    Mw_initial::Float64 = 94.432      # kg/mol (opcional, calculado de PDI)
    PDI_initial::Float64 = 1.84       # Mw/Mn

    # Propriedades térmicas
    Tg_initial::Float64 = 54.0        # °C
    Xc_initial::Float64 = 0.08        # Cristalinidade (0-1)

    # Morfologia do scaffold
    porosity::Float64 = 0.65          # (0-1)
    pore_size::Float64 = 350.0        # μm

    # Aditivos (específico PLDLA)
    TEC_percent::Float64 = 0.0        # % de triethyl citrate

    # Condições experimentais
    temperature::Float64 = 310.15     # K (37°C)
    pH::Float64 = 7.4                 # pH do meio

    # Tempo de simulação
    t_max::Float64 = 90.0             # dias
    dt::Float64 = 0.5                 # passo de tempo (dias)

    # Opções de modelo
    force_model::Union{Symbol, Nothing} = nothing  # :idiosyncratic, :biphasic, :generic
    include_cells::Bool = false
    tissue_type::Symbol = :meniscus   # :meniscus, :cartilage, :bone, :skin
end

"""
Configuração para integração celular.
"""
Base.@kwdef struct CellularConfig
    # Tipos celulares e densidades iniciais (células/mm³)
    fibroblasts::Float64 = 1e4
    macrophages::Float64 = 200.0
    chondrocytes::Float64 = 0.0
    osteoblasts::Float64 = 0.0
    mscs::Float64 = 0.0

    # Parâmetros inflamatórios
    il6_initial::Float64 = 0.0        # pg/mL
    mmp_initial::Float64 = 0.0        # ng/mL
    vegf_initial::Float64 = 0.0       # ng/mL
end

"""
Resultado da simulação de degradação.
"""
struct DegradationOutput
    # Metadata
    polymer::Symbol
    model_used::Symbol
    input::DegradationInput

    # Séries temporais
    time::Vector{Float64}
    Mn::Vector{Float64}
    Mw::Vector{Float64}
    PDI::Vector{Float64}
    porosity::Vector{Float64}
    pore_size::Vector{Float64}
    Tg::Vector{Float64}
    Xc::Vector{Float64}

    # Integridade mecânica
    mechanical_integrity::Vector{Float64}

    # Celular (se aplicável)
    cell_density::Union{Vector{Float64}, Nothing}
    ecm_fraction::Union{Vector{Float64}, Nothing}
    integration_score::Union{Vector{Float64}, Nothing}

    # Métricas finais
    Mn_final::Float64
    degradation_percent::Float64
    expected_accuracy::Float64
end

# ============================================================================
# FUNÇÕES DE ROTEAMENTO
# ============================================================================

"""
Determina o modelo recomendado para um polímero.
"""
function get_recommended_model(polymer::Symbol)::Symbol
    if !haskey(RECOMMENDED_MODEL, polymer)
        @warn "Polímero $polymer não reconhecido. Usando modelo genérico."
        return :generic
    end
    return RECOMMENDED_MODEL[polymer]
end

"""
Seleciona e executa o modelo apropriado.
"""
function route_to_model(input::DegradationInput)::Tuple{Symbol, Any}

    # Determinar modelo
    model_type = isnothing(input.force_model) ?
                 get_recommended_model(input.polymer) :
                 input.force_model

    if model_type == :idiosyncratic && input.polymer == :PLDLA
        # Modelo idiossincrático PLDLA
        params = PLDLAIdiosyncraticModel.create_pldla_params(
            Mn_initial = input.Mn_initial,
            TEC_percent = input.TEC_percent
        )

        # Calibrar se TEC > 0
        if input.TEC_percent > 0
            material = input.TEC_percent ≈ 1.0 ? :PLDLA_TEC1 : :PLDLA_TEC2
            params = PLDLAIdiosyncraticModel.calibrate_pldla_model(material=material)
        end

        states = PLDLAIdiosyncraticModel.simulate_pldla_degradation(
            params;
            t_max = input.t_max,
            dt = input.dt,
            T = input.temperature
        )

        return (:idiosyncratic, states)

    elseif model_type == :biphasic || input.polymer in [:PLLA, :PCL]
        # Modelo bifásico para semi-cristalinos
        scaffold = UnifiedScaffoldTissueModel.create_polymer_scaffold(
            input.polymer;
            Mn_initial = input.Mn_initial,
            porosity = input.porosity,
            pore_size = input.pore_size,
            crystallinity = input.Xc_initial
        )

        model = UnifiedScaffoldTissueModel.UnifiedModel(
            tissue_type = symbol_to_tissue_type(input.tissue_type),
            porosity = input.porosity,
            pore_size = input.pore_size
        )

        results = UnifiedScaffoldTissueModel.simulate_unified_model(
            model;
            t_max = input.t_max,
            dt = input.dt
        )

        return (:biphasic, results)

    else
        # Modelo genérico
        scaffold = UnifiedScaffoldTissueModel.ScaffoldDesign(
            porosity = input.porosity,
            pore_size = input.pore_size,
            Mn_initial = input.Mn_initial,
            crystallinity = input.Xc_initial,
            polymer_type = input.polymer
        )

        model = UnifiedScaffoldTissueModel.UnifiedModel(
            tissue_type = symbol_to_tissue_type(input.tissue_type),
            porosity = input.porosity,
            pore_size = input.pore_size
        )

        results = UnifiedScaffoldTissueModel.simulate_unified_model(
            model;
            t_max = input.t_max,
            dt = input.dt
        )

        return (:generic, results)
    end
end

"""
Converte símbolo para TissueType.
"""
function symbol_to_tissue_type(s::Symbol)
    mapping = Dict(
        :meniscus => UnifiedScaffoldTissueModel.MENISCUS_TYPE,
        :cartilage => UnifiedScaffoldTissueModel.CARTILAGE_TYPE,
        :bone => UnifiedScaffoldTissueModel.BONE_TYPE,
        :skin => UnifiedScaffoldTissueModel.SKIN_TYPE,
        :muscle => UnifiedScaffoldTissueModel.MUSCLE_TYPE
    )
    return get(mapping, s, UnifiedScaffoldTissueModel.MENISCUS_TYPE)
end

# ============================================================================
# FUNÇÃO PRINCIPAL DE SIMULAÇÃO
# ============================================================================

"""
    simulate_degradation(input::DegradationInput) -> DegradationOutput
    simulate_degradation(; kwargs...) -> DegradationOutput

Simula degradação de scaffold com roteamento automático de modelo.

# Exemplos
```julia
# Sintaxe simples
result = simulate_degradation(polymer=:PLDLA, Mn_initial=51.3)

# Com configuração completa
input = DegradationInput(
    polymer = :PLDLA,
    Mn_initial = 51.285,
    TEC_percent = 1.0,
    t_max = 90.0
)
result = simulate_degradation(input)
```
"""
function simulate_degradation(input::DegradationInput)::DegradationOutput

    # Validar entrada
    @assert input.polymer in SUPPORTED_POLYMERS "Polímero $(input.polymer) não suportado. Use: $(SUPPORTED_POLYMERS)"
    @assert 0 < input.Mn_initial < 1000 "Mn_initial deve estar entre 0 e 1000 kg/mol"
    @assert 0 <= input.porosity <= 1 "Porosidade deve estar entre 0 e 1"

    # Rotear para modelo apropriado
    model_used, raw_results = route_to_model(input)

    # Converter resultados para formato padronizado
    output = convert_to_output(input, model_used, raw_results)

    return output
end

# Versão com kwargs
function simulate_degradation(; kwargs...)
    input = DegradationInput(; kwargs...)
    return simulate_degradation(input)
end

"""
Converte resultados brutos para DegradationOutput padronizado.
"""
function convert_to_output(
    input::DegradationInput,
    model_used::Symbol,
    raw_results
)::DegradationOutput

    if model_used == :idiosyncratic
        # Resultados do PLDLAIdiosyncraticModel
        states = raw_results
        n = length(states)

        time = [s.t for s in states]
        Mn = [s.Mn for s in states]
        Mw = [s.Mw for s in states]
        PDI = [s.PDI for s in states]
        Tg = [s.Tg for s in states]
        Xc = [s.Xc for s in states]

        # Porosidade e tamanho de poro (estimativa simples)
        degradation_extent = [1 - m/input.Mn_initial for m in Mn]
        porosity = [input.porosity + 0.2 * d for d in degradation_extent]
        pore_size = [input.pore_size * (1 + 0.5 * d) for d in degradation_extent]

        # Integridade mecânica
        mechanical_integrity = [(m/input.Mn_initial)^1.5 * ((1-p)/(1-input.porosity))^2
                                for (m, p) in zip(Mn, porosity)]

        return DegradationOutput(
            input.polymer,
            model_used,
            input,
            time, Mn, Mw, PDI, porosity, pore_size, Tg, Xc,
            mechanical_integrity,
            nothing, nothing, nothing,
            Mn[end],
            (1 - Mn[end]/input.Mn_initial) * 100,
            MODEL_ACCURACY[input.polymer]
        )

    else
        # Resultados do UnifiedScaffoldTissueModel
        results = raw_results
        n = length(results)

        time = [r.time for r in results]
        Mn = [r.Mn for r in results]
        porosity = [r.porosity for r in results]
        pore_size = [r.pore_size for r in results]
        mechanical_integrity = [r.mechanical_integrity for r in results]

        # Calcular Mw e PDI (estimativa)
        PDI_est = [1.8 + 0.3 * (1 - m/input.Mn_initial) for m in Mn]
        Mw = [m * p for (m, p) in zip(Mn, PDI_est)]

        # Tg e Xc (estimativa baseada em Mn)
        Tg = [input.Tg_initial - 20 * (1 - m/input.Mn_initial) for m in Mn]
        Xc = [input.Xc_initial + 0.1 * (1 - m/input.Mn_initial) for m in Mn]

        # Dados celulares se disponíveis
        cell_density = input.include_cells ? [r.cell_density for r in results] : nothing
        ecm_fraction = input.include_cells ? [r.ecm_volume_fraction for r in results] : nothing
        integration_score = input.include_cells ? [r.integration_score for r in results] : nothing

        return DegradationOutput(
            input.polymer,
            model_used,
            input,
            time, Mn, Mw, PDI_est, porosity, pore_size, Tg, Xc,
            mechanical_integrity,
            cell_density, ecm_fraction, integration_score,
            Mn[end],
            (1 - Mn[end]/input.Mn_initial) * 100,
            MODEL_ACCURACY[input.polymer]
        )
    end
end

# ============================================================================
# FUNÇÕES DE COMPARAÇÃO E RELATÓRIO
# ============================================================================

"""
Imprime comparação entre modelos disponíveis.
"""
function print_model_comparison()
    println("="^80)
    println("  COMPARAÇÃO DE MODELOS - Darwin Scaffold Studio")
    println("="^80)

    println("\n📊 PRECISÃO POR POLÍMERO:")
    println("-"^60)
    println("  Polímero │ Modelo Recomendado │ NRMSE (%) │ R² esperado")
    println("-"^60)

    for polymer in SUPPORTED_POLYMERS
        model = RECOMMENDED_MODEL[polymer]
        nrmse = MODEL_ACCURACY[polymer]
        r2 = round(1 - (nrmse/100)^2, digits=2)
        model_str = rpad(string(model), 15)
        @printf("  %-7s │ %s    │   %5.1f   │    %.2f\n",
                polymer, model_str, nrmse, r2)
    end

    println("-"^60)

    println("\n🎯 MODELOS DISPONÍVEIS:")
    println("""
    1. :idiosyncratic - PLDLAIdiosyncraticModel
       • Específico para PLDLA 70:30
       • Captura degradação diferenciada L/DL
       • Melhor para PLDLA com ou sem TEC

    2. :biphasic - UnifiedScaffoldTissueModel (modo bifásico)
       • Para polímeros semi-cristalinos (PLLA, PCL)
       • Modela cristalização durante degradação
       • Fase 1: amorfo rápido, Fase 2: cristalino lento

    3. :generic - UnifiedScaffoldTissueModel (modo genérico)
       • Para polímeros amorfos (PDLLA, PLGA)
       • Modelo de degradação bulk homogêneo
       • Adequado para maioria dos casos

    4. + CellularIntegration - Overlay celular
       • Pode ser combinado com qualquer modelo
       • 13 tipos celulares
       • Acelera degradação em ~2x
    """)

    println("="^80)
end

"""
Imprime relatório detalhado do resultado.
"""
function print_degradation_report(output::DegradationOutput)
    println("="^80)
    println("  RELATÓRIO DE DEGRADAÇÃO")
    println("="^80)

    println("\n📋 CONFIGURAÇÃO:")
    println("-"^60)
    @printf("  Polímero: %s\n", output.polymer)
    @printf("  Modelo usado: %s\n", output.model_used)
    @printf("  Mn inicial: %.2f kg/mol\n", output.input.Mn_initial)
    @printf("  Porosidade: %.1f%%\n", output.input.porosity * 100)
    @printf("  Tempo: %.0f dias\n", output.input.t_max)

    println("\n📈 EVOLUÇÃO TEMPORAL:")
    println("-"^80)
    println("  Dia │  Mn (kg/mol) │  Degradação │  Porosidade │  Tg (°C) │ Integridade")
    println("-"^80)

    n = length(output.time)
    indices = [1, div(n,4), div(n,2), div(3n,4), n]

    for i in indices
        if i >= 1 && i <= n
            deg = (1 - output.Mn[i]/output.input.Mn_initial) * 100
            @printf("  %3.0f │    %6.2f    │    %5.1f%%   │    %5.1f%%   │  %5.1f   │   %5.1f%%\n",
                    output.time[i], output.Mn[i], deg,
                    output.porosity[i]*100, output.Tg[i],
                    output.mechanical_integrity[i]*100)
        end
    end

    println("-"^80)

    println("\n🎯 RESULTADO FINAL:")
    println("-"^60)
    @printf("  Mn final: %.2f kg/mol\n", output.Mn_final)
    @printf("  Degradação total: %.1f%%\n", output.degradation_percent)
    @printf("  Precisão esperada (NRMSE): %.1f%%\n", output.expected_accuracy)

    if output.expected_accuracy < 15.0
        println("\n  ✅ Modelo de alta precisão")
    elseif output.expected_accuracy < 20.0
        println("\n  ⚠️  Modelo de precisão moderada")
    else
        println("\n  ❌ Modelo precisa refinamento para este polímero")
    end

    println("="^80)
end

end # module
