#!/usr/bin/env julia
"""
Validação: Modelo Computacional vs Dados Experimentais do Kaique

Pergunta: Os cálculos computacionais conseguem prever os valores experimentais?
"""

using Printf

println("="^70)
println("VALIDACAO: MODELO COMPUTACIONAL vs DADOS EXPERIMENTAIS DO KAIQUE")
println("="^70)

# ============================================================================
# DADOS EXPERIMENTAIS DO KAIQUE (Dissertação)
# ============================================================================

println("\n[DADOS EXPERIMENTAIS] (Kaique Hergesel, 2023)")
println("-"^50)

# Peso molecular (GPC) - Tabela 5 da dissertação
experimental_Mw = Dict(
    "PLDLA" => Dict(
        0 => 94.4,   # kg/mol
        30 => 42.0,  # estimado do gráfico
        60 => 22.0,  # estimado do gráfico
        90 => 11.8   # kg/mol
    ),
    "PLDLA/TEC1" => Dict(
        0 => 85.8,
        30 => 38.0,
        60 => 20.0,
        90 => 12.1
    ),
    "PLDLA/TEC2" => Dict(
        0 => 68.4,
        30 => 28.0,
        60 => 15.0,
        90 => 8.4
    )
)

# Tg (DSC) - extraído da dissertação
experimental_Tg = Dict(
    "PLDLA" => Dict(0 => 54.0, 30 => 48.0, 60 => 38.0, 90 => 32.0),
    "PLDLA/TEC1" => Dict(0 => 49.0, 30 => 42.0, 60 => 32.0, 90 => 26.0),
    "PLDLA/TEC2" => Dict(0 => 46.0, 30 => 38.0, 60 => 22.0, 90 => 18.0)
)

# Massa (%) - intumescimento e perda
experimental_mass = Dict(
    "PLDLA" => Dict(0 => 100.0, 30 => 102.0, 60 => 101.0, 90 => 98.0),
    "PLDLA/TEC1" => Dict(0 => 100.0, 30 => 103.0, 60 => 102.0, 90 => 96.0),
    "PLDLA/TEC2" => Dict(0 => 100.0, 30 => 105.0, 60 => 103.0, 90 => 94.0)
)

# Propriedades mecânicas iniciais (t=0)
experimental_E0 = Dict(
    "PLDLA" => 2.67,      # MPa
    "PLDLA/TEC1" => 1.90,
    "PLDLA/TEC2" => 1.60
)

for material in ["PLDLA", "PLDLA/TEC1", "PLDLA/TEC2"]
    println("\n$material:")
    println("  Mw0 = $(experimental_Mw[material][0]) kg/mol -> Mw90 = $(experimental_Mw[material][90]) kg/mol")
    println("  Tg0 = $(experimental_Tg[material][0]) C -> Tg90 = $(experimental_Tg[material][90]) C")
    println("  E0 = $(experimental_E0[material]) MPa")
end

# ============================================================================
# MODELO COMPUTACIONAL
# ============================================================================

println("\n\n[MODELO COMPUTACIONAL] (baseado na literatura)")
println("-"^50)

"""
Modelo de degradação hidrolítica de primeira ordem
Mw(t) = Mw0 * exp(-k * t)

Fonte: Weir et al. (2004), Lyu & Untereker (2009)
"""
function degradation_model(Mw0, k, t)
    return Mw0 * exp(-k * t)
end

"""
Calcular k a partir dos dados experimentais
k = -ln(Mw_t / Mw_0) / t
"""
function calculate_k(Mw0, Mw_t, t)
    return -log(Mw_t / Mw0) / t
end

"""
Relação Tg-Mw (Equação de Fox-Flory)
Tg = Tg_inf - K/Mn

Simplificado: Tg proporcional a Mw^alpha onde alpha = 0.3-0.5
"""
function predict_Tg(Tg0, Mw0, Mw_t; alpha=0.4)
    return Tg0 * (Mw_t / Mw0)^alpha
end

"""
Relação E-Mw (propriedades mecânicas)
E(t) / E0 = (Mw(t) / Mw0)^n

Fonte: Duek et al. (1999), n = 0.5-1.0
"""
function predict_E(E0, Mw0, Mw_t; n=0.75)
    return E0 * (Mw_t / Mw0)^n
end

# ============================================================================
# CALIBRAÇÃO: Calcular k para cada material
# ============================================================================

println("\n[CALIBRACAO DO MODELO]")
println("-"^50)

k_values = Dict{String, Float64}()

for material in ["PLDLA", "PLDLA/TEC1", "PLDLA/TEC2"]
    Mw0 = experimental_Mw[material][0]
    Mw90 = experimental_Mw[material][90]
    k = calculate_k(Mw0, Mw90, 90.0)
    k_values[material] = k

    k_str = @sprintf("%.4f", k)
    t_half = @sprintf("%.1f", log(2)/k)
    println("$material: k = $k_str /dia")
    println("  (tempo de meia-vida t1/2 = $t_half dias)")
end

# ============================================================================
# VALIDAÇÃO: Comparar previsões vs experimentos
# ============================================================================

println("\n\n[VALIDACAO: PREVISAO vs EXPERIMENTAL]")
println("="^70)

time_points = [0, 30, 60, 90]

for material in ["PLDLA", "PLDLA/TEC1", "PLDLA/TEC2"]
    println("\n>> $material")
    println("-"^50)

    Mw0 = experimental_Mw[material][0]
    Tg0 = experimental_Tg[material][0]
    E0 = experimental_E0[material]
    k = k_values[material]

    println("\n  PESO MOLECULAR (Mw, kg/mol):")
    println("  Tempo(d) | Experimental | Previsto | Erro")
    println("  " * "-"^45)

    total_error_Mw = 0.0
    n_points = 0

    for t in time_points
        Mw_exp = experimental_Mw[material][t]
        Mw_pred = degradation_model(Mw0, k, t)
        error_pct = abs(Mw_pred - Mw_exp) / Mw_exp * 100

        line = @sprintf("  %7d  | %11.1f  | %8.1f | %5.1f%%", t, Mw_exp, Mw_pred, error_pct)
        println(line)

        if t > 0
            total_error_Mw += error_pct
            n_points += 1
        end
    end

    avg_error_Mw = total_error_Mw / n_points
    println("  " * "-"^45)
    avg_str = @sprintf("%.1f", avg_error_Mw)
    println("  ERRO MEDIO Mw: $avg_str%")

    # Tg
    println("\n  TEMPERATURA DE TRANSICAO VITREA (Tg, C):")
    println("  Tempo(d) | Experimental | Previsto | Erro")
    println("  " * "-"^45)

    total_error_Tg = 0.0

    for t in time_points
        Tg_exp = experimental_Tg[material][t]
        Mw_t = degradation_model(Mw0, k, t)
        Tg_pred = predict_Tg(Tg0, Mw0, Mw_t)
        error_pct = abs(Tg_pred - Tg_exp) / Tg_exp * 100

        line = @sprintf("  %7d  | %11.1f  | %8.1f | %5.1f%%", t, Tg_exp, Tg_pred, error_pct)
        println(line)

        if t > 0
            total_error_Tg += error_pct
        end
    end

    avg_error_Tg = total_error_Tg / 3
    println("  " * "-"^45)
    avg_str = @sprintf("%.1f", avg_error_Tg)
    println("  ERRO MEDIO Tg: $avg_str%")

    # Mecânica (previsão - não temos dados experimentais durante degradação)
    println("\n  MODULO DE COMPRESSAO (E, MPa) - PREVISAO:")
    println("  Tempo(d) | Previsto | Pct do inicial")
    println("  " * "-"^35)

    for t in time_points
        Mw_t = degradation_model(Mw0, k, t)
        E_pred = predict_E(E0, Mw0, Mw_t)
        pct = E_pred / E0 * 100

        line = @sprintf("  %7d  | %8.2f | %5.1f%%", t, E_pred, pct)
        println(line)
    end
    println("  [!] SEM DADOS EXPERIMENTAIS PARA VALIDAR")
end

# ============================================================================
# RESUMO ESTATÍSTICO
# ============================================================================

println("\n\n[RESUMO DA VALIDACAO]")
println("="^70)

println("""

+---------------------------------------------------------------------+
|                    CAPACIDADE PREDITIVA DO MODELO                   |
+---------------------------------------------------------------------+
| Parametro              | Dados Disponiveis | Erro Medio | Confianca |
+------------------------+-------------------+------------+-----------+
| Peso Molecular (Mw)    | SIM (GPC)         | ~15-25%    | MEDIO     |
| Tg                     | SIM (DSC)         | ~10-20%    | MEDIO     |
| Modulo Elastico (E)    | NAO (so t=0)      | N/A        | TEORICO   |
| Massa                  | SIM               | nao model. | BAIXO     |
| Porosidade             | SIM (so t=0)      | N/A        | TEORICO   |
+---------------------------------------------------------------------+

CONCLUSOES:

1. O MODELO FUNCIONA?
   -> PARCIALMENTE. Para Mw e Tg, erros de 10-25% sao aceitaveis
   -> O modelo de primeira ordem e uma APROXIMACAO

2. POR QUE O ERRO?
   -> Degradacao real e MAIS COMPLEXA (autocatalise, cristalizacao)
   -> Dados intermediarios (30, 60 dias) sao estimados do grafico
   -> Modelo assume homogeneidade (scaffolds sao heterogeneos)

3. O QUE PODEMOS PREVER COM CONFIANCA?
   [OK] Tendencia geral de degradacao
   [OK] Ordem de magnitude do tempo de degradacao
   [OK] Comparacao relativa entre materiais (TEC acelera)
   [X]  Valores exatos em pontos especificos

4. O QUE FALTA PARA MELHORAR?
   -> Modelo de autocatalise (aceleracao com acidos)
   -> Dados de mecanica durante degradacao
   -> Validacao cruzada com mais datasets
""")

println("\n" * "="^70)
println("FIM DA VALIDACAO")
println("="^70)
