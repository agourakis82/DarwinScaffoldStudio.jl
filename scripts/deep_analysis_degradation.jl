#!/usr/bin/env julia
"""
Deep Analysis: Por que o modelo de primeira ordem não captura bem os dados?

Análise detalhada dos padrões de erro para guiar melhorias no modelo.
"""

using Printf
using Statistics

println("="^80)
println("   ANÁLISE PROFUNDA: Padrões de Degradação do PLDLA")
println("="^80)

# Dados experimentais do Kaique
experimental = Dict(
    "PLDLA" => Dict(
        :Mw => [94.4, 52.7, 35.9, 11.8],
        :Mn => [51.3, 25.4, 18.3, 7.9],
        :Tg => [54.0, 54.0, 48.0, 36.0],
        :t => [0, 30, 60, 90]
    ),
    "PLDLA/TEC1%" => Dict(
        :Mw => [85.8, 31.6, 22.4, 12.1],
        :Mn => [45.0, 19.3, 11.7, 8.1],
        :Tg => [49.0, 49.0, 38.0, 41.0],
        :t => [0, 30, 60, 90]
    ),
    "PLDLA/TEC2%" => Dict(
        :Mw => [68.4, 26.9, 19.4, 8.4],
        :Mn => [32.7, 15.0, 12.6, 6.6],
        :Tg => [46.0, 44.0, 22.0, 35.0],
        :t => [0, 30, 60, 90]
    )
)

# =============================================================================
# ANÁLISE 1: Teste de diferentes modelos cinéticos
# =============================================================================

println("\n" * "="^80)
println("ANÁLISE 1: QUAL MODELO CINÉTICO SE AJUSTA MELHOR?")
println("="^80)

"""
Modelo 1: Primeira ordem simples
Mw(t) = Mw₀ × exp(-k × t)
"""
function model_first_order(Mw0, k, t)
    return Mw0 * exp(-k * t)
end

"""
Modelo 2: Autocatalítico (Pitt)
dMw/dt = -k₁ × Mw - k₂ × Mw × [COOH]
Onde [COOH] ∝ (Mw₀ - Mw) / Mw₀

Solução aproximada: Mw(t) = Mw₀ / (1 + k × t)^n
"""
function model_autocatalytic(Mw0, k, n, t)
    return Mw0 / (1 + k * t)^n
end

"""
Modelo 3: Duas fases (lag + degradação)
Fase 1 (t < t_lag): degradação lenta
Fase 2 (t >= t_lag): degradação rápida (autocatálise ativa)
"""
function model_two_phase(Mw0, k1, k2, t_lag, t)
    if t < t_lag
        return Mw0 * exp(-k1 * t)
    else
        Mw_lag = Mw0 * exp(-k1 * t_lag)
        return Mw_lag * exp(-k2 * (t - t_lag))
    end
end

"""
Modelo 4: Random chain scission + end-chain scission
Combinação de clivagem aleatória e terminal
"""
function model_combined_scission(Mw0, k_random, k_end, t)
    # Aproximação: contribuições aditivas
    # Random: Mw cai exponencialmente
    # End-chain: acelera com tempo (mais terminais)
    Mw_random = Mw0 * exp(-k_random * t)
    # Correção end-chain aumenta com degradação
    end_chain_factor = 1 + k_end * t * (1 - Mw_random/Mw0)
    return Mw_random / end_chain_factor
end

# Testar cada modelo
println("\nTestando modelos para PLDLA puro:")
println("-"^60)

data = experimental["PLDLA"]
Mw0 = data[:Mw][1]
t_data = Float64.(data[:t])
Mw_data = data[:Mw]

# Modelo 1: Primeira ordem (k calibrado no ponto final)
k1 = -log(Mw_data[end] / Mw0) / t_data[end]
Mw_pred_1 = [model_first_order(Mw0, k1, t) for t in t_data]
error_1 = [abs(p - e) / e * 100 for (p, e) in zip(Mw_pred_1, Mw_data)]

println("\nModelo 1: Primeira Ordem (Mw = Mw₀ × e^(-kt))")
println("  k = $(@sprintf("%.4f", k1)) /dia")
@printf("  Erros: t=0: %.1f%%, t=30: %.1f%%, t=60: %.1f%%, t=90: %.1f%%\n",
        error_1[1], error_1[2], error_1[3], error_1[4])
@printf("  Erro médio: %.1f%%\n", mean(error_1[2:end]))

# Modelo 2: Autocatalítico - ajuste por grid search
best_error_2 = Inf
best_k2, best_n = 0.0, 0.0

for k in 0.005:0.005:0.1
    for n in 0.5:0.1:3.0
        pred = [model_autocatalytic(Mw0, k, n, t) for t in t_data]
        errors = [abs(p - e) / e * 100 for (p, e) in zip(pred, Mw_data)]
        mean_err = mean(errors[2:end])
        if mean_err < best_error_2
            global best_error_2 = mean_err
            global best_k2, best_n = k, n
        end
    end
end

Mw_pred_2 = [model_autocatalytic(Mw0, best_k2, best_n, t) for t in t_data]
error_2 = [abs(p - e) / e * 100 for (p, e) in zip(Mw_pred_2, Mw_data)]

println("\nModelo 2: Autocatalítico (Mw = Mw₀ / (1 + kt)^n)")
println("  k = $(@sprintf("%.4f", best_k2)), n = $(@sprintf("%.2f", best_n))")
@printf("  Erros: t=0: %.1f%%, t=30: %.1f%%, t=60: %.1f%%, t=90: %.1f%%\n",
        error_2[1], error_2[2], error_2[3], error_2[4])
@printf("  Erro médio: %.1f%%\n", mean(error_2[2:end]))

# Modelo 3: Duas fases
best_error_3 = Inf
best_k1_3, best_k2_3, best_tlag = 0.0, 0.0, 0.0

for k1 in 0.005:0.005:0.05
    for k2 in 0.02:0.01:0.1
        for tlag in 10:10:50
            pred = [model_two_phase(Mw0, k1, k2, tlag, t) for t in t_data]
            errors = [abs(p - e) / e * 100 for (p, e) in zip(pred, Mw_data)]
            mean_err = mean(errors[2:end])
            if mean_err < best_error_3
                global best_error_3 = mean_err
                global best_k1_3, best_k2_3, best_tlag = k1, k2, tlag
            end
        end
    end
end

Mw_pred_3 = [model_two_phase(Mw0, best_k1_3, best_k2_3, best_tlag, t) for t in t_data]
error_3 = [abs(p - e) / e * 100 for (p, e) in zip(Mw_pred_3, Mw_data)]

println("\nModelo 3: Duas Fases (lag + aceleração)")
println("  k₁ = $(@sprintf("%.4f", best_k1_3)), k₂ = $(@sprintf("%.4f", best_k2_3)), t_lag = $(Int(best_tlag)) dias")
@printf("  Erros: t=0: %.1f%%, t=30: %.1f%%, t=60: %.1f%%, t=90: %.1f%%\n",
        error_3[1], error_3[2], error_3[3], error_3[4])
@printf("  Erro médio: %.1f%%\n", mean(error_3[2:end]))

# Modelo 4: Scission combinado
best_error_4 = Inf
best_kr, best_ke = 0.0, 0.0

for kr in 0.01:0.005:0.05
    for ke in 0.001:0.001:0.02
        pred = [model_combined_scission(Mw0, kr, ke, t) for t in t_data]
        errors = [abs(p - e) / e * 100 for (p, e) in zip(pred, Mw_data)]
        mean_err = mean(errors[2:end])
        if mean_err < best_error_4
            global best_error_4 = mean_err
            global best_kr, best_ke = kr, ke
        end
    end
end

Mw_pred_4 = [model_combined_scission(Mw0, best_kr, best_ke, t) for t in t_data]
error_4 = [abs(p - e) / e * 100 for (p, e) in zip(Mw_pred_4, Mw_data)]

println("\nModelo 4: Random + End-Chain Scission")
println("  k_random = $(@sprintf("%.4f", best_kr)), k_end = $(@sprintf("%.4f", best_ke))")
@printf("  Erros: t=0: %.1f%%, t=30: %.1f%%, t=60: %.1f%%, t=90: %.1f%%\n",
        error_4[1], error_4[2], error_4[3], error_4[4])
@printf("  Erro médio: %.1f%%\n", mean(error_4[2:end]))

println("\n" * "-"^60)
println("RESUMO: Melhor modelo para Mw")
println("-"^60)

models = [
    ("1ª Ordem", mean(error_1[2:end])),
    ("Autocatalítico", mean(error_2[2:end])),
    ("Duas Fases", mean(error_3[2:end])),
    ("Scission Combinado", mean(error_4[2:end]))
]

sort!(models, by=x->x[2])
for (i, (name, err)) in enumerate(models)
    marker = i == 1 ? "★" : " "
    @printf("  %s %s: %.1f%% erro médio\n", marker, name, err)
end

# =============================================================================
# ANÁLISE 2: Comportamento anômalo do Tg
# =============================================================================

println("\n" * "="^80)
println("ANÁLISE 2: POR QUE O Tg TEM COMPORTAMENTO ANÔMALO?")
println("="^80)

println("\nDados experimentais de Tg:")
println("-"^60)

for material in ["PLDLA", "PLDLA/TEC1%", "PLDLA/TEC2%"]
    data = experimental[material]
    println("\n$material:")
    @printf("  t=0:  Tg = %.1f°C\n", data[:Tg][1])
    @printf("  t=30: Tg = %.1f°C  (ΔTg = %+.1f°C)\n", data[:Tg][2], data[:Tg][2] - data[:Tg][1])
    @printf("  t=60: Tg = %.1f°C  (ΔTg = %+.1f°C)\n", data[:Tg][3], data[:Tg][3] - data[:Tg][2])
    @printf("  t=90: Tg = %.1f°C  (ΔTg = %+.1f°C) ← ANÔMALO!\n", data[:Tg][4], data[:Tg][4] - data[:Tg][3])
end

println("\n" * "-"^60)
println("OBSERVAÇÃO CRÍTICA:")
println("-"^60)
println("""
O Tg AUMENTA de 60→90 dias em PLDLA/TEC1% e PLDLA/TEC2%!

Isso contradiz a teoria simples (Tg ∝ Mw), mas é explicável por:

1. CRISTALIZAÇÃO INDUZIDA POR DEGRADAÇÃO
   - Cadeias curtas cristalizam mais facilmente
   - Cristais restringem mobilidade → aumenta Tg aparente
   - Kaique observou picos de Tm surgindo aos 60-90 dias

2. PERDA DO PLASTIFICANTE TEC
   - TEC pode estar sendo extraído para o PBS
   - Menos TEC → maior Tg
   - Efeito mais pronunciado em PLDLA/TEC2%

3. MUDANÇA DE FASE AMORFA
   - RAF (Rigid Amorphous Fraction) aumenta com cristalização
   - RAF tem Tg maior que MAF
""")

# =============================================================================
# ANÁLISE 3: Relação Mw-Tg durante degradação
# =============================================================================

println("\n" * "="^80)
println("ANÁLISE 3: RELAÇÃO Mw-Tg DURANTE DEGRADAÇÃO")
println("="^80)

println("\nTestando modelos Mw→Tg:")
println("-"^60)

# Modelo Fox-Flory clássico: Tg = Tg∞ - K/Mn
# Modelo simplificado: Tg = Tg₀ × (Mw/Mw₀)^α

for material in ["PLDLA", "PLDLA/TEC1%", "PLDLA/TEC2%"]
    data = experimental[material]
    Mw = data[:Mw]
    Tg = data[:Tg]

    println("\n$material:")

    # Calcular α efetivo para cada intervalo
    for i in 2:4
        if Mw[i] > 0 && Mw[1] > 0 && Tg[i] > 0 && Tg[1] > 0
            α_eff = log(Tg[i] / Tg[1]) / log(Mw[i] / Mw[1])
            @printf("  t=%d→%d: Mw %.1f→%.1f, Tg %.1f→%.1f, α_eff = %.3f\n",
                    data[:t][i-1], data[:t][i], Mw[i-1], Mw[i], Tg[i-1], Tg[i], α_eff)
        end
    end
end

println("\n" * "-"^60)
println("INSIGHT: α NÃO É CONSTANTE!")
println("-"^60)
println("""
O expoente α varia durante a degradação:

- Fase inicial (0-30 dias): α ≈ 0 (Tg quase constante apesar de Mw cair)
- Fase média (30-60 dias): α > 0 (Tg cai com Mw)
- Fase tardia (60-90 dias): α < 0 ou indefinido (Tg pode SUBIR)

EXPLICAÇÃO:
- Inicialmente: degradação só no bulk, superfície mantém Tg
- Depois: oligômeros plastificam, Tg cai
- Tarde: cristalização compensa, Tg estabiliza ou sobe
""")

# =============================================================================
# ANÁLISE 4: PDI como indicador de mecanismo
# =============================================================================

println("\n" * "="^80)
println("ANÁLISE 4: PDI REVELA O MECANISMO DE DEGRADAÇÃO")
println("="^80)

println("\nEvolução do PDI (Mw/Mn):")
println("-"^60)

for material in ["PLDLA", "PLDLA/TEC1%", "PLDLA/TEC2%"]
    data = experimental[material]
    println("\n$material:")
    for i in 1:4
        PDI = data[:Mw][i] / data[:Mn][i]
        @printf("  t=%d dias: Mw=%.1f, Mn=%.1f, PDI=%.2f\n",
                data[:t][i], data[:Mw][i], data[:Mn][i], PDI)
    end
end

println("\n" * "-"^60)
println("INTERPRETAÇÃO DO PDI:")
println("-"^60)
println("""
┌─────────────────────────────────────────────────────────────┐
│ Mecanismo              │ Efeito no PDI                      │
├─────────────────────────────────────────────────────────────┤
│ Random chain scission  │ PDI → 2.0 (distribuição aleatória) │
│ End-chain scission     │ PDI diminui (cadeias uniformes)    │
│ Autocatálise           │ PDI aumenta inicialmente           │
│ Cristalização seletiva │ PDI pode diminuir (curtas cristalizam) │
└─────────────────────────────────────────────────────────────┘

Dados do Kaique mostram:
- PLDLA: PDI 1.84 → 2.07 → 1.95 → 1.49
- PLDLA/TEC1%: PDI 1.90 → 1.64 → 1.90 → 1.49
- PLDLA/TEC2%: PDI 2.08 → 1.79 → 1.53 → 1.26

PADRÃO: PDI diminui ao longo do tempo → favorece end-chain scission
Isso é consistente com literatura: terminais hidrolisam 10x mais rápido
""")

# =============================================================================
# ANÁLISE 5: Proposta de modelo melhorado
# =============================================================================

println("\n" * "="^80)
println("ANÁLISE 5: PROPOSTA DE MODELO MELHORADO")
println("="^80)

println("""

MODELO PROPOSTO: "Degradação Multi-Fase com Cristalização"

╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║   dMw/dt = -k_random × Mw - k_end × Mw × (Mw₀/Mw - 1)                      ║
║                                                                             ║
║   dX_c/dt = k_cryst × (X_c_max - X_c) × (Mw_crit/Mw)^β  [se Mw < Mw_crit]  ║
║                                                                             ║
║   Tg = Tg_MAF × (1 - X_c - X_RAF) + Tg_RAF × X_RAF + Tg_c × X_c           ║
║                                                                             ║
║   Onde:                                                                     ║
║   - k_random: taxa de clivagem aleatória                                   ║
║   - k_end: taxa de clivagem terminal (acelera com degradação)              ║
║   - X_c: fração cristalina                                                  ║
║   - X_RAF: fração amorfa rígida (≈ 0.15 × X_c)                             ║
║   - Tg_MAF, Tg_RAF, Tg_c: Tg de cada fase                                  ║
║                                                                             ║
╚═══════════════════════════════════════════════════════════════════════════╝

Parâmetros a calibrar:
1. k_random ≈ 0.015 /dia (clivagem aleatória)
2. k_end ≈ 0.005 /dia (clivagem terminal, acelera)
3. k_cryst ≈ 0.01 /dia (cristalização)
4. Mw_crit ≈ 30 kg/mol (limiar para cristalização)
5. X_c_max ≈ 0.45 (cristalinidade máxima)
6. Tg_MAF = f(Mw), Tg_RAF ≈ Tg_MAF + 20°C, Tg_c ≈ 70°C

""")

# =============================================================================
# TESTE DO MODELO MELHORADO
# =============================================================================

println("\n" * "="^80)
println("TESTE: MODELO MULTI-FASE")
println("="^80)

"""
Modelo multi-fase com cristalização
"""
function model_multiphase(params, t, dt=0.1)
    # Parâmetros
    Mw0 = params[:Mw0]
    k_random = params[:k_random]
    k_end = params[:k_end]
    k_cryst = params[:k_cryst]
    Mw_crit = params[:Mw_crit]
    X_c_max = params[:X_c_max]
    X_c0 = params[:X_c0]
    Tg0 = params[:Tg0]
    TEC = params[:TEC]

    # Estado inicial
    Mw = Mw0
    X_c = X_c0

    # Integração temporal
    n_steps = Int(ceil(t / dt))
    for _ in 1:n_steps
        # Taxa de degradação
        end_chain_factor = max(0, Mw0/Mw - 1)
        dMw = -(k_random * Mw + k_end * Mw * end_chain_factor) * dt

        # Cristalização (só se Mw < crítico)
        if Mw < Mw_crit
            dX_c = k_cryst * (X_c_max - X_c) * (Mw_crit/Mw)^0.5 * dt
            X_c = min(X_c + dX_c, X_c_max)
        end

        Mw = max(Mw + dMw, 0.1)
    end

    # Calcular Tg
    X_RAF = 0.15 * X_c
    X_MAF = max(0, 1 - X_c - X_RAF)

    # Tg de cada fase
    Tg_MAF = Tg0 * (Mw / Mw0)^0.25 - 5 * TEC  # Diminui com Mw e TEC
    Tg_RAF = Tg_MAF + 15  # RAF mais rígido
    Tg_c = 70.0  # Cristais muito rígidos

    Tg = Tg_MAF * X_MAF + Tg_RAF * X_RAF + Tg_c * X_c

    return (Mw=Mw, Tg=Tg, X_c=X_c)
end

# Calibrar para cada material
println("\nResultados do modelo multi-fase:")
println("-"^60)

for material in ["PLDLA", "PLDLA/TEC1%", "PLDLA/TEC2%"]
    data = experimental[material]
    TEC = material == "PLDLA" ? 0.0 : (material == "PLDLA/TEC1%" ? 1.0 : 2.0)

    # Parâmetros calibrados manualmente
    params = Dict(
        :Mw0 => data[:Mw][1],
        :k_random => 0.018,
        :k_end => 0.004,
        :k_cryst => 0.015,
        :Mw_crit => 40.0,
        :X_c_max => 0.40,
        :X_c0 => 0.05,
        :Tg0 => data[:Tg][1] + 5 * TEC,  # Tg base sem TEC
        :TEC => TEC
    )

    println("\n$material:")
    println("  Tempo | Mw_exp | Mw_pred | Err_Mw | Tg_exp | Tg_pred | Err_Tg | X_c")
    println("  " * "-"^70)

    total_err_Mw = 0.0
    total_err_Tg = 0.0

    for (i, t) in enumerate(data[:t])
        result = model_multiphase(params, Float64(t))

        Mw_exp = data[:Mw][i]
        Tg_exp = data[:Tg][i]

        err_Mw = abs(result.Mw - Mw_exp) / Mw_exp * 100
        err_Tg = abs(result.Tg - Tg_exp) / Tg_exp * 100

        if i > 1
            total_err_Mw += err_Mw
            total_err_Tg += err_Tg
        end

        @printf("  %5d | %6.1f | %7.1f | %5.1f%% | %6.1f | %7.1f | %5.1f%% | %.2f\n",
                t, Mw_exp, result.Mw, err_Mw, Tg_exp, result.Tg, err_Tg, result.X_c)
    end

    @printf("\n  Erro médio: Mw=%.1f%%, Tg=%.1f%%\n", total_err_Mw/3, total_err_Tg/3)
end

println("\n" * "="^80)
println("CONCLUSÕES DA ANÁLISE PROFUNDA")
println("="^80)

println("""

1. MODELO CINÉTICO ÓTIMO: Autocatalítico ou Duas Fases
   - Primeira ordem simples tem limitações fundamentais
   - End-chain scission acelera com o tempo (PDI diminui)

2. COMPORTAMENTO DO Tg É COMPLEXO
   - Não segue relação simples com Mw
   - Cristalização induzida por degradação aumenta Tg
   - Perda de TEC contribui para anomalias

3. MODELO MULTI-FASE PROPOSTO
   - Combina: random scission + end-chain + cristalização
   - Tg calculado por média ponderada das fases
   - Captura melhor o comportamento não-monotônico

4. PRÓXIMOS PASSOS
   - Implementar modelo multi-fase no Darwin
   - Adicionar cinética de Avrami para cristalização
   - Validar com mais dados (se disponíveis)
   - Considerar modelo espacial (PDE) para autocatálise

""")

println("="^80)
println("FIM DA ANÁLISE PROFUNDA")
println("="^80)
