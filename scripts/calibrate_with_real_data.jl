#!/usr/bin/env julia
"""
Calibração do modelo com dados REAIS de GPC da tese do Kaique.

DADOS EXPERIMENTAIS EXTRAÍDOS (Mn em g/mol):
============================================

PLDLA puro:
  0 dias:  51285 g/mol = 51.3 kg/mol
  30 dias: 25447 g/mol = 25.4 kg/mol  (-50.4%)
  60 dias: 18313 g/mol = 18.3 kg/mol  (-64.3%)
  90 dias:  7904 g/mol =  7.9 kg/mol  (-84.6%)

PLDLA/TEC1%:
  0 dias:  44998 g/mol = 45.0 kg/mol
  30 dias: 19257 g/mol = 19.3 kg/mol  (-57.2%)
  60 dias: 11749 g/mol = 11.7 kg/mol  (-73.9%)
  90 dias:  8122 g/mol =  8.1 kg/mol  (-81.9%)

PLDLA/TEC2%:
  0 dias:  32733 g/mol = 32.7 kg/mol
  (mais dados a extrair...)

CONCLUSÃO: Degradação MUITO mais rápida que o modelo atual!
- 50% de perda em 30 dias (modelo previa ~10%)
- 85% de perda em 90 dias (modelo previa ~30%)
"""

using Pkg
Pkg.activate(".")

using Statistics
using Printf

println("="^80)
println("  CALIBRAÇÃO COM DADOS REAIS - TESE DO KAIQUE")
println("  Dados de GPC: Mn vs tempo para PLDLA 70:30")
println("="^80)

# ============================================================================
# DADOS EXPERIMENTAIS REAIS
# ============================================================================

# PLDLA puro (scaffold 3D-printed)
experimental_pldla = [
    (0.0,  51.285),   # Mn em kg/mol
    (30.0, 25.447),
    (60.0, 18.313),
    (90.0,  7.904),
]

# PLDLA/TEC1%
experimental_tec1 = [
    (0.0,  44.998),
    (30.0, 19.257),
    (60.0, 11.749),
    (90.0,  8.122),
]

# PLDLA/TEC2%
experimental_tec2 = [
    (0.0,  32.733),
    # Mais dados se disponíveis
]

println("\n📊 DADOS EXPERIMENTAIS (GPC):")
println("-"^50)
println("Tempo (dias)  │  PLDLA   │  PLDLA/TEC1%  │  Redução")
println("-"^50)
for i in 1:length(experimental_pldla)
    t, mn_pldla = experimental_pldla[i]
    mn_tec1 = i <= length(experimental_tec1) ? experimental_tec1[i][2] : NaN
    reduction = (1 - mn_pldla/experimental_pldla[1][2]) * 100
    @printf("    %3.0f       │  %5.1f   │     %5.1f     │  -%4.1f%%\n",
            t, mn_pldla, mn_tec1, reduction)
end

# ============================================================================
# MODELO CALIBRADO
# ============================================================================

"""
Modelo de degradação calibrado com dados reais.
Usa cinética de primeira ordem com autocatálise.

dMn/dt = -k * Mn * (1 + α * [COOH])

Onde [COOH] ∝ (Mn₀ - Mn) representa ácidos gerados.
"""
function mn_model_calibrated(t, Mn0; k=0.025, alpha=0.8, T=310.15)
    # Parâmetros calibrados para PLDLA 70:30
    # k: taxa base de degradação
    # alpha: fator de autocatálise

    # Efeito de temperatura (Arrhenius)
    Ea = 80.0  # kJ/mol
    R = 8.314e-3
    T_ref = 310.15
    k_T = k * exp(-Ea/R * (1/T - 1/T_ref))

    # Integração numérica simples (Euler)
    dt = 0.1
    Mn = Mn0

    for time in 0:dt:t
        # Concentração de ácidos (proporcional à degradação)
        acid_conc = (Mn0 - Mn) / Mn0

        # Taxa efetiva com autocatálise
        k_eff = k_T * (1 + alpha * acid_conc)

        # Atualização
        dMn = -k_eff * Mn * dt
        Mn = max(Mn + dMn, 1.0)  # Mn mínimo 1 kg/mol
    end

    return Mn
end

# ============================================================================
# OTIMIZAÇÃO DE PARÂMETROS
# ============================================================================

println("\n\n🔧 OTIMIZAÇÃO DE PARÂMETROS:")
println("-"^50)

function calculate_rmse(k, alpha)
    errors = Float64[]
    Mn0 = experimental_pldla[1][2]

    for (t, Mn_exp) in experimental_pldla[2:end]  # Excluir t=0
        Mn_pred = mn_model_calibrated(t, Mn0; k=k, alpha=alpha)
        push!(errors, (Mn_pred - Mn_exp)^2)
    end

    return sqrt(mean(errors))
end

# Grid search para encontrar melhores parâmetros
best_k, best_alpha, best_rmse = 0.0, 0.0, Inf

println("Buscando parâmetros ótimos...")
for k in 0.01:0.005:0.08
    for alpha in 0.0:0.2:3.0
        rmse = calculate_rmse(k, alpha)
        if rmse < best_rmse
            global best_k, best_alpha, best_rmse = k, alpha, rmse
        end
    end
end

@printf("\nParâmetros ótimos encontrados:\n")
@printf("  k     = %.4f /dia\n", best_k)
@printf("  α     = %.2f\n", best_alpha)
@printf("  RMSE  = %.2f kg/mol\n", best_rmse)

# ============================================================================
# VALIDAÇÃO
# ============================================================================

println("\n\n📈 VALIDAÇÃO DO MODELO CALIBRADO:")
println("-"^60)
println("Tempo │  Mn Exp  │  Mn Pred  │  Erro   │  Erro %")
println("-"^60)

Mn0 = experimental_pldla[1][2]
total_error = 0.0
n_points = 0

for (t, Mn_exp) in experimental_pldla
    Mn_pred = mn_model_calibrated(t, Mn0; k=best_k, alpha=best_alpha)
    erro = Mn_pred - Mn_exp
    erro_pct = abs(erro) / Mn_exp * 100
    global total_error += erro_pct
    global n_points += 1

    @printf(" %3.0f  │  %5.1f   │   %5.1f   │  %+5.1f  │  %4.1f%%\n",
            t, Mn_exp, Mn_pred, erro, erro_pct)
end

mean_error = total_error / n_points
accuracy = 100 - mean_error

println("-"^60)
@printf("\nErro médio: %.1f%%\n", mean_error)
@printf("ACURÁCIA: %.1f%%\n", accuracy)

# ============================================================================
# PREDIÇÃO DE DEGRADAÇÃO TOTAL
# ============================================================================

println("\n\n⏱️  PREDIÇÃO DE MARCOS CRÍTICOS:")
println("-"^50)

# Tempo para Mn < 5 kg/mol (perda de integridade)
for t in 1:200
    Mn = mn_model_calibrated(Float64(t), Mn0; k=best_k, alpha=best_alpha)
    if Mn < 5.0
        @printf("Mn < 5 kg/mol (perda integridade): %d dias\n", t)
        break
    end
end

# Tempo para Mn < 2 kg/mol (degradação avançada)
for t in 1:300
    Mn = mn_model_calibrated(Float64(t), Mn0; k=best_k, alpha=best_alpha)
    if Mn < 2.0
        @printf("Mn < 2 kg/mol (degradação avançada): %d dias\n", t)
        break
    end
end

# ============================================================================
# CURVA COMPLETA
# ============================================================================

println("\n\n📊 CURVA DE DEGRADAÇÃO CALIBRADA:")
println("-"^50)
println("Tempo (dias) │  Mn (kg/mol)  │  Redução (%)")
println("-"^50)

for t in [0, 7, 14, 21, 30, 45, 60, 75, 90, 105, 120]
    Mn = mn_model_calibrated(Float64(t), Mn0; k=best_k, alpha=best_alpha)
    reduction = (1 - Mn/Mn0) * 100
    @printf("    %3d      │     %5.1f     │    %5.1f%%\n", t, Mn, reduction)
end

println("\n" * "="^80)
println("  MODELO CALIBRADO COM SUCESSO!")
println("  Use k=$(best_k), α=$(best_alpha) no MorphologyDegradationModel")
println("="^80)
