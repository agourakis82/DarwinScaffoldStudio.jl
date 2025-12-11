"""
test_bayesian_pldla.jl

Teste de Integração: Inferência Bayesiana + Modelo PLDLA Idiossincrático

OBJETIVO:
=========
Demonstrar quantificação de incerteza completa:
  Mn(90 dias) = 7.9 kg/mol  →  Mn(90 dias) = 7.9 ± X kg/mol (95% CI)

WORKFLOW:
=========
1. Carregar dados experimentais do Kaique
2. Definir priors informativos para PLDLA
3. Executar MCMC (Metropolis-Hastings)
4. Verificar convergência (R-hat)
5. Calcular distribuição posterior dos parâmetros
6. Propagar incerteza para previsões
7. Gerar intervalos de credibilidade 95%
8. Análise de sensibilidade (Sobol)

Author: Darwin Scaffold Studio
Date: 2025-12-11
"""

using Printf
using Statistics
using Random

# Seed para reprodutibilidade
Random.seed!(42)

# Incluir módulos
include("../src/DarwinScaffoldStudio/Science/PLDLAIdiosyncraticModel.jl")
include("../src/DarwinScaffoldStudio/Science/BayesianUncertainty.jl")

using .PLDLAIdiosyncraticModel
using .BayesianUncertainty

println("="^80)
println("  QUANTIFICAÇÃO DE INCERTEZA BAYESIANA PARA PLDLA")
println("  Modelo Idiossincrático + MCMC + Intervalos de Credibilidade 95%")
println("="^80)

# ============================================================================
# DADOS EXPERIMENTAIS (KAIQUE 2025)
# ============================================================================

println("\n📊 DADOS EXPERIMENTAIS (Kaique Hergesel, PUC-SP 2025):")
println("-"^60)

const TIMES_EXP = [0.0, 30.0, 60.0, 90.0]
const MN_EXP = [51.285, 25.447, 18.313, 7.904]

for i in eachindex(TIMES_EXP)
    @printf("  Dia %3.0f: Mn = %.3f kg/mol\n", TIMES_EXP[i], MN_EXP[i])
end

# ============================================================================
# FUNÇÃO MODELO PARA MCMC
# ============================================================================

"""
Modelo trifásico com transições suaves para degradação de PLDLA.

Perfil observado (dados Kaique):
- Fase 1 (0-30d): Queda RÁPIDA 51->25 kg/mol (hidrólise regiões amorfas DL)
- Fase 2 (30-60d): Queda LENTA 25->18 kg/mol (cristalização protege)
- Fase 3 (60-90d): Queda RÁPIDA 18->8 kg/mol (colapso autocatalítico)

Usa funções sigmóide para transições suaves entre fases.
"""
function pldla_model_for_mcmc(params::Dict{Symbol, Float64}, times::Vector{Float64})::Vector{Float64}
    # Parâmetros do modelo trifásico (calibrados)
    k1 = get(params, :k1, 0.026)         # Taxa fase 1 (rápida inicial)
    k2 = get(params, :k2, 0.006)         # Taxa fase 2 (lenta - platô)
    k3 = get(params, :k3, 0.028)         # Taxa fase 3 (rápida final)

    # Tempos de transição entre fases
    t_trans1 = 30.0   # Transição fase 1 -> 2
    t_trans2 = 60.0   # Transição fase 2 -> 3
    w_trans = 8.0     # Largura da transição (dias)

    Mn0 = 51.285
    Mn_min = 5.0
    dt = 0.5
    t_max = maximum(times) + 1.0

    # Função sigmóide para transição suave
    sigmoid(t, t_mid, width) = 1.0 / (1.0 + exp(-(t - t_mid) / width))

    # Simular com Euler
    Mn = Mn0
    t_vals = Float64[0.0]
    Mn_vals = Float64[Mn0]

    t_curr = 0.0
    while t_curr < t_max
        # Pesos das fases (transição suave)
        w1 = 1.0 - sigmoid(t_curr, t_trans1, w_trans)
        w2 = sigmoid(t_curr, t_trans1, w_trans) * (1.0 - sigmoid(t_curr, t_trans2, w_trans))
        w3 = sigmoid(t_curr, t_trans2, w_trans)

        # Taxa efetiva (média ponderada)
        k = w1 * k1 + w2 * k2 + w3 * k3

        dMn = -k * Mn * dt
        Mn = max(Mn_min, Mn + dMn)
        t_curr += dt
        push!(t_vals, t_curr)
        push!(Mn_vals, Mn)
    end

    # Interpolar nos tempos solicitados
    Mn_pred = Float64[]
    for t in times
        if t <= 0.0
            push!(Mn_pred, Mn0)
        else
            idx = searchsortedfirst(t_vals, t)
            if idx > length(t_vals)
                push!(Mn_pred, Mn_vals[end])
            elseif idx == 1
                push!(Mn_pred, Mn0)
            else
                t1_interp, t2_interp = t_vals[idx-1], t_vals[idx]
                w = (t - t1_interp) / (t2_interp - t1_interp)
                push!(Mn_pred, (1-w) * Mn_vals[idx-1] + w * Mn_vals[idx])
            end
        end
    end

    return Mn_pred
end

# ============================================================================
# DEFINIR PRIORS INFORMATIVOS
# ============================================================================

println("\n📐 PRIORS INFORMATIVOS (baseados em literatura e dados preliminares):")
println("-"^60)

# Priors para modelo trifásico - amplos para explorar incerteza
const PLDLA_PRIORS = [
    BayesianUncertainty.PriorDistribution(:k1, :normal, 0.026, 0.008, 0.010, 0.050),
    BayesianUncertainty.PriorDistribution(:k2, :normal, 0.006, 0.003, 0.001, 0.015),
    BayesianUncertainty.PriorDistribution(:k3, :normal, 0.028, 0.010, 0.010, 0.060),
]

for prior in PLDLA_PRIORS
    @printf("  %s: %s(μ=%.3f, σ=%.3f) ∈ [%.3f, %.3f]\n",
            prior.name, prior.type, prior.mean, prior.std, prior.lower, prior.upper)
end

# ============================================================================
# EXECUTAR MCMC
# ============================================================================

println("\n🔄 EXECUTANDO MCMC (Metropolis-Hastings):")
println("-"^60)

# Configuração MCMC - otimizada para boa exploração
mcmc_config = BayesianUncertainty.BayesianConfig(
    n_samples = 10000,
    n_burnin = 3000,
    n_chains = 1,
    proposal_scale = 0.50,  # Amplo para explorar
    sigma_likelihood = 1.5  # Erro experimental estimado
)

@printf("  Amostras: %d (+ %d burn-in)\n", mcmc_config.n_samples, mcmc_config.n_burnin)
@printf("  Escala proposta: %.2f\n", mcmc_config.proposal_scale)
@printf("  σ likelihood: %.1f kg/mol\n", mcmc_config.sigma_likelihood)
println()

# Executar MCMC
@time posterior = BayesianUncertainty.run_mcmc(
    PLDLA_PRIORS,
    TIMES_EXP,
    MN_EXP,
    pldla_model_for_mcmc;
    config = mcmc_config
)

# ============================================================================
# DIAGNÓSTICOS DE CONVERGÊNCIA
# ============================================================================

println("\n📈 DIAGNÓSTICOS DE CONVERGÊNCIA:")
println("-"^60)
@printf("  Taxa de aceitação: %.1f%% (ótimo: ~23%%)\n", posterior.acceptance_rate * 100)

if posterior.acceptance_rate < 0.15
    println("  ⚠️  Taxa baixa - considere aumentar proposal_scale")
elseif posterior.acceptance_rate > 0.35
    println("  ⚠️  Taxa alta - considere diminuir proposal_scale")
else
    println("  ✓ Taxa de aceitação adequada")
end

println("\n  Convergência (R̂ - Gelman-Rubin):")
global all_converged = true
for (name, rhat) in posterior.r_hat
    status = rhat < 1.1 ? "✓" : "⚠"
    if rhat >= 1.1
        global all_converged = false
    end
    @printf("    %s: %.3f %s\n", name, rhat, status)
end

if all_converged
    println("\n  ✓ Todas as cadeias convergiram (R̂ < 1.1)")
else
    println("\n  ⚠️  Algumas cadeias não convergiram - aumentar n_samples")
end

# ============================================================================
# DISTRIBUIÇÃO POSTERIOR DOS PARÂMETROS
# ============================================================================

println("\n📊 DISTRIBUIÇÃO POSTERIOR DOS PARÂMETROS:")
println("-"^60)
println("  Parâmetro    │   Prior Mean │ Posterior Mean │ Posterior Std │   IC 95%")
println("-"^80)

for prior in PLDLA_PRIORS
    samples = posterior.parameters[prior.name]
    m = mean(samples)
    s = std(samples)
    ci = BayesianUncertainty.credible_interval(samples; level=0.95)

    @printf("  %-12s │     %.4f   │      %.4f    │     %.4f    │ [%.4f, %.4f]\n",
            prior.name, prior.mean, m, s, ci[1], ci[2])
end

# ============================================================================
# PREVISÕES COM INCERTEZA
# ============================================================================

println("\n🎯 PREVISÕES COM INTERVALOS DE CREDIBILIDADE 95%:")
println("-"^60)

# Tempos para previsão (resolução fina)
times_pred = collect(0.0:5.0:90.0)

# Gerar previsões
prediction = BayesianUncertainty.predict_with_uncertainty(
    posterior,
    times_pred,
    pldla_model_for_mcmc;
    ci_level = 0.95
)

println("  Tempo (dias) │ Mn Médio │   IC 95%          │ Largura IC")
println("-"^70)

key_times = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
for t in key_times
    idx = findfirst(x -> x >= t, times_pred)
    if idx !== nothing
        width = prediction.ci_upper[idx] - prediction.ci_lower[idx]
        @printf("      %3.0f      │  %6.2f  │ [%5.2f, %6.2f]   │   %.2f\n",
                prediction.times[idx], prediction.mean[idx],
                prediction.ci_lower[idx], prediction.ci_upper[idx], width)
    end
end

# ============================================================================
# COMPARAÇÃO COM DADOS EXPERIMENTAIS
# ============================================================================

println("\n📋 COMPARAÇÃO PREDIÇÃO vs EXPERIMENTAL:")
println("-"^70)
println("  Dia │ Mn Exp │ Mn Pred │  IC 95%           │ Experimental em IC?")
println("-"^70)

for i in eachindex(TIMES_EXP)
    t = TIMES_EXP[i]
    idx = findfirst(x -> x >= t, times_pred)
    if idx !== nothing
        in_ci = prediction.ci_lower[idx] <= MN_EXP[i] <= prediction.ci_upper[idx]
        status = in_ci ? "✓" : "✗"
        @printf("  %3.0f │ %5.2f  │  %5.2f  │ [%5.2f, %6.2f]  │      %s\n",
                t, MN_EXP[i], prediction.mean[idx],
                prediction.ci_lower[idx], prediction.ci_upper[idx], status)
    end
end

# Cobertura
global coverage = 0
for i in eachindex(TIMES_EXP)
    t = TIMES_EXP[i]
    idx = findfirst(x -> x >= t, times_pred)
    if idx !== nothing
        if prediction.ci_lower[idx] <= MN_EXP[i] <= prediction.ci_upper[idx]
            global coverage += 1
        end
    end
end
coverage_pct = coverage / length(TIMES_EXP) * 100

println("-"^70)
@printf("  Cobertura do IC 95%%: %.0f%% (%d/%d pontos)\n", coverage_pct, coverage, length(TIMES_EXP))

if coverage_pct >= 90
    println("  ✓ Cobertura adequada para IC 95%")
else
    println("  ⚠️  Cobertura abaixo do esperado - revisar σ_likelihood ou modelo")
end

# ============================================================================
# ANÁLISE DE SENSIBILIDADE GLOBAL (SOBOL)
# ============================================================================

println("\n🔬 ANÁLISE DE SENSIBILIDADE GLOBAL (SOBOL - Índices de Primeira Ordem):")
println("-"^60)

sobol = BayesianUncertainty.sensitivity_analysis_sobol(
    PLDLA_PRIORS,
    TIMES_EXP,
    pldla_model_for_mcmc;
    n_samples = 500,
    output_time_index = 4  # Mn em t=90 dias
)

println("  Parâmetro    │ Índice S₁ │ Contribuição para variância Mn(90d)")
println("-"^70)

sorted_sobol = sort(collect(sobol), by=x->x[2], rev=true)
for (name, s1) in sorted_sobol
    bar_len = max(0, round(Int, s1 * 30))
    bar = "█"^bar_len
    @printf("  %-12s │   %.3f   │ %s\n", name, s1, bar)
end

println()
println("  Interpretação:")
principal = sorted_sobol[1]
@printf("  → %s é o parâmetro mais influente (%.1f%% da variância)\n",
        principal[1], principal[2] * 100)

# ============================================================================
# RESUMO FINAL
# ============================================================================

println("\n" * "="^80)
println("  RESUMO: QUANTIFICAÇÃO DE INCERTEZA COMPLETA")
println("="^80)

println("\n  📌 RESULTADO PRINCIPAL:")
idx_90 = findfirst(x -> x >= 90.0, times_pred)
@printf("\n     Mn(90 dias) = %.2f ± %.2f kg/mol (IC 95%%: [%.2f, %.2f])\n",
        prediction.mean[idx_90],
        (prediction.ci_upper[idx_90] - prediction.ci_lower[idx_90]) / 2,
        prediction.ci_lower[idx_90],
        prediction.ci_upper[idx_90])

println("\n  📊 MÉTRICAS DE QUALIDADE:")
@printf("     • Taxa de aceitação MCMC: %.1f%%\n", posterior.acceptance_rate * 100)
@printf("     • Convergência (R̂ < 1.1): %s\n", all_converged ? "✓ Sim" : "⚠ Não")
@printf("     • Cobertura IC 95%%: %.0f%%\n", coverage_pct)

println("\n  🎯 PARÂMETROS MAIS SENSÍVEIS:")
for (i, (name, s1)) in enumerate(sorted_sobol[1:min(2, length(sorted_sobol))])
    @printf("     %d. %s (%.1f%% da variância)\n", i, name, s1 * 100)
end

println("\n  ✅ PRONTO PARA PUBLICAÇÃO:")
println("     • Previsões com incerteza quantificada")
println("     • Intervalos de credibilidade 95%")
println("     • Análise de sensibilidade global")
println("     • Diagnósticos de convergência MCMC")

println("\n" * "="^80)
