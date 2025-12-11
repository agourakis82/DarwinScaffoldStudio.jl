"""
BayesianUncertainty.jl

Quantificação de Incerteza Bayesiana para Modelos de Degradação

OBJETIVO:
========
Transformar previsões pontuais em intervalos de confiança 95%:
  Mn(90 dias) = 7.9 kg/mol  →  Mn(90 dias) = 7.9 ± 2.1 kg/mol (95% CI)

METODOLOGIA:
===========
1. Definir priors para parâmetros (k₀, α, Ea, Xc)
2. Likelihood baseada em dados experimentais
3. MCMC (Metropolis-Hastings) para amostrar posterior
4. Propagação de incerteza para previsões

REFERÊNCIAS:
===========
- Gelman et al. 2013: Bayesian Data Analysis
- Saltelli et al. 2008: Global Sensitivity Analysis
- MacKay 2003: Information Theory, Inference, and Learning Algorithms

Author: Darwin Scaffold Studio
Date: 2025-12-11
"""
module BayesianUncertainty

using Statistics
using Random
using Printf

export BayesianParams, PriorDistribution, PosteriorSample
export run_mcmc, predict_with_uncertainty, sensitivity_analysis_sobol
export print_uncertainty_report, credible_interval

# ============================================================================
# ESTRUTURAS DE DADOS
# ============================================================================

"""
Distribuição prior para um parâmetro.
Suporta: :normal, :lognormal, :uniform
"""
struct PriorDistribution
    name::Symbol
    type::Symbol
    mean::Float64
    std::Float64
    lower::Float64
    upper::Float64
end

# Construtor para Normal
PriorNormal(name, mean, std) = PriorDistribution(name, :normal, mean, std, -Inf, Inf)

# Construtor para LogNormal
PriorLogNormal(name, mean, std) = PriorDistribution(name, :lognormal, mean, std, 0.0, Inf)

# Construtor para Uniform
PriorUniform(name, lower, upper) = PriorDistribution(name, :uniform, (lower+upper)/2, (upper-lower)/sqrt(12), lower, upper)

"""
Configuração para inferência Bayesiana.
"""
Base.@kwdef struct BayesianConfig
    # MCMC settings
    n_samples::Int = 5000
    n_burnin::Int = 1000
    n_chains::Int = 4

    # Proposal distribution
    proposal_scale::Float64 = 0.1  # Escala da proposta (fração do prior std)

    # Convergence diagnostics
    target_acceptance::Float64 = 0.234  # Ótimo para Metropolis

    # Model error
    sigma_likelihood::Float64 = 3.0  # kg/mol (erro observacional)
end

"""
Amostra da distribuição posterior.
"""
struct PosteriorSample
    parameters::Dict{Symbol, Vector{Float64}}
    log_likelihood::Vector{Float64}
    acceptance_rate::Float64
    r_hat::Dict{Symbol, Float64}  # Convergence diagnostic
end

"""
Resultado de previsão com incerteza.
"""
struct UncertainPrediction
    times::Vector{Float64}
    mean::Vector{Float64}
    std::Vector{Float64}
    ci_lower::Vector{Float64}  # 2.5%
    ci_upper::Vector{Float64}  # 97.5%
    median::Vector{Float64}
end

# ============================================================================
# PRIORS PADRÃO PARA PLDLA
# ============================================================================

"""
Priors informativos baseados em literatura e dados experimentais.
"""
const DEFAULT_PRIORS_PLDLA = [
    PriorDistribution(:k_L, :lognormal, 0.028, 0.010, 0.005, 0.100),
    PriorDistribution(:k_DL, :lognormal, 0.065, 0.020, 0.020, 0.200),
    PriorDistribution(:alpha_L, :normal, 0.18, 0.05, 0.0, 0.5),
    PriorDistribution(:alpha_DL, :normal, 0.35, 0.10, 0.0, 1.0),
    PriorDistribution(:Ea, :normal, 72.0, 5.0, 50.0, 100.0),
    PriorDistribution(:Xc_initial, :normal, 0.08, 0.02, 0.0, 0.3),
]

const DEFAULT_PRIORS_PLLA = [
    PriorDistribution(:k0, :lognormal, 0.0075, 0.002, 0.001, 0.030),
    PriorDistribution(:alpha, :normal, 0.045, 0.015, 0.0, 0.2),
    PriorDistribution(:Ea, :normal, 82.0, 5.0, 60.0, 100.0),
    PriorDistribution(:Xc_initial, :normal, 0.55, 0.10, 0.3, 0.8),
]

# ============================================================================
# FUNÇÕES DE PROBABILIDADE
# ============================================================================

"""
Calcula log-prior para um conjunto de parâmetros.
"""
function log_prior(params::Dict{Symbol, Float64}, priors::Vector{PriorDistribution})::Float64
    lp = 0.0

    for prior in priors
        if !haskey(params, prior.name)
            continue
        end

        x = params[prior.name]

        # Verificar limites
        if x < prior.lower || x > prior.upper
            return -Inf
        end

        if prior.type == :normal
            lp += -0.5 * ((x - prior.mean) / prior.std)^2
        elseif prior.type == :lognormal
            if x <= 0
                return -Inf
            end
            lp += -0.5 * ((log(x) - log(prior.mean)) / prior.std)^2 - log(x)
        elseif prior.type == :uniform
            lp += 0.0  # Constante, não afeta
        end
    end

    return lp
end

"""
Calcula log-likelihood dados os parâmetros e dados experimentais.
"""
function log_likelihood(
    params::Dict{Symbol, Float64},
    times::Vector{Float64},
    Mn_observed::Vector{Float64},
    model_function::Function,
    sigma::Float64
)::Float64

    # Rodar modelo com parâmetros
    try
        Mn_predicted = model_function(params, times)

        # Likelihood Gaussiana
        residuals = Mn_predicted .- Mn_observed
        ll = -0.5 * sum((residuals ./ sigma).^2)
        ll -= length(Mn_observed) * log(sigma)

        return ll
    catch e
        # Se modelo falhar, retornar -Inf
        return -Inf
    end
end

"""
Calcula log-posterior (prior + likelihood).
"""
function log_posterior(
    params::Dict{Symbol, Float64},
    priors::Vector{PriorDistribution},
    times::Vector{Float64},
    Mn_observed::Vector{Float64},
    model_function::Function,
    sigma::Float64
)::Float64

    lp = log_prior(params, priors)

    if lp == -Inf
        return -Inf
    end

    ll = log_likelihood(params, times, Mn_observed, model_function, sigma)

    return lp + ll
end

# ============================================================================
# MCMC - METROPOLIS-HASTINGS
# ============================================================================

"""
Propõe novo valor para parâmetro (random walk).
Usa proposta proporcional para parâmetros lognormal.
"""
function propose_parameter(
    current::Float64,
    prior::PriorDistribution,
    scale::Float64
)::Float64

    if prior.type == :lognormal
        # Para lognormal: proposta multiplicativa (log-space random walk)
        # Usar coeficiente de variação do prior como escala base
        cv = prior.std / prior.mean
        log_step = randn() * cv * scale
        proposed = current * exp(log_step)
    else
        # Para normal/uniform: proposta aditiva
        step = randn() * prior.std * scale
        proposed = current + step
    end

    # Refletir nos limites
    if proposed < prior.lower
        proposed = prior.lower + abs(prior.lower - proposed)
    end
    if proposed > prior.upper
        proposed = prior.upper - abs(proposed - prior.upper)
    end

    return clamp(proposed, prior.lower, prior.upper)
end

"""
Executa MCMC (Metropolis-Hastings) para amostrar posterior.
"""
function run_mcmc(
    priors::Vector{PriorDistribution},
    times::Vector{Float64},
    Mn_observed::Vector{Float64},
    model_function::Function;
    config::BayesianConfig = BayesianConfig()
)::PosteriorSample

    n_total = config.n_samples + config.n_burnin
    param_names = [p.name for p in priors]
    n_params = length(param_names)

    # Inicializar com médias dos priors
    current_params = Dict{Symbol, Float64}()
    for prior in priors
        current_params[prior.name] = prior.mean
    end

    # Storage para samples
    samples = Dict{Symbol, Vector{Float64}}()
    for name in param_names
        samples[name] = Float64[]
    end
    log_likelihoods = Float64[]

    # Log-posterior atual
    current_lp = log_posterior(
        current_params, priors, times, Mn_observed,
        model_function, config.sigma_likelihood
    )

    # Contadores
    accepted = 0

    # MCMC loop
    for i in 1:n_total
        # Propor novos parâmetros
        proposed_params = copy(current_params)

        for prior in priors
            proposed_params[prior.name] = propose_parameter(
                current_params[prior.name],
                prior,
                config.proposal_scale
            )
        end

        # Calcular log-posterior proposto
        proposed_lp = log_posterior(
            proposed_params, priors, times, Mn_observed,
            model_function, config.sigma_likelihood
        )

        # Critério de aceitação Metropolis
        log_alpha = proposed_lp - current_lp

        if log(rand()) < log_alpha
            # Aceitar
            current_params = proposed_params
            current_lp = proposed_lp
            accepted += 1
        end

        # Salvar após burnin
        if i > config.n_burnin
            for name in param_names
                push!(samples[name], current_params[name])
            end
            push!(log_likelihoods, current_lp)
        end

        # Progress (a cada 1000)
        if i % 1000 == 0
            acc_rate = accepted / i
            @printf("  MCMC: %d/%d (%.1f%% aceito)\r", i, n_total, acc_rate * 100)
        end
    end
    println()

    acceptance_rate = accepted / n_total

    # R-hat simplificado (intra-chain)
    r_hat = Dict{Symbol, Float64}()
    for name in param_names
        s = samples[name]
        n = length(s)
        first_half = s[1:div(n,2)]
        second_half = s[div(n,2)+1:end]

        var1 = var(first_half)
        var2 = var(second_half)
        mean1 = mean(first_half)
        mean2 = mean(second_half)

        W = (var1 + var2) / 2
        B = (mean1 - mean2)^2

        r_hat[name] = sqrt((W + B) / W)
    end

    return PosteriorSample(samples, log_likelihoods, acceptance_rate, r_hat)
end

# ============================================================================
# PREVISÃO COM INCERTEZA
# ============================================================================

"""
Gera previsões com intervalos de confiança.
"""
function predict_with_uncertainty(
    posterior::PosteriorSample,
    times::Vector{Float64},
    model_function::Function;
    ci_level::Float64 = 0.95
)::UncertainPrediction

    param_names = collect(keys(posterior.parameters))
    n_samples = length(posterior.parameters[param_names[1]])
    n_times = length(times)

    # Matrix de previsões
    predictions = zeros(n_samples, n_times)

    for i in 1:n_samples
        # Construir parâmetros para esta amostra
        params = Dict{Symbol, Float64}()
        for name in param_names
            params[name] = posterior.parameters[name][i]
        end

        # Rodar modelo
        try
            predictions[i, :] = model_function(params, times)
        catch
            predictions[i, :] .= NaN
        end
    end

    # Remover NaNs
    valid_rows = .!any(isnan.(predictions), dims=2)[:]
    predictions = predictions[valid_rows, :]

    # Estatísticas
    mean_pred = vec(mean(predictions, dims=1))
    std_pred = vec(std(predictions, dims=1))
    median_pred = vec(mapslices(median, predictions, dims=1))

    # Intervalos de credibilidade
    alpha = (1 - ci_level) / 2
    ci_lower = vec(mapslices(x -> quantile(x, alpha), predictions, dims=1))
    ci_upper = vec(mapslices(x -> quantile(x, 1-alpha), predictions, dims=1))

    return UncertainPrediction(times, mean_pred, std_pred, ci_lower, ci_upper, median_pred)
end

"""
Calcula intervalo de credibilidade para um valor.
"""
function credible_interval(samples::Vector{Float64}; level::Float64=0.95)
    alpha = (1 - level) / 2
    lower = quantile(samples, alpha)
    upper = quantile(samples, 1 - alpha)
    return (lower, upper)
end

# ============================================================================
# ANÁLISE DE SENSIBILIDADE GLOBAL (SOBOL)
# ============================================================================

"""
Análise de sensibilidade de Sobol (primeiro ordem).
Baseado em Saltelli et al. 2008.
"""
function sensitivity_analysis_sobol(
    priors::Vector{PriorDistribution},
    times::Vector{Float64},
    model_function::Function;
    n_samples::Int = 1000,
    output_time_index::Int = -1  # -1 = último tempo
)::Dict{Symbol, Float64}

    param_names = [p.name for p in priors]
    n_params = length(param_names)

    if output_time_index < 0
        output_time_index = length(times)
    end

    # Gerar amostras base (A) e (B)
    A = zeros(n_samples, n_params)
    B = zeros(n_samples, n_params)

    for (j, prior) in enumerate(priors)
        if prior.type == :uniform
            A[:, j] = rand(n_samples) .* (prior.upper - prior.lower) .+ prior.lower
            B[:, j] = rand(n_samples) .* (prior.upper - prior.lower) .+ prior.lower
        else
            A[:, j] = prior.mean .+ randn(n_samples) .* prior.std
            B[:, j] = prior.mean .+ randn(n_samples) .* prior.std
            # Clamp
            A[:, j] = clamp.(A[:, j], prior.lower, prior.upper)
            B[:, j] = clamp.(B[:, j], prior.lower, prior.upper)
        end
    end

    # Calcular saídas para A e B
    function evaluate_model(sample_matrix)
        outputs = zeros(size(sample_matrix, 1))
        for i in 1:size(sample_matrix, 1)
            params = Dict{Symbol, Float64}()
            for (j, name) in enumerate(param_names)
                params[name] = sample_matrix[i, j]
            end
            try
                result = model_function(params, times)
                outputs[i] = result[output_time_index]
            catch
                outputs[i] = NaN
            end
        end
        return outputs
    end

    Y_A = evaluate_model(A)
    Y_B = evaluate_model(B)

    # Variância total
    Y_all = vcat(Y_A, Y_B)
    Y_all = filter(!isnan, Y_all)
    V_total = var(Y_all)

    if V_total == 0
        return Dict(name => 0.0 for name in param_names)
    end

    # Índices de Sobol de primeira ordem
    S1 = Dict{Symbol, Float64}()

    for (i, name) in enumerate(param_names)
        # Matriz C_i: coluna i de B, resto de A
        C_i = copy(A)
        C_i[:, i] = B[:, i]

        Y_Ci = evaluate_model(C_i)

        # Remover NaNs
        valid = .!isnan.(Y_A) .& .!isnan.(Y_Ci)

        if sum(valid) > 10
            # Estimador de Sobol
            numerator = mean(Y_B[valid] .* (Y_Ci[valid] .- Y_A[valid]))
            S1[name] = numerator / V_total
        else
            S1[name] = 0.0
        end
    end

    # Normalizar para somar ~1 (pode não somar exatamente devido a interações)
    total_S1 = sum(values(S1))
    if total_S1 > 0
        for name in param_names
            S1[name] = S1[name] / total_S1
        end
    end

    return S1
end

# ============================================================================
# RELATÓRIO
# ============================================================================

"""
Imprime relatório de incerteza.
"""
function print_uncertainty_report(
    posterior::PosteriorSample,
    prediction::UncertainPrediction
)
    println("="^80)
    println("  QUANTIFICAÇÃO DE INCERTEZA BAYESIANA")
    println("="^80)

    println("\n📊 DIAGNÓSTICOS MCMC:")
    println("-"^60)
    @printf("  Taxa de aceitação: %.1f%% (ótimo: ~23%%)\n", posterior.acceptance_rate * 100)

    println("\n  Convergência (R̂ - deve ser < 1.1):")
    for (name, rhat) in posterior.r_hat
        status = rhat < 1.1 ? "✓" : "⚠"
        @printf("    %s: %.3f %s\n", name, rhat, status)
    end

    println("\n📈 DISTRIBUIÇÃO POSTERIOR DOS PARÂMETROS:")
    println("-"^60)
    println("  Parâmetro      │  Média  │   Std   │   IC 95%%")
    println("-"^60)

    for (name, samples) in posterior.parameters
        m = mean(samples)
        s = std(samples)
        ci = credible_interval(samples)
        @printf("  %-14s │ %7.4f │ %7.4f │ [%.4f, %.4f]\n",
                name, m, s, ci[1], ci[2])
    end

    println("\n🎯 PREVISÕES COM INCERTEZA:")
    println("-"^60)
    println("  Tempo (dias) │ Média Mn │   IC 95%%")
    println("-"^60)

    # Mostrar alguns tempos chave
    key_indices = [1, div(length(prediction.times), 4),
                   div(length(prediction.times), 2),
                   div(3*length(prediction.times), 4),
                   length(prediction.times)]

    for i in key_indices
        if i >= 1 && i <= length(prediction.times)
            @printf("      %3.0f      │  %6.2f  │ [%.2f, %.2f]\n",
                    prediction.times[i], prediction.mean[i],
                    prediction.ci_lower[i], prediction.ci_upper[i])
        end
    end

    println("="^80)
end

end # module
