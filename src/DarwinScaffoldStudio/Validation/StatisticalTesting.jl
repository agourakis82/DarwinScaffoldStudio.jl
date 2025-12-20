"""
StatisticalTesting.jl - Statistical Testing Framework for Scaffold Comparison

Provides rigorous statistical analysis for comparing scaffold properties,
including effect sizes, multiple comparison corrections, and power analysis.

SOTA 2024-2025 Features:
- Effect sizes with interpretation (Cohen's d, Glass's delta, Hedges' g, Cliff's delta)
- Multiple comparison corrections (Bonferroni, Holm, FDR)
- Power analysis for experimental design
- Bootstrap confidence intervals
- Scaffold-specific comparison tests

References:
- Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"
- Benjamini & Hochberg (1995) "Controlling the False Discovery Rate"
- Romano et al. (2006) "Cliff's delta" for non-parametric effect size
- VanderWeele & Ding (2017) "E-values for sensitivity analysis"
"""
module StatisticalTesting

using Statistics
using Random
using Distributions
using Printf

export EffectSize, HypothesisTestResult, PowerAnalysisResult
export cohens_d, glass_delta, hedges_g, cliff_delta
export bonferroni_correction, holm_bonferroni, fdr_correction
export power_analysis, required_sample_size
export scaffold_comparison_test, batch_scaffold_comparison
export bootstrap_ci, permutation_test
export generate_statistical_report

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

"""
    EffectSize

Container for effect size metrics with interpretation.

# Fields
- `metric::Symbol`: Effect size type (:cohens_d, :glass_delta, :hedges_g, :cliff_delta)
- `value::Float64`: Effect size value
- `interpretation::String`: Qualitative interpretation
- `ci_lower::Float64`: Lower bound of confidence interval
- `ci_upper::Float64`: Upper bound of confidence interval
"""
struct EffectSize
    metric::Symbol
    value::Float64
    interpretation::String
    ci_lower::Float64
    ci_upper::Float64
end

"""
    HypothesisTestResult

Result of a hypothesis test.
"""
struct HypothesisTestResult
    test_name::String
    statistic::Float64
    p_value::Float64
    df::Union{Float64, Nothing}
    significant_005::Bool
    significant_001::Bool
    effect_size::Union{EffectSize, Nothing}
end

"""
    PowerAnalysisResult

Result of power analysis.
"""
struct PowerAnalysisResult
    effect_size::Float64
    n1::Int
    n2::Int
    alpha::Float64
    power::Float64
    sufficient::Bool  # power >= 0.80
end

# ============================================================================
# EFFECT SIZE CALCULATIONS
# ============================================================================

"""
    cohens_d(group1, group2; n_bootstrap=1000, alpha=0.05) -> EffectSize

Cohen's d effect size: standardized mean difference.

d = (M₁ - M₂) / s_pooled

Interpretation (Cohen 1988):
- |d| < 0.2: negligible
- 0.2 ≤ |d| < 0.5: small
- 0.5 ≤ |d| < 0.8: medium
- |d| ≥ 0.8: large

# Arguments
- `group1`, `group2`: Data vectors
- `n_bootstrap`: Number of bootstrap samples for CI
- `alpha`: Significance level for CI
"""
function cohens_d(group1::Vector{<:Real}, group2::Vector{<:Real};
                  n_bootstrap::Int=1000, alpha::Float64=0.05)
    n1, n2 = length(group1), length(group2)
    m1, m2 = mean(group1), mean(group2)
    s1, s2 = std(group1), std(group2)

    # Pooled standard deviation
    s_pooled = sqrt(((n1 - 1) * s1^2 + (n2 - 1) * s2^2) / (n1 + n2 - 2))

    # Avoid division by zero
    if s_pooled < 1e-10
        d = 0.0
    else
        d = (m1 - m2) / s_pooled
    end

    # Bootstrap CI
    bootstrap_d = Float64[]
    for _ in 1:n_bootstrap
        boot1 = group1[rand(1:n1, n1)]
        boot2 = group2[rand(1:n2, n2)]
        m1_b, m2_b = mean(boot1), mean(boot2)
        s1_b, s2_b = std(boot1), std(boot2)
        s_b = sqrt(((n1 - 1) * s1_b^2 + (n2 - 1) * s2_b^2) / (n1 + n2 - 2))
        if s_b > 1e-10
            push!(bootstrap_d, (m1_b - m2_b) / s_b)
        end
    end

    if isempty(bootstrap_d)
        ci_lower, ci_upper = d, d
    else
        ci = quantile(bootstrap_d, [alpha/2, 1 - alpha/2])
        ci_lower, ci_upper = ci[1], ci[2]
    end

    interp = interpret_cohens_d(abs(d))

    return EffectSize(:cohens_d, d, interp, ci_lower, ci_upper)
end

"""
    interpret_cohens_d(d::Float64) -> String

Interpret Cohen's d value.
"""
function interpret_cohens_d(d::Float64)
    if d < 0.2
        return "negligible"
    elseif d < 0.5
        return "small"
    elseif d < 0.8
        return "medium"
    else
        return "large"
    end
end

"""
    glass_delta(treatment, control) -> EffectSize

Glass's delta: uses only control group SD as denominator.
Better when treatment may affect variance.
"""
function glass_delta(treatment::Vector{<:Real}, control::Vector{<:Real})
    s_control = std(control)
    if s_control < 1e-10
        delta = 0.0
    else
        delta = (mean(treatment) - mean(control)) / s_control
    end

    interp = interpret_cohens_d(abs(delta))
    return EffectSize(:glass_delta, delta, interp, NaN, NaN)
end

"""
    hedges_g(group1, group2) -> EffectSize

Hedges' g: bias-corrected Cohen's d for small samples.
"""
function hedges_g(group1::Vector{<:Real}, group2::Vector{<:Real})
    d_result = cohens_d(group1, group2; n_bootstrap=0)
    n1, n2 = length(group1), length(group2)

    # Correction factor (Hedges & Olkin, 1985)
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    g = J * d_result.value

    interp = interpret_cohens_d(abs(g))
    return EffectSize(:hedges_g, g, interp, NaN, NaN)
end

"""
    cliff_delta(group1, group2) -> EffectSize

Cliff's delta: non-parametric effect size (ordinal data compatible).

δ = P(X > Y) - P(X < Y)

Range: [-1, 1]

Interpretation (Romano et al. 2006):
- |δ| < 0.147: negligible
- 0.147 ≤ |δ| < 0.33: small
- 0.33 ≤ |δ| < 0.474: medium
- |δ| ≥ 0.474: large
"""
function cliff_delta(group1::Vector{<:Real}, group2::Vector{<:Real})
    n1, n2 = length(group1), length(group2)

    # Count dominance pairs
    greater = 0
    less = 0
    for x in group1
        for y in group2
            if x > y
                greater += 1
            elseif x < y
                less += 1
            end
        end
    end

    delta = (greater - less) / (n1 * n2)

    interp = if abs(delta) < 0.147
        "negligible"
    elseif abs(delta) < 0.33
        "small"
    elseif abs(delta) < 0.474
        "medium"
    else
        "large"
    end

    return EffectSize(:cliff_delta, delta, interp, NaN, NaN)
end

# ============================================================================
# MULTIPLE COMPARISON CORRECTIONS
# ============================================================================

"""
    bonferroni_correction(p_values; alpha=0.05) -> (adjusted, significant)

Bonferroni correction: most conservative.
α_adj = α / n_tests
"""
function bonferroni_correction(p_values::Vector{Float64}; alpha::Float64=0.05)
    n = length(p_values)
    adjusted = min.(p_values .* n, 1.0)
    significant = adjusted .< alpha
    return adjusted, significant
end

"""
    holm_bonferroni(p_values; alpha=0.05) -> (adjusted, significant)

Holm-Bonferroni step-down procedure: less conservative than Bonferroni.
"""
function holm_bonferroni(p_values::Vector{Float64}; alpha::Float64=0.05)
    n = length(p_values)
    order = sortperm(p_values)
    adjusted = zeros(n)

    for (i, idx) in enumerate(order)
        adjusted[idx] = min(p_values[idx] * (n - i + 1), 1.0)
    end

    # Ensure monotonicity (step-down)
    for i in 2:n
        adjusted[order[i]] = max(adjusted[order[i]], adjusted[order[i-1]])
    end

    significant = adjusted .< alpha
    return adjusted, significant
end

"""
    fdr_correction(p_values; method=:benjamini_hochberg, alpha=0.05)

False Discovery Rate control.

# Methods
- `:benjamini_hochberg`: Original BH procedure (1995) - independent tests
- `:benjamini_yekutieli`: BY procedure (2001) - dependent tests
"""
function fdr_correction(p_values::Vector{Float64};
                        method::Symbol=:benjamini_hochberg, alpha::Float64=0.05)
    n = length(p_values)
    order = sortperm(p_values)
    adjusted = zeros(n)

    if method == :benjamini_hochberg
        for (rank, idx) in enumerate(order)
            adjusted[idx] = p_values[idx] * n / rank
        end
    elseif method == :benjamini_yekutieli
        c_m = sum(1.0 / k for k in 1:n)  # Correction for dependence
        for (rank, idx) in enumerate(order)
            adjusted[idx] = p_values[idx] * n * c_m / rank
        end
    else
        error("Unknown FDR method: $method. Use :benjamini_hochberg or :benjamini_yekutieli")
    end

    # Ensure monotonicity (step-up: work backwards)
    for i in (n-1):-1:1
        adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i+1]])
    end

    adjusted = min.(adjusted, 1.0)
    significant = adjusted .< alpha

    return adjusted, significant
end

# ============================================================================
# POWER ANALYSIS
# ============================================================================

"""
    power_analysis(effect_size, n1, n2; alpha=0.05) -> PowerAnalysisResult

Compute statistical power for given sample sizes and effect size.

Power = P(reject H₀ | H₁ is true)

For dissertation: target power ≥ 0.80
"""
function power_analysis(effect_size::Float64, n1::Int, n2::Int;
                        alpha::Float64=0.05)
    # Non-central t-distribution approach
    df = n1 + n2 - 2

    # Non-centrality parameter
    ncp = effect_size * sqrt(n1 * n2 / (n1 + n2))

    # Critical t-value for two-tailed test
    t_crit = quantile(TDist(df), 1 - alpha / 2)

    # Power = P(|T| > t_crit | H1)
    # Using non-central t distribution
    nct = NoncentralT(df, ncp)
    power = 1 - cdf(nct, t_crit) + cdf(nct, -t_crit)

    return PowerAnalysisResult(effect_size, n1, n2, alpha, power, power >= 0.80)
end

"""
    required_sample_size(effect_size; power=0.80, alpha=0.05, ratio=1.0)

Compute required sample size per group to achieve target power.

# Arguments
- `effect_size`: Expected effect size (Cohen's d)
- `power`: Target power (default 0.80)
- `alpha`: Significance level (default 0.05)
- `ratio`: n2/n1 ratio (default 1.0 for equal groups)

# Returns
- Tuple (n1, n2, achieved_power)
"""
function required_sample_size(effect_size::Float64;
                              power::Float64=0.80, alpha::Float64=0.05,
                              ratio::Float64=1.0)
    # Binary search for required n
    for n in 5:1000
        n2 = max(5, round(Int, n * ratio))
        result = power_analysis(effect_size, n, n2; alpha=alpha)
        if result.power >= power
            return (n, n2, result.power)
        end
    end

    # If not found, return max checked
    return (1000, round(Int, 1000 * ratio), power_analysis(effect_size, 1000, round(Int, 1000 * ratio); alpha=alpha).power)
end

# ============================================================================
# BOOTSTRAP AND PERMUTATION TESTS
# ============================================================================

"""
    bootstrap_ci(data, statistic; n_bootstrap=10000, alpha=0.05)

Compute bootstrap confidence interval for any statistic.

# Arguments
- `data`: Data vector or matrix
- `statistic`: Function to compute statistic (e.g., mean, median)
- `n_bootstrap`: Number of bootstrap samples
- `alpha`: Significance level (default 0.05 for 95% CI)
"""
function bootstrap_ci(data::Vector{<:Real}, statistic::Function;
                      n_bootstrap::Int=10000, alpha::Float64=0.05)
    n = length(data)
    bootstrap_stats = Float64[]

    for _ in 1:n_bootstrap
        boot_sample = data[rand(1:n, n)]
        push!(bootstrap_stats, statistic(boot_sample))
    end

    ci = quantile(bootstrap_stats, [alpha/2, 1 - alpha/2])
    point_estimate = statistic(data)

    return (estimate=point_estimate, ci_lower=ci[1], ci_upper=ci[2],
            se=std(bootstrap_stats))
end

"""
    permutation_test(group1, group2, statistic; n_permutations=10000)

Non-parametric permutation test for comparing two groups.

# Arguments
- `group1`, `group2`: Data vectors
- `statistic`: Function to compute test statistic (default: difference of means)
- `n_permutations`: Number of permutations
"""
function permutation_test(group1::Vector{<:Real}, group2::Vector{<:Real},
                          statistic::Function=((x, y) -> mean(x) - mean(y));
                          n_permutations::Int=10000)
    # Observed statistic
    observed = statistic(group1, group2)

    # Combined data
    combined = vcat(group1, group2)
    n1 = length(group1)
    n_total = length(combined)

    # Permutation distribution
    perm_stats = Float64[]
    for _ in 1:n_permutations
        perm = shuffle(combined)
        perm_g1 = perm[1:n1]
        perm_g2 = perm[n1+1:end]
        push!(perm_stats, statistic(perm_g1, perm_g2))
    end

    # Two-tailed p-value
    p_value = mean(abs.(perm_stats) .>= abs(observed))

    return (observed=observed, p_value=p_value,
            null_mean=mean(perm_stats), null_std=std(perm_stats))
end

# ============================================================================
# SCAFFOLD COMPARISON TESTS
# ============================================================================

"""
    scaffold_comparison_test(metrics1, metrics2; test=:ttest, paired=false)

Statistical comparison of scaffold metrics between two groups.

Returns comprehensive test results including p-value, effect size, and power.

# Arguments
- `metrics1`, `metrics2`: Vectors of metric values
- `test`: Test type (:ttest, :wilcoxon, :permutation)
- `paired`: Whether samples are paired
"""
function scaffold_comparison_test(metrics1::Vector{<:Real}, metrics2::Vector{<:Real};
                                  test::Symbol=:ttest, paired::Bool=false)
    n1, n2 = length(metrics1), length(metrics2)

    # Convert to Float64
    g1 = Float64.(metrics1)
    g2 = Float64.(metrics2)

    # Compute test statistic and p-value
    if test == :ttest
        if paired
            @assert n1 == n2 "Paired test requires equal sample sizes"
            diff = g1 .- g2
            t_stat = mean(diff) / (std(diff) / sqrt(n1))
            df = n1 - 1
            p_value = 2 * (1 - cdf(TDist(df), abs(t_stat)))
            test_name = "Paired t-test"
        else
            # Welch's t-test (unequal variances)
            m1, m2 = mean(g1), mean(g2)
            s1, s2 = std(g1), std(g2)
            se = sqrt(s1^2/n1 + s2^2/n2)

            if se < 1e-10
                t_stat = 0.0
                df = n1 + n2 - 2
            else
                t_stat = (m1 - m2) / se
                # Welch-Satterthwaite degrees of freedom
                df = (s1^2/n1 + s2^2/n2)^2 /
                     ((s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1))
            end
            p_value = 2 * (1 - cdf(TDist(df), abs(t_stat)))
            test_name = "Welch's t-test"
        end
        statistic = t_stat
    elseif test == :permutation
        result = permutation_test(g1, g2)
        statistic = result.observed
        p_value = result.p_value
        df = nothing
        test_name = "Permutation test"
    else
        error("Unknown test: $test. Use :ttest or :permutation")
    end

    # Effect size
    effect = cohens_d(g1, g2)

    # Power
    power_result = power_analysis(abs(effect.value), n1, n2)

    return Dict(
        "test_name" => test_name,
        "statistic" => statistic,
        "p_value" => p_value,
        "df" => test == :ttest ? df : nothing,
        "effect_size" => effect,
        "power" => power_result.power,
        "n_samples" => (n1, n2),
        "significant_005" => p_value < 0.05,
        "significant_001" => p_value < 0.01,
        "means" => (mean(g1), mean(g2)),
        "stds" => (std(g1), std(g2))
    )
end

"""
    batch_scaffold_comparison(data; correction=:fdr, alpha=0.05)

Compare multiple scaffold groups with multiple comparison correction.

# Arguments
- `data`: Dict mapping group names to metric vectors
- `correction`: Correction method (:bonferroni, :holm, :fdr)
- `alpha`: Significance level
"""
function batch_scaffold_comparison(data::Dict{String, Vector{<:Real}};
                                   correction::Symbol=:fdr, alpha::Float64=0.05)
    groups = collect(keys(data))
    n_groups = length(groups)

    # Pairwise comparisons
    comparisons = []
    p_values = Float64[]

    for i in 1:(n_groups-1)
        for j in (i+1):n_groups
            result = scaffold_comparison_test(
                Float64.(data[groups[i]]),
                Float64.(data[groups[j]])
            )
            push!(comparisons, (groups[i], groups[j], result))
            push!(p_values, result["p_value"])
        end
    end

    # Multiple comparison correction
    if correction == :bonferroni
        adjusted, significant = bonferroni_correction(p_values; alpha=alpha)
    elseif correction == :holm
        adjusted, significant = holm_bonferroni(p_values; alpha=alpha)
    else
        adjusted, significant = fdr_correction(p_values; alpha=alpha)
    end

    # Compile results
    results = []
    for (i, (g1, g2, res)) in enumerate(comparisons)
        push!(results, Dict(
            "group1" => g1,
            "group2" => g2,
            "p_value_raw" => res["p_value"],
            "p_value_adjusted" => adjusted[i],
            "significant" => significant[i],
            "effect_size" => res["effect_size"],
            "effect_interpretation" => res["effect_size"].interpretation,
            "power" => res["power"],
            "means" => res["means"],
            "stds" => res["stds"]
        ))
    end

    return results
end

# ============================================================================
# REPORT GENERATION
# ============================================================================

"""
    generate_statistical_report(results; format=:markdown)

Generate publication-ready statistical report.

# Arguments
- `results`: Results from batch_scaffold_comparison
- `format`: Output format (:markdown, :latex, :text)
"""
function generate_statistical_report(results::Vector; format::Symbol=:markdown)
    io = IOBuffer()

    if format == :markdown
        println(io, "# Statistical Comparison Report")
        println(io, "\n## Pairwise Comparisons\n")
        println(io, "| Group 1 | Group 2 | p (raw) | p (adj) | Effect Size | Interpretation | Significant |")
        println(io, "|---------|---------|---------|---------|-------------|----------------|-------------|")

        for r in results
            sig_mark = r["significant"] ? "**Yes**" : "No"
            println(io, @sprintf("| %s | %s | %.4f | %.4f | %.3f | %s | %s |",
                r["group1"], r["group2"],
                r["p_value_raw"], r["p_value_adjusted"],
                r["effect_size"].value, r["effect_interpretation"],
                sig_mark))
        end

        println(io, "\n## Summary Statistics\n")
        println(io, "| Group | Mean | SD |")
        println(io, "|-------|------|-----|")

        # Extract unique groups
        seen = Set{String}()
        for r in results
            for (g, m, s) in [(r["group1"], r["means"][1], r["stds"][1]),
                              (r["group2"], r["means"][2], r["stds"][2])]
                if !(g in seen)
                    push!(seen, g)
                    println(io, @sprintf("| %s | %.4f | %.4f |", g, m, s))
                end
            end
        end

    elseif format == :text
        println(io, "Statistical Comparison Report")
        println(io, "=" ^ 60)
        println(io)

        for r in results
            println(io, "$(r["group1"]) vs $(r["group2"]):")
            println(io, "  p-value (raw):      $(round(r["p_value_raw"], digits=4))")
            println(io, "  p-value (adjusted): $(round(r["p_value_adjusted"], digits=4))")
            println(io, "  Effect size:        $(round(r["effect_size"].value, digits=3)) ($(r["effect_interpretation"]))")
            println(io, "  Significant:        $(r["significant"] ? "Yes" : "No")")
            println(io, "  Power:              $(round(r["power"], digits=3))")
            println(io)
        end
    end

    return String(take!(io))
end

end # module
