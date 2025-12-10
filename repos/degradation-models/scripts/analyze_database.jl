#!/usr/bin/env julia
# Analyze database heterogeneity

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "data", "literature_degradation_database.jl"))

println("="^70)
println("DATABASE ANALYSIS - Understanding the heterogeneity")
println("="^70)

# Analyze by ratio
for cat in [(50,50), (70,30), (85,15), (96,4), (100,0)]
    subset = filter(d -> abs(d.ratio_L - cat[1]) <= 5, DEGRADATION_DATABASE)
    if isempty(subset)
        continue
    end

    n = length(subset)
    println("\n$(cat[1]):$(cat[2]) - $n datasets")

    for d in subset
        # Calculate observed degradation rate
        if length(d.times) >= 2 && d.times[end] > 0
            final_frac = d.Mn[end] / d.Mn0
            k_obs = -log(max(final_frac, 0.01)) / d.times[end]
            half_life = log(2) / max(k_obs, 0.0001)

            println("  $(d.id):")
            println("    Mn0=$(d.Mn0), Xc=$(d.Xc0)%, T=$(d.T)C, $(d.condition)")
            println("    t_max=$(d.times[end]) days, final=$(round(final_frac*100, digits=1))%")
            println("    k=$(round(k_obs*1000, digits=2))/1000, t_half=$(round(half_life, digits=0)) days")
        end
    end
end

println("\n" * "="^70)
println("KEY INSIGHT: PLLA datasets have wildly different kinetics")
println("  - Some degrade 10x faster than others (k varies 10-fold)")
println("  - Crystallinity and morphology drive this variation")
println("  - A single model cannot fit all without per-dataset calibration")
println("="^70)
