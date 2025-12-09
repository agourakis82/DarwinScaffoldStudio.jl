#!/usr/bin/env julia
"""Test Fast Cubical TDA for tortuosity prediction"""

using Pkg
Pkg.activate(".")

using TiffImages, CSV, DataFrames, Statistics, Printf, Random

include("../src/DarwinScaffoldStudio/Science/FastCubicalTDA.jl")
using .FastCubicalTDA

# Load data
df = CSV.read("data/soil_pore_space/characteristics.csv", DataFrame)

println("="^65)
println("FAST CUBICAL TDA - TORTUOSITY PREDICTION")
println("="^65)

n_test = min(50, nrow(df))
Random.seed!(42)
indices = randperm(nrow(df))[1:n_test]

errors = Float64[]
times = Float64[]
predictions = Float64[]
ground_truths = Float64[]

for (i, idx) in enumerate(indices)
    row = df[idx, :]
    filepath = joinpath("data/soil_pore_space", row.file)

    if !isfile(filepath)
        continue
    end

    try
        img = TiffImages.load(filepath)
        binary = Array{Bool,3}(Float64.(img) .> 0.5)

        # Time the computation
        t1 = time()
        features = fast_topological_features(binary)
        t2 = time()

        elapsed_ms = (t2 - t1) * 1000
        push!(times, elapsed_ms)

        # Ground truth
        tau_gt = row["mean geodesic tortuosity"]
        porosity = features.porosity

        # Topology-informed tortuosity estimate
        # Calibrated formula based on cubical homology
        alpha = 0.12
        beta = 0.015
        gamma = 0.02

        # τ ≈ 1 + α*(1-φ)/φ - β*log(1+β₁) + γ*surface_complexity
        tau_pred = 1.0 + alpha * (1 - porosity) / (porosity + 0.1)
        tau_pred -= beta * log(1 + features.betti_1)
        tau_pred += gamma * features.surface_area
        tau_pred = max(1.0, tau_pred)

        push!(predictions, tau_pred)
        push!(ground_truths, tau_gt)

        rel_err = abs(tau_pred - tau_gt) / tau_gt * 100
        push!(errors, rel_err)

        status = rel_err < 5.0 ? "OK" : "MISS"
        @printf("[%2d] B=(%d,%d,%d) pred=%.4f gt=%.4f err=%.1f%% [%s] %dms\n",
                i, features.betti_0, features.betti_1, features.betti_2,
                tau_pred, tau_gt, rel_err, status, round(Int, elapsed_ms))
    catch e
        @printf("[%2d] Error: %s\n", i, string(e)[1:min(50, length(string(e)))])
    end
end

println("\n" * "="^65)
println("RESULTS")
println("="^65)
@printf("Samples tested: %d\n", length(errors))
@printf("Mean inference time: %.1f ms\n", mean(times))
@printf("MAE: %.4f\n", mean(abs.(predictions .- ground_truths)))
@printf("MRE: %.2f%%\n", mean(errors))
@printf("Within 5%%: %d/%d (%.1f%%)\n", count(x->x<5, errors), length(errors), count(x->x<5, errors)/length(errors)*100)
@printf("Within 10%%: %d/%d (%.1f%%)\n", count(x->x<10, errors), length(errors), count(x->x<10, errors)/length(errors)*100)

# Compare to FMM time (~5000ms)
fmm_time = 5000.0
speedup = fmm_time / mean(times)
@printf("\nSpeedup vs FMM: %.0fx\n", speedup)
