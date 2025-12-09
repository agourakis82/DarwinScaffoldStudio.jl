#!/usr/bin/env julia
"""Quick tortuosity validation test"""

using Pkg
Pkg.activate(".")

using TiffImages, Statistics, CSV, DataFrames, Random, Printf

include("../src/DarwinScaffoldStudio/Science/GeodesicTortuosity.jl")
using .GeodesicTortuosity

# Load ground truth
df = CSV.read("data/soil_pore_space/characteristics.csv", DataFrame)
println("Total samples: ", nrow(df))

# Test on random samples
Random.seed!(42)
n_test = parse(Int, get(ARGS, 1, "30"))
indices = randperm(nrow(df))[1:n_test]

errors = Float64[]
rel_errors = Float64[]

println("\n=== VALIDATION ON $n_test SAMPLES ===\n")

for (i, idx) in enumerate(indices)
    row = df[idx, :]
    filepath = joinpath("data/soil_pore_space", row.file)

    if !isfile(filepath)
        @printf("[%2d] File not found\n", i)
        continue
    end

    try
        img = TiffImages.load(filepath)
        binary = BitArray(Float64.(img) .> 0.5)

        gt_tort = row["mean geodesic tortuosity"]
        result = compute_geodesic_tortuosity(binary, direction=:z, n_samples=50)

        abs_err = abs(result.mean - gt_tort)
        rel_err = abs_err / gt_tort * 100

        push!(errors, abs_err)
        push!(rel_errors, rel_err)

        status = rel_err < 5.0 ? "OK" : "MISS"
        @printf("[%2d] pred=%.4f, gt=%.4f, err=%.2f%% [%s]\n", i, result.mean, gt_tort, rel_err, status)
    catch e
        @printf("[%2d] Error: %s\n", i, string(e)[1:min(50,length(string(e)))])
    end
end

println("\n" * "="^55)
println("VALIDATION SUMMARY (n=$(length(errors)))")
println("="^55)
@printf("MAE (Mean Absolute Error): %.4f\n", mean(errors))
@printf("MRE (Mean Relative Error): %.2f%%\n", mean(rel_errors))
@printf("Max Error: %.2f%%\n", maximum(rel_errors))
println()
within_1 = count(x -> x < 1.0, rel_errors)
within_5 = count(x -> x < 5.0, rel_errors)
within_10 = count(x -> x < 10.0, rel_errors)
n = length(rel_errors)
@printf("Within 1%%:  %3d/%d (%.1f%%)\n", within_1, n, within_1/n*100)
@printf("Within 5%%:  %3d/%d (%.1f%%)\n", within_5, n, within_5/n*100)
@printf("Within 10%%: %3d/%d (%.1f%%)\n", within_10, n, within_10/n*100)
println()

if within_5/n >= 0.95
    println("TARGET ACHIEVED: >95% within 5% error")
else
    @printf("TARGET NOT MET: %.1f%% within 5%% (need 95%%)\n", within_5/n*100)
end
