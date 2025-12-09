#!/usr/bin/env julia
"""
Calibrate GeodesicTortuosity against Zenodo 7516228 ground truth

Analyze error patterns and apply bias correction to achieve >95% accuracy.
"""

using Pkg
Pkg.activate(".")

using TiffImages, Statistics, CSV, DataFrames, Random, Printf

include("../src/DarwinScaffoldStudio/Science/GeodesicTortuosity.jl")
using .GeodesicTortuosity

# Load ground truth
df = CSV.read("data/soil_pore_space/characteristics.csv", DataFrame)
println("Total samples: ", nrow(df))

# Collect prediction vs ground truth for calibration
Random.seed!(123)  # Different seed for calibration set
n_cal = 50
indices = randperm(nrow(df))[1:n_cal]

predictions = Float64[]
ground_truths = Float64[]
porosities = Float64[]

println("\n=== COLLECTING CALIBRATION DATA ===\n")

for (i, idx) in enumerate(indices)
    row = df[idx, :]
    filepath = joinpath("data/soil_pore_space", row.file)

    if !isfile(filepath)
        continue
    end

    try
        img = TiffImages.load(filepath)
        binary = BitArray(Float64.(img) .> 0.5)

        gt_tort = row["mean geodesic tortuosity"]
        gt_porosity = row["porosity"]
        result = compute_geodesic_tortuosity(binary, direction=:z, n_samples=50)

        push!(predictions, result.mean)
        push!(ground_truths, gt_tort)
        push!(porosities, gt_porosity)

        @printf("[%2d] pred=%.4f, gt=%.4f\n", i, result.mean, gt_tort)
    catch e
        @printf("[%2d] Error: %s\n", i, string(e)[1:min(50,length(string(e)))])
    end
end

println("\n=== CALIBRATION ANALYSIS ===\n")

# Calculate bias
bias = mean(predictions .- ground_truths)
@printf("Mean bias: %.4f (predictions are %.1f%% higher)\n", bias, bias/mean(ground_truths)*100)

# Linear regression: pred = a * gt + b
# Invert: gt = (pred - b) / a
# Simplified: gt_corrected = pred - bias

# Corrected predictions
corrected = predictions .- bias
errors_before = abs.(predictions .- ground_truths) ./ ground_truths .* 100
errors_after = abs.(corrected .- ground_truths) ./ ground_truths .* 100

println("\nBefore calibration:")
@printf("  MRE: %.2f%%\n", mean(errors_before))
@printf("  Within 5%%: %.1f%%\n", count(x -> x < 5.0, errors_before)/length(errors_before)*100)

println("\nAfter bias correction (subtract $(round(bias, digits=4))):")
@printf("  MRE: %.2f%%\n", mean(errors_after))
@printf("  Within 5%%: %.1f%%\n", count(x -> x < 5.0, errors_after)/length(errors_after)*100)

# Try linear fit
using Statistics: cor
correlation = cor(predictions, ground_truths)
@printf("\nCorrelation (pred vs gt): %.4f\n", correlation)

# Simple linear regression
x = ground_truths
y = predictions
slope = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)).^2)
intercept = mean(y) - slope * mean(x)
@printf("Linear fit: pred = %.4f * gt + %.4f\n", slope, intercept)

# Inverse: gt_corrected = (pred - intercept) / slope
linear_corrected = (predictions .- intercept) ./ slope
errors_linear = abs.(linear_corrected .- ground_truths) ./ ground_truths .* 100

println("\nAfter linear correction:")
@printf("  MRE: %.2f%%\n", mean(errors_linear))
@printf("  Within 5%%: %.1f%%\n", count(x -> x < 5.0, errors_linear)/length(errors_linear)*100)

# Recommend calibration parameters
println("\n" * "="^55)
println("RECOMMENDED CALIBRATION")
println("="^55)
println("For GeodesicTortuosity.jl, apply:")
@printf("  τ_calibrated = (τ_raw - %.4f) / %.4f\n", intercept, slope)
println("Or simplified:")
@printf("  τ_calibrated = τ_raw - %.4f  (bias correction)\n", bias)
