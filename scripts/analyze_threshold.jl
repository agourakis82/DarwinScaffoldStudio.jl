#!/usr/bin/env julia
"""
Analyze optimal threshold for pore detection on PoreScript images.
"""

using Pkg
Pkg.activate(".")

using Images
using Statistics
using Printf

DATA_DIR = joinpath(@__DIR__, "..", "data", "validation", "porescript")
PIXEL_SIZE_UM = 3.5

function test_threshold(image_path::String, threshold::Float64; scale::Int=4)
    img = load(image_path)
    gray = Gray.(img)
    img_array = Float64.(gray)

    if scale > 1
        h, w = size(img_array)
        new_h, new_w = h ÷ scale, w ÷ scale
        img_small = imresize(img_array, (new_h, new_w))
    else
        img_small = img_array
    end

    # Apply threshold
    pore_mask = img_small .< threshold

    # Label connected components
    labels = label_components(pore_mask)
    n_pores = maximum(labels)

    if n_pores == 0
        return 0.0, 0, 0.0
    end

    # Calculate pore sizes
    pore_diameters = Float64[]
    for i in 1:n_pores
        area = sum(labels .== i)
        if area > 10
            diameter = 2.0 * sqrt(area / π)
            push!(pore_diameters, diameter)
        end
    end

    if isempty(pore_diameters)
        return 0.0, 0, 0.0
    end

    effective_pixel_size = PIXEL_SIZE_UM * scale
    pore_diameters_um = pore_diameters .* effective_pixel_size

    # Porosity (fraction of pores)
    porosity = sum(pore_mask) / length(pore_mask) * 100

    return mean(pore_diameters_um), length(pore_diameters), porosity
end

println("Threshold Analysis for PoreScript SEM Images")
println("=" ^ 60)
println("\nGround truth mean: ~175 μm")
println("Target: Find threshold that gives ~175 μm mean pore size\n")

image_path = joinpath(DATA_DIR, "S1_27x.tif")

println("Sample: S1_27x.tif")
println("-" ^ 40)
@printf("%-12s %10s %8s %10s\n", "Threshold", "Mean (μm)", "n_pores", "Porosity%")
println("-" ^ 40)

for threshold in 0.3:0.05:0.7
    mean_size, n_pores, porosity = test_threshold(image_path, threshold)
    @printf("%-12.2f %10.1f %8d %10.1f\n", threshold, mean_size, n_pores, porosity)
end

println("\nOptimizing threshold...")
best_threshold = 0.5
best_error = Inf
gt_mean = 175.0

for threshold in 0.30:0.01:0.70
    mean_size, _, _ = test_threshold(image_path, threshold)
    error = abs(mean_size - gt_mean)
    if error < best_error
        best_error = error
        best_threshold = threshold
    end
end

mean_size, n_pores, porosity = test_threshold(image_path, best_threshold)
@printf("\nOptimal threshold: %.2f\n", best_threshold)
@printf("Mean pore size: %.1f μm (error: %.1f%%)\n", mean_size, abs(mean_size - gt_mean)/gt_mean*100)
@printf("Number of pores: %d\n", n_pores)
@printf("Porosity: %.1f%%\n", porosity)
