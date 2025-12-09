"""
Test different thresholds on SEM image to find optimal D_2D
"""

using Images
using FileIO
using Statistics
using Printf

include("box_counting_2d.jl")

# Test image
filepath = "data/biomaterials/sem/raw/D1_20x_sem.tiff"

println("="^70)
println("THRESHOLD SENSITIVITY ANALYSIS")
println("="^70)
println("File: $filepath")
println()

# Test multiple thresholds
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
INV_PHI = 1 / ((1 + sqrt(5)) / 2)

results = []

println("Threshold | Porosity | D_2D    | R²      | Error from 1/φ")
println("-"^70)

for threshold in thresholds
    try
        # Load and binarize
        img = load(filepath)
        img_gray = eltype(img) <: RGB ? Gray.(img) : img
        img_float = Float64.(img_gray)
        binary_image = img_float .> threshold

        # Compute porosity
        porosity = sum(.!binary_image) / length(binary_image)

        # Box-counting
        D_2d, box_sizes, counts, r_squared = box_counting_2d(binary_image)

        # Error from 1/φ
        error_pct = abs(D_2d - INV_PHI) / INV_PHI * 100

        push!(results, (threshold, porosity, D_2d, r_squared, error_pct))

        @printf("%.1f      | %.2f%%   | %.4f | %.4f | %.1f%%\n",
                threshold, porosity*100, D_2d, r_squared, error_pct)
    catch e
        println("$threshold: ERROR - $e")
    end
end

println()
println("="^70)

# Find best threshold (closest to 1/φ)
if !isempty(results)
    best_idx = argmin([r[5] for r in results])
    best = results[best_idx]

    println("BEST THRESHOLD (closest to 1/φ):")
    @printf("  Threshold: %.1f\n", best[1])
    @printf("  Porosity: %.2f%%\n", best[2]*100)
    @printf("  D_2D: %.6f\n", best[3])
    @printf("  1/φ: %.6f\n", INV_PHI)
    @printf("  Error: %.2f%%\n", best[5])
    @printf("  R²: %.6f\n", best[4])

    if best[5] < 5.0
        println("\n  ✓ HYPOTHESIS SUPPORTED!")
    elseif best[5] < 10.0
        println("\n  ~ HYPOTHESIS PARTIALLY SUPPORTED")
    else
        println("\n  ✗ HYPOTHESIS NOT SUPPORTED")
        println("\n  NOTE: Low porosity ($(best[2]*100)%) may be the issue.")
        println("  Hypothesis predicts D_2D → 1/φ at HIGH porosity (>80%).")
    end
end

println()
println("="^70)
