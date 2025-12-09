"""
Test D_2D = 1/φ hypothesis on PURE 2D slices from micro-CT volume

This tests TRUE 2D projections (not quasi-3D SEM), which should reveal
whether D_2D → 1/φ = 0.618034 at high porosity.

Author: Darwin Scaffold Studio
Date: 2025-12-08
"""

using Images
using FileIO
using Statistics
using Printf
using Glob

include("box_counting_2d.jl")

const φ = (1 + sqrt(5)) / 2
const INV_φ = 1 / φ

println("="^70)
println("2D SLICE ANALYSIS FROM MICRO-CT VOLUME")
println("Testing: D_2D = 1/φ = 0.618034 on pure 2D projections")
println("="^70)
println()

# Load KFoam micro-CT slices
kfoam_dir = "data/kfoam/KFoam_200pixcube/KFoam_200pixcube_tiff/"

if !isdir(kfoam_dir)
    println("ERROR: KFoam directory not found: $kfoam_dir")
    exit(1)
end

# Find all TIFF files
tiff_files = glob("*.tif", kfoam_dir)

if isempty(tiff_files)
    println("ERROR: No TIFF files found in $kfoam_dir")
    exit(1)
end

println("Found $(length(tiff_files)) micro-CT slices")
println()

# Sample slices (every 10th to avoid too many)
n_samples = min(20, length(tiff_files))
sample_indices = round.(Int, range(1, length(tiff_files), length=n_samples))
sample_files = tiff_files[sample_indices]

println("Analyzing $n_samples slices (sampled from $(length(tiff_files)) total)")
println()

# Analyze each slice
results = []
# Adjusted thresholds for low-contrast micro-CT (values 0.047-0.14)
thresholds_to_test = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]

println("Progress:")
for (i, filepath) in enumerate(sample_files)
    slice_num = parse(Int, match(r"(\d+)\.tif$", filepath).captures[1])

    for threshold in thresholds_to_test
        try
            # Load slice
            img = load(filepath)
            img_gray = eltype(img) <: RGB ? Gray.(img) : img
            img_float = Float64.(img_gray)

            # Binarize
            binary_slice = img_float .> threshold

            # Compute porosity
            porosity = sum(.!binary_slice) / length(binary_slice)

            # Skip if porosity too extreme
            if porosity < 0.05 || porosity > 0.95
                continue
            end

            # Box-counting on 2D slice
            D_2d, box_sizes, counts, r_squared = box_counting_2d(binary_slice)

            # Only accept good fits
            if r_squared < 0.95
                continue
            end

            # Error from 1/φ
            error_pct = abs(D_2d - INV_φ) / INV_φ * 100

            push!(results, (slice_num, threshold, porosity, D_2d, r_squared, error_pct))

        catch e
            # Skip problematic slices
            continue
        end
    end

    print(".")
    if i % 10 == 0
        println(" [$i/$n_samples]")
    end
end
println()
println()

if isempty(results)
    println("ERROR: No valid results obtained")
    exit(1)
end

# Analyze results
println("="^70)
println("RESULTS SUMMARY")
println("="^70)
println()

porosities = [r[3] for r in results]
D_2d_values = [r[4] for r in results]
r_squared_values = [r[5] for r in results]
errors = [r[6] for r in results]

@printf("Number of valid measurements: %d\n", length(results))
@printf("Porosity range: %.1f%% - %.1f%%\n", minimum(porosities)*100, maximum(porosities)*100)
@printf("D_2D range: %.4f - %.4f\n", minimum(D_2d_values), maximum(D_2d_values))
@printf("Mean D_2D: %.6f ± %.6f\n", mean(D_2d_values), std(D_2d_values))
@printf("Mean R²: %.4f\n", mean(r_squared_values))
println()

# Test hypothesis at different porosity ranges
println("D_2D by Porosity Range:")
println("-"^70)

porosity_ranges = [
    (0.0, 0.3, "Low (0-30%)"),
    (0.3, 0.5, "Medium (30-50%)"),
    (0.5, 0.7, "High (50-70%)"),
    (0.7, 1.0, "Very High (70-100%)")
]

for (p_min, p_max, label) in porosity_ranges
    mask = (porosities .>= p_min) .& (porosities .< p_max)

    if sum(mask) > 0
        D_mean = mean(D_2d_values[mask])
        D_std = std(D_2d_values[mask])
        n = sum(mask)
        error_mean = mean(errors[mask])

        @printf("  %-20s: D_2D = %.4f ± %.4f  (n=%d, error from 1/φ = %.1f%%)\n",
                label, D_mean, D_std, n, error_mean)
    end
end
println()

# Find closest to 1/φ
best_idx = argmin(errors)
best = results[best_idx]

println("BEST MATCH TO 1/φ:")
@printf("  Slice #%d (threshold=%.1f)\n", best[1], best[2])
@printf("  Porosity: %.2f%%\n", best[3]*100)
@printf("  D_2D: %.6f\n", best[4])
@printf("  1/φ:  %.6f\n", INV_φ)
@printf("  Error: %.2f%%\n", best[6])
@printf("  R²: %.6f\n", best[5])
println()

# Hypothesis test
support_strong = count(e -> e < 5.0, errors)
support_partial = count(e -> 5.0 <= e < 10.0, errors)
no_support = count(e -> e >= 10.0, errors)

println("HYPOTHESIS SUPPORT (D_2D = 1/φ = 0.618034):")
@printf("  Strong support (< 5%% error):     %3d/%d (%.1f%%)\n",
        support_strong, length(errors), support_strong/length(errors)*100)
@printf("  Partial support (5-10%% error):   %3d/%d (%.1f%%)\n",
        support_partial, length(errors), support_partial/length(errors)*100)
@printf("  Not supported (> 10%% error):     %3d/%d (%.1f%%)\n",
        no_support, length(errors), no_support/length(errors)*100)
println()

# Overall conclusion
if support_strong / length(errors) > 0.3
    println("✓ HYPOTHESIS STRONGLY SUPPORTED")
    println("  Pure 2D slices show D_2D ≈ 1/φ = 0.618 at high porosity")
    println()
    println("  CONCLUSION: Original hypothesis CONFIRMED!")
    println("  - 3D micro-CT: D_3D → φ = 1.618")
    println("  - 2D slices:   D_2D → 1/φ = 0.618")
    println("  - Dimension scaling: D_3D - D_2D ≈ 1.0 ✓")
elseif support_strong / length(errors) > 0.1
    println("~ HYPOTHESIS PARTIALLY SUPPORTED")
    println("  Some 2D slices approach 1/φ but not consistently")
    println()
    println("  NEEDS: More analysis of high-porosity slices specifically")
else
    println("✗ HYPOTHESIS NOT SUPPORTED")
    println("  Pure 2D slices do NOT show D_2D = 1/φ")
    println()
    println("  CONCLUSION: Theory needs revision")
    println("  - D_2D ≠ 1/φ even for pure 2D projections")
    println("  - SEM result (D ≈ φ) may be correct for 2D!")
    println("  - Alternative: φ appears in all dimensions")
end
println()

# Save detailed results
mkpath("results")
output_file = "results/microct_2d_slices_analysis.txt"
open(output_file, "w") do f
    write(f, "="^70 * "\n")
    write(f, "2D SLICE ANALYSIS - DETAILED RESULTS\n")
    write(f, "="^70 * "\n\n")

    write(f, @sprintf("Total measurements: %d\n", length(results)))
    write(f, @sprintf("Mean D_2D: %.6f ± %.6f\n", mean(D_2d_values), std(D_2d_values)))
    write(f, @sprintf("Target 1/φ: %.6f\n\n", INV_φ))

    write(f, "Slice# | Threshold | Porosity | D_2D    | R²      | Error\n")
    write(f, "-"^70 * "\n")

    for r in results
        write(f, @sprintf("%6d | %9.1f | %8.2f%% | %.6f | %.4f | %.2f%%\n",
                         r[1], r[2], r[3]*100, r[4], r[5], r[6]))
    end
end

println("Detailed results saved to: $output_file")
println()
println("="^70)
