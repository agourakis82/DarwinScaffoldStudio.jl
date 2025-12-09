#!/usr/bin/env julia
"""
FAST REAL DATA VALIDATION
Focuses on Pore Space 3D dataset with ground truth
"""

println("="^70)
println("REAL DATA VALIDATION - FAST VERSION")
println("="^70)

const PROJECT_ROOT = dirname(dirname(@__FILE__))
const DATA_DIR = joinpath(PROJECT_ROOT, "data/real_datasets")

using Statistics
using Printf
using CSV
using DataFrames
using TiffImages
using Dates

# Metrics
compute_porosity(binary) = 1.0 - sum(binary) / length(binary)
compute_tortuosity_ga(binary) = 1.0 + 0.5 * (sum(binary) / length(binary))

# Statistics
pearson_r(x, y) = cor(x, y)
rmse(p, a) = sqrt(mean((p .- a).^2))
mae(p, a) = mean(abs.(p .- a))

# Load ground truth
gt_df = CSV.read(joinpath(DATA_DIR, "pore_characteristics.csv"), DataFrame)
println("\nGround truth: $(nrow(gt_df)) samples from Zenodo 7516228")
println("Paper: 'Quantifying the impact of 3D pore space morphology...'")

# Validate
n_val = 300
println("\nValidating $n_val samples...")

por_gt, por_comp = Float64[], Float64[]
tort_gt, tort_comp = Float64[], Float64[]
tiff_base = joinpath(DATA_DIR, "pore_space_3d")

for (i, row) in enumerate(eachrow(gt_df[1:n_val, :]))
    tiff_path = joinpath(tiff_base, row.file)
    isfile(tiff_path) || continue

    try
        img = TiffImages.load(tiff_path)
        ndims(img) == 3 || continue
        binary = img .> 0

        push!(por_gt, row.porosity)
        push!(por_comp, compute_porosity(binary))
        push!(tort_gt, row[Symbol("mean geodesic tortuosity")])
        push!(tort_comp, compute_tortuosity_ga(binary))

        i % 100 == 0 && println("  $i/$n_val done...")
    catch; end
end

n = length(por_gt)
println("\nValidated: $n samples")

# Results
println("\n" * "="^70)
println("POROSITY VALIDATION (N=$n)")
println("="^70)
r_por = pearson_r(por_comp, por_gt)
println("Pearson r: $(@sprintf("%.4f", r_por))")
println("R²: $(@sprintf("%.4f", r_por^2))")
println("RMSE: $(@sprintf("%.6f", rmse(por_comp, por_gt)))")
println("MAE: $(@sprintf("%.6f", mae(por_comp, por_gt)))")

println("\n" * "="^70)
println("TORTUOSITY VALIDATION (N=$n)")
println("="^70)
r_tort = pearson_r(tort_comp, tort_gt)
println("Pearson r: $(@sprintf("%.4f", r_tort))")
println("R²: $(@sprintf("%.4f", r_tort^2))")
println("RMSE: $(@sprintf("%.4f", rmse(tort_comp, tort_gt)))")
println("Note: Gibson-Ashby approximation vs geodesic measurement")

# Save report
report = """
# Real Data Validation Report

**Generated:** $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
**Dataset:** Pore Space 3D (Zenodo 7516228)
**Paper:** "Quantifying the impact of 3D pore space morphology on diffusive mass transport in loam and sand"
**Total Available:** $(nrow(gt_df)) samples
**Validated:** $n samples

## Porosity Validation

| Metric | Value |
|--------|-------|
| Pearson r | $(@sprintf("%.4f", r_por)) |
| R² | $(@sprintf("%.4f", r_por^2)) |
| RMSE | $(@sprintf("%.6f", rmse(por_comp, por_gt))) |
| MAE | $(@sprintf("%.6f", mae(por_comp, por_gt))) |

**Ground Truth Range:** $(@sprintf("%.3f", minimum(por_gt))) - $(@sprintf("%.3f", maximum(por_gt)))
**Computed Range:** $(@sprintf("%.3f", minimum(por_comp))) - $(@sprintf("%.3f", maximum(por_comp)))

## Tortuosity Validation

| Metric | Value |
|--------|-------|
| Pearson r | $(@sprintf("%.4f", r_tort)) |
| R² | $(@sprintf("%.4f", r_tort^2)) |
| RMSE | $(@sprintf("%.4f", rmse(tort_comp, tort_gt))) |

**Note:** Darwin uses Gibson-Ashby approximation (τ = 1 + 0.5ρ), ground truth uses geodesic tortuosity from actual path tracing.

## Conclusion

$(r_por^2 > 0.99 ? "✅ **EXCELLENT**: Porosity measurement achieves R² > 0.99 against published ground truth" : r_por^2 > 0.95 ? "✅ **GOOD**: Porosity measurement achieves R² > 0.95" : "⚠️ Review needed")

$(r_tort^2 > 0.5 ? "✅ Tortuosity approximation shows significant correlation (r² = $(@sprintf("%.2f", r_tort^2))) with geodesic measurements" : "⚠️ Tortuosity approximation differs from geodesic method")

## Reference

Prifling B, Röding M, Townsend P, et al. (2023). Quantifying the impact of 3D pore space morphology on diffusive mass transport in loam and sand. Zenodo. https://doi.org/10.5281/zenodo.7516228
"""

mkpath(joinpath(PROJECT_ROOT, "results"))
open(joinpath(PROJECT_ROOT, "results/real_data_validation.md"), "w") do io
    write(io, report)
end

println("\n✅ Report saved: results/real_data_validation.md")
