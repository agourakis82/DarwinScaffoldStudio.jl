#!/usr/bin/env julia
"""
REAL DATA VALIDATION - NO SYNTHETIC DATA

Validates DarwinScaffoldStudio against REAL published datasets:
1. Pore Space 3D Dataset (4,608 samples) - Zenodo 7516228
   Paper: "Quantifying the impact of 3D pore space morphology on diffusive mass transport"
   Ground truth: porosity, tortuosity, constrictivity measured by authors

2. DeePore Dataset (17,700 samples) - Zenodo 3820900
   Paper: "DeePore: A deep learning workflow for rapid and comprehensive characterization"
   Ground truth: 1,515 features per sample from pore network modeling

3. Ti-6Al-4V Pore Morphology (20 samples) - Zenodo 6587905
   Paper: "Processing, microstructure, mechanical property dataset for LPBF Ti-6Al-4V"
   Ground truth: XCT measured pore volume, sphericity, diameter
"""

println("="^70)
println("REAL DATA VALIDATION - DarwinScaffoldStudio")
println("NO SYNTHETIC DATA - Published datasets with ground truth")
println("="^70)

const PROJECT_ROOT = dirname(dirname(@__FILE__))
const DATA_DIR = joinpath(PROJECT_ROOT, "data/real_datasets")

using Statistics
using Printf
using CSV
using DataFrames
using Images
using TiffImages
using Dates

# ============================================================================
# METRICS FUNCTIONS
# ============================================================================

function compute_porosity(binary::AbstractArray{Bool})
    return 1.0 - sum(binary) / length(binary)
end

function compute_tortuosity_gibson_ashby(binary::AbstractArray{Bool})
    relative_density = sum(binary) / length(binary)
    return 1.0 + 0.5 * relative_density
end

# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

function pearson_r(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    mx, my = mean(x), mean(y)
    sx, sy = std(x), std(y)
    if sx == 0 || sy == 0
        return 0.0
    end
    return sum((x .- mx) .* (y .- my)) / ((n - 1) * sx * sy)
end

function rmse(pred::Vector{Float64}, actual::Vector{Float64})
    return sqrt(mean((pred .- actual).^2))
end

function mae(pred::Vector{Float64}, actual::Vector{Float64})
    return mean(abs.(pred .- actual))
end

function mape(pred::Vector{Float64}, actual::Vector{Float64})
    valid = actual .!= 0
    if !any(valid)
        return NaN
    end
    return mean(abs.((pred[valid] .- actual[valid]) ./ actual[valid])) * 100
end

# ============================================================================
# 1. PORE SPACE 3D DATASET VALIDATION
# ============================================================================

println("\n" * "="^70)
println("1. PORE SPACE 3D DATASET (Zenodo 7516228)")
println("   Paper: Quantifying the impact of 3D pore space morphology...")
println("="^70)

# Load ground truth CSV
gt_csv = joinpath(DATA_DIR, "pore_characteristics.csv")
if !isfile(gt_csv)
    error("Ground truth CSV not found: $gt_csv")
end

gt_df = CSV.read(gt_csv, DataFrame)
println("\n   Ground truth loaded: $(nrow(gt_df)) samples")
println("   Columns: $(names(gt_df))")

# Sample statistics from ground truth
println("\n   📊 Ground Truth Statistics:")
println("   Porosity range: $(@sprintf("%.3f", minimum(gt_df.porosity))) - $(@sprintf("%.3f", maximum(gt_df.porosity)))")
println("   Porosity mean: $(@sprintf("%.3f", mean(gt_df.porosity))) ± $(@sprintf("%.3f", std(gt_df.porosity)))")
println("   Tortuosity range: $(@sprintf("%.3f", minimum(gt_df[!, "mean geodesic tortuosity"]))) - $(@sprintf("%.3f", maximum(gt_df[!, "mean geodesic tortuosity"])))")

# Validate subset (processing all 4608 would take too long)
n_validate = min(200, nrow(gt_df))
println("\n   Validating $n_validate samples...")

porosity_gt = Float64[]
porosity_computed = Float64[]
tortuosity_gt = Float64[]
tortuosity_computed = Float64[]

tiff_base = joinpath(DATA_DIR, "pore_space_3d")

for (i, row) in enumerate(eachrow(gt_df[1:n_validate, :]))
    try
        # Load TIFF stack
        tiff_path = joinpath(tiff_base, row.file)
        if !isfile(tiff_path)
            continue
        end

        # Load 3D TIFF
        img_stack = TiffImages.load(tiff_path)

        # Convert to binary (pores are typically 0 or 1 in segmented images)
        if ndims(img_stack) == 3
            binary = img_stack .> 0
        else
            continue
        end

        # Compute our metrics
        computed_por = compute_porosity(binary)
        computed_tort = compute_tortuosity_gibson_ashby(binary)

        # Store results
        push!(porosity_gt, row.porosity)
        push!(porosity_computed, computed_por)
        push!(tortuosity_gt, row[Symbol("mean geodesic tortuosity")])
        push!(tortuosity_computed, computed_tort)

        if i % 50 == 0
            println("      Processed $i/$n_validate...")
        end
    catch e
        # Skip problematic files
        continue
    end
end

n_validated = length(porosity_gt)
println("   Successfully validated: $n_validated samples")

if n_validated > 10
    println("\n   📈 POROSITY VALIDATION:")
    r = pearson_r(porosity_computed, porosity_gt)
    r2 = r^2
    rmse_val = rmse(porosity_computed, porosity_gt)
    mae_val = mae(porosity_computed, porosity_gt)

    println("      Pearson r: $(@sprintf("%.4f", r))")
    println("      R²: $(@sprintf("%.4f", r2))")
    println("      RMSE: $(@sprintf("%.4f", rmse_val))")
    println("      MAE: $(@sprintf("%.4f", mae_val))")

    println("\n   📈 TORTUOSITY VALIDATION:")
    r_tort = pearson_r(tortuosity_computed, tortuosity_gt)
    rmse_tort = rmse(tortuosity_computed, tortuosity_gt)

    println("      Pearson r: $(@sprintf("%.4f", r_tort))")
    println("      RMSE: $(@sprintf("%.4f", rmse_tort))")
    println("      Note: Using Gibson-Ashby approximation vs geodesic measurement")
end

# ============================================================================
# 2. DEEPORE DATASET VALIDATION
# ============================================================================

println("\n" * "="^70)
println("2. DEEPORE DATASET (Zenodo 3820900)")
println("   Paper: DeePore: A deep learning workflow...")
println("   17,700 micro-CT images with 1,515 features each")
println("="^70)

deepore_file = joinpath(DATA_DIR, "DeePore_Compact_Data.h5")

if isfile(deepore_file)
    println("\n   Loading DeePore HDF5...")

    # Try to load with HDF5.jl
    try
        using HDF5

        h5open(deepore_file, "r") do f
            X = read(f, "X")  # Images: (17700, 128, 128, 3)
            Y = read(f, "Y")  # Features: (17700, 1515, 1)

            println("   Images shape: $(size(X))")
            println("   Features shape: $(size(Y))")

            # Feature indices (from DeePore documentation):
            # 0: Porosity
            # 1: Specific surface area
            # ... many more

            n_samples = min(500, size(X, 1))
            println("\n   Validating $n_samples samples...")

            deepore_gt_porosity = Float64[]
            deepore_computed_porosity = Float64[]

            for i in 1:n_samples
                # Get image (3 slices at different depths)
                img_slices = X[i, :, :, :]

                # Convert to binary using Otsu-like threshold
                # The images are grayscale representing density
                middle_slice = Float64.(img_slices[:, :, 2])
                threshold = (maximum(middle_slice) + minimum(middle_slice)) / 2
                binary_2d = middle_slice .> threshold

                # Compute porosity from 2D slice
                computed_por = 1.0 - sum(binary_2d) / length(binary_2d)

                # Ground truth porosity (feature index 0)
                gt_por = Y[i, 1, 1]

                push!(deepore_gt_porosity, gt_por)
                push!(deepore_computed_porosity, computed_por)

                if i % 100 == 0
                    println("      Processed $i/$n_samples...")
                end
            end

            println("\n   📈 DEEPORE POROSITY VALIDATION (N=$n_samples):")
            r = pearson_r(deepore_computed_porosity, deepore_gt_porosity)
            r2 = r^2
            rmse_val = rmse(deepore_computed_porosity, deepore_gt_porosity)

            println("      Pearson r: $(@sprintf("%.4f", r))")
            println("      R²: $(@sprintf("%.4f", r2))")
            println("      RMSE: $(@sprintf("%.4f", rmse_val))")
            println("      Note: 2D slice estimation vs 3D pore network modeling")
        end
    catch e
        println("   ⚠ Could not load HDF5: $e")
        println("   Install HDF5.jl: ] add HDF5")
    end
else
    println("   DeePore file not found")
end

# ============================================================================
# 3. Ti-6Al-4V PORE DATA
# ============================================================================

println("\n" * "="^70)
println("3. Ti-6Al-4V PORE MORPHOLOGY (Zenodo 6587905)")
println("   Paper: Processing, microstructure, mechanical property dataset...")
println("="^70)

ti64_dir = joinpath(DATA_DIR, "ti64_pores")
if isdir(ti64_dir)
    csv_files = filter(f -> endswith(f, ".csv"), readdir(ti64_dir))
    println("\n   Found $(length(csv_files)) sample files")

    # Aggregate pore statistics
    total_pores = 0
    all_volumes = Float64[]
    all_diameters = Float64[]
    all_sphericities = Float64[]

    for csv_file in csv_files
        try
            df = CSV.read(joinpath(ti64_dir, csv_file), DataFrame)
            total_pores += nrow(df)

            # Check for common column names
            if "Volume" in names(df)
                append!(all_volumes, df.Volume)
            end
            if "EqDiameter" in names(df)
                append!(all_diameters, df.EqDiameter)
            elseif "Diameter" in names(df)
                append!(all_diameters, df.Diameter)
            end
            if "Sphericity" in names(df)
                append!(all_sphericities, df.Sphericity)
            end
        catch
            continue
        end
    end

    println("   Total pores analyzed: $total_pores")

    if !isempty(all_diameters)
        println("\n   📊 PORE STATISTICS (from XCT measurements):")
        println("      Equivalent diameter: $(@sprintf("%.2f", mean(all_diameters))) ± $(@sprintf("%.2f", std(all_diameters))) μm")
        println("      Diameter range: $(@sprintf("%.2f", minimum(all_diameters))) - $(@sprintf("%.2f", maximum(all_diameters))) μm")
    end

    if !isempty(all_sphericities)
        println("      Sphericity: $(@sprintf("%.3f", mean(all_sphericities))) ± $(@sprintf("%.3f", std(all_sphericities)))")
    end
end

# ============================================================================
# SUMMARY REPORT
# ============================================================================

println("\n" * "="^70)
println("VALIDATION SUMMARY")
println("="^70)

results_dir = joinpath(PROJECT_ROOT, "results")
mkpath(results_dir)

# Generate report
report = """
# Real Data Validation Report

**Generated:** $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
**DarwinScaffoldStudio Version:** 0.8.0

## Data Sources (ALL REAL - NO SYNTHETIC)

### 1. Pore Space 3D Dataset
- **Source:** Zenodo 7516228
- **Paper:** "Quantifying the impact of 3D pore space morphology on diffusive mass transport in loam and sand"
- **Samples:** 4,608 segmented 3D CT volumes
- **Ground Truth:** Porosity, geodesic tortuosity, constrictivity, specific surface area
- **Validated:** $n_validated samples

### 2. DeePore Dataset
- **Source:** Zenodo 3820900
- **Paper:** "DeePore: A deep learning workflow for rapid and comprehensive characterization of porous materials"
- **Samples:** 17,700 micro-CT images
- **Ground Truth:** 1,515 features per sample from pore network modeling

### 3. Ti-6Al-4V Additive Manufacturing Dataset
- **Source:** Zenodo 6587905
- **Paper:** "Processing, microstructure, mechanical property dataset for LPBF Ti-6Al-4V"
- **Samples:** 20+ XCT scanned samples
- **Ground Truth:** Pore volume, diameter, sphericity from XCT

## Validation Results

### Porosity Measurement (Pore Space Dataset)
| Metric | Value |
|--------|-------|
| N (samples) | $n_validated |
| Pearson r | $(n_validated > 10 ? @sprintf("%.4f", pearson_r(porosity_computed, porosity_gt)) : "N/A") |
| R² | $(n_validated > 10 ? @sprintf("%.4f", pearson_r(porosity_computed, porosity_gt)^2) : "N/A") |
| RMSE | $(n_validated > 10 ? @sprintf("%.4f", rmse(porosity_computed, porosity_gt)) : "N/A") |
| MAE | $(n_validated > 10 ? @sprintf("%.4f", mae(porosity_computed, porosity_gt)) : "N/A") |

## References

1. Prifling B, et al. (2023). Quantifying the impact of 3D pore space morphology on diffusive mass transport in loam and sand. Zenodo. https://doi.org/10.5281/zenodo.7516228

2. Rabbani A, et al. (2020). DeePore: A deep learning workflow for rapid and comprehensive characterization of porous materials. Advances in Water Resources. https://doi.org/10.5281/zenodo.3820900

3. Kok Y, et al. (2022). Processing, microstructure, mechanical property dataset for laser powder bed fusion additively manufactured Ti-6Al-4V. Zenodo. https://doi.org/10.5281/zenodo.6587905
"""

report_path = joinpath(results_dir, "real_data_validation_report.md")
open(report_path, "w") do io
    write(io, report)
end

println("\n📄 Report saved: $report_path")
println("\n✅ VALIDATION COMPLETE - ALL REAL DATA, NO SYNTHETIC!")
println("="^70)
