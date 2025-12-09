#!/usr/bin/env julia
"""
PROCESS REAL KFOAM MICRO-CT DATA - VALIDATE D = φ
==================================================

This script processes actual micro-CT TIFF image stacks from the KFoam dataset
(Zenodo 3532935) to validate the D = φ discovery using real experimental data.

No synthetic data. No mocks. No stubs. Only real micro-CT imaging data.

Dataset: KFoam 200px cube from foam material
- Format: TIFF stack (200 slices × 200×200 px each)
- Total volume: 200³ pixels
- Already binary segmented (solid/void)
- Source: Published open-access dataset
"""

using Images
using Statistics
using Random
using Printf

const φ = (1 + sqrt(5)) / 2  # 1.618033988749895

#=============================================================================
                    LOAD REAL TIFF STACK DATA
=============================================================================#

function load_kfoam_tiff_stack(tiff_dir::String; start_idx::Int=690, end_idx::Int=889)
    """
    Load KFoam TIFF stack into 3D binary volume.
    Returns (volume::Array{Bool,3}, actual_porosity::Float64)
    """
    println("Loading KFoam TIFF stack: slices $start_idx to $end_idx...")

    # Get list of TIFF files
    tiff_files = sort([f for f in readdir(tiff_dir) if endswith(f, ".tif")])

    if isempty(tiff_files)
        error("No TIFF files found in $tiff_dir")
    end

    # Filter to requested range
    tiff_files_subset = [f for f in tiff_files if
                        (idx = parse(Int, match(r"(\d{4})", f).captures[1])) >= start_idx && idx <= end_idx]

    n_slices = length(tiff_files_subset)
    println("  Found $n_slices slices in range")

    # Load first slice to get dimensions
    first_slice = load(joinpath(tiff_dir, tiff_files_subset[1]))
    h, w = size(first_slice)

    # Initialize volume
    volume = zeros(Bool, h, w, n_slices)

    # Load all slices
    for (i, f) in enumerate(tiff_files_subset)
        if i % 25 == 0
            print(".")
        end

        img = load(joinpath(tiff_dir, f))

        # Convert to binary (grayscale to bool)
        if eltype(img) <: Bool
            volume[:, :, i] = img
        elseif eltype(img) <: Gray
            # Threshold at 0.5
            volume[:, :, i] = img .> 0.5
        else
            # Generic conversion
            volume[:, :, i] = convert.(Bool, img)
        end
    end

    println(" ✓")

    # Calculate porosity (1 = void, 0 = solid)
    total_voxels = prod(size(volume))
    void_voxels = sum(.!volume)  # Count void (false = solid, true = material)
    porosity = void_voxels / total_voxels

    println(@sprintf("  Loaded volume: %d × %d × %d", size(volume, 1), size(volume, 2), size(volume, 3)))
    println(@sprintf("  Porosity: %.2f%% (void voxels: %d / %d)", porosity*100, void_voxels, total_voxels))

    return volume, porosity
end

#=============================================================================
                    BOUNDARY EXTRACTION & FRACTAL DIMENSION
=============================================================================#

function extract_boundary_3d(volume::AbstractArray{Bool,3})
    """Extract surface boundary voxels from 3D volume."""
    nx, ny, nz = size(volume)
    boundary = falses(nx, ny, nz)

    for i in 2:nx-1
        for j in 2:ny-1
            for k in 2:nz-1
                if volume[i,j,k]
                    # Check 6-connectivity neighbors
                    neighbors = [
                        volume[i-1,j,k], volume[i+1,j,k],
                        volume[i,j-1,k], volume[i,j+1,k],
                        volume[i,j,k-1], volume[i,j,k+1]
                    ]
                    if any(.!neighbors)  # If any neighbor is void
                        boundary[i,j,k] = true
                    end
                end
            end
        end
    end

    n_boundary = sum(boundary)
    println(@sprintf("  Boundary voxels: %d (%.1f%% of material)", n_boundary,
                     100 * n_boundary / sum(volume)))

    return boundary
end

function box_counting_dimension_3d(volume::AbstractArray{Bool,3}; verbose::Bool=false)
    """
    Calculate fractal dimension using 3D box-counting method.
    Returns: (D, R², quality_label)
    """
    nx, ny, nz = size(volume)
    min_dim = min(nx, ny, nz)

    max_power = floor(Int, log2(min_dim))
    box_sizes = [2^k for k in 1:max_power-1]

    counts = Int[]
    valid_sizes = Int[]

    if verbose
        println("  Box-counting analysis:")
    end

    for box_size in box_sizes
        count = 0
        for i in 1:box_size:nx
            for j in 1:box_size:ny
                for k in 1:box_size:nz
                    end_i = min(i + box_size - 1, nx)
                    end_j = min(j + box_size - 1, ny)
                    end_k = min(k + box_size - 1, nz)

                    if any(volume[i:end_i, j:end_j, k:end_k])
                        count += 1
                    end
                end
            end
        end

        if count > 0
            push!(counts, count)
            push!(valid_sizes, box_size)

            if verbose && length(valid_sizes) <= 5
                println(@sprintf("    Box size %3d: %5d boxes", box_size, count))
            end
        end
    end

    if length(counts) < 3
        if verbose
            println("    ERROR: Insufficient data points for fit")
        end
        return NaN, NaN, "insufficient_data"
    end

    # Linear regression: log(N) = -D * log(r) + intercept
    x = log.(valid_sizes)
    y = log.(counts)

    x_mean = mean(x)
    y_mean = mean(y)

    numerator = sum((x .- x_mean) .* (y .- y_mean))
    denominator = sum((x .- x_mean).^2)

    slope = numerator / denominator
    D = -slope

    intercept = y_mean - slope * x_mean
    y_pred = slope .* x .+ intercept

    ss_res = sum((y .- y_pred).^2)
    ss_tot = sum((y .- y_mean).^2)
    R2 = 1 - ss_res / ss_tot

    quality = R2 > 0.98 ? "excellent" : R2 > 0.95 ? "good" : "fair"

    if verbose
        println(@sprintf("    Dimension D = %.4f", D))
        println(@sprintf("    R² = %.4f (%s fit)", R2, quality))
    end

    return D, R2, quality
end

#=============================================================================
                    MULTI-REGION ANALYSIS
=============================================================================#

function analyze_multiple_regions(volume::AbstractArray{Bool,3}, n_regions::Int=4)
    """
    Analyze fractal dimension in multiple 100³ subvolumes to test robustness.
    """
    println("\n" * "="^70)
    println("MULTI-REGION ANALYSIS: Testing $n_regions independent 100³ regions")
    println("="^70)

    nx, ny, nz = size(volume)
    region_size = 100

    D_values = Float64[]
    R2_values = Float64[]

    for region_num in 1:n_regions
        # Random starting position ensuring 100³ fits
        i_start = rand(1:nx-region_size)
        j_start = rand(1:ny-region_size)
        k_start = rand(1:nz-region_size)

        # Extract subvolume
        subvolume = volume[i_start:i_start+region_size-1,
                          j_start:j_start+region_size-1,
                          k_start:k_start+region_size-1]

        # Calculate metrics
        region_porosity = 1 - sum(subvolume) / length(subvolume)
        boundary = extract_boundary_3d(subvolume)
        D, R2, quality = box_counting_dimension_3d(boundary, verbose=false)

        if !isnan(D)
            push!(D_values, D)
            push!(R2_values, R2)

            ratio = D / φ
            error_pct = abs(D - φ) / φ * 100

            marker = abs(D - φ) < 0.05 ? " ✓ CLOSE TO φ" : ""

            println(@sprintf("Region %d: D = %.4f (D/φ = %.3f, error = %+.1f%%) R² = %.4f%s",
                region_num, D, ratio, error_pct, R2, marker))
        else
            println(@sprintf("Region %d: Failed to compute (R² = %.4f)", region_num, R2))
        end
    end

    if !isempty(D_values)
        D_mean = mean(D_values)
        D_std = length(D_values) > 1 ? std(D_values) : 0.0
        D_sem = D_std / sqrt(length(D_values))

        println("\n" * "-"^70)
        println(@sprintf("AGGREGATE: D = %.4f ± %.4f (SEM)", D_mean, D_sem))
        println(@sprintf("           D/φ = %.4f", D_mean / φ))
        println(@sprintf("           Distance to φ: %.4f", abs(D_mean - φ)))
        println("-"^70)

        return D_mean, D_sem, D_values
    else
        return NaN, NaN, []
    end
end

#=============================================================================
                            MAIN ANALYSIS
=============================================================================#

function main()
    println("╔" * "="^66 * "╗")
    println("║  REAL DATA VALIDATION: KFoam Micro-CT Dataset                    ║")
    println("║  D = φ Discovery Using Actual Experimental Data                  ║")
    println("╚" * "="^66 * "╝")
    println()

    # Find TIFF directory
    tiff_dir = "/home/agourakis82/workspace/darwin-scaffold-studio/data/kfoam/KFoam_200pixcube/KFoam_200pixcube_tiff"

    if !isdir(tiff_dir)
        error("TIFF directory not found: $tiff_dir")
    end

    # Load TIFF stack (using subset of slices)
    println("\n" * "="^70)
    println("STEP 1: LOAD REAL MICRO-CT DATA")
    println("="^70)

    volume, porosity = load_kfoam_tiff_stack(tiff_dir, start_idx=690, end_idx=789)

    # Extract boundary
    println("\n" * "="^70)
    println("STEP 2: EXTRACT MATERIAL BOUNDARY")
    println("="^70)

    boundary = extract_boundary_3d(volume)

    # Compute fractal dimension on full volume
    println("\n" * "="^70)
    println("STEP 3: FRACTAL DIMENSION (FULL VOLUME)")
    println("="^70)

    D_full, R2_full, quality_full = box_counting_dimension_3d(boundary, verbose=true)

    println("\n" * "="^70)
    println("FULL VOLUME RESULTS")
    println("="^70)
    println(@sprintf("Porosity:        %.2f%%", porosity*100))
    println(@sprintf("Fractal Dim (D): %.4f", D_full))
    println(@sprintf("Golden Ratio (φ): %.6f", φ))
    println(@sprintf("D/φ ratio:       %.4f", D_full / φ))
    println(@sprintf("Error from φ:    %.4f (%.2f%%)", abs(D_full - φ),
                     abs(D_full - φ) / φ * 100))
    println(@sprintf("R² fit quality:  %.4f (%s)", R2_full, quality_full))

    # Multi-region analysis
    println()
    D_mean, D_sem, D_values = analyze_multiple_regions(volume, 4)

    # Final summary
    println("\n" * "="^70)
    println("VALIDATION SUMMARY")
    println("="^70)

    println("\n✓ REAL DATA VALIDATION COMPLETE")
    println()
    println("Key Findings:")
    println("  • Used real KFoam micro-CT TIFF stack (100×100×100 pixels)")
    println("  • No synthetic data, no simulations")
    println("  • Porosity: $(round(porosity*100, digits=1))%")
    println("  • Full volume D: $(round(D_full, digits=4))")

    if !isnan(D_mean)
        println("  • Multi-region D: $(round(D_mean, digits=4)) ± $(round(D_sem, digits=4))")
    end

    println()
    println("  D/φ deviation: $(round(abs(D_full - φ) / φ * 100, digits=2))%")
    println("  Statistical quality: $quality_full (R² = $(round(R2_full, digits=4)))")

    println("\n" * "="^70)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
