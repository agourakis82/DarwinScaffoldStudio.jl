#!/usr/bin/env julia
"""
Analyze DeePore Dataset for D = φ Validation
=============================================

DeePore dataset contains 17,700 3D porous media images with properties.
We'll extract porosity and compute fractal dimension to test our prediction:

    D = φ at ~95.8% porosity

This provides REAL DATA validation of our computational finding.
"""

using HDF5
using Statistics
using Printf
using Random

const φ = (1 + sqrt(5)) / 2  # 1.618...
const DATA_PATH = joinpath(@__DIR__, "..", "data", "deepore", "DeePore_Compact_Data.h5")

#=============================================================================
                        DATA LOADING
=============================================================================#

function load_deepore_data()
    println("Loading DeePore dataset...")

    if !isfile(DATA_PATH)
        error("Dataset not found: $DATA_PATH\nPlease download from https://zenodo.org/record/3820900")
    end

    h5open(DATA_PATH, "r") do file
        println("\nDataset structure:")
        for name in keys(file)
            obj = file[name]
            if isa(obj, HDF5.Dataset)
                println("  $name: $(size(obj))")
            else
                println("  $name: (group)")
            end
        end

        # Try to find the data
        if haskey(file, "X")
            X = read(file["X"])
            println("\nLoaded X with shape: $(size(X))")
            return X, nothing
        elseif haskey(file, "images")
            images = read(file["images"])
            println("\nLoaded images with shape: $(size(images))")
            return images, nothing
        else
            # List all datasets
            println("\nAvailable datasets:")
            function list_datasets(g, prefix="")
                for name in keys(g)
                    obj = g[name]
                    if isa(obj, HDF5.Dataset)
                        println("  $prefix$name: $(size(obj))")
                    elseif isa(obj, HDF5.Group)
                        list_datasets(obj, "$prefix$name/")
                    end
                end
            end
            list_datasets(file)
            return nothing, nothing
        end
    end
end

#=============================================================================
                        FRACTAL DIMENSION
=============================================================================#

function extract_boundary_3d(volume::AbstractArray{<:Number,3}; threshold=0.5)
    # Binarize
    binary = volume .> threshold

    nx, ny, nz = size(binary)
    boundary = falses(nx, ny, nz)

    for i in 2:nx-1
        for j in 2:ny-1
            for k in 2:nz-1
                if binary[i,j,k]
                    neighbors = [
                        binary[i-1,j,k], binary[i+1,j,k],
                        binary[i,j-1,k], binary[i,j+1,k],
                        binary[i,j,k-1], binary[i,j,k+1]
                    ]
                    if any(.!neighbors)
                        boundary[i,j,k] = true
                    end
                end
            end
        end
    end

    return boundary
end

function box_counting_dimension_3d(volume::AbstractArray{Bool,3})
    nx, ny, nz = size(volume)
    min_dim = min(nx, ny, nz)

    if min_dim < 8
        return NaN, NaN
    end

    max_power = floor(Int, log2(min_dim))
    box_sizes = [2^k for k in 1:max_power-1]

    counts = Int[]
    valid_sizes = Int[]

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
        end
    end

    if length(counts) < 3
        return NaN, NaN
    end

    x = log.(valid_sizes)
    y = log.(counts)

    x_mean = mean(x)
    y_mean = mean(y)

    slope = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
    D = -slope

    intercept = y_mean - slope * x_mean
    y_pred = slope .* x .+ intercept
    ss_res = sum((y .- y_pred).^2)
    ss_tot = sum((y .- y_mean).^2)
    R2 = ss_tot > 0 ? 1 - ss_res / ss_tot : 0.0

    return D, R2
end

function compute_porosity(volume::AbstractArray{<:Number,3}; threshold=0.5)
    binary = volume .> threshold
    return 1 - sum(binary) / length(binary)
end

#=============================================================================
                        ANALYSIS
=============================================================================#

function analyze_samples(data::AbstractArray, n_samples::Int=100)
    println("\n" * "="^70)
    println("ANALYZING DEEPORE SAMPLES FOR D = φ")
    println("="^70)
    println("\nPrediction: D = φ = $(round(φ, digits=4)) at porosity ~95.8%")
    println()

    # Determine data shape
    data_shape = size(data)
    println("Data shape: $data_shape")

    # Assuming format is (samples, channels, x, y, z) or (samples, x, y, z)
    n_total = data_shape[1]
    println("Total samples available: $n_total")

    # Random sample
    sample_indices = randperm(n_total)[1:min(n_samples, n_total)]

    results = []

    println("\nProcessing $n_samples samples...")
    println("-"^50)

    for (i, idx) in enumerate(sample_indices)
        # Extract volume based on data dimensions
        if length(data_shape) == 5
            # (samples, channels, x, y, z) - use first channel
            volume = data[idx, 1, :, :, :]
        elseif length(data_shape) == 4
            # (samples, x, y, z)
            volume = data[idx, :, :, :]
        else
            println("Unknown data format with $(length(data_shape)) dimensions")
            continue
        end

        # Compute porosity
        porosity = compute_porosity(volume)

        # Extract boundary
        boundary = extract_boundary_3d(volume)

        # Compute fractal dimension
        D, R2 = box_counting_dimension_3d(boundary)

        if !isnan(D) && R2 > 0.9
            push!(results, Dict(
                "idx" => idx,
                "porosity" => porosity,
                "D" => D,
                "R2" => R2,
                "D_over_phi" => D / φ,
                "distance_to_phi" => abs(D - φ)
            ))

            # Progress indicator
            marker = abs(D - φ) < 0.1 ? " ★" : ""
            if i % 10 == 0 || abs(D - φ) < 0.1
                println(@sprintf("  Sample %4d: porosity=%.1f%%, D=%.3f, D/φ=%.3f%s",
                    idx, porosity*100, D, D/φ, marker))
            end
        end

        if i % 20 == 0
            print(".")
        end
    end
    println()

    return results
end

function analyze_by_porosity(results::Vector)
    println("\n" * "="^70)
    println("D vs POROSITY ANALYSIS")
    println("="^70)

    if isempty(results)
        println("No valid results to analyze")
        return
    end

    # Sort by porosity
    sorted = sort(results, by=r->r["porosity"])

    # Bin by porosity ranges
    bins = [
        (0.0, 0.5, "0-50%"),
        (0.5, 0.7, "50-70%"),
        (0.7, 0.8, "70-80%"),
        (0.8, 0.9, "80-90%"),
        (0.9, 0.95, "90-95%"),
        (0.95, 1.0, "95-100%")
    ]

    println("\nD by porosity range:")
    println("-"^60)
    println("Porosity Range | N samples |    D mean ± std    |  D/φ")
    println("-"^60)

    for (low, high, label) in bins
        in_range = filter(r -> low <= r["porosity"] < high, sorted)

        if !isempty(in_range)
            D_vals = [r["D"] for r in in_range]
            D_mean = mean(D_vals)
            D_std = length(D_vals) > 1 ? std(D_vals) : 0.0

            marker = abs(D_mean - φ) < 0.1 ? " ★" : ""
            println(@sprintf("  %10s   |    %3d    | %6.3f ± %5.3f   | %5.3f%s",
                label, length(in_range), D_mean, D_std, D_mean/φ, marker))
        else
            println(@sprintf("  %10s   |    %3d    |       -          |   -", label, 0))
        end
    end

    # Find best match to φ
    println("\n" * "="^70)
    println("BEST MATCHES TO φ = $(round(φ, digits=4))")
    println("="^70)

    sorted_by_phi = sort(results, by=r->r["distance_to_phi"])

    println("\nTop 10 closest to φ:")
    for (i, r) in enumerate(sorted_by_phi[1:min(10, length(sorted_by_phi))])
        println(@sprintf("  %2d. Porosity=%5.1f%%, D=%.4f, |D-φ|=%.4f",
            i, r["porosity"]*100, r["D"], r["distance_to_phi"]))
    end

    # Check if high porosity samples are closer to φ
    high_p = filter(r -> r["porosity"] > 0.9, results)
    low_p = filter(r -> r["porosity"] <= 0.9, results)

    if !isempty(high_p) && !isempty(low_p)
        high_dist = mean([r["distance_to_phi"] for r in high_p])
        low_dist = mean([r["distance_to_phi"] for r in low_p])

        println("\n" * "="^70)
        println("HYPOTHESIS TEST: Is D closer to φ at high porosity?")
        println("="^70)
        println("\n  Mean |D - φ| for porosity > 90%:  $(round(high_dist, digits=4))")
        println("  Mean |D - φ| for porosity ≤ 90%:  $(round(low_dist, digits=4))")

        if high_dist < low_dist
            println("\n  ✓ YES! High porosity samples are $(round((1-high_dist/low_dist)*100, digits=1))% closer to φ")
        else
            println("\n  ✗ No, low porosity samples are closer to φ")
        end
    end
end

#=============================================================================
                            MAIN
=============================================================================#

function main()
    Random.seed!(42)

    println("╔══════════════════════════════════════════════════════════════════╗")
    println("║  DeePore Dataset Analysis: Validating D = φ Prediction           ║")
    println("╚══════════════════════════════════════════════════════════════════╝")
    println()

    # Load data
    data, properties = load_deepore_data()

    if data === nothing
        println("\nCould not load data. Please check file format.")
        return
    end

    # Analyze samples
    results = analyze_samples(data, 200)

    if !isempty(results)
        analyze_by_porosity(results)

        # Final summary
        println("\n" * "="^70)
        println("CONCLUSION")
        println("="^70)

        D_mean = mean([r["D"] for r in results])
        p_mean = mean([r["porosity"] for r in results])

        println("\n  Analyzed $(length(results)) valid samples")
        println("  Mean porosity: $(round(p_mean*100, digits=1))%")
        println("  Mean D: $(round(D_mean, digits=3))")
        println("  φ = $(round(φ, digits=3))")
        println("  Mean D/φ = $(round(D_mean/φ, digits=3))")
    end

    println("\n" * "="^70)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
