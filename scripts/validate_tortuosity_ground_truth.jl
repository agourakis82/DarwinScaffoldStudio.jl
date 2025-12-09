#!/usr/bin/env julia
"""
Validate GeodesicTortuosity.jl against Zenodo 7516228 Ground Truth Dataset

Dataset: Soil Pore Space 3D with geodesic tortuosity ground truth
- 4,608 samples with porosity 14.4-51.0%
- Geodesic tortuosity: 1.059 - 1.257 (scaffold-like range)
- Ground truth computed using validated methods

Target: >95% accuracy (error < 5%)
"""

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Statistics
using Images
using FileIO
using Printf
using ProgressMeter
using Random
using TiffImages

# Load our module
include("../src/DarwinScaffoldStudio/Science/GeodesicTortuosity.jl")
using .GeodesicTortuosity

const DATA_DIR = joinpath(@__DIR__, "..", "data", "soil_pore_space")
const RESULTS_FILE = joinpath(@__DIR__, "..", "results", "tortuosity_validation.csv")

"""
Load ground truth from characteristics.csv
"""
function load_ground_truth()
    csv_path = joinpath(DATA_DIR, "characteristics.csv")
    if !isfile(csv_path)
        error("Ground truth file not found: $csv_path")
    end

    df = CSV.read(csv_path, DataFrame)
    println("Loaded $(nrow(df)) samples with ground truth")

    # Statistics
    println("\n=== GROUND TRUTH STATISTICS ===")
    println("Porosity: $(round(minimum(df.porosity), digits=4)) - $(round(maximum(df.porosity), digits=4))")
    println("Geodesic τ: $(round(minimum(df[!, "mean geodesic tortuosity"]), digits=4)) - $(round(maximum(df[!, "mean geodesic tortuosity"]), digits=4))")
    println("Geometric τ: $(round(minimum(df[!, "mean geometric tortuosity"]), digits=4)) - $(round(maximum(df[!, "mean geometric tortuosity"]), digits=4))")

    return df
end

"""
Load a 3D TIFF stack (multi-page TIFF with 128 slices)
"""
function load_tiff_stack(filepath::String)
    if !isfile(filepath)
        return nothing
    end

    try
        # Use TiffImages for multi-page TIFF support
        img = TiffImages.load(filepath)

        # img is typically Array{Gray{N0f8},3} for multi-page TIFF
        if ndims(img) == 3
            # Convert to binary (pore=black=0, solid=white=1 in dataset)
            # In the dataset: black=pore, white=solid
            binary = Float64.(img) .> 0.5
            return BitArray(binary)
        elseif ndims(img) == 2
            # Single slice, not usable for tortuosity
            return nothing
        end
    catch e
        # Fallback to Images.jl
        try
            img = load(filepath)
            if ndims(img) == 3
                binary = Float64.(Gray.(img)) .> 0.5
                return BitArray(binary)
            end
        catch e2
            @warn "Failed to load $filepath: $e2"
        end
    end

    return nothing
end

"""
Compute tortuosity using our FMM implementation
"""
function compute_our_tortuosity(binary::AbstractArray{<:Any,3})
    try
        result = compute_geodesic_tortuosity(binary,
            direction=:z,
            n_samples=100
        )
        return result.mean, result.std
    catch e
        @warn "Tortuosity computation failed: $e"
        return NaN, NaN
    end
end

"""
Validate against a subset of samples
"""
function validate_samples(df::DataFrame; n_samples::Int=100, seed::Int=42)
    Random.seed!(seed)

    # Check if stacks directory exists
    stacks_dir = joinpath(DATA_DIR, "segmented_stacks")
    if !isdir(stacks_dir)
        # Try to extract zip
        zip_file = joinpath(DATA_DIR, "segmented_stacks.zip")
        if isfile(zip_file)
            println("Extracting segmented_stacks.zip...")
            run(`unzip -q -o $zip_file -d $DATA_DIR`)
        else
            error("Segmented stacks not found. Please download from Zenodo.")
        end
    end

    # Sample rows
    n_available = min(n_samples, nrow(df))
    indices = randperm(nrow(df))[1:n_available]

    results = DataFrame(
        sample_id = Int[],
        porosity_gt = Float64[],
        geodesic_tort_gt = Float64[],
        geodesic_tort_pred = Float64[],
        absolute_error = Float64[],
        relative_error = Float64[],
        file = String[]
    )

    println("\n=== VALIDATING $n_available SAMPLES ===")

    errors = Float64[]
    rel_errors = Float64[]

    @showprogress for (i, idx) in enumerate(indices)
        row = df[idx, :]

        # Get file path
        file_path = joinpath(DATA_DIR, row.file)

        # Load 3D volume
        binary = load_tiff_stack(file_path)
        if binary === nothing
            continue
        end

        # Ground truth
        gt_porosity = row.porosity
        gt_tortuosity = row["mean geodesic tortuosity"]

        # Our prediction
        pred_tortuosity, pred_std = compute_our_tortuosity(binary)

        if isnan(pred_tortuosity)
            continue
        end

        # Compute errors
        abs_error = abs(pred_tortuosity - gt_tortuosity)
        rel_error = abs_error / gt_tortuosity * 100

        push!(errors, abs_error)
        push!(rel_errors, rel_error)

        push!(results, (
            sample_id = idx,
            porosity_gt = gt_porosity,
            geodesic_tort_gt = gt_tortuosity,
            geodesic_tort_pred = pred_tortuosity,
            absolute_error = abs_error,
            relative_error = rel_error,
            file = row.file
        ))

        # Print progress every 10 samples
        if i % 10 == 0
            current_mae = mean(errors)
            current_mre = mean(rel_errors)
            @printf("  [%d/%d] MAE=%.4f, MRE=%.2f%%\n", i, n_available, current_mae, current_mre)
        end
    end

    return results, errors, rel_errors
end

"""
Compute validation metrics
"""
function compute_metrics(errors::Vector{Float64}, rel_errors::Vector{Float64})
    println("\n" * "="^60)
    println("VALIDATION RESULTS")
    println("="^60)

    n = length(errors)
    mae = mean(errors)
    mre = mean(rel_errors)
    rmse = sqrt(mean(errors.^2))
    max_error = maximum(errors)

    # Accuracy metrics
    within_1pct = count(x -> x < 1.0, rel_errors) / n * 100
    within_5pct = count(x -> x < 5.0, rel_errors) / n * 100
    within_10pct = count(x -> x < 10.0, rel_errors) / n * 100

    println("Samples validated: $n")
    println()
    println("Error Metrics:")
    @printf("  MAE (Mean Absolute Error): %.4f\n", mae)
    @printf("  MRE (Mean Relative Error): %.2f%%\n", mre)
    @printf("  RMSE: %.4f\n", rmse)
    @printf("  Max Error: %.4f\n", max_error)
    println()
    println("Accuracy Metrics:")
    @printf("  Within 1%%: %.1f%% of samples\n", within_1pct)
    @printf("  Within 5%%: %.1f%% of samples\n", within_5pct)
    @printf("  Within 10%%: %.1f%% of samples\n", within_10pct)
    println()

    # Pass/Fail
    target_accuracy = 95.0
    if within_5pct >= target_accuracy
        println("✓ TARGET ACHIEVED: $(round(within_5pct, digits=1))% accuracy (>$target_accuracy% within 5% error)")
    else
        println("✗ TARGET NOT MET: $(round(within_5pct, digits=1))% accuracy (<$target_accuracy% within 5% error)")
        println("  Gap: $(round(target_accuracy - within_5pct, digits=1)) percentage points")
    end

    return Dict(
        :n => n,
        :mae => mae,
        :mre => mre,
        :rmse => rmse,
        :max_error => max_error,
        :within_1pct => within_1pct,
        :within_5pct => within_5pct,
        :within_10pct => within_10pct
    )
end

"""
Main validation pipeline
"""
function main(; n_samples::Int=100)
    println("="^60)
    println("GEODESIC TORTUOSITY VALIDATION")
    println("Dataset: Zenodo 7516228 (Soil Pore Space 3D)")
    println("="^60)

    # Load ground truth
    df = load_ground_truth()

    # Run validation
    results, errors, rel_errors = validate_samples(df, n_samples=n_samples)

    if isempty(errors)
        println("\nNo samples could be validated. Check data files.")
        return nothing
    end

    # Compute metrics
    metrics = compute_metrics(errors, rel_errors)

    # Save results
    mkpath(dirname(RESULTS_FILE))
    CSV.write(RESULTS_FILE, results)
    println("\nResults saved to: $RESULTS_FILE")

    return metrics
end

# Run validation
if abspath(PROGRAM_FILE) == @__FILE__
    n = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 50
    main(n_samples=n)
end
