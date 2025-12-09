#!/usr/bin/env julia
"""
Train and Validate TopoTort on Zenodo 7516228 Dataset
=====================================================

Novel contribution: First TDA+GNN model for scaffold tortuosity prediction.
Target: >95% accuracy with 1000x speedup over FMM.
"""

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Statistics
using Random
using Printf
using TiffImages
using ProgressMeter

# Load modules
include("../src/DarwinScaffoldStudio/Science/TDA.jl")
include("../src/DarwinScaffoldStudio/Science/TopoTort.jl")
include("../src/DarwinScaffoldStudio/Science/GeodesicTortuosity.jl")

using .TDA
using .TopoTort
using .GeodesicTortuosity

const DATA_DIR = joinpath(@__DIR__, "..", "data", "soil_pore_space")

# ============================================================================
# DATA LOADING
# ============================================================================

function load_dataset(; max_samples::Int=500)
    println("Loading Zenodo 7516228 dataset...")

    csv_path = joinpath(DATA_DIR, "characteristics.csv")
    df = CSV.read(csv_path, DataFrame)

    println("Total available samples: $(nrow(df))")

    # Sample subset
    n_samples = min(max_samples, nrow(df))
    Random.seed!(42)
    indices = randperm(nrow(df))[1:n_samples]

    data = Tuple{Array{Bool,3}, Float64}[]

    println("Loading $n_samples volumes...")
    @showprogress for idx in indices
        row = df[idx, :]
        filepath = joinpath(DATA_DIR, row.file)

        if !isfile(filepath)
            continue
        end

        try
            img = TiffImages.load(filepath)
            binary = Array{Bool,3}(Float64.(img) .> 0.5)
            τ_gt = row["mean geodesic tortuosity"]

            push!(data, (binary, τ_gt))
        catch e
            # Skip failed loads
        end
    end

    println("Loaded $(length(data)) samples")
    return data
end

# ============================================================================
# TRAINING
# ============================================================================

function train_and_evaluate(; n_train::Int=300, n_test::Int=100)
    # Load data
    all_data = load_dataset(max_samples=n_train + n_test + 50)

    if length(all_data) < n_train + n_test
        n_train = Int(floor(0.75 * length(all_data)))
        n_test = length(all_data) - n_train
    end

    # Split
    train_data = all_data[1:n_train]
    test_data = all_data[n_train+1:n_train+n_test]

    println("\n" * "="^60)
    println("TOPOTORT TRAINING")
    println("="^60)
    println("Training samples: $n_train")
    println("Test samples: $n_test")

    # Initialize model
    config = TopoTortConfig(
        n_pore_samples = 1500,
        connectivity_radius = 6.0,
        persistence_image_size = 15,
        n_message_passing = 3,
        hidden_dim = 64,
        n_epochs = 50
    )

    model = TopoTortModel(config)

    # Train
    println("\nTraining TopoTort...")
    train_topotort!(model, train_data; config=config, verbose=true)

    # Evaluate
    println("\n" * "="^60)
    println("EVALUATION")
    println("="^60)

    results = evaluate_topotort(model, test_data; config=config)

    println("\nTopoTort Results:")
    @printf("  MAE: %.4f\n", results["MAE"])
    @printf("  RMSE: %.4f\n", results["RMSE"])
    @printf("  R²: %.4f\n", results["R2"])
    @printf("  MRE: %.2f%%\n", results["MRE"])
    @printf("  Within 5%%: %.1f%%\n", results["within_5pct"])
    @printf("  Mean inference time: %.1f ms\n", results["mean_inference_ms"])
    @printf("  Speedup vs FMM: %.0fx\n", results["speedup_vs_fmm"])

    return model, results
end

# ============================================================================
# QUICK TOPOTORT (Topology-informed heuristic)
# ============================================================================

function evaluate_quick_topotort(; n_samples::Int=100)
    println("\n" * "="^60)
    println("QUICK TOPOTORT EVALUATION (Topology-Informed Heuristic)")
    println("="^60)

    # Load data
    csv_path = joinpath(DATA_DIR, "characteristics.csv")
    df = CSV.read(csv_path, DataFrame)

    Random.seed!(123)
    indices = randperm(nrow(df))[1:n_samples]

    predictions = Float64[]
    ground_truths = Float64[]
    inference_times = Float64[]

    println("Evaluating on $n_samples samples...")
    @showprogress for idx in indices
        row = df[idx, :]
        filepath = joinpath(DATA_DIR, row.file)

        if !isfile(filepath)
            continue
        end

        try
            img = TiffImages.load(filepath)
            binary = Array{Bool,3}(Float64.(img) .> 0.5)
            τ_gt = row["mean geodesic tortuosity"]

            # Time the prediction
            t_start = time()
            τ_pred = quick_topotort(binary; use_topology=true)
            t_end = time()

            push!(predictions, τ_pred)
            push!(ground_truths, τ_gt)
            push!(inference_times, (t_end - t_start) * 1000)
        catch e
            # Skip failed
        end
    end

    # Compute metrics
    errors = predictions .- ground_truths
    abs_errors = abs.(errors)
    rel_errors = abs_errors ./ ground_truths .* 100

    mae = mean(abs_errors)
    rmse = sqrt(mean(errors.^2))
    mre = mean(rel_errors)

    within_5pct = count(x -> x < 5.0, rel_errors) / length(rel_errors) * 100
    within_10pct = count(x -> x < 10.0, rel_errors) / length(rel_errors) * 100

    mean_time = mean(inference_times)

    # FMM baseline: ~5000ms for 128³
    fmm_time = 5000.0
    speedup = fmm_time / mean_time

    println("\nQuick TopoTort Results:")
    @printf("  Samples: %d\n", length(predictions))
    @printf("  MAE: %.4f\n", mae)
    @printf("  RMSE: %.4f\n", rmse)
    @printf("  MRE: %.2f%%\n", mre)
    @printf("  Within 5%%: %.1f%%\n", within_5pct)
    @printf("  Within 10%%: %.1f%%\n", within_10pct)
    @printf("  Mean inference time: %.1f ms\n", mean_time)
    @printf("  Speedup vs FMM: %.0fx\n", speedup)

    # Show some examples
    println("\nSample predictions:")
    for i in 1:min(10, length(predictions))
        status = rel_errors[i] < 5.0 ? "OK" : "MISS"
        @printf("  [%2d] pred=%.4f, gt=%.4f, err=%.2f%% [%s]\n",
                i, predictions[i], ground_truths[i], rel_errors[i], status)
    end

    return Dict(
        "MAE" => mae,
        "RMSE" => rmse,
        "MRE" => mre,
        "within_5pct" => within_5pct,
        "speedup" => speedup
    )
end

# ============================================================================
# COMPARISON: FMM vs TopoTort
# ============================================================================

function benchmark_comparison(; n_samples::Int=30)
    println("\n" * "="^60)
    println("BENCHMARK: FMM vs Quick TopoTort")
    println("="^60)

    csv_path = joinpath(DATA_DIR, "characteristics.csv")
    df = CSV.read(csv_path, DataFrame)

    Random.seed!(999)
    indices = randperm(nrow(df))[1:n_samples]

    fmm_times = Float64[]
    fmm_errors = Float64[]

    topo_times = Float64[]
    topo_errors = Float64[]

    println("Running comparison on $n_samples samples...")

    for (i, idx) in enumerate(indices)
        row = df[idx, :]
        filepath = joinpath(DATA_DIR, row.file)

        if !isfile(filepath)
            continue
        end

        try
            img = TiffImages.load(filepath)
            binary = BitArray(Float64.(img) .> 0.5)
            τ_gt = row["mean geodesic tortuosity"]

            # FMM
            t1 = time()
            result_fmm = compute_geodesic_tortuosity(binary, direction=:z, n_samples=50)
            t2 = time()
            fmm_time = (t2 - t1) * 1000
            fmm_error = abs(result_fmm.mean - τ_gt) / τ_gt * 100

            push!(fmm_times, fmm_time)
            push!(fmm_errors, fmm_error)

            # TopoTort
            t3 = time()
            τ_topo = quick_topotort(Array{Bool,3}(binary); use_topology=true)
            t4 = time()
            topo_time = (t4 - t3) * 1000
            topo_error = abs(τ_topo - τ_gt) / τ_gt * 100

            push!(topo_times, topo_time)
            push!(topo_errors, topo_error)

            @printf("[%2d] FMM: %.2f%% (%.0fms) | TopoTort: %.2f%% (%.0fms) | GT: %.4f\n",
                    i, fmm_error, fmm_time, topo_error, topo_time, τ_gt)

        catch e
            @printf("[%2d] Error: %s\n", i, string(e)[1:min(50, length(string(e)))])
        end
    end

    println("\n" * "="^60)
    println("BENCHMARK SUMMARY")
    println("="^60)

    println("\nFMM (Calibrated):")
    @printf("  MRE: %.2f%%\n", mean(fmm_errors))
    @printf("  Within 5%%: %.1f%%\n", count(x -> x < 5.0, fmm_errors) / length(fmm_errors) * 100)
    @printf("  Mean time: %.1f ms\n", mean(fmm_times))

    println("\nQuick TopoTort:")
    @printf("  MRE: %.2f%%\n", mean(topo_errors))
    @printf("  Within 5%%: %.1f%%\n", count(x -> x < 5.0, topo_errors) / length(topo_errors) * 100)
    @printf("  Mean time: %.1f ms\n", mean(topo_times))

    speedup = mean(fmm_times) / mean(topo_times)
    @printf("\nSpeedup: %.1fx\n", speedup)
end

# ============================================================================
# MAIN
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    mode = length(ARGS) > 0 ? ARGS[1] : "quick"

    if mode == "full"
        # Full GNN training (slow)
        model, results = train_and_evaluate(n_train=200, n_test=50)
    elseif mode == "benchmark"
        # Compare FMM vs TopoTort
        benchmark_comparison(n_samples=30)
    else
        # Quick topology-informed heuristic (fast)
        evaluate_quick_topotort(n_samples=100)
    end
end
