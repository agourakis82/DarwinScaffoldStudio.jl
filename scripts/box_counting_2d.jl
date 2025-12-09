"""
2D Box-Counting Fractal Dimension Analysis for SEM Images

Implements box-counting method to compute fractal dimension D_2D for 2D images.
Tests the hypothesis: D_2D = 1/φ = 0.618034 at high surface porosity in SEM images.

Author: Darwin Scaffold Studio
Date: 2025-12-08
"""

using Images
using FileIO
using Statistics
using Plots
using Printf

# Golden ratio and its reciprocal
const φ = (1 + sqrt(5)) / 2  # 1.618034...
const INV_φ = 1 / φ           # 0.618034... (φ - 1)

"""
    box_counting_2d(image::Matrix{Bool}; min_box_size=2, max_box_size=nothing)

Compute 2D fractal dimension using box-counting method.

# Arguments
- `image::Matrix{Bool}`: Binary image (true = solid, false = void/pore)
- `min_box_size::Int`: Minimum box size in pixels (default: 2)
- `max_box_size::Int`: Maximum box size (default: min(height, width)/4)

# Returns
- `D_2d::Float64`: Fractal dimension
- `box_sizes::Vector{Int}`: Box sizes used
- `counts::Vector{Int}`: Number of boxes at each size
- `r_squared::Float64`: R² goodness of fit
"""
function box_counting_2d(image::AbstractMatrix{Bool};
                         min_box_size::Int=2,
                         max_box_size::Union{Int,Nothing}=nothing)

    h, w = size(image)

    # Determine box size range
    if isnothing(max_box_size)
        max_box_size = min(h, w) ÷ 4
    end

    # Generate box sizes (powers of 2)
    max_power = floor(Int, log2(max_box_size))
    min_power = floor(Int, log2(min_box_size))
    box_sizes = [2^k for k in min_power:max_power]

    counts = Int[]

    # Count boxes at each scale
    for box_size in box_sizes
        count = 0

        # Slide box across image
        for i in 1:box_size:h
            for j in 1:box_size:w
                # Define box boundaries
                i_end = min(i + box_size - 1, h)
                j_end = min(j + box_size - 1, w)

                # Check if box contains any solid material
                box_region = @view image[i:i_end, j:j_end]
                if any(box_region)
                    count += 1
                end
            end
        end

        push!(counts, count)
    end

    # Log-log regression: log(N) = -D * log(ε) + c
    # where N = count, ε = box_size, D = fractal dimension
    x = log.(Float64.(box_sizes))
    y = log.(Float64.(counts))

    # Linear regression
    n = length(x)
    x_mean = mean(x)
    y_mean = mean(y)

    slope = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
    intercept = y_mean - slope * x_mean

    # Fractal dimension is negative slope
    D_2d = -slope

    # R² goodness of fit
    y_pred = slope .* x .+ intercept
    ss_res = sum((y .- y_pred).^2)
    ss_tot = sum((y .- y_mean).^2)
    r_squared = 1 - ss_res / ss_tot

    return D_2d, box_sizes, counts, r_squared
end

"""
    compute_surface_porosity(image::AbstractMatrix{Bool})

Compute surface porosity as fraction of void pixels.

# Arguments
- `image::AbstractMatrix{Bool}`: Binary image (true = solid, false = void)

# Returns
- `porosity::Float64`: Porosity fraction (0 to 1)
"""
function compute_surface_porosity(image::AbstractMatrix{Bool})
    total_pixels = length(image)
    void_pixels = count(.!image)  # Count false (void) pixels
    return void_pixels / total_pixels
end

"""
    load_and_binarize(filepath::String; threshold=0.5)

Load image and convert to binary.

# Arguments
- `filepath::String`: Path to image file
- `threshold::Float64`: Binarization threshold (0 to 1)

# Returns
- `binary_image::Matrix{Bool}`: Binary image
"""
function load_and_binarize(filepath::String; threshold::Float64=0.5)
    # Load image
    img = load(filepath)

    # Convert to grayscale if needed
    if eltype(img) <: RGB
        img_gray = Gray.(img)
    else
        img_gray = img
    end

    # Convert to float matrix
    img_float = Float64.(img_gray)

    # Binarize (true = solid, false = void)
    binary_image = img_float .> threshold

    return binary_image
end

"""
    plot_box_counting(box_sizes, counts, D_2d, r_squared;
                      save_path=nothing)

Create log-log plot of box-counting analysis.

# Arguments
- `box_sizes::Vector{Int}`: Box sizes
- `counts::Vector{Int}`: Counts at each box size
- `D_2d::Float64`: Computed fractal dimension
- `r_squared::Float64`: R² value
- `save_path::String`: Path to save plot (optional)
"""
function plot_box_counting(box_sizes, counts, D_2d, r_squared;
                           save_path::Union{String,Nothing}=nothing)

    x = log.(Float64.(box_sizes))
    y = log.(Float64.(counts))

    # Linear fit
    x_mean = mean(x)
    y_mean = mean(y)
    slope = -D_2d
    intercept = y_mean - slope * x_mean

    x_fit = range(minimum(x), maximum(x), length=100)
    y_fit = slope .* x_fit .+ intercept

    p = plot(x, y,
             seriestype=:scatter,
             label="Data",
             markersize=6,
             xlabel="log(Box Size)",
             ylabel="log(Count)",
             title=@sprintf("2D Box-Counting: D = %.4f (R² = %.4f)", D_2d, r_squared),
             legend=:topright,
             size=(600, 500),
             dpi=300)

    plot!(p, x_fit, y_fit,
          label=@sprintf("Fit: slope = %.4f", slope),
          linewidth=2,
          linestyle=:dash)

    # Add golden ratio reference
    hline!(p, [log(INV_φ)],
           label=@sprintf("1/φ = %.4f", INV_φ),
           linewidth=2,
           linestyle=:dot,
           color=:red)

    if !isnothing(save_path)
        savefig(p, save_path)
        println("Plot saved to: $save_path")
    end

    return p
end

"""
    analyze_sem_image(filepath::String; threshold=0.5, save_plot=true)

Complete analysis of SEM image: porosity + fractal dimension.

# Arguments
- `filepath::String`: Path to SEM image
- `threshold::Float64`: Binarization threshold
- `save_plot::Bool`: Whether to save diagnostic plots

# Returns
- `results::Dict`: Dictionary with all results
"""
function analyze_sem_image(filepath::String;
                           threshold::Float64=0.5,
                           save_plot::Bool=true)

    println("="^70)
    println("2D FRACTAL DIMENSION ANALYSIS - SEM IMAGE")
    println("="^70)
    println("File: $filepath")
    println()

    # Load and binarize
    println("Loading and binarizing image (threshold = $threshold)...")
    binary_image = load_and_binarize(filepath, threshold=threshold)
    h, w = size(binary_image)
    println("Image size: $h × $w pixels")
    println()

    # Compute surface porosity
    porosity = compute_surface_porosity(binary_image)
    println(@sprintf("Surface Porosity: %.2f%%", porosity * 100))
    println()

    # Box-counting analysis
    println("Running box-counting analysis...")
    D_2d, box_sizes, counts, r_squared = box_counting_2d(binary_image)

    println("Results:")
    println(@sprintf("  D_2D = %.6f", D_2d))
    println(@sprintf("  R² = %.6f", r_squared))
    println(@sprintf("  Box sizes: %d to %d pixels", minimum(box_sizes), maximum(box_sizes)))
    println(@sprintf("  Number of scales: %d", length(box_sizes)))
    println()

    # Compare to golden ratio
    delta_from_inv_phi = abs(D_2d - INV_φ)
    percent_error = (delta_from_inv_phi / INV_φ) * 100

    println("Comparison to 1/φ hypothesis:")
    println(@sprintf("  1/φ = %.6f", INV_φ))
    println(@sprintf("  D_2D = %.6f", D_2d))
    println(@sprintf("  |D_2D - 1/φ| = %.6f", delta_from_inv_phi))
    println(@sprintf("  Percent error: %.2f%%", percent_error))

    if percent_error < 5.0
        println("  ✓ HYPOTHESIS SUPPORTED (< 5% error)")
    elseif percent_error < 10.0
        println("  ~ HYPOTHESIS PARTIALLY SUPPORTED (5-10% error)")
    else
        println("  ✗ HYPOTHESIS NOT SUPPORTED (> 10% error)")
    end
    println()

    # Create plot
    if save_plot
        base_name = splitext(basename(filepath))[1]
        plot_path = "results/sem_analysis_$(base_name).png"
        mkpath("results")
        plot_box_counting(box_sizes, counts, D_2d, r_squared, save_path=plot_path)
    end

    # Return results
    results = Dict(
        "filepath" => filepath,
        "image_size" => (h, w),
        "threshold" => threshold,
        "porosity" => porosity,
        "D_2d" => D_2d,
        "r_squared" => r_squared,
        "box_sizes" => box_sizes,
        "counts" => counts,
        "inv_phi" => INV_φ,
        "error_from_inv_phi" => delta_from_inv_phi,
        "percent_error" => percent_error
    )

    println("="^70)

    return results
end

"""
    batch_analyze_sem_images(image_dir::String; threshold=0.5)

Analyze multiple SEM images and test D_2D vs porosity relationship.

# Arguments
- `image_dir::String`: Directory containing SEM images
- `threshold::Float64`: Binarization threshold

# Returns
- `batch_results::Vector{Dict}`: Results for each image
"""
function batch_analyze_sem_images(image_dir::String; threshold::Float64=0.5)

    println("\n" * "="^70)
    println("BATCH SEM ANALYSIS - D_2D = 1/φ HYPOTHESIS TEST")
    println("="^70)
    println()

    # Find all image files
    image_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    image_files = String[]

    for file in readdir(image_dir)
        ext = lowercase(splitext(file)[2])
        if ext in image_extensions
            push!(image_files, joinpath(image_dir, file))
        end
    end

    if isempty(image_files)
        println("No image files found in: $image_dir")
        return []
    end

    println("Found $(length(image_files)) images")
    println()

    # Analyze each image
    batch_results = []

    for (i, filepath) in enumerate(image_files)
        println("\n[$i/$(length(image_files))] Processing: $(basename(filepath))")

        try
            results = analyze_sem_image(filepath, threshold=threshold, save_plot=true)
            push!(batch_results, results)
        catch e
            println("ERROR processing $filepath: $e")
        end
    end

    # Summary statistics
    if !isempty(batch_results)
        println("\n" * "="^70)
        println("BATCH SUMMARY")
        println("="^70)

        porosities = [r["porosity"] for r in batch_results]
        D_2d_values = [r["D_2d"] for r in batch_results]
        r_squared_values = [r["r_squared"] for r in batch_results]
        errors = [r["percent_error"] for r in batch_results]

        println(@sprintf("Porosity range: %.1f%% - %.1f%%",
                        minimum(porosities)*100, maximum(porosities)*100))
        println(@sprintf("D_2D range: %.4f - %.4f",
                        minimum(D_2d_values), maximum(D_2d_values)))
        println(@sprintf("Mean D_2D: %.4f ± %.4f",
                        mean(D_2d_values), std(D_2d_values)))
        println(@sprintf("Mean R²: %.4f", mean(r_squared_values)))
        println(@sprintf("Mean error from 1/φ: %.2f%%", mean(errors)))
        println()

        # Count support for hypothesis
        supported = count(e -> e < 5.0, errors)
        partial = count(e -> 5.0 <= e < 10.0, errors)
        not_supported = count(e -> e >= 10.0, errors)

        println("Hypothesis support:")
        println(@sprintf("  Strong support (< 5%% error): %d/%d (%.1f%%)",
                        supported, length(errors), supported/length(errors)*100))
        println(@sprintf("  Partial support (5-10%% error): %d/%d (%.1f%%)",
                        partial, length(errors), partial/length(errors)*100))
        println(@sprintf("  Not supported (> 10%% error): %d/%d (%.1f%%)",
                        not_supported, length(errors), not_supported/length(errors)*100))
        println()

        # Create summary plot: D_2D vs Porosity
        p = scatter(porosities .* 100, D_2d_values,
                   xlabel="Surface Porosity (%)",
                   ylabel="Fractal Dimension D_2D",
                   title="D_2D vs Porosity - SEM Images",
                   label="Measured",
                   markersize=8,
                   legend=:topright,
                   size=(700, 500),
                   dpi=300)

        hline!(p, [INV_φ],
               label=@sprintf("1/φ = %.4f", INV_φ),
               linewidth=2,
               linestyle=:dash,
               color=:red)

        savefig(p, "results/batch_sem_D2d_vs_porosity.png")
        println("Summary plot saved to: results/batch_sem_D2d_vs_porosity.png")
        println()
    end

    return batch_results
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    println("2D Box-Counting Fractal Dimension Analysis")
    println("Testing D_2D = 1/φ = 0.618034 hypothesis")
    println()
    println("Usage:")
    println("  Single image:")
    println("    julia box_counting_2d.jl path/to/sem_image.tif")
    println()
    println("  Batch analysis:")
    println("    julia box_counting_2d.jl path/to/image_directory/")
    println()

    if length(ARGS) > 0
        path = ARGS[1]

        if isfile(path)
            # Single image analysis
            results = analyze_sem_image(path)
        elseif isdir(path)
            # Batch analysis
            results = batch_analyze_sem_images(path)
        else
            println("Error: Path not found: $path")
        end
    end
end
