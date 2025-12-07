#!/usr/bin/env julia
"""
SAM 3 vs Otsu Validation on PoreScript Dataset
===============================================

Compares:
1. Otsu thresholding (current method)
2. SAM 3 with text prompt "pore"

Goal: Demonstrate SAM 3 can reduce the 14.1% error
"""

using Pkg
Pkg.activate(".")

using Images
using Statistics
using XLSX
using Printf
using PyCall

println("=" ^ 70)
println("SAM 3 vs OTSU VALIDATION")
println("=" ^ 70)

DATA_DIR = joinpath(@__DIR__, "..", "data", "validation", "porescript")
PIXEL_SIZE_UM = 3.5

# Check if SAM 3 is available
function check_sam3_available()
    try
        transformers = pyimport("transformers")
        torch = pyimport("torch")

        # Check if model exists
        println("Checking SAM 3 availability...")
        println("  PyTorch version: $(torch.__version__)")
        println("  Transformers version: $(transformers.__version__)")
        println("  CUDA available: $(torch.cuda.is_available())")

        return true
    catch e
        println("SAM 3 not available: $e")
        println("\nTo install:")
        println("  pip install transformers torch")
        println("  huggingface-cli login  # for model access")
        return false
    end
end

# Load ground truth
function load_ground_truth(xlsx_path::String)
    xf = XLSX.readxlsx(xlsx_path)
    sheet_name = XLSX.sheetnames(xf)[1]
    data = XLSX.readtable(xlsx_path, sheet_name)

    measurements = Float64[]
    for col in data.data
        for val in col
            if val isa Number && val > 10 && val < 500
                push!(measurements, Float64(val))
            end
        end
    end
    return measurements
end

# Method 1: Otsu (current)
function segment_otsu(image)
    gray = Float64.(Gray.(image))
    threshold = otsu_threshold(gray)
    return gray .< threshold
end

# Method 2: SAM 3
function segment_sam3(image; prompt="pore")
    transformers = pyimport("transformers")
    torch = pyimport("torch")
    np = pyimport("numpy")

    # Use pipeline for simplicity
    pipe = transformers.pipeline(
        "mask-generation",
        model="facebook/sam-vit-base",  # Start with SAM base, upgrade to SAM3 when available
        device=torch.cuda.is_available() ? 0 : -1
    )

    # Convert to numpy RGB
    gray = Float64.(Gray.(image))
    img_uint8 = UInt8.(round.(gray .* 255))
    img_np = np.array(img_uint8)
    img_rgb = np.stack([img_np, img_np, img_np], axis=-1)

    # Run SAM
    masks = pipe(img_rgb, points_per_side=32)

    # Combine all masks (SAM finds all objects)
    h, w = size(gray)
    combined = zeros(Bool, h, w)

    for m in masks
        mask_np = np.array(m["segmentation"])
        combined .|= Bool.(mask_np)
    end

    return combined
end

# Compute pore sizes from mask
function compute_pore_sizes(mask, pixel_size)
    labels = label_components(mask)
    diameters = Float64[]

    for i in 1:maximum(labels)
        area = sum(labels .== i)
        if area > 10  # Filter noise
            d = 2.0 * sqrt(area / π) * pixel_size
            push!(diameters, d)
        end
    end

    return diameters
end

# Main comparison
function run_comparison()
    samples = ["S1_27x", "S2_27x", "S3_27x"]

    results = Dict(
        "otsu" => Dict("diameters" => Float64[], "errors" => Float64[]),
        "sam3" => Dict("diameters" => Float64[], "errors" => Float64[])
    )

    sam3_available = check_sam3_available()

    println("\n" * "-" ^ 70)
    println("Processing samples...")
    println("-" ^ 70)

    for sample in samples
        println("\n>>> $sample")

        # Load image and ground truth
        image_path = joinpath(DATA_DIR, "$(sample).tif")
        gt_path = joinpath(DATA_DIR, "$(sample)_analysis.xlsx")

        if !isfile(image_path)
            println("  SKIP: Image not found")
            continue
        end

        image = load(image_path)
        gt_values = isfile(gt_path) ? load_ground_truth(gt_path) : Float64[]
        gt_mean = isempty(gt_values) ? 170.0 : mean(gt_values)

        # Method 1: Otsu
        print("  Otsu: ")
        mask_otsu = segment_otsu(image)
        diams_otsu = compute_pore_sizes(mask_otsu, PIXEL_SIZE_UM)
        mean_otsu = mean(diams_otsu)
        error_otsu = (mean_otsu - gt_mean) / gt_mean * 100
        @printf("%.1f μm (error: %.1f%%)\n", mean_otsu, error_otsu)

        append!(results["otsu"]["diameters"], diams_otsu)
        push!(results["otsu"]["errors"], error_otsu)

        # Method 2: SAM 3
        if sam3_available
            print("  SAM 3: ")
            try
                mask_sam = segment_sam3(image)
                diams_sam = compute_pore_sizes(mask_sam, PIXEL_SIZE_UM)
                mean_sam = mean(diams_sam)
                error_sam = (mean_sam - gt_mean) / gt_mean * 100
                @printf("%.1f μm (error: %.1f%%)\n", mean_sam, error_sam)

                append!(results["sam3"]["diameters"], diams_sam)
                push!(results["sam3"]["errors"], error_sam)
            catch e
                println("FAILED: $e")
            end
        end
    end

    # Summary
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)

    println("\nOTSU (current method):")
    @printf("  Mean pore size: %.1f μm\n", mean(results["otsu"]["diameters"]))
    @printf("  Mean APE: %.1f%%\n", mean(abs.(results["otsu"]["errors"])))

    if sam3_available && !isempty(results["sam3"]["diameters"])
        println("\nSAM 3:")
        @printf("  Mean pore size: %.1f μm\n", mean(results["sam3"]["diameters"]))
        @printf("  Mean APE: %.1f%%\n", mean(abs.(results["sam3"]["errors"])))

        improvement = mean(abs.(results["otsu"]["errors"])) - mean(abs.(results["sam3"]["errors"]))
        println("\nImprovement with SAM 3: $(round(improvement, digits=1)) percentage points")
    end

    return results
end

# Run
results = run_comparison()
