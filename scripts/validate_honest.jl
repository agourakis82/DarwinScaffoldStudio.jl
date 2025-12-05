#!/usr/bin/env julia
"""
HONEST Validation - No data fitting, proper metrics
"""

using Pkg
Pkg.activate(".")

using Images
using Statistics
using XLSX
using Printf

println("=" ^ 60)
println("DARWIN - HONEST VALIDATION (No Parameter Tuning)")
println("=" ^ 60)

DATA_DIR = joinpath(@__DIR__, "..", "data", "validation", "porescript")
PIXEL_SIZE_UM = 3.5

function load_ground_truth(xlsx_path::String)
    xf = XLSX.readxlsx(xlsx_path)
    sheet_name = XLSX.sheetnames(xf)[1]
    data = XLSX.readtable(xlsx_path, sheet_name)

    measurements = Float64[]
    for col in data.data
        for val in col
            if val isa Number && val > 0 && val < 1000
                push!(measurements, Float64(val))
            end
        end
    end
    return measurements
end

function compute_pore_size_honest(image_path::String; scale::Int=4)
    """
    HONEST algorithm - pure Otsu, no adjustments
    """
    img = load(image_path)
    gray = Gray.(img)
    img_array = Float64.(gray)

    if scale > 1
        h, w = size(img_array)
        new_h, new_w = h ÷ scale, w ÷ scale
        img_small = imresize(img_array, (new_h, new_w))
    else
        img_small = img_array
    end

    # PURE Otsu - no adjustment
    threshold = otsu_threshold(img_small)
    pore_mask = img_small .< threshold

    labels = label_components(pore_mask)
    n_pores = maximum(labels)

    if n_pores == 0
        return 0.0, Float64[]
    end

    pore_diameters = Float64[]
    for i in 1:n_pores
        area = sum(labels .== i)
        if area > 10  # Minimal noise filter only
            diameter = 2.0 * sqrt(area / π)
            push!(pore_diameters, diameter)
        end
    end

    if isempty(pore_diameters)
        return 0.0, Float64[]
    end

    effective_pixel_size = PIXEL_SIZE_UM * scale
    pore_diameters_um = pore_diameters .* effective_pixel_size

    return mean(pore_diameters_um), pore_diameters_um
end

# Load all ground truth
println("\n1. Loading ground truth...")
all_gt = Float64[]
for sample in ["S1_27x", "S2_27x", "S3_27x"]
    analysis_path = joinpath(DATA_DIR, "$(sample)_analysis.xlsx")
    if isfile(analysis_path)
        gt = load_ground_truth(analysis_path)
        append!(all_gt, gt)
        @printf("   %s: %d measurements, mean %.1f μm\n", sample, length(gt), mean(gt))
    end
end
@printf("   Total GT: %d measurements, mean %.1f ± %.1f μm\n",
        length(all_gt), mean(all_gt), std(all_gt))

# Process images with HONEST algorithm
println("\n2. Darwin analysis (PURE Otsu, no tuning)...")
all_darwin = Float64[]
for sample in ["S1_27x", "S2_27x", "S3_27x"]
    image_path = joinpath(DATA_DIR, "$(sample).tif")
    if isfile(image_path)
        darwin_mean, darwin_pores = compute_pore_size_honest(image_path, scale=4)
        append!(all_darwin, darwin_pores)
        @printf("   %s: %d pores, mean %.1f μm\n", sample, length(darwin_pores), darwin_mean)
    end
end

println("\n" * "=" ^ 60)
println("HONEST RESULTS")
println("=" ^ 60)

darwin_mean = mean(all_darwin)
darwin_median = median(all_darwin)
darwin_std = std(all_darwin)

gt_mean = mean(all_gt)
gt_median = median(all_gt)
gt_std = std(all_gt)

# APE (what we were using - less rigorous)
ape_mean = abs(darwin_mean - gt_mean) / gt_mean * 100
ape_median = abs(darwin_median - gt_median) / gt_median * 100

@printf("\nDarwin:       %.1f ± %.1f μm (median: %.1f)\n", darwin_mean, darwin_std, darwin_median)
@printf("Ground Truth: %.1f ± %.1f μm (median: %.1f)\n", gt_mean, gt_std, gt_median)

println("\n--- Metrics (HONEST) ---")
@printf("APE (mean):   %.1f%%\n", ape_mean)
@printf("APE (median): %.1f%%\n", ape_median)

# Bias direction
if darwin_mean < gt_mean
    @printf("\nBIAS: Darwin UNDERESTIMATES by %.1f μm (%.1f%%)\n",
            gt_mean - darwin_mean, (gt_mean - darwin_mean)/gt_mean * 100)
else
    @printf("\nBIAS: Darwin OVERESTIMATES by %.1f μm (%.1f%%)\n",
            darwin_mean - gt_mean, (darwin_mean - gt_mean)/gt_mean * 100)
end

# Distribution comparison
println("\n--- Distribution Analysis ---")
@printf("Darwin range:  %.0f - %.0f μm\n", minimum(all_darwin), maximum(all_darwin))
@printf("GT range:      %.0f - %.0f μm\n", minimum(all_gt), maximum(all_gt))
@printf("Darwin IQR:    %.0f - %.0f μm\n", quantile(all_darwin, 0.25), quantile(all_darwin, 0.75))
@printf("GT IQR:        %.0f - %.0f μm\n", quantile(all_gt, 0.25), quantile(all_gt, 0.75))

# Sample size warning
println("\n--- Limitations ---")
println("WARNING: Only 3 SEM images analyzed")
println("WARNING: Ground truth assignment to images uncertain")
println("WARNING: 2D SEM analysis, not 3D microCT")

# Honest comparison to PoreScript
println("\n--- Honest PoreScript Comparison ---")
println("PoreScript MAPE: 15.5% (mean of individual pore errors)")
println("Darwin APE: $(round(ape_mean, digits=1))% (error of mean values)")
println("")
println("THESE ARE DIFFERENT METRICS - not directly comparable!")
println("For fair comparison, would need to calculate Darwin MAPE")
println("against paired pore-by-pore measurements (not available).")

# Final honest assessment
println("\n" * "=" ^ 60)
println("HONEST ASSESSMENT")
println("=" ^ 60)
println("""
Darwin's pore size measurement shows:
- $(round(ape_mean, digits=1))% error on mean pore size (without tuning)
- Systematic $(darwin_mean < gt_mean ? "under" : "over")estimation
- High variance (σ = $(round(darwin_std, digits=1)) μm vs GT σ = $(round(gt_std, digits=1)) μm)

For a first paper, this is:
- ACCEPTABLE for comparing scaffold designs
- NOT gold-standard for absolute measurements
- Needs validation on more diverse datasets

Recommendation: Report as "preliminary validation" with limitations.
""")
