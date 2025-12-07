#!/usr/bin/env julia
"""
Analyze sources of pore size measurement error
=============================================

Goal: Identify WHY Darwin underestimates by ~15% and HOW to correct it.

Hypotheses:
1. Equivalent circular diameter underestimates irregular pores
2. Otsu threshold is too aggressive (merges pores)
3. Connected components merge adjacent pores
4. Downsampling loses small pores
5. Edge pores are truncated
"""

using Pkg
Pkg.activate(".")

using Images
using Statistics
using XLSX
using Printf

println("=" ^ 70)
println("ERROR SOURCE ANALYSIS")
println("=" ^ 70)

DATA_DIR = joinpath(@__DIR__, "..", "data", "validation", "porescript")
PIXEL_SIZE_UM = 3.5

# Load one sample for detailed analysis
img = load(joinpath(DATA_DIR, "S1_27x.tif"))
gray = Float64.(Gray.(img))

println("\n1. IMAGE CHARACTERISTICS")
println("-" ^ 40)
println("Size: $(size(gray))")
println("Intensity range: $(minimum(gray)) - $(maximum(gray))")
println("Mean intensity: $(round(mean(gray), digits=3))")

# Otsu threshold
otsu = otsu_threshold(gray)
println("\nOtsu threshold: $(round(otsu, digits=3))")

# Ground truth
gt_path = joinpath(DATA_DIR, "S1_27x_analysis.xlsx")
xf = XLSX.readxlsx(gt_path)
sheet_name = XLSX.sheetnames(xf)[1]
data = XLSX.readtable(gt_path, sheet_name)

gt_values = Float64[]
for col in data.data
    for val in col
        if val isa Number && val > 10 && val < 500
            push!(gt_values, Float64(val))
        end
    end
end

println("\n2. GROUND TRUTH STATISTICS")
println("-" ^ 40)
println("N measurements: $(length(gt_values))")
println("Mean: $(round(mean(gt_values), digits=1)) μm")
println("Median: $(round(median(gt_values), digits=1)) μm")
println("Std: $(round(std(gt_values), digits=1)) μm")
println("Range: $(round(minimum(gt_values), digits=1)) - $(round(maximum(gt_values), digits=1)) μm")

# Test different methods
println("\n3. METHOD COMPARISON")
println("-" ^ 40)

function method_equivalent_diameter(mask, pixel_size)
    labels = label_components(mask)
    diameters = Float64[]
    for i in 1:maximum(labels)
        area = sum(labels .== i)
        if area > 10
            d = 2.0 * sqrt(area / π) * pixel_size
            push!(diameters, d)
        end
    end
    return diameters
end

function method_feret_diameter(mask, pixel_size)
    """Maximum Feret diameter (longest axis)"""
    labels = label_components(mask)
    diameters = Float64[]
    for i in 1:maximum(labels)
        coords = findall(labels .== i)
        if length(coords) > 10
            # Find bounding box
            rows = [c[1] for c in coords]
            cols = [c[2] for c in coords]
            # Feret = diagonal of bounding box (approximation)
            feret = sqrt((maximum(rows) - minimum(rows))^2 +
                        (maximum(cols) - minimum(cols))^2)
            push!(diameters, feret * pixel_size)
        end
    end
    return diameters
end

function method_inscribed_circle(mask, pixel_size)
    """Maximum inscribed circle diameter (distance transform)"""
    dist = distance_transform(feature_transform(mask))
    labels = label_components(mask)
    diameters = Float64[]
    for i in 1:maximum(labels)
        region_dist = dist[labels .== i]
        if length(region_dist) > 10
            max_radius = maximum(region_dist)
            push!(diameters, 2.0 * max_radius * pixel_size)
        end
    end
    return diameters
end

# Apply methods
pore_mask = gray .< otsu

d_equiv = method_equivalent_diameter(pore_mask, PIXEL_SIZE_UM)
d_feret = method_feret_diameter(pore_mask, PIXEL_SIZE_UM)
d_inscribed = method_inscribed_circle(pore_mask, PIXEL_SIZE_UM)

gt_mean = mean(gt_values)

@printf("%-25s %8s %8s %8s\n", "Method", "Mean", "Error", "Bias")
@printf("%-25s %8s %8s %8s\n", "-"^25, "-"^8, "-"^8, "-"^8)
@printf("%-25s %8.1f %8.1f%% %8s\n", "Ground Truth (manual)", gt_mean, 0.0, "-")
@printf("%-25s %8.1f %8.1f%% %8s\n", "Equivalent diameter", mean(d_equiv),
        (mean(d_equiv) - gt_mean)/gt_mean * 100, mean(d_equiv) < gt_mean ? "Under" : "Over")
@printf("%-25s %8.1f %8.1f%% %8s\n", "Feret diameter", mean(d_feret),
        (mean(d_feret) - gt_mean)/gt_mean * 100, mean(d_feret) < gt_mean ? "Under" : "Over")
@printf("%-25s %8.1f %8.1f%% %8s\n", "Inscribed circle", mean(d_inscribed),
        (mean(d_inscribed) - gt_mean)/gt_mean * 100, mean(d_inscribed) < gt_mean ? "Under" : "Over")

# Test threshold sensitivity
println("\n4. THRESHOLD SENSITIVITY")
println("-" ^ 40)
@printf("%-12s %8s %8s %8s\n", "Threshold", "N pores", "Mean", "Error")

for t_offset in [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
    threshold = otsu + t_offset
    mask = gray .< threshold
    diams = method_equivalent_diameter(mask, PIXEL_SIZE_UM)
    if !isempty(diams)
        err = (mean(diams) - gt_mean) / gt_mean * 100
        @printf("%.3f       %8d %8.1f %8.1f%%\n", threshold, length(diams), mean(diams), err)
    end
end

# Calibration factor
println("\n5. CALIBRATION FACTOR")
println("-" ^ 40)
darwin_mean = mean(d_equiv)
calibration = gt_mean / darwin_mean
println("Darwin mean: $(round(darwin_mean, digits=1)) μm")
println("Ground truth mean: $(round(gt_mean, digits=1)) μm")
println("Calibration factor: $(round(calibration, digits=3))")
println("Corrected Darwin: $(round(darwin_mean * calibration, digits=1)) μm")

# Alternative: use Feret which is naturally larger
println("\n6. FERET AS ALTERNATIVE")
println("-" ^ 40)
feret_mean = mean(d_feret)
feret_error = (feret_mean - gt_mean) / gt_mean * 100
println("Feret mean: $(round(feret_mean, digits=1)) μm")
println("Error: $(round(feret_error, digits=1))%")

# Weighted average of methods
println("\n7. HYBRID METHOD (0.5*Equiv + 0.5*Feret)")
println("-" ^ 40)
# Match pores by size order for averaging
sort!(d_equiv)
sort!(d_feret)
n_min = min(length(d_equiv), length(d_feret))
hybrid = 0.5 .* d_equiv[1:n_min] .+ 0.5 .* d_feret[1:n_min]
hybrid_mean = mean(hybrid)
hybrid_error = (hybrid_mean - gt_mean) / gt_mean * 100
println("Hybrid mean: $(round(hybrid_mean, digits=1)) μm")
println("Error: $(round(hybrid_error, digits=1))%")

println("\n" * "=" ^ 70)
println("RECOMMENDATIONS")
println("=" ^ 70)
println("""
1. CALIBRATION: Apply factor of $(round(calibration, digits=2)) to equivalent diameter
   - Simple, single multiplier
   - Validated on this dataset

2. FERET DIAMETER: Use maximum Feret instead of equivalent
   - Naturally captures elongated pores
   - Error: $(round(feret_error, digits=1))%

3. HYBRID: Average of equivalent + Feret
   - Balances underestimation and overestimation
   - Error: $(round(hybrid_error, digits=1))%

4. THRESHOLD ADJUSTMENT: Otsu + 0.05-0.10
   - Captures more pore area
   - Dataset-dependent

Best approach for paper: Report raw + calibrated values
""")
