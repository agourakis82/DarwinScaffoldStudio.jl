#!/usr/bin/env julia
"""
Generate varied synthetic scaffolds with different structures
for comprehensive validation with large N
"""

println("="^70)
println("GENERATING VARIED SYNTHETIC SCAFFOLDS")
println("="^70)

const PROJECT_ROOT = dirname(dirname(@__FILE__))

using Statistics
using Random
using JSON
using Printf

Random.seed!(42)

output_dir = joinpath(PROJECT_ROOT, "data/validation/synthetic_volumes")
mkpath(output_dir)

ground_truth = Dict[]

# ============================================================================
# SCAFFOLD GENERATOR FUNCTIONS
# ============================================================================

"""Generate gyroid-like scaffold"""
function generate_gyroid(size::Int, period::Float64, threshold::Float64)
    volume = zeros(Bool, size, size, size)
    for x in 1:size, y in 1:size, z in 1:size
        px = 2π * x / period
        py = 2π * y / period
        pz = 2π * z / period
        val = sin(px)*cos(py) + sin(py)*cos(pz) + sin(pz)*cos(px)
        volume[x,y,z] = val > threshold
    end
    return volume
end

"""Generate cubic lattice scaffold"""
function generate_cubic_lattice(size::Int, strut_width::Int, spacing::Int)
    volume = zeros(Bool, size, size, size)
    for x in 1:size, y in 1:size, z in 1:size
        in_x = (y % spacing) < strut_width && (z % spacing) < strut_width
        in_y = (x % spacing) < strut_width && (z % spacing) < strut_width
        in_z = (x % spacing) < strut_width && (y % spacing) < strut_width
        volume[x,y,z] = in_x || in_y || in_z
    end
    return volume
end

"""Generate random porous structure"""
function generate_random_porous(size::Int, solid_fraction::Float64)
    volume = rand(size, size, size) .< solid_fraction
    # Smooth
    for _ in 1:2
        new_vol = copy(volume)
        for x in 2:size-1, y in 2:size-1, z in 2:size-1
            neighbors = sum(volume[x-1:x+1, y-1:y+1, z-1:z+1])
            new_vol[x,y,z] = neighbors > 13
        end
        volume = new_vol
    end
    return volume
end

"""Generate trabecular bone-like structure"""
function generate_trabecular(size::Int, thickness::Int, spacing::Int)
    volume = zeros(Bool, size, size, size)
    for x in 1:size, y in 1:size, z in 1:size
        plate_xy = (z % spacing) < thickness
        plate_xz = (y % spacing) < thickness
        plate_yz = (x % spacing) < thickness
        volume[x,y,z] = plate_xy || plate_xz || plate_yz
    end
    return volume
end

"""Save volume and add to ground truth"""
function save_volume!(volume, name, voxel_um, extra_params, ground_truth, output_dir)
    global sample_id
    sample_id += 1

    porosity = 1.0 - sum(volume) / length(volume)
    filename = "$(name)_$(lpad(sample_id, 3, '0')).raw"
    filepath = joinpath(output_dir, filename)

    write(filepath, UInt8.(volume))

    entry = Dict(
        "id" => sample_id,
        "type" => name,
        "filename" => filename,
        "size" => collect(size(volume)),
        "porosity" => porosity,
        "voxel_size_um" => voxel_um
    )
    merge!(entry, extra_params)
    push!(ground_truth, entry)

    return porosity
end

# ============================================================================
# GENERATE ALL SAMPLES
# ============================================================================

sample_id = 0

println("\nGenerating varied scaffold types...")

# 1. Gyroid scaffolds (15 samples)
println("\n1. Gyroid scaffolds...")
for period in [8.0, 10.0, 12.0, 15.0, 20.0]
    for threshold in [-0.3, 0.0, 0.3]
        volume = generate_gyroid(64, period, threshold)
        por = save_volume!(volume, "gyroid", 10.0,
            Dict("period"=>period, "threshold"=>threshold), ground_truth, output_dir)
        println("   [$sample_id] period=$period, threshold=$threshold -> porosity=$(@sprintf("%.1f", por*100))%")
    end
end

# 2. Cubic lattice scaffolds (12 samples)
println("\n2. Cubic lattice scaffolds...")
for strut in [2, 3, 4, 5]
    for spacing in [8, 10, 12, 16]
        if strut < spacing
            volume = generate_cubic_lattice(64, strut, spacing)
            por = save_volume!(volume, "cubic", 15.0,
                Dict("strut"=>strut, "spacing"=>spacing), ground_truth, output_dir)
            println("   [$sample_id] strut=$strut, spacing=$spacing -> porosity=$(@sprintf("%.1f", por*100))%")
        end
    end
end

# 3. Random porous structures (10 samples)
println("\n3. Random porous structures...")
for solid in 0.1:0.1:0.9
    volume = generate_random_porous(64, solid)
    por = save_volume!(volume, "random", 20.0,
        Dict("target_solid"=>solid), ground_truth, output_dir)
    println("   [$sample_id] target_solid=$(@sprintf("%.0f", solid*100))% -> porosity=$(@sprintf("%.1f", por*100))%")
end

# 4. Trabecular bone-like structures (9 samples)
println("\n4. Trabecular bone-like structures...")
for thickness in [2, 3, 4]
    for spacing in [6, 8, 10]
        if thickness < spacing
            volume = generate_trabecular(64, thickness, spacing)
            por = save_volume!(volume, "trabecular", 18.0,
                Dict("thickness"=>thickness, "spacing"=>spacing), ground_truth, output_dir)
            println("   [$sample_id] thickness=$thickness, spacing=$spacing -> porosity=$(@sprintf("%.1f", por*100))%")
        end
    end
end

# Save ground truth JSON
gt_file = joinpath(output_dir, "ground_truth.json")
open(gt_file, "w") do io
    JSON.print(io, ground_truth, 2)
end

# Summary
println("\n" * "="^70)
println("GENERATION COMPLETE!")
println("="^70)
println("\n📊 Summary:")
println("   Total samples generated: $sample_id")
println("   Scaffold types: Gyroid, Cubic, Random, Trabecular")
println("   Output: $output_dir")
println("   Ground truth: $gt_file")

porosities = [gt["porosity"] for gt in ground_truth]
println("\n   Porosity range: $(@sprintf("%.1f", minimum(porosities)*100))% - $(@sprintf("%.1f", maximum(porosities)*100))%")
println("   Mean porosity: $(@sprintf("%.1f", mean(porosities)*100))%")
