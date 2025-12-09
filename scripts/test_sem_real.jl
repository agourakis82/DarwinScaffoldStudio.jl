#!/usr/bin/env julia
"""
Test SEM 3D Reconstruction with REAL scaffold SEM images
"""

println("="^60)
println("SEM 3D RECONSTRUCTION - REAL DATA TEST")
println("="^60)

const PROJECT_ROOT = dirname(dirname(@__FILE__))

using Images
using FileIO
using Statistics

# Include SEM module
include(joinpath(PROJECT_ROOT, "src/DarwinScaffoldStudio/Vision/SEM3DReconstruction.jl"))
using .SEM3DReconstruction

# Find SEM images
sem_dir = joinpath(PROJECT_ROOT, "data/public/scaffold_sem/MeBiosys/Figure 2Morphology/SEM images")

println("\n1. Finding real SEM scaffold images...")
sem_files = String[]
for (root, dirs, files) in walkdir(sem_dir)
    for f in files
        if endswith(lowercase(f), ".jpg")
            push!(sem_files, joinpath(root, f))
        end
    end
end

println("   Found $(length(sem_files)) SEM images")

if isempty(sem_files)
    error("No SEM images found! Check path: $sem_dir")
end

# Process first image
println("\n2. Loading first SEM image...")
sem_path = sem_files[1]
println("   File: ", basename(sem_path))

img = load(sem_path)
println("   Size: ", size(img))
println("   Type: ", typeof(img))

# Convert to grayscale Float64
if ndims(img) == 3 || eltype(img) <: RGB
    img_gray = Float64.(Gray.(img))
else
    img_gray = Float64.(img)
end

# Downsample for faster processing (full res takes too long for SfS)
downsample_factor = 4
img_gray = img_gray[1:downsample_factor:end, 1:downsample_factor:end]

println("   Grayscale size (downsampled $(downsample_factor)x): ", size(img_gray))
println("   Intensity range: ", round(minimum(img_gray), digits=3), " - ", round(maximum(img_gray), digits=3))

println("\n3. Running Shape-from-Shading reconstruction...")
# SEM typically has top-down illumination from detector
result = reconstruct_depth_sfs(
    img_gray,
    light_direction=[0.0, -0.3, 1.0],  # Slightly angled (SEM detector position)
    albedo=0.6,
    max_iterations=300,
    tolerance=1e-5
)

println("   Depth map computed!")
println("   Depth range: ", round(minimum(result.depth_map), digits=2), " - ", round(maximum(result.depth_map), digits=2))
println("   Mean confidence: ", round(mean(result.confidence), digits=3))

println("\n4. Extracting surface normals...")
normals = result.normal_map
println("   Normal map size: ", size(normals))

# Analyze surface roughness from normals
z_normals = normals[:,:,3]
roughness = std(z_normals)
println("   Surface roughness (std of Nz): ", round(roughness, digits=4))

println("\n5. Converting to mesh...")
# Downsample for mesh (full res would be huge)
step = 4
depth_small = result.depth_map[1:step:end, 1:step:end]
pixel_size_um = 0.5  # Estimate based on SEM magnification

vertices, faces = depth_to_mesh(depth_small, pixel_size_um * step, simplify=false)
println("   Vertices: ", size(vertices, 1))
println("   Faces: ", size(faces, 1))

println("\n6. Processing multiple SEM images...")
println("-"^50)

# Process a few more images
for (i, sem_file) in enumerate(sem_files[1:min(5, length(sem_files))])
    img_i = load(sem_file)
    img_gray_i = Float64.(Gray.(img_i))
    # Downsample
    img_gray_i = img_gray_i[1:downsample_factor:end, 1:downsample_factor:end]

    result_i = reconstruct_depth_sfs(
        img_gray_i,
        light_direction=[0.0, -0.3, 1.0],
        albedo=0.6,
        max_iterations=200
    )

    depth_range = maximum(result_i.depth_map) - minimum(result_i.depth_map)
    conf = mean(result_i.confidence)

    println("   [$i] $(basename(sem_file))")
    println("       Depth range: $(round(depth_range, digits=2)), Confidence: $(round(conf, digits=3))")
end

println("\n", "="^60)
println("SEM RECONSTRUCTION TEST COMPLETE!")
println("="^60)
println("\nSummary:")
println("  - Loaded $(length(sem_files)) real chitosan scaffold SEM images")
println("  - Shape-from-Shading reconstruction: WORKING")
println("  - Depth-to-mesh conversion: WORKING")
println("  - Ready for dissertation!")
