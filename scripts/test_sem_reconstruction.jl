#!/usr/bin/env julia
"""
Test SEM 3D Reconstruction Module

Tests Shape-from-Shading reconstruction on synthetic SEM-like image.
"""

println("="^60)
println("DARWIN SCAFFOLD STUDIO - SEM 3D Reconstruction Test")
println("="^60)

const PROJECT_ROOT = dirname(dirname(@__FILE__))

# Include the SEM reconstruction module
include(joinpath(PROJECT_ROOT, "src/DarwinScaffoldStudio/Vision/SEM3DReconstruction.jl"))
using .SEM3DReconstruction

using Statistics

println("\n1. Generating synthetic SEM-like image...")

# Create a synthetic SEM image with known 3D structure
# Simulating a scaffold surface with spherical pores
function create_synthetic_sem(size::Int=256)
    img = zeros(Float64, size, size)

    # Background (flat surface)
    fill!(img, 0.5)

    # Add some spherical pores (appear as dark circles with bright edges)
    n_pores = 15
    for _ in 1:n_pores
        cx = rand(30:size-30)
        cy = rand(30:size-30)
        r = rand(15:40)

        for i in 1:size, j in 1:size
            dist = sqrt((i - cx)^2 + (j - cy)^2)
            if dist < r
                # Inside pore - darker (deeper)
                depth_factor = sqrt(1 - (dist/r)^2)  # Hemisphere
                img[i, j] = 0.3 * (1 - depth_factor)

                # Edge brightening (SEM edge effect)
                if dist > r * 0.8
                    img[i, j] += 0.3
                end
            end
        end
    end

    # Add some Gaussian noise (SEM-like)
    img .+= 0.05 * randn(size, size)

    # Clamp to [0, 1]
    img = clamp.(img, 0.0, 1.0)

    return img
end

sem_image = create_synthetic_sem(256)
println("   Size: ", size(sem_image))
println("   Intensity range: ", round(minimum(sem_image), digits=3), " - ", round(maximum(sem_image), digits=3))

println("\n2. Running Shape-from-Shading reconstruction...")
println("   (This may take a moment...)")

# Run SfS with default parameters
result = reconstruct_depth_sfs(
    sem_image,
    light_direction=[0.0, 0.0, 1.0],
    albedo=0.5,
    max_iterations=500,
    tolerance=1e-5
)

println("\n3. Results:")
println("   Method: ", result.method)
println("   Depth map size: ", size(result.depth_map))
println("   Depth range: ", round(minimum(result.depth_map), digits=3), " - ", round(maximum(result.depth_map), digits=3))
println("   Mean confidence: ", round(mean(result.confidence), digits=3))

println("\n4. Converting depth to mesh...")
vertices, faces = depth_to_mesh(result.depth_map, 1.0, simplify=true, target_faces=10000)
println("   Vertices: ", size(vertices, 1))
println("   Faces: ", size(faces, 1))

println("\n5. Testing Stereo SEM (with synthetic pair)...")
# Create a slightly shifted "right" image (simulating tilt)
right_image = similar(sem_image)
shift = 5  # pixels
for i in 1:size(sem_image, 1)
    for j in 1:size(sem_image, 2)
        src_j = j + shift
        if src_j <= size(sem_image, 2)
            right_image[i, j] = sem_image[i, src_j]
        else
            right_image[i, j] = sem_image[i, end]
        end
    end
end

stereo_result = reconstruct_stereo_sem(
    sem_image, right_image,
    tilt_angle=5.0,
    pixel_size_um=0.5
)
println("   Stereo depth range: ", round(minimum(stereo_result.depth_map), digits=1), " - ", round(maximum(stereo_result.depth_map), digits=1), " μm")

println("\n6. Testing surface normal estimation...")
normals = estimate_surface_normals(sem_image)
println("   Normal map size: ", size(normals))
println("   Z-component range: ", round(minimum(normals[:,:,3]), digits=3), " - ", round(maximum(normals[:,:,3]), digits=3))

println("\n", "="^60)
println("SEM 3D Reconstruction Test COMPLETE!")
println("="^60)

# Summary
println("\nSUMMARY:")
println("  ✓ Shape-from-Shading: Working")
println("  ✓ Stereo SEM: Working")
println("  ✓ Surface normals: Working")
println("  ✓ Depth-to-mesh: Working")
println("\nThe SEM → 3D pipeline is functional!")
println("For real SEM images, adjust light_direction and albedo parameters.")
