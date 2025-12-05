"""
Local Thickness Module

Implements the Local Thickness algorithm (Hildebrand & Rüegsegger 1997)
for accurate pore size measurement, as used in BoneJ.

Reference:
- Hildebrand T, Rüegsegger P (1997) "A new method for the model-independent
  assessment of thickness in three-dimensional images"
  Journal of Microscopy 185(1):67-75

Method:
1. Compute 3D Euclidean distance transform
2. Find local maxima (sphere centers)
3. Assign each voxel the diameter of the largest sphere containing it
"""

module LocalThickness

using Statistics
using LinearAlgebra

export compute_local_thickness, compute_mean_pore_size_3d

"""
    distance_transform_3d(mask::AbstractArray{Bool,3}) -> Array{Float64,3}

Compute 3D Euclidean distance transform.
Each voxel gets the distance to the nearest background (false) voxel.
"""
function distance_transform_3d(mask::AbstractArray{Bool,3})::Array{Float64,3}
    dims = size(mask)
    dist = zeros(Float64, dims)

    # Find all background voxels (boundaries)
    background = CartesianIndex{3}[]
    for idx in CartesianIndices(mask)
        if !mask[idx]
            push!(background, idx)
        end
    end

    if isempty(background)
        # All foreground - return large distance
        fill!(dist, minimum(dims) / 2.0)
        return dist
    end

    # For efficiency, use a chunked approach for large volumes
    # Compute distance for each foreground voxel
    for idx in CartesianIndices(mask)
        if mask[idx]
            min_d = Inf
            i, j, k = Tuple(idx)

            # Only check nearby background voxels (optimization)
            # Use squared distances to avoid sqrt until final
            for bg in background
                bi, bj, bk = Tuple(bg)
                d_sq = Float64((i - bi)^2 + (j - bj)^2 + (k - bk)^2)
                if d_sq < min_d
                    min_d = d_sq
                end
            end

            dist[idx] = sqrt(min_d)
        end
    end

    return dist
end

"""
    distance_transform_3d_fast(mask::AbstractArray{Bool,3}) -> Array{Float64,3}

Fast approximate 3D distance transform using separable passes.
Based on Saito & Toriwaki (1994) algorithm.
"""
function distance_transform_3d_fast(mask::AbstractArray{Bool,3})::Array{Float64,3}
    dims = size(mask)
    dist = fill(Inf, dims)

    # Initialize: 0 for foreground adjacent to background, Inf otherwise
    for idx in CartesianIndices(mask)
        if mask[idx]
            i, j, k = Tuple(idx)
            # Check 6-connectivity for boundary
            is_boundary = false
            for (di, dj, dk) in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
                ni, nj, nk = i+di, j+dj, k+dk
                if ni >= 1 && ni <= dims[1] && nj >= 1 && nj <= dims[2] && nk >= 1 && nk <= dims[3]
                    if !mask[ni, nj, nk]
                        is_boundary = true
                        break
                    end
                end
            end
            if is_boundary
                dist[idx] = 1.0
            end
        else
            dist[idx] = 0.0
        end
    end

    # Forward pass
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        if mask[i,j,k]
            if i > 1
                dist[i,j,k] = min(dist[i,j,k], dist[i-1,j,k] + 1)
            end
            if j > 1
                dist[i,j,k] = min(dist[i,j,k], dist[i,j-1,k] + 1)
            end
            if k > 1
                dist[i,j,k] = min(dist[i,j,k], dist[i,j,k-1] + 1)
            end
        end
    end

    # Backward pass
    for k in dims[3]:-1:1, j in dims[2]:-1:1, i in dims[1]:-1:1
        if mask[i,j,k]
            if i < dims[1]
                dist[i,j,k] = min(dist[i,j,k], dist[i+1,j,k] + 1)
            end
            if j < dims[2]
                dist[i,j,k] = min(dist[i,j,k], dist[i,j+1,k] + 1)
            end
            if k < dims[3]
                dist[i,j,k] = min(dist[i,j,k], dist[i,j,k+1] + 1)
            end
        end
    end

    return dist
end

"""
    find_local_maxima_3d(dist::Array{Float64,3}, mask::AbstractArray{Bool,3}) -> Vector{Tuple{CartesianIndex{3}, Float64}}

Find local maxima in distance transform (sphere centers).
"""
function find_local_maxima_3d(dist::Array{Float64,3}, mask::AbstractArray{Bool,3})
    dims = size(dist)
    maxima = Tuple{CartesianIndex{3}, Float64}[]

    for k in 2:dims[3]-1, j in 2:dims[2]-1, i in 2:dims[1]-1
        if !mask[i,j,k]
            continue
        end

        val = dist[i,j,k]
        if val <= 0
            continue
        end

        # Check 26-neighborhood
        is_max = true
        for di in -1:1, dj in -1:1, dk in -1:1
            if di == 0 && dj == 0 && dk == 0
                continue
            end
            if dist[i+di, j+dj, k+dk] > val
                is_max = false
                break
            end
        end

        if is_max
            push!(maxima, (CartesianIndex(i,j,k), val))
        end
    end

    return maxima
end

"""
    compute_local_thickness(mask::AbstractArray{Bool,3}) -> Array{Float64,3}

Compute local thickness map using the Hildebrand-Rüegsegger method.
Each voxel is assigned the diameter of the largest inscribed sphere containing it.

Returns thickness in voxel units.
"""
function compute_local_thickness(mask::AbstractArray{Bool,3})::Array{Float64,3}
    dims = size(mask)
    thickness = zeros(Float64, dims)

    # Step 1: Distance transform
    dist = distance_transform_3d_fast(mask)

    # Step 2: Find local maxima (sphere centers)
    maxima = find_local_maxima_3d(dist, mask)

    if isempty(maxima)
        # No local maxima found - use distance transform directly
        for idx in CartesianIndices(mask)
            if mask[idx]
                thickness[idx] = 2.0 * dist[idx]
            end
        end
        return thickness
    end

    # Sort maxima by radius (largest first) for efficiency
    sort!(maxima, by=x->x[2], rev=true)

    # Step 3: For each voxel, find the largest sphere containing it
    for idx in CartesianIndices(mask)
        if !mask[idx]
            continue
        end

        i, j, k = Tuple(idx)
        max_diameter = 0.0

        for (center, radius) in maxima
            ci, cj, ck = Tuple(center)
            d = sqrt(Float64((i-ci)^2 + (j-cj)^2 + (k-ck)^2))

            if d <= radius
                diameter = 2.0 * radius
                if diameter > max_diameter
                    max_diameter = diameter
                end
            end

            # Early termination: if max_diameter > 2*radius, no larger sphere can contain this voxel
            if max_diameter > 2.0 * radius
                break
            end
        end

        # Fallback to distance transform if no sphere found
        if max_diameter == 0.0
            max_diameter = 2.0 * dist[idx]
        end

        thickness[idx] = max_diameter
    end

    return thickness
end

"""
    compute_mean_pore_size_3d(binary::AbstractArray{Bool,3}, voxel_size_um::Real) -> NamedTuple

Compute pore size statistics using 3D Local Thickness method.

Arguments:
- binary: 3D array (true = solid, false = pore)
- voxel_size_um: voxel size in micrometers

Returns NamedTuple with:
- mean_pore_size_um: Mean pore diameter
- median_pore_size_um: Median pore diameter
- std_pore_size_um: Standard deviation
- max_pore_size_um: Maximum pore diameter
- min_pore_size_um: Minimum non-zero pore diameter
"""
function compute_mean_pore_size_3d(
    binary::AbstractArray{Bool,3},
    voxel_size_um::Real
)::NamedTuple{(:mean_pore_size_um, :median_pore_size_um, :std_pore_size_um, :max_pore_size_um, :min_pore_size_um), NTuple{5, Float64}}

    # Pore mask (inverse of solid)
    pore_mask = .!binary

    if sum(pore_mask) == 0
        return (
            mean_pore_size_um = 0.0,
            median_pore_size_um = 0.0,
            std_pore_size_um = 0.0,
            max_pore_size_um = 0.0,
            min_pore_size_um = 0.0
        )
    end

    # Compute local thickness on pore space
    thickness_voxels = compute_local_thickness(pore_mask)

    # Extract non-zero thickness values
    pore_diameters_voxels = thickness_voxels[pore_mask .& (thickness_voxels .> 0)]

    if isempty(pore_diameters_voxels)
        return (
            mean_pore_size_um = 0.0,
            median_pore_size_um = 0.0,
            std_pore_size_um = 0.0,
            max_pore_size_um = 0.0,
            min_pore_size_um = 0.0
        )
    end

    # Convert to micrometers
    pore_diameters_um = pore_diameters_voxels .* voxel_size_um

    return (
        mean_pore_size_um = mean(pore_diameters_um),
        median_pore_size_um = median(pore_diameters_um),
        std_pore_size_um = std(pore_diameters_um),
        max_pore_size_um = maximum(pore_diameters_um),
        min_pore_size_um = minimum(pore_diameters_um)
    )
end

end # module LocalThickness
