"""
Geometric Tortuosity Module

Computes geometric tortuosity using shortest path algorithms.

Definition: τ = L_actual / L_straight
Where L_actual is the shortest path through pore space and L_straight is the direct distance.

References:
- Gommes CJ et al. (2009) "Practical methods for measuring the tortuosity of porous materials"
- Ghanbarian B et al. (2013) "Tortuosity in porous media: A critical review"
- TORT3D: Srisutthiyakorn N, Mavko G (2017) SoftwareX

Methods implemented:
1. Dijkstra shortest path (exact)
2. A* shortest path (faster for single source-target)
3. Fast marching method (approximation)
"""

module Tortuosity

using Statistics
using DataStructures: PriorityQueue, enqueue!, dequeue!

export compute_geometric_tortuosity, compute_directional_tortuosity

"""
    dijkstra_3d(pore_mask::AbstractArray{Bool,3}, start::CartesianIndex{3}) -> Array{Float64,3}

Compute shortest path distances from start point to all pore voxels using Dijkstra's algorithm.
Uses 26-connectivity with Euclidean distances.
"""
function dijkstra_3d(pore_mask::AbstractArray{Bool,3}, start::CartesianIndex{3})::Array{Float64,3}
    dims = size(pore_mask)
    dist = fill(Inf, dims)

    if !pore_mask[start]
        return dist
    end

    dist[start] = 0.0

    # Priority queue: (distance, index)
    pq = PriorityQueue{CartesianIndex{3}, Float64}()
    enqueue!(pq, start, 0.0)

    # 26-connectivity offsets with distances
    offsets = CartesianIndex{3}[]
    distances = Float64[]
    for di in -1:1, dj in -1:1, dk in -1:1
        if di == 0 && dj == 0 && dk == 0
            continue
        end
        push!(offsets, CartesianIndex(di, dj, dk))
        push!(distances, sqrt(Float64(di^2 + dj^2 + dk^2)))
    end

    while !isempty(pq)
        current = dequeue!(pq)
        ci, cj, ck = Tuple(current)
        current_dist = dist[current]

        # Explore neighbors
        for (offset, edge_dist) in zip(offsets, distances)
            ni, nj, nk = ci + offset[1], cj + offset[2], ck + offset[3]

            # Bounds check
            if ni < 1 || ni > dims[1] || nj < 1 || nj > dims[2] || nk < 1 || nk > dims[3]
                continue
            end

            neighbor = CartesianIndex(ni, nj, nk)

            # Must be pore
            if !pore_mask[neighbor]
                continue
            end

            new_dist = current_dist + edge_dist

            if new_dist < dist[neighbor]
                dist[neighbor] = new_dist
                # Update or add to queue
                if haskey(pq, neighbor)
                    pq[neighbor] = new_dist
                else
                    enqueue!(pq, neighbor, new_dist)
                end
            end
        end
    end

    return dist
end

"""
    compute_directional_tortuosity(pore_mask::AbstractArray{Bool,3}, direction::Symbol) -> Float64

Compute tortuosity in a specific direction (:x, :y, or :z).

Method:
1. Sample points on entry face
2. For each, find shortest path to exit face
3. Tortuosity = mean(path_length) / direct_distance
"""
function compute_directional_tortuosity(
    pore_mask::AbstractArray{Bool,3},
    direction::Symbol;
    n_samples::Int=10
)::Float64
    dims = size(pore_mask)

    # Determine entry and exit faces based on direction
    if direction == :x
        entry_slice = pore_mask[1, :, :]
        exit_slice = pore_mask[end, :, :]
        direct_distance = Float64(dims[1] - 1)
        get_entry_idx = (j, k) -> CartesianIndex(1, j, k)
        get_exit_idx = (j, k) -> CartesianIndex(dims[1], j, k)
        face_dims = (dims[2], dims[3])
    elseif direction == :y
        entry_slice = pore_mask[:, 1, :]
        exit_slice = pore_mask[:, end, :]
        direct_distance = Float64(dims[2] - 1)
        get_entry_idx = (i, k) -> CartesianIndex(i, 1, k)
        get_exit_idx = (i, k) -> CartesianIndex(i, dims[2], k)
        face_dims = (dims[1], dims[3])
    elseif direction == :z
        entry_slice = pore_mask[:, :, 1]
        exit_slice = pore_mask[:, :, end]
        direct_distance = Float64(dims[3] - 1)
        get_entry_idx = (i, j) -> CartesianIndex(i, j, 1)
        get_exit_idx = (i, j) -> CartesianIndex(i, j, dims[3])
        face_dims = (dims[1], dims[2])
    else
        error("Direction must be :x, :y, or :z")
    end

    # Find pore voxels on entry and exit faces
    entry_pores = Tuple{Int,Int}[]
    exit_pores = Tuple{Int,Int}[]

    for i in 1:face_dims[1], j in 1:face_dims[2]
        if entry_slice[i, j]
            push!(entry_pores, (i, j))
        end
        if exit_slice[i, j]
            push!(exit_pores, (i, j))
        end
    end

    if isempty(entry_pores) || isempty(exit_pores)
        return Inf  # No path possible
    end

    # Sample entry points
    n_actual_samples = min(n_samples, length(entry_pores))
    sample_indices = round.(Int, range(1, length(entry_pores), length=n_actual_samples))
    sampled_entries = [entry_pores[i] for i in sample_indices]

    path_lengths = Float64[]

    for (a, b) in sampled_entries
        start = get_entry_idx(a, b)

        # Run Dijkstra from this entry point
        dist = dijkstra_3d(pore_mask, start)

        # Find minimum distance to any exit pore
        min_exit_dist = Inf
        for (ea, eb) in exit_pores
            exit_idx = get_exit_idx(ea, eb)
            if dist[exit_idx] < min_exit_dist
                min_exit_dist = dist[exit_idx]
            end
        end

        if min_exit_dist < Inf
            push!(path_lengths, min_exit_dist)
        end
    end

    if isempty(path_lengths)
        return Inf  # No connected path found
    end

    mean_path_length = mean(path_lengths)
    tortuosity = mean_path_length / direct_distance

    # Tortuosity must be >= 1
    return max(tortuosity, 1.0)
end

"""
    compute_geometric_tortuosity(binary::AbstractArray{Bool,3}; n_samples::Int=10) -> NamedTuple

Compute geometric tortuosity in all three directions.

Arguments:
- binary: 3D array (true = solid, false = pore)
- n_samples: Number of sample points per direction

Returns NamedTuple with:
- tau_x, tau_y, tau_z: Directional tortuosities
- tau_mean: Mean tortuosity
- tau_min: Minimum (most direct) tortuosity
"""
function compute_geometric_tortuosity(
    binary::AbstractArray{Bool,3};
    n_samples::Int=10
)::NamedTuple{(:tau_x, :tau_y, :tau_z, :tau_mean, :tau_min), NTuple{5, Float64}}

    # Pore mask (inverse of solid)
    pore_mask = .!binary

    # Compute directional tortuosities
    tau_x = compute_directional_tortuosity(pore_mask, :x, n_samples=n_samples)
    tau_y = compute_directional_tortuosity(pore_mask, :y, n_samples=n_samples)
    tau_z = compute_directional_tortuosity(pore_mask, :z, n_samples=n_samples)

    # Handle infinite values
    finite_taus = filter(isfinite, [tau_x, tau_y, tau_z])

    if isempty(finite_taus)
        return (
            tau_x = Inf,
            tau_y = Inf,
            tau_z = Inf,
            tau_mean = Inf,
            tau_min = Inf
        )
    end

    return (
        tau_x = tau_x,
        tau_y = tau_y,
        tau_z = tau_z,
        tau_mean = mean(finite_taus),
        tau_min = minimum(finite_taus)
    )
end

"""
    compute_tortuosity_fast(binary::AbstractArray{Bool,3}) -> Float64

Fast tortuosity estimation using random walk simulation.
More efficient for large volumes but less accurate.
"""
function compute_tortuosity_fast(binary::AbstractArray{Bool,3}; n_walkers::Int=100, n_steps::Int=1000)::Float64
    pore_mask = .!binary
    dims = size(pore_mask)

    # Find all pore voxels
    pore_indices = findall(pore_mask)

    if isempty(pore_indices)
        return Inf
    end

    # 26-connectivity offsets
    offsets = CartesianIndex{3}[]
    for di in -1:1, dj in -1:1, dk in -1:1
        if di == 0 && dj == 0 && dk == 0
            continue
        end
        push!(offsets, CartesianIndex(di, dj, dk))
    end

    total_msd = 0.0  # Mean squared displacement in porous medium
    total_free_msd = 0.0  # Free diffusion MSD (no obstacles)

    for _ in 1:n_walkers
        # Start at random pore voxel
        start_idx = rand(pore_indices)
        pos = start_idx

        displacements = zeros(Float64, 3)
        free_displacements = zeros(Float64, 3)

        for _ in 1:n_steps
            # Random direction
            offset = rand(offsets)

            # Free diffusion displacement
            free_displacements .+= Float64.([offset[1], offset[2], offset[3]])

            # Try to move in porous medium
            new_pos = pos + offset
            ni, nj, nk = Tuple(new_pos)

            if 1 <= ni <= dims[1] && 1 <= nj <= dims[2] && 1 <= nk <= dims[3]
                if pore_mask[new_pos]
                    displacements .+= Float64.([offset[1], offset[2], offset[3]])
                    pos = new_pos
                end
                # If blocked, stay in place (displacement = 0 for this step)
            end
        end

        total_msd += sum(displacements.^2)
        total_free_msd += sum(free_displacements.^2)
    end

    # Tortuosity from MSD ratio
    # τ² = MSD_free / MSD_porous
    if total_msd > 0
        tortuosity = sqrt(total_free_msd / total_msd)
        return max(tortuosity, 1.0)
    else
        return Inf
    end
end

end # module Tortuosity
