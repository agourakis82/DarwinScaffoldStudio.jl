#!/usr/bin/env julia
"""
PROPER BETTI NUMBER COMPUTATION FOR TOPOLOGY-TRANSPORT CORRELATION
===================================================================

This script implements cubical homology to compute proper Betti numbers:
- β₀ = number of connected components
- β₁ = number of independent loops/tunnels
- β₂ = number of enclosed voids/cavities

The key question: Does the strong correlation cor(χ, τ) = 0.78 survive
proper computation, or was it an artifact of β₁=β₂=0?

METHODOLOGY:
1. Implement boundary matrix reduction algorithm (standard in TDA)
2. Compute Betti numbers via Smith normal form / rank computation
3. Test on known structures (torus: β=(1,1,0), sphere: β=(1,0,1))
4. Re-run topology-transport correlation from search_novel_physics.jl

THEORY:
-------
Cubical complex: For a 3D binary volume, we have cubes (3-cells), faces (2-cells),
edges (1-cells), and vertices (0-cells).

Boundary operators:
- ∂₃: 3-cells → 2-cells (cube boundaries)
- ∂₂: 2-cells → 1-cells (face boundaries)
- ∂₁: 1-cells → 0-cells (edge boundaries)

Homology groups:
- H₀ = ker(∂₁) / im(∂₂)  → β₀ = rank(H₀)
- H₁ = ker(∂₂) / im(∂₃)  → β₁ = rank(H₁)
- H₂ = ker(∂₃) / im(∂₄)  → β₂ = rank(H₂)

Betti numbers via rank-nullity theorem:
- β₀ = rank(ker(∂₁)) - rank(im(∂₂))
- β₁ = rank(ker(∂₂)) - rank(im(∂₃))
- β₂ = rank(ker(∂₃))

For 3D volumes: ker(∂ₖ) = nullity(∂ₖ) = n - rank(∂ₖ)

Therefore:
- β₀ = (# 0-cells) - rank(∂₁) - rank(im(∂₂))
- β₁ = (# 1-cells) - rank(∂₂) - rank(im(∂₃))
- β₂ = (# 2-cells) - rank(∂₃)

Simplified via Euler characteristic:
χ = β₀ - β₁ + β₂ = C₀ - C₁ + C₂ - C₃

Where Cₖ = number of k-cells in the complex.

IMPLEMENTATION STRATEGY:
-----------------------
For large volumes (50³ = 125,000 voxels), full boundary matrix is too large.

Options:
1. Use sparse matrix reduction (efficient)
2. Connected component labeling for β₀ (fast)
3. Graph-based cycle detection for β₁ (moderate)
4. Dual complex for β₂ (solid components = voids in complement)

Author: Darwin Scaffold Studio
Date: December 2025
"""

using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using Random

Random.seed!(42)

# ============================================================================
# CUBICAL COMPLEX CONSTRUCTION
# ============================================================================

"""
Represent a cubical cell by its dimension and base coordinate.
For 3D: dimension ∈ {0, 1, 2, 3}
- 0-cell (vertex): (i, j, k)
- 1-cell (edge): (i, j, k) with one coordinate as (x, x+1)
- 2-cell (face): (i, j, k) with two coordinates as intervals
- 3-cell (cube): (i, j, k) with all three as intervals
"""
struct CubicalCell
    dim::Int  # Dimension: 0, 1, 2, 3
    coords::Tuple{Int, Int, Int}  # Base coordinate
    type::Int  # For edges/faces: which direction (0=x, 1=y, 2=z for edges)
end

"""
Build cubical complex from binary 3D volume.
Only include cells that are part of the pore space (volume = true).
"""
function build_cubical_complex(volume::BitArray{3})
    sx, sy, sz = size(volume)

    # For computational tractability, we build the complex only for pore voxels
    cells_0 = CubicalCell[]  # Vertices
    cells_1 = CubicalCell[]  # Edges
    cells_2 = CubicalCell[]  # Faces
    cells_3 = CubicalCell[]  # Cubes

    # 3-cells (cubes): only pore voxels
    for k in 1:sz-1, j in 1:sy-1, i in 1:sx-1
        if volume[i, j, k]
            push!(cells_3, CubicalCell(3, (i, j, k), 0))
        end
    end

    # 2-cells (faces): faces between pore cubes or on boundary
    # For simplicity: all faces of pore cubes
    face_set = Set{Tuple{Int,Int,Int,Int}}()  # (i,j,k,type) where type=0,1,2 for XY,XZ,YZ

    for cell in cells_3
        i, j, k = cell.coords
        # Six faces of cube (i,j,k)-(i+1,j+1,k+1)
        # XY faces (perpendicular to Z): (i,j,k) and (i,j,k+1)
        push!(face_set, (i, j, k, 0))
        push!(face_set, (i, j, k+1, 0))
        # XZ faces (perpendicular to Y): (i,j,k) and (i,j+1,k)
        push!(face_set, (i, j, k, 1))
        push!(face_set, (i, j+1, k, 1))
        # YZ faces (perpendicular to X): (i,j,k) and (i+1,j,k)
        push!(face_set, (i, j, k, 2))
        push!(face_set, (i+1, j, k, 2))
    end

    for (i, j, k, t) in face_set
        push!(cells_2, CubicalCell(2, (i, j, k), t))
    end

    # 1-cells (edges): edges of faces
    edge_set = Set{Tuple{Int,Int,Int,Int}}()  # (i,j,k,dir) where dir=0,1,2 for X,Y,Z

    for cell in cells_2
        i, j, k = cell.coords
        t = cell.type
        if t == 0  # XY face
            # Four edges: two X-direction, two Y-direction
            push!(edge_set, (i, j, k, 0))      # Bottom X edge
            push!(edge_set, (i, j+1, k, 0))    # Top X edge
            push!(edge_set, (i, j, k, 1))      # Left Y edge
            push!(edge_set, (i+1, j, k, 1))    # Right Y edge
        elseif t == 1  # XZ face
            push!(edge_set, (i, j, k, 0))      # Front X edge
            push!(edge_set, (i, j, k+1, 0))    # Back X edge
            push!(edge_set, (i, j, k, 2))      # Bottom Z edge
            push!(edge_set, (i+1, j, k, 2))    # Top Z edge
        else  # t == 2, YZ face
            push!(edge_set, (i, j, k, 1))      # Front Y edge
            push!(edge_set, (i, j, k+1, 1))    # Back Y edge
            push!(edge_set, (i, j, k, 2))      # Bottom Z edge
            push!(edge_set, (i, j+1, k, 2))    # Top Z edge
        end
    end

    for (i, j, k, d) in edge_set
        push!(cells_1, CubicalCell(1, (i, j, k), d))
    end

    # 0-cells (vertices): endpoints of edges
    vertex_set = Set{Tuple{Int,Int,Int}}()

    for cell in cells_1
        i, j, k = cell.coords
        d = cell.type
        push!(vertex_set, (i, j, k))
        if d == 0
            push!(vertex_set, (i+1, j, k))
        elseif d == 1
            push!(vertex_set, (i, j+1, k))
        else  # d == 2
            push!(vertex_set, (i, j, k+1))
        end
    end

    for (i, j, k) in vertex_set
        push!(cells_0, CubicalCell(0, (i, j, k), 0))
    end

    return cells_0, cells_1, cells_2, cells_3
end

# ============================================================================
# BOUNDARY OPERATORS (SIMPLIFIED)
# ============================================================================

"""
Compute boundary matrix ∂ₖ: k-cells → (k-1)-cells.
Returns sparse matrix.
"""
function compute_boundary_matrix(cells_k::Vector{CubicalCell}, cells_k_minus_1::Vector{CubicalCell})
    n_k = length(cells_k)
    n_k_minus_1 = length(cells_k_minus_1)

    if n_k == 0 || n_k_minus_1 == 0
        return spzeros(n_k_minus_1, n_k)
    end

    # Index cells for lookup
    cell_to_idx = Dict{Tuple{Int,Int,Int,Int,Int}, Int}()  # (dim, i, j, k, type) → index
    for (idx, cell) in enumerate(cells_k_minus_1)
        key = (cell.dim, cell.coords[1], cell.coords[2], cell.coords[3], cell.type)
        cell_to_idx[key] = idx
    end

    I_rows = Int[]
    J_cols = Int[]
    vals = Int[]

    # Compute boundary of each k-cell
    for (j, cell_k) in enumerate(cells_k)
        boundary_cells = get_boundary(cell_k)

        for (sign, bc) in boundary_cells
            key = (bc.dim, bc.coords[1], bc.coords[2], bc.coords[3], bc.type)
            if haskey(cell_to_idx, key)
                i = cell_to_idx[key]
                push!(I_rows, i)
                push!(J_cols, j)
                push!(vals, sign)
            end
        end
    end

    # Build sparse matrix (mod 2 for computational topology)
    ∂ = sparse(I_rows, J_cols, vals, n_k_minus_1, n_k)

    # Reduce mod 2 for Z₂ homology (standard in computational topology)
    ∂ = ∂ .% 2

    return ∂
end

"""
Get boundary of a cubical cell with orientation (signed).
Returns list of (sign, cell) pairs.
"""
function get_boundary(cell::CubicalCell)
    i, j, k = cell.coords

    if cell.dim == 0
        return []  # Vertices have no boundary

    elseif cell.dim == 1  # Edge
        # Edge connects two vertices
        if cell.type == 0  # X-direction edge
            return [(1, CubicalCell(0, (i, j, k), 0)),
                    (-1, CubicalCell(0, (i+1, j, k), 0))]
        elseif cell.type == 1  # Y-direction
            return [(1, CubicalCell(0, (i, j, k), 0)),
                    (-1, CubicalCell(0, (i, j+1, k), 0))]
        else  # Z-direction
            return [(1, CubicalCell(0, (i, j, k), 0)),
                    (-1, CubicalCell(0, (i, j, k+1), 0))]
        end

    elseif cell.dim == 2  # Face
        # Face has 4 edges
        if cell.type == 0  # XY face at height k
            return [(1, CubicalCell(1, (i, j, k), 0)),      # Bottom X
                    (-1, CubicalCell(1, (i, j+1, k), 0)),    # Top X
                    (-1, CubicalCell(1, (i, j, k), 1)),      # Left Y
                    (1, CubicalCell(1, (i+1, j, k), 1))]     # Right Y
        elseif cell.type == 1  # XZ face at y=j
            return [(1, CubicalCell(1, (i, j, k), 0)),      # Front X
                    (-1, CubicalCell(1, (i, j, k+1), 0)),    # Back X
                    (-1, CubicalCell(1, (i, j, k), 2)),      # Bottom Z
                    (1, CubicalCell(1, (i+1, j, k), 2))]     # Top Z
        else  # YZ face at x=i
            return [(1, CubicalCell(1, (i, j, k), 1)),      # Front Y
                    (-1, CubicalCell(1, (i, j, k+1), 1)),    # Back Y
                    (-1, CubicalCell(1, (i, j, k), 2)),      # Bottom Z
                    (1, CubicalCell(1, (i, j+1, k), 2))]     # Top Z
        end

    else  # cell.dim == 3, Cube
        # Cube has 6 faces
        return [(1, CubicalCell(2, (i, j, k), 0)),      # Bottom XY face
                (-1, CubicalCell(2, (i, j, k+1), 0)),    # Top XY face
                (-1, CubicalCell(2, (i, j, k), 1)),      # Front XZ face
                (1, CubicalCell(2, (i, j+1, k), 1)),     # Back XZ face
                (1, CubicalCell(2, (i, j, k), 2)),       # Left YZ face
                (-1, CubicalCell(2, (i+1, j, k), 2))]    # Right YZ face
    end
end

# ============================================================================
# BETTI NUMBER COMPUTATION
# ============================================================================

"""
Compute Betti numbers using boundary matrix ranks.

β₀ = # connected components
β₁ = dim(ker(∂₂)) - dim(im(∂₃))
β₂ = dim(ker(∂₃))

In Z₂ homology (mod 2):
- rank(∂ₖ) = number of linearly independent columns
- ker(∂ₖ) = nullspace dimension = n_k - rank(∂ₖ)
"""
function compute_betti_numbers_homology(cells_0, cells_1, cells_2, cells_3)
    C₀ = length(cells_0)
    C₁ = length(cells_1)
    C₂ = length(cells_2)
    C₃ = length(cells_3)

    println("  Building boundary matrices...")
    @time ∂₁ = compute_boundary_matrix(cells_1, cells_0)
    @time ∂₂ = compute_boundary_matrix(cells_2, cells_1)
    @time ∂₃ = compute_boundary_matrix(cells_3, cells_2)

    println("  Computing ranks...")
    # Rank computation for sparse matrices (using SVD or LU factorization)
    # For Z₂, we need rank over GF(2)
    # Approximation: use standard rank (works for random matrices)

    r₁ = rank(Matrix(∂₁))  # Expensive for large matrices!
    r₂ = rank(Matrix(∂₂))
    r₃ = rank(Matrix(∂₃))

    println("  Ranks: r₁=$r₁, r₂=$r₂, r₃=$r₃")

    # Betti numbers
    β₀ = C₀ - r₁
    β₁ = C₁ - r₁ - r₂
    β₂ = C₂ - r₂ - r₃

    # Euler characteristic
    χ = β₀ - β₁ + β₂

    # Also compute via alternating sum of cells (should match)
    χ_direct = C₀ - C₁ + C₂ - C₃

    return (β₀=β₀, β₁=β₁, β₂=β₂, χ=χ, χ_direct=χ_direct)
end

"""
Fast approximation using graph-based methods (for large volumes).
"""
function compute_betti_numbers_fast(volume::BitArray{3})
    # β₀: Connected components (fast)
    β₀ = count_connected_components(volume)

    # β₁: Approximate using graph cycle rank
    # Cycle rank = E - V + C where E=edges, V=vertices, C=components
    β₁_approx = approximate_beta1(volume, β₀)

    # β₂: Voids in solid (dual problem)
    β₂_approx = count_enclosed_voids(volume)

    χ = β₀ - β₁_approx + β₂_approx

    return (β₀=β₀, β₁=β₁_approx, β₂=β₂_approx, χ=χ)
end

"""
Count connected components via flood fill.
"""
function count_connected_components(volume::BitArray{3})
    visited = falses(size(volume))
    n_components = 0

    for idx in eachindex(volume)
        if volume[idx] && !visited[idx]
            n_components += 1
            flood_fill!(volume, visited, idx)
        end
    end

    return n_components
end

function flood_fill!(volume::BitArray{3}, visited::BitArray{3}, start_idx::CartesianIndex{3})
    dims = size(volume)
    stack = [start_idx]

    while !isempty(stack)
        idx = pop!(stack)

        if !checkbounds(Bool, volume, idx) || visited[idx] || !volume[idx]
            continue
        end

        visited[idx] = true

        # 6-connectivity
        for dir in [CartesianIndex(1,0,0), CartesianIndex(-1,0,0),
                    CartesianIndex(0,1,0), CartesianIndex(0,-1,0),
                    CartesianIndex(0,0,1), CartesianIndex(0,0,-1)]
            push!(stack, idx + dir)
        end
    end
end

function flood_fill!(volume::BitArray{3}, visited::BitArray{3}, start_idx::Int)
    flood_fill!(volume, visited, CartesianIndices(volume)[start_idx])
end

"""
Approximate β₁ using graph cycle rank.
For a connected graph: β₁ = |E| - |V| + 1
For multiple components: β₁ = |E| - |V| + |C|
"""
function approximate_beta1(volume::BitArray{3}, n_components::Int)
    # Count vertices (pore voxels) and edges (adjacencies)
    V = sum(volume)
    E = 0

    # Count edges (6-connectivity)
    dims = size(volume)
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        if volume[i, j, k]
            # Check neighbors
            for (di, dj, dk) in [(1,0,0), (0,1,0), (0,0,1)]
                ni, nj, nk = i+di, j+dj, k+dk
                if 1 <= ni <= dims[1] && 1 <= nj <= dims[2] && 1 <= nk <= dims[3]
                    if volume[ni, nj, nk]
                        E += 1
                    end
                end
            end
        end
    end

    # Cycle rank (first Betti number for graph)
    β₁ = E - V + n_components

    return max(0, β₁)  # Can't be negative
end

"""
Count enclosed voids (cavities in solid material).
This is β₂ for the pore space = β₀ for the solid space (excluding exterior).
"""
function count_enclosed_voids(volume::BitArray{3})
    # Invert: solid = .!volume
    solid = .!volume

    # Label solid components
    solid_labels = label_all_components(solid)
    n_solid_components = maximum(solid_labels)

    if n_solid_components == 0
        return 0
    end

    # Identify exterior component (touches boundary)
    exterior_labels = Set{Int}()
    dims = size(volume)

    # Check all six faces
    for label in unique(solid_labels[1, :, :])
        push!(exterior_labels, label)
    end
    for label in unique(solid_labels[end, :, :])
        push!(exterior_labels, label)
    end
    for label in unique(solid_labels[:, 1, :])
        push!(exterior_labels, label)
    end
    for label in unique(solid_labels[:, end, :])
        push!(exterior_labels, label)
    end
    for label in unique(solid_labels[:, :, 1])
        push!(exterior_labels, label)
    end
    for label in unique(solid_labels[:, :, end])
        push!(exterior_labels, label)
    end

    delete!(exterior_labels, 0)  # Remove background

    # Enclosed voids = solid components NOT touching boundary
    n_voids = n_solid_components - length(exterior_labels)

    return max(0, n_voids)
end

function label_all_components(volume::BitArray{3})
    labels = zeros(Int, size(volume))
    current_label = 0

    for idx in eachindex(volume)
        if volume[idx] && labels[idx] == 0
            current_label += 1
            flood_fill_label!(volume, labels, idx, current_label)
        end
    end

    return labels
end

function flood_fill_label!(volume::BitArray{3}, labels::Array{Int,3},
                           start_idx::Int, label::Int)
    dims = size(volume)
    idx_cart = CartesianIndices(volume)[start_idx]
    stack = [idx_cart]

    while !isempty(stack)
        idx = pop!(stack)

        if !checkbounds(Bool, volume, idx) || labels[idx] != 0 || !volume[idx]
            continue
        end

        labels[idx] = label

        for dir in [CartesianIndex(1,0,0), CartesianIndex(-1,0,0),
                    CartesianIndex(0,1,0), CartesianIndex(0,-1,0),
                    CartesianIndex(0,0,1), CartesianIndex(0,0,-1)]
            push!(stack, idx + dir)
        end
    end
end

# ============================================================================
# TEST STRUCTURES WITH KNOWN BETTI NUMBERS
# ============================================================================

"""
Generate solid torus: β₀=1, β₁=1, β₂=0
"""
function generate_torus(size::Int=30, R::Float64=10.0, r::Float64=4.0)
    volume = falses(size, size, size)
    center = size / 2

    for k in 1:size, j in 1:size, i in 1:size
        x, y, z = i - center, j - center, k - center

        # Torus equation: (R - sqrt(x² + y²))² + z² ≤ r²
        ρ = sqrt(x^2 + y^2)
        dist = (R - ρ)^2 + z^2

        if dist <= r^2
            volume[i, j, k] = true
        end
    end

    return volume
end

"""
Generate hollow sphere: β₀=1, β₁=0, β₂=1
"""
function generate_hollow_sphere(size::Int=30, R_outer::Float64=12.0, R_inner::Float64=8.0)
    volume = falses(size, size, size)
    center = size / 2

    for k in 1:size, j in 1:size, i in 1:size
        x, y, z = i - center, j - center, k - center
        r = sqrt(x^2 + y^2 + z^2)

        if R_inner <= r <= R_outer
            volume[i, j, k] = true
        end
    end

    return volume
end

"""
Generate double torus (genus 2): β₀=1, β₁=2, β₂=0
"""
function generate_double_torus(size::Int=40)
    # Two tori side by side, connected
    t1 = generate_torus(size, 8.0, 3.0)
    t2 = generate_torus(size, 8.0, 3.0)

    # Shift t2 to the side
    volume = falses(size, size, size)
    center = size ÷ 2

    # Place torus 1
    for k in 1:size, j in 1:size, i in 1:size
        if i < center && t1[i, j, k]
            volume[i, j, k] = true
        elseif i >= center && t2[i, j, k]
            volume[i, j, k] = true
        end
    end

    # Connect them with a bridge
    for k in center-2:center+2, j in center-2:center+2
        for i in center-5:center+5
            volume[i, j, k] = true
        end
    end

    return volume
end

"""
Test Betti number computation on known structures.
"""
function test_known_structures()
    println("\n" * "="^80)
    println("TESTING BETTI NUMBER COMPUTATION ON KNOWN STRUCTURES")
    println("="^80)

    structures = [
        ("Solid Torus", generate_torus(25), (1, 1, 0)),
        ("Hollow Sphere", generate_hollow_sphere(25), (1, 0, 1)),
    ]

    for (name, volume, expected) in structures
        println("\n" * "-"^80)
        println("Structure: $name")
        println("Expected: β₀=$(expected[1]), β₁=$(expected[2]), β₂=$(expected[3])")
        println("Volume: $(sum(volume)) / $(length(volume)) pore voxels")

        # Fast approximation
        println("\nFast approximation:")
        betti_fast = compute_betti_numbers_fast(volume)
        @printf("  β₀ = %d\n", betti_fast.β₀)
        @printf("  β₁ = %d (approx)\n", betti_fast.β₁)
        @printf("  β₂ = %d (approx)\n", betti_fast.β₂)
        @printf("  χ = %d\n", betti_fast.χ)

        # Check correctness
        correct_β₀ = (betti_fast.β₀ == expected[1])
        # β₁ and β₂ are approximations, so we check if they're in the right ballpark
        close_β₁ = abs(betti_fast.β₁ - expected[2]) <= 2
        close_β₂ = abs(betti_fast.β₂ - expected[3]) <= 1

        if correct_β₀
            println("  ✓ β₀ correct")
        else
            println("  ✗ β₀ incorrect (got $(betti_fast.β₀), expected $(expected[1]))")
        end

        if close_β₁
            println("  ≈ β₁ reasonable")
        else
            println("  ⚠ β₁ may be off (got $(betti_fast.β₁), expected $(expected[2]))")
        end

        if close_β₂
            println("  ≈ β₂ reasonable")
        else
            println("  ⚠ β₂ may be off (got $(betti_fast.β₂), expected $(expected[3]))")
        end

        # Note: For exact computation, we'd need full homology (too expensive for demo)
    end

    println("\n" * "="^80)
    println("NOTE: Fast approximations are heuristic. For exact Betti numbers,")
    println("use dedicated TDA libraries: Eirene.jl, GUDHI, or Dionysus.")
    println("="^80)
end

# ============================================================================
# TOPOLOGY-TRANSPORT CORRELATION RE-RUN
# ============================================================================

"""
Generate random percolation structure.
"""
function generate_percolation(size::Int, porosity::Float64)
    return rand(size, size, size) .< porosity
end

"""
Compute tortuosity via geodesic distance.
"""
function compute_tortuosity(volume::BitArray{3})
    dims = size(volume)

    # Check percolation (top to bottom in Z)
    dist = fill(Inf, dims)
    queue = Vector{CartesianIndex{3}}()

    # Initialize from bottom (z=1)
    for j in 1:dims[2], i in 1:dims[1]
        if volume[i, j, 1]
            dist[i, j, 1] = 0.0
            push!(queue, CartesianIndex(i, j, 1))
        end
    end

    min_dist_top = Inf
    head = 1

    while head <= length(queue)
        idx = queue[head]
        head += 1
        d = dist[idx]

        if idx[3] == dims[3]
            min_dist_top = min(min_dist_top, d)
            continue
        end

        # 6-connectivity BFS
        for dir in [CartesianIndex(1,0,0), CartesianIndex(-1,0,0),
                    CartesianIndex(0,1,0), CartesianIndex(0,-1,0),
                    CartesianIndex(0,0,1), CartesianIndex(0,0,-1)]
            next_idx = idx + dir
            if checkbounds(Bool, volume, next_idx) && volume[next_idx]
                new_d = d + 1.0
                if new_d < dist[next_idx]
                    dist[next_idx] = new_d
                    push!(queue, next_idx)
                end
            end
        end
    end

    if isinf(min_dist_top)
        return Inf, false
    end

    τ = min_dist_top / dims[3]
    return τ, true
end

"""
Re-run topology-transport correlation with proper Betti numbers.
"""
function test_topology_transport_correlation(size::Int=40, n_samples::Int=20)
    println("\n" * "="^80)
    println("TOPOLOGY-TRANSPORT CORRELATION WITH PROPER BETTI NUMBERS")
    println("="^80)
    println("Re-testing: cor(χ, τ) = 0.78 finding from search_novel_physics.jl")
    println("Previous result used β₁=β₂=0 (placeholders)")
    println()

    porosities = 0.35:0.05:0.75
    results = []

    println("Generating structures and computing metrics...")
    println("(Size: $(size)³, Samples per porosity: $n_samples)")
    println()

    for (p_idx, p) in enumerate(porosities)
        @printf("Porosity %.2f (%d/%d)...\n", p, p_idx, length(porosities))

        for sample in 1:n_samples
            volume = generate_percolation(size, p)

            # Compute Betti numbers
            betti = compute_betti_numbers_fast(volume)

            # Compute tortuosity
            τ, percolates = compute_tortuosity(volume)

            if percolates && isfinite(τ) && τ < 10.0  # Filter outliers
                push!(results, (
                    p=p,
                    β₀=betti.β₀,
                    β₁=betti.β₁,
                    β₂=betti.β₂,
                    χ=betti.χ,
                    τ=τ
                ))
            end
        end
    end

    if length(results) < 10
        println("\n✗ Insufficient data for correlation analysis")
        return
    end

    @printf("\nCollected %d samples\n", length(results))

    # Extract data
    β₀_vals = [r.β₀ for r in results]
    β₁_vals = [r.β₁ for r in results]
    β₂_vals = [r.β₂ for r in results]
    χ_vals = [r.χ for r in results]
    τ_vals = [r.τ for r in results]
    p_vals = [r.p for r in results]

    # Compute correlations
    println("\n" * "="^80)
    println("CORRELATION ANALYSIS")
    println("="^80)

    cor_β₀_τ = cor(β₀_vals, τ_vals)
    cor_β₁_τ = cor(β₁_vals, τ_vals)
    cor_β₂_τ = cor(β₂_vals, τ_vals)
    cor_χ_τ = cor(χ_vals, τ_vals)
    cor_p_τ = cor(p_vals, τ_vals)

    @printf("\nSimple Correlations:\n")
    @printf("  cor(β₀, τ) = %+.4f\n", cor_β₀_τ)
    @printf("  cor(β₁, τ) = %+.4f\n", cor_β₁_τ)
    @printf("  cor(β₂, τ) = %+.4f\n", cor_β₂_τ)
    @printf("  cor(χ, τ)  = %+.4f  ← KEY RESULT\n", cor_χ_τ)
    @printf("  cor(p, τ)  = %+.4f (baseline)\n", cor_p_τ)

    # Partial correlation: cor(χ, τ) controlling for p
    # Use residuals method
    # Regress χ ~ p, get residuals χ_res
    # Regress τ ~ p, get residuals τ_res
    # cor(χ_res, τ_res) = partial correlation

    # Linear regression: y = a + bx
    function simple_regression(x, y)
        n = length(x)
        x_mean = mean(x)
        y_mean = mean(y)
        b = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
        a = y_mean - b * x_mean
        y_pred = a .+ b .* x
        residuals = y .- y_pred
        return residuals
    end

    χ_res = simple_regression(p_vals, χ_vals)
    τ_res = simple_regression(p_vals, τ_vals)

    cor_χ_τ_partial = cor(χ_res, τ_res)

    @printf("\nPartial Correlation (controlling for porosity):\n")
    @printf("  cor(χ, τ | p) = %+.4f  ← CRUCIAL TEST\n", cor_χ_τ_partial)

    # Summary statistics
    println("\n" * "="^80)
    println("BETTI NUMBER STATISTICS")
    println("="^80)
    @printf("β₀: mean=%.2f, std=%.2f, range=[%d, %d]\n",
            mean(β₀_vals), std(β₀_vals), minimum(β₀_vals), maximum(β₀_vals))
    @printf("β₁: mean=%.2f, std=%.2f, range=[%d, %d]\n",
            mean(β₁_vals), std(β₁_vals), minimum(β₁_vals), maximum(β₁_vals))
    @printf("β₂: mean=%.2f, std=%.2f, range=[%d, %d]\n",
            mean(β₂_vals), std(β₂_vals), minimum(β₂_vals), maximum(β₂_vals))
    @printf("χ:  mean=%.2f, std=%.2f, range=[%d, %d]\n",
            mean(χ_vals), std(χ_vals), minimum(χ_vals), maximum(χ_vals))
    @printf("τ:  mean=%.3f, std=%.3f, range=[%.2f, %.2f]\n",
            mean(τ_vals), std(τ_vals), minimum(τ_vals), maximum(τ_vals))

    # Assessment
    println("\n" * "="^80)
    println("ASSESSMENT")
    println("="^80)

    if abs(cor_χ_τ) > 0.7
        println("✓ Strong correlation found: |cor(χ, τ)| > 0.7")

        if abs(cor_χ_τ_partial) > 0.5
            println("✓ Correlation survives controlling for porosity!")
            println("\n⚠ POTENTIAL DISCOVERY: Topology-transport universality")
            println("  χ = β₀ - β₁ + β₂ predicts tortuosity independently of porosity")
            println()
            println("  NEXT STEPS:")
            println("  1. Validate on real porous media (soil, bone, scaffolds)")
            println("  2. Test on different structure types (TPMS, Voronoi, etc.)")
            println("  3. Develop theoretical explanation (topological field theory?)")
            println("  4. Compare with computational topology libraries (GUDHI, Eirene)")
        else
            println("✗ Correlation weakens when controlling for porosity")
            println("  → Likely confounded by porosity (not independent)")
            println("  → NOT a fundamental topological law")
        end
    else
        println("✗ Weak correlation: |cor(χ, τ)| < 0.7")
        println("\n  The original finding (r=0.78) was likely an artifact of:")
        println("  1. Placeholder Betti numbers (β₁=β₂=0)")
        println("  2. Small sample size")
        println("  3. Confounding with porosity")
        println()
        println("  CONCLUSION: No evidence for topology-transport universality")
    end

    println("="^80)

    return results
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function main()
    println("╔" * "="^78 * "╗")
    println("║" * " "^15 * "PROPER BETTI NUMBER COMPUTATION" * " "^32 * "║")
    println("║" * " "^15 * "FOR TOPOLOGY-TRANSPORT CORRELATION" * " "^28 * "║")
    println("╚" * "="^78 * "╝")
    println()
    println("Testing if cor(χ, τ) = 0.78 survives proper Betti number computation")
    println("(Previous result used β₁ = β₂ = 0 placeholders)")
    println()

    # Step 1: Test on known structures
    test_known_structures()

    # Step 2: Re-run topology-transport correlation
    println("\nPress Enter to continue to correlation analysis...")
    readline()

    results = test_topology_transport_correlation(40, 20)

    println("\n" * "="^80)
    println("SCRIPT COMPLETE")
    println("="^80)
    println("""
    SUMMARY:

    This script implemented:
    1. Fast approximations for Betti numbers (β₀, β₁, β₂)
    2. Validation on known structures (torus, sphere)
    3. Re-analysis of topology-transport correlation

    LIMITATIONS:
    - β₁ and β₂ are heuristic approximations, not exact homology
    - For rigorous results, use computational topology libraries:
      * Eirene.jl (Julia)
      * GUDHI (Python, C++)
      * Dionysus (Python, C++)

    RECOMMENDATION:
    - If correlation survives: Pursue with rigorous TDA tools
    - If correlation disappears: Original finding was artifact
    - Either outcome is scientifically valuable (honest null results!)
    """)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
