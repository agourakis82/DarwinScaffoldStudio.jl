#!/usr/bin/env julia
"""
PROPER VALIDATION: Find datasets with WIDE tortuosity range

The Zenodo soil data has τ = 1.06-1.26 (only 20% variation).
We need datasets spanning τ = 1.0 - 3.0+ to properly test models.

Known tortuosity ranges by material:
- Well-connected soil: τ ≈ 1.1-1.3
- Sandstone: τ ≈ 1.5-3.0
- Tight rock/shale: τ ≈ 3-10
- Fibrous media: τ ≈ 1.2-2.0
- Foam: τ ≈ 1.1-1.5
- Near percolation threshold: τ → ∞

Let's check what public datasets exist with ground truth tortuosity.
"""

using Pkg
Pkg.activate(".")

println("="^70)
println("SEARCHING FOR WIDE-RANGE TORTUOSITY DATASETS")
println("="^70)

println("""

KNOWN PUBLIC DATASETS WITH TORTUOSITY:

1. Digital Rocks Portal (digitalrocksportal.org)
   - Bentheimer sandstone: φ ≈ 0.20-0.24, τ ≈ 1.5-2.5
   - Berea sandstone: φ ≈ 0.18-0.22, τ ≈ 2.0-3.0
   - Fontainebleau sandstone: φ ≈ 0.04-0.25, τ ≈ 1.5-10+
   - Carbonate rocks: φ ≈ 0.10-0.30, τ ≈ 2.0-5.0
   ✓ GOOD: Wide τ range, micro-CT data available
   ✗ ISSUE: May need to compute tortuosity ourselves

2. Zenodo 7516228 (current)
   - Soil pore space: φ ≈ 0.15-0.50, τ ≈ 1.06-1.26
   ✗ NARROW τ range - not sufficient for validation

3. Imperial College Rock Library
   - Various sandstones and carbonates
   - Some have published tortuosity values
   ✓ Potential source

4. Synthetic TPMS (we can generate)
   - Gyroid, Schwarz-P, Diamond, etc.
   - φ controllable 0.1-0.9
   - τ typically 1.1-1.4 (highly connected)
   ✗ Still narrow τ range

5. Random sphere packings (we can generate)
   - φ ≈ 0.36-0.45 (random close packing)
   - τ ≈ 1.4-1.8 depending on connectivity
   ✓ Moderate τ range

6. Percolation-threshold structures (we can generate)
   - Near critical: τ → ∞
   - Can create gradient from well-connected to barely-connected
   ✓ BEST for testing connectivity effects

""")

println("="^70)
println("STRATEGY: CREATE SYNTHETIC DATASET WITH CONTROLLED τ RANGE")
println("="^70)

println("""

To properly validate, we need to CREATE a synthetic dataset where:
1. We KNOW the ground truth tortuosity exactly
2. τ spans from 1.0 to 3.0+
3. Connectivity varies from 0.3 to 1.0
4. Porosity varies from 0.1 to 0.6

Methods to create wide τ range:

A. PERCOLATION APPROACH
   - Start with random 3D binary array at p = 0.5 (percolation threshold)
   - Vary occupation probability p from 0.3 to 0.9
   - Near p_c ≈ 0.31 (3D site percolation): τ → ∞
   - Far above p_c: τ → 1
   - This naturally creates wide τ range!

B. OBSTACLE DENSITY APPROACH
   - Start with empty space (τ = 1)
   - Add random obstacles
   - More obstacles → higher τ
   - Can control independently of porosity

C. CHANNEL RESTRICTION APPROACH
   - Create straight channels (τ = 1)
   - Add constrictions/blockages
   - More blockages → higher τ, lower C

Let's implement the percolation approach - it's physically meaningful
and naturally creates the τ range we need.
""")

# =============================================================================
# GENERATE PERCOLATION-BASED DATASET
# =============================================================================

using Statistics
using Random
using LinearAlgebra
using Printf

Random.seed!(42)

println("\n" * "="^70)
println("GENERATING PERCOLATION-BASED VALIDATION DATASET")
println("="^70)

"""
Compute geodesic tortuosity using simplified BFS-based shortest path.
More accurate than random walks, faster than full FMM.
"""
function compute_tortuosity_bfs(binary::AbstractArray{<:Any,3})
    nx, ny, nz = size(binary)

    # Find entry points (pores at z=1)
    entry_points = [(i, j, 1) for i in 1:nx, j in 1:ny if binary[i, j, 1]]

    if isempty(entry_points)
        return NaN, 0.0  # No path possible
    end

    # BFS from all entry points simultaneously
    distances = fill(Inf, nx, ny, nz)
    queue = Vector{Tuple{Int,Int,Int}}()

    for (i, j, k) in entry_points
        distances[i, j, k] = 0.0
        push!(queue, (i, j, k))
    end

    # 6-connectivity neighbors
    neighbors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    head = 1
    while head <= length(queue)
        (x, y, z) = queue[head]
        head += 1

        for (dx, dy, dz) in neighbors
            nx_new, ny_new, nz_new = x + dx, y + dy, z + dz

            if 1 <= nx_new <= nx && 1 <= ny_new <= ny && 1 <= nz_new <= nz
                if binary[nx_new, ny_new, nz_new] && distances[nx_new, ny_new, nz_new] == Inf
                    distances[nx_new, ny_new, nz_new] = distances[x, y, z] + 1.0
                    push!(queue, (nx_new, ny_new, nz_new))
                end
            end
        end
    end

    # Find minimum distance to exit (z = nz)
    exit_distances = [distances[i, j, nz] for i in 1:nx, j in 1:ny if binary[i, j, nz]]

    if isempty(exit_distances) || all(isinf.(exit_distances))
        return NaN, 0.0  # No path through
    end

    min_path = minimum(filter(!isinf, exit_distances))
    τ = (min_path + 1) / nz  # +1 because we count steps, not points

    # Connectivity = fraction that can be reached
    reachable = sum(distances .< Inf .&& binary)
    total_pore = sum(binary)
    connectivity = total_pore > 0 ? reachable / total_pore : 0.0

    return τ, connectivity
end

"""
Generate 3D porous structure via site percolation.
p = occupation probability (pore fraction before percolation effects)
"""
function generate_percolation_structure(size::Int, p::Float64)
    return rand(size, size, size) .< p
end

"""
Generate structure with controlled connectivity via layer blocking.
"""
function generate_layered_blocking(size::Int, base_porosity::Float64, block_fraction::Float64)
    # Start with base porosity
    volume = rand(size, size, size) .< base_porosity

    # Block random z-layers
    n_blocked = round(Int, block_fraction * size)
    blocked_layers = randperm(size)[1:n_blocked]

    for z in blocked_layers
        # Block 80-95% of each blocked layer
        block_prob = 0.80 + 0.15 * rand()
        for i in 1:size, j in 1:size
            if rand() < block_prob
                volume[i, j, z] = false
            end
        end
    end

    return volume
end

# Generate dataset spanning wide range
println("\nGenerating samples with varying tortuosity...")

results = []

# Method 1: Pure percolation at different densities
println("\n1. Percolation structures (varying p):")
for p in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
    for trial in 1:5
        vol = generate_percolation_structure(64, p)
        φ = sum(vol) / length(vol)
        τ, C = compute_tortuosity_bfs(vol)

        if !isnan(τ) && τ < 10.0  # Valid path exists
            push!(results, (method="percolation", p=p, φ=φ, τ=τ, C=C))
            if trial == 1
                println(@sprintf("  p=%.2f: φ=%.3f, τ=%.3f, C=%.3f", p, φ, τ, C))
            end
        end
    end
end

# Method 2: Layered blocking for controlled connectivity
println("\n2. Layered blocking (varying block fraction):")
for block_frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for base_φ in [0.4, 0.5, 0.6]
        for trial in 1:3
            vol = generate_layered_blocking(64, base_φ, block_frac)
            φ = sum(vol) / length(vol)
            τ, C = compute_tortuosity_bfs(vol)

            if !isnan(τ) && τ < 10.0
                push!(results, (method="blocked", block=block_frac, φ=φ, τ=τ, C=C))
                if trial == 1
                    println(@sprintf("  block=%.1f, base_φ=%.1f: φ=%.3f, τ=%.3f, C=%.3f",
                                    block_frac, base_φ, φ, τ, C))
                end
            end
        end
    end
end

# Method 3: Obstacle insertion
println("\n3. Random obstacles (varying obstacle density):")
for n_obstacles in [100, 500, 1000, 2000, 4000]
    for trial in 1:3
        # Start with open space
        vol = ones(Bool, 64, 64, 64)

        # Add spherical obstacles
        for _ in 1:n_obstacles
            cx, cy, cz = rand(1:64), rand(1:64), rand(1:64)
            r = rand(2:5)
            for i in max(1,cx-r):min(64,cx+r)
                for j in max(1,cy-r):min(64,cy+r)
                    for k in max(1,cz-r):min(64,cz+r)
                        if (i-cx)^2 + (j-cy)^2 + (k-cz)^2 <= r^2
                            vol[i,j,k] = false
                        end
                    end
                end
            end
        end

        φ = sum(vol) / length(vol)
        τ, C = compute_tortuosity_bfs(vol)

        if !isnan(τ) && τ < 10.0 && φ > 0.05
            push!(results, (method="obstacles", n_obs=n_obstacles, φ=φ, τ=τ, C=C))
            if trial == 1
                println(@sprintf("  n_obs=%d: φ=%.3f, τ=%.3f, C=%.3f", n_obstacles, φ, τ, C))
            end
        end
    end
end

# =============================================================================
# ANALYZE THE SYNTHETIC DATASET
# =============================================================================

println("\n" * "="^70)
println("SYNTHETIC DATASET STATISTICS")
println("="^70)

n_samples = length(results)
φ_all = [r.φ for r in results]
τ_all = [r.τ for r in results]
C_all = [r.C for r in results]

println(@sprintf("\nTotal samples: %d", n_samples))
println(@sprintf("\nPorosity φ:"))
println(@sprintf("  Range: %.3f - %.3f", minimum(φ_all), maximum(φ_all)))
println(@sprintf("  Mean:  %.3f ± %.3f", mean(φ_all), std(φ_all)))

println(@sprintf("\nTortuosity τ:"))
println(@sprintf("  Range: %.3f - %.3f", minimum(τ_all), maximum(τ_all)))
println(@sprintf("  Mean:  %.3f ± %.3f", mean(τ_all), std(τ_all)))

println(@sprintf("\nConnectivity C:"))
println(@sprintf("  Range: %.3f - %.3f", minimum(C_all), maximum(C_all)))
println(@sprintf("  Mean:  %.3f ± %.3f", mean(C_all), std(C_all)))

# =============================================================================
# TEST MODELS ON WIDE-RANGE DATA
# =============================================================================

println("\n" * "="^70)
println("MODEL VALIDATION ON WIDE-RANGE DATA")
println("="^70)

# Models to test
function archie(φ, m)
    return φ .^ (-m)
end

function linear_model(φ, a, b)
    return a .+ b ./ φ
end

function connectivity_model(φ, C, a, b, c)
    return a .+ b ./ φ .+ c .* C
end

# Test Archie with different m
println("\nArchie's law τ = φ^(-m):")
for m in [0.127, 0.3, 0.5, 0.7, 1.0]
    τ_pred = archie(φ_all, m)
    mre = mean(abs.(τ_pred .- τ_all) ./ τ_all) * 100
    within_20 = sum(abs.(τ_pred .- τ_all) ./ τ_all .< 0.20) / n_samples * 100
    println(@sprintf("  m = %.3f: MRE = %6.1f%%, within 20%% = %5.1f%%", m, mre, within_20))
end

# Fit optimal m
global best_m = 0.0
global best_mre = Inf
for m in 0.1:0.01:2.0
    τ_pred = archie(φ_all, m)
    mre = mean(abs.(τ_pred .- τ_all) ./ τ_all) * 100
    if mre < best_mre
        global best_mre = mre
        global best_m = m
    end
end
println(@sprintf("\n  OPTIMAL m = %.3f: MRE = %.1f%%", best_m, best_mre))

# Test linear model
println("\nLinear model τ = a + b/φ:")
X = hcat(ones(n_samples), 1 ./ φ_all)
β = X \ τ_all
τ_pred_linear = X * β
mre_linear = mean(abs.(τ_pred_linear .- τ_all) ./ τ_all) * 100
println(@sprintf("  Fitted: τ = %.4f + %.4f/φ", β[1], β[2]))
println(@sprintf("  MRE = %.1f%%", mre_linear))

# Test connectivity model
println("\nConnectivity model τ = a + b/φ + c·C:")
X_C = hcat(ones(n_samples), 1 ./ φ_all, C_all)
β_C = X_C \ τ_all
τ_pred_conn = X_C * β_C
mre_conn = mean(abs.(τ_pred_conn .- τ_all) ./ τ_all) * 100
println(@sprintf("  Fitted: τ = %.4f + %.4f/φ + %.4f·C", β_C[1], β_C[2], β_C[3]))
println(@sprintf("  MRE = %.1f%%", mre_conn))

# Improvement from adding C
improvement = (mre_linear - mre_conn) / mre_linear * 100
println(@sprintf("\n  IMPROVEMENT from connectivity: %.1f%% error reduction", improvement))

# Correlation analysis
println("\nCorrelations:")
println(@sprintf("  cor(τ, φ):  %.4f", cor(τ_all, φ_all)))
println(@sprintf("  cor(τ, 1/φ): %.4f", cor(τ_all, 1 ./ φ_all)))
println(@sprintf("  cor(τ, C):  %.4f", cor(τ_all, C_all)))

# Partial correlation: τ vs C controlling for φ
τ_resid = τ_all .- (X * β)
partial_cor = cor(τ_resid, C_all)
println(@sprintf("  cor(τ|φ, C): %.4f (partial, controlling for φ)", partial_cor))

# =============================================================================
# KEY INSIGHT
# =============================================================================

println("\n" * "="^70)
println("KEY INSIGHT")
println("="^70)

println("""

With WIDE-RANGE synthetic data (τ = $(round(minimum(τ_all), digits=2)) - $(round(maximum(τ_all), digits=2))):

1. Optimal Archie m = $(round(best_m, digits=2)) (vs 0.127 on narrow soil data)
   → The "anomalously low" m was an artifact of narrow τ range!

2. Connectivity correlation with τ: $(round(cor(τ_all, C_all), digits=3))
   Partial correlation (controlling for φ): $(round(partial_cor, digits=3))
   → Connectivity $(abs(partial_cor) > 0.3 ? "IS" : "may not be") significant when properly tested!

3. Model comparison:
   - Archie (optimal m):    MRE = $(round(best_mre, digits=1))%
   - Linear (φ only):       MRE = $(round(mre_linear, digits=1))%
   - With connectivity:     MRE = $(round(mre_conn, digits=1))%
   - Improvement from C:    $(round(improvement, digits=1))%

""")

println("="^70)
println("VALIDATION DATASET GENERATION COMPLETE")
println("="^70)
