#!/usr/bin/env julia
"""
CRITICAL VALIDATION: Does the connectivity term actually matter?

Test the formula on structures with VARYING connectivity.
This is the key test - if β·C doesn't improve predictions,
the contribution is not novel.
"""

using Pkg
Pkg.activate(".")

using Statistics
using Printf
using Random

Random.seed!(42)

println("="^70)
println("CRITICAL TEST: VARYING CONNECTIVITY STRUCTURES")
println("="^70)

# The discovered formula
const τ₀ = 1.04
const α = 0.045
const β = 0.07

predict_with_C(φ, C) = τ₀ + α/φ - β*C
predict_without_C(φ) = τ₀ + α/φ  # Baseline without connectivity

function calculate_tortuosity_montecarlo(binary::Array{Bool,3}; n_walks=500)
    """
    Monte Carlo tortuosity estimation via random walks.
    More robust than simple geometric estimation.
    """
    nx, ny, nz = size(binary)
    φ = sum(binary) / length(binary)

    if φ < 0.05 || φ > 0.99
        return NaN
    end

    successful_walks = Float64[]

    # Find starting points (pores in z=1)
    start_points = [(i,j) for i in 1:nx, j in 1:ny if binary[i,j,1]]

    if isempty(start_points)
        return NaN
    end

    for _ in 1:n_walks
        # Random start
        x, y = start_points[rand(1:length(start_points))]
        z = 1
        path_length = 0.0
        steps = 0
        max_steps = 10 * nz

        while z < nz && steps < max_steps
            steps += 1

            # Collect valid moves (6-connectivity)
            moves = Tuple{Int,Int,Int,Float64}[]

            # Prefer upward (z+1)
            if z + 1 <= nz && binary[x, y, z+1]
                push!(moves, (0, 0, 1, 1.0))
            end

            # Lateral moves
            for (dx, dy) in [(1,0), (-1,0), (0,1), (0,-1)]
                nx_new, ny_new = x + dx, y + dy
                if 1 <= nx_new <= nx && 1 <= ny_new <= ny && binary[nx_new, ny_new, z]
                    push!(moves, (dx, dy, 0, 1.0))
                end
            end

            # Down (penalty)
            if z > 1 && binary[x, y, z-1]
                push!(moves, (0, 0, -1, 1.0))
            end

            if isempty(moves)
                break
            end

            # Weighted selection (prefer up)
            weights = [m[3] > 0 ? 3.0 : (m[3] < 0 ? 0.5 : 1.0) for m in moves]
            total = sum(weights)
            r = rand() * total
            cumsum = 0.0
            selected = moves[1]
            for (m, w) in zip(moves, weights)
                cumsum += w
                if r <= cumsum
                    selected = m
                    break
                end
            end

            x += selected[1]
            y += selected[2]
            z += selected[3]
            path_length += selected[4]
        end

        if z >= nz
            push!(successful_walks, path_length / (nz - 1))
        end
    end

    if length(successful_walks) < n_walks / 10
        # Fallback
        return 1 / sqrt(φ)
    end

    return mean(successful_walks)
end

function generate_layered_structure(size::Int, layer_connectivity::Float64, porosity_target::Float64)
    """
    Generate structure with controlled z-connectivity.

    layer_connectivity: fraction of layers that are fully connected
    """
    volume = zeros(Bool, size, size, size)

    # Fill with random porosity
    for i in 1:size, j in 1:size, k in 1:size
        volume[i,j,k] = rand() < porosity_target
    end

    # Add blocking layers at random z positions
    n_blocked = round(Int, (1 - layer_connectivity) * size)
    blocked_z = randperm(size)[1:n_blocked]

    for z in blocked_z
        # Block most of the layer (leave small holes)
        for i in 1:size, j in 1:size
            if rand() > 0.1  # Block 90% of each blocked layer
                volume[i,j,z] = false
            end
        end
    end

    return volume
end

function generate_channel_structure(size::Int, n_channels::Int, channel_radius::Int)
    """
    Generate structure with vertical channels (high connectivity).
    """
    volume = zeros(Bool, size, size, size)

    # Create random vertical channels
    for _ in 1:n_channels
        cx = rand(1:size)
        cy = rand(1:size)

        for z in 1:size
            for dx in -channel_radius:channel_radius
                for dy in -channel_radius:channel_radius
                    if dx^2 + dy^2 <= channel_radius^2
                        nx, ny = cx + dx, cy + dy
                        if 1 <= nx <= size && 1 <= ny <= size
                            volume[nx, ny, z] = true
                        end
                    end
                end
            end

            # Add some wandering
            cx = clamp(cx + rand(-1:1), 1, size)
            cy = clamp(cy + rand(-1:1), 1, size)
        end
    end

    return volume
end

function calculate_connectivity_z(binary::Array{Bool,3})
    pore_z = vec(sum(binary, dims=(1,2)) .> 0)
    return sum(pore_z) / length(pore_z)
end

# =============================================================================
# TEST 1: LAYERED STRUCTURES WITH VARYING CONNECTIVITY
# =============================================================================

println("\n" * "-"^70)
println("TEST 1: LAYERED STRUCTURES (Varying Z-Connectivity)")
println("-"^70)

println("\nGenerating structures with C ∈ [0.5, 1.0]...")

layered_results = []

for target_C in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for target_φ in [0.3, 0.4, 0.5]
        for _ in 1:3  # Multiple instances
            vol = generate_layered_structure(48, target_C, target_φ)

            φ = sum(vol) / length(vol)
            C = calculate_connectivity_z(vol)

            if 0.2 < φ < 0.7 && C > 0.3
                τ_mc = calculate_tortuosity_montecarlo(vol, n_walks=300)

                if !isnan(τ_mc) && τ_mc > 1.0 && τ_mc < 3.0
                    τ_with_C = predict_with_C(φ, C)
                    τ_without_C = predict_without_C(φ)

                    push!(layered_results, (
                        target_C=target_C,
                        φ=φ,
                        C=C,
                        τ_mc=τ_mc,
                        τ_with_C=τ_with_C,
                        τ_without_C=τ_without_C
                    ))
                end
            end
        end
    end
end

println(@sprintf("  Generated %d valid samples", length(layered_results)))

if length(layered_results) > 5
    println("\n  Results sorted by connectivity:")
    println(@sprintf("  %-6s %-6s %-8s %-10s %-10s %-10s %-8s %-8s",
                    "C_tgt", "C", "φ", "τ_MC", "τ_with_C", "τ_no_C", "err_w_C", "err_no_C"))

    sorted_results = sort(layered_results, by=x->x.C)

    err_with_C = Float64[]
    err_without_C = Float64[]

    for r in sorted_results
        ew = abs(r.τ_with_C - r.τ_mc) / r.τ_mc * 100
        eno = abs(r.τ_without_C - r.τ_mc) / r.τ_mc * 100
        push!(err_with_C, ew)
        push!(err_without_C, eno)

        println(@sprintf("  %-6.1f %-6.3f %-8.3f %-10.4f %-10.4f %-10.4f %-8.2f%% %-8.2f%%",
                        r.target_C, r.C, r.φ, r.τ_mc, r.τ_with_C, r.τ_without_C, ew, eno))
    end

    println("\n  SUMMARY:")
    println(@sprintf("    MRE with connectivity:    %.2f%%", mean(err_with_C)))
    println(@sprintf("    MRE without connectivity: %.2f%%", mean(err_without_C)))
    println(@sprintf("    IMPROVEMENT from C term:  %.1f%%", (mean(err_without_C) - mean(err_with_C)) / mean(err_without_C) * 100))
end

# =============================================================================
# TEST 2: CHANNEL VS RANDOM STRUCTURES
# =============================================================================

println("\n" * "-"^70)
println("TEST 2: HIGH-CONNECTIVITY CHANNELS vs LOW-CONNECTIVITY RANDOM")
println("-"^70)

channel_results = []
random_results = []

# High-connectivity channels
println("\nGenerating channel structures (high C)...")
for n_ch in [5, 10, 15, 20]
    for r in [2, 3, 4]
        vol = generate_channel_structure(48, n_ch, r)
        φ = sum(vol) / length(vol)
        C = calculate_connectivity_z(vol)

        if 0.1 < φ < 0.6
            τ_mc = calculate_tortuosity_montecarlo(vol, n_walks=300)
            if !isnan(τ_mc) && τ_mc > 1.0 && τ_mc < 2.5
                push!(channel_results, (φ=φ, C=C, τ_mc=τ_mc))
            end
        end
    end
end

# Low-connectivity random
println("Generating random structures (varying C)...")
for _ in 1:20
    vol = generate_layered_structure(48, rand(0.4:0.1:0.8), rand(0.25:0.05:0.5))
    φ = sum(vol) / length(vol)
    C = calculate_connectivity_z(vol)

    if 0.1 < φ < 0.6 && C < 0.95
        τ_mc = calculate_tortuosity_montecarlo(vol, n_walks=300)
        if !isnan(τ_mc) && τ_mc > 1.0 && τ_mc < 2.5
            push!(random_results, (φ=φ, C=C, τ_mc=τ_mc))
        end
    end
end

# Combined analysis
all_test2 = vcat(channel_results, random_results)

if length(all_test2) > 5
    println(@sprintf("\n  Total samples: %d (channels: %d, random: %d)",
                    length(all_test2), length(channel_results), length(random_results)))

    # Calculate errors for both models
    err_with_C = Float64[]
    err_without_C = Float64[]

    for r in all_test2
        τ_w = predict_with_C(r.φ, r.C)
        τ_wo = predict_without_C(r.φ)

        push!(err_with_C, abs(τ_w - r.τ_mc) / r.τ_mc * 100)
        push!(err_without_C, abs(τ_wo - r.τ_mc) / r.τ_mc * 100)
    end

    # Correlation analysis
    C_values = [r.C for r in all_test2]
    τ_values = [r.τ_mc for r in all_test2]
    φ_values = [r.φ for r in all_test2]

    # Partial correlation: τ vs C, controlling for φ
    # Residuals of τ after linear fit on φ
    X_φ = hcat(ones(length(φ_values)), 1 ./ φ_values)
    coef = X_φ \ τ_values
    τ_resid = τ_values .- X_φ * coef

    cor_τ_C = cor(τ_values, C_values)
    cor_resid_C = cor(τ_resid, C_values)

    println("\n  CORRELATION ANALYSIS:")
    println(@sprintf("    cor(τ, C):        %.4f", cor_τ_C))
    println(@sprintf("    cor(τ-τ_φ, C):    %.4f  (partial, controlling for φ)", cor_resid_C))

    println("\n  MODEL COMPARISON:")
    println(@sprintf("    MRE with C:       %.2f%%", mean(err_with_C)))
    println(@sprintf("    MRE without C:    %.2f%%", mean(err_without_C)))

    improvement = (mean(err_without_C) - mean(err_with_C)) / mean(err_without_C) * 100
    println(@sprintf("    IMPROVEMENT:      %.1f%%", improvement))
end

# =============================================================================
# FINAL VERDICT
# =============================================================================

println("\n" * "="^70)
println("FINAL VERDICT: IS THE CONNECTIVITY TERM SIGNIFICANT?")
println("="^70)

# Aggregate all results
all_results = []
if !isempty(layered_results)
    for r in layered_results
        push!(all_results, (φ=r.φ, C=r.C, τ_gt=r.τ_mc))
    end
end
for r in all_test2
    push!(all_results, (φ=r.φ, C=r.C, τ_gt=r.τ_mc))
end

if length(all_results) > 10
    # Final comparison
    err_with = [abs(predict_with_C(r.φ, r.C) - r.τ_gt) / r.τ_gt * 100 for r in all_results]
    err_without = [abs(predict_without_C(r.φ) - r.τ_gt) / r.τ_gt * 100 for r in all_results]

    # Statistical test (paired t-test equivalent)
    diff = err_without .- err_with
    mean_diff = mean(diff)
    std_diff = std(diff)
    t_stat = mean_diff / (std_diff / sqrt(length(diff)))

    println("\n┌────────────────────────────────────────────────────────────────────┐")
    println("│                       STATISTICAL VERDICT                           │")
    println("├────────────────────────────────────────────────────────────────────┤")
    println(@sprintf("│                                                                     │"))
    println(@sprintf("│   Total test samples: %d                                            │", length(all_results)))
    println(@sprintf("│   Connectivity range: %.2f - %.2f                                   │",
                    minimum(r.C for r in all_results), maximum(r.C for r in all_results)))
    println(@sprintf("│                                                                     │"))
    println(@sprintf("│   MODEL: τ = τ₀ + α/φ - β·C                                        │"))
    println(@sprintf("│     MRE = %.2f%%                                                    │", mean(err_with)))
    println(@sprintf("│                                                                     │"))
    println(@sprintf("│   BASELINE: τ = τ₀ + α/φ (no connectivity)                         │"))
    println(@sprintf("│     MRE = %.2f%%                                                    │", mean(err_without)))
    println(@sprintf("│                                                                     │"))
    println(@sprintf("│   IMPROVEMENT: %.1f%% reduction in error                            │",
                    (mean(err_without) - mean(err_with)) / mean(err_without) * 100))
    println(@sprintf("│   t-statistic: %.2f (|t| > 2 is significant)                        │", t_stat))
    println(@sprintf("│                                                                     │"))

    if abs(t_stat) > 2
        println("│   ✓ CONNECTIVITY TERM IS STATISTICALLY SIGNIFICANT                │")
    else
        println("│   ✗ Connectivity term may not be significant                      │")
    end

    println("│                                                                     │")
    println("└────────────────────────────────────────────────────────────────────┘")
end

println("\n" * "="^70)
println("VALIDATION COMPLETE")
println("="^70)
