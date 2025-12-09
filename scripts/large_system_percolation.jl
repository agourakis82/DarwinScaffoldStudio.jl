#!/usr/bin/env julia
"""
Large System Percolation Analysis
==================================

Tests whether the percolation exponent μ for tortuosity converges to:
- μ ≈ 0.25 (anomalous/fractal regime)
- μ ≈ 1.3 (standard 3D percolation theory)

Key Physics:
- At percolation threshold p_c: τ ~ |p - p_c|^(-μ)
- Standard 3D percolation: μ ≈ 1.3
- Fractal pore networks: μ ≈ 0.25 (if φ-based scaling)

Strategy:
1. Large system sizes: 64³, 100³, 150³
2. Fine sampling near p_c ∈ [0.32, 0.50]
3. Memory-efficient BFS for geodesic tortuosity
4. Power law fits for each system size
5. Finite-size scaling analysis
"""

using Statistics
using Random
using Printf

#=============================================================================
                            DATA STRUCTURES
=============================================================================#

struct PercolationSample
    L::Int                    # System size
    porosity::Float64         # Target porosity
    realized_porosity::Float64 # Actual porosity
    n_samples::Int            # Number of samples at this (L, p)
    tortuosities::Vector{Float64}  # Valid tortuosity measurements
    percolation_fraction::Float64  # Fraction of samples that percolated
end

struct PowerLawFit
    L::Int                    # System size
    p_c::Float64             # Estimated critical porosity
    mu::Float64              # Fitted exponent
    mu_err::Float64          # Error estimate
    R2::Float64              # Goodness of fit
    n_points::Int            # Number of data points
end

#=============================================================================
                        PERCOLATION GENERATION
=============================================================================#

"""
Generate binary scaffold with target porosity using random voxel removal.
Memory-efficient: uses BitArray for storage.
"""
function generate_scaffold(L::Int, target_porosity::Float64)::BitArray{3}
    n_voxels = L^3
    n_pore = round(Int, target_porosity * n_voxels)

    # Start with all solid (false = solid, true = pore)
    scaffold = falses(L, L, L)

    # Randomly select pore voxels
    pore_indices = randperm(n_voxels)[1:n_pore]
    scaffold[pore_indices] .= true

    return scaffold
end

"""
Compute actual porosity of scaffold.
"""
function compute_porosity(scaffold::BitArray{3})::Float64
    return sum(scaffold) / length(scaffold)
end

#=============================================================================
                        CONNECTED COMPONENTS
=============================================================================#

"""
Find largest connected component using BFS.
Returns mask of largest pore cluster (memory-efficient).
"""
function find_largest_component(scaffold::BitArray{3})::BitArray{3}
    L = size(scaffold, 1)
    visited = falses(L, L, L)
    largest_component = falses(L, L, L)
    largest_size = 0

    # 6-connectivity neighbors
    neighbors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    for i in 1:L, j in 1:L, k in 1:L
        if scaffold[i,j,k] && !visited[i,j,k]
            # BFS from this seed
            queue = [(i,j,k)]
            component = falses(L, L, L)
            component_size = 0
            idx = 1

            while idx <= length(queue)
                ci, cj, ck = queue[idx]
                idx += 1

                if visited[ci,cj,ck]
                    continue
                end

                visited[ci,cj,ck] = true
                component[ci,cj,ck] = true
                component_size += 1

                # Check neighbors
                for (di, dj, dk) in neighbors
                    ni, nj, nk = ci+di, cj+dj, ck+dk
                    if 1 <= ni <= L && 1 <= nj <= L && 1 <= nk <= L
                        if scaffold[ni,nj,nk] && !visited[ni,nj,nk]
                            push!(queue, (ni,nj,nk))
                        end
                    end
                end
            end

            if component_size > largest_size
                largest_size = component_size
                largest_component = component
            end
        end
    end

    return largest_component
end

"""
Check if scaffold percolates in z-direction.
"""
function check_percolation(component::BitArray{3})::Bool
    L = size(component, 1)

    # Check if component spans from z=1 to z=L
    has_bottom = any(component[:, :, 1])
    has_top = any(component[:, :, L])

    return has_bottom && has_top
end

#=============================================================================
                        GEODESIC TORTUOSITY
=============================================================================#

"""
Compute geodesic tortuosity using BFS shortest path.
Memory-efficient: doesn't store full distance matrix.

Tortuosity τ = L_geodesic / L_euclidean

For z-direction percolation:
- Start from random pore voxel at z=1
- Find shortest path to z=L through pore network
- Compare to straight-line distance (L-1)
"""
function compute_geodesic_tortuosity(component::BitArray{3})::Union{Float64, Nothing}
    L = size(component, 1)

    # Find starting point at bottom (z=1)
    start_candidates = findall(component[:, :, 1])
    if isempty(start_candidates)
        return nothing
    end
    start_2d = start_candidates[rand(1:length(start_candidates))]
    start = CartesianIndex(start_2d[1], start_2d[2], 1)

    # Find target points at top (z=L)
    end_candidates = findall(component[:, :, L])
    if isempty(end_candidates)
        return nothing
    end

    # BFS to find shortest path
    L_size = size(component, 1)
    distances = fill(typemax(Int32), L_size, L_size, L_size)
    si, sj, sk = start.I
    distances[si, sj, sk] = 0

    queue = [start]
    neighbors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    idx = 1
    min_distance_to_top = typemax(Int32)

    while idx <= length(queue)
        current = queue[idx]
        idx += 1

        ci, cj, ck = current.I
        current_dist = distances[ci, cj, ck]

        # Check if we reached the top
        if ck == L
            min_distance_to_top = min(min_distance_to_top, current_dist)
            continue  # Keep searching for potentially shorter paths
        end

        # Early termination if we've already found a shorter path to top
        if current_dist >= min_distance_to_top
            continue
        end

        # Explore neighbors
        for (di, dj, dk) in neighbors
            ni, nj, nk = ci+di, cj+dj, ck+dk

            if 1 <= ni <= L_size && 1 <= nj <= L_size && 1 <= nk <= L_size
                if component[ni, nj, nk] && distances[ni, nj, nk] > current_dist + 1
                    distances[ni, nj, nk] = current_dist + 1
                    push!(queue, CartesianIndex(ni, nj, nk))
                end
            end
        end
    end

    if min_distance_to_top == typemax(Int32)
        return nothing
    end

    # Tortuosity = geodesic_length / euclidean_length
    euclidean_length = L - 1
    geodesic_length = min_distance_to_top

    return geodesic_length / euclidean_length
end

#=============================================================================
                        SAMPLING & MEASUREMENT
=============================================================================#

"""
Run percolation simulation for given (L, porosity) combination.
Returns PercolationSample with statistics.
"""
function run_percolation_samples(L::Int, porosity::Float64, n_samples::Int)::PercolationSample
    tortuosities = Float64[]
    realized_porosities = Float64[]
    n_percolated = 0

    print("  L=$L, p=$(round(porosity, digits=3)): ")

    for i in 1:n_samples
        # Generate scaffold
        scaffold = generate_scaffold(L, porosity)
        real_p = compute_porosity(scaffold)
        push!(realized_porosities, real_p)

        # Find largest component
        component = find_largest_component(scaffold)

        # Check percolation
        if check_percolation(component)
            n_percolated += 1

            # Compute tortuosity
            tau = compute_geodesic_tortuosity(component)
            if tau !== nothing
                push!(tortuosities, tau)
            end
        end

        if i % max(1, n_samples ÷ 10) == 0
            print(".")
        end
    end

    perc_fraction = n_percolated / n_samples
    avg_realized_p = mean(realized_porosities)

    println(" $(length(tortuosities))/$(n_samples) valid ($(round(perc_fraction*100, digits=1))% percolated)")

    return PercolationSample(
        L, porosity, avg_realized_p, n_samples,
        tortuosities, perc_fraction
    )
end

#=============================================================================
                        POWER LAW FITTING
=============================================================================#

"""
Fit power law: τ ~ |p - p_c|^(-μ)

Using log-log regression:
log(τ) = -μ * log(|p - p_c|) + const

We'll estimate p_c from the data and fit μ.
"""
function fit_power_law(samples::Vector{PercolationSample})::PowerLawFit
    if isempty(samples)
        return PowerLawFit(0, 0.0, 0.0, Inf, 0.0, 0)
    end

    L = samples[1].L

    # Filter samples with valid measurements
    valid_samples = filter(s -> !isempty(s.tortuosities), samples)

    if length(valid_samples) < 3
        return PowerLawFit(L, 0.0, 0.0, Inf, 0.0, length(valid_samples))
    end

    # Estimate p_c as the porosity where percolation probability ≈ 0.5
    # Sort by porosity
    sorted_samples = sort(valid_samples, by=s->s.realized_porosity)

    # Find crossing point
    p_c_est = 0.31  # Standard 3D site percolation threshold
    for i in 1:length(sorted_samples)-1
        if sorted_samples[i].percolation_fraction < 0.5 &&
           sorted_samples[i+1].percolation_fraction >= 0.5
            p_c_est = (sorted_samples[i].realized_porosity +
                      sorted_samples[i+1].realized_porosity) / 2
            break
        end
    end

    # Prepare data for power law fit
    porosities = Float64[]
    mean_tortuosities = Float64[]

    for sample in valid_samples
        if !isempty(sample.tortuosities)
            p = sample.realized_porosity
            tau_mean = mean(sample.tortuosities)

            # Only use points away from threshold (|p - p_c| > 0.02)
            if abs(p - p_c_est) > 0.02
                push!(porosities, p)
                push!(mean_tortuosities, tau_mean)
            end
        end
    end

    if length(porosities) < 3
        return PowerLawFit(L, p_c_est, 0.0, Inf, 0.0, length(porosities))
    end

    # Log-log regression: log(τ) = -μ * log(|p - p_c|) + C
    x = log.(abs.(porosities .- p_c_est))
    y = log.(mean_tortuosities)

    # Linear regression
    n = length(x)
    x_mean = mean(x)
    y_mean = mean(y)

    numerator = sum((x .- x_mean) .* (y .- y_mean))
    denominator = sum((x .- x_mean).^2)

    mu = -numerator / denominator  # Negative because τ ~ |p-pc|^(-μ)
    intercept = y_mean + mu * x_mean

    # Compute R²
    y_pred = -mu .* x .+ intercept
    ss_res = sum((y .- y_pred).^2)
    ss_tot = sum((y .- y_mean).^2)
    R2 = 1 - ss_res / ss_tot

    # Error estimate (simplified)
    mu_err = abs(mu) * sqrt((1/R2 - 1) / n)

    return PowerLawFit(L, p_c_est, mu, mu_err, R2, n)
end

#=============================================================================
                        FINITE-SIZE SCALING
=============================================================================#

"""
Analyze finite-size scaling: μ(L) → μ(∞)

In finite-size scaling theory:
μ(L) = μ(∞) + A/L^(1/ν)

where ν ≈ 0.88 is the correlation length exponent in 3D percolation.
"""
function finite_size_scaling(fits::Vector{PowerLawFit})
    if length(fits) < 2
        println("\n⚠ Need at least 2 system sizes for finite-size scaling")
        return
    end

    println("\n" * "="^70)
    println("FINITE-SIZE SCALING ANALYSIS")
    println("="^70)

    # Sort by system size
    sorted_fits = sort(fits, by=f->f.L)

    println("\nμ(L) values:")
    for fit in sorted_fits
        println("  L=$(fit.L): μ = $(round(fit.mu, digits=3)) ± $(round(fit.mu_err, digits=3)) (R² = $(round(fit.R2, digits=3)))")
    end

    # Check for trend
    if length(sorted_fits) >= 2
        mu_first = sorted_fits[1].mu
        mu_last = sorted_fits[end].mu

        println("\nTrend analysis:")
        println("  μ(L=$(sorted_fits[1].L)) = $(round(mu_first, digits=3))")
        println("  μ(L=$(sorted_fits[end].L)) = $(round(mu_last, digits=3))")
        println("  Change: $(round(mu_last - mu_first, digits=3))")

        if abs(mu_last - mu_first) < 0.1
            println("  → μ appears to be converging")
        elseif mu_last > mu_first
            println("  → μ is increasing with L")
        else
            println("  → μ is decreasing with L")
        end
    end

    # Extrapolation attempt (if we have 3+ sizes)
    if length(sorted_fits) >= 3
        println("\n⚡ Attempting extrapolation to L→∞:")

        # Simple linear extrapolation in 1/L
        L_inv = [1.0/fit.L for fit in sorted_fits]
        mu_vals = [fit.mu for fit in sorted_fits]

        # Linear fit: μ(L) = μ_inf + A/L
        L_inv_mean = mean(L_inv)
        mu_mean = mean(mu_vals)

        A = sum((L_inv .- L_inv_mean) .* (mu_vals .- mu_mean)) /
            sum((L_inv .- L_inv_mean).^2)
        mu_inf = mu_mean - A * L_inv_mean

        println("  Linear extrapolation: μ(∞) ≈ $(round(mu_inf, digits=3))")
        println("  Slope A ≈ $(round(A, digits=2))")
    end

    # Theoretical comparison
    println("\n📊 Comparison to theory:")
    println("  Standard 3D percolation: μ ≈ 1.3")
    println("  Fractal/anomalous regime: μ ≈ 0.25")

    mu_avg = mean([fit.mu for fit in sorted_fits])
    println("\n  Average μ = $(round(mu_avg, digits=3))")

    if abs(mu_avg - 0.25) < abs(mu_avg - 1.3)
        println("  ✓ CLOSER TO μ = 0.25 (fractal regime)")
    else
        println("  ✓ CLOSER TO μ = 1.3 (standard percolation)")
    end
end

#=============================================================================
                            MAIN EXECUTION
=============================================================================#

function main()
    println("="^70)
    println("LARGE SYSTEM PERCOLATION EXPONENT ANALYSIS")
    println("="^70)
    println("Question: Does μ converge to ~0.25 or ~1.3 as L → ∞?")
    println("="^70)

    # Configuration
    system_sizes = [64, 100]  # Start with these, add 150 if memory allows
    porosity_range = 0.32:0.02:0.50  # Fine sampling near threshold
    n_samples_per_point = 5  # Balance between statistics and runtime

    println("\n⚙ Configuration:")
    println("  System sizes: $(system_sizes)")
    println("  Porosity range: $(first(porosity_range)) to $(last(porosity_range))")
    println("  Samples per (L,p): $(n_samples_per_point)")
    println("  Total simulations: $(length(system_sizes) * length(porosity_range) * n_samples_per_point)")

    # Check memory estimate
    println("\n💾 Memory estimate:")
    for L in system_sizes
        mem_mb = (L^3 * 1) / 1024^2  # BitArray: 1 bit per voxel
        println("  L=$L: ~$(round(mem_mb, digits=1)) MB per scaffold")
    end

    println("\n🚀 Starting simulations...")
    println("="^70)

    # Storage for all results
    all_samples = PercolationSample[]
    power_law_fits = PowerLawFit[]

    # Run simulations for each system size
    for L in system_sizes
        println("\n" * "─"^70)
        println("SYSTEM SIZE L = $L")
        println("─"^70)

        L_samples = PercolationSample[]

        for p in porosity_range
            sample = run_percolation_samples(L, p, n_samples_per_point)
            push!(L_samples, sample)
            push!(all_samples, sample)
        end

        # Fit power law for this system size
        println("\n📈 Fitting power law for L=$L...")
        fit = fit_power_law(L_samples)
        push!(power_law_fits, fit)

        println("  p_c ≈ $(round(fit.p_c, digits=3))")
        println("  μ = $(round(fit.mu, digits=3)) ± $(round(fit.mu_err, digits=3))")
        println("  R² = $(round(fit.R2, digits=3))")
        println("  n_points = $(fit.n_points)")
    end

    # Finite-size scaling analysis
    finite_size_scaling(power_law_fits)

    # Summary
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)

    total_sims = sum(s.n_samples for s in all_samples)
    total_valid = sum(length(s.tortuosities) for s in all_samples)

    println("Total simulations: $(total_sims)")
    println("Valid measurements: $(total_valid) ($(round(100*total_valid/total_sims, digits=1))%)")

    println("\n✅ Analysis complete!")
    println("\n💡 Key finding:")

    mu_avg = mean([fit.mu for fit in power_law_fits])
    if abs(mu_avg - 0.25) < 0.3
        println("  μ ≈ $(round(mu_avg, digits=2)) suggests ANOMALOUS/FRACTAL scaling")
        println("  This supports the φ-based scaling hypothesis")
    elseif abs(mu_avg - 1.3) < 0.3
        println("  μ ≈ $(round(mu_avg, digits=2)) matches STANDARD 3D percolation")
        println("  This suggests classical critical behavior")
    else
        println("  μ ≈ $(round(mu_avg, digits=2)) is INTERMEDIATE")
        println("  Need larger systems or more statistics to resolve")
    end

    println("="^70)
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(42)  # Reproducibility
    main()
end
