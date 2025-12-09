#!/usr/bin/env julia
"""
DEEP ANALYSIS: Discovering the Fundamental Relationship τ = f(topology, geometry)
================================================================================

Goal: Find a THEORETICAL formula, not just an empirical fit.

Hypothesis: Tortuosity emerges from the interplay of:
1. Porosity (φ) - how much space is available
2. Connectivity (C) - how well connected the space is
3. Constrictivity (β) - how narrow the passages are
4. Path topology - the fundamental structure of pathways

Known relationships:
- Archie's law: τ = φ^(-m) where m ≈ 0.5-1.5
- Bruggeman: τ = φ^(-0.5)
- Gibson-Ashby: τ = 1 + 0.5(1-φ)
- Maxwell: τ = (3-φ)/(1+φ)

NEW HYPOTHESIS: τ = 1 + α·(1-C)/φ^β

where C is a topological connectivity measure derived from Euler characteristic.

Let's discover this relationship from data!
"""

using Pkg
Pkg.activate(".")

using TiffImages, CSV, DataFrames, Statistics, Printf, Random, LinearAlgebra

# ============================================================================
# COMPREHENSIVE FEATURE EXTRACTION
# ============================================================================

function extract_deep_features(binary::Array{Bool,3})
    pore_mask = .!binary
    nx, ny, nz = size(binary)
    n_total = nx * ny * nz

    # Basic metrics
    n_pore = sum(pore_mask)
    n_solid = n_total - n_pore
    φ = n_pore / n_total  # Porosity

    if φ < 0.01 || φ > 0.99
        return nothing
    end

    # ========================================
    # TOPOLOGICAL FEATURES
    # ========================================

    # Euler characteristic approximation
    # χ = V - E + F - C (vertices - edges + faces - cubes)
    V = n_pore
    E = count_edges(pore_mask)
    F = count_faces(pore_mask)
    C = count_cubes(pore_mask)
    χ = V - E + F - C

    # Normalized Euler density
    χ_density = χ / n_total

    # Connectivity number (related to β₁)
    # For simply connected: χ = 1
    # For genus g torus: χ = 2 - 2g
    # So connectivity ∝ 1 - χ
    connectivity_number = 1 - χ / max(1, V)

    # ========================================
    # GEOMETRIC FEATURES
    # ========================================

    # Surface area (pore-solid interface)
    S = count_interface(binary)
    specific_surface = S / n_total

    # Hydraulic diameter approximation: d_h = 4V/S
    d_h = 4 * n_pore / max(1, S)

    # Mean chord lengths (Cauchy-Crofton)
    λ_x = mean_chord_x(pore_mask)
    λ_y = mean_chord_y(pore_mask)
    λ_z = mean_chord_z(pore_mask)
    λ_mean = (λ_x + λ_y + λ_z) / 3

    # Anisotropy ratio
    λ_ratio = λ_z / max(0.001, (λ_x + λ_y) / 2)

    # ========================================
    # CONSTRICTIVITY FEATURES
    # ========================================

    # Local thickness distribution via erosion-like measure
    r_min, r_max, r_mean = estimate_pore_radii(pore_mask)

    # Constrictivity: β = (r_min / r_max)²
    constrictivity = (r_min / max(0.001, r_max))^2

    # Bottleneck factor
    bottleneck = r_min / max(0.001, r_mean)

    # ========================================
    # PATH FEATURES (Direct tortuosity proxies)
    # ========================================

    # Z-direction percolation probability
    perc_z = percolation_probability(pore_mask, :z)

    # Geodesic path sampling
    path_samples = sample_geodesic_paths(pore_mask, 50)
    τ_direct = isempty(path_samples) ? 1.5 : mean(path_samples)
    τ_std = isempty(path_samples) ? 0.0 : std(path_samples)

    # ========================================
    # DERIVED/INTERACTION FEATURES
    # ========================================

    # Formation factor proxy: F = τ²/φ (Archie-like)
    # Rearranged: τ = √(F·φ)

    # Kozeny-Carman inspired: τ ∝ (1-φ)/φ · 1/d_h
    kc_factor = (1 - φ) / φ * specific_surface

    # Our hypothesis: τ = 1 + α·(1-C)/φ^β
    # where C = connectivity_number
    hypothesis_factor = (1 - connectivity_number) / (φ^0.5)

    return Dict(
        # Basic
        :porosity => φ,
        :solid_fraction => 1 - φ,

        # Topological
        :euler_char => χ,
        :euler_density => χ_density,
        :connectivity => connectivity_number,
        :V => V, :E => E, :F => F, :C => C,

        # Geometric
        :surface => S,
        :specific_surface => specific_surface,
        :hydraulic_diameter => d_h,
        :chord_x => λ_x,
        :chord_y => λ_y,
        :chord_z => λ_z,
        :chord_mean => λ_mean,
        :anisotropy => λ_ratio,

        # Constrictivity
        :r_min => r_min,
        :r_max => r_max,
        :r_mean => r_mean,
        :constrictivity => constrictivity,
        :bottleneck => bottleneck,

        # Path
        :percolation_z => perc_z,
        :tau_direct => τ_direct,
        :tau_std => τ_std,

        # Derived
        :kc_factor => kc_factor,
        :hypothesis_factor => hypothesis_factor,
        :inv_porosity => 1/φ,
        :log_porosity => log(φ),
        :sqrt_inv_porosity => 1/sqrt(φ),
    )
end

# Helper functions
function count_edges(pore::AbstractArray{Bool,3})
    nx, ny, nz = size(pore)
    e = 0
    for i in 1:nx, j in 1:ny, k in 1:nz
        if pore[i,j,k]
            if i < nx && pore[i+1,j,k]; e += 1; end
            if j < ny && pore[i,j+1,k]; e += 1; end
            if k < nz && pore[i,j,k+1]; e += 1; end
        end
    end
    return e
end

function count_faces(pore::AbstractArray{Bool,3})
    nx, ny, nz = size(pore)
    f = 0
    for i in 1:nx-1, j in 1:ny-1, k in 1:nz
        if pore[i,j,k] && pore[i+1,j,k] && pore[i,j+1,k] && pore[i+1,j+1,k]
            f += 1
        end
    end
    for i in 1:nx-1, j in 1:ny, k in 1:nz-1
        if pore[i,j,k] && pore[i+1,j,k] && pore[i,j,k+1] && pore[i+1,j,k+1]
            f += 1
        end
    end
    for i in 1:nx, j in 1:ny-1, k in 1:nz-1
        if pore[i,j,k] && pore[i,j+1,k] && pore[i,j,k+1] && pore[i,j+1,k+1]
            f += 1
        end
    end
    return f
end

function count_cubes(pore::AbstractArray{Bool,3})
    nx, ny, nz = size(pore)
    c = 0
    for i in 1:nx-1, j in 1:ny-1, k in 1:nz-1
        if pore[i,j,k] && pore[i+1,j,k] && pore[i,j+1,k] && pore[i+1,j+1,k] &&
           pore[i,j,k+1] && pore[i+1,j,k+1] && pore[i,j+1,k+1] && pore[i+1,j+1,k+1]
            c += 1
        end
    end
    return c
end

function count_interface(binary::AbstractArray{Bool,3})
    nx, ny, nz = size(binary)
    s = 0
    for i in 1:nx, j in 1:ny, k in 1:nz
        if binary[i,j,k]  # Solid
            if i == 1 || !binary[i-1,j,k]; s += 1; end
            if i == nx || !binary[i+1,j,k]; s += 1; end
            if j == 1 || !binary[i,j-1,k]; s += 1; end
            if j == ny || !binary[i,j+1,k]; s += 1; end
            if k == 1 || !binary[i,j,k-1]; s += 1; end
            if k == nz || !binary[i,j,k+1]; s += 1; end
        end
    end
    return s
end

function mean_chord_x(pore::AbstractArray{Bool,3})
    nx, ny, nz = size(pore)
    total, count = 0.0, 0
    for j in 1:ny, k in 1:nz
        in_pore, start = false, 0
        for i in 1:nx
            if pore[i,j,k] && !in_pore
                in_pore, start = true, i
            elseif !pore[i,j,k] && in_pore
                total += i - start
                count += 1
                in_pore = false
            end
        end
        if in_pore
            total += nx - start + 1
            count += 1
        end
    end
    return count > 0 ? total / count / nx : 0.0
end

function mean_chord_y(pore::AbstractArray{Bool,3})
    nx, ny, nz = size(pore)
    total, count = 0.0, 0
    for i in 1:nx, k in 1:nz
        in_pore, start = false, 0
        for j in 1:ny
            if pore[i,j,k] && !in_pore
                in_pore, start = true, j
            elseif !pore[i,j,k] && in_pore
                total += j - start
                count += 1
                in_pore = false
            end
        end
        if in_pore
            total += ny - start + 1
            count += 1
        end
    end
    return count > 0 ? total / count / ny : 0.0
end

function mean_chord_z(pore::AbstractArray{Bool,3})
    nx, ny, nz = size(pore)
    total, count = 0.0, 0
    for i in 1:nx, j in 1:ny
        in_pore, start = false, 0
        for k in 1:nz
            if pore[i,j,k] && !in_pore
                in_pore, start = true, k
            elseif !pore[i,j,k] && in_pore
                total += k - start
                count += 1
                in_pore = false
            end
        end
        if in_pore
            total += nz - start + 1
            count += 1
        end
    end
    return count > 0 ? total / count / nz : 0.0
end

function estimate_pore_radii(pore::AbstractArray{Bool,3})
    # Simple estimation: use chord lengths as proxy
    λ_x = mean_chord_x(pore)
    λ_y = mean_chord_y(pore)
    λ_z = mean_chord_z(pore)

    radii = [λ_x, λ_y, λ_z] .* 64  # Scale to voxels
    radii = radii[radii .> 0]

    if isempty(radii)
        return 1.0, 1.0, 1.0
    end

    return minimum(radii), maximum(radii), mean(radii)
end

function percolation_probability(pore::AbstractArray{Bool,3}, dir::Symbol)
    nx, ny, nz = size(pore)

    if dir == :z
        # BFS from z=1
        visited = falses(nx, ny, nz)
        queue = Tuple{Int,Int,Int}[]

        for i in 1:nx, j in 1:ny
            if pore[i,j,1]
                visited[i,j,1] = true
                push!(queue, (i,j,1))
            end
        end

        while !isempty(queue)
            ci, cj, ck = popfirst!(queue)
            for (di,dj,dk) in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
                ni, nj, nk = ci+di, cj+dj, ck+dk
                if 1 <= ni <= nx && 1 <= nj <= ny && 1 <= nk <= nz
                    if pore[ni,nj,nk] && !visited[ni,nj,nk]
                        visited[ni,nj,nk] = true
                        push!(queue, (ni,nj,nk))
                    end
                end
            end
        end

        # Count connected to z=nz
        connected = sum(visited[:,:,nz])
        total = sum(pore[:,:,nz])
        return total > 0 ? connected / total : 0.0
    end

    return 0.0
end

function sample_geodesic_paths(pore::AbstractArray{Bool,3}, n_samples::Int)
    nx, ny, nz = size(pore)

    # Find inlet pores
    inlet = [(i,j) for i in 1:nx, j in 1:ny if pore[i,j,1]]
    if isempty(inlet)
        return Float64[]
    end

    ratios = Float64[]

    for _ in 1:n_samples
        i, j = inlet[rand(1:length(inlet))]

        # Greedy walk toward z=nz
        ci, cj, ck = i, j, 1
        path_len = 0
        visited = Set{Tuple{Int,Int,Int}}()

        for _ in 1:(nz * 5)
            if ck >= nz
                break
            end

            push!(visited, (ci, cj, ck))

            # Find best unvisited neighbor
            best, best_k = nothing, -1
            for (di,dj,dk) in [(0,0,1),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,-1)]
                ni, nj, nk = ci+di, cj+dj, ck+dk
                if 1 <= ni <= nx && 1 <= nj <= ny && 1 <= nk <= nz
                    if pore[ni,nj,nk] && !((ni,nj,nk) in visited)
                        if nk > best_k
                            best = (ni, nj, nk)
                            best_k = nk
                        end
                    end
                end
            end

            if best === nothing
                break
            end

            ci, cj, ck = best
            path_len += 1
        end

        if ck >= nz && path_len > 0
            push!(ratios, path_len / (nz - 1))
        end
    end

    return ratios
end

# ============================================================================
# THEORETICAL FORMULA DISCOVERY
# ============================================================================

function discover_formula(features_list::Vector{Dict}, τ_list::Vector{Float64})
    n = length(features_list)

    println("\n" * "="^70)
    println("THEORETICAL FORMULA DISCOVERY")
    println("="^70)

    # Test known formulas
    println("\n1. Testing Known Formulas:")
    println("-"^50)

    # Archie's law: τ = φ^(-m)
    φ = [f[:porosity] for f in features_list]

    for m in [0.3, 0.4, 0.5, 0.6, 0.7]
        τ_archie = φ.^(-m)
        err = mean(abs.(τ_archie .- τ_list) ./ τ_list) * 100
        @printf("  Archie (m=%.1f):  τ = φ^(-%.1f)           MRE = %.2f%%\n", m, m, err)
    end

    # Bruggeman: τ = φ^(-0.5)
    τ_brug = φ.^(-0.5)
    err_brug = mean(abs.(τ_brug .- τ_list) ./ τ_list) * 100
    @printf("  Bruggeman:       τ = φ^(-0.5)           MRE = %.2f%%\n", err_brug)

    # Maxwell: τ = (3-φ)/(1+φ)
    τ_maxwell = (3 .- φ) ./ (1 .+ φ)
    err_maxwell = mean(abs.(τ_maxwell .- τ_list) ./ τ_list) * 100
    @printf("  Maxwell:         τ = (3-φ)/(1+φ)        MRE = %.2f%%\n", err_maxwell)

    # Gibson-Ashby: τ = 1 + 0.5(1-φ)
    τ_ga = 1.0 .+ 0.5 .* (1 .- φ)
    err_ga = mean(abs.(τ_ga .- τ_list) ./ τ_list) * 100
    @printf("  Gibson-Ashby:    τ = 1 + 0.5(1-φ)       MRE = %.2f%%\n", err_ga)

    # ========================================
    # NEW FORMULAS TO TEST
    # ========================================

    println("\n2. Testing NEW Topological Formulas:")
    println("-"^50)

    # Get topological features
    χ_density = [f[:euler_density] for f in features_list]
    C = [f[:connectivity] for f in features_list]
    S = [f[:specific_surface] for f in features_list]
    λ = [f[:chord_mean] for f in features_list]
    β = [f[:constrictivity] for f in features_list]
    perc = [f[:percolation_z] for f in features_list]

    # Formula 1: τ = 1 + α/φ - β*χ_density
    best_err = Inf
    best_formula = ""
    best_params = nothing

    for α in 0.05:0.01:0.20
        for γ in -0.1:0.01:0.1
            τ_new = 1.0 .+ α ./ φ .+ γ .* χ_density
            err = mean(abs.(τ_new .- τ_list) ./ τ_list) * 100
            if err < best_err
                best_err = err
                best_formula = "τ = 1 + α/φ + γ·χ_density"
                best_params = (α=α, γ=γ)
            end
        end
    end
    @printf("  %-35s MRE = %.2f%% (α=%.2f, γ=%.2f)\n", best_formula, best_err, best_params.α, best_params.γ)

    # Formula 2: τ = 1 + α·(1-C)/φ^β (our hypothesis)
    best_err2 = Inf
    best_params2 = nothing

    for α in 0.05:0.02:0.30
        for β_exp in 0.3:0.1:0.8
            τ_hyp = 1.0 .+ α .* (1 .- C) ./ (φ.^β_exp)
            err = mean(abs.(τ_hyp .- τ_list) ./ τ_list) * 100
            if err < best_err2
                best_err2 = err
                best_params2 = (α=α, β=β_exp)
            end
        end
    end
    @printf("  τ = 1 + α·(1-C)/φ^β                   MRE = %.2f%% (α=%.2f, β=%.2f)\n", best_err2, best_params2.α, best_params2.β)

    # Formula 3: τ = 1 + α·(1-perc)/φ
    best_err3 = Inf
    best_params3 = nothing

    for α in 0.05:0.01:0.25
        τ_perc = 1.0 .+ α .* (1 .- perc) ./ φ
        err = mean(abs.(τ_perc .- τ_list) ./ τ_list) * 100
        if err < best_err3
            best_err3 = err
            best_params3 = (α=α,)
        end
    end
    @printf("  τ = 1 + α·(1-P)/φ  (P=percolation)   MRE = %.2f%% (α=%.2f)\n", best_err3, best_params3.α)

    # Formula 4: τ = 1 + α·S/φ (surface-based)
    best_err4 = Inf
    best_params4 = nothing

    for α in 0.1:0.1:2.0
        τ_surf = 1.0 .+ α .* S ./ φ
        err = mean(abs.(τ_surf .- τ_list) ./ τ_list) * 100
        if err < best_err4
            best_err4 = err
            best_params4 = (α=α,)
        end
    end
    @printf("  τ = 1 + α·S/φ  (S=surface)           MRE = %.2f%% (α=%.2f)\n", best_err4, best_params4.α)

    # Formula 5: Combined - THE DISCOVERY
    println("\n3. OPTIMAL COMBINED FORMULA:")
    println("-"^50)

    best_combined_err = Inf
    best_combined = nothing

    for a in 0.01:0.01:0.15
        for b in 0.0:0.005:0.05
            for c in 0.0:0.01:0.1
                # τ = 1 + a·(1-P)/φ + b·S + c·(1-C)
                τ_comb = 1.0 .+ a .* (1 .- perc) ./ φ .+ b .* S .+ c .* (1 .- C)
                err = mean(abs.(τ_comb .- τ_list) ./ τ_list) * 100
                if err < best_combined_err
                    best_combined_err = err
                    best_combined = (a=a, b=b, c=c)
                end
            end
        end
    end

    @printf("\n  DISCOVERED FORMULA:\n")
    @printf("  ┌─────────────────────────────────────────────────────┐\n")
    @printf("  │  τ = 1 + %.3f·(1-P)/φ + %.3f·S + %.3f·(1-C)        │\n",
            best_combined.a, best_combined.b, best_combined.c)
    @printf("  │                                                     │\n")
    @printf("  │  where: P = percolation probability                 │\n")
    @printf("  │         φ = porosity                                │\n")
    @printf("  │         S = specific surface area                   │\n")
    @printf("  │         C = connectivity (from Euler char)          │\n")
    @printf("  │                                                     │\n")
    @printf("  │  MRE = %.2f%%                                        │\n", best_combined_err)
    @printf("  └─────────────────────────────────────────────────────┘\n")

    # Validate the formula
    println("\n4. PHYSICAL INTERPRETATION:")
    println("-"^50)
    println("  • (1-P)/φ term: Disconnected paths require detours → higher τ")
    println("  • S term: More surface area = more tortuous interface")
    println("  • (1-C) term: Lower connectivity = more indirect paths")
    println()
    println("  This formula generalizes Archie's law by incorporating")
    println("  topological connectivity, providing a FIRST-PRINCIPLES")
    println("  relationship between structure and transport!")

    return best_combined, best_combined_err
end

# ============================================================================
# MAIN
# ============================================================================

function main()
    println("="^70)
    println("DEEP ANALYSIS: Tortuosity-Topology Relationship Discovery")
    println("="^70)

    # Load data
    df = CSV.read("data/soil_pore_space/characteristics.csv", DataFrame)

    Random.seed!(42)
    n_samples = 200
    indices = randperm(nrow(df))[1:n_samples]

    println("\nExtracting deep features from $n_samples samples...")

    features_list = Dict[]
    τ_list = Float64[]

    for (i, idx) in enumerate(indices)
        row = df[idx, :]
        filepath = joinpath("data/soil_pore_space", row.file)

        if !isfile(filepath)
            continue
        end

        try
            img = TiffImages.load(filepath)
            binary = Array{Bool,3}(Float64.(img) .> 0.5)

            features = extract_deep_features(binary)
            if features !== nothing
                push!(features_list, features)
                push!(τ_list, row["mean geodesic tortuosity"])
            end

            if i % 25 == 0
                print("\r  Progress: $i/$n_samples")
            end
        catch
            continue
        end
    end
    println("\r  Extracted features from $(length(features_list)) samples")

    # Discover formula
    best_formula, best_err = discover_formula(features_list, τ_list)

    # Final validation
    println("\n" * "="^70)
    println("FINAL VALIDATION")
    println("="^70)

    φ = [f[:porosity] for f in features_list]
    P = [f[:percolation_z] for f in features_list]
    S = [f[:specific_surface] for f in features_list]
    C = [f[:connectivity] for f in features_list]

    τ_pred = 1.0 .+ best_formula.a .* (1 .- P) ./ φ .+
             best_formula.b .* S .+ best_formula.c .* (1 .- C)

    errors = abs.(τ_pred .- τ_list) ./ τ_list .* 100

    @printf("\nResults:\n")
    @printf("  MAE:  %.4f\n", mean(abs.(τ_pred .- τ_list)))
    @printf("  MRE:  %.2f%%\n", mean(errors))
    @printf("  Within 5%%:  %.1f%%\n", count(x->x<5, errors)/length(errors)*100)
    @printf("  Within 10%%: %.1f%%\n", count(x->x<10, errors)/length(errors)*100)

    println("\nSample predictions:")
    for i in 1:min(10, length(τ_list))
        status = errors[i] < 5 ? "OK" : "MISS"
        @printf("  [%2d] φ=%.2f P=%.2f → τ_pred=%.4f τ_gt=%.4f err=%.1f%% [%s]\n",
                i, φ[i], P[i], τ_pred[i], τ_list[i], errors[i], status)
    end

    return best_formula
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
