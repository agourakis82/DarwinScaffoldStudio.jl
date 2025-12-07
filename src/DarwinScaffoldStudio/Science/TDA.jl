"""
Topological Data Analysis (TDA) for Scaffold Characterization
==============================================================

Uses persistent homology to characterize scaffold pore networks:
- H₀: Connected components (pore clusters)
- H₁: Loops/tunnels (interconnectivity pathways)
- H₂: Voids/cavities (enclosed spaces)

Key metrics:
- Betti numbers (β₀, β₁, β₂): Count of topological features
- Persistence: Lifetime of features (robustness measure)
- Euler characteristic: χ = β₀ - β₁ + β₂

References:
- Edelsbrunner & Harer (2010) "Computational Topology"
- Carlsson (2009) "Topology and Data"
- Ghrist (2014) "Elementary Applied Topology"
"""
module TDA

using Ripserer
using PersistenceDiagrams
using Statistics
using LinearAlgebra
using Random: randperm

export compute_persistent_homology, analyze_pore_topology
export betti_numbers, persistence_entropy, bottleneck_distance
export PersistenceSummary, plot_persistence_diagram, plot_betti_barcode

# ============================================================================
# Data Structures
# ============================================================================

"""
    PersistenceSummary

Summary of persistence diagram for a single homology dimension.
"""
struct PersistenceSummary
    dimension::Int
    n_features::Int
    n_essential::Int          # Features that persist to infinity
    births::Vector{Float64}
    deaths::Vector{Float64}
    persistence::Vector{Float64}
    total_persistence::Float64
    mean_persistence::Float64
    max_persistence::Float64
    entropy::Float64          # Persistence entropy
end

# ============================================================================
# Core Computation
# ============================================================================

"""
    compute_persistent_homology(scaffold_volume; max_dim=2, n_samples=5000)

Compute persistence diagrams for scaffold pore space.

Arguments:
- scaffold_volume: 3D array (>0 = pore space)
- max_dim: Maximum homology dimension (default 2 for H₀, H₁, H₂)
- n_samples: Maximum points to sample (for performance)

Returns Dict with:
- diagrams: Dict of "H0", "H1", "H2" → [(birth, death), ...]
- betti_numbers: [β₀, β₁, β₂]
- summaries: Dict of PersistenceSummary per dimension
- euler_characteristic: χ = β₀ - β₁ + β₂
"""
function compute_persistent_homology(
    scaffold_volume::AbstractArray;
    max_dim::Int=2,
    n_samples::Int=5000,
    threshold::Float64=0.0
)
    # Extract pore space coordinates
    pore_coords = findall(scaffold_volume .> threshold)

    if isempty(pore_coords)
        return Dict(
            "diagrams" => Dict("H0" => [], "H1" => [], "H2" => []),
            "betti_numbers" => [0, 0, 0],
            "summaries" => Dict(),
            "euler_characteristic" => 0,
            "error" => "No pore space found"
        )
    end

    # Convert to vector of tuples (Ripserer expects this format)
    points = [(Float64(p[1]), Float64(p[2]), Float64(p[3])) for p in pore_coords]

    # Subsample if too large (Ripserer scales as O(n³))
    if length(points) > n_samples
        indices = randperm(length(points))[1:n_samples]
        points = points[indices]
        @info "TDA: Subsampled to $n_samples points for performance"
    end

    # Compute Vietoris-Rips filtration
    try
        result = ripserer(points; dim_max=max_dim)

        # Extract persistence diagrams
        diagrams = Dict{String, Vector{Tuple{Float64, Float64}}}()
        summaries = Dict{String, PersistenceSummary}()
        betti = Int[]

        for dim in 0:max_dim
            key = "H$(dim)"

            if dim + 1 <= length(result)
                # Extract (birth, death) pairs
                pairs = [(Float64(interval.birth),
                         isinf(interval.death) ? Inf : Float64(interval.death))
                        for interval in result[dim + 1]]

                diagrams[key] = pairs
                summaries[key] = summarize_persistence(pairs, dim)

                # Betti number = count of infinite persistence features
                push!(betti, sum(isinf(d) for (b, d) in pairs))
            else
                diagrams[key] = []
                summaries[key] = PersistenceSummary(dim, 0, 0, [], [], [], 0.0, 0.0, 0.0, 0.0)
                push!(betti, 0)
            end
        end

        # Euler characteristic
        χ = length(betti) >= 3 ? betti[1] - betti[2] + betti[3] : betti[1]

        return Dict(
            "diagrams" => diagrams,
            "betti_numbers" => betti,
            "summaries" => summaries,
            "euler_characteristic" => χ,
            "n_points" => size(points, 1)
        )

    catch e
        @warn "TDA computation failed" exception=e
        return Dict(
            "diagrams" => Dict("H0" => [], "H1" => [], "H2" => []),
            "betti_numbers" => [0, 0, 0],
            "summaries" => Dict(),
            "euler_characteristic" => 0,
            "error" => string(e)
        )
    end
end

"""
    summarize_persistence(pairs, dim)

Create PersistenceSummary from (birth, death) pairs.
"""
function summarize_persistence(pairs::Vector{Tuple{Float64, Float64}}, dim::Int)
    if isempty(pairs)
        return PersistenceSummary(dim, 0, 0, [], [], [], 0.0, 0.0, 0.0, 0.0)
    end

    births = [p[1] for p in pairs]
    deaths = [p[2] for p in pairs]

    # Finite persistence values
    finite_pairs = [(b, d) for (b, d) in pairs if !isinf(d)]
    persistence = [d - b for (b, d) in finite_pairs]

    n_essential = sum(isinf(d) for (b, d) in pairs)

    total_pers = isempty(persistence) ? 0.0 : sum(persistence)
    mean_pers = isempty(persistence) ? 0.0 : mean(persistence)
    max_pers = isempty(persistence) ? 0.0 : maximum(persistence)

    # Persistence entropy
    ent = persistence_entropy(persistence)

    return PersistenceSummary(
        dim, length(pairs), n_essential,
        births, deaths, persistence,
        total_pers, mean_pers, max_pers, ent
    )
end

# ============================================================================
# Betti Numbers and Derived Metrics
# ============================================================================

"""
    betti_numbers(scaffold_volume; kwargs...)

Compute Betti numbers (β₀, β₁, β₂) for scaffold.

- β₀: Number of connected components
- β₁: Number of independent loops/tunnels
- β₂: Number of enclosed voids

Higher β₁ indicates better interconnectivity.
"""
function betti_numbers(scaffold_volume::AbstractArray; kwargs...)
    ph = compute_persistent_homology(scaffold_volume; kwargs...)
    return get(ph, "betti_numbers", [0, 0, 0])
end

"""
    persistence_entropy(persistence_values)

Compute Shannon entropy of persistence diagram.

High entropy = many features with similar persistence (complex structure)
Low entropy = few dominant features (simple structure)

Formula: H = -Σ pᵢ log(pᵢ) where pᵢ = persᵢ / Σpers
"""
function persistence_entropy(persistence::Vector{Float64})
    if isempty(persistence) || sum(persistence) == 0
        return 0.0
    end

    # Normalize to probabilities
    total = sum(persistence)
    p = persistence ./ total

    # Shannon entropy
    H = -sum(pᵢ * log(pᵢ) for pᵢ in p if pᵢ > 0)

    return H
end

"""
    bottleneck_distance(diagram1, diagram2)

Compute bottleneck distance between two persistence diagrams.

This measures the similarity of topological features between scaffolds.
"""
function bottleneck_distance(
    diagram1::Vector{Tuple{Float64, Float64}},
    diagram2::Vector{Tuple{Float64, Float64}}
)
    # Convert to finite pairs only (exclude essential features)
    pairs1 = [(b, d) for (b, d) in diagram1 if !isinf(d)]
    pairs2 = [(b, d) for (b, d) in diagram2 if !isinf(d)]

    if isempty(pairs1) && isempty(pairs2)
        return 0.0
    end

    # Simple approximation: compare sorted persistence values
    pers1 = sort([d - b for (b, d) in pairs1], rev=true)
    pers2 = sort([d - b for (b, d) in pairs2], rev=true)

    # Pad shorter list with zeros
    n = max(length(pers1), length(pers2))
    while length(pers1) < n; push!(pers1, 0.0); end
    while length(pers2) < n; push!(pers2, 0.0); end

    # Maximum difference in sorted persistence
    return maximum(abs.(pers1 .- pers2))
end

# ============================================================================
# High-Level Analysis
# ============================================================================

"""
    analyze_pore_topology(scaffold_volume; kwargs...)

Comprehensive topological analysis with interpretable metrics.

Returns Dict with:
- num_components: β₀ (connected pore clusters)
- num_loops: β₁ (tunnel/channel count)
- num_voids: β₂ (enclosed cavity count)
- interconnectivity_score: Normalized β₁ metric
- mean_loop_persistence: Average robustness of tunnels
- euler_characteristic: χ = β₀ - β₁ + β₂
- topological_complexity: Total feature count
- persistence_entropy_H1: Complexity of tunnel network
"""
function analyze_pore_topology(scaffold_volume::AbstractArray; kwargs...)
    ph = compute_persistent_homology(scaffold_volume; kwargs...)

    betti = get(ph, "betti_numbers", [0, 0, 0])
    summaries = get(ph, "summaries", Dict())

    # Extract H1 summary for interconnectivity analysis
    H1_summary = get(summaries, "H1", PersistenceSummary(1, 0, 0, [], [], [], 0.0, 0.0, 0.0, 0.0))
    H2_summary = get(summaries, "H2", PersistenceSummary(2, 0, 0, [], [], [], 0.0, 0.0, 0.0, 0.0))

    # Interconnectivity score: normalized β₁
    # Higher is better for nutrient transport
    porosity = sum(scaffold_volume .> 0) / length(scaffold_volume)
    expected_loops = porosity * 10  # Heuristic baseline
    interconnectivity_score = betti[2] / max(1, expected_loops)

    return Dict(
        "num_components" => betti[1],
        "num_loops" => betti[2],
        "num_voids" => length(betti) >= 3 ? betti[3] : 0,
        "interconnectivity_score" => min(1.0, interconnectivity_score),
        "mean_loop_persistence" => H1_summary.mean_persistence,
        "max_loop_persistence" => H1_summary.max_persistence,
        "mean_void_persistence" => H2_summary.mean_persistence,
        "euler_characteristic" => get(ph, "euler_characteristic", 0),
        "topological_complexity" => H1_summary.n_features + H2_summary.n_features,
        "persistence_entropy_H1" => H1_summary.entropy,
        "persistence_entropy_H2" => H2_summary.entropy,
        "diagrams" => get(ph, "diagrams", Dict()),
        "betti_numbers" => betti
    )
end

# ============================================================================
# Visualization Helpers (Text-based for terminal)
# ============================================================================

"""
    plot_persistence_diagram(diagrams; dim=1)

Generate ASCII persistence diagram for terminal display.
"""
function plot_persistence_diagram(diagrams::Dict; dim::Int=1)
    key = "H$(dim)"
    pairs = get(diagrams, key, [])

    if isempty(pairs)
        return "No features in $key"
    end

    # Filter finite pairs
    finite_pairs = [(b, d) for (b, d) in pairs if !isinf(d)]
    essential_count = length(pairs) - length(finite_pairs)

    if isempty(finite_pairs)
        return "$key: $essential_count essential features (infinite persistence)"
    end

    # Scale to ASCII grid
    births = [p[1] for p in finite_pairs]
    deaths = [p[2] for p in finite_pairs]

    b_min, b_max = minimum(births), maximum(births)
    d_min, d_max = minimum(deaths), maximum(deaths)

    width = 40
    height = 20

    # Create grid
    grid = fill(' ', height, width)

    # Draw diagonal (birth = death)
    for i in 1:min(width, height)
        grid[height - i + 1, i] = '/'
    end

    # Plot points
    for (b, d) in finite_pairs
        x = round(Int, (b - b_min) / (b_max - b_min + 1e-10) * (width - 1)) + 1
        y = height - round(Int, (d - d_min) / (d_max - d_min + 1e-10) * (height - 1))
        if 1 <= x <= width && 1 <= y <= height
            grid[y, x] = '*'
        end
    end

    # Build output string
    output = "\n  Persistence Diagram $key\n"
    output *= "  " * "─" ^ width * "\n"

    for row in 1:height
        output *= "  │" * String(grid[row, :]) * "\n"
    end

    output *= "  └" * "─" ^ width * "\n"
    output *= "   Birth →\n"
    output *= "\n  Features: $(length(finite_pairs)) finite, $essential_count essential\n"
    output *= "  Persistence range: $(round(minimum(deaths .- births), digits=3)) - $(round(maximum(deaths .- births), digits=3))\n"

    return output
end

"""
    plot_betti_barcode(diagrams; dim=1, max_bars=20)

Generate ASCII barcode plot showing feature lifetimes.
"""
function plot_betti_barcode(diagrams::Dict; dim::Int=1, max_bars::Int=20)
    key = "H$(dim)"
    pairs = get(diagrams, key, [])

    if isempty(pairs)
        return "No features in $key"
    end

    # Sort by persistence (longest first)
    sorted_pairs = sort([(b, d) for (b, d) in pairs],
                        by=p -> isinf(p[2]) ? Inf : p[2] - p[1], rev=true)

    # Limit number of bars
    display_pairs = sorted_pairs[1:min(max_bars, length(sorted_pairs))]

    # Find scale
    all_births = [p[1] for p in display_pairs]
    all_deaths = [p[2] for p in display_pairs if !isinf(p[2])]

    min_val = minimum(all_births)
    max_val = isempty(all_deaths) ? min_val + 1 : maximum(all_deaths)

    width = 50

    output = "\n  Betti Barcode $key ($(length(pairs)) features, showing $(length(display_pairs)))\n"
    output *= "  " * "─" ^ (width + 10) * "\n"

    for (i, (b, d)) in enumerate(display_pairs)
        # Scale to width
        start = round(Int, (b - min_val) / (max_val - min_val + 1e-10) * width) + 1

        if isinf(d)
            stop = width
            bar = "─" ^ (stop - start) * "→"  # Arrow for infinite
        else
            stop = round(Int, (d - min_val) / (max_val - min_val + 1e-10) * width) + 1
            bar = "─" ^ max(1, stop - start)
        end

        # Pad and align
        prefix = " " ^ (start - 1)
        label = lpad(i, 3)

        output *= "  $label │$prefix$bar\n"
    end

    output *= "  " * "─" ^ (width + 10) * "\n"
    output *= "      $(round(min_val, digits=2))" * " " ^ (width - 10) * "$(round(max_val, digits=2))\n"

    return output
end

# ============================================================================
# Scaffold Comparison
# ============================================================================

"""
    compare_scaffolds(scaffold1, scaffold2; kwargs...)

Compare topological features of two scaffolds.

Returns Dict with:
- bottleneck_H0, H1, H2: Bottleneck distances per dimension
- betti_diff: Difference in Betti numbers
- similarity_score: Overall topological similarity [0, 1]
"""
function compare_scaffolds(
    scaffold1::AbstractArray,
    scaffold2::AbstractArray;
    kwargs...
)
    ph1 = compute_persistent_homology(scaffold1; kwargs...)
    ph2 = compute_persistent_homology(scaffold2; kwargs...)

    diagrams1 = get(ph1, "diagrams", Dict())
    diagrams2 = get(ph2, "diagrams", Dict())

    betti1 = get(ph1, "betti_numbers", [0, 0, 0])
    betti2 = get(ph2, "betti_numbers", [0, 0, 0])

    # Bottleneck distances per dimension
    bottleneck = Dict{String, Float64}()
    for dim in 0:2
        key = "H$(dim)"
        d1 = get(diagrams1, key, [])
        d2 = get(diagrams2, key, [])
        bottleneck[key] = bottleneck_distance(d1, d2)
    end

    # Betti number differences
    betti_diff = abs.(betti1 .- betti2)

    # Overall similarity (heuristic)
    total_bottleneck = sum(values(bottleneck))
    total_betti_diff = sum(betti_diff)

    # Normalize to [0, 1] where 1 = identical
    similarity = 1.0 / (1.0 + total_bottleneck + total_betti_diff)

    return Dict(
        "bottleneck_H0" => bottleneck["H0"],
        "bottleneck_H1" => bottleneck["H1"],
        "bottleneck_H2" => bottleneck["H2"],
        "betti_diff" => betti_diff,
        "betti_scaffold1" => betti1,
        "betti_scaffold2" => betti2,
        "similarity_score" => similarity
    )
end

end # module
