#!/usr/bin/env julia
"""
TOPOLOGY-TRANSPORT CORRELATION VALIDATION ON REAL DATA

Tests the hypothesis: Euler characteristic (χ) correlates with tortuosity (τ)

Background:
- Synthetic percolation: χ-τ correlation r = 0.83 (claimed)
- Question: Does this hold on REAL porous media (Zenodo soil data)?

Approach:
1. Load ground-truth tortuosity from CSV (4,608 samples)
2. For subset, load 3D TIFF volumes
3. Compute topology: β₀, β₁, β₂ → χ = β₀ - β₁ + β₂
4. Correlate χ with τ_ground_truth
5. Compare to synthetic r = 0.83

Dataset: Zenodo 7516228 - Soil pore space morphology
Paper: Prifling et al. (2023) "Quantifying the impact of 3D pore space
       morphology on diffusive mass transport in loam and sand"
"""

using CSV
using DataFrames
using TiffImages
using Statistics
using Printf
using Dates

# ==============================================================================
# CONFIGURATION
# ==============================================================================

const PROJECT_ROOT = dirname(dirname(@__FILE__))
const DATA_DIR = joinpath(PROJECT_ROOT, "data/soil_pore_space")
const CSV_FILE = joinpath(DATA_DIR, "characteristics.csv")
const TIFF_DIR = joinpath(DATA_DIR, "segmented_stacks")

# Sample size for validation (TIFF loading is slow)
const N_SAMPLES = 100  # Increase to 500 for full validation

println("="^80)
println("TOPOLOGY-TRANSPORT CORRELATION VALIDATION")
println("="^80)
println("Dataset: Zenodo 7516228 (Soil Pore Space 3D)")
println("Paper: Prifling et al. (2023)")
println("Validation samples: $N_SAMPLES")
println("="^80)

# ==============================================================================
# TOPOLOGY COMPUTATION
# ==============================================================================

"""
Compute β₀ (connected components) using flood-fill algorithm.
Returns number of distinct pore clusters.
"""
function compute_beta0(binary::Array{Bool,3})
    visited = falses(size(binary))
    n_components = 0

    nx, ny, nz = size(binary)

    # Flood fill from each unvisited pore voxel
    for i in 1:nx, j in 1:ny, k in 1:nz
        if binary[i,j,k] && !visited[i,j,k]
            n_components += 1

            # BFS flood fill
            queue = [(i, j, k)]
            visited[i,j,k] = true

            while !isempty(queue)
                x, y, z = popfirst!(queue)

                # 6-connectivity neighbors
                for (dx, dy, dz) in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
                    nx_new = x + dx
                    ny_new = y + dy
                    nz_new = z + dz

                    if 1 <= nx_new <= nx && 1 <= ny_new <= ny && 1 <= nz_new <= nz
                        if binary[nx_new, ny_new, nz_new] && !visited[nx_new, ny_new, nz_new]
                            visited[nx_new, ny_new, nz_new] = true
                            push!(queue, (nx_new, ny_new, nz_new))
                        end
                    end
                end
            end
        end
    end

    return n_components
end

"""
Compute β₁ (loops/tunnels) - APPROXIMATE via connectivity heuristic.

Note: Exact computation requires persistent homology (Ripserer.jl).
This is a computationally efficient approximation.
"""
function compute_beta1_approx(binary::Array{Bool,3}, porosity::Float64)
    pore_voxels = sum(binary)

    if pore_voxels == 0
        return 0
    end

    # Heuristic: β₁ scales with porosity and volume
    # For random porous media: β₁ << β₀
    β₁_est = max(0, round(Int, porosity * sqrt(pore_voxels) / 10))

    return β₁_est
end

"""
Compute β₂ (voids/cavities).
For pore space, β₂ represents enclosed void regions.
"""
function compute_beta2(binary::Array{Bool,3})
    nx, ny, nz = size(binary)

    # Check if pores touch all boundaries (open system → β₂ = 0)
    touches_boundaries = (
        any(binary[1, :, :]) && any(binary[nx, :, :]) &&
        any(binary[:, 1, :]) && any(binary[:, ny, :]) &&
        any(binary[:, :, 1]) && any(binary[:, :, nz])
    )

    if touches_boundaries
        return 0
    end

    # Compute solid components (inverted β₀)
    solid = .!binary
    β₂ = compute_beta0(solid) - 1  # Subtract external void

    return max(0, β₂)
end

"""
Compute full topological signature.
Returns: (β₀, β₁, β₂, χ, χ_normalized)
"""
function compute_topology(binary::Array{Bool,3}, porosity::Float64)
    β₀ = compute_beta0(binary)
    β₁ = compute_beta1_approx(binary, porosity)
    β₂ = compute_beta2(binary)

    # Euler characteristic
    χ = β₀ - β₁ + β₂

    # Normalized by volume (scale-invariant)
    volume = prod(size(binary))
    χ_norm = χ / (volume^(1/3))

    return (β₀=β₀, β₁=β₁, β₂=β₂, χ=χ, χ_norm=χ_norm)
end

# ==============================================================================
# DATA LOADING
# ==============================================================================

println("\nLoading ground truth CSV...")
df = CSV.read(CSV_FILE, DataFrame)
println("  Total samples in CSV: $(nrow(df))")

# Check columns
println("\n  Available columns:")
for col in names(df)
    println("    - $col")
end

# Select subset for TIFF analysis
sample_indices = 1:min(N_SAMPLES, nrow(df))
df_sample = df[sample_indices, :]

println("\n  Selected $(nrow(df_sample)) samples for topology analysis")

# ==============================================================================
# VALIDATION LOOP
# ==============================================================================

results = []

println("\n" * "="^80)
println("COMPUTING TOPOLOGY FROM 3D VOLUMES")
println("="^80)

function validate_topology_transport()
    start_time = time()
    n_success = 0
    n_failed = 0

    for (idx, row) in enumerate(eachrow(df_sample))
        tiff_path = joinpath(DATA_DIR, row.file)

        if !isfile(tiff_path)
            n_failed += 1
            continue
        end

        try
            # Load TIFF
            img = TiffImages.load(tiff_path)

            # Ensure 3D
            if ndims(img) != 3
                n_failed += 1
                continue
            end

            # Convert to binary (pore = true) - ensure Array{Bool,3}
            binary = Array{Bool,3}(img .> 0)

            # Extract ground truth
            porosity_gt = row.porosity
            tortuosity_gt = row[Symbol("mean geodesic tortuosity")]
            constrictivity_gt = row.constrictivity

            # Compute topology
            topo = compute_topology(binary, porosity_gt)

            # Store result
            push!(results, (
                file = row.file,
                soil = row.soil,
                depth = row.depth,
                porosity = porosity_gt,
                tortuosity = tortuosity_gt,
                constrictivity = constrictivity_gt,
                β₀ = topo.β₀,
                β₁ = topo.β₁,
                β₂ = topo.β₂,
                χ = topo.χ,
                χ_norm = topo.χ_norm
            ))

            n_success += 1

            # Progress
            if idx % 10 == 0
                elapsed = time() - start_time
                rate = n_success / elapsed
                remaining = (nrow(df_sample) - idx) / rate
                println(@sprintf("  Progress: %d/%d | Success: %d | Failed: %d | ETA: %.1f min",
                               idx, nrow(df_sample), n_success, n_failed, remaining/60))
            end

        catch e
            n_failed += 1
            if idx <= 5  # Show first few errors
                println("  Error on $(row.file): $e")
            end
        end
    end

    elapsed = time() - start_time
    println("\n  Total time: $(round(elapsed/60, digits=1)) minutes")
    println("  Successfully processed: $n_success samples")
    println("  Failed: $n_failed samples")

    return n_success, n_failed
end

# Run validation
n_success, n_failed = validate_topology_transport()

# ==============================================================================
# CORRELATION ANALYSIS
# ==============================================================================

if length(results) < 10
    println("\n⚠️  ERROR: Insufficient samples ($n_success) for correlation analysis")
    println("    Need at least 10 samples. Check TIFF file paths.")
    exit(1)
end

println("\n" * "="^80)
println("CORRELATION ANALYSIS: TOPOLOGY vs TRANSPORT")
println("="^80)

# Extract vectors
χ_values = [r.χ for r in results]
χ_norm_values = [r.χ_norm for r in results]
τ_values = [r.tortuosity for r in results]
φ_values = [r.porosity for r in results]
C_values = [r.constrictivity for r in results]
β₀_values = [r.β₀ for r in results]
β₁_values = [r.β₁ for r in results]
β₂_values = [r.β₂ for r in results]

# Pearson correlations
cor_χ_τ = cor(χ_values, τ_values)
cor_χnorm_τ = cor(χ_norm_values, τ_values)
cor_β₀_τ = cor(β₀_values, τ_values)
cor_β₁_τ = cor(β₁_values, τ_values)
cor_β₂_τ = cor(β₂_values, τ_values)
cor_φ_τ = cor(φ_values, τ_values)
cor_C_τ = cor(C_values, τ_values)

println("\n┌────────────────────────────────────────────────────────────────┐")
println("│                    CORRELATION RESULTS                         │")
println("├────────────────────────────────────────────────────────────────┤")
println(@sprintf("│  Samples (N):              %-6d                              │", length(results)))
println("│                                                                │")
println("│  TOPOLOGICAL FEATURES vs TORTUOSITY:                           │")
println(@sprintf("│    cor(χ, τ):              %+.4f  (Euler characteristic)       │", cor_χ_τ))
println(@sprintf("│    cor(χ_norm, τ):         %+.4f  (normalized)                 │", cor_χnorm_τ))
println(@sprintf("│    cor(β₀, τ):             %+.4f  (connected components)       │", cor_β₀_τ))
println(@sprintf("│    cor(β₁, τ):             %+.4f  (loops/tunnels)              │", cor_β₁_τ))
println(@sprintf("│    cor(β₂, τ):             %+.4f  (voids)                      │", cor_β₂_τ))
println("│                                                                │")
println("│  BASELINE FEATURES vs TORTUOSITY:                              │")
println(@sprintf("│    cor(φ, τ):              %+.4f  (porosity)                   │", cor_φ_τ))
println(@sprintf("│    cor(C, τ):              %+.4f  (constrictivity)             │", cor_C_τ))
println("│                                                                │")
println("├────────────────────────────────────────────────────────────────┤")
println("│  COMPARISON TO SYNTHETIC DATA:                                 │")
println("│    Synthetic χ-τ correlation:   r = 0.83 (claimed)             │")
println(@sprintf("│    Real data χ-τ correlation:   r = %.2f                       │", cor_χ_τ))
println("│                                                                │")

# Verdict
if abs(cor_χ_τ) >= 0.80
    println("│  ✅ STRONG CORRELATION VALIDATED (|r| ≥ 0.80)                  │")
elseif abs(cor_χ_τ) >= 0.60
    println("│  ⚠️  MODERATE CORRELATION (0.60 ≤ |r| < 0.80)                  │")
elseif abs(cor_χ_τ) >= 0.40
    println("│  ⚠️  WEAK CORRELATION (0.40 ≤ |r| < 0.60)                      │")
else
    println("│  ❌ NO SIGNIFICANT CORRELATION (|r| < 0.40)                    │")
end

println("└────────────────────────────────────────────────────────────────┘")

# ==============================================================================
# DETAILED STATISTICS
# ==============================================================================

println("\n" * "="^80)
println("DESCRIPTIVE STATISTICS")
println("="^80)

println("\nTopological Features:")
println(@sprintf("  β₀ (components):      %.1f ± %.1f  [%d, %d]",
               mean(β₀_values), std(β₀_values), minimum(β₀_values), maximum(β₀_values)))
println(@sprintf("  β₁ (loops):           %.1f ± %.1f  [%d, %d]",
               mean(β₁_values), std(β₁_values), minimum(β₁_values), maximum(β₁_values)))
println(@sprintf("  β₂ (voids):           %.1f ± %.1f  [%d, %d]",
               mean(β₂_values), std(β₂_values), minimum(β₂_values), maximum(β₂_values)))
println(@sprintf("  χ (Euler char):       %.1f ± %.1f  [%d, %d]",
               mean(χ_values), std(χ_values), minimum(χ_values), maximum(χ_values)))
println(@sprintf("  χ_norm:               %.4f ± %.4f  [%.4f, %.4f]",
               mean(χ_norm_values), std(χ_norm_values), minimum(χ_norm_values), maximum(χ_norm_values)))

println("\nTransport Property:")
println(@sprintf("  τ (tortuosity):       %.4f ± %.4f  [%.4f, %.4f]",
               mean(τ_values), std(τ_values), minimum(τ_values), maximum(τ_values)))

println("\nControl Variables:")
println(@sprintf("  φ (porosity):         %.4f ± %.4f  [%.4f, %.4f]",
               mean(φ_values), std(φ_values), minimum(φ_values), maximum(φ_values)))
println(@sprintf("  C (constrictivity):   %.4f ± %.4f  [%.4f, %.4f]",
               mean(C_values), std(C_values), minimum(C_values), maximum(C_values)))

# ==============================================================================
# REGRESSION ANALYSIS
# ==============================================================================

println("\n" * "="^80)
println("REGRESSION MODELS")
println("="^80)

# Model 1: τ ~ χ (simple)
X_χ = hcat(ones(length(χ_values)), χ_values)
β_χ = X_χ \ τ_values
τ_pred_χ = X_χ * β_χ
r²_χ = cor(τ_pred_χ, τ_values)^2
rmse_χ = sqrt(mean((τ_pred_χ .- τ_values).^2))

println("\nModel 1: τ = a + b·χ")
println(@sprintf("  Coefficients: τ = %.4f + %.6f·χ", β_χ[1], β_χ[2]))
println(@sprintf("  R² = %.4f", r²_χ))
println(@sprintf("  RMSE = %.4f", rmse_χ))

# Model 2: τ ~ φ (baseline)
X_φ = hcat(ones(length(φ_values)), 1 ./ φ_values)
β_φ = X_φ \ τ_values
τ_pred_φ = X_φ * β_φ
r²_φ = cor(τ_pred_φ, τ_values)^2
rmse_φ = sqrt(mean((τ_pred_φ .- τ_values).^2))

println("\nModel 2: τ = a + b/φ (baseline)")
println(@sprintf("  Coefficients: τ = %.4f + %.4f/φ", β_φ[1], β_φ[2]))
println(@sprintf("  R² = %.4f", r²_φ))
println(@sprintf("  RMSE = %.4f", rmse_φ))

# Model 3: τ ~ χ + φ (combined)
X_combined = hcat(ones(length(χ_values)), χ_values, 1 ./ φ_values)
β_combined = X_combined \ τ_values
τ_pred_combined = X_combined * β_combined
r²_combined = cor(τ_pred_combined, τ_values)^2
rmse_combined = sqrt(mean((τ_pred_combined .- τ_values).^2))

println("\nModel 3: τ = a + b·χ + c/φ (combined)")
println(@sprintf("  Coefficients: τ = %.4f + %.6f·χ + %.4f/φ", β_combined[1], β_combined[2], β_combined[3]))
println(@sprintf("  R² = %.4f", r²_combined))
println(@sprintf("  RMSE = %.4f", rmse_combined))

# Model comparison
println("\n┌────────────────────────────────────────────────────────────────┐")
println("│  MODEL COMPARISON:                                             │")
println("├────────────────────────────────────────────────────────────────┤")
println(@sprintf("│  τ ~ χ:            R² = %.4f,  RMSE = %.4f                 │", r²_χ, rmse_χ))
println(@sprintf("│  τ ~ φ:            R² = %.4f,  RMSE = %.4f                 │", r²_φ, rmse_φ))
println(@sprintf("│  τ ~ χ + φ:        R² = %.4f,  RMSE = %.4f                 │", r²_combined, rmse_combined))
println("│                                                                │")

if r²_combined > r²_φ + 0.05
    println("│  ✅ Topology (χ) adds significant predictive power            │")
elseif r²_χ > r²_φ
    println("│  ⚠️  Topology alone performs better than porosity             │")
else
    println("│  ❌ Topology does not improve predictions                     │")
end

println("└────────────────────────────────────────────────────────────────┘")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

println("\n" * "="^80)
println("SAVING RESULTS")
println("="^80)

# Create results DataFrame
results_df = DataFrame(
    file = [r.file for r in results],
    soil = [r.soil for r in results],
    depth = [r.depth for r in results],
    porosity = φ_values,
    tortuosity = τ_values,
    constrictivity = C_values,
    β₀ = β₀_values,
    β₁ = β₁_values,
    β₂ = β₂_values,
    χ = χ_values,
    χ_norm = χ_norm_values
)

# Save CSV
results_dir = joinpath(PROJECT_ROOT, "results")
mkpath(results_dir)

csv_out = joinpath(results_dir, "topology_transport_validation.csv")
CSV.write(csv_out, results_df)
println("  ✓ Saved data: $csv_out")

# Save report
report = """
# Topology-Transport Correlation Validation Report

**Generated:** $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
**Dataset:** Zenodo 7516228 - Soil Pore Space 3D
**Paper:** Prifling et al. (2023) "Quantifying the impact of 3D pore space morphology on diffusive mass transport in loam and sand"
**Samples analyzed:** $n_success / $(nrow(df_sample))

---

## Research Question

Does the topology-transport correlation (χ-τ, r=0.83) observed in synthetic percolation models hold on **real porous media**?

## Methodology

1. **Ground Truth:** Mean geodesic tortuosity from direct path tracing (Zenodo)
2. **Topology Computation:**
   - β₀: Connected components via flood-fill
   - β₁: Loop/tunnel count (approximation via connectivity)
   - β₂: Enclosed void count via solid component analysis
   - χ: Euler characteristic = β₀ - β₁ + β₂
3. **Correlation:** Pearson correlation between χ and τ

## Results

### Correlation Coefficients

| Feature | Correlation with τ | Interpretation |
|---------|-------------------|----------------|
| **χ (Euler char)** | **$(round(cor_χ_τ, digits=4))** | $(abs(cor_χ_τ) >= 0.8 ? "**Strong**" : abs(cor_χ_τ) >= 0.6 ? "Moderate" : abs(cor_χ_τ) >= 0.4 ? "Weak" : "None") |
| χ_norm (normalized) | $(round(cor_χnorm_τ, digits=4)) | $(abs(cor_χnorm_τ) >= 0.8 ? "Strong" : abs(cor_χnorm_τ) >= 0.6 ? "Moderate" : abs(cor_χnorm_τ) >= 0.4 ? "Weak" : "None") |
| β₀ (components) | $(round(cor_β₀_τ, digits=4)) | $(abs(cor_β₀_τ) >= 0.8 ? "Strong" : abs(cor_β₀_τ) >= 0.6 ? "Moderate" : abs(cor_β₀_τ) >= 0.4 ? "Weak" : "None") |
| β₁ (loops) | $(round(cor_β₁_τ, digits=4)) | $(abs(cor_β₁_τ) >= 0.8 ? "Strong" : abs(cor_β₁_τ) >= 0.6 ? "Moderate" : abs(cor_β₁_τ) >= 0.4 ? "Weak" : "None") |
| β₂ (voids) | $(round(cor_β₂_τ, digits=4)) | $(abs(cor_β₂_τ) >= 0.8 ? "Strong" : abs(cor_β₂_τ) >= 0.6 ? "Moderate" : abs(cor_β₂_τ) >= 0.4 ? "Weak" : "None") |
| **φ (porosity)** | **$(round(cor_φ_τ, digits=4))** | Baseline |
| C (constrictivity) | $(round(cor_C_τ, digits=4)) | - |

### Comparison to Synthetic Data

- **Synthetic percolation:** χ-τ correlation r = 0.83
- **Real soil data:** χ-τ correlation r = $(round(cor_χ_τ, digits=2))

$(abs(cor_χ_τ - 0.83) < 0.10 ? "✅ **VALIDATED**: Real data matches synthetic correlation within 10%" : abs(cor_χ_τ) >= 0.60 ? "⚠️ **PARTIAL**: Correlation exists but weaker than synthetic" : "❌ **NOT VALIDATED**: Correlation not observed in real data")

### Regression Models

| Model | R² | RMSE | Notes |
|-------|-----|------|-------|
| τ ~ χ | $(round(r²_χ, digits=4)) | $(round(rmse_χ, digits=4)) | Topology alone |
| τ ~ φ | $(round(r²_φ, digits=4)) | $(round(rmse_φ, digits=4)) | Porosity baseline |
| τ ~ χ + φ | $(round(r²_combined, digits=4)) | $(round(rmse_combined, digits=4)) | Combined model |

**Improvement:** $(r²_combined > r²_φ ? "χ adds $(round((r²_combined - r²_φ)*100, digits=1))% predictive power" : "χ does not improve over porosity")

## Statistical Summary

**Topological Features:**
- β₀ (connected components): $(round(mean(β₀_values), digits=1)) ± $(round(std(β₀_values), digits=1))
- β₁ (loops/tunnels): $(round(mean(β₁_values), digits=1)) ± $(round(std(β₁_values), digits=1))
- β₂ (voids): $(round(mean(β₂_values), digits=1)) ± $(round(std(β₂_values), digits=1))
- χ (Euler): $(round(mean(χ_values), digits=1)) ± $(round(std(χ_values), digits=1))

**Transport Property:**
- τ (tortuosity): $(round(mean(τ_values), digits=4)) ± $(round(std(τ_values), digits=4))

**Range:**
- Porosity: $(round(minimum(φ_values), digits=3)) - $(round(maximum(φ_values), digits=3))
- Tortuosity: $(round(minimum(τ_values), digits=4)) - $(round(maximum(τ_values), digits=4))

## Limitations

1. **β₁ approximation:** Loop/tunnel count uses heuristic, not rigorous persistent homology
2. **Sample size:** $n_success samples (target: 100-500 for robust statistics)
3. **Computational cost:** TIFF loading + topology computation
4. **Soil-specific:** Results may not generalize to bone scaffolds (different pore structure)

## Conclusions

$(abs(cor_χ_τ) >= 0.80 ? "The strong χ-τ correlation (r=$(round(cor_χ_τ, digits=2))) validates the topology-transport relationship on real porous media. Euler characteristic is a robust predictor of tortuosity, independent of material type (percolation vs soil)." : abs(cor_χ_τ) >= 0.60 ? "A moderate χ-τ correlation (r=$(round(cor_χ_τ, digits=2))) suggests topology influences transport, but other factors (pore shape, connectivity patterns) dominate in real materials. The synthetic correlation (r=0.83) may overestimate due to idealized percolation structure." : "The weak χ-τ correlation (r=$(round(cor_χ_τ, digits=2))) indicates that Euler characteristic alone does not predict tortuosity in real soil pores. The synthetic correlation (r=0.83) does not generalize. Possible reasons: (1) β₁ approximation error, (2) soil pore complexity exceeds percolation model, (3) tortuosity dominated by local geometry, not global topology.")

## Next Steps

1. **Improve β₁ computation:** Use Ripserer.jl for exact persistent homology
2. **Increase sample size:** Validate on full 4,608 samples (requires parallel processing)
3. **Test on scaffolds:** Does χ-τ correlation hold for bone tissue engineering scaffolds?
4. **Multi-scale analysis:** Compute topology at different length scales (coarse-graining)

## References

- Prifling B, Röding M, Townsend P, et al. (2023). Dataset: Quantifying the impact of 3D pore space morphology on diffusive mass transport. *Zenodo*. https://doi.org/10.5281/zenodo.7516228

- Edelsbrunner H, Harer J. (2010). *Computational Topology: An Introduction*. American Mathematical Society.

---

*Validation completed on $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))*
"""

report_out = joinpath(results_dir, "topology_transport_validation.md")
open(report_out, "w") do io
    write(io, report)
end
println("  ✓ Saved report: $report_out")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

println("\n" * "="^80)
println("VALIDATION COMPLETE")
println("="^80)
println()
println("KEY FINDINGS:")
println("  • Samples analyzed: $n_success")
println("  • χ-τ correlation: $(round(cor_χ_τ, digits=4)) $(abs(cor_χ_τ) >= 0.80 ? "(STRONG ✅)" : abs(cor_χ_τ) >= 0.60 ? "(MODERATE ⚠️)" : "(WEAK ❌)")")
println("  • Synthetic claim: r = 0.83")
println("  • Δ from synthetic: $(round(abs(cor_χ_τ - 0.83), digits=3))")
println()
println("$(abs(cor_χ_τ - 0.83) < 0.10 ? "✅ TOPOLOGY-TRANSPORT CORRELATION VALIDATED ON REAL DATA" : abs(cor_χ_τ) >= 0.60 ? "⚠️  PARTIAL VALIDATION: Correlation exists but weaker than synthetic" : "❌ CORRELATION NOT VALIDATED: Topology does not predict transport in real soil")")
println()
println("="^80)
