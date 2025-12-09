#!/usr/bin/env julia
"""
Generate publication-quality figures for the tortuosity paper.
"""

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Statistics
using Printf
using Random
using LinearAlgebra

# Check if CairoMakie is available, otherwise use text-based output
USE_MAKIE = false
try
    @eval using CairoMakie
    global USE_MAKIE = true
catch
    println("CairoMakie not available, using text-based output")
end

Random.seed!(42)

println("="^70)
println("GENERATING PUBLICATION FIGURES")
println("="^70)

# Load data
csv_path = expanduser("~/workspace/darwin-scaffold-studio/data/soil_pore_space/characteristics.csv")
df = CSV.read(csv_path, DataFrame)

φ = Float64.(df.porosity)
τ = Float64.(df[!, "mean geodesic tortuosity"])
ψ = Float64.(df.constrictivity)
soil = String.(df.soil)

n = length(τ)
println("Loaded $n samples")

# Output directory
fig_dir = expanduser("~/workspace/darwin-scaffold-studio/paper/figures")
mkpath(fig_dir)

# =============================================================================
# FIGURE 1: τ vs φ with all model fits
# =============================================================================

println("\n" * "-"^70)
println("FIGURE 1: Tortuosity vs Porosity")
println("-"^70)

# Calculate model predictions
φ_range = range(0.15, 0.55, length=100)

# Classical models
τ_archie = φ_range .^ (-0.5)
τ_maxwell = 3 ./ (2 .+ φ_range)
τ_weiss = 1 .- 0.5 .* log.(φ_range)

# Fitted models
τ_fitted_power = 0.962 .* φ_range .^ (-0.127)
τ_fitted_linear = 0.977 .+ 0.043 ./ φ_range

# Data statistics
println("\nData range:")
println(@sprintf("  φ: %.3f - %.3f", minimum(φ), maximum(φ)))
println(@sprintf("  τ: %.4f - %.4f", minimum(τ), maximum(τ)))

println("\nModel predictions at φ = 0.30:")
println(@sprintf("  Ground truth mean: %.4f", mean(τ[abs.(φ .- 0.30) .< 0.02])))
println(@sprintf("  Archie (m=0.5):    %.4f", 0.30^(-0.5)))
println(@sprintf("  Maxwell:           %.4f", 3/(2+0.30)))
println(@sprintf("  Weissberg:         %.4f", 1 - 0.5*log(0.30)))
println(@sprintf("  This work (power): %.4f", 0.962 * 0.30^(-0.127)))
println(@sprintf("  This work (linear):%.4f", 0.977 + 0.043/0.30))

if USE_MAKIE
    fig1 = Figure(size=(800, 600))
    ax1 = Axis(fig1[1,1],
        xlabel = "Porosity φ",
        ylabel = "Tortuosity τ",
        title = "Tortuosity-Porosity Relationships")

    # Data points (subsample for clarity)
    idx_loam = soil .== "loam"
    idx_sand = soil .== "sand"

    scatter!(ax1, φ[idx_loam][1:10:end], τ[idx_loam][1:10:end],
             color=:blue, alpha=0.3, markersize=4, label="Loam")
    scatter!(ax1, φ[idx_sand][1:10:end], τ[idx_sand][1:10:end],
             color=:red, alpha=0.3, markersize=4, label="Sand")

    # Model lines
    lines!(ax1, φ_range, τ_archie, color=:gray, linestyle=:dash,
           linewidth=2, label="Archie (m=0.5)")
    lines!(ax1, φ_range, τ_maxwell, color=:orange, linestyle=:dash,
           linewidth=2, label="Maxwell")
    lines!(ax1, φ_range, τ_fitted_linear, color=:black,
           linewidth=3, label="This work")

    axislegend(ax1, position=:rt)
    ylims!(ax1, 1.0, 2.0)

    save(joinpath(fig_dir, "fig1_tau_vs_phi.pdf"), fig1)
    save(joinpath(fig_dir, "fig1_tau_vs_phi.png"), fig1, px_per_unit=3)
    println("\nSaved: fig1_tau_vs_phi.pdf/png")
else
    println("\n[TEXT OUTPUT - Figure 1 data]")
    println("φ\tτ_data\tτ_archie\tτ_linear")
    for p in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        mask = abs.(φ .- p) .< 0.02
        τ_mean = sum(mask) > 0 ? mean(τ[mask]) : NaN
        println(@sprintf("%.2f\t%.4f\t%.4f\t%.4f", p, τ_mean, p^(-0.5), 0.977 + 0.043/p))
    end
end

# =============================================================================
# FIGURE 2: Archie exponent sensitivity
# =============================================================================

println("\n" * "-"^70)
println("FIGURE 2: Archie Exponent Sensitivity")
println("-"^70)

m_range = range(0.05, 0.80, length=50)
mre_vs_m = Float64[]

for m in m_range
    τ_pred = φ .^ (-m)
    mre = mean(abs.(τ_pred .- τ) ./ τ) * 100
    push!(mre_vs_m, mre)
end

# Find optimal
m_opt_idx = argmin(mre_vs_m)
m_opt = m_range[m_opt_idx]
mre_opt = mre_vs_m[m_opt_idx]

println(@sprintf("\nOptimal exponent: m = %.3f (MRE = %.2f%%)", m_opt, mre_opt))
println(@sprintf("Bruggeman (m=0.5): MRE = %.2f%%", mre_vs_m[findfirst(m_range .>= 0.5)]))

if USE_MAKIE
    fig2 = Figure(size=(600, 400))
    ax2 = Axis(fig2[1,1],
        xlabel = "Archie exponent m",
        ylabel = "Mean Relative Error (%)",
        title = "Sensitivity to Archie Exponent")

    lines!(ax2, collect(m_range), mre_vs_m, color=:blue, linewidth=2)
    vlines!(ax2, [m_opt], color=:green, linestyle=:solid, linewidth=2,
            label=@sprintf("Optimal m = %.3f", m_opt))
    vlines!(ax2, [0.5], color=:red, linestyle=:dash, linewidth=2,
            label="Bruggeman m = 0.5")

    axislegend(ax2, position=:rt)
    ylims!(ax2, 0, 80)

    save(joinpath(fig_dir, "fig2_archie_sensitivity.pdf"), fig2)
    save(joinpath(fig_dir, "fig2_archie_sensitivity.png"), fig2, px_per_unit=3)
    println("\nSaved: fig2_archie_sensitivity.pdf/png")
else
    println("\n[TEXT OUTPUT - Figure 2 data]")
    println("m\tMRE(%)")
    for (m, mre) in zip(m_range[1:5:end], mre_vs_m[1:5:end])
        marker = m ≈ m_opt ? " *OPT*" : (m ≈ 0.5 ? " *BRUGG*" : "")
        println(@sprintf("%.3f\t%.2f%s", m, mre, marker))
    end
end

# =============================================================================
# FIGURE 3: Residuals comparison
# =============================================================================

println("\n" * "-"^70)
println("FIGURE 3: Residual Analysis")
println("-"^70)

# Calculate residuals
τ_pred_archie = φ .^ (-0.5)
τ_pred_linear = 0.977 .+ 0.043 ./ φ

resid_archie = (τ .- τ_pred_archie) ./ τ .* 100
resid_linear = (τ .- τ_pred_linear) ./ τ .* 100

println("\nResidual statistics:")
println(@sprintf("  Archie:  mean = %+.2f%%, std = %.2f%%", mean(resid_archie), std(resid_archie)))
println(@sprintf("  Linear:  mean = %+.2f%%, std = %.2f%%", mean(resid_linear), std(resid_linear)))

if USE_MAKIE
    fig3 = Figure(size=(800, 400))

    ax3a = Axis(fig3[1,1],
        xlabel = "Porosity φ",
        ylabel = "Relative Error (%)",
        title = "Archie (m=0.5)")

    ax3b = Axis(fig3[1,2],
        xlabel = "Porosity φ",
        ylabel = "Relative Error (%)",
        title = "This work (linear)")

    scatter!(ax3a, φ[1:10:end], resid_archie[1:10:end],
             color=:red, alpha=0.5, markersize=3)
    hlines!(ax3a, [0], color=:black, linestyle=:dash)
    ylims!(ax3a, -70, 10)

    scatter!(ax3b, φ[1:10:end], resid_linear[1:10:end],
             color=:blue, alpha=0.5, markersize=3)
    hlines!(ax3b, [0], color=:black, linestyle=:dash)
    ylims!(ax3b, -5, 5)

    save(joinpath(fig_dir, "fig3_residuals.pdf"), fig3)
    save(joinpath(fig_dir, "fig3_residuals.png"), fig3, px_per_unit=3)
    println("\nSaved: fig3_residuals.pdf/png")
end

# =============================================================================
# FIGURE 4: Material-specific fits
# =============================================================================

println("\n" * "-"^70)
println("FIGURE 4: Material-Specific Relationships")
println("-"^70)

for s in ["loam", "sand"]
    mask = soil .== s
    X = hcat(ones(sum(mask)), 1 ./ φ[mask])
    β = X \ τ[mask]
    τ_pred = X * β
    mre = mean(abs.(τ_pred .- τ[mask]) ./ τ[mask]) * 100
    println(@sprintf("  %s: τ = %.4f + %.4f/φ (MRE = %.2f%%)", s, β[1], β[2], mre))
end

# =============================================================================
# TABLE: Complete model comparison
# =============================================================================

println("\n" * "="^70)
println("TABLE: Complete Model Comparison")
println("="^70)

models = [
    ("Archie (m=0.5)", φ .^ (-0.5)),
    ("Archie (m=0.3)", φ .^ (-0.3)),
    ("Maxwell", 3 ./ (2 .+ φ)),
    ("Weissberg", 1 .- 0.5 .* log.(φ)),
    ("Fitted power", 0.962 .* φ .^ (-0.127)),
    ("Fitted linear", 0.977 .+ 0.043 ./ φ),
]

println("\n┌─────────────────────┬────────┬────────┬──────────┐")
println("│ Model               │ MRE(%) │ <5%    │ <10%     │")
println("├─────────────────────┼────────┼────────┼──────────┤")

for (name, τ_pred) in models
    mre = mean(abs.(τ_pred .- τ) ./ τ) * 100
    within5 = sum(abs.(τ_pred .- τ) ./ τ .< 0.05) / n * 100
    within10 = sum(abs.(τ_pred .- τ) ./ τ .< 0.10) / n * 100
    println(@sprintf("│ %-19s │ %6.2f │ %5.1f%% │ %6.1f%% │", name, mre, within5, within10))
end

println("└─────────────────────┴────────┴────────┴──────────┘")

# =============================================================================
# Save numerical results
# =============================================================================

results_file = joinpath(fig_dir, "numerical_results.txt")
open(results_file, "w") do f
    println(f, "TORTUOSITY PAPER - NUMERICAL RESULTS")
    println(f, "="^50)
    println(f, "\nDataset: Zenodo 7516228")
    println(f, "Samples: $n")
    println(f, "\nGround truth statistics:")
    println(f, @sprintf("  τ: %.4f ± %.4f (range: %.4f - %.4f)", mean(τ), std(τ), minimum(τ), maximum(τ)))
    println(f, @sprintf("  φ: %.3f ± %.3f (range: %.3f - %.3f)", mean(φ), std(φ), minimum(φ), maximum(φ)))
    println(f, "\nKey findings:")
    println(f, @sprintf("  Optimal Archie exponent: m = %.3f", m_opt))
    println(f, @sprintf("  Linear model: τ = 0.977 + 0.043/φ"))
    println(f, @sprintf("  MRE (Archie m=0.5): %.1f%%", mre_vs_m[findfirst(m_range .>= 0.5)]))
    println(f, @sprintf("  MRE (This work): %.2f%%", mre_opt))
end

println("\nSaved: numerical_results.txt")

println("\n" * "="^70)
println("FIGURE GENERATION COMPLETE")
println("="^70)
println("\nFigures saved to: $fig_dir")
