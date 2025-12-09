#!/usr/bin/env julia
# Generate publication-quality figures for the Dimensional Duality paper

using Pkg
Pkg.activate(".")

using Plots
using Printf

# Set high-quality plot defaults
gr(size=(800, 600), dpi=300)
default(fontfamily="Computer Modern", titlefontsize=14, guidefontsize=12,
        tickfontsize=10, legendfontsize=10, linewidth=2)

# Golden ratio constant
const φ = (1 + √5) / 2

# Create output directory
mkpath("paper/figures")

println("=" ^ 60)
println("GENERATING PUBLICATION FIGURES")
println("=" ^ 60)

# ============================================================================
# FIGURE 1: Fractal Dimension vs Porosity (D vs p)
# ============================================================================
println("\n📊 Figure 1: D(p) power-law model...")

# Experimental data points
data_points = [
    (0.05, 2.854, "Shale (ACS Omega)"),
    (0.35, 2.56, "Soil pore"),
    (0.69, 2.10, "High-porosity scaffold"),
    (0.96, 1.625, "Salt-leached scaffold")
]

porosities = [d[1] for d in data_points]
dimensions = [d[2] for d in data_points]
labels = [d[3] for d in data_points]

# Model: D(p) = φ + (3-φ)(1-p)^α
α = 1.0  # Linear approximation works well
D_model(p) = φ + (3 - φ) * (1 - p)^α

p_range = 0:0.01:1
D_range = D_model.(p_range)

fig1 = plot(p_range, D_range,
    label="Model: D(p) = φ + (3-φ)(1-p)",
    xlabel="Porosity p",
    ylabel="Fractal Dimension D",
    title="Porosity-Dependent Fractal Dimension",
    color=:blue,
    linewidth=3,
    legend=:topright
)

# Add horizontal lines for key values
hline!([φ], label="D = φ ≈ 1.618", linestyle=:dash, color=:gold, linewidth=2)
hline!([3φ - 2], label="D = 3φ-2 ≈ 2.854", linestyle=:dash, color=:red, linewidth=2)
hline!([2/φ], label="D = 2/φ ≈ 1.236", linestyle=:dot, color=:green, linewidth=2)

# Add experimental points
scatter!(porosities, dimensions,
    label="Experimental data",
    markersize=10,
    color=:red,
    markerstrokewidth=2
)

# Annotate points
for (p, d, lab) in data_points
    annotate!(p, d + 0.08, text(lab, 8, :center))
end

savefig(fig1, "paper/figures/fig1_D_vs_porosity.png")
savefig(fig1, "paper/figures/fig1_D_vs_porosity.pdf")
println("   ✓ Saved: paper/figures/fig1_D_vs_porosity.{png,pdf}")

# ============================================================================
# FIGURE 2: Dimensional Duality Schematic (3D → 2D projection)
# ============================================================================
println("\n📊 Figure 2: Dimensional duality schematic...")

fig2 = plot(
    xlim=(0, 10), ylim=(0, 6),
    aspect_ratio=:equal,
    axis=false,
    grid=false,
    legend=false,
    title="Dimensional Duality: 3D → 2D Projection"
)

# 3D box (represented as isometric cube)
x3d = [1, 3, 4, 2, 1, 1, 3, 3, 4, 4, 2, 2, 1]
y3d = [1, 1, 2.5, 2.5, 1, 4, 4, 1, 2.5, 5.5, 5.5, 2.5, 4]
plot!(fig2, x3d, y3d, color=:blue, linewidth=2, fill=false)

# 2D projection (rectangle)
plot!(fig2, [6, 9, 9, 6, 6], [1.5, 1.5, 4.5, 4.5, 1.5],
    color=:orange, linewidth=3, fill=(:orange, 0.3))

# Arrow from 3D to 2D
plot!(fig2, [4.2, 5.8], [3, 3], arrow=true, color=:black, linewidth=2)

# Labels
annotate!(fig2, 2.5, 0.3, text("3D: D₃D = φ", 12, :center))
annotate!(fig2, 7.5, 0.7, text("2D: D₂D = 2/φ", 12, :center))
annotate!(fig2, 5, 3.5, text("Projection", 10, :center))

# Key relations box
annotate!(fig2, 5, 5.5, text("D₃D × D₂D = 2", 11, :center, :blue))

savefig(fig2, "paper/figures/fig2_dimensional_duality.png")
savefig(fig2, "paper/figures/fig2_dimensional_duality.pdf")
println("   ✓ Saved: paper/figures/fig2_dimensional_duality.{png,pdf}")

# ============================================================================
# FIGURE 3: Walk Dimension Validation
# ============================================================================
println("\n📊 Figure 3: Walk dimension validation...")

# Data: dimension d, predicted d_w, measured d_w
walk_data = [
    (2, 2 + 1/φ^2, 2.37),  # 2D estimated
    (3, 3 + 1/φ^2, 3.31),  # 3D measured
]

dims = [d[1] for d in walk_data]
predicted = [d[2] for d in walk_data]
measured = [d[3] for d in walk_data]

fig3 = plot(
    xlabel="Euclidean Dimension d",
    ylabel="Walk Dimension d_w",
    title="Walk Dimension: Theory vs Measurement",
    legend=:topleft,
    xlim=(1.5, 3.5),
    ylim=(2, 4)
)

# Theoretical line: d_w = d + 1/φ²
d_range = 1.5:0.1:3.5
dw_theory = d_range .+ 1/φ^2

plot!(fig3, d_range, dw_theory,
    label="Theory: d_w = d + 1/φ²",
    color=:blue, linewidth=3)

# Identity line for reference
plot!(fig3, d_range, d_range,
    label="d_w = d (Brownian)",
    color=:gray, linestyle=:dash, linewidth=2)

# Measured points
scatter!(fig3, dims, measured,
    label="Measured (percolation)",
    markersize=12, color=:red, markerstrokewidth=2)

# Predicted points
scatter!(fig3, dims, predicted,
    label="Predicted",
    markersize=10, color=:blue, marker=:diamond, markerstrokewidth=2)

# Annotate error
annotate!(fig3, 3.1, 3.35, text("2.2% error", 10, :left, :red))

savefig(fig3, "paper/figures/fig3_walk_dimension.png")
savefig(fig3, "paper/figures/fig3_walk_dimension.pdf")
println("   ✓ Saved: paper/figures/fig3_walk_dimension.{png,pdf}")

# ============================================================================
# FIGURE 4: Golden Ratio Relations Summary
# ============================================================================
println("\n📊 Figure 4: Golden ratio relations summary...")

relations = [
    ("D₃D × D₂D", 2.0, φ * (2/φ), "Conservation"),
    ("D₃D + D₂D", 3φ - 2, φ + 2/φ, "Totality"),
    ("D₃D - D₂D", 1/φ^2, φ - 2/φ, "Complementarity"),
    ("D₃D / D₂D", φ^2/2, φ / (2/φ), "Proportion")
]

names = [r[1] for r in relations]
exact_vals = [r[2] for r in relations]
computed_vals = [r[3] for r in relations]
labels_r = [r[4] for r in relations]

fig4 = bar(names, exact_vals,
    label="Theoretical",
    xlabel="Relation",
    ylabel="Value",
    title="Dimensional Duality Relations",
    color=:blue,
    alpha=0.7,
    bar_width=0.4
)

bar!(names, computed_vals,
    label="Computed (D₃D=φ, D₂D=2/φ)",
    color=:orange,
    alpha=0.7,
    bar_width=0.3
)

# Add value annotations
for (i, (name, exact, computed, lab)) in enumerate(relations)
    annotate!(fig4, i-0.5, exact + 0.1, text(@sprintf("%.3f", exact), 9, :center))
end

savefig(fig4, "paper/figures/fig4_relations_summary.png")
savefig(fig4, "paper/figures/fig4_relations_summary.pdf")
println("   ✓ Saved: paper/figures/fig4_relations_summary.{png,pdf}")

# ============================================================================
# Summary
# ============================================================================
println("\n" * "=" ^ 60)
println("✅ ALL FIGURES GENERATED SUCCESSFULLY")
println("=" ^ 60)
println("\nFigures saved in: paper/figures/")
println("  • fig1_D_vs_porosity.{png,pdf}")
println("  • fig2_dimensional_duality.{png,pdf}")
println("  • fig3_walk_dimension.{png,pdf}")
println("  • fig4_relations_summary.{png,pdf}")
println("\nReady for manuscript inclusion!")
