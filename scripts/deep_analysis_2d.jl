"""
Deep analysis of 2D fractal dimension vs porosity relationship
Testing if D_2D → some function of φ at high porosity
"""

using Statistics
using Printf

# Data extracted from results (90%+ porosity points)
high_porosity_data = [
    # (porosity, D_2D)
    (0.8955, 1.2558),
    (0.9025, 1.2398),
    (0.9013, 1.2263),
    (0.9004, 1.2478),
    (0.9069, 1.2216),
    (0.9086, 1.2185),  # minimum D
]

# All data from results file
all_data = [
    (0.1750, 1.9060), (0.4632, 1.7730), (0.5909, 1.6622), (0.6489, 1.6106),
    (0.7270, 1.5303), (0.8158, 1.4124), (0.8955, 1.2558),
    (0.1558, 1.9109), (0.4957, 1.7581), (0.6254, 1.6274), (0.6797, 1.5763),
    (0.7532, 1.4918), (0.8306, 1.3696), (0.9025, 1.2398),
    (0.1522, 1.9114), (0.4730, 1.7726), (0.6084, 1.6483), (0.6675, 1.5957),
    (0.7460, 1.4975), (0.8278, 1.3744), (0.9013, 1.2263),
]

φ = (1 + sqrt(5)) / 2
INV_φ = 1 / φ

println("="^70)
println("DEEP ANALYSIS: D_2D vs POROSITY RELATIONSHIP")
println("="^70)
println()

# Extract arrays
porosities = [d[1] for d in all_data]
D_values = [d[2] for d in all_data]

# Linear regression: D = a * porosity + b
n = length(porosities)
x_mean = mean(porosities)
y_mean = mean(D_values)

slope = sum((porosities .- x_mean) .* (D_values .- y_mean)) / sum((porosities .- x_mean).^2)
intercept = y_mean - slope * x_mean

println("LINEAR MODEL: D_2D = slope × porosity + intercept")
@printf("  slope = %.6f\n", slope)
@printf("  intercept = %.6f\n", intercept)
println()

# R² calculation
y_pred = slope .* porosities .+ intercept
ss_res = sum((D_values .- y_pred).^2)
ss_tot = sum((D_values .- y_mean).^2)
r_squared = 1 - ss_res / ss_tot
@printf("  R² = %.6f\n", r_squared)
println()

# Extrapolate to 100% porosity
D_at_100pct = slope * 1.0 + intercept
@printf("EXTRAPOLATION TO 100%% POROSITY:\n")
@printf("  D_2D at p=1.0: %.6f\n", D_at_100pct)
println()

# Compare to golden ratio expressions
println("COMPARISON TO GOLDEN RATIO EXPRESSIONS:")
println("-"^70)

expressions = [
    ("1/φ", INV_φ, 0.618034),
    ("φ - 1 (= 1/φ)", φ - 1, 0.618034),
    ("2 - φ", 2 - φ, 0.381966),
    ("φ/2", φ/2, 0.809017),
    ("1/φ²", 1/φ^2, 0.381966),
    ("√φ - 1", sqrt(φ) - 1, 0.272020),
    ("φ² - 2", φ^2 - 2, 0.618034),
    ("3 - φ²", 3 - φ^2, 0.381966),
    ("(φ + 1)/φ²", (φ + 1)/φ^2, 1.0),
    ("2/φ", 2/φ, 1.236068),
    ("φ/√2", φ/sqrt(2), 1.144123),
    ("√(φ)", sqrt(φ), 1.272020),
    ("φ - 0.5", φ - 0.5, 1.118034),
    ("2φ - 2", 2*φ - 2, 1.236068),
    ("φ²/2", φ^2/2, 1.309017),
    ("1 + 1/φ²", 1 + 1/φ^2, 1.381966),
    ("3 - φ", 3 - φ, 1.381966),  # INTERESTING!
    ("φ² - 1", φ^2 - 1, 1.618034),
    ("2/φ²", 2/φ^2, 0.763932),
]

# Sort by closeness to D_at_100pct
sorted_expr = sort(expressions, by=x -> abs(x[3] - D_at_100pct))

println("\nBest matches for D_2D at 100% porosity (D = $(round(D_at_100pct, digits=4))):")
for (name, formula, value) in sorted_expr[1:10]
    error_pct = abs(value - D_at_100pct) / D_at_100pct * 100
    @printf("  %-15s = %.6f  (error: %6.2f%%)\n", name, value, error_pct)
end
println()

# THE BIG INSIGHT
println("="^70)
println("KEY INSIGHT")
println("="^70)
println()

# D_2D linear model
@printf("2D Linear Model:  D_2D = %.4f × p + %.4f\n", slope, intercept)
@printf("3D Linear Model:  D_3D = -1.25 × p + 2.98\n")  # From previous analysis
println()

# At what porosity does D_2D = 1/φ?
p_at_inv_phi = (INV_φ - intercept) / slope
@printf("D_2D = 1/φ = 0.618 at porosity = %.2f%%\n", p_at_inv_phi * 100)

if p_at_inv_phi > 1.0
    println("  ⚠ This is BEYOND 100%% porosity!")
    println("  → D_2D = 1/φ is NOT physically achievable")
    println()
    println("  CONCLUSION: D_2D NEVER reaches 1/φ in real scaffolds")
else
    println("  ✓ Achievable porosity")
end
println()

# What IS D_2D at same porosity where D_3D = φ?
# D_3D = φ at p = 95.76% (from previous analysis)
p_critical = 0.9576
D_2D_at_critical = slope * p_critical + intercept

@printf("At p = 95.76%% (where D_3D = φ):\n")
@printf("  D_2D = %.6f\n", D_2D_at_critical)
@printf("  φ - D_2D = %.6f\n", φ - D_2D_at_critical)
@printf("  D_3D - D_2D = %.6f (expected: 1.0 for isotropic fractal)\n", φ - D_2D_at_critical)
println()

# NEW HYPOTHESIS
println("="^70)
println("NEW HYPOTHESIS")
println("="^70)
println()

# Check if D_3D - D_2D is constant
println("If salt-leached scaffolds are anisotropic fractals:")
println()
println("  D_3D - D_2D ≈ 0.4 - 0.5 (NOT 1.0)")
println()
println("This means the fractal 'loses' less dimension in projection")
println("because pores are elongated (anisotropic).")
println()

# Alternative: Check D_2D = f(D_3D)
println("ALTERNATIVE: D_2D = D_3D - Δ")
println()
println("  If D_3D = φ = 1.618 and D_2D ≈ 1.1-1.2 at same porosity:")
println("  Δ = D_3D - D_2D ≈ 0.4 - 0.5")
println()
println("  This is consistent with anisotropic pore structure!")
println()

# Slope comparison
println("="^70)
println("SLOPE COMPARISON")
println("="^70)
println()
@printf("3D slope: %.4f (D decreases with porosity)\n", -1.25)
@printf("2D slope: %.4f (D decreases with porosity)\n", slope)
println()
@printf("Ratio of slopes: 2D/3D = %.4f\n", abs(slope / -1.25))
println()

if abs(slope) < abs(-1.25)
    println("2D slope is SHALLOWER than 3D slope")
    println("→ D_2D decreases slower with porosity")
    println("→ 2D structure maintains more complexity at high porosity")
end
println()

# Final model
println("="^70)
println("PROPOSED UNIFIED MODEL")
println("="^70)
println()
println("3D: D_3D = -1.25p + 2.98")
println("    D_3D → φ at p = 95.76%")
println()
@printf("2D: D_2D = %.4fp + %.4f\n", slope, intercept)
@printf("    D_2D → %.4f at p = 100%%\n", D_at_100pct)
println()
println("Relationship:")
@printf("    D_3D - D_2D ≈ %.2f at high porosity\n", φ - D_2D_at_critical)
println("    (NOT 1.0 as expected for isotropic fractal)")
println()

# Check 3-φ hypothesis
println("="^70)
println("TESTING: D_2D → 3 - φ at high porosity")
println("="^70)
println()
target = 3 - φ
@printf("3 - φ = %.6f\n", target)
@printf("D_2D at 90%% porosity ≈ %.6f\n", mean([d[2] for d in high_porosity_data]))
error_pct = abs(mean([d[2] for d in high_porosity_data]) - target) / target * 100
@printf("Error: %.2f%%\n", error_pct)
println()

if error_pct < 10
    println("✓ HYPOTHESIS SUPPORTED: D_2D → 3 - φ = 1.382 at high porosity")
    println()
    println("This is BEAUTIFUL because:")
    println("  3D: D → φ")
    println("  2D: D → 3 - φ")
    println("  Sum: φ + (3-φ) = 3 (total dimensions!)")
else
    println("✗ Hypothesis not strongly supported")
    println("  D_2D at high porosity is closer to 1.2, not 1.38")
end

println()
println("="^70)
