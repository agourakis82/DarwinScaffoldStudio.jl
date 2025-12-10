#!/usr/bin/env julia
# Train and validate PLDLA 3D-print model V2

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using DegradationModels
using Printf

println("="^70)
println("  PLDLA 70:30 3D-Printed Scaffold Degradation Model")
println("  Data: Kaique Hergesel PhD Thesis")
println("="^70)

# Train model
model = train(PLDLA3DPrintModelV2, epochs=3000, verbose=true)

# Validate
results = validate(model, verbose=true)

# Practical applications
println("\n" * "="^60)
println("  PRACTICAL APPLICATIONS FOR SCAFFOLD DESIGN")
println("="^60)

println("\n  Half-life (time to 50% Mn loss):")
for (Mn0, TEC, label) in [
    (50.0, 0.0, "Pure PLDLA"),
    (45.0, 1.0, "PLDLA + 1% TEC"),
    (35.0, 2.0, "PLDLA + 2% TEC"),
]
    t_half = estimate_halflife(model, Mn0, TEC=TEC)
    @printf("    %-20s: t½ = %.1f days\n", label, t_half)
end

println("\n  Degradation timeline (Mn0=50 kg/mol, pure PLDLA):")
@printf("    %8s  %12s  %12s\n", "Time", "Mn (kg/mol)", "% Remaining")
for t in [0, 7, 14, 21, 30, 45, 60, 75, 90]
    Mn = predict(model, 50.0, Float64(t), TEC=0.0)
    pct = Mn / 50.0 * 100
    @printf("    %5d d   %10.1f     %8.1f%%\n", t, Mn, pct)
end

println("\n  Effect of TEC plasticizer at t=60 days:")
@printf("    %10s  %12s  %12s  %12s\n", "TEC %", "Mn0", "Mn(60d)", "% Remaining")
for (Mn0, TEC, label) in [
    (51.3, 0.0, "0%"),
    (45.0, 1.0, "1%"),
    (32.7, 2.0, "2%"),
]
    Mn = predict(model, Mn0, 60.0, TEC=TEC)
    pct = Mn / Mn0 * 100
    @printf("    %10s  %10.1f     %10.1f     %8.1f%%\n", label, Mn0, Mn, pct)
end

println("\n" * "="^60)
println("  Model ready for scaffold optimization!")
println("="^60)
