#!/usr/bin/env julia
# Train and validate PLDLA 3D-print model

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using DegradationModels
using Printf

# Train focused model
model = train(PLDLA3DPrintModel, epochs=2000, verbose=true)

# Validate
results = validate(model, verbose=true)

# Practical examples
println("\n" * "="^60)
println("  PRACTICAL APPLICATIONS")
println("="^60)

println("\n  Half-life estimates:")
for (Mn0, TEC) in [(50.0, 0.0), (50.0, 1.0), (50.0, 2.0), (40.0, 0.0)]
    t_half = estimate_halflife(model, Mn0, TEC=TEC)
    @printf("    Mn0=%.0f kg/mol, TEC=%.0f%% → t½ ≈ %.1f days\n", Mn0, TEC, t_half)
end

println("\n  Design prediction (Mn0=50 kg/mol, pure PLDLA):")
for t in [7, 14, 30, 60, 90]
    Mn = predict(model, 50.0, Float64(t))
    pct = Mn / 50.0 * 100
    @printf("    t=%2d days → Mn=%.1f kg/mol (%.1f%% remaining)\n", t, Mn, pct)
end

println("\n  Effect of TEC plasticizer (Mn0=45 kg/mol, t=60 days):")
for TEC in [0.0, 0.5, 1.0, 1.5, 2.0]
    Mn = predict(model, 45.0, 60.0, TEC=TEC)
    pct = Mn / 45.0 * 100
    @printf("    TEC=%.1f%% → Mn=%.1f kg/mol (%.1f%% remaining)\n", TEC, Mn, pct)
end
