#!/usr/bin/env julia
# Quick training of Neural ODE

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using DegradationModels
using Printf

println("="^70)
println("  PLDLA Neural ODE - Quick Training")
println("="^70)

# Train with fewer epochs for speed
model = train(PLDLANeuralODE, epochs=1500, verbose=true)

# Validate
results = validate(model, verbose=true)

# Half-life
println("\nHalf-life estimates (Mn0=50 kg/mol):")
for (cond, region, label) in [(:in_vitro, nothing, "In vitro 37°C"),
                               (:in_vivo, :bone, "In vivo bone"),
                               (:in_vivo, :inflammation, "In vivo inflammation")]
    t_half = estimate_halflife(model, 50.0, condition=cond, region=region)
    @printf("  %-25s: t½ = %.1f days\n", label, t_half)
end

# Uncertainty example
println("\nPrediction with uncertainty (in vitro, t=60d):")
r = predict(model, 50.0, 60.0, with_uncertainty=true)
@printf("  Mn = %.1f ± %.1f kg/mol (95%% CI: %.1f - %.1f)\n",
        r.Mn, 1.96*r.σ, r.lower, r.upper)
