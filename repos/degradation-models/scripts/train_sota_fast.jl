#!/usr/bin/env julia
# Train SOTA Neural ODE (Fast version)

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using DegradationModels
using Printf

println("="^70)
println("  PLDLA NEURAL ODE (Fast) - SOTA Physics-Informed Model")
println("="^70)

# Train
model = train(PLDLANeuralODEFast, epochs=2000, verbose=true)

# Validate
results = validate(model, verbose=true)

# Compare conditions
compare_conditions(model, 50.0)

# Predictions with uncertainty
println("\n" * "="^70)
println("  PREDICTIONS WITH UNCERTAINTY")
println("="^70)

println("\n  In vitro (Mn0=50 kg/mol):")
for t in [30.0, 60.0, 90.0]
    r = predict(model, 50.0, t, with_uncertainty=true)
    @printf("    t=%2.0fd: Mn=%.1f ± %.1f (95%% CI: %.1f-%.1f)\n",
            t, r.Mn, 1.96*r.σ, r.lower, r.upper)
end

println("\n  In vivo bone (Mn0=50 kg/mol):")
for t in [30.0, 60.0, 90.0]
    r = predict(model, 50.0, t, condition=:in_vivo, region=:bone, with_uncertainty=true)
    @printf("    t=%2.0fd: Mn=%.1f ± %.1f (95%% CI: %.1f-%.1f)\n",
            t, r.Mn, 1.96*r.σ, r.lower, r.upper)
end

# Final summary
println("\n" * "="^70)
println("  MODEL SUMMARY")
println("="^70)
@printf("\n  Global Accuracy: %.1f%%\n", results["overall_accuracy"])
println("\n  Key features:")
println("    ✓ Wang-Han physics (interpretable parameters)")
println("    ✓ Neural correction for unmodeled dynamics")
println("    ✓ Temperature via Arrhenius (Ea from literature)")
println("    ✓ In vivo enzymatic acceleration (1.35x)")
println("    ✓ Uncertainty quantification")
println("="^70)
