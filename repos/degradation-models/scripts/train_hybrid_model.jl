#!/usr/bin/env julia
# Train and demonstrate PLDLA Hybrid Model

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using DegradationModels
using Printf

println("="^70)
println("  PLDLA HYBRID MODEL")
println("  In vitro + In vivo + Temperatura por região do corpo")
println("="^70)

# Train
model = train(PLDLAHybridModel, epochs=2000, verbose=true)

# Validate
results = validate(model, verbose=true)

# Show body regions
list_body_regions()

# Compare conditions
compare_conditions(model, 50.0)

# Practical predictions
println("\n" * "="^70)
println("  APLICAÇÕES PRÁTICAS PARA SCAFFOLD DESIGN")
println("="^70)

println("\n  Cenário 1: Scaffold para regeneração óssea")
println("  " * "-"^50)
Mn0 = 50.0
for t in [30, 60, 90, 120]
    # In vitro (lab testing)
    Mn_vitro = predict(model, Mn0, Float64(t), condition=:in_vitro)
    # In vivo bone
    Mn_bone = predict(model, Mn0, Float64(t), condition=:in_vivo, region=:bone)
    # In vivo with inflammation (early healing)
    Mn_inflam = predict(model, Mn0, Float64(t), condition=:in_vivo, region=:inflammation)

    @printf("  t=%3d dias: in vitro=%.1f | osso=%.1f | inflamação=%.1f kg/mol\n",
            t, Mn_vitro, Mn_bone, Mn_inflam)
end

println("\n  Cenário 2: Scaffold para cartilagem articular")
println("  " * "-"^50)
for t in [30, 60, 90, 120]
    Mn_vitro = predict(model, Mn0, Float64(t), condition=:in_vitro)
    Mn_cart = predict(model, Mn0, Float64(t), condition=:in_vivo, region=:cartilage)

    @printf("  t=%3d dias: in vitro=%.1f | cartilagem(35°C)=%.1f kg/mol\n",
            t, Mn_vitro, Mn_cart)
end

println("\n  Cenário 3: Implante subcutâneo (liberação de fármaco)")
println("  " * "-"^50)
for t in [7, 14, 30, 60, 90]
    result = predict_with_uncertainty(model, Mn0, Float64(t),
                                      condition=:in_vivo, region=:subcutaneous)
    @printf("  t=%2d dias: Mn = %.1f ± %.1f kg/mol (95%% CI: %.1f - %.1f)\n",
            t, result.mean, 2*result.σ, result.lower, result.upper)
end

println("\n  Half-life por condição:")
println("  " * "-"^50)
conditions = [
    (:in_vitro, nothing, "In vitro (37°C)"),
    (:in_vivo, :bone, "In vivo osso"),
    (:in_vivo, :cartilage, "In vivo cartilagem"),
    (:in_vivo, :subcutaneous, "In vivo subcutâneo"),
    (:in_vivo, :inflammation, "In vivo inflamação"),
]

for (cond, region, label) in conditions
    t_half = estimate_halflife(model, 50.0, condition=cond, region=region)
    @printf("  %-25s: t½ = %.1f dias\n", label, t_half)
end

println("\n" * "="^70)
println("  Modelo pronto para otimização de scaffolds!")
println("="^70)
