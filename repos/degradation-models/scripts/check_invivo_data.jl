#!/usr/bin/env julia
# Check available in vivo data

include(joinpath(@__DIR__, "..", "data", "literature_degradation_database.jl"))

println("="^60)
println("DADOS IN VIVO DISPONÍVEIS NA DATABASE")
println("="^60)

for d in DEGRADATION_DATABASE
    if d.condition == :in_vivo
        println("\n$(d.id):")
        println("  Material: $(d.material)")
        println("  L:DL = $(d.ratio_L):$(100-d.ratio_L)")
        println("  Mn0 = $(d.Mn0) kg/mol")
        println("  Times: $(d.times) days")
        println("  Mn: $(d.Mn) kg/mol")

        # Calculate degradation rate
        if length(d.times) >= 2 && d.times[end] > 0
            final_frac = d.Mn[end] / d.Mn0
            println("  Final fraction: $(round(final_frac*100, digits=1))%")
        end
    end
end

println("\n" * "="^60)
println("COMPARAÇÃO IN VITRO vs IN VIVO (mesma razão L:DL)")
println("="^60)

# Group by ratio
for ratio in [70, 96, 100]
    invitro = filter(d -> d.ratio_L == ratio && d.condition == :in_vitro, DEGRADATION_DATABASE)
    invivo = filter(d -> d.ratio_L == ratio && d.condition == :in_vivo, DEGRADATION_DATABASE)

    if !isempty(invitro) && !isempty(invivo)
        println("\nL:DL = $ratio:$(100-ratio)")
        println("  In vitro: $(length(invitro)) datasets")
        println("  In vivo: $(length(invivo)) datasets")
    end
end
