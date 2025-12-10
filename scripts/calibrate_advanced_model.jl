"""
Calibração das constantes cinéticas para o modelo avançado de degradação.
"""

using Statistics
using Printf

# Dados experimentais Kaique
const KAIQUE_DATA = Dict(
    "PLDLA" => Dict(
        :Mw => [94.4, 52.7, 35.9, 11.8],
        :Tg => [54.0, 54.0, 48.0, 36.0],
        :t => [0, 30, 60, 90]
    ),
    "PLDLA/TEC1%" => Dict(
        :Mw => [85.8, 31.6, 22.4, 12.1],
        :Tg => [49.0, 49.0, 38.0, 41.0],
        :t => [0, 30, 60, 90]
    ),
    "PLDLA/TEC2%" => Dict(
        :Mw => [68.4, 26.9, 19.4, 8.4],
        :Tg => [46.0, 44.0, 22.0, 35.0],
        :t => [0, 30, 60, 90]
    )
)

# Modelo combinado: dMw/dt = -k_r*Mw - k_e*Mw*(Mw0/Mw - 1)
function simulate_combined(Mw0, k_r, k_e, times)
    Mw_values = Float64[]
    for t_target in times
        Mw = Mw0
        dt = 0.5
        t_current = 0.0
        while t_current < t_target
            end_factor = max(0.0, Mw0/Mw - 1.0)
            dMw = (-k_r * Mw - k_e * Mw * end_factor) * dt
            Mw = max(0.1, Mw + dMw)
            t_current += dt
        end
        push!(Mw_values, Mw)
    end
    return Mw_values
end

println("="^70)
println("        CALIBRAÇÃO DO MODELO AVANÇADO DE DEGRADAÇÃO PLDLA")
println("="^70)

# =========================================================================
# PARTE 1: Calibração individual por material
# =========================================================================

println("\n" * "="^70)
println("PARTE 1: CALIBRAÇÃO INDIVIDUAL POR MATERIAL")
println("="^70)

calibrated_params = Dict{String, Dict}()

for (name, data) in KAIQUE_DATA
    Mw_exp = data[:Mw]
    t = Float64.(data[:t])

    best_err = Inf
    best_kr = 0.0
    best_ke = 0.0

    # Grid search refinado
    for k_r in 0.001:0.0005:0.045
        for k_e in 0.000:0.001:0.025
            Mw_pred = simulate_combined(Mw_exp[1], k_r, k_e, t)
            errors = [abs(Mw_pred[i] - Mw_exp[i])/Mw_exp[i] * 100 for i in 2:4]
            mean_err = mean(errors)

            if mean_err < best_err
                best_err = mean_err
                best_kr = k_r
                best_ke = k_e
            end
        end
    end

    calibrated_params[name] = Dict(
        :k_random => best_kr,
        :k_end => best_ke,
        :error => best_err
    )

    println("\n$name:")
    @printf("  k_random = %.4f /dia\n", best_kr)
    @printf("  k_end = %.4f /dia\n", best_ke)
    @printf("  Erro médio Mw = %.1f%%\n", best_err)

    # Mostrar detalhes
    Mw_pred = simulate_combined(Mw_exp[1], best_kr, best_ke, t)
    println("  Detalhes:")
    for i in 1:4
        err = abs(Mw_pred[i] - Mw_exp[i]) / Mw_exp[i] * 100
        @printf("    t=%d dias: Mw_exp=%.1f, Mw_pred=%.1f, erro=%.1f%%\n",
                data[:t][i], Mw_exp[i], Mw_pred[i], err)
    end
end

# =========================================================================
# PARTE 2: Análise do padrão de k com TEC
# =========================================================================

println("\n" * "="^70)
println("PARTE 2: RELAÇÃO ENTRE k E CONCENTRAÇÃO DE TEC")
println("="^70)

TEC_levels = [0.0, 1.0, 2.0]
materials = ["PLDLA", "PLDLA/TEC1%", "PLDLA/TEC2%"]

println("\n┌────────────────┬──────────┬──────────┬──────────┐")
println("│    Material    │ TEC (%)  │ k_random │  k_end   │")
println("├────────────────┼──────────┼──────────┼──────────┤")

for (i, name) in enumerate(materials)
    p = calibrated_params[name]
    @printf("│ %-14s │ %8.1f │ %8.4f │ %8.4f │\n",
            name, TEC_levels[i], p[:k_random], p[:k_end])
end
println("└────────────────┴──────────┴──────────┴──────────┘")

# Calcular correlação TEC vs k
k_randoms = [calibrated_params[m][:k_random] for m in materials]
k_ends = [calibrated_params[m][:k_end] for m in materials]

# Ajuste linear: k = a + b*TEC
function linear_fit(x, y)
    n = length(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x .* y)
    sum_xx = sum(x .* x)

    b = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x^2)
    a = (sum_y - b * sum_x) / n

    return a, b
end

a_r, b_r = linear_fit(TEC_levels, k_randoms)
a_e, b_e = linear_fit(TEC_levels, k_ends)

println("\nRelação k vs TEC:")
@printf("  k_random = %.4f + %.4f × TEC%%  (taxa base + aceleração por TEC)\n", a_r, b_r)
@printf("  k_end = %.4f + %.4f × TEC%%\n", a_e, b_e)

# =========================================================================
# PARTE 3: Modelo Tg três fases - Calibração
# =========================================================================

println("\n" * "="^70)
println("PARTE 3: CALIBRAÇÃO DO MODELO Tg TRÊS FASES")
println("="^70)

# Para cada material, calcular parâmetros do modelo Tg
for (name, data) in KAIQUE_DATA
    Mw_exp = data[:Mw]
    Tg_exp = data[:Tg]
    t = Float64.(data[:t])

    println("\n$name:")
    println("  Dados experimentais Tg vs Mw:")
    for i in 1:4
        @printf("    t=%d: Mw=%.1f, Tg=%.1f\n", data[:t][i], Mw_exp[i], Tg_exp[i])
    end

    # Análise: Tg deveria aumentar se cristalização aumenta
    # Mas dados mostram que Tg diminui em alguns casos

    # Calcular alpha empírico (Tg ~ Mw^α)
    # Tg/Tg0 = (Mw/Mw0)^α → α = ln(Tg/Tg0) / ln(Mw/Mw0)

    println("  Expoente α aparente (Tg ~ Mw^α):")
    for i in 2:4
        if Mw_exp[i] > 0 && Tg_exp[i] > 0
            alpha = log(Tg_exp[i]/Tg_exp[1]) / log(Mw_exp[i]/Mw_exp[1])
            @printf("    t=%d: α = %.3f\n", data[:t][i], alpha)
        end
    end
end

# =========================================================================
# PARTE 4: Modelo final calibrado
# =========================================================================

println("\n" * "="^70)
println("PARTE 4: PARÂMETROS FINAIS PARA AdvancedPLDLADegradation.jl")
println("="^70)

println("\n# Constantes cinéticas calibradas:")
@printf("k_random_base = %.4f  # /dia (PLDLA puro)\n", calibrated_params["PLDLA"][:k_random])
@printf("k_end_base = %.4f     # /dia\n", calibrated_params["PLDLA"][:k_end])
@printf("k_TEC_factor = %.4f   # incremento por %% TEC\n", b_r)

println("\n# Para usar no código:")
println("# k_random = k_random_base + k_TEC_factor * TEC%")
println("# k_end = k_end_base + k_TEC_factor_end * TEC%")

# Calcular erro médio global com parâmetros calibrados
global_errors = [calibrated_params[m][:error] for m in materials]
println("\n# Erro médio Mw com calibração:")
for (i, m) in enumerate(materials)
    @printf("#   %s: %.1f%%\n", m, calibrated_params[m][:error])
end
@printf("#   GLOBAL: %.1f%%\n", mean(global_errors))

# =========================================================================
# PARTE 5: Teste com parâmetros universais
# =========================================================================

println("\n" * "="^70)
println("PARTE 5: TESTE COM PARÂMETROS UNIVERSAIS")
println("="^70)

# Usar valores médios para um modelo universal
k_r_universal = a_r
k_e_universal = a_e + 0.003  # pequeno ajuste

println("\nParâmetros universais (todos os materiais):")
@printf("  k_random_base = %.4f /dia\n", k_r_universal)
@printf("  k_end_base = %.4f /dia\n", k_e_universal)
@printf("  k_TEC_acceleration = %.4f /dia por %%TEC\n", b_r)

println("\nValidação com modelo universal:")

for (i, name) in enumerate(materials)
    data = KAIQUE_DATA[name]
    Mw_exp = data[:Mw]
    t = Float64.(data[:t])
    TEC = TEC_levels[i]

    k_r = k_r_universal + b_r * TEC
    k_e = k_e_universal + b_e * TEC

    Mw_pred = simulate_combined(Mw_exp[1], k_r, k_e, t)
    errors = [abs(Mw_pred[j] - Mw_exp[j])/Mw_exp[j] * 100 for j in 2:4]

    @printf("\n  %s (k_r=%.4f, k_e=%.4f):\n", name, k_r, k_e)
    for j in 1:4
        err = abs(Mw_pred[j] - Mw_exp[j]) / Mw_exp[j] * 100
        @printf("    t=%d: exp=%.1f, pred=%.1f, err=%.1f%%\n",
                data[:t][j], Mw_exp[j], Mw_pred[j], err)
    end
    @printf("    Erro médio: %.1f%%\n", mean(errors))
end

println("\n" * "="^70)
println("CALIBRAÇÃO COMPLETA")
println("="^70)
