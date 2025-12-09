#!/usr/bin/env julia
# Validação do modelo power-law com dados reais de materiais porosos

using Statistics

# Dados de solo (pore space) - porosidade e tortuosidade
# Extraídos do CSV de dados reais

porosities = [0.251, 0.284, 0.349, 0.382, 0.266, 0.281, 0.276, 0.272, 0.219, 0.237,
              0.260, 0.243, 0.299, 0.349, 0.293, 0.286, 0.264, 0.234, 0.241, 0.250,
              0.284, 0.304, 0.304, 0.254, 0.265, 0.250, 0.246, 0.265, 0.270, 0.258,
              0.271, 0.254, 0.335, 0.391, 0.323, 0.314, 0.353, 0.351, 0.354, 0.296]

tortuosities = [1.149, 1.130, 1.105, 1.095, 1.128, 1.123, 1.133, 1.128, 1.152, 1.145,
                1.136, 1.155, 1.115, 1.098, 1.126, 1.120, 1.150, 1.156, 1.169, 1.152,
                1.105, 1.110, 1.116, 1.151, 1.146, 1.156, 1.160, 1.143, 1.134, 1.138,
                1.126, 1.143, 1.094, 1.084, 1.110, 1.113, 1.112, 1.092, 1.101, 1.095]

φ = (1 + √5) / 2

println("="^80)
println("  VALIDAÇÃO DO MODELO POWER-LAW COM DADOS REAIS")
println("="^80)

println("\n" * "─"^80)
println("ESTATÍSTICAS DOS DADOS")
println("─"^80)

n = length(porosities)
println("\nPorosidade:")
println("  n = $n")
println("  Média = $(round(mean(porosities), digits=3))")
println("  Min = $(round(minimum(porosities), digits=3))")
println("  Max = $(round(maximum(porosities), digits=3))")
println("  Std = $(round(std(porosities), digits=3))")

println("\nTortuosidade:")
println("  Média = $(round(mean(tortuosities), digits=4))")
println("  Min = $(round(minimum(tortuosities), digits=4))")
println("  Max = $(round(maximum(tortuosities), digits=4))")
println("  Std = $(round(std(tortuosities), digits=4))")

println("\n" * "─"^80)
println("MODELO 1: TORTUOSIDADE LINEAR")
println("─"^80)

# Fit linear: τ = a + b*p
sum_p = sum(porosities)
sum_t = sum(tortuosities)
sum_pp = sum(porosities .^ 2)
sum_pt = sum(porosities .* tortuosities)

b_lin = (n * sum_pt - sum_p * sum_t) / (n * sum_pp - sum_p^2)
a_lin = (sum_t - b_lin * sum_p) / n

println("\nτ(p) = $(round(a_lin, digits=4)) + $(round(b_lin, digits=4)) × p")

# R² para linear
predicted_lin = a_lin .+ b_lin .* porosities
ss_res_lin = sum((tortuosities .- predicted_lin) .^ 2)
ss_tot = sum((tortuosities .- mean(tortuosities)) .^ 2)
r2_lin = 1 - ss_res_lin / ss_tot
println("R² = $(round(r2_lin, digits=4))")

println("\n" * "─"^80)
println("MODELO 2: TORTUOSIDADE POWER-LAW")
println("─"^80)

# Modelo: τ(p) = τ_min + A × (1-p)^α
# Assumindo τ_min → 1 quando p → 1

τ_shifted = tortuosities .- 1.0
log_1_minus_p = log.(1.0 .- porosities)

# Filtrar valores válidos (τ > 1)
valid = τ_shifted .> 0.001
log_τ_shifted = log.(τ_shifted[valid])
log_1_mp_valid = log_1_minus_p[valid]

# Fit linear no log
n_v = sum(valid)
sum_x = sum(log_1_mp_valid)
sum_y = sum(log_τ_shifted)
sum_xx = sum(log_1_mp_valid .^ 2)
sum_xy = sum(log_1_mp_valid .* log_τ_shifted)

α_fit = (n_v * sum_xy - sum_x * sum_y) / (n_v * sum_xx - sum_x^2)
log_A = (sum_y - α_fit * sum_x) / n_v
A_fit = exp(log_A)

println("\nτ(p) = 1 + $(round(A_fit, digits=4)) × (1-p)^$(round(α_fit, digits=4))")

# Predições
predicted_pow = 1.0 .+ A_fit .* (1.0 .- porosities) .^ α_fit
ss_res_pow = sum((tortuosities .- predicted_pow) .^ 2)
r2_pow = 1 - ss_res_pow / ss_tot
println("R² = $(round(r2_pow, digits=4))")

println("\n" * "─"^80)
println("MODELO 3: DIMENSÃO FRACTAL D(p)")
println("─"^80)

# Se D(p) = φ + (3-φ)(1-p)^α como proposto
println("\nModelo proposto para D:")
println("  D(p) = φ + (3-φ)(1-p)^α")
println("  com α ≈ 0.88 (calibrado de KFoam)")

α_D = 0.88
D_predicted = φ .+ (3-φ) .* (1.0 .- porosities) .^ α_D

println("\nPrevisões de D para os dados:")
println("  p_min = $(round(minimum(porosities), digits=2)) → D = $(round(maximum(D_predicted), digits=3))")
println("  p_max = $(round(maximum(porosities), digits=2)) → D = $(round(minimum(D_predicted), digits=3))")
println("  p_mean = $(round(mean(porosities), digits=2)) → D = $(round(mean(D_predicted), digits=3))")

println("\n" * "─"^80)
println("CONEXÃO TORTUOSIDADE-DIMENSÃO FRACTAL")
println("─"^80)

println("\nTestando relação τ vs D:")
println("  Se D ~ $(round(mean(D_predicted), digits=2)) (média), τ ~ $(round(mean(tortuosities), digits=2))")
println("  Razão (D-1)/(τ-1) = $(round((mean(D_predicted)-1)/(mean(tortuosities)-1), digits=2))")

# Correlação entre D_predicted e τ observada
corr_D_τ = sum((D_predicted .- mean(D_predicted)) .* (tortuosities .- mean(tortuosities))) /
           (std(D_predicted) * std(tortuosities) * (n-1))
println("  Correlação D(p) vs τ(p): $(round(corr_D_τ, digits=3))")

println("\n" * "─"^80)
println("VERIFICAÇÃO: 3φ-2 NA LITERATURA")
println("─"^80)

println("\nEncontrado na busca:")
println("  • Análise de shales: D₂ = 2.854 a 2.863 (ACS Omega)")
println("  • Formação Longmaxi: D₂ avg = 2.830")
println("  • 3φ - 2 = $(round(3φ-2, digits=3))")
println()
println("  COINCIDÊNCIA? D₂ em shales ≈ 3φ - 2 !")

println("\n" * "─"^80)
println("RESUMO")
println("─"^80)

melhor_lin = r2_lin > r2_pow ? "MELHOR" : "pior"
melhor_pow = r2_pow > r2_lin ? "MELHOR" : "pior"

println("""

┌─────────────────────────────────────────────────────────────────────┐
│  VALIDAÇÃO DO MODELO POWER-LAW                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DADOS: Solo poroso (n=$n)                                          │
│    Porosidade: $(round(minimum(porosities), digits=2)) - $(round(maximum(porosities), digits=2))                                          │
│    Tortuosidade: $(round(minimum(tortuosities), digits=2)) - $(round(maximum(tortuosities), digits=2))                                        │
│                                                                     │
│  MODELO LINEAR: τ = $(round(a_lin, digits=3)) + $(round(b_lin, digits=3))×p                            │
│    R² = $(round(r2_lin, digits=3)) ($melhor_lin)                                               │
│                                                                     │
│  MODELO POWER-LAW: τ = 1 + $(round(A_fit, digits=3))×(1-p)^$(round(α_fit, digits=2))                      │
│    R² = $(round(r2_pow, digits=3)) ($melhor_pow)                                               │
│                                                                     │
│  DESCOBERTA: D ≈ 2.854 aparece em shales naturais!                  │
│    Isso sugere que 3φ-2 pode ser universal para                     │
│    materiais porosos naturais/estocásticos                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

println("="^80)
