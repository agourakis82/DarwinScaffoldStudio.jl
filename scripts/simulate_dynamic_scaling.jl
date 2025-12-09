#!/usr/bin/env julia
# Simulação da predição dinâmica τ ~ L^φ
# Baseado na universalidade de Fibonacci (Spohn 2024)

using Statistics
using Random

Random.seed!(42)

φ = (1 + √5) / 2

println("="^80)
println("  SIMULAÇÃO DA PREDIÇÃO DINÂMICA: τ ~ L^φ")
println("="^80)

println("""
Teoria (Spohn 2024):
  Em sistemas com universalidade de Fibonacci, o expoente dinâmico z = φ.
  Isso implica que o tempo de relaxação escala como:
    τ_relax ~ L^z = L^φ

  Para scaffolds φ-fractais, predizemos:
    τ_tortuosity ~ L^φ (tempo de difusão)
    ξ ~ L^φ (comprimento de correlação)
""")

# ============================================================================
# PARTE 1: Simulação de Random Walk em estrutura fractal
# ============================================================================
println("\n" * "─"^80)
println("PARTE 1: RANDOM WALK EM ESTRUTURA FRACTAL")
println("─"^80)

"""
Gera uma estrutura fractal 2D usando Percolation
"""
function generate_fractal_structure(L::Int, p::Float64)
    # Percolação simples
    structure = rand(L, L) .< p
    return structure
end

"""
Simula random walk e retorna tempo médio para atravessar
"""
function random_walk_time(structure::AbstractMatrix{Bool}, n_walks::Int=1000)
    L = size(structure, 1)
    times = Int[]

    for _ in 1:n_walks
        # Começar na borda esquerda
        start_positions = findall(structure[:, 1])
        if isempty(start_positions)
            continue
        end

        pos = [rand(start_positions)[1], 1]
        t = 0
        max_steps = L * L * 10

        while pos[2] < L && t < max_steps
            # Movimento aleatório
            moves = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            shuffle!(moves)

            moved = false
            for move in moves
                new_pos = pos .+ move
                if 1 <= new_pos[1] <= L && 1 <= new_pos[2] <= L
                    if structure[new_pos[1], new_pos[2]]
                        pos = new_pos
                        moved = true
                        break
                    end
                end
            end

            t += 1
            if !moved
                break
            end
        end

        if pos[2] >= L
            push!(times, t)
        end
    end

    return isempty(times) ? NaN : mean(times)
end

# Simular para diferentes tamanhos L
L_values = [16, 32, 64, 128]
p_percolation = 0.7  # Acima do limiar crítico

println("\nSimulando random walks para diferentes L (p = $p_percolation)...")
println("(Isso pode demorar alguns segundos)")

τ_values = Float64[]
for L in L_values
    structure = generate_fractal_structure(L, p_percolation)
    τ = random_walk_time(structure, 500)
    push!(τ_values, τ)
    println("  L = $L: τ = $(isnan(τ) ? "N/A" : round(τ, digits=1))")
end

# Ajustar expoente z
valid = .!isnan.(τ_values)
if sum(valid) >= 3
    log_L = log.(L_values[valid])
    log_τ = log.(τ_values[valid])

    n = length(log_L)
    sum_x = sum(log_L)
    sum_y = sum(log_τ)
    sum_xx = sum(log_L .^ 2)
    sum_xy = sum(log_L .* log_τ)

    z_fit = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x^2)

    println("\n  Expoente z ajustado: $(round(z_fit, digits=3))")
    println("  Valor teórico (φ): $(round(φ, digits=3))")
    println("  Diferença: $(round(abs(z_fit - φ), digits=3))")
else
    z_fit = NaN
    println("\n  Dados insuficientes para ajuste")
end

# ============================================================================
# PARTE 2: Modelo teórico de difusão anômala
# ============================================================================
println("\n" * "─"^80)
println("PARTE 2: MODELO DE DIFUSÃO ANÔMALA")
println("─"^80)

println("""
Em meios fractais, a difusão é anômala:
  ⟨r²(t)⟩ ~ t^(2/d_w)

onde d_w é a dimensão de walk.

Para scaffold φ-fractal:
  d_w = 2 + θ, onde θ é o expoente de resistência

Da relação Einstein: d_w = d_f + ζ
  onde d_f = dimensão fractal, ζ = expoente de tortuosidade

Se d_f = φ ≈ 1.618 e predizemos d_w ≈ 2 + 1/φ² ≈ 2.38:
""")

d_f = φ
θ_predicted = 1/φ^2  # ≈ 0.382
d_w_predicted = 2 + θ_predicted

println("  d_f (dimensão fractal) = $(round(d_f, digits=4))")
println("  θ (expoente resistência) = 1/φ² = $(round(θ_predicted, digits=4))")
println("  d_w (dimensão walk) = 2 + θ = $(round(d_w_predicted, digits=4))")

# Expoente de difusão
α_diff = 2 / d_w_predicted
println("\n  Expoente de difusão α = 2/d_w = $(round(α_diff, digits=4))")
println("  (α = 1 → difusão normal, α < 1 → subdifusão)")

# ============================================================================
# PARTE 3: Predições para experimentos
# ============================================================================
println("\n" * "─"^80)
println("PARTE 3: PREDIÇÕES EXPERIMENTAIS")
println("─"^80)

println("""
PREDIÇÃO 1: Scaling de tortuosidade com tamanho do sistema
  τ(L) = τ₀ × L^z, onde z = φ ≈ 1.618

  Para scaffold de tamanho L (em unidades de poro):
""")

L_exp = [10, 50, 100, 500, 1000]
τ_0 = 1.0  # Constante de normalização

println("  L (poros)    τ/τ₀")
println("  " * "─"^25)
for L in L_exp
    τ_pred = L^φ
    println("  $(lpad(L, 6))       $(round(τ_pred, digits=1))")
end

println("""

PREDIÇÃO 2: Tempo de difusão molecular
  t_diff ~ L^(d_w) = L^$(round(d_w_predicted, digits=2))

  Para célula migrando em scaffold:
  Se L = 100 poros e t₀ = 1 min para L = 1:
""")

t_0 = 1.0  # minutos
for L in [10, 50, 100]
    t_diff = t_0 * L^d_w_predicted
    println("  L = $L poros: t ≈ $(round(t_diff/60, digits=1)) horas")
end

println("""

PREDIÇÃO 3: Correlação comprimento-tempo
  ξ(t) ~ t^(1/z) = t^(1/φ) ≈ t^0.618

  Comprimento de correlação após tempo t:
""")

for t in [1, 10, 100, 1000]
    ξ = t^(1/φ)
    println("  t = $t: ξ ≈ $(round(ξ, digits=2))")
end

# ============================================================================
# PARTE 4: Conexão com expoente de percolação
# ============================================================================
println("\n" * "─"^80)
println("PARTE 4: CONEXÃO COM PERCOLAÇÃO")
println("─"^80)

println("""
Nosso resultado anterior: μ ≈ 0.31 (expoente de tortuosidade)

Relação com dimensão walk:
  d_w = d + μ, onde d = dimensão do espaço

Para d = 3 (scaffold 3D):
  d_w = 3 + 0.31 = 3.31

Comparação com predição φ-based:
  d_w (percolação) = 3.31
  d_w (φ theory) = 2 + 1/φ² = 2.38 (para 2D)

Para 3D, predizemos:
  d_w (3D) = 3 + 1/φ² = $(round(3 + 1/φ^2, digits=3))
""")

d_w_3D_predicted = 3 + 1/φ^2
d_w_3D_measured = 3.31

println("  d_w predito (φ): $(round(d_w_3D_predicted, digits=3))")
println("  d_w medido: $(round(d_w_3D_measured, digits=3))")
println("  Diferença: $(round(abs(d_w_3D_predicted - d_w_3D_measured), digits=3))")
println("  Erro relativo: $(round(100*abs(d_w_3D_predicted - d_w_3D_measured)/d_w_3D_measured, digits=1))%")

# ============================================================================
# RESUMO
# ============================================================================
println("\n" * "─"^80)
println("RESUMO")
println("─"^80)

println("""

╔════════════════════════════════════════════════════════════════════════╗
║  PREDIÇÕES DINÂMICAS PARA SCAFFOLDS φ-FRACTAIS                        ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  EXPOENTE DINÂMICO:                                                    ║
║    z = φ ≈ 1.618 (universalidade Fibonacci)                           ║
║                                                                        ║
║  SCALING DE TORTUOSIDADE:                                              ║
║    τ(L) ~ L^φ                                                         ║
║                                                                        ║
║  DIMENSÃO DE WALK:                                                     ║
║    d_w = d + 1/φ² (d = dimensão do espaço)                            ║
║    2D: d_w ≈ 2.38                                                      ║
║    3D: d_w ≈ 3.38                                                      ║
║                                                                        ║
║  DIFUSÃO ANÔMALA:                                                      ║
║    ⟨r²(t)⟩ ~ t^(2/d_w) (subdifusão)                                   ║
║                                                                        ║
║  CORRELAÇÃO ESPAÇO-TEMPO:                                              ║
║    ξ(t) ~ t^(1/φ) ≈ t^0.618                                           ║
║                                                                        ║
║  VALIDAÇÃO:                                                            ║
║    d_w (3D) predito: 3.38                                              ║
║    d_w (3D) medido: 3.31                                              ║
║    Erro: ~2%                                                          ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")

println("="^80)
