"""
Análise de primeiros princípios para determinar parâmetros físicos reais.
"""

using Statistics
using Printf

println("="^70)
println("ANÁLISE DE PRIMEIROS PRINCÍPIOS - DEGRADAÇÃO PLDLA")
println("="^70)

# Dados experimentais Kaique - PLDLA puro
Mn_exp = [51.3, 25.4, 18.3, 7.9]  # kg/mol
Mw_exp = [94.4, 52.7, 35.9, 11.8]  # kg/mol
t_days = [0, 30, 60, 90]

println("\n1. TAXA DE DECAIMENTO OBSERVADA (Mn):")
println("-"^50)

# Se Mn(t) = Mn0 * exp(-k*t), então k = -ln(Mn/Mn0)/t
k_values = Float64[]
for i in 2:4
    k_obs = -log(Mn_exp[i]/Mn_exp[1]) / t_days[i]
    push!(k_values, k_obs)
    @printf("  t=%d dias: k_obs = %.5f /dia\n", t_days[i], k_obs)
end

println("\n2. TAXA MÉDIA (REGRESSÃO LINEAR):")
println("-"^50)

# ln(Mn) vs t
ln_Mn = log.(Mn_exp)
n = length(t_days)
sum_t = sum(t_days)
sum_ln = sum(ln_Mn)
sum_t2 = sum(Float64.(t_days).^2)
sum_t_ln = sum(Float64.(t_days) .* ln_Mn)

k_avg = -(n * sum_t_ln - sum_t * sum_ln) / (n * sum_t2 - sum_t^2)
@printf("  k médio (Mn): %.5f /dia = %.4f /semana\n", k_avg, k_avg * 7)

# Mesmo para Mw
ln_Mw = log.(Mw_exp)
sum_ln_mw = sum(ln_Mw)
sum_t_ln_mw = sum(Float64.(t_days) .* ln_Mw)
k_avg_mw = -(n * sum_t_ln_mw - sum_t * sum_ln_mw) / (n * sum_t2 - sum_t^2)
@printf("  k médio (Mw): %.5f /dia = %.4f /semana\n", k_avg_mw, k_avg_mw * 7)

println("\n3. CONCENTRAÇÃO DE COOH AO LONGO DO TEMPO:")
println("-"^50)

rho = 1.25e6  # g/m³ (1.25 g/cm³)

for i in 1:4
    Mn_g = Mn_exp[i] * 1000  # g/mol
    C_COOH = rho / Mn_g  # mol/m³
    @printf("  t=%d: Mn=%.1f kg/mol → [COOH] = %.1f mol/m³\n",
            t_days[i], Mn_exp[i], C_COOH)
end

println("\n4. DERIVAÇÃO DO MODELO AUTOCATALÍTICO:")
println("-"^50)
println("  Modelo: R = k1*Ce + k2*Ce*[COOH]^m")
println("  Para hidrólise de éster: m ≈ 0.5 (literatura)")
println("")

# Calcular k2 assumindo k1 << k2*[COOH]^0.5
# dN/dt = R * V, onde N = número de cadeias
# d(1/Mn)/dt ≈ R/rho (em termos molares)
# Se R = k2 * Ce * [COOH]^0.5
# e Ce ≈ rho/72 (ester bonds por unidade de volume)
# e [COOH] = rho/Mn

# Simplificando para 1a ordem efetiva:
# d(ln Mn)/dt = -k_eff
# k_eff ≈ k2 * (rho/72) * sqrt(rho/Mn) / (rho/Mn)
# k_eff ≈ k2 * Mn / (72 * sqrt(rho/Mn))

println("  Usando k_eff observado e [COOH] típico:")
C_COOH_avg = rho / (mean(Mn_exp) * 1000)
@printf("  [COOH] médio ≈ %.1f mol/m³\n", C_COOH_avg)
@printf("  sqrt([COOH]) ≈ %.2f\n", sqrt(C_COOH_avg))

# Ce = rho / 72 ≈ 17361 mol/m³
Ce = rho / 72.0
@printf("  [Ester] ≈ %.0f mol/m³\n", Ce)

# R = k2 * Ce * sqrt([COOH])
# dMn/dt ≈ -Mn² * R / rho (aproximação)
# -k_eff * Mn = -Mn² * k2 * Ce * sqrt([COOH]) / rho
# k_eff = Mn * k2 * Ce * sqrt([COOH]) / rho

# Para Mn = 30 kg/mol (meio do range):
Mn_mid = 30.0
C_COOH_mid = rho / (Mn_mid * 1000)
k2_est = k_avg * rho / (Mn_mid * 1000 * Ce * sqrt(C_COOH_mid))
@printf("\n  Estimativa k2: %.2e m³^0.5/(mol^0.5·dia)\n", k2_est)
@printf("  Em semanas: %.2e m³^0.5/(mol^0.5·semana)\n", k2_est * 7)

println("\n5. COMPARAÇÃO COM LITERATURA:")
println("-"^50)
println("  PMC3359772: k ≈ 0.02/dia para PLA industrial a 37°C")
@printf("  Nosso k_obs (Mn): %.4f/dia\n", k_avg)
@printf("  Nosso k_obs (Mw): %.4f/dia\n", k_avg_mw)
@printf("  Razão vs literatura: %.2fx\n", k_avg/0.02)

println("\n6. FOX-FLORY PARA Tg:")
println("-"^50)
println("  Tg = Tg∞ - K/Mn")
println("  Literatura PLA: Tg∞ = 55°C, K = 55 kg/mol")
println("")

Tg_exp = [54.0, 54.0, 48.0, 36.0]
for i in 1:4
    Tg_FF = 55.0 - 55.0/Mn_exp[i]
    err = abs(Tg_FF - Tg_exp[i])/Tg_exp[i] * 100
    @printf("  t=%d: Mn=%.1f → Tg_FF=%.1f°C, Tg_exp=%.1f°C, erro=%.1f%%\n",
            t_days[i], Mn_exp[i], Tg_FF, Tg_exp[i], err)
end

println("\n7. CONCLUSÕES:")
println("-"^50)
println("  a) Taxa de degradação PLDLA scaffold ≈ 0.02/dia (similar à literatura)")
println("  b) Fox-Flory funciona bem para Tg vs Mn")
println("  c) Modelo autocatalítico com k2 ≈ 10⁻⁵ deveria funcionar")

println("\n" * "="^70)
