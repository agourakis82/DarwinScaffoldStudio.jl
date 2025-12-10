"""
    ThermodynamicDegradation

Degradação de PLDLA baseada em primeiros princípios:
- Termodinâmica: ΔG, ΔH, ΔS da hidrólise
- Cinética: Teoria do Estado de Transição (Eyring)
- Físico-química: Difusão de água, atividade química
- Química: Mecanismo AAC2 de hidrólise de éster

FUNDAMENTOS:
============

1. TERMODINÂMICA DA HIDRÓLISE
   R-COO-R' + H₂O → R-COOH + R'-OH

   ΔG° = ΔH° - T·ΔS°

   Para ésteres alifáticos:
   ΔH° ≈ -10 a -15 kJ/mol (exotérmico)
   ΔS° ≈ +50 a +80 J/(mol·K) (aumento de entropia)
   ΔG° ≈ -25 a -40 kJ/mol (espontâneo)

2. TEORIA DE EYRING (Estado de Transição)
   k = (kB·T/h) · exp(-ΔG‡/RT)

   Onde ΔG‡ = ΔH‡ - T·ΔS‡ (barreira de ativação)

   Para hidrólise ácida de éster:
   ΔH‡ ≈ 75-85 kJ/mol
   ΔS‡ ≈ -50 a -100 J/(mol·K) (estado de transição ordenado)

3. MECANISMO AAC2 (Acid-catalyzed Acyl Cleavage, bimolecular)

   Etapa 1: Protonação da carbonila
   R-C(=O)-OR' + H⁺ ⇌ R-C(=OH⁺)-OR'     (rápido, K_prot)

   Etapa 2: Ataque nucleofílico da água (determinante)
   R-C(=OH⁺)-OR' + H₂O → R-C(OH)₂-OR'   (lento, k₂)

   Etapa 3: Eliminação do álcool
   R-C(OH)₂-OR' → R-COOH + R'-OH + H⁺   (rápido)

4. EQUAÇÃO DE TAXA COMPLETA

   r = k₂ · K_prot · [Éster] · [H⁺] · [H₂O] · f_amorfo · f_difusão

   Onde:
   - K_prot = constante de protonação da carbonila
   - k₂ = constante da etapa determinante
   - f_amorfo = fração amorfa acessível
   - f_difusão = fator de limitação difusional

Author: Darwin Scaffold Studio
Date: December 2025
"""
module ThermodynamicDegradation

export validate_thermodynamic_model, predict_thermodynamic
export calculate_eyring_rate, calculate_diffusion_factor
export calculate_protonation_equilibrium

using Statistics
using Printf

# =============================================================================
# CONSTANTES FUNDAMENTAIS
# =============================================================================

const CONSTANTS = (
    R = 8.314,              # J/(mol·K) - constante dos gases
    kB = 1.381e-23,         # J/K - constante de Boltzmann
    h = 6.626e-34,          # J·s - constante de Planck
    NA = 6.022e23,          # mol⁻¹ - número de Avogadro
    T_ref = 310.15          # K (37°C)
)

# =============================================================================
# TERMODINÂMICA DA HIDRÓLISE
# =============================================================================

const THERMODYNAMICS = (
    # Reação: Éster + H₂O → Ácido + Álcool
    # Valores para lactídeo/ácido lático

    # Entalpia de reação (exotérmico)
    ΔH_rxn = -12.0e3,       # J/mol (-12 kJ/mol)

    # Entropia de reação (favorável - mais moléculas)
    ΔS_rxn = 65.0,          # J/(mol·K)

    # Energia livre de reação a 37°C
    # ΔG = ΔH - T·ΔS = -12000 - 310.15*65 = -32.2 kJ/mol
    # Reação espontânea mas cineticamente lenta

    # Estado de transição (barreira de ativação)
    ΔH_act = 78.0e3,        # J/mol (78 kJ/mol) - barreira entálpica
    ΔS_act = -80.0,         # J/(mol·K) - estado de transição ordenado

    # pKa do intermediário protonado
    pKa_carbonyl = -6.5,    # Carbonila protonada é muito ácida
    pKa_lactic = 3.86       # Ácido lático produto
)

# =============================================================================
# PROPRIEDADES DO POLÍMERO
# =============================================================================

const POLYMER = (
    # PLDLA específico
    M_repeat = 72.06,       # g/mol - massa do monômero (lactídeo/2)
    ρ = 1.25e6,             # g/m³ - densidade do polímero

    # Difusão de água
    D_water_amorphous = 1.0e-12,   # m²/s - coef. difusão em região amorfa
    D_water_crystalline = 1.0e-14, # m²/s - coef. difusão em região cristalina

    # Concentração de ligações éster
    C_ester_initial = 13.9e3,      # mol/m³ (≈ ρ/M_repeat)

    # Parâmetros de solubilidade de água
    S_water_37C = 0.02,     # g H₂O / g polímero (2% em massa)

    # Fox-Flory
    Tg_inf = 330.15,        # K (57°C)
    K_ff = 55.0e3           # g/mol
)

# =============================================================================
# DADOS EXPERIMENTAIS
# =============================================================================

const DATASETS = Dict(
    "Kaique_PLDLA" => (
        Mn = [51.3, 25.4, 18.3, 7.9],
        t = [0.0, 30.0, 60.0, 90.0],
        T = 310.15,  # K
        pH = 7.4,
        condition = :in_vitro,
        source = "Kaique PhD thesis"
    ),
    "Kaique_TEC1" => (
        Mn = [45.0, 19.3, 11.7, 8.1],
        t = [0.0, 30.0, 60.0, 90.0],
        T = 310.15,
        pH = 7.4,
        TEC = 1.0,
        condition = :in_vitro,
        source = "Kaique PhD thesis"
    ),
    "Kaique_TEC2" => (
        Mn = [32.7, 15.0, 12.6, 6.6],
        t = [0.0, 30.0, 60.0, 90.0],
        T = 310.15,
        pH = 7.4,
        TEC = 2.0,
        condition = :in_vitro,
        source = "Kaique PhD thesis"
    ),
    "BioEval_InVivo" => (
        Mn = [99.0, 92.0, 85.0],
        t = [0.0, 28.0, 56.0],
        T = 310.15,
        pH = 7.35,
        condition = :subcutaneous,
        source = "BioEval in vivo"
    )
)

# =============================================================================
# TEORIA DE EYRING - TAXA DO ESTADO DE TRANSIÇÃO
# =============================================================================

"""
Calcula a constante de taxa usando a Teoria do Estado de Transição (Eyring).

k = (kB·T/h) · exp(-ΔG‡/RT)
  = (kB·T/h) · exp(-ΔH‡/RT) · exp(ΔS‡/R)

A 37°C:
- kB·T/h ≈ 6.4 × 10¹² s⁻¹ (frequência de tentativa)
- exp(-ΔH‡/RT) = fator de Boltzmann
- exp(ΔS‡/R) = fator entrópico
"""
function calculate_eyring_rate(T::Float64, ΔH_act::Float64, ΔS_act::Float64)
    kB = CONSTANTS.kB
    h = CONSTANTS.h
    R = CONSTANTS.R

    # Frequência de tentativa (teoria do estado de transição)
    ν = kB * T / h  # ≈ 6.4e12 s⁻¹ a 310K

    # Fator de Boltzmann (barreira entálpica)
    boltzmann = exp(-ΔH_act / (R * T))

    # Fator entrópico (ordenamento do estado de transição)
    entropy_factor = exp(ΔS_act / R)

    # Taxa intrínseca (s⁻¹)
    k = ν * boltzmann * entropy_factor

    return k
end

# =============================================================================
# EQUILÍBRIO DE PROTONAÇÃO (Brønsted)
# =============================================================================

"""
Calcula a fração de carbonilas protonadas.

Para catálise ácida, a carbonila precisa ser protonada:
C=O + H⁺ ⇌ C=OH⁺

K_prot = [C=OH⁺] / ([C=O] · [H⁺])

A fração protonada é:
f_prot = K_prot · [H⁺] / (1 + K_prot · [H⁺])

Como pKa ≈ -6.5, K_prot = 10^6.5 ≈ 3×10⁶
Mas [H⁺] a pH 7.4 = 4×10⁻⁸ M

f_prot ≈ 3×10⁶ × 4×10⁻⁸ ≈ 0.12 (12% protonado)

A pH local mais baixo (ex: 3.5), f_prot aumenta significativamente.
"""
function calculate_protonation_equilibrium(pH::Float64, pH_local::Float64)
    # pKa da carbonila protonada
    pKa = THERMODYNAMICS.pKa_carbonyl

    # [H⁺] do pH local (dentro do polímero)
    H_conc = 10^(-pH_local)

    # Constante de protonação
    K_prot = 10^(-pKa)  # = 10^6.5

    # Fração protonada
    f_prot = K_prot * H_conc / (1.0 + K_prot * H_conc)

    return f_prot
end

"""
Calcula o pH local dentro do polímero baseado na concentração de COOH.

À medida que a degradação prossegue:
1. Grupos COOH são gerados
2. Eles se dissociam: COOH ⇌ COO⁻ + H⁺ (pKa = 3.86)
3. O pH local cai

[H⁺] = √(Ka × C_COOH) para ácido fraco
"""
function calculate_local_pH(Mn::Float64, Mn0::Float64, pH_bulk::Float64)
    # Concentração de grupos COOH (mol/m³)
    # Cada cadeia tem ~1 COOH terminal + mais gerados por cisão

    # Número de cisões por cadeia original
    n_scissions = max(0.0, Mn0/Mn - 1.0)

    # Concentração de COOH (mol/L)
    # Assumindo ~1 mol COOH por mol de cadeia clivada
    C_COOH = 0.01 * n_scissions  # mol/L (aproximação)

    if C_COOH < 1e-8
        return pH_bulk
    end

    # Dissociação do ácido lático
    Ka = 10^(-THERMODYNAMICS.pKa_lactic)

    # Equação do ácido fraco: [H⁺]² + Ka[H⁺] - Ka·C = 0
    H_acid = (-Ka + sqrt(Ka^2 + 4*Ka*C_COOH)) / 2

    # Combinar com buffer do meio
    H_bulk = 10^(-pH_bulk)
    H_total = H_bulk + H_acid

    pH_local = -log10(H_total)

    # Limite físico-químico realista:
    # 1. O buffer PBS mantém pH mínimo ~3.5 mesmo com ácido em excesso
    # 2. Difusão de H⁺ para fora limita acidificação
    # 3. pKa do ácido lático (3.86) define o equilíbrio máximo
    return max(pH_local, 3.4)  # Não pode ir muito abaixo do pKa
end

# =============================================================================
# DIFUSÃO DE ÁGUA NO POLÍMERO
# =============================================================================

"""
Calcula o fator de limitação difusional.

A água precisa difundir para dentro do polímero para reagir.
O perfil de concentração segue a Lei de Fick.

Para uma esfera/cilindro de raio R:
τ_diff = R² / D

Se τ_diff >> τ_rxn: difusão limita (erosão de superfície)
Se τ_diff << τ_rxn: reação limita (degradação bulk)

Para scaffolds porosos, a difusão é rápida → degradação bulk.
"""
function calculate_diffusion_factor(t::Float64, Xc::Float64,
                                    characteristic_length::Float64=100e-6)
    # Coeficiente de difusão efetivo (média ponderada)
    D_eff = (1 - Xc) * POLYMER.D_water_amorphous +
            Xc * POLYMER.D_water_crystalline

    # Tempo característico de difusão
    τ_diff = characteristic_length^2 / D_eff  # segundos
    τ_diff_days = τ_diff / 86400

    # Fator de saturação (água atinge equilíbrio)
    # Para scaffold poroso, isso é rápido (~horas)
    if t > τ_diff_days
        f_diff = 1.0  # Totalmente saturado
    else
        # Transiente inicial
        f_diff = sqrt(t / τ_diff_days)
    end

    return f_diff
end

# =============================================================================
# CONCENTRAÇÃO DE ÁGUA NO POLÍMERO
# =============================================================================

"""
Calcula a atividade da água no polímero.

A água absorvida depende de:
1. Solubilidade (termodinâmica)
2. Difusão (cinética)
3. Estado do polímero (amorfo vs cristalino)

A atividade efetiva é:
a_water = S × f_amorfo × f_plasticização
"""
function calculate_water_activity(Xc::Float64, extent::Float64, TEC::Float64=0.0)
    # Solubilidade base
    S = POLYMER.S_water_37C  # 2% em massa

    # Água só penetra região amorfa
    f_amorfo = 1.0 - Xc

    # Degradação aumenta hidrofilicidade (mais COOH)
    f_hydrophilic = 1.0 + 0.5 * extent

    # TEC aumenta absorção de água (plastificante hidrofílico)
    f_TEC = 1.0 + 0.15 * TEC

    # Atividade efetiva (normalizada)
    a_water = S * f_amorfo * f_hydrophilic * f_TEC

    # Converter para mol/L
    # 2% em massa ≈ 20 g/L / 18 g/mol ≈ 1.1 mol/L
    C_water = a_water * 55.5  # mol/L (referência: água pura = 55.5 M)

    return C_water
end

# =============================================================================
# CRISTALINIDADE (Avrami)
# =============================================================================

"""
Calcula a fração cristalina usando cinética de Avrami.

Xc(t) = Xc_∞ × (1 - exp(-k_av × t^n))

A cristalização é favorecida por:
1. Temperatura (máximo entre Tg e Tm)
2. Cadeias curtas (maior mobilidade)
3. Tempo (nucleação e crescimento)
"""
function calculate_crystallinity(t::Float64, Mn::Float64, T::Float64)
    # Parâmetros de Avrami
    n = 1.5  # Expoente (crescimento 2D a partir de núcleos preexistentes)

    # Taxa de cristalização depende de Mn
    # Cadeias curtas cristalizam mais rápido
    Mn_ref = 50.0  # kg/mol
    k_av_base = 5e-4  # dia^-n
    k_av = k_av_base * (Mn_ref / max(Mn, 5.0))^0.5

    # Cristalinidade máxima
    Xc_inf = 0.45  # 45% máximo para PLDLA

    # Avrami
    Xc = Xc_inf * (1.0 - exp(-k_av * t^n))

    # Cristalização inicial
    Xc_initial = 0.05

    return Xc_initial + Xc * (1 - Xc_initial)
end

# =============================================================================
# MODELO COMPLETO DE TAXA
# =============================================================================

"""
Taxa de hidrólise baseada em primeiros princípios.

r = k_Eyring × f_prot × C_water × f_amorfo × f_diff × [Éster]

Onde:
- k_Eyring: constante de Eyring (teoria do estado de transição)
- f_prot: fração de carbonilas protonadas (Brønsted)
- C_water: concentração de água no polímero (Lewis nucleophile)
- f_amorfo: fração amorfa acessível
- f_diff: fator de limitação difusional
- [Éster]: concentração de ligações éster
"""
function calculate_hydrolysis_rate(Mn::Float64, Mn0::Float64, t::Float64,
                                   T::Float64, pH_bulk::Float64;
                                   TEC::Float64=0.0,
                                   condition::Symbol=:in_vitro)

    # 1. Constante de Eyring (intrínseca)
    k_eyring = calculate_eyring_rate(T, THERMODYNAMICS.ΔH_act, THERMODYNAMICS.ΔS_act)

    # Converter de s⁻¹ para dia⁻¹
    k_eyring_day = k_eyring * 86400

    # 2. pH local e protonação
    pH_local = calculate_local_pH(Mn, Mn0, pH_bulk)
    f_prot = calculate_protonation_equilibrium(pH_bulk, pH_local)

    # 3. Cristalinidade
    Xc = calculate_crystallinity(t, Mn, T)
    f_amorfo = 1.0 - Xc

    # 4. Concentração de água
    extent = 1.0 - Mn/Mn0
    C_water = calculate_water_activity(Xc, extent, TEC)

    # 5. Fator difusional
    f_diff = calculate_diffusion_factor(t, Xc)

    # 6. Fator in vivo (redução de água acessível)
    if condition == :subcutaneous
        f_vivo = 0.25
    elseif condition == :bone
        f_vivo = 0.15
    else
        f_vivo = 1.0
    end

    # 7. Fator de acessibilidade estérica
    # No polímero sólido, nem todas as ligações éster são acessíveis
    # A fração acessível AUMENTA com a degradação (mais interfaces/poros)
    #
    # Modelo físico: água penetra por microporos e interfaces
    # À medida que cadeias quebram, mais superfície interna é exposta

    # Acessibilidade base depende da massa molar inicial
    # Polímeros de menor Mn têm mais extremidades de cadeia (mais hidrofílicas)
    f_accessibility_base = 5e-3 * (50.0 / max(Mn0, 30.0))  # Normalizado para Mn=50
    f_accessibility = f_accessibility_base * (1.0 + 6.0 * extent)  # Aumenta com degradação

    # 8. Fator de emaranhamento (reptação de cadeias)
    # Cadeias longas têm menor mobilidade segmental
    # Massa molar crítica de emaranhamento para PLA ≈ 9 kg/mol (literatura)
    Mc = 9.0  # kg/mol
    if Mn > 3 * Mc
        f_entanglement = 0.4  # Altamente emaranhado
    elseif Mn > Mc
        f_entanglement = 0.4 + 0.6 * (3*Mc - Mn) / (2*Mc)
    else
        f_entanglement = 1.0  # Abaixo de Mc: sem emaranhamento, alta mobilidade
    end

    f_calibration = f_accessibility * f_entanglement

    # Taxa efetiva
    k_eff = k_eyring_day * f_prot * C_water * f_amorfo * f_diff * f_vivo * f_calibration

    return (k_eff=k_eff, pH_local=pH_local, Xc=Xc, f_prot=f_prot, C_water=C_water)
end

# =============================================================================
# SIMULAÇÃO
# =============================================================================

"""
Simula degradação usando modelo termodinâmico.
"""
function predict_thermodynamic(dataset::String)
    data = DATASETS[dataset]
    Mn0 = data.Mn[1]
    T = data.T
    pH = data.pH
    condition = data.condition
    TEC = haskey(data, :TEC) ? data.TEC : 0.0

    dt = 0.5  # dias
    Mn = Mn0
    t_current = 0.0

    results = Dict{String, Vector{Float64}}(
        "t" => Float64[],
        "Mn" => Float64[],
        "pH_local" => Float64[],
        "Xc" => Float64[],
        "k_eff" => Float64[]
    )

    for t_target in data.t
        while t_current < t_target - dt/2
            rate_info = calculate_hydrolysis_rate(Mn, Mn0, t_current, T, pH,
                                                  TEC=TEC, condition=condition)

            # dMn/dt = -k × Mn (primeira ordem em Mn)
            dMn = -rate_info.k_eff * Mn
            Mn = max(Mn + dt * dMn, 0.5)
            t_current += dt
        end

        rate_info = calculate_hydrolysis_rate(Mn, Mn0, t_current, T, pH,
                                              TEC=TEC, condition=condition)

        push!(results["t"], t_target)
        push!(results["Mn"], Mn)
        push!(results["pH_local"], rate_info.pH_local)
        push!(results["Xc"], rate_info.Xc)
        push!(results["k_eff"], rate_info.k_eff)

        t_current = t_target
    end

    return results
end

# =============================================================================
# VALIDAÇÃO
# =============================================================================

"""
Valida o modelo termodinâmico contra dados experimentais.
"""
function validate_thermodynamic_model()
    println("\n" * "="^80)
    println("       MODELO TERMODINÂMICO DE DEGRADAÇÃO")
    println("       Baseado em Primeiros Princípios")
    println("="^80)

    println("\n┌─────────────────────────────────────────────────────────────────────────────────┐")
    println("│  FUNDAMENTOS TEÓRICOS                                                           │")
    println("├─────────────────────────────────────────────────────────────────────────────────┤")
    println("│                                                                                 │")
    println("│  1. TERMODINÂMICA: ΔG° = -32 kJ/mol (espontâneo)                               │")
    println("│     ΔH° = -12 kJ/mol (exotérmico)                                              │")
    println("│     ΔS° = +65 J/(mol·K) (aumento entropia)                                     │")
    println("│                                                                                 │")
    println("│  2. EYRING: k = (kB·T/h)·exp(-ΔG‡/RT)                                          │")
    println("│     ΔH‡ = 78 kJ/mol (barreira)                                                 │")
    println("│     ΔS‡ = -80 J/(mol·K) (estado transição ordenado)                            │")
    println("│                                                                                 │")
    println("│  3. BRØNSTED: Catálise por H⁺ (pKa carbonila ≈ -6.5)                           │")
    println("│                                                                                 │")
    println("│  4. LEWIS: H₂O como nucleófilo atacando C=O                                    │")
    println("│                                                                                 │")
    println("│  5. FICK: Difusão de água (D ≈ 10⁻¹² m²/s em amorfo)                          │")
    println("│                                                                                 │")
    println("└─────────────────────────────────────────────────────────────────────────────────┘")

    # Mostrar cálculos intermediários para um caso
    println("\n--- Cálculos para PLDLA a t=0, T=37°C ---")

    T = 310.15
    k_eyring = calculate_eyring_rate(T, THERMODYNAMICS.ΔH_act, THERMODYNAMICS.ΔS_act)
    @printf("  k_Eyring = %.2e s⁻¹\n", k_eyring)
    @printf("  k_Eyring = %.2e dia⁻¹\n", k_eyring * 86400)

    f_prot = calculate_protonation_equilibrium(7.4, 7.4)
    @printf("  f_protonação (pH 7.4) = %.2e\n", f_prot)

    f_prot_low = calculate_protonation_equilibrium(7.4, 3.5)
    @printf("  f_protonação (pH 3.5) = %.2e (autocatálise)\n", f_prot_low)

    Xc = calculate_crystallinity(0.0, 50.0, T)
    @printf("  Xc inicial = %.1f%%\n", Xc * 100)

    C_water = calculate_water_activity(Xc, 0.0, 0.0)
    @printf("  C_água no polímero = %.2f mol/L\n", C_water)

    # Validação contra datasets
    results = Dict{String, Float64}()

    for (name, data) in DATASETS
        println("\n--- $name ---")
        println("  Source: $(data.source)")

        pred = predict_thermodynamic(name)

        errors = Float64[]
        println("  ┌─────────┬──────────┬──────────┬─────────┬─────────┬──────────┐")
        println("  │ Time(d) │ Mn_exp   │ Mn_pred  │ pH_loc  │ Xc (%)  │  Error   │")
        println("  ├─────────┼──────────┼──────────┼─────────┼─────────┼──────────┤")

        for i in 1:length(data.t)
            err = abs(pred["Mn"][i] - data.Mn[i]) / data.Mn[i] * 100
            push!(errors, err)
            @printf("  │ %7.0f │ %8.1f │ %8.1f │ %7.2f │ %6.1f%% │ %6.1f%%  │\n",
                    data.t[i], data.Mn[i], pred["Mn"][i],
                    pred["pH_local"][i], pred["Xc"][i]*100, err)
        end
        println("  └─────────┴──────────┴──────────┴─────────┴─────────┴──────────┘")

        mape = mean(errors[2:end])
        @printf("  MAPE: %.1f%%\n", mape)
        results[name] = mape
    end

    # Sumário
    println("\n" * "="^80)
    println("  SUMÁRIO")
    println("="^80)

    println("\n┌────────────────────────────┬────────────┬────────────────────┐")
    println("│ Dataset                    │ MAPE (%)   │ Quality            │")
    println("├────────────────────────────┼────────────┼────────────────────┤")

    for (name, mape) in sort(collect(results), by=x->x[2])
        quality = mape < 15 ? "Excellent" : mape < 25 ? "Good" :
                  mape < 35 ? "Acceptable" : "Needs work"
        @printf("│ %-26s │ %8.1f%% │ %-18s │\n", name, mape, quality)
    end

    global_mape = mean(values(results))
    @printf("├────────────────────────────┼────────────┼────────────────────┤\n")
    @printf("│ %-26s │ %8.1f%% │                    │\n", "GLOBAL MEAN", global_mape)
    println("└────────────────────────────┴────────────┴────────────────────┘")

    # Análise dos parâmetros termodinâmicos
    println("\n" * "="^80)
    println("  ANÁLISE TERMODINÂMICA")
    println("="^80)

    println("\n┌─────────────────────────────────────────────────────────────────────────────────┐")
    println("│  ENERGIA LIVRE DE ATIVAÇÃO                                                      │")
    println("├─────────────────────────────────────────────────────────────────────────────────┤")

    T = 310.15
    ΔG_act = THERMODYNAMICS.ΔH_act - T * THERMODYNAMICS.ΔS_act
    @printf("│  ΔG‡ = ΔH‡ - T·ΔS‡ = %.1f - %.1f × (%.1f) = %.1f kJ/mol            │\n",
            THERMODYNAMICS.ΔH_act/1000, T, THERMODYNAMICS.ΔS_act, ΔG_act/1000)
    println("│                                                                                 │")
    println("│  A barreira é dominada pela entalpia (ΔH‡ >> |T·ΔS‡|)                          │")
    println("│  O estado de transição é ordenado (ΔS‡ < 0) → geometria precisa                │")
    println("└─────────────────────────────────────────────────────────────────────────────────┘")

    println("\n┌─────────────────────────────────────────────────────────────────────────────────┐")
    println("│  POR QUE A REAÇÃO É LENTA APESAR DE ΔG° < 0?                                   │")
    println("├─────────────────────────────────────────────────────────────────────────────────┤")
    println("│                                                                                 │")
    println("│  ΔG° = -32 kJ/mol → Reação TERMODINAMICAMENTE favorável                        │")
    println("│  ΔG‡ = +103 kJ/mol → Barreira CINÉTICA alta                                    │")
    println("│                                                                                 │")
    println("│  A catálise ácida reduz ΔG‡ protonando a carbonila,                            │")
    println("│  tornando-a mais eletrofílica para o ataque nucleofílico.                      │")
    println("│                                                                                 │")
    println("└─────────────────────────────────────────────────────────────────────────────────┘")

    return results
end

end # module
