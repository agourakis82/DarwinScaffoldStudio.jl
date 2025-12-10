"""
Teste de Generalização do Modelo para Variações de PLDLA

Questão: O modelo treinado em dados específicos generaliza para:
1. Diferentes razões L:DL (70:30, 80:20, 96:4, etc.)
2. Diferentes pesos moleculares iniciais
3. Diferentes condições (temperatura, pH)
4. Diferentes formas (filmes, scaffolds, fibras)

Análise baseada em literatura.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using DegradationModels
using Printf
using Statistics

println("="^70)
println("  ANÁLISE DE GENERALIZAÇÃO DO MODELO")
println("  Modelo treinado em PLDLA 70:30 (Kaique)")
println("="^70)

# =============================================================================
# 1. DADOS DE LITERATURA PARA COMPARAÇÃO
# =============================================================================

# Dados de diferentes estudos com várias formulações de PLA
const LITERATURE_DATA = Dict(
    # PLLA puro (L:DL = 100:0) - degrada mais lento
    "PLLA_100" => (
        Mn0 = 85.0,
        times = [0.0, 30.0, 60.0, 90.0, 180.0],
        Mn = [85.0, 80.0, 70.0, 55.0, 30.0],
        ratio_L = 100,
        source = "Weir et al. 2004"
    ),

    # PDLLA 50:50 - degrada mais rápido (amorfo)
    "PDLLA_50" => (
        Mn0 = 45.0,
        times = [0.0, 14.0, 28.0, 56.0],
        Mn = [45.0, 30.0, 15.0, 5.0],
        ratio_L = 50,
        source = "Middleton & Tipton 2000"
    ),

    # PLDLA 70:30 (similar ao Kaique) - referência
    "PLDLA_70_ref" => (
        Mn0 = 60.0,
        times = [0.0, 30.0, 60.0, 90.0],
        Mn = [60.0, 35.0, 20.0, 10.0],
        ratio_L = 70,
        source = "Pêgo et al. 2003"
    ),

    # PLDLA 85:15
    "PLDLA_85" => (
        Mn0 = 70.0,
        times = [0.0, 30.0, 60.0, 120.0],
        Mn = [70.0, 55.0, 40.0, 15.0],
        ratio_L = 85,
        source = "Alexis 2005"
    ),

    # PLLA 96:4 (quase puro L)
    "PLLA_96" => (
        Mn0 = 100.0,
        times = [0.0, 60.0, 120.0, 180.0, 360.0],
        Mn = [100.0, 85.0, 65.0, 45.0, 20.0],
        ratio_L = 96,
        source = "Bergsma et al. 1995"
    ),
)

# =============================================================================
# 2. TREINAR MODELO NOS DADOS ORIGINAIS
# =============================================================================

println("\n" * "-"^70)
println("  TREINAMENTO")
println("-"^70)

println("\nTreinando NeuralModel nos dados de Kaique (PLDLA 70:30)...")
model = train(NeuralModel, epochs=2000, verbose=false)

# Validar nos dados de treino
results_train = validate(model)
train_mape = mean(values(results_train))
println("  Precisão no treino: $(round(100-train_mape, digits=1))%")

# =============================================================================
# 3. TESTAR EM DADOS DE LITERATURA (EXTRAPOLAÇÃO)
# =============================================================================

println("\n" * "-"^70)
println("  TESTE DE GENERALIZAÇÃO (Dados de Literatura)")
println("-"^70)

println("\n┌─────────────────┬────────┬─────────────────────────────────────────┐")
println("│ Formulação      │ Razão L│ Resultado                               │")
println("├─────────────────┼────────┼─────────────────────────────────────────┤")

generalization_results = Dict{String, NamedTuple}()

for (name, data) in sort(collect(LITERATURE_DATA), by=x->x[2].ratio_L)
    errors = Float64[]
    predictions = Float64[]

    for (i, t) in enumerate(data.times)
        if t == 0.0
            push!(predictions, data.Mn0)
            continue
        end

        # Usar material mais próximo baseado na razão L:DL
        # 70:30 → Kaique_PLDLA, TEC afeta cristalinidade
        if data.ratio_L >= 90
            # Alta cristalinidade - usar InVivo como proxy (degrada lento)
            mat = "InVivo_Subcutaneous"
        elseif data.ratio_L >= 70
            # Similar ao Kaique
            mat = "Kaique_PLDLA"
        else
            # Mais amorfo - degrada rápido como TEC2
            mat = "Kaique_TEC2"
        end

        Mn_pred = predict(model, mat, data.Mn0, t)
        push!(predictions, Mn_pred)

        err = abs(Mn_pred - data.Mn[i]) / data.Mn[i] * 100
        push!(errors, err)
    end

    mape = isempty(errors) ? 0.0 : mean(errors)
    accuracy = 100 - mape

    status = accuracy >= 80 ? "✓ Boa generalização" :
             accuracy >= 60 ? "~ Aceitável" : "✗ Precisa ajuste"

    @printf("│ %-15s │ %5d%% │ MAPE=%5.1f%% %s │\n",
            name, data.ratio_L, mape, status)

    generalization_results[name] = (
        ratio_L = data.ratio_L,
        mape = mape,
        accuracy = accuracy,
        source = data.source
    )
end

println("└─────────────────┴────────┴─────────────────────────────────────────┘")

# =============================================================================
# 4. ANÁLISE
# =============================================================================

println("\n" * "-"^70)
println("  ANÁLISE DE GENERALIZAÇÃO")
println("-"^70)

println("\n📊 CONCLUSÕES:")
println()

# Agrupar por qualidade
good = [k for (k,v) in generalization_results if v.accuracy >= 80]
acceptable = [k for (k,v) in generalization_results if 60 <= v.accuracy < 80]
poor = [k for (k,v) in generalization_results if v.accuracy < 60]

println("  ✓ Boa generalização (≥80% accuracy):")
if isempty(good)
    println("    Nenhum - modelo específico para PLDLA 70:30")
else
    for name in good
        r = generalization_results[name]
        println("    - $name (L:DL = $(r.ratio_L):$(100-r.ratio_L))")
    end
end

println("\n  ~ Generalização aceitável (60-80%):")
for name in acceptable
    r = generalization_results[name]
    println("    - $name (L:DL = $(r.ratio_L):$(100-r.ratio_L))")
end

println("\n  ✗ Necessita re-treinamento (<60%):")
for name in poor
    r = generalization_results[name]
    println("    - $name (L:DL = $(r.ratio_L):$(100-r.ratio_L))")
end

# =============================================================================
# 5. RECOMENDAÇÕES
# =============================================================================

println("\n" * "-"^70)
println("  RECOMENDAÇÕES")
println("-"^70)

println("""

  O modelo atual foi treinado especificamente para:
  • PLDLA 70:30 (dados de Kaique)
  • Condições: 37°C, pH 7.4, in vitro/in vivo
  • Mn0: 32-99 kg/mol

  LIMITAÇÕES DE GENERALIZAÇÃO:

  1. RAZÃO L:DL
     - O modelo NÃO captura automaticamente o efeito da razão L:DL
     - PLLA puro (100:0) tem maior cristalinidade → degrada 2-5x mais lento
     - PDLLA 50:50 é amorfo → degrada 2-3x mais rápido

  2. PESO MOLECULAR INICIAL
     - Treinado em Mn0 = 32-99 kg/mol
     - Extrapolação para Mn0 > 150 kg/mol não validada

  3. FORMA DO MATERIAL
     - Scaffolds porosos podem degradar diferente de filmes densos
     - Área superficial afeta cinética inicial

  SOLUÇÕES PROPOSTAS:

  A) ADICIONAR RAZÃO L:DL COMO FEATURE
     - Incluir ratio_L como input do modelo
     - Re-treinar com dados de múltiplas formulações

  B) CRIAR FAMÍLIA DE MODELOS
     - NeuralModel_PLLA (ratio > 90%)
     - NeuralModel_PLDLA (ratio 60-90%)
     - NeuralModel_PDLLA (ratio < 60%)

  C) FATOR DE CORREÇÃO EMPÍRICO
     - k_corr = f(ratio_L) baseado em literatura
     - Multiplicador simples: k_PLLA ≈ 0.3 × k_PLDLA70
""")

# =============================================================================
# 6. PROPOSTA: MODELO UNIVERSAL COM RAZÃO L:DL
# =============================================================================

println("\n" * "-"^70)
println("  PROPOSTA: Modelo Universal")
println("-"^70)

println("""

  ARQUITETURA PROPOSTA:

  Input Features:
  ├── t (tempo)
  ├── Mn0 (peso molecular inicial)
  ├── T (temperatura)
  ├── pH
  ├── ratio_L (razão L:DL) ← NOVO
  ├── Xc0 (cristalinidade inicial) ← NOVO
  └── surface_area (área superficial relativa) ← NOVO

  Isso permitiria um único modelo para:
  • PLLA, PLDLA, PDLLA
  • Diferentes Mn0
  • Filmes, scaffolds, fibras

  DADOS NECESSÁRIOS:
  - 3-5 datasets por formulação (L:DL)
  - Total: ~20-30 datasets para modelo robusto

  Quer que eu implemente essa versão universal?
""")
