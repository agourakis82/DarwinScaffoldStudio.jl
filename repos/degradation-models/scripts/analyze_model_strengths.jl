#!/usr/bin/env julia
# Analyze strengths of each model

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using DegradationModels
using Printf

println("="^70)
println("ANÁLISE DOS PONTOS FORTES DE CADA MODELO")
println("="^70)

# Models we have:
println("""

MODELOS DISPONÍVEIS:
====================

1. NeuralModel (95.5% para PLDLA 70:30 Kaique)
   - Melhor para: dados específicos do Kaique
   - Limitação: não generaliza para outras condições

2. PLDLA3DPrintModelV2 (87.3% global)
   - Melhor para: PLDLA 70:30 3D-printed, com TEC
   - Física: k, n, autocatálise
   - Limitação: só in vitro 37°C

3. UniversalModelV3 (physics-informed)
   - Melhor para: diferentes L:DL ratios
   - Física: cinética baseada em literatura
   - Limitação: baixa acurácia em PLLA cristalino

4. Database de Literatura
   - Melhor para: dados in vivo, diferentes temperaturas
   - Informação: Bergsma (in vivo), Tsuji (temperatura)

PROPOSTA DE MODELO HÍBRIDO:
===========================

PLDLAHybridModel combina:
├── Base: PLDLA3DPrintModelV2 (k, n, autocatálise)
├── Temperatura: Arrhenius (Ea da literatura)
├── In vivo: fator enzimático (1.3-1.5x)
├── Região do corpo: temperaturas específicas
└── Incerteza: propagação de erro

Temperaturas por região:
- Pele superficial: 32-34°C
- Subcutâneo: 35-36°C
- Músculo/osso: 37°C (referência)
- Fígado/rim: 37-38°C
- Inflamação local: 38-40°C

""")
