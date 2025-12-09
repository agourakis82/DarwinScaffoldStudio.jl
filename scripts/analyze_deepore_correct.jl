#!/usr/bin/env julia
"""
Análise CORRETA do DeePore Dataset
===================================

Propriedades (índices 1-15):
1. Absolute Permeability
2. Formation Factor  
3. Cementation Factor
4. Pore density (relacionado a porosidade!)
5. Tortuosity
6. Average Coordination Number
7. Average Throat Radius
8. Average Pore Radius
9. Average Throat Length
10. Pore to Throat Aspect ratio
11. Specific Surface
12. Grain Sphericity
13. Pore Sphericity (corrigido)
14. Average Grain Radius
15. Relative Young Module

O dataset X contém as imagens 3D: (3, 128, 128, 17700)
- 3 slices ortogonais de 128x128 para cada uma das 17700 amostras
"""

using HDF5
using Statistics

const φ = (1 + sqrt(5)) / 2

println("="^80)
println("  ANÁLISE CORRETA DO DEEPORE - PROPRIEDADES FÍSICAS")
println("="^80)

h5path = "data/deepore/DeePore_Compact_Data.h5"
h5file = h5open(h5path, "r")

# Ler Y: shape (1, 1515, 17700)
Y_raw = read(h5file["Y"])
n_samples = size(Y_raw, 3)
n_props = size(Y_raw, 2)

println("\n📊 Dataset: ", n_samples, " amostras, ", n_props, " propriedades cada")

# Extrair propriedades principais (índices 1-15)
prop_names = [
    "Abs. Permeability",
    "Formation Factor",
    "Cementation Factor",
    "Pore Density",
    "Tortuosity",
    "Avg Coord. Number",
    "Avg Throat Radius",
    "Avg Pore Radius",
    "Avg Throat Length",
    "Pore/Throat Aspect",
    "Specific Surface",
    "Grain Sphericity",
    "Pore Sphericity",
    "Avg Grain Radius",
    "Rel. Young Module"
]

println("\n📐 Propriedades Principais (1-15):")
println("-"^70)

props = zeros(n_samples, 15)
for i in 1:n_samples
    for j in 1:15
        props[i, j] = Y_raw[1, j, i]
    end
end

for j in 1:15
    col = props[:, j]
    valid = filter(x -> isfinite(x), col)
    if length(valid) > 0
        println("  ", j, ". ", prop_names[j])
        println("     Range: ", round(minimum(valid), digits=4), " - ", round(maximum(valid), digits=4))
        println("     Mean:  ", round(mean(valid), digits=4))
    else
        println("  ", j, ". ", prop_names[j], " - NO VALID DATA")
    end
end

# Pore Density (índice 4) - mais próximo de porosidade
pore_density = props[:, 4]
valid_pd = filter(x -> isfinite(x), pore_density)

println("\n" * "="^80)
println("  ANÁLISE DE PORE DENSITY (proxy para porosidade)")
println("="^80)

println("\n📈 Estatísticas:")
println("   Valid samples: ", length(valid_pd), " / ", n_samples)
println("   Min:  ", round(minimum(valid_pd), digits=6))
println("   Max:  ", round(maximum(valid_pd), digits=6))
println("   Mean: ", round(mean(valid_pd), digits=6))

# Tortuosity (índice 5) - importante para nossa análise
tortuosity = props[:, 5]
valid_tau = filter(x -> isfinite(x), tortuosity)

println("\n📈 Tortuosidade:")
println("   Valid samples: ", length(valid_tau), " / ", n_samples)
println("   Min:  ", round(minimum(valid_tau), digits=4))
println("   Max:  ", round(maximum(valid_tau), digits=4))
println("   Mean: ", round(mean(valid_tau), digits=4))

# Analisar imagens X para calcular porosidade diretamente
println("\n" * "="^80)
println("  CALCULANDO POROSIDADE DAS IMAGENS")
println("="^80)

X = read(h5file["X"])
println("\nDataset X shape: ", size(X))
println("   3 slices × 128×128 × 17700 amostras")

# Calcular porosidade de cada amostra (fração de pixels escuros)
# Assumindo: 0 = poro, 1 = sólido (ou vice-versa)
println("\n📊 Calculando porosidade para cada amostra...")

porosities = zeros(n_samples)
for i in 1:n_samples
    # Pegar os 3 slices e calcular média
    slices = X[:, :, :, i]  # 3 x 128 x 128
    # Assumindo que valores baixos = poro
    # Normalizar para 0-1 se necessário
    vals = vec(slices)
    if maximum(vals) > 1
        vals = vals ./ maximum(vals)
    end
    # Porosidade = fração de valores < 0.5 (assumindo poros são escuros)
    porosities[i] = mean(vals .< 0.5)
end

println("\n📈 Estatísticas de Porosidade Calculada:")
println("   Min:  ", round(minimum(porosities) * 100, digits=2), "%")
println("   Max:  ", round(maximum(porosities) * 100, digits=2), "%")
println("   Mean: ", round(mean(porosities) * 100, digits=2), "%")

# Histograma
println("\n📊 Distribuição de Porosidade:")
bins = 0:0.1:1.0
for i in 1:length(bins)-1
    count = sum((porosities .>= bins[i]) .& (porosities .< bins[i+1]))
    pct = round(count / n_samples * 100, digits=1)
    bar = repeat("█", Int(ceil(pct / 2)))
    println("   ", Int(bins[i] * 100), "-", Int(bins[i+1] * 100), "%: ", count, " (", pct, "%) ", bar)
end

# Alta porosidade
high_p = sum(porosities .> 0.85)
very_high_p = sum(porosities .> 0.90)
ultra_high_p = sum(porosities .> 0.95)

println("\n🎯 Amostras de ALTA POROSIDADE:")
println("   p > 85%: ", high_p, " amostras")
println("   p > 90%: ", very_high_p, " amostras")
println("   p > 95%: ", ultra_high_p, " amostras")

if very_high_p > 0
    println("\n📋 Índices das amostras com p > 90%:")
    high_idx = findall(porosities .> 0.90)
    println("   ", high_idx[1:min(30, length(high_idx))])
end

close(h5file)

# Salvar resultados
println("\n💾 Salvando porosidades calculadas...")
open("results/deepore_porosities.csv", "w") do f
    println(f, "sample_idx,porosity,tortuosity,pore_density")
    for i in 1:n_samples
        tau = isfinite(tortuosity[i]) ? tortuosity[i] : ""
        pd = isfinite(pore_density[i]) ? pore_density[i] : ""
        println(f, i, ",", porosities[i], ",", tau, ",", pd)
    end
end
println("   Salvo em results/deepore_porosities.csv")

println("\n" * "="^80)
println("  CONCLUSÃO")
println("="^80)

if very_high_p > 0
    println("\n✅ Encontramos ", very_high_p, " amostras com porosidade > 90%")
    println("   Podemos usar para validar D = φ!")
else
    println("\n⚠️ DeePore contém principalmente rochas de baixa/média porosidade")
    println("   Precisamos de outros datasets para alta porosidade (>90%)")
end
