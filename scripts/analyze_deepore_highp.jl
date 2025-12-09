#!/usr/bin/env julia
"""
Análise do DeePore Dataset - Buscando Alta Porosidade para validar D = φ
"""

using HDF5
using Statistics

println("="^80)
println("  ANÁLISE DO DEEPORE DATASET - BUSCANDO ALTA POROSIDADE")
println("="^80)

# Abrir arquivo
h5path = "data/deepore/DeePore_Compact_Data.h5"
h5file = h5open(h5path, "r")

println("\n📁 Estrutura do arquivo HDF5:")
for name in keys(h5file)
    obj = h5file[name]
    if isa(obj, HDF5.Dataset)
        data = read(obj)
        s = size(data)
        println("  Dataset: ", name, " -> shape ", s)
    else
        println("  Group: ", name)
    end
end

# Ler dados
println("\n📊 Carregando dados...")

# Dataset Y contém as propriedades (incluindo porosidade)
# Shape é (1, 1515, 17700) - 1515 propriedades para 17700 amostras
Y_raw = read(h5file["Y"])
println("Dataset Y shape: ", size(Y_raw))

# Reorganizar para (n_samples, n_props)
if ndims(Y_raw) == 3
    n_samples = size(Y_raw, 3)
    n_props = size(Y_raw, 2)
    println("   Amostras: ", n_samples)
    println("   Propriedades por amostra: ", n_props)

    # Extrair porosidade (primeira propriedade de cada amostra)
    porosity = [Y_raw[1, 1, i] for i in 1:n_samples]
    Y = reshape(Y_raw[1, :, :], n_props, n_samples)'  # (n_samples, n_props)
else
    Y = Y_raw
    porosity = Y[:, 1]
end

println("\n📈 Estatísticas de Porosidade:")
println("   Total amostras: ", length(porosity))
println("   Min:  ", round(minimum(porosity) * 100, digits=2), "%")
println("   Max:  ", round(maximum(porosity) * 100, digits=2), "%")
println("   Mean: ", round(mean(porosity) * 100, digits=2), "%")
println("   Std:  ", round(std(porosity) * 100, digits=2), "%")

# Histograma simples
println("\n📊 Distribuição de Porosidade:")
bins = 0:0.1:1.0
for i in 1:length(bins)-1
    count = sum((porosity .>= bins[i]) .& (porosity .< bins[i+1]))
    pct = round(count / length(porosity) * 100, digits=1)
    bar = repeat("█", Int(ceil(pct / 2)))
    println("   ", Int(bins[i] * 100), "-", Int(bins[i+1] * 100), "%: ", count, " (", pct, "%) ", bar)
end

# Contar amostras com alta porosidade
high_p = sum(porosity .> 0.85)
very_high_p = sum(porosity .> 0.90)
ultra_high_p = sum(porosity .> 0.95)

println("\n🎯 Amostras de ALTA POROSIDADE (target para D = φ):")
println("   p > 85%: ", high_p, " amostras")
println("   p > 90%: ", very_high_p, " amostras")
println("   p > 95%: ", ultra_high_p, " amostras")

# Mostrar algumas amostras de alta porosidade
if very_high_p > 0
    println("\n📋 Amostras com porosidade > 90%:")
    high_idx = findall(porosity .> 0.90)
    for (j, i) in enumerate(high_idx[1:min(20, length(high_idx))])
        println("   [", j, "] Amostra ", i, ": p = ", round(porosity[i] * 100, digits=2), "%")
    end
elseif high_p > 0
    println("\n📋 Amostras com porosidade > 85%:")
    high_idx = findall(porosity .> 0.85)
    for (j, i) in enumerate(high_idx[1:min(20, length(high_idx))])
        println("   [", j, "] Amostra ", i, ": p = ", round(porosity[i] * 100, digits=2), "%")
    end
end

# Verificar estrutura completa do Y
println("\n📐 Propriedades disponíveis no Y (", size(Y, 2), " colunas):")
println("   Coluna 1 (porosidade?): range ", round(minimum(Y[:, 1]), digits=3), " - ", round(maximum(Y[:, 1]), digits=3))
if size(Y, 2) > 1
    for col in 2:min(10, size(Y, 2))
        println("   Coluna ", col, ": range ", round(minimum(Y[:, col]), digits=3), " - ", round(maximum(Y[:, col]), digits=3))
    end
end

close(h5file)

println("\n" * "="^80)
println("  CONCLUSÃO")
println("="^80)

if ultra_high_p > 0
    println("\n✅ PERFEITO! Temos ", ultra_high_p, " amostras com porosidade > 95%")
    println("   Isso é IDEAL para validar D = φ!")
elseif very_high_p > 0
    println("\n✅ EXCELENTE! Temos ", very_high_p, " amostras com porosidade > 90%")
    println("   Isso é suficiente para validar D = φ em alta porosidade!")
elseif high_p > 0
    println("\n⚠️  Temos ", high_p, " amostras com porosidade > 85%")
    println("   Útil, mas idealmente precisamos > 90%")
else
    println("\n❌ Não há amostras com porosidade > 85% neste dataset")
    println("   Precisamos buscar outros datasets")
end
