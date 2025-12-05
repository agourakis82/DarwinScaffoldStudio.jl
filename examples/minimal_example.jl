#!/usr/bin/env julia
"""
Darwin Scaffold Studio - Minimal Reproducible Example
======================================================

Demonstrates core scaffold analysis without heavy dependencies.

Run with: julia --project=. examples/minimal_example.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Images
using Statistics

println("=" ^ 60)
println("Darwin Scaffold Studio - Minimal Example")
println("=" ^ 60)

println("\n1. Generate synthetic scaffold (Gyroid TPMS)")
println("-" ^ 40)

# Create a 64x64x64 gyroid scaffold
size = 64
volume = zeros(Bool, size, size, size)

for i in 1:size, j in 1:size, k in 1:size
    x = 2π * i / size
    y = 2π * j / size
    z = 2π * k / size
    gyroid = sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)
    volume[i, j, k] = gyroid > 0.3
end

println("   Volume: $(size)x$(size)x$(size) voxels")
println("   Structure: Gyroid TPMS")

println("\n2. Compute metrics")
println("-" ^ 40)

voxel_size_um = 10.0

# Porosity
solid_voxels = sum(volume)
total_voxels = length(volume)
porosity = 1.0 - solid_voxels / total_voxels
println("   Porosity: $(round(porosity * 100, digits=1))%")

# Pore size via connected components
pore_mask = .!volume
labels = label_components(pore_mask)
n_pores = maximum(labels)

pore_sizes = Float64[]
for i in 1:n_pores
    vol = sum(labels .== i)
    # Equivalent spherical diameter
    d = 2 * (3 * vol / (4 * π))^(1/3) * voxel_size_um
    push!(pore_sizes, d)
end
mean_pore_size = isempty(pore_sizes) ? 0.0 : mean(pore_sizes)
println("   Mean pore size: $(round(mean_pore_size, digits=1)) μm (n=$(n_pores) pores)")

# Interconnectivity
labels = label_components(pore_mask)
n_components = maximum(labels)
largest = maximum([sum(labels .== i) for i in 1:n_components])
interconnectivity = largest / sum(pore_mask)
println("   Interconnectivity: $(round(interconnectivity * 100, digits=1))%")

println("\n3. Literature comparison")
println("-" ^ 40)

# Murphy et al. 2010 criteria
println("   Murphy et al. 2010 (bone scaffolds):")
println("   - Optimal porosity: 85-95%")
println("   - Optimal pore size: 100-300 μm")

if 0.85 <= porosity <= 0.95
    println("   → Porosity: OPTIMAL")
else
    println("   → Porosity: outside optimal range")
end

println("\n" * "=" ^ 60)
println("Example completed!")
println("=" ^ 60)
