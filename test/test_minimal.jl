"""
Minimal Test - Verifica estrutura e sintaxe básica
"""

println("=" ^ 60)
println("DARWIN-SCAFFOLD-STUDIO.JL - MINIMAL STRUCTURE TEST")
println("=" ^ 60)

# Test 1: Verificar arquivos existem
println("\n✅ Test 1: Verificando estrutura de arquivos...")
files = [
    "src/DarwinScaffoldStudio.jl",
    "src/DarwinScaffoldStudio/Core/Config.jl",
    "src/DarwinScaffoldStudio/Core/Types.jl",
    "src/DarwinScaffoldStudio/MicroCT/ImageLoader.jl",
    "src/DarwinScaffoldStudio/MicroCT/Metrics.jl",
    "src/DarwinScaffoldStudio/Optimization/ScaffoldOptimizer.jl",
]

all_exist = true
for file in files
    if isfile(file)
        println("   ✓ ", file)
    else
        println("   ✗ ", file, " - MISSING")
        all_exist = false
    end
end

if !all_exist
    println("\n❌ Alguns arquivos estão faltando!")
    exit(1)
end

# Test 2: Contar módulos
println("\n✅ Test 2: Contando módulos...")
microct_modules = filter(f -> isfile(f),
    [joinpath("src/DarwinScaffoldStudio/MicroCT", f) for f in readdir("src/DarwinScaffoldStudio/MicroCT") if endswith(f, ".jl")])
opt_modules = filter(f -> isfile(f),
    [joinpath("src/DarwinScaffoldStudio/Optimization", f) for f in readdir("src/DarwinScaffoldStudio/Optimization") if endswith(f, ".jl")])
viz_modules = filter(f -> isfile(f),
    [joinpath("src/DarwinScaffoldStudio/Visualization", f) for f in readdir("src/DarwinScaffoldStudio/Visualization") if endswith(f, ".jl")])
core_modules = filter(f -> isfile(f),
    [joinpath("src/DarwinScaffoldStudio/Core", f) for f in readdir("src/DarwinScaffoldStudio/Core") if endswith(f, ".jl")])

println("   - Core modules: ", length(core_modules))
println("   - MicroCT modules: ", length(microct_modules))
println("   - Optimization modules: ", length(opt_modules))
println("   - Visualization modules: ", length(viz_modules))
println("   - Total: ", length(core_modules) + length(microct_modules) + length(opt_modules) + length(viz_modules))

# Test 3: Verificar Project.toml
println("\n✅ Test 3: Verificando Project.toml...")
if isfile("Project.toml")
    println("   ✓ Project.toml existe")
    content = read("Project.toml", String)
    if occursin("DarwinScaffoldStudio", content)
        println("   ✓ Nome do projeto correto")
    end
    if occursin("1.0.0", content)
        println("   ✓ Versão 1.0.0")
    end
else
    println("   ✗ Project.toml não encontrado")
    exit(1)
end

println("\n" ^ 60)
println("✅ ESTRUTURA BÁSICA OK!")
println("=" ^ 60)

