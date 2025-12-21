using DarwinScaffoldStudio
using DarwinScaffoldStudio.Demetrios.CompilerBridge: run_demetrios_json

program = joinpath(@__DIR__, "darwin_json_gibson_ashby.d")
input = Dict("porosity" => 0.75, "E_solid" => 1000.0)

result = run_demetrios_json(program, input)
println(result)
