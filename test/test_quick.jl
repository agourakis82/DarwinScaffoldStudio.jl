"""
Quick Test Suite for CI/CD
Runs fast tests without loading heavy dependencies.

Run with: julia --project=. test/test_quick.jl
"""

using Test

println("=" ^ 50)
println("Darwin Scaffold Studio - Quick Tests")
println("=" ^ 50)

@testset "Quick Tests" begin
    @testset "Module Structure" begin
        # Test that main module file exists and is valid Julia
        main_file = joinpath(@__DIR__, "..", "src", "DarwinScaffoldStudio.jl")
        @test isfile(main_file)

        # Parse without executing (syntax check)
        content = read(main_file, String)
        @test occursin("module DarwinScaffoldStudio", content)
        @test occursin("export", content)
    end

    @testset "Core Types" begin
        # Test types can be included standalone
        include("../src/DarwinScaffoldStudio/Core/Types.jl")
        using .Types

        # Create ScaffoldMetrics (8 positional fields)
        metrics = ScaffoldMetrics(
            0.85,    # porosity
            150.0,   # mean_pore_size_um
            0.92,    # interconnectivity
            1.15,    # tortuosity
            10.0,    # specific_surface_area
            50.0,    # elastic_modulus
            2.0,     # yield_strength
            1e-10    # permeability
        )

        @test metrics.porosity == 0.85
        @test metrics.interconnectivity == 0.92

        # Create ScaffoldParameters (6 positional fields)
        params = ScaffoldParameters(0.9, 200.0, 0.95, 1.1, (5.0, 5.0, 5.0), 10.0)
        @test params.porosity_target == 0.9
    end

    @testset "TPMS Functions" begin
        # Test TPMS implicit functions
        gyroid(x, y, z) = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)
        diamond(x, y, z) = sin(x)*sin(y)*sin(z) + sin(x)*cos(y)*cos(z) + cos(x)*sin(y)*cos(z) + cos(x)*cos(y)*sin(z)
        schwarz_p(x, y, z) = cos(x) + cos(y) + cos(z)

        # Test at origin
        @test gyroid(0, 0, 0) == 0.0
        @test schwarz_p(0, 0, 0) == 3.0

        # Test symmetry
        @test abs(gyroid(π, π, π) - gyroid(-π, -π, -π)) < 1e-10
    end

    @testset "Data Files" begin
        # Check validation data exists
        data_dir = joinpath(@__DIR__, "..", "data", "validation", "synthetic")
        @test isdir(data_dir)

        # Check at least one scaffold exists
        raw_files = filter(f -> endswith(f, ".raw"), readdir(data_dir))
        @test length(raw_files) >= 1

        json_files = filter(f -> endswith(f, ".json"), readdir(data_dir))
        @test length(json_files) >= 1
    end

    @testset "Ontology Files" begin
        # Check ontology files exist
        onto_dir = joinpath(@__DIR__, "..", "src", "DarwinScaffoldStudio", "Ontology")
        @test isdir(onto_dir)

        required_files = [
            "OBOFoundry.jl",
            "TissueLibrary.jl",
            "CellLibrary.jl",
            "MaterialLibrary.jl",
            "OntologyManager.jl"
        ]

        for file in required_files
            @test isfile(joinpath(onto_dir, file))
        end
    end

    @testset "Scripts" begin
        scripts_dir = joinpath(@__DIR__, "..", "scripts")
        @test isdir(scripts_dir)

        # Check key scripts exist
        @test isfile(joinpath(scripts_dir, "download_sample_data.sh"))
        @test isfile(joinpath(scripts_dir, "setup_llm.sh"))
    end
end

println()
println("✅ Quick tests passed!")
println("=" ^ 50)
