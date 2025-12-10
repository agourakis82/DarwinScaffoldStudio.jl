"""
Test suite for DegradationModels

Run with:
    julia --project=. test/runtests.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Test
using DegradationModels

@testset "DegradationModels" begin

    @testset "Data" begin
        @test haskey(EXPERIMENTAL_DATA, "Kaique_PLDLA")
        @test haskey(EXPERIMENTAL_DATA, "InVivo_Subcutaneous")
        @test length(EXPERIMENTAL_DATA) == 4

        data = EXPERIMENTAL_DATA["Kaique_PLDLA"]
        @test data.Mn0 ≈ 51.3
        @test length(data.times) == 4
        @test data.times[1] ≈ 0.0
    end

    @testset "NeuralModel" begin
        # Quick training with few epochs
        model = train(NeuralModel, epochs=100, verbose=false)

        # Test prediction at t=0 returns Mn0
        Mn = predict(model, "Kaique_PLDLA", 51.3, 0.0)
        @test Mn ≈ 51.3

        # Test prediction decreases over time
        Mn30 = predict(model, "Kaique_PLDLA", 51.3, 30.0)
        Mn60 = predict(model, "Kaique_PLDLA", 51.3, 60.0)
        @test Mn30 < 51.3
        @test Mn60 < Mn30

        # Test bounds
        Mn90 = predict(model, "Kaique_PLDLA", 51.3, 90.0)
        @test Mn90 > 0.0
        @test Mn90 <= 51.3
    end

    @testset "BronstedModel" begin
        model = train(BronstedModel, verbose=false)

        Mn = predict(model, "Kaique_PLDLA", 51.3, 0.0)
        @test Mn ≈ 51.3

        Mn30 = predict(model, "Kaique_PLDLA", 51.3, 30.0)
        @test Mn30 < 51.3
        @test Mn30 > 0.0
    end

    @testset "ThermodynamicModel" begin
        model = train(ThermodynamicModel, verbose=false)

        Mn = predict(model, "Kaique_PLDLA", 51.3, 0.0)
        @test Mn ≈ 51.3

        Mn30 = predict(model, "Kaique_PLDLA", 51.3, 30.0)
        @test Mn30 < 51.3
    end

    @testset "ConservativeModel" begin
        model = train(ConservativeModel, verbose=false)

        Mn = predict(model, "Kaique_PLDLA", 51.3, 0.0)
        @test Mn ≈ 51.3

        # Test material-specific behavior
        Mn_pldla = predict(model, "Kaique_PLDLA", 51.3, 30.0)
        Mn_invivo = predict(model, "InVivo_Subcutaneous", 99.0, 30.0)

        # In vivo degrades slower (lower k)
        frac_pldla = Mn_pldla / 51.3
        frac_invivo = Mn_invivo / 99.0
        @test frac_invivo > frac_pldla
    end

    @testset "Validation" begin
        model = train(NeuralModel, epochs=500, verbose=false)
        results = validate(model)

        @test haskey(results, "Kaique_PLDLA")
        @test all(v -> v >= 0, values(results))
        @test all(v -> v <= 100, values(results))
    end

    @testset "Material IDs" begin
        model = train(ConservativeModel, verbose=false)

        # Test integer material IDs
        Mn1 = predict(model, 1, 51.3, 30.0)  # Kaique_PLDLA
        Mn2 = predict(model, "Kaique_PLDLA", 51.3, 30.0)

        @test Mn1 ≈ Mn2
    end

    @testset "Physical Bounds" begin
        model = train(NeuralModel, epochs=100, verbose=false)

        # Very long time should not go negative
        Mn = predict(model, "Kaique_PLDLA", 51.3, 365.0)
        @test Mn >= 0.5

        # Should not exceed initial
        Mn0 = predict(model, "Kaique_PLDLA", 51.3, 1.0)
        @test Mn0 <= 51.3
    end

end

println("\n✓ All tests passed!")
