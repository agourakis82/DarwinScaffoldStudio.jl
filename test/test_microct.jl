"""
MicroCT Module Tests
Tests for image loading, preprocessing, segmentation, and metrics
"""

using Test
using DarwinScaffoldStudio
using Random

Random.seed!(42)

@testset "MicroCT Module" begin
    @testset "Image Preprocessing" begin
        # Create synthetic noisy image
        image = rand(Float64, 50, 50, 50)

        # Test preprocessing pipeline
        processed = preprocess_image(image; denoise=true, normalize=true)

        @test size(processed) == size(image)
        @test minimum(processed) >= 0.0
        @test maximum(processed) <= 1.0
        @test eltype(processed) <: AbstractFloat
    end

    @testset "Segmentation Methods" begin
        # Create bimodal image (simulating scaffold)
        image = zeros(Float64, 50, 50, 50)
        image[10:40, 10:40, 10:40] .= 0.2  # Background (pores)
        image[15:35, 15:35, 15:35] .= 0.8  # Foreground (material)
        image .+= 0.05 * rand(50, 50, 50)  # Add noise

        # Test Otsu segmentation
        binary = segment_scaffold(image, "otsu")

        @test size(binary) == size(image)
        @test eltype(binary) == Bool
        @test sum(binary) > 0  # Should have some material
        @test sum(binary) < length(binary)  # Should have some pores
    end

    @testset "Porosity Computation" begin
        # Create scaffold with known porosity
        scaffold = zeros(Bool, 100, 100, 100)
        # Fill 50% with material
        scaffold[1:50, :, :] .= true

        metrics = compute_metrics(scaffold, 10.0)

        # Porosity should be ~50%
        @test abs(metrics.porosity - 0.5) < 0.01
    end

    @testset "Specific Surface Area" begin
        # Create simple cube
        scaffold = zeros(Bool, 30, 30, 30)
        scaffold[10:20, 10:20, 10:20] .= true

        metrics = compute_metrics(scaffold, 10.0)

        # Should have positive specific surface area
        @test metrics.specific_surface_area >= 0.0
    end

    @testset "Interconnectivity" begin
        # Create connected structure
        scaffold = zeros(Bool, 30, 30, 30)
        scaffold[5:25, 5:25, 5:25] .= true  # Main block
        scaffold[1:30, 14:16, 14:16] .= true  # Channel through

        metrics = compute_metrics(scaffold, 10.0)

        @test metrics.interconnectivity >= 0.0
        @test metrics.interconnectivity <= 1.0
    end

    @testset "Tortuosity" begin
        scaffold = zeros(Bool, 30, 30, 30)
        scaffold[5:25, 5:25, 5:25] .= true

        metrics = compute_metrics(scaffold, 10.0)

        # Tortuosity >= 1.0 (straight path = 1.0)
        @test metrics.tortuosity >= 1.0
    end

    @testset "Mechanical Properties (Gibson-Ashby)" begin
        # Create scaffold with ~85% porosity
        scaffold = falses(100, 100, 100)
        # Add sparse material (15% solid)
        for i in 1:100, j in 1:100, k in 1:100
            if (i + j + k) % 7 == 0
                scaffold[i,j,k] = true
            end
        end

        metrics = compute_metrics(scaffold, 10.0)

        # Mechanical properties should be positive
        @test metrics.elastic_modulus > 0.0
        @test metrics.yield_strength > 0.0
    end
end

println("✅ MicroCT tests passed!")
