"""
Science Module Tests
Tests for Topology, Percolation, and ML modules
"""

using Test
using DarwinScaffoldStudio
using Random

Random.seed!(42)

@testset "Science Module" begin
    @testset "Topology Analysis (KEC Metrics)" begin
        # Create simple connected structure
        scaffold = zeros(Bool, 20, 20, 20)
        scaffold[5:15, 5:15, 5:15] .= true

        # Add a channel
        scaffold[1:20, 9:11, 9:11] .= true

        # Test KEC metrics computation
        kec = compute_kec_metrics(scaffold, 10.0)
        @test haskey(kec, "curvature_mean")
        @test haskey(kec, "entropy_shannon")
        @test haskey(kec, "coherence_spatial")
        @test isa(kec["entropy_shannon"], Number)
    end

    @testset "Percolation Analysis" begin
        # Create percolating structure (connected from one side to other)
        scaffold = zeros(Bool, 20, 20, 20)
        scaffold[8:12, 8:12, 1:20] .= true  # Column through z

        # Invert for pore space
        pores = .!scaffold

        # Test percolation metrics
        perc = compute_percolation_metrics(scaffold, 10.0)
        @test haskey(perc, "percolation_probability")
        @test haskey(perc, "largest_cluster_fraction")
    end

    @testset "Connected Components" begin
        # Create two separate blobs
        scaffold = zeros(Bool, 30, 30, 30)
        scaffold[5:10, 5:10, 5:10] .= true   # Blob 1
        scaffold[20:25, 20:25, 20:25] .= true  # Blob 2 (disconnected)

        # Test metrics computes something reasonable
        metrics = compute_metrics(scaffold, 10.0)
        @test metrics.porosity >= 0.0 && metrics.porosity <= 1.0
    end

    @testset "Pore Size Distribution" begin
        # Create scaffold with varying pore sizes
        scaffold = zeros(Bool, 40, 40, 40)

        # Material with regular holes
        scaffold[5:35, 5:35, 5:35] .= true

        # Small pore
        scaffold[10:12, 10:12, 10:30] .= false

        # Medium pore
        scaffold[20:25, 20:25, 10:30] .= false

        # Large pore
        scaffold[30:35, 10:20, 10:30] .= false

        # Compute metrics
        metrics = compute_metrics(scaffold, 10.0)

        # Should detect pores
        @test metrics.mean_pore_size_um >= 0.0
        @test metrics.interconnectivity >= 0.0 && metrics.interconnectivity <= 1.0
    end
end

println("✅ Science tests passed!")
