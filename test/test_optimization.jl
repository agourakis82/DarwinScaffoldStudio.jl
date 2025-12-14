"""
Optimization Module Tests
Tests for scaffold optimization and parametric design
"""

using Test
using DarwinScaffoldStudio
using Random

Random.seed!(42)

@testset "Optimization Module" begin
    @testset "ScaffoldOptimizer Creation" begin
        optimizer = Optimizer(voxel_size_um=10.0)
        @test optimizer !== nothing
    end

    @testset "Basic Optimization" begin
        # Create simple scaffold
        scaffold = zeros(Bool, 30, 30, 30)
        scaffold[5:25, 5:25, 5:25] .= true

        optimizer = Optimizer(voxel_size_um=10.0)

        target = ScaffoldParameters(
            0.85,           # target_porosity
            150.0,          # target_pore_size_um
            0.90,           # target_interconnectivity
            1.2,            # target_tortuosity
            (3.0, 3.0, 3.0), # volume_mm
            10.0            # resolution_um
        )

        results = optimize_scaffold(optimizer, scaffold, target)

        @test size(results.optimized_volume) == size(scaffold)
        @test haskey(results.improvement_percent, "porosity")
        @test results.optimized_metrics !== nothing
    end

    @testset "Porosity Optimization" begin
        # Start with solid block (0% porosity)
        scaffold = ones(Bool, 30, 30, 30)

        optimizer = Optimizer(voxel_size_um=10.0)

        target = ScaffoldParameters(
            0.70,           # target 70% porosity
            100.0,
            0.85,
            1.3,
            (3.0, 3.0, 3.0),
            10.0
        )

        results = optimize_scaffold(optimizer, scaffold, target)

        # Optimized should have higher porosity than original
        original_porosity = 1.0 - sum(scaffold) / length(scaffold)
        optimized_porosity = 1.0 - sum(results.optimized_volume) / length(results.optimized_volume)

        @test optimized_porosity > original_porosity
    end

    @testset "Optimization Constraints" begin
        scaffold = zeros(Bool, 20, 20, 20)
        scaffold[5:15, 5:15, 5:15] .= true

        optimizer = Optimizer(voxel_size_um=10.0)

        # Target with extreme values
        target = ScaffoldParameters(
            0.99,           # Very high porosity
            500.0,          # Large pores
            0.99,           # Very high interconnectivity
            1.0,            # Minimum tortuosity
            (2.0, 2.0, 2.0),
            10.0
        )

        results = optimize_scaffold(optimizer, scaffold, target)

        # Should still produce valid output
        @test size(results.optimized_volume) == size(scaffold)
        @test sum(results.optimized_volume) >= 0
    end

    @testset "Improvement Tracking" begin
        scaffold = ones(Bool, 25, 25, 25)
        # Add some pores
        scaffold[10:15, 10:15, 10:15] .= false

        optimizer = Optimizer(voxel_size_um=10.0)

        target = ScaffoldParameters(
            0.80,
            120.0,
            0.90,
            1.15,
            (2.5, 2.5, 2.5),
            10.0
        )

        results = optimize_scaffold(optimizer, scaffold, target)

        # Should track improvements
        @test haskey(results.improvement_percent, "porosity")
        @test isa(results.improvement_percent["porosity"], Number)
    end
end

println("✅ Optimization tests passed!")
