"""
Core Module Tests
Tests for Types, Config, Utils, and ErrorHandling
"""

using Test
using DarwinScaffoldStudio

@testset "Core Module" begin
    @testset "ScaffoldMetrics Type" begin
        # ScaffoldMetrics uses positional arguments (8 fields):
        # porosity, mean_pore_size_um, interconnectivity, tortuosity,
        # specific_surface_area, elastic_modulus, yield_strength, permeability
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
        @test metrics.mean_pore_size_um == 150.0
        @test metrics.interconnectivity == 0.92
        @test metrics.tortuosity == 1.15
    end

    @testset "ScaffoldParameters Type" begin
        params = ScaffoldParameters(
            0.90,           # porosity_target
            200.0,          # pore_size_target_um
            0.95,           # interconnectivity_target
            1.1,            # tortuosity_target
            (5.0, 5.0, 5.0),  # volume_mm3
            10.0            # resolution_um
        )

        @test params.porosity_target == 0.90
        @test params.pore_size_target_um == 200.0
        @test params.volume_mm3 == (5.0, 5.0, 5.0)
    end

    @testset "GlobalConfig" begin
        config = get_config()

        @test haskey(config, :enable_gpu)
        @test haskey(config, :default_voxel_size)
    end
end

println("✅ Core tests passed!")
