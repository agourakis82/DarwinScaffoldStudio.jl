"""
Tests for DarwinScaffoldStudio

Comprehensive test suite.
"""

using Test
using DarwinScaffoldStudio
using Random

Random.seed!(42)

@testset "DarwinScaffoldStudio" begin
    @testset "Image Loading" begin
        # Create mock image
        image = rand(50, 50, 50)
        
        # Test preprocessing
        processed = preprocess_image(image; denoise=true, normalize=true)
        
        @test size(processed) == size(image)
        @test minimum(processed) >= 0.0
        @test maximum(processed) <= 1.0
    end
    
    @testset "Segmentation" begin
        image = rand(50, 50, 50)
        binary = segment_scaffold(image, "otsu")
        
        @test size(binary) == size(image)
        @test eltype(binary) == Bool
    end
    
    @testset "Metrics" begin
        # Create test scaffold
        binary = zeros(Bool, 30, 30, 30)
        binary[10:20, 10:20, 10:20] .= true  # Solid block
        
        metrics = compute_metrics(binary, 10.0)
        
        @test metrics.porosity >= 0.0 && metrics.porosity <= 1.0
        @test metrics.mean_pore_size_um >= 0.0
        @test metrics.interconnectivity >= 0.0 && metrics.interconnectivity <= 1.0
        @test metrics.tortuosity >= 1.0
        @test metrics.elastic_modulus >= 0.0
        @test metrics.yield_strength >= 0.0
    end
    
    @testset "Optimization" begin
        # Create test scaffold
        binary = zeros(Bool, 20, 20, 20)
        binary[5:15, 5:15, 5:15] .= true
        
        optimizer = Optimizer(voxel_size_um=10.0)
        
        target_params = ScaffoldParameters(
            0.92,  # porosity
            150.0,  # pore size
            0.95,   # interconnectivity
            1.1,    # tortuosity
            (2.0, 2.0, 2.0),  # volume mm³
            10.0    # resolution
        )
        
        results = optimize_scaffold(optimizer, binary, target_params)
        
        @test size(results.optimized_volume) == size(binary)
        @test haskey(results.improvement_percent, "porosity")
    end
    
    @testset "Mesh Creation" begin
        binary = zeros(Bool, 10, 10, 10)
        binary[3:7, 3:7, 3:7] .= true
        
        vertices, faces = create_mesh_simple(binary, 10.0)
        
        @test size(vertices, 2) == 3
        @test size(faces, 2) == 3
    end
end

println("✅ DarwinScaffoldStudio tests passed!")

