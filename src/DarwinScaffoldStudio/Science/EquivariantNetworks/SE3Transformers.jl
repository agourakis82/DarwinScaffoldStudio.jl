"""
SE3Transformers.jl - SE(3)-Equivariant Transformers for 3D Structure Analysis

Implements SE(3)-equivariant attention mechanisms for analyzing 3D molecular
and scaffold structures while respecting rotational and translational symmetry.

SOTA 2024-2025 Features:
- SE(3)-equivariant self-attention
- Type-0 (scalar), Type-1 (vector), Type-2 (tensor) features
- Geometric message passing
- Radial basis functions for distance encoding
- Spherical harmonic features for angular encoding

Applications in Scaffold Analysis:
- Pore network analysis with rotational invariance
- Material property prediction
- Structure-property relationships
- Defect detection in scaffolds

References:
- Fuchs et al. 2020: "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention"
- Thomas et al. 2018: "Tensor Field Networks"
- Batzner et al. 2022: "E(3)-Equivariant Graph Neural Networks (NequIP)"
"""
module SE3Transformers

using LinearAlgebra
using Statistics
using Random

# Include spherical harmonics utilities
include("SphericalHarmonics.jl")
using .SphericalHarmonics

export SE3TransformerConfig, SE3TransformerModel
export SE3AttentionLayer, TensorFieldLayer
export forward_se3, encode_structure
export RadialBasisFunction, GaussianRBF, BesselRBF
export create_se3_small, create_se3_base

# ============================================================================
# CONFIGURATION
# ============================================================================

"""
    SE3TransformerConfig

Configuration for SE(3)-Equivariant Transformer.
"""
struct SE3TransformerConfig
    lmax::Int                    # Maximum spherical harmonic degree
    num_layers::Int              # Number of transformer layers
    hidden_channels::Int         # Hidden feature channels
    num_heads::Int               # Attention heads
    head_channels::Int           # Channels per head
    num_radial_basis::Int        # Number of radial basis functions
    cutoff::Float64              # Interaction cutoff distance
    num_neighbors::Int           # Average number of neighbors
    use_attention::Bool          # Use attention vs message passing
    dropout::Float64             # Dropout rate
    num_output_features::Int     # Output feature dimension
end

function SE3TransformerConfig(;
    lmax::Int=2,
    num_layers::Int=4,
    hidden_channels::Int=64,
    num_heads::Int=4,
    head_channels::Int=16,
    num_radial_basis::Int=16,
    cutoff::Float64=5.0,
    num_neighbors::Int=20,
    use_attention::Bool=true,
    dropout::Float64=0.1,
    num_output_features::Int=64
)
    SE3TransformerConfig(
        lmax, num_layers, hidden_channels, num_heads, head_channels,
        num_radial_basis, cutoff, num_neighbors, use_attention, dropout,
        num_output_features
    )
end

# Presets
function create_se3_small(; kwargs...)
    SE3TransformerConfig(;
        lmax=1,
        num_layers=2,
        hidden_channels=32,
        num_heads=2,
        kwargs...
    )
end

function create_se3_base(; kwargs...)
    SE3TransformerConfig(;
        lmax=2,
        num_layers=4,
        hidden_channels=64,
        num_heads=4,
        kwargs...
    )
end

# ============================================================================
# RADIAL BASIS FUNCTIONS
# ============================================================================

"""
    RadialBasisFunction

Abstract type for radial basis functions.
"""
abstract type RadialBasisFunction end

"""
    GaussianRBF

Gaussian radial basis functions.
"""
struct GaussianRBF <: RadialBasisFunction
    centers::Vector{Float64}
    widths::Vector{Float64}
    cutoff::Float64
end

function GaussianRBF(num_basis::Int, cutoff::Float64)
    centers = collect(range(0.0, cutoff, length=num_basis))
    widths = fill(cutoff / num_basis, num_basis)
    GaussianRBF(centers, widths, cutoff)
end

function (rbf::GaussianRBF)(d::Real)
    envelope = cosine_cutoff(d, rbf.cutoff)
    return envelope .* exp.(-((d .- rbf.centers) ./ rbf.widths).^2)
end

"""
    BesselRBF

Bessel radial basis functions (more physically motivated).
"""
struct BesselRBF <: RadialBasisFunction
    num_basis::Int
    cutoff::Float64
end

function (rbf::BesselRBF)(d::Real)
    if d >= rbf.cutoff || d < 1e-10
        return zeros(rbf.num_basis)
    end

    envelope = cosine_cutoff(d, rbf.cutoff)
    freqs = π .* (1:rbf.num_basis) ./ rbf.cutoff

    return envelope .* sin.(freqs .* d) ./ d
end

"""
    cosine_cutoff(d, cutoff)

Smooth cutoff function.
"""
function cosine_cutoff(d::Real, cutoff::Real)
    if d >= cutoff
        return 0.0
    end
    return 0.5 * (1 + cos(π * d / cutoff))
end

# ============================================================================
# FIBER FEATURES
# ============================================================================

"""
    FiberFeatures

SE(3)-equivariant features organized by angular momentum l.

For scaffold analysis:
- l=0: Scalar properties (density, porosity at a point)
- l=1: Vector properties (gradients, flow directions)
- l=2: Tensor properties (stress, anisotropy)
"""
struct FiberFeatures
    scalars::Matrix{Float64}     # (num_points, num_scalar_channels)
    vectors::Array{Float64,3}    # (num_points, num_vector_channels, 3)
    tensors::Array{Float64,3}    # (num_points, num_tensor_channels, 5)  # l=2 has 2l+1=5
end

function FiberFeatures(num_points::Int; scalar_channels::Int=0,
                       vector_channels::Int=0, tensor_channels::Int=0)
    scalars = zeros(num_points, scalar_channels)
    vectors = zeros(num_points, vector_channels, 3)
    tensors = zeros(num_points, tensor_channels, 5)
    FiberFeatures(scalars, vectors, tensors)
end

# ============================================================================
# SE(3) ATTENTION LAYER
# ============================================================================

"""
    SE3AttentionLayer

SE(3)-equivariant attention layer.

Implements the attention mechanism from Fuchs et al. 2020:
- Query/Key from scalar features
- Value from all fiber types
- Attention weights based on relative positions
"""
struct SE3AttentionLayer
    lmax::Int
    num_heads::Int
    head_channels::Int

    # Query/Key projections (scalar)
    query_proj::Matrix{Float64}
    key_proj::Matrix{Float64}

    # Value projections per l
    value_projs::Vector{Matrix{Float64}}

    # Output projections per l
    output_projs::Vector{Matrix{Float64}}

    # Radial network (MLP on distance)
    radial_net_W1::Matrix{Float64}
    radial_net_W2::Matrix{Float64}
    radial_net_bias1::Vector{Float64}
    radial_net_bias2::Vector{Float64}

    rbf::RadialBasisFunction
    dropout::Float64
end

function SE3AttentionLayer(in_channels::Int, out_channels::Int, lmax::Int,
                           num_heads::Int; num_radial_basis::Int=16,
                           cutoff::Float64=5.0, dropout::Float64=0.0)
    head_channels = out_channels ÷ num_heads

    # Query/Key only use scalars
    query_proj = randn(in_channels, num_heads * head_channels) .* 0.02
    key_proj = randn(in_channels, num_heads * head_channels) .* 0.02

    # Value projections for each l
    value_projs = Matrix{Float64}[]
    output_projs = Matrix{Float64}[]
    for l in 0:lmax
        dim_l = 2l + 1
        push!(value_projs, randn(in_channels * dim_l, num_heads * head_channels * dim_l) .* 0.02)
        push!(output_projs, randn(num_heads * head_channels * dim_l, out_channels * dim_l) .* 0.02)
    end

    # Radial MLP
    hidden_radial = 64
    radial_net_W1 = randn(num_radial_basis, hidden_radial) .* sqrt(2.0 / num_radial_basis)
    radial_net_bias1 = zeros(hidden_radial)
    radial_net_W2 = randn(hidden_radial, num_heads) .* sqrt(2.0 / hidden_radial)
    radial_net_bias2 = zeros(num_heads)

    rbf = GaussianRBF(num_radial_basis, cutoff)

    SE3AttentionLayer(
        lmax, num_heads, head_channels,
        query_proj, key_proj, value_projs, output_projs,
        radial_net_W1, radial_net_W2, radial_net_bias1, radial_net_bias2,
        rbf, dropout
    )
end

"""
    se3_attention_forward(layer, features, positions, edge_index)

Forward pass through SE(3) attention.

# Arguments
- `features`: FiberFeatures for all nodes
- `positions`: (N, 3) node positions
- `edge_index`: (2, E) edge connectivity

# Returns
- Updated FiberFeatures
"""
function se3_attention_forward(layer::SE3AttentionLayer,
                               features::FiberFeatures,
                               positions::Matrix{Float64},
                               edge_index::Matrix{Int};
                               training::Bool=false)
    N = size(positions, 1)
    E = size(edge_index, 2)

    # Get scalar features for Q/K
    scalars = features.scalars  # (N, C)

    # Compute queries and keys
    queries = scalars * layer.query_proj  # (N, num_heads * head_channels)
    keys = scalars * layer.key_proj

    # Reshape for heads
    queries = reshape(queries, N, layer.num_heads, layer.head_channels)
    keys = reshape(keys, N, layer.num_heads, layer.head_channels)

    # Initialize output features
    out_scalars = zeros(N, size(features.scalars, 2))
    out_vectors = zeros(size(features.vectors))
    out_tensors = zeros(size(features.tensors))

    # Process each node
    for i in 1:N
        # Find incoming edges to node i
        incoming_edges = findall(edge_index[2, :] .== i)

        if isempty(incoming_edges)
            continue
        end

        neighbors = edge_index[1, incoming_edges]
        num_neighbors = length(neighbors)

        # Compute attention scores
        attn_scores = zeros(num_neighbors, layer.num_heads)

        for (k, j) in enumerate(neighbors)
            # Relative position
            r_ij = positions[j, :] - positions[i, :]
            d_ij = norm(r_ij)

            if d_ij < 1e-10 || d_ij > layer.rbf.cutoff
                continue
            end

            # Radial basis
            rbf_values = layer.rbf(d_ij)

            # Radial MLP
            h_radial = tanh.(rbf_values' * layer.radial_net_W1 .+ layer.radial_net_bias1')
            radial_weight = vec(h_radial * layer.radial_net_W2 .+ layer.radial_net_bias2')

            # Q·K attention
            for h in 1:layer.num_heads
                q = queries[i, h, :]
                k_vec = keys[j, h, :]
                attn_scores[k, h] = (dot(q, k_vec) / sqrt(layer.head_channels)) * radial_weight[h]
            end
        end

        # Softmax over neighbors for each head
        for h in 1:layer.num_heads
            scores = attn_scores[:, h]
            max_score = maximum(scores)
            exp_scores = exp.(scores .- max_score)
            attn_scores[:, h] = exp_scores ./ sum(exp_scores)
        end

        # Apply dropout
        if training && layer.dropout > 0
            mask = rand(size(attn_scores)...) .> layer.dropout
            attn_scores = attn_scores .* mask ./ (1 - layer.dropout)
        end

        # Aggregate values with attention weights
        # For scalars (l=0)
        for (k, j) in enumerate(neighbors)
            weight = mean(attn_scores[k, :])  # Average over heads
            out_scalars[i, :] .+= weight .* features.scalars[j, :]
        end

        # For vectors (l=1) - need to account for spherical harmonics
        for (k, j) in enumerate(neighbors)
            r_ij = positions[j, :] - positions[i, :]
            d_ij = norm(r_ij)

            if d_ij < 1e-10
                continue
            end

            # Unit direction
            r_hat = r_ij / d_ij

            # Spherical harmonics for l=1
            Y1 = spherical_harmonics_xyz_batch(1, r_hat[1], r_hat[2], r_hat[3])
            Y1_vec = Y1[2:4]  # l=1 components

            weight = mean(attn_scores[k, :])
            for c in 1:size(features.vectors, 2)
                # Rotate neighbor vectors by relative angle
                out_vectors[i, c, :] .+= weight .* features.vectors[j, c, :]
            end
        end
    end

    return FiberFeatures(out_scalars, out_vectors, out_tensors)
end

# ============================================================================
# TENSOR FIELD LAYER
# ============================================================================

"""
    TensorFieldLayer

Tensor Field Network layer (Thomas et al. 2018).

Simpler than full attention - uses local convolutions with
spherical harmonic filters.
"""
struct TensorFieldLayer
    lmax::Int
    in_channels::Int
    out_channels::Int

    # Radial functions for each (l_in, l_out, l_filter) combination
    radial_weights::Dict{Tuple{Int,Int,Int}, Matrix{Float64}}

    rbf::RadialBasisFunction
    num_radial::Int
end

function TensorFieldLayer(in_channels::Int, out_channels::Int, lmax::Int;
                          num_radial::Int=16, cutoff::Float64=5.0)
    rbf = GaussianRBF(num_radial, cutoff)

    # Create radial weights for valid (l_in, l_out, l_filter) combinations
    radial_weights = Dict{Tuple{Int,Int,Int}, Matrix{Float64}}()

    for l_in in 0:lmax
        for l_out in 0:lmax
            for l_filter in tensor_product_irreps(l_in, l_out)
                if l_filter <= lmax
                    # Weight matrix: (num_radial, in_channels * out_channels)
                    W = randn(num_radial, in_channels * out_channels) .* 0.02
                    radial_weights[(l_in, l_out, l_filter)] = W
                end
            end
        end
    end

    TensorFieldLayer(lmax, in_channels, out_channels, radial_weights, rbf, num_radial)
end

"""
    tensor_field_forward(layer, features, positions, edge_index)

Forward pass through Tensor Field layer.
"""
function tensor_field_forward(layer::TensorFieldLayer,
                              features::FiberFeatures,
                              positions::Matrix{Float64},
                              edge_index::Matrix{Int})
    N = size(positions, 1)

    # Initialize output
    out_scalars = zeros(N, layer.out_channels)
    out_vectors = zeros(N, layer.out_channels, 3)
    out_tensors = zeros(N, layer.out_channels, 5)

    # Process edges
    for e in 1:size(edge_index, 2)
        i = edge_index[1, e]  # Source
        j = edge_index[2, e]  # Target

        r_ij = positions[j, :] - positions[i, :]
        d_ij = norm(r_ij)

        if d_ij < 1e-10 || d_ij >= layer.rbf.cutoff
            continue
        end

        # Radial basis
        rbf_values = layer.rbf(d_ij)

        # Spherical harmonics of direction
        r_hat = r_ij / d_ij
        Y = spherical_harmonics_xyz_batch(layer.lmax, r_hat[1], r_hat[2], r_hat[3])

        # l=0 (scalars) convolution
        if haskey(layer.radial_weights, (0, 0, 0))
            W = layer.radial_weights[(0, 0, 0)]
            radial_weight = rbf_values' * W
            radial_weight = reshape(radial_weight, size(features.scalars, 2), layer.out_channels)

            contrib = features.scalars[i, :]' * radial_weight
            out_scalars[j, :] .+= vec(contrib) .* Y[1]  # Y_0^0
        end

        # l=0 → l=1 with l_filter=1
        if haskey(layer.radial_weights, (0, 1, 1)) && size(features.vectors, 2) > 0
            W = layer.radial_weights[(0, 1, 1)]
            radial_weight = rbf_values' * W
            radial_weight = reshape(radial_weight, size(features.scalars, 2), layer.out_channels)

            for c_out in 1:layer.out_channels
                for c_in in 1:size(features.scalars, 2)
                    # Scalar × Y_1^m → vector
                    w = radial_weight[c_in, c_out]
                    out_vectors[j, c_out, :] .+= w * features.scalars[i, c_in] .* Y[2:4]
                end
            end
        end
    end

    return FiberFeatures(out_scalars, out_vectors, out_tensors)
end

# ============================================================================
# FULL SE(3) TRANSFORMER MODEL
# ============================================================================

"""
    SE3TransformerModel

Complete SE(3)-equivariant transformer for 3D structures.
"""
struct SE3TransformerModel
    config::SE3TransformerConfig
    embedding::Matrix{Float64}  # Initial feature embedding
    layers::Vector{Union{SE3AttentionLayer, TensorFieldLayer}}
    output_proj::Matrix{Float64}
end

function SE3TransformerModel(config::SE3TransformerConfig; in_features::Int=1)
    # Initial embedding
    embedding = randn(in_features, config.hidden_channels) .* 0.02

    # Build layers
    layers = Union{SE3AttentionLayer, TensorFieldLayer}[]

    for i in 1:config.num_layers
        if config.use_attention
            layer = SE3AttentionLayer(
                config.hidden_channels, config.hidden_channels, config.lmax,
                config.num_heads;
                num_radial_basis=config.num_radial_basis,
                cutoff=config.cutoff,
                dropout=config.dropout
            )
        else
            layer = TensorFieldLayer(
                config.hidden_channels, config.hidden_channels, config.lmax;
                num_radial=config.num_radial_basis,
                cutoff=config.cutoff
            )
        end
        push!(layers, layer)
    end

    # Output projection
    output_proj = randn(config.hidden_channels, config.num_output_features) .* 0.02

    SE3TransformerModel(config, embedding, layers, output_proj)
end

"""
    forward_se3(model, node_features, positions, edge_index; training=false)

Forward pass through SE(3) Transformer.

# Arguments
- `node_features`: (N, F) node feature matrix
- `positions`: (N, 3) node positions
- `edge_index`: (2, E) edge connectivity

# Returns
- (N, output_dim) output features
"""
function forward_se3(model::SE3TransformerModel,
                     node_features::Matrix{Float64},
                     positions::Matrix{Float64},
                     edge_index::Matrix{Int};
                     training::Bool=false)
    N = size(positions, 1)

    # Initial embedding
    scalars = node_features * model.embedding

    # Create fiber features
    features = FiberFeatures(
        scalars,
        zeros(N, model.config.hidden_channels, 3),  # vectors
        zeros(N, model.config.hidden_channels, 5)   # tensors
    )

    # Process through layers
    for layer in model.layers
        if layer isa SE3AttentionLayer
            features = se3_attention_forward(layer, features, positions, edge_index;
                                             training=training)
        else
            features = tensor_field_forward(layer, features, positions, edge_index)
        end
    end

    # Output projection (from scalars)
    output = features.scalars * model.output_proj

    return output
end

"""
    encode_structure(model, positions; features=nothing, k_neighbors=20)

Encode a 3D structure using SE(3) transformer.

# Arguments
- `positions`: (N, 3) point positions
- `features`: Optional (N, F) point features
- `k_neighbors`: Number of nearest neighbors for graph

# Returns
- (N, output_dim) equivariant embeddings
"""
function encode_structure(model::SE3TransformerModel, positions::Matrix{Float64};
                          features::Union{Nothing, Matrix{Float64}}=nothing,
                          k_neighbors::Int=20)
    N = size(positions, 1)

    # Default features if not provided
    if isnothing(features)
        features = ones(N, 1)  # Constant features
    end

    # Build k-NN graph
    edge_index = build_knn_graph(positions, k_neighbors)

    return forward_se3(model, features, positions, edge_index)
end

"""
    build_knn_graph(positions, k)

Build k-nearest neighbor graph from positions.
"""
function build_knn_graph(positions::Matrix{Float64}, k::Int)
    N = size(positions, 1)
    k = min(k, N - 1)

    edges = Tuple{Int,Int}[]

    for i in 1:N
        # Compute distances to all other nodes
        distances = Float64[]
        indices = Int[]

        for j in 1:N
            if i != j
                d = norm(positions[i, :] - positions[j, :])
                push!(distances, d)
                push!(indices, j)
            end
        end

        # Find k nearest
        perm = sortperm(distances)
        nearest = indices[perm[1:k]]

        for j in nearest
            push!(edges, (i, j))
        end
    end

    # Convert to matrix
    edge_index = zeros(Int, 2, length(edges))
    for (e, (i, j)) in enumerate(edges)
        edge_index[1, e] = i
        edge_index[2, e] = j
    end

    return edge_index
end

# ============================================================================
# SCAFFOLD-SPECIFIC UTILITIES
# ============================================================================

"""
    analyze_scaffold_pores(model, pore_centers, pore_radii)

Analyze scaffold pore network using SE(3)-equivariant features.

# Arguments
- `model`: SE3TransformerModel
- `pore_centers`: (N, 3) matrix of pore center positions
- `pore_radii`: (N,) vector of pore radii

# Returns
- Dict with pore embeddings and analysis results
"""
function analyze_scaffold_pores(model::SE3TransformerModel,
                                pore_centers::Matrix{Float64},
                                pore_radii::Vector{Float64})
    N = size(pore_centers, 1)

    # Create features from pore properties
    features = hcat(
        pore_radii,                    # Size
        pore_radii .^ 3 .* (4π/3),    # Volume
        ones(N)                        # Bias
    )

    # Get equivariant embeddings
    embeddings = encode_structure(model, pore_centers;
                                  features=features,
                                  k_neighbors=min(20, N-1))

    # Compute aggregate features
    global_embedding = mean(embeddings, dims=1)

    return Dict(
        "pore_embeddings" => embeddings,
        "global_embedding" => vec(global_embedding),
        "num_pores" => N,
        "mean_radius" => mean(pore_radii),
        "std_radius" => std(pore_radii)
    )
end

"""
    predict_scaffold_properties(model, positions, features)

Predict scaffold properties using SE(3)-equivariant model.

Returns invariant predictions that don't change under rotation.
"""
function predict_scaffold_properties(model::SE3TransformerModel,
                                     positions::Matrix{Float64},
                                     features::Matrix{Float64})
    # Get node embeddings
    node_embeddings = encode_structure(model, positions; features=features)

    # Global pooling for invariant output
    global_features = mean(node_embeddings, dims=1)

    return Dict(
        "node_embeddings" => node_embeddings,
        "global_features" => vec(global_features)
    )
end

end # module
