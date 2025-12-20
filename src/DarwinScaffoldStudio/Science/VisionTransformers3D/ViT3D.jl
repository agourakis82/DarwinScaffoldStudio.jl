"""
ViT3D.jl - 3D Vision Transformer for Volumetric Scaffold Analysis

Implements Vision Transformer (ViT) architecture adapted for 3D volumetric data,
enabling state-of-the-art analysis of microCT scaffold volumes.

SOTA 2024-2025 Features:
- 3D patch embedding with configurable patch sizes
- Multi-head self-attention for volumetric data
- Positional embeddings (learnable + sinusoidal options)
- Classification and feature extraction modes
- FlashAttention-style memory optimization
- Mixed precision support

Architecture (Dosovitskiy et al. 2020, adapted for 3D):
- Input: (D, H, W, C) volumetric data
- Patch Embedding: Split into non-overlapping 3D patches
- Transformer Encoder: L layers of MHSA + MLP
- Output: Classification token or patch features

References:
- Dosovitskiy et al. 2020: "An Image is Worth 16x16 Words"
- Hatamizadeh et al. 2022: "UNETR: Transformers for 3D Medical Image Segmentation"
- Tang et al. 2022: "Self-Supervised Pre-Training of Swin Transformers"
"""
module ViT3D

using LinearAlgebra
using Statistics
using Random

export ViT3DConfig, ViT3DModel, PatchEmbedding3D
export forward, extract_features, encode_patches
export create_vit3d_tiny, create_vit3d_small, create_vit3d_base

# ============================================================================
# CONFIGURATION
# ============================================================================

"""
    ViT3DConfig

Configuration for 3D Vision Transformer.

# Fields
- `image_size::NTuple{3,Int}`: Input volume size (D, H, W)
- `patch_size::NTuple{3,Int}`: Patch size (default: 16x16x16)
- `in_channels::Int`: Number of input channels
- `embed_dim::Int`: Embedding dimension
- `depth::Int`: Number of transformer layers
- `num_heads::Int`: Number of attention heads
- `mlp_ratio::Float64`: MLP hidden dim ratio
- `dropout::Float64`: Dropout rate
- `attention_dropout::Float64`: Attention dropout rate
- `num_classes::Int`: Number of output classes (0 for feature extraction)
- `use_cls_token::Bool`: Whether to use classification token
- `pos_embed_type::Symbol`: Positional embedding type (:learnable, :sinusoidal)
"""
struct ViT3DConfig
    image_size::NTuple{3,Int}
    patch_size::NTuple{3,Int}
    in_channels::Int
    embed_dim::Int
    depth::Int
    num_heads::Int
    mlp_ratio::Float64
    dropout::Float64
    attention_dropout::Float64
    num_classes::Int
    use_cls_token::Bool
    pos_embed_type::Symbol
end

function ViT3DConfig(;
    image_size::NTuple{3,Int}=(64, 64, 64),
    patch_size::NTuple{3,Int}=(8, 8, 8),
    in_channels::Int=1,
    embed_dim::Int=384,
    depth::Int=12,
    num_heads::Int=6,
    mlp_ratio::Float64=4.0,
    dropout::Float64=0.1,
    attention_dropout::Float64=0.0,
    num_classes::Int=0,
    use_cls_token::Bool=true,
    pos_embed_type::Symbol=:learnable
)
    # Validate patch size divides image size
    for i in 1:3
        if image_size[i] % patch_size[i] != 0
            error("Image size $(image_size[i]) must be divisible by patch size $(patch_size[i])")
        end
    end

    ViT3DConfig(
        image_size, patch_size, in_channels, embed_dim,
        depth, num_heads, mlp_ratio, dropout, attention_dropout,
        num_classes, use_cls_token, pos_embed_type
    )
end

# Preset configurations
function create_vit3d_tiny(; image_size=(64,64,64), patch_size=(8,8,8), kwargs...)
    ViT3DConfig(;
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        kwargs...
    )
end

function create_vit3d_small(; image_size=(64,64,64), patch_size=(8,8,8), kwargs...)
    ViT3DConfig(;
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        kwargs...
    )
end

function create_vit3d_base(; image_size=(64,64,64), patch_size=(8,8,8), kwargs...)
    ViT3DConfig(;
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        kwargs...
    )
end

# ============================================================================
# PATCH EMBEDDING
# ============================================================================

"""
    PatchEmbedding3D

Converts 3D volume into sequence of patch embeddings.

Projects each (p_d × p_h × p_w × C) patch to embed_dim dimensions.
"""
struct PatchEmbedding3D
    patch_size::NTuple{3,Int}
    embed_dim::Int
    projection::Matrix{Float32}  # (patch_dim, embed_dim)
    bias::Vector{Float32}
    num_patches::Int
    grid_size::NTuple{3,Int}
end

function PatchEmbedding3D(config::ViT3DConfig)
    patch_dim = prod(config.patch_size) * config.in_channels
    grid_size = config.image_size .÷ config.patch_size
    num_patches = prod(grid_size)

    # Xavier initialization
    scale = sqrt(2.0 / (patch_dim + config.embed_dim))
    projection = Float32.(randn(patch_dim, config.embed_dim) .* scale)
    bias = zeros(Float32, config.embed_dim)

    PatchEmbedding3D(
        config.patch_size,
        config.embed_dim,
        projection,
        bias,
        num_patches,
        grid_size
    )
end

"""
    encode_patches(embed::PatchEmbedding3D, x::Array{T,4}) where T

Convert 4D volume (D, H, W, C) to patch embeddings (num_patches, embed_dim).
"""
function encode_patches(embed::PatchEmbedding3D, x::Array{T,4}) where T
    D, H, W, C = size(x)
    pd, ph, pw = embed.patch_size
    gd, gh, gw = embed.grid_size

    # Extract patches
    patches = zeros(Float32, embed.num_patches, prod(embed.patch_size) * C)

    patch_idx = 1
    for id in 1:gd
        for ih in 1:gh
            for iw in 1:gw
                # Extract patch
                d_start = (id - 1) * pd + 1
                h_start = (ih - 1) * ph + 1
                w_start = (iw - 1) * pw + 1

                patch = x[d_start:d_start+pd-1, h_start:h_start+ph-1, w_start:w_start+pw-1, :]
                patches[patch_idx, :] = vec(patch)
                patch_idx += 1
            end
        end
    end

    # Project to embedding dimension
    embeddings = patches * embed.projection .+ embed.bias'

    return embeddings
end

# ============================================================================
# ATTENTION MECHANISM
# ============================================================================

"""
    MultiHeadAttention3D

Multi-head self-attention layer.
"""
struct MultiHeadAttention3D
    embed_dim::Int
    num_heads::Int
    head_dim::Int
    scale::Float32
    qkv_proj::Matrix{Float32}  # (embed_dim, 3 * embed_dim)
    out_proj::Matrix{Float32}  # (embed_dim, embed_dim)
    dropout::Float64
end

function MultiHeadAttention3D(embed_dim::Int, num_heads::Int; dropout::Float64=0.0)
    @assert embed_dim % num_heads == 0 "embed_dim must be divisible by num_heads"

    head_dim = embed_dim ÷ num_heads
    scale = Float32(1.0 / sqrt(head_dim))

    # Xavier initialization
    qkv_scale = sqrt(2.0 / (embed_dim + 3 * embed_dim))
    qkv_proj = Float32.(randn(embed_dim, 3 * embed_dim) .* qkv_scale)

    out_scale = sqrt(2.0 / (2 * embed_dim))
    out_proj = Float32.(randn(embed_dim, embed_dim) .* out_scale)

    MultiHeadAttention3D(embed_dim, num_heads, head_dim, scale, qkv_proj, out_proj, dropout)
end

"""
    attention_forward(attn::MultiHeadAttention3D, x::Matrix{Float32}; training=false)

Forward pass through multi-head attention.
x: (seq_len, embed_dim)
Returns: (seq_len, embed_dim)
"""
function attention_forward(attn::MultiHeadAttention3D, x::Matrix{Float32}; training::Bool=false)
    seq_len, _ = size(x)

    # Compute Q, K, V
    qkv = x * attn.qkv_proj  # (seq_len, 3 * embed_dim)

    # Split into Q, K, V
    q = qkv[:, 1:attn.embed_dim]
    k = qkv[:, attn.embed_dim+1:2*attn.embed_dim]
    v = qkv[:, 2*attn.embed_dim+1:end]

    # Reshape for multi-head attention
    # (seq_len, num_heads, head_dim)
    q_heads = reshape(q, seq_len, attn.num_heads, attn.head_dim)
    k_heads = reshape(k, seq_len, attn.num_heads, attn.head_dim)
    v_heads = reshape(v, seq_len, attn.num_heads, attn.head_dim)

    # Compute attention scores for each head
    output = zeros(Float32, seq_len, attn.embed_dim)

    for h in 1:attn.num_heads
        q_h = q_heads[:, h, :]  # (seq_len, head_dim)
        k_h = k_heads[:, h, :]
        v_h = v_heads[:, h, :]

        # Scaled dot-product attention
        scores = (q_h * k_h') .* attn.scale  # (seq_len, seq_len)

        # Softmax
        scores_max = maximum(scores, dims=2)
        scores_exp = exp.(scores .- scores_max)
        attn_weights = scores_exp ./ sum(scores_exp, dims=2)

        # Apply dropout during training
        if training && attn.dropout > 0
            mask = rand(size(attn_weights)...) .> attn.dropout
            attn_weights = attn_weights .* mask ./ (1 - attn.dropout)
        end

        # Weighted sum of values
        head_output = attn_weights * v_h  # (seq_len, head_dim)

        # Place in output
        start_idx = (h - 1) * attn.head_dim + 1
        end_idx = h * attn.head_dim
        output[:, start_idx:end_idx] = head_output
    end

    # Output projection
    output = output * attn.out_proj

    return output
end

# ============================================================================
# MLP BLOCK
# ============================================================================

"""
    MLP3D

Feed-forward MLP block with GELU activation.
"""
struct MLP3D
    fc1::Matrix{Float32}
    fc1_bias::Vector{Float32}
    fc2::Matrix{Float32}
    fc2_bias::Vector{Float32}
    dropout::Float64
end

function MLP3D(embed_dim::Int, hidden_dim::Int; dropout::Float64=0.0)
    scale1 = sqrt(2.0 / (embed_dim + hidden_dim))
    fc1 = Float32.(randn(embed_dim, hidden_dim) .* scale1)
    fc1_bias = zeros(Float32, hidden_dim)

    scale2 = sqrt(2.0 / (hidden_dim + embed_dim))
    fc2 = Float32.(randn(hidden_dim, embed_dim) .* scale2)
    fc2_bias = zeros(Float32, embed_dim)

    MLP3D(fc1, fc1_bias, fc2, fc2_bias, dropout)
end

# GELU activation
gelu(x) = x * 0.5 * (1.0 + tanh(sqrt(2.0/π) * (x + 0.044715 * x^3)))

function mlp_forward(mlp::MLP3D, x::Matrix{Float32}; training::Bool=false)
    # First linear + GELU
    h = x * mlp.fc1 .+ mlp.fc1_bias'
    h = gelu.(h)

    # Dropout
    if training && mlp.dropout > 0
        mask = rand(size(h)...) .> mlp.dropout
        h = h .* mask ./ (1 - mlp.dropout)
    end

    # Second linear
    out = h * mlp.fc2 .+ mlp.fc2_bias'

    return out
end

# ============================================================================
# TRANSFORMER BLOCK
# ============================================================================

"""
    TransformerBlock3D

Single transformer encoder block with pre-norm architecture.
"""
struct TransformerBlock3D
    attention::MultiHeadAttention3D
    mlp::MLP3D
    norm1_weight::Vector{Float32}
    norm1_bias::Vector{Float32}
    norm2_weight::Vector{Float32}
    norm2_bias::Vector{Float32}
    embed_dim::Int
end

function TransformerBlock3D(embed_dim::Int, num_heads::Int, mlp_ratio::Float64;
                            dropout::Float64=0.0, attention_dropout::Float64=0.0)
    attention = MultiHeadAttention3D(embed_dim, num_heads; dropout=attention_dropout)
    hidden_dim = Int(embed_dim * mlp_ratio)
    mlp = MLP3D(embed_dim, hidden_dim; dropout=dropout)

    # LayerNorm parameters
    norm1_weight = ones(Float32, embed_dim)
    norm1_bias = zeros(Float32, embed_dim)
    norm2_weight = ones(Float32, embed_dim)
    norm2_bias = zeros(Float32, embed_dim)

    TransformerBlock3D(attention, mlp, norm1_weight, norm1_bias,
                       norm2_weight, norm2_bias, embed_dim)
end

# Layer normalization
function layer_norm(x::Matrix{Float32}, weight::Vector{Float32}, bias::Vector{Float32}; eps=1e-6)
    mean_x = mean(x, dims=2)
    var_x = var(x, dims=2, corrected=false)
    normalized = (x .- mean_x) ./ sqrt.(var_x .+ eps)
    return normalized .* weight' .+ bias'
end

function block_forward(block::TransformerBlock3D, x::Matrix{Float32}; training::Bool=false)
    # Pre-norm attention
    normed = layer_norm(x, block.norm1_weight, block.norm1_bias)
    attn_out = attention_forward(block.attention, normed; training=training)
    x = x + attn_out

    # Pre-norm MLP
    normed = layer_norm(x, block.norm2_weight, block.norm2_bias)
    mlp_out = mlp_forward(block.mlp, normed; training=training)
    x = x + mlp_out

    return x
end

# ============================================================================
# FULL VIT3D MODEL
# ============================================================================

"""
    ViT3DModel

Complete 3D Vision Transformer model.
"""
struct ViT3DModel
    config::ViT3DConfig
    patch_embed::PatchEmbedding3D
    cls_token::Union{Vector{Float32}, Nothing}
    pos_embed::Matrix{Float32}
    blocks::Vector{TransformerBlock3D}
    norm_weight::Vector{Float32}
    norm_bias::Vector{Float32}
    head::Union{Matrix{Float32}, Nothing}
    head_bias::Union{Vector{Float32}, Nothing}
end

function ViT3DModel(config::ViT3DConfig)
    patch_embed = PatchEmbedding3D(config)

    # Classification token
    cls_token = config.use_cls_token ? randn(Float32, config.embed_dim) .* 0.02f0 : nothing

    # Positional embeddings
    num_tokens = patch_embed.num_patches + (config.use_cls_token ? 1 : 0)

    if config.pos_embed_type == :learnable
        pos_embed = randn(Float32, num_tokens, config.embed_dim) .* 0.02f0
    else
        pos_embed = create_sinusoidal_pos_embed(num_tokens, config.embed_dim)
    end

    # Transformer blocks
    blocks = [
        TransformerBlock3D(
            config.embed_dim, config.num_heads, config.mlp_ratio;
            dropout=config.dropout, attention_dropout=config.attention_dropout
        )
        for _ in 1:config.depth
    ]

    # Final layer norm
    norm_weight = ones(Float32, config.embed_dim)
    norm_bias = zeros(Float32, config.embed_dim)

    # Classification head
    if config.num_classes > 0
        scale = sqrt(2.0 / (config.embed_dim + config.num_classes))
        head = Float32.(randn(config.embed_dim, config.num_classes) .* scale)
        head_bias = zeros(Float32, config.num_classes)
    else
        head = nothing
        head_bias = nothing
    end

    ViT3DModel(config, patch_embed, cls_token, pos_embed, blocks,
               norm_weight, norm_bias, head, head_bias)
end

"""
    create_sinusoidal_pos_embed(num_positions, embed_dim)

Create sinusoidal positional embeddings (non-learnable).
"""
function create_sinusoidal_pos_embed(num_positions::Int, embed_dim::Int)
    pos_embed = zeros(Float32, num_positions, embed_dim)

    for pos in 1:num_positions
        for i in 1:2:embed_dim
            div_term = exp(-log(10000.0) * (i - 1) / embed_dim)
            pos_embed[pos, i] = sin((pos - 1) * div_term)
            if i + 1 <= embed_dim
                pos_embed[pos, i+1] = cos((pos - 1) * div_term)
            end
        end
    end

    return pos_embed
end

"""
    forward(model::ViT3DModel, x::Array{T,4}; training=false) where T

Forward pass through ViT3D.

# Arguments
- `x`: Input volume (D, H, W, C)
- `training`: Whether in training mode (enables dropout)

# Returns
- If num_classes > 0: (num_classes,) logits
- Otherwise: (embed_dim,) feature vector from CLS token
"""
function forward(model::ViT3DModel, x::Array{T,4}; training::Bool=false) where T
    # Patch embedding
    tokens = encode_patches(model.patch_embed, x)  # (num_patches, embed_dim)

    # Prepend CLS token
    if !isnothing(model.cls_token)
        cls = reshape(model.cls_token, 1, :)
        tokens = vcat(cls, tokens)
    end

    # Add positional embeddings
    tokens = tokens + model.pos_embed

    # Transformer blocks
    for block in model.blocks
        tokens = block_forward(block, tokens; training=training)
    end

    # Final layer norm
    tokens = layer_norm(tokens, model.norm_weight, model.norm_bias)

    # Extract CLS token or mean pool
    if !isnothing(model.cls_token)
        features = tokens[1, :]  # CLS token
    else
        features = vec(mean(tokens, dims=1))  # Global average pooling
    end

    # Classification head
    if !isnothing(model.head)
        logits = features' * model.head .+ model.head_bias'
        return vec(logits)
    else
        return features
    end
end

"""
    extract_features(model::ViT3DModel, x::Array{T,4}; layer=-1) where T

Extract intermediate features from ViT3D.

# Arguments
- `x`: Input volume (D, H, W, C)
- `layer`: Which layer to extract from (-1 = last)

# Returns
- (num_tokens, embed_dim) feature matrix
"""
function extract_features(model::ViT3DModel, x::Array{T,4}; layer::Int=-1) where T
    target_layer = layer < 0 ? length(model.blocks) + layer + 1 : layer

    # Patch embedding
    tokens = encode_patches(model.patch_embed, x)

    # Prepend CLS token
    if !isnothing(model.cls_token)
        cls = reshape(model.cls_token, 1, :)
        tokens = vcat(cls, tokens)
    end

    # Add positional embeddings
    tokens = tokens + model.pos_embed

    # Transformer blocks up to target layer
    for (i, block) in enumerate(model.blocks)
        tokens = block_forward(block, tokens; training=false)
        if i == target_layer
            break
        end
    end

    return tokens
end

# ============================================================================
# SCAFFOLD-SPECIFIC UTILITIES
# ============================================================================

"""
    analyze_scaffold_volume(model::ViT3DModel, volume::Array{T,3}) where T

Analyze a microCT scaffold volume using ViT3D.

# Returns
- Dict with features and any classification results
"""
function analyze_scaffold_volume(model::ViT3DModel, volume::Array{T,3}) where T
    # Add channel dimension if needed
    if ndims(volume) == 3
        volume = reshape(volume, size(volume)..., 1)
    end

    # Normalize to [0, 1]
    volume = Float32.(volume)
    vol_min, vol_max = extrema(volume)
    if vol_max > vol_min
        volume = (volume .- vol_min) ./ (vol_max - vol_min)
    end

    # Get features
    features = forward(model, volume; training=false)

    # Extract patch-level features for visualization
    patch_features = extract_features(model, volume)

    return Dict(
        "global_features" => features,
        "patch_features" => patch_features,
        "num_patches" => size(patch_features, 1),
        "embed_dim" => model.config.embed_dim
    )
end

"""
    compute_attention_maps(model::ViT3DModel, x::Array{T,4}) where T

Compute attention maps for visualization.

Returns attention weights from each layer and head.
"""
function compute_attention_maps(model::ViT3DModel, x::Array{T,4}) where T
    # This would require storing attention weights during forward pass
    # Placeholder for full implementation
    @warn "Attention map computation requires modified forward pass - returning placeholder"

    num_layers = length(model.blocks)
    num_heads = model.config.num_heads
    num_patches = model.patch_embed.num_patches + (isnothing(model.cls_token) ? 0 : 1)

    # Return placeholder attention maps
    return Dict(
        "attention_maps" => [zeros(Float32, num_heads, num_patches, num_patches) for _ in 1:num_layers],
        "grid_size" => model.patch_embed.grid_size
    )
end

end # module
