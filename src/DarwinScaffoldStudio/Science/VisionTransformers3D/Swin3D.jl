"""
Swin3D.jl - 3D Shifted Window Transformer for Hierarchical Scaffold Analysis

Implements Swin Transformer architecture adapted for 3D volumetric data,
providing efficient hierarchical feature extraction through shifted windows.

SOTA 2024-2025 Features:
- Shifted window self-attention (O(n) complexity vs O(n²) for ViT)
- Hierarchical feature maps via patch merging
- Multi-scale feature extraction for dense prediction
- Relative position bias for translation equivariance
- Compatible with downstream segmentation/detection

Architecture (Liu et al. 2021, extended to 3D):
- Stage 1: Patch partition + Linear embed
- Stages 2-4: Swin blocks + Patch merging (2x downsampling)
- Each stage: alternating W-MSA and SW-MSA blocks

References:
- Liu et al. 2021: "Swin Transformer: Hierarchical Vision Transformer"
- Hatamizadeh et al. 2022: "Swin UNETR" (3D medical imaging)
- Tang et al. 2022: "Self-Supervised Swin for 3D Medical Images"
"""
module Swin3D

using LinearAlgebra
using Statistics
using Random

export Swin3DConfig, Swin3DModel
export forward_swin3d, extract_multiscale_features
export window_partition_3d, window_reverse_3d
export create_swin3d_tiny, create_swin3d_small, create_swin3d_base

# ============================================================================
# CONFIGURATION
# ============================================================================

"""
    Swin3DConfig

Configuration for 3D Swin Transformer.
"""
struct Swin3DConfig
    image_size::NTuple{3,Int}
    patch_size::NTuple{3,Int}
    in_channels::Int
    embed_dim::Int
    depths::Vector{Int}  # Number of blocks per stage
    num_heads::Vector{Int}  # Heads per stage
    window_size::NTuple{3,Int}
    mlp_ratio::Float64
    dropout::Float64
    attention_dropout::Float64
    drop_path_rate::Float64
    num_classes::Int
    use_checkpoint::Bool
end

function Swin3DConfig(;
    image_size::NTuple{3,Int}=(64, 64, 64),
    patch_size::NTuple{3,Int}=(4, 4, 4),
    in_channels::Int=1,
    embed_dim::Int=96,
    depths::Vector{Int}=[2, 2, 6, 2],
    num_heads::Vector{Int}=[3, 6, 12, 24],
    window_size::NTuple{3,Int}=(7, 7, 7),
    mlp_ratio::Float64=4.0,
    dropout::Float64=0.0,
    attention_dropout::Float64=0.0,
    drop_path_rate::Float64=0.1,
    num_classes::Int=0,
    use_checkpoint::Bool=false
)
    Swin3DConfig(
        image_size, patch_size, in_channels, embed_dim,
        depths, num_heads, window_size, mlp_ratio,
        dropout, attention_dropout, drop_path_rate,
        num_classes, use_checkpoint
    )
end

# Preset configurations
function create_swin3d_tiny(; image_size=(64,64,64), kwargs...)
    Swin3DConfig(;
        image_size=image_size,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        kwargs...
    )
end

function create_swin3d_small(; image_size=(64,64,64), kwargs...)
    Swin3DConfig(;
        image_size=image_size,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        kwargs...
    )
end

function create_swin3d_base(; image_size=(64,64,64), kwargs...)
    Swin3DConfig(;
        image_size=image_size,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        kwargs...
    )
end

# ============================================================================
# WINDOW UTILITIES
# ============================================================================

"""
    window_partition_3d(x, window_size)

Partition 3D feature map into non-overlapping windows.

# Arguments
- `x`: (D, H, W, C) feature map
- `window_size`: (wd, wh, ww) window dimensions

# Returns
- `windows`: (num_windows, wd*wh*ww, C) tensor
- `num_windows_dims`: (nwd, nwh, nww) number of windows per dimension
"""
function window_partition_3d(x::Array{Float32,4}, window_size::NTuple{3,Int})
    D, H, W, C = size(x)
    wd, wh, ww = window_size

    nwd = D ÷ wd
    nwh = H ÷ wh
    nww = W ÷ ww
    num_windows = nwd * nwh * nww
    tokens_per_window = wd * wh * ww

    windows = zeros(Float32, num_windows, tokens_per_window, C)

    win_idx = 1
    for id in 1:nwd
        for ih in 1:nwh
            for iw in 1:nww
                d_start = (id - 1) * wd + 1
                h_start = (ih - 1) * wh + 1
                w_start = (iw - 1) * ww + 1

                window = x[d_start:d_start+wd-1, h_start:h_start+wh-1, w_start:w_start+ww-1, :]

                # Flatten spatial dimensions
                windows[win_idx, :, :] = reshape(window, tokens_per_window, C)
                win_idx += 1
            end
        end
    end

    return windows, (nwd, nwh, nww)
end

"""
    window_reverse_3d(windows, window_size, original_size)

Reverse window partition back to original feature map.
"""
function window_reverse_3d(windows::Array{Float32,3}, window_size::NTuple{3,Int},
                           original_size::NTuple{3,Int})
    D, H, W = original_size
    wd, wh, ww = window_size
    C = size(windows, 3)

    nwd = D ÷ wd
    nwh = H ÷ wh
    nww = W ÷ ww

    x = zeros(Float32, D, H, W, C)

    win_idx = 1
    for id in 1:nwd
        for ih in 1:nwh
            for iw in 1:nww
                d_start = (id - 1) * wd + 1
                h_start = (ih - 1) * wh + 1
                w_start = (iw - 1) * ww + 1

                window = reshape(windows[win_idx, :, :], wd, wh, ww, C)
                x[d_start:d_start+wd-1, h_start:h_start+wh-1, w_start:w_start+ww-1, :] = window
                win_idx += 1
            end
        end
    end

    return x
end

"""
    cyclic_shift_3d(x, shift_size)

Cyclic shift for shifted window attention.
"""
function cyclic_shift_3d(x::Array{Float32,4}, shift_size::NTuple{3,Int})
    D, H, W, C = size(x)
    sd, sh, sw = shift_size

    shifted = similar(x)

    for d in 1:D
        for h in 1:H
            for w in 1:W
                # Compute source indices with wraparound
                src_d = mod1(d + sd, D)
                src_h = mod1(h + sh, H)
                src_w = mod1(w + sw, W)
                shifted[d, h, w, :] = x[src_d, src_h, src_w, :]
            end
        end
    end

    return shifted
end

# ============================================================================
# WINDOW ATTENTION
# ============================================================================

"""
    WindowAttention3D

Window-based multi-head self-attention with relative position bias.
"""
struct WindowAttention3D
    dim::Int
    num_heads::Int
    head_dim::Int
    scale::Float32
    window_size::NTuple{3,Int}
    qkv_proj::Matrix{Float32}
    out_proj::Matrix{Float32}
    relative_position_bias_table::Matrix{Float32}
    relative_position_index::Array{Int,3}
    dropout::Float64
end

function WindowAttention3D(dim::Int, window_size::NTuple{3,Int}, num_heads::Int;
                           dropout::Float64=0.0)
    head_dim = dim ÷ num_heads
    scale = Float32(1.0 / sqrt(head_dim))

    # QKV projection
    qkv_scale = sqrt(2.0 / (dim + 3 * dim))
    qkv_proj = Float32.(randn(dim, 3 * dim) .* qkv_scale)

    # Output projection
    out_scale = sqrt(2.0 / (2 * dim))
    out_proj = Float32.(randn(dim, dim) .* out_scale)

    # Relative position bias table
    # Size: (2*wd-1) * (2*wh-1) * (2*ww-1), num_heads
    wd, wh, ww = window_size
    num_relative_positions = (2*wd - 1) * (2*wh - 1) * (2*ww - 1)
    relative_position_bias_table = randn(Float32, num_relative_positions, num_heads) .* 0.02f0

    # Compute relative position index
    relative_position_index = compute_relative_position_index_3d(window_size)

    WindowAttention3D(dim, num_heads, head_dim, scale, window_size,
                      qkv_proj, out_proj, relative_position_bias_table,
                      relative_position_index, dropout)
end

"""
    compute_relative_position_index_3d(window_size)

Compute relative position indices for 3D window attention.
"""
function compute_relative_position_index_3d(window_size::NTuple{3,Int})
    wd, wh, ww = window_size
    tokens = wd * wh * ww

    # Create coordinate grids
    coords_d = collect(0:wd-1)
    coords_h = collect(0:wh-1)
    coords_w = collect(0:ww-1)

    # Compute relative positions for each pair of positions
    index = zeros(Int, tokens, tokens, 1)

    for i in 1:tokens
        id1 = (i - 1) ÷ (wh * ww)
        ih1 = ((i - 1) ÷ ww) % wh
        iw1 = (i - 1) % ww

        for j in 1:tokens
            id2 = (j - 1) ÷ (wh * ww)
            ih2 = ((j - 1) ÷ ww) % wh
            iw2 = (j - 1) % ww

            # Relative position
            rd = id1 - id2 + wd - 1
            rh = ih1 - ih2 + wh - 1
            rw = iw1 - iw2 + ww - 1

            # Flatten to 1D index
            rel_idx = rd * (2*wh - 1) * (2*ww - 1) + rh * (2*ww - 1) + rw + 1
            index[i, j, 1] = rel_idx
        end
    end

    return index
end

"""
    window_attention_forward(attn, x; mask=nothing, training=false)

Forward pass through window attention.

# Arguments
- `x`: (num_windows, tokens_per_window, dim) input
- `mask`: Optional attention mask for shifted windows
"""
function window_attention_forward(attn::WindowAttention3D, x::Array{Float32,3};
                                  mask::Union{Nothing, Array{Float32,2}}=nothing,
                                  training::Bool=false)
    num_windows, tokens, _ = size(x)

    # Reshape x for batch processing: (num_windows * tokens, dim)
    x_flat = reshape(x, num_windows * tokens, attn.dim)

    # Compute QKV
    qkv = x_flat * attn.qkv_proj
    qkv = reshape(qkv, num_windows, tokens, 3, attn.dim)

    q = qkv[:, :, 1, :]
    k = qkv[:, :, 2, :]
    v = qkv[:, :, 3, :]

    output = zeros(Float32, num_windows, tokens, attn.dim)

    # Process each window
    for w in 1:num_windows
        q_w = reshape(q[w, :, :], tokens, attn.num_heads, attn.head_dim)
        k_w = reshape(k[w, :, :], tokens, attn.num_heads, attn.head_dim)
        v_w = reshape(v[w, :, :], tokens, attn.num_heads, attn.head_dim)

        # Compute attention for each head
        head_outputs = zeros(Float32, tokens, attn.num_heads, attn.head_dim)

        for h in 1:attn.num_heads
            q_h = q_w[:, h, :]  # (tokens, head_dim)
            k_h = k_w[:, h, :]
            v_h = v_w[:, h, :]

            # Scaled dot-product attention
            scores = (q_h * k_h') .* attn.scale  # (tokens, tokens)

            # Add relative position bias
            for i in 1:tokens
                for j in 1:tokens
                    rel_idx = attn.relative_position_index[i, j, 1]
                    scores[i, j] += attn.relative_position_bias_table[rel_idx, h]
                end
            end

            # Apply mask if provided (for shifted windows)
            if !isnothing(mask)
                scores = scores .+ mask
            end

            # Softmax
            scores_max = maximum(scores, dims=2)
            scores_exp = exp.(scores .- scores_max)
            attn_weights = scores_exp ./ sum(scores_exp, dims=2)

            # Dropout
            if training && attn.dropout > 0
                drop_mask = rand(size(attn_weights)...) .> attn.dropout
                attn_weights = attn_weights .* drop_mask ./ (1 - attn.dropout)
            end

            # Weighted values
            head_outputs[:, h, :] = attn_weights * v_h
        end

        # Concatenate heads
        output[w, :, :] = reshape(head_outputs, tokens, attn.dim)
    end

    # Output projection
    output_flat = reshape(output, num_windows * tokens, attn.dim)
    output_flat = output_flat * attn.out_proj
    output = reshape(output_flat, num_windows, tokens, attn.dim)

    return output
end

# ============================================================================
# SWIN TRANSFORMER BLOCK
# ============================================================================

"""
    SwinBlock3D

Single Swin Transformer block with W-MSA or SW-MSA.
"""
struct SwinBlock3D
    dim::Int
    num_heads::Int
    window_size::NTuple{3,Int}
    shift_size::NTuple{3,Int}
    attention::WindowAttention3D
    mlp_fc1::Matrix{Float32}
    mlp_fc1_bias::Vector{Float32}
    mlp_fc2::Matrix{Float32}
    mlp_fc2_bias::Vector{Float32}
    norm1_weight::Vector{Float32}
    norm1_bias::Vector{Float32}
    norm2_weight::Vector{Float32}
    norm2_bias::Vector{Float32}
    drop_path_rate::Float64
end

function SwinBlock3D(dim::Int, num_heads::Int, window_size::NTuple{3,Int};
                     shift_size::NTuple{3,Int}=(0,0,0), mlp_ratio::Float64=4.0,
                     dropout::Float64=0.0, attention_dropout::Float64=0.0,
                     drop_path_rate::Float64=0.0)

    attention = WindowAttention3D(dim, window_size, num_heads; dropout=attention_dropout)

    # MLP
    hidden_dim = Int(dim * mlp_ratio)
    mlp_fc1 = Float32.(randn(dim, hidden_dim) .* sqrt(2.0 / (dim + hidden_dim)))
    mlp_fc1_bias = zeros(Float32, hidden_dim)
    mlp_fc2 = Float32.(randn(hidden_dim, dim) .* sqrt(2.0 / (hidden_dim + dim)))
    mlp_fc2_bias = zeros(Float32, dim)

    # Layer norms
    norm1_weight = ones(Float32, dim)
    norm1_bias = zeros(Float32, dim)
    norm2_weight = ones(Float32, dim)
    norm2_bias = zeros(Float32, dim)

    SwinBlock3D(dim, num_heads, window_size, shift_size, attention,
                mlp_fc1, mlp_fc1_bias, mlp_fc2, mlp_fc2_bias,
                norm1_weight, norm1_bias, norm2_weight, norm2_bias,
                drop_path_rate)
end

# GELU activation
gelu(x) = x * 0.5 * (1.0 + tanh(sqrt(2.0/π) * (x + 0.044715 * x^3)))

function layer_norm_4d(x::Array{Float32,4}, weight::Vector{Float32}, bias::Vector{Float32}; eps=1e-6)
    D, H, W, C = size(x)
    x_flat = reshape(x, D * H * W, C)

    mean_x = mean(x_flat, dims=2)
    var_x = var(x_flat, dims=2, corrected=false)
    normalized = (x_flat .- mean_x) ./ sqrt.(var_x .+ eps)
    normed = normalized .* weight' .+ bias'

    return reshape(normed, D, H, W, C)
end

function swin_block_forward(block::SwinBlock3D, x::Array{Float32,4};
                            training::Bool=false)
    D, H, W, C = size(x)
    shortcut = x

    # Layer norm 1
    x = layer_norm_4d(x, block.norm1_weight, block.norm1_bias)

    # Cyclic shift if needed
    if any(block.shift_size .> 0)
        shifted_x = cyclic_shift_3d(x, .-block.shift_size)
    else
        shifted_x = x
    end

    # Partition windows
    windows, num_windows_dims = window_partition_3d(shifted_x, block.window_size)

    # Window attention
    attn_windows = window_attention_forward(block.attention, windows; training=training)

    # Reverse windows
    shifted_x = window_reverse_3d(attn_windows, block.window_size, (D, H, W))

    # Reverse cyclic shift
    if any(block.shift_size .> 0)
        x = cyclic_shift_3d(shifted_x, block.shift_size)
    else
        x = shifted_x
    end

    # Drop path (stochastic depth)
    if training && block.drop_path_rate > 0 && rand() < block.drop_path_rate
        x = shortcut
    else
        x = shortcut + x
    end

    # MLP block
    shortcut = x
    x = layer_norm_4d(x, block.norm2_weight, block.norm2_bias)

    # Flatten for MLP
    x_flat = reshape(x, D * H * W, C)
    h = x_flat * block.mlp_fc1 .+ block.mlp_fc1_bias'
    h = gelu.(h)
    h = h * block.mlp_fc2 .+ block.mlp_fc2_bias'
    x = reshape(h, D, H, W, C)

    # Drop path
    if training && block.drop_path_rate > 0 && rand() < block.drop_path_rate
        x = shortcut
    else
        x = shortcut + x
    end

    return x
end

# ============================================================================
# PATCH MERGING (DOWNSAMPLING)
# ============================================================================

"""
    PatchMerging3D

Merges 2x2x2 patches to reduce resolution by 2x.
"""
struct PatchMerging3D
    dim::Int
    reduction::Matrix{Float32}
    norm_weight::Vector{Float32}
    norm_bias::Vector{Float32}
end

function PatchMerging3D(dim::Int)
    # Concatenate 8 patches (2x2x2) then project to 2*dim
    input_dim = 8 * dim
    output_dim = 2 * dim
    reduction = Float32.(randn(input_dim, output_dim) .* sqrt(2.0 / (input_dim + output_dim)))
    norm_weight = ones(Float32, input_dim)
    norm_bias = zeros(Float32, input_dim)

    PatchMerging3D(dim, reduction, norm_weight, norm_bias)
end

function patch_merge_forward(pm::PatchMerging3D, x::Array{Float32,4})
    D, H, W, C = size(x)

    # Ensure divisible by 2
    @assert D % 2 == 0 && H % 2 == 0 && W % 2 == 0 "Dimensions must be divisible by 2"

    # Extract 8 groups of patches
    x0 = x[1:2:end, 1:2:end, 1:2:end, :]
    x1 = x[2:2:end, 1:2:end, 1:2:end, :]
    x2 = x[1:2:end, 2:2:end, 1:2:end, :]
    x3 = x[2:2:end, 2:2:end, 1:2:end, :]
    x4 = x[1:2:end, 1:2:end, 2:2:end, :]
    x5 = x[2:2:end, 1:2:end, 2:2:end, :]
    x6 = x[1:2:end, 2:2:end, 2:2:end, :]
    x7 = x[2:2:end, 2:2:end, 2:2:end, :]

    # Concatenate along channel dimension
    new_D, new_H, new_W = D ÷ 2, H ÷ 2, W ÷ 2
    x_merged = cat(x0, x1, x2, x3, x4, x5, x6, x7, dims=4)  # (D/2, H/2, W/2, 8*C)

    # Flatten, normalize, reduce
    x_flat = reshape(x_merged, new_D * new_H * new_W, 8 * C)

    # Layer norm
    mean_x = mean(x_flat, dims=2)
    var_x = var(x_flat, dims=2, corrected=false)
    x_normed = (x_flat .- mean_x) ./ sqrt.(var_x .+ 1e-6)
    x_normed = x_normed .* pm.norm_weight' .+ pm.norm_bias'

    # Linear reduction
    x_reduced = x_normed * pm.reduction

    return reshape(x_reduced, new_D, new_H, new_W, 2 * C)
end

# ============================================================================
# SWIN STAGE
# ============================================================================

"""
    SwinStage3D

A stage of Swin Transformer blocks followed by optional patch merging.
"""
struct SwinStage3D
    blocks::Vector{SwinBlock3D}
    downsample::Union{PatchMerging3D, Nothing}
end

function SwinStage3D(dim::Int, depth::Int, num_heads::Int, window_size::NTuple{3,Int};
                     mlp_ratio::Float64=4.0, dropout::Float64=0.0,
                     attention_dropout::Float64=0.0, drop_path_rates::Vector{Float64}=Float64[],
                     downsample::Bool=true)

    if isempty(drop_path_rates)
        drop_path_rates = zeros(depth)
    end

    blocks = SwinBlock3D[]
    for i in 1:depth
        # Alternate between W-MSA (shift=0) and SW-MSA (shift=window_size/2)
        shift_size = (i % 2 == 0) ? window_size .÷ 2 : (0, 0, 0)

        block = SwinBlock3D(dim, num_heads, window_size;
                            shift_size=shift_size, mlp_ratio=mlp_ratio,
                            dropout=dropout, attention_dropout=attention_dropout,
                            drop_path_rate=drop_path_rates[i])
        push!(blocks, block)
    end

    ds = downsample ? PatchMerging3D(dim) : nothing

    SwinStage3D(blocks, ds)
end

function stage_forward(stage::SwinStage3D, x::Array{Float32,4}; training::Bool=false)
    for block in stage.blocks
        x = swin_block_forward(block, x; training=training)
    end

    if !isnothing(stage.downsample)
        x = patch_merge_forward(stage.downsample, x)
    end

    return x
end

# ============================================================================
# FULL SWIN3D MODEL
# ============================================================================

"""
    Swin3DModel

Complete 3D Swin Transformer model.
"""
struct Swin3DModel
    config::Swin3DConfig
    patch_embed::Matrix{Float32}
    patch_embed_bias::Vector{Float32}
    stages::Vector{SwinStage3D}
    norm_weight::Vector{Float32}
    norm_bias::Vector{Float32}
    head::Union{Matrix{Float32}, Nothing}
    head_bias::Union{Vector{Float32}, Nothing}
end

function Swin3DModel(config::Swin3DConfig)
    # Patch embedding (linear projection)
    patch_dim = prod(config.patch_size) * config.in_channels
    patch_embed = Float32.(randn(patch_dim, config.embed_dim) .* sqrt(2.0 / (patch_dim + config.embed_dim)))
    patch_embed_bias = zeros(Float32, config.embed_dim)

    # Compute drop path rates per block
    total_blocks = sum(config.depths)
    drop_path_rates = range(0, config.drop_path_rate, length=total_blocks)

    # Build stages
    stages = SwinStage3D[]
    current_dim = config.embed_dim
    block_idx = 1

    for (i, (depth, num_heads)) in enumerate(zip(config.depths, config.num_heads))
        # Adjust window size to not exceed feature map size
        resolution = config.image_size .÷ config.patch_size .÷ (2^(i-1))
        window_size = min.(config.window_size, resolution)

        stage_drop_rates = collect(drop_path_rates[block_idx:block_idx+depth-1])

        # Downsample for all but last stage
        downsample = i < length(config.depths)

        stage = SwinStage3D(current_dim, depth, num_heads, window_size;
                            mlp_ratio=config.mlp_ratio, dropout=config.dropout,
                            attention_dropout=config.attention_dropout,
                            drop_path_rates=stage_drop_rates, downsample=downsample)
        push!(stages, stage)

        if downsample
            current_dim *= 2
        end

        block_idx += depth
    end

    # Final layer norm
    final_dim = config.embed_dim * (2^(length(config.depths) - 1))
    norm_weight = ones(Float32, final_dim)
    norm_bias = zeros(Float32, final_dim)

    # Classification head
    if config.num_classes > 0
        head = Float32.(randn(final_dim, config.num_classes) .* sqrt(2.0 / (final_dim + config.num_classes)))
        head_bias = zeros(Float32, config.num_classes)
    else
        head = nothing
        head_bias = nothing
    end

    Swin3DModel(config, patch_embed, patch_embed_bias, stages,
                norm_weight, norm_bias, head, head_bias)
end

"""
    forward_swin3d(model, x; training=false)

Forward pass through Swin3D model.

# Arguments
- `x`: Input volume (D, H, W, C)
- `training`: Enable dropout/drop path

# Returns
- Classification logits or feature vector
"""
function forward_swin3d(model::Swin3DModel, x::Array{T,4}; training::Bool=false) where T
    x = Float32.(x)
    D, H, W, C = size(x)
    pd, ph, pw = model.config.patch_size

    # Patch embedding
    num_patches_d = D ÷ pd
    num_patches_h = H ÷ ph
    num_patches_w = W ÷ pw
    num_patches = num_patches_d * num_patches_h * num_patches_w

    # Extract and project patches
    patch_dim = pd * ph * pw * C
    patches = zeros(Float32, num_patches, patch_dim)

    patch_idx = 1
    for id in 1:num_patches_d
        for ih in 1:num_patches_h
            for iw in 1:num_patches_w
                d_start = (id - 1) * pd + 1
                h_start = (ih - 1) * ph + 1
                w_start = (iw - 1) * pw + 1

                patch = x[d_start:d_start+pd-1, h_start:h_start+ph-1, w_start:w_start+pw-1, :]
                patches[patch_idx, :] = vec(patch)
                patch_idx += 1
            end
        end
    end

    embedded = patches * model.patch_embed .+ model.patch_embed_bias'

    # Reshape to (D', H', W', embed_dim)
    x = reshape(embedded, num_patches_d, num_patches_h, num_patches_w, model.config.embed_dim)

    # Process through stages
    for stage in model.stages
        x = stage_forward(stage, x; training=training)
    end

    # Final norm
    x = layer_norm_4d(x, model.norm_weight, model.norm_bias)

    # Global average pooling
    features = vec(mean(x, dims=(1,2,3)))

    # Classification head
    if !isnothing(model.head)
        logits = features' * model.head .+ model.head_bias'
        return vec(logits)
    else
        return features
    end
end

"""
    extract_multiscale_features(model, x)

Extract hierarchical features from each stage.

Returns a list of feature maps at different scales.
"""
function extract_multiscale_features(model::Swin3DModel, x::Array{T,4}) where T
    x = Float32.(x)
    D, H, W, C = size(x)
    pd, ph, pw = model.config.patch_size

    # Patch embedding
    num_patches_d = D ÷ pd
    num_patches_h = H ÷ ph
    num_patches_w = W ÷ pw
    num_patches = num_patches_d * num_patches_h * num_patches_w
    patch_dim = pd * ph * pw * C

    patches = zeros(Float32, num_patches, patch_dim)
    patch_idx = 1
    for id in 1:num_patches_d
        for ih in 1:num_patches_h
            for iw in 1:num_patches_w
                d_start = (id - 1) * pd + 1
                h_start = (ih - 1) * ph + 1
                w_start = (iw - 1) * pw + 1

                patch = x[d_start:d_start+pd-1, h_start:h_start+ph-1, w_start:w_start+pw-1, :]
                patches[patch_idx, :] = vec(patch)
                patch_idx += 1
            end
        end
    end

    embedded = patches * model.patch_embed .+ model.patch_embed_bias'
    x = reshape(embedded, num_patches_d, num_patches_h, num_patches_w, model.config.embed_dim)

    # Collect features from each stage
    multiscale_features = Array{Float32,4}[]
    push!(multiscale_features, copy(x))

    for stage in model.stages
        x = stage_forward(stage, x; training=false)
        push!(multiscale_features, copy(x))
    end

    return multiscale_features
end

end # module
