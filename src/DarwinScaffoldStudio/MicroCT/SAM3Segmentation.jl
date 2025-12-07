module SAM3Segmentation

using PyCall
using Images
using Statistics

export segment_pores_sam3, SAM3Config, initialize_sam3

"""
SAM 3 (Segment Anything Model 3) Integration for Pore Segmentation
===================================================================

Meta AI, November 2024
- 848M parameters
- Text + visual prompts
- 4M+ concepts trained
- 2x accuracy improvement over SAM 2

Reference: https://huggingface.co/facebook/sam3
Paper: arxiv.org/abs/2511.16719
"""

# Configuration
Base.@kwdef struct SAM3Config
    model_id::String = "facebook/sam3-hiera-base"
    device::String = "cuda"  # or "cpu"
    text_prompt::String = "pore"
    confidence_threshold::Float64 = 0.5
end

# Global model cache
const SAM3_MODEL = Ref{Any}(nothing)
const SAM3_PROCESSOR = Ref{Any}(nothing)

"""
    initialize_sam3(config::SAM3Config)

Initialize SAM 3 model from Hugging Face.
Requires: pip install transformers torch
"""
function initialize_sam3(config::SAM3Config=SAM3Config())
    @info "Initializing SAM 3 from Hugging Face..."

    transformers = pyimport("transformers")
    torch = pyimport("torch")

    # Load processor and model
    SAM3_PROCESSOR[] = transformers.Sam3Processor.from_pretrained(config.model_id)
    SAM3_MODEL[] = transformers.Sam3Model.from_pretrained(config.model_id)

    # Move to device
    if config.device == "cuda" && torch.cuda.is_available()
        SAM3_MODEL[].to("cuda")
        @info "SAM 3 loaded on CUDA"
    else
        @info "SAM 3 loaded on CPU"
    end

    return nothing
end

"""
    segment_pores_sam3(image; config=SAM3Config())

Segment pores in SEM/microCT image using SAM 3 with text prompt "pore".

Arguments:
- image: 2D grayscale array (SEM image)
- config: SAM3Config with model settings

Returns:
- Binary mask where true = pore
- Vector of individual pore masks
- Confidence scores
"""
function segment_pores_sam3(image::AbstractMatrix; config::SAM3Config=SAM3Config())

    # Initialize model if needed
    if SAM3_MODEL[] === nothing
        initialize_sam3(config)
    end

    torch = pyimport("torch")
    np = pyimport("numpy")

    # Convert Julia array to numpy
    if eltype(image) <: AbstractFloat
        img_np = np.array(image .* 255, dtype=np.uint8)
    else
        img_np = np.array(image)
    end

    # Convert grayscale to RGB (SAM expects RGB)
    if ndims(img_np) == 2
        img_rgb = np.stack([img_np, img_np, img_np], axis=2)
    else
        img_rgb = img_np
    end

    # Process with SAM 3
    inputs = SAM3_PROCESSOR[](
        images=img_rgb,
        text=[config.text_prompt],
        return_tensors="pt"
    )

    # Move to device
    if config.device == "cuda"
        inputs = Dict(k => v.to("cuda") for (k, v) in inputs)
    end

    # Run inference (no_grad context for PyTorch)
    torch.set_grad_enabled(false)
    outputs = SAM3_MODEL[](inputs...)
    torch.set_grad_enabled(true)

    # Extract masks
    masks = outputs.pred_masks.cpu().numpy()
    scores = outputs.scores.cpu().numpy()

    # Filter by confidence
    valid_idx = scores .> config.confidence_threshold

    # Combine all valid masks into single binary mask
    h, w = size(image)
    combined_mask = zeros(Bool, h, w)
    individual_masks = []
    valid_scores = Float64[]

    for (i, valid) in enumerate(valid_idx)
        if valid
            mask_i = masks[i, :, :] .> 0.5
            push!(individual_masks, mask_i)
            push!(valid_scores, scores[i])
            combined_mask .|= mask_i
        end
    end

    @info "SAM 3 segmentation complete" n_pores=length(individual_masks) mean_confidence=mean(valid_scores)

    return combined_mask, individual_masks, valid_scores
end

"""
    segment_pores_sam3_fallback(image; config=SAM3Config())

Fallback using transformers pipeline API (simpler but may be slower).
"""
function segment_pores_sam3_fallback(image::AbstractMatrix; config::SAM3Config=SAM3Config())

    transformers = pyimport("transformers")
    np = pyimport("numpy")
    PIL = pyimport("PIL.Image")

    # Create pipeline
    pipe = transformers.pipeline(
        "image-segmentation",
        model=config.model_id,
        device=0  # GPU
    )

    # Convert to PIL Image
    img_uint8 = np.array(image .* 255, dtype=np.uint8)
    pil_image = PIL.fromarray(img_uint8)

    # Run with text prompt
    results = pipe(pil_image, text=config.text_prompt)

    # Extract masks
    h, w = size(image)
    combined_mask = zeros(Bool, h, w)

    for result in results
        mask = np.array(result["mask"])
        combined_mask .|= (mask .> 0)
    end

    return combined_mask
end

"""
    compute_pore_metrics_sam3(image, pixel_size_um; config=SAM3Config())

Complete pore analysis using SAM 3 segmentation.
"""
function compute_pore_metrics_sam3(image::AbstractMatrix, pixel_size_um::Float64;
                                    config::SAM3Config=SAM3Config())

    # Segment with SAM 3
    mask, individual_masks, scores = segment_pores_sam3(image; config=config)

    # Compute metrics for each pore
    pore_diameters = Float64[]
    pore_areas = Float64[]

    for pore_mask in individual_masks
        area_pixels = sum(pore_mask)
        area_um2 = area_pixels * pixel_size_um^2

        # Equivalent circular diameter
        diameter_um = 2 * sqrt(area_um2 / π)

        push!(pore_areas, area_um2)
        push!(pore_diameters, diameter_um)
    end

    # Porosity
    porosity = sum(mask) / length(mask)

    return (
        porosity = porosity,
        n_pores = length(individual_masks),
        mean_pore_size = isempty(pore_diameters) ? 0.0 : mean(pore_diameters),
        std_pore_size = isempty(pore_diameters) ? 0.0 : std(pore_diameters),
        pore_diameters = pore_diameters,
        pore_areas = pore_areas,
        confidence_scores = scores,
        mask = mask
    )
end

end # module
