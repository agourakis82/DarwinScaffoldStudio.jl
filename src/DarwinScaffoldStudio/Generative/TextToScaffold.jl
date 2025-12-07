"""
    TextToScaffold

Text-to-Scaffold generation using LLM + parametric design.

Converts natural language descriptions into 3D scaffold geometries:
- "Design a bone scaffold with 80% porosity and 200um pores"
- "Create a skin regeneration scaffold using collagen hydrogel"
- "Generate a vascularized cardiac patch with aligned channels"

# Pipeline
1. LLM extracts parameters from text
2. Ontology validates tissue/material compatibility
3. Parametric generator creates geometry
4. TPMS or salt-leaching model applied
5. Validation against Q1 literature

# Author: Dr. Demetrios Agourakis
# Darwin Scaffold Studio v0.5.0
"""
module TextToScaffold

using ..OllamaClient: OllamaModel, generate, chat
using ..Types: ScaffoldMetrics, ScaffoldParameters
using JSON3
using LinearAlgebra
using Statistics

export generate_scaffold_from_text, ScaffoldPrompt, GenerationResult
export parse_scaffold_description, validate_parameters
export TPMSType, generate_tpms, generate_salt_leaching, generate_bioprinting_lattice

#=============================================================================
  CONFIGURATION
=============================================================================#

"""TPMS (Triply Periodic Minimal Surface) types."""
@enum TPMSType begin
    GYROID
    SCHWARZ_P
    SCHWARZ_D
    NEOVIUS
    LIDINOID
    IWP
    FISCHER_KOCH
end

"""
    ScaffoldPrompt

Structured representation of a scaffold design request.
"""
struct ScaffoldPrompt
    raw_text::String
    target_tissue::Symbol
    material::String
    porosity::Float64
    pore_size_um::Float64
    dimensions_mm::Tuple{Float64, Float64, Float64}
    special_features::Vector{String}
    generation_method::Symbol  # :tpms, :salt_leaching, :bioprinting, :freeze_casting
end

"""
    GenerationResult

Result of scaffold generation.
"""
struct GenerationResult
    success::Bool
    volume::Union{Array{Bool,3}, Nothing}
    parameters::ScaffoldPrompt
    metrics::Union{ScaffoldMetrics, Nothing}
    warnings::Vector{String}
    generation_time_s::Float64
end

#=============================================================================
  LLM PARAMETER EXTRACTION
=============================================================================#

const EXTRACTION_SYSTEM_PROMPT = """
You are a tissue engineering expert. Extract scaffold design parameters from the user's description.

Return ONLY valid JSON with these fields:
{
    "target_tissue": "bone|cartilage|skin|cardiac|neural|vascular|tendon|general",
    "material": "PCL|PLA|PLGA|Collagen|Hydrogel|HA_TCP|Ti6Al4V|Chitosan|Alginate|Fibrin",
    "porosity": 0.0-1.0,
    "pore_size_um": 50-1000,
    "dimensions_mm": [x, y, z],
    "special_features": ["aligned_channels", "gradient_porosity", "vascularization", "drug_delivery"],
    "generation_method": "tpms|salt_leaching|bioprinting|freeze_casting"
}

Use these defaults if not specified:
- porosity: 0.75
- pore_size_um: 200
- dimensions_mm: [10, 10, 10]
- generation_method: "tpms" for bone, "salt_leaching" for soft tissues

Literature guidance:
- Bone: porosity 0.85-0.95, pore size 100-500um (Murphy 2010)
- Cartilage: porosity 0.80-0.90, pore size 200-400um
- Skin: porosity 0.70-0.90, pore size 100-200um
- Neural: porosity 0.85-0.95, pore size 50-150um
"""

"""
    parse_scaffold_description(text::String; model_name="llama3.2:3b") -> ScaffoldPrompt

Use LLM to extract scaffold parameters from natural language.
"""
function parse_scaffold_description(text::String; model_name::String="llama3.2:3b")
    model = OllamaModel(model_name)

    prompt = """
    User request: "$text"

    Extract the scaffold design parameters as JSON.
    """

    try
        response = generate(model, prompt, system=EXTRACTION_SYSTEM_PROMPT, temperature=0.1)

        # Extract JSON from response
        json_match = match(r"\{[^{}]*\}", response)
        if isnothing(json_match)
            # Try to find JSON with nested braces
            json_match = match(r"\{.*\}", response, s=true)
        end

        if isnothing(json_match)
            @warn "Could not extract JSON from LLM response, using defaults"
            return default_prompt(text)
        end

        params = JSON3.read(json_match.match, Dict)

        return ScaffoldPrompt(
            text,
            Symbol(get(params, "target_tissue", "general")),
            get(params, "material", "PCL"),
            get(params, "porosity", 0.75),
            get(params, "pore_size_um", 200.0),
            tuple(get(params, "dimensions_mm", [10.0, 10.0, 10.0])...),
            get(params, "special_features", String[]),
            Symbol(get(params, "generation_method", "tpms"))
        )
    catch e
        @warn "LLM extraction failed, using defaults" error=e
        return default_prompt(text)
    end
end

"""Default prompt when LLM fails."""
function default_prompt(text::String)
    # Simple keyword extraction
    text_lower = lowercase(text)

    tissue = if contains(text_lower, "bone")
        :bone
    elseif contains(text_lower, "cartilage")
        :cartilage
    elseif contains(text_lower, "skin")
        :skin
    elseif contains(text_lower, "cardiac") || contains(text_lower, "heart")
        :cardiac
    elseif contains(text_lower, "neural") || contains(text_lower, "nerve")
        :neural
    else
        :general
    end

    material = if contains(text_lower, "collagen")
        "Collagen"
    elseif contains(text_lower, "hydrogel")
        "Hydrogel"
    elseif contains(text_lower, "pla")
        "PLA"
    elseif contains(text_lower, "titanium")
        "Ti6Al4V"
    else
        "PCL"
    end

    # Extract numbers
    porosity = 0.75
    pore_size = 200.0

    porosity_match = match(r"(\d+)\s*%?\s*poros", text_lower)
    if !isnothing(porosity_match)
        porosity = parse(Float64, porosity_match[1]) / 100
    end

    pore_match = match(r"(\d+)\s*[uμ]?m\s*pore", text_lower)
    if !isnothing(pore_match)
        pore_size = parse(Float64, pore_match[1])
    end

    ScaffoldPrompt(text, tissue, material, porosity, pore_size,
                   (10.0, 10.0, 10.0), String[], :tpms)
end

#=============================================================================
  PARAMETER VALIDATION
=============================================================================#

"""Q1 literature parameter ranges."""
const Q1_RANGES = Dict(
    :bone => (porosity=(0.85, 0.95), pore_size=(100.0, 500.0)),
    :cartilage => (porosity=(0.80, 0.90), pore_size=(200.0, 400.0)),
    :skin => (porosity=(0.70, 0.90), pore_size=(100.0, 200.0)),
    :neural => (porosity=(0.85, 0.95), pore_size=(50.0, 150.0)),
    :cardiac => (porosity=(0.80, 0.90), pore_size=(150.0, 300.0)),
    :vascular => (porosity=(0.70, 0.85), pore_size=(100.0, 200.0)),
    :tendon => (porosity=(0.60, 0.80), pore_size=(200.0, 400.0)),
    :general => (porosity=(0.50, 0.95), pore_size=(50.0, 500.0))
)

"""
    validate_parameters(prompt::ScaffoldPrompt) -> Tuple{Bool, Vector{String}}

Validate parameters against Q1 literature.
"""
function validate_parameters(prompt::ScaffoldPrompt)
    warnings = String[]

    ranges = get(Q1_RANGES, prompt.target_tissue, Q1_RANGES[:general])

    # Check porosity
    if prompt.porosity < ranges.porosity[1]
        push!(warnings, "Porosity $(round(prompt.porosity*100, digits=1))% below recommended minimum $(round(ranges.porosity[1]*100, digits=1))% for $(prompt.target_tissue)")
    elseif prompt.porosity > ranges.porosity[2]
        push!(warnings, "Porosity $(round(prompt.porosity*100, digits=1))% above recommended maximum $(round(ranges.porosity[2]*100, digits=1))% for $(prompt.target_tissue)")
    end

    # Check pore size
    if prompt.pore_size_um < ranges.pore_size[1]
        push!(warnings, "Pore size $(round(prompt.pore_size_um, digits=1))um below recommended minimum $(ranges.pore_size[1])um for $(prompt.target_tissue)")
    elseif prompt.pore_size_um > ranges.pore_size[2]
        push!(warnings, "Pore size $(round(prompt.pore_size_um, digits=1))um above recommended maximum $(ranges.pore_size[2])um for $(prompt.target_tissue)")
    end

    # Material-tissue compatibility
    incompatible = check_material_tissue_compatibility(prompt.material, prompt.target_tissue)
    if !isnothing(incompatible)
        push!(warnings, incompatible)
    end

    return (isempty(warnings), warnings)
end

"""Check material-tissue compatibility."""
function check_material_tissue_compatibility(material::String, tissue::Symbol)
    # Soft tissues need soft materials
    soft_tissues = [:skin, :neural, :cardiac]
    hard_materials = ["Ti6Al4V", "HA_TCP"]

    if tissue in soft_tissues && material in hard_materials
        return "Material $material may be too stiff for $tissue tissue. Consider softer materials like Collagen or Hydrogel."
    end

    # Bone can use hard materials
    if tissue == :bone && material in ["Hydrogel", "Alginate", "Fibrin"]
        return "Material $material may be too soft for bone scaffolds. Consider PCL, PLA, or HA_TCP."
    end

    return nothing
end

#=============================================================================
  TPMS GENERATION
=============================================================================#

"""
    generate_tpms(; tpms_type, resolution, porosity, unit_cell_size, dimensions) -> Array{Bool,3}

Generate TPMS scaffold geometry.
"""
function generate_tpms(;
    tpms_type::TPMSType=GYROID,
    resolution::Int=100,
    porosity::Float64=0.75,
    unit_cell_size::Float64=1.0,
    dimensions::Tuple{Float64,Float64,Float64}=(10.0, 10.0, 10.0)
)
    # Calculate voxel counts
    nx = round(Int, dimensions[1] / unit_cell_size * resolution / 10)
    ny = round(Int, dimensions[2] / unit_cell_size * resolution / 10)
    nz = round(Int, dimensions[3] / unit_cell_size * resolution / 10)

    nx = max(nx, 10)
    ny = max(ny, 10)
    nz = max(nz, 10)

    # Create coordinate grids
    x = range(0, 2π * dimensions[1] / unit_cell_size, length=nx)
    y = range(0, 2π * dimensions[2] / unit_cell_size, length=ny)
    z = range(0, 2π * dimensions[3] / unit_cell_size, length=nz)

    # Initialize volume
    volume = zeros(Float64, nx, ny, nz)

    # Compute TPMS function
    for (i, xi) in enumerate(x)
        for (j, yj) in enumerate(y)
            for (k, zk) in enumerate(z)
                volume[i, j, k] = tpms_function(tpms_type, xi, yj, zk)
            end
        end
    end

    # Find threshold for target porosity
    threshold = find_porosity_threshold(volume, porosity)

    # Binarize: pore = true (where TPMS > threshold)
    return volume .> threshold
end

"""TPMS implicit functions."""
function tpms_function(tpms_type::TPMSType, x::Real, y::Real, z::Real)
    if tpms_type == GYROID
        return sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)
    elseif tpms_type == SCHWARZ_P
        return cos(x) + cos(y) + cos(z)
    elseif tpms_type == SCHWARZ_D
        return cos(x) * cos(y) * cos(z) - sin(x) * sin(y) * sin(z)
    elseif tpms_type == NEOVIUS
        return 3 * (cos(x) + cos(y) + cos(z)) + 4 * cos(x) * cos(y) * cos(z)
    elseif tpms_type == LIDINOID
        return sin(2x) * cos(y) * sin(z) + sin(2y) * cos(z) * sin(x) + sin(2z) * cos(x) * sin(y) -
               cos(2x) * cos(2y) - cos(2y) * cos(2z) - cos(2z) * cos(2x) + 0.3
    elseif tpms_type == IWP
        return 2 * (cos(x) * cos(y) + cos(y) * cos(z) + cos(z) * cos(x)) -
               (cos(2x) + cos(2y) + cos(2z))
    elseif tpms_type == FISCHER_KOCH
        return cos(2x) * sin(y) * cos(z) + cos(x) * cos(2y) * sin(z) + sin(x) * cos(y) * cos(2z)
    else
        return sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)  # Default: Gyroid
    end
end

"""Find threshold for target porosity using binary search."""
function find_porosity_threshold(volume::Array{Float64,3}, target_porosity::Float64)
    sorted_vals = sort(vec(volume))
    n = length(sorted_vals)

    # Target index for threshold
    target_idx = round(Int, n * (1 - target_porosity))
    target_idx = clamp(target_idx, 1, n)

    return sorted_vals[target_idx]
end

#=============================================================================
  SALT LEACHING GENERATION
=============================================================================#

"""
    generate_salt_leaching(; porosity, pore_size_voxels, resolution, dimensions) -> Array{Bool,3}

Generate scaffold using salt-leaching model (random spherical pores).
"""
function generate_salt_leaching(;
    porosity::Float64=0.75,
    pore_size_voxels::Int=10,
    resolution::Int=100,
    dimensions::Tuple{Float64,Float64,Float64}=(10.0, 10.0, 10.0)
)
    # Calculate voxel counts
    scale = resolution / 10.0
    nx = round(Int, dimensions[1] * scale)
    ny = round(Int, dimensions[2] * scale)
    nz = round(Int, dimensions[3] * scale)

    nx = max(nx, 20)
    ny = max(ny, 20)
    nz = max(nz, 20)

    # Start with solid
    volume = trues(nx, ny, nz)

    # Calculate number of pores needed
    total_voxels = nx * ny * nz
    target_pore_voxels = round(Int, total_voxels * porosity)

    # Approximate pore volume
    pore_radius = pore_size_voxels ÷ 2
    single_pore_volume = 4/3 * π * pore_radius^3
    num_pores = round(Int, target_pore_voxels / single_pore_volume * 1.5)  # Overshoot for overlap

    # Place random spherical pores
    current_pore_voxels = 0

    for _ in 1:num_pores
        if current_pore_voxels >= target_pore_voxels
            break
        end

        # Random center
        cx = rand(pore_radius+1:nx-pore_radius)
        cy = rand(pore_radius+1:ny-pore_radius)
        cz = rand(pore_radius+1:nz-pore_radius)

        # Create spherical pore
        for dx in -pore_radius:pore_radius
            for dy in -pore_radius:pore_radius
                for dz in -pore_radius:pore_radius
                    if dx^2 + dy^2 + dz^2 <= pore_radius^2
                        x = cx + dx
                        y = cy + dy
                        z = cz + dz
                        if 1 <= x <= nx && 1 <= y <= ny && 1 <= z <= nz
                            if volume[x, y, z]
                                volume[x, y, z] = false
                                current_pore_voxels += 1
                            end
                        end
                    end
                end
            end
        end
    end

    return .!volume  # Pore = true
end

#=============================================================================
  BIOPRINTING LATTICE
=============================================================================#

"""
    generate_bioprinting_lattice(; strand_diameter, strand_spacing, layer_angle, dimensions) -> Array{Bool,3}

Generate bioprinting-style lattice scaffold.
"""
function generate_bioprinting_lattice(;
    strand_diameter::Float64=0.2,  # mm
    strand_spacing::Float64=0.5,   # mm
    layer_angle::Float64=90.0,     # degrees rotation between layers
    resolution::Int=100,
    dimensions::Tuple{Float64,Float64,Float64}=(10.0, 10.0, 10.0)
)
    scale = resolution / 10.0
    nx = round(Int, dimensions[1] * scale)
    ny = round(Int, dimensions[2] * scale)
    nz = round(Int, dimensions[3] * scale)

    # Voxel sizes
    strand_r = round(Int, strand_diameter * scale / 2)
    spacing = round(Int, strand_spacing * scale)
    layer_height = strand_r * 2

    # Initialize as pore (true = pore in our convention)
    volume = trues(nx, ny, nz)

    layer = 0
    for z in 1:layer_height:nz
        angle = layer * deg2rad(layer_angle)
        layer += 1

        # Direction vectors for this layer
        dir_x = cos(angle)
        dir_y = sin(angle)

        # Draw strands
        if abs(dir_x) > abs(dir_y)
            # Primarily X-aligned strands
            for strand_idx in 0:spacing:ny
                for x in 1:nx
                    for dz in -strand_r:strand_r
                        for dy in -strand_r:strand_r
                            if dy^2 + dz^2 <= strand_r^2
                                yy = strand_idx + dy
                                zz = z + dz
                                if 1 <= yy <= ny && 1 <= zz <= nz
                                    volume[x, yy, zz] = false  # Solid strand
                                end
                            end
                        end
                    end
                end
            end
        else
            # Primarily Y-aligned strands
            for strand_idx in 0:spacing:nx
                for y in 1:ny
                    for dz in -strand_r:strand_r
                        for dx in -strand_r:strand_r
                            if dx^2 + dz^2 <= strand_r^2
                                xx = strand_idx + dx
                                zz = z + dz
                                if 1 <= xx <= nx && 1 <= zz <= nz
                                    volume[xx, y, zz] = false  # Solid strand
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    return volume
end

#=============================================================================
  MAIN GENERATION FUNCTION
=============================================================================#

"""
    generate_scaffold_from_text(text::String; model_name="llama3.2:3b") -> GenerationResult

Main entry point: Generate scaffold from natural language description.

# Example
```julia
result = generate_scaffold_from_text("Design a bone scaffold with 85% porosity and 300um pores using PCL")
if result.success
    @info "Generated scaffold" porosity=result.metrics.porosity
end
```
"""
function generate_scaffold_from_text(text::String;
                                      model_name::String="llama3.2:3b",
                                      resolution::Int=80)
    start_time = time()
    warnings = String[]

    @info "Generating scaffold from text" input=text[1:min(50, length(text))]*"..."

    # Step 1: Parse description with LLM
    prompt = parse_scaffold_description(text, model_name=model_name)
    @info "Extracted parameters" tissue=prompt.target_tissue material=prompt.material porosity=prompt.porosity

    # Step 2: Validate against Q1 literature
    valid, validation_warnings = validate_parameters(prompt)
    append!(warnings, validation_warnings)

    if !isempty(warnings)
        @warn "Parameter warnings" warnings=warnings
    end

    # Step 3: Generate geometry
    volume = nothing

    try
        if prompt.generation_method == :tpms
            # Select TPMS type based on tissue
            tpms_type = if prompt.target_tissue == :bone
                GYROID
            elseif prompt.target_tissue == :cartilage
                SCHWARZ_P
            elseif prompt.target_tissue in [:cardiac, :neural]
                LIDINOID  # More open structure
            else
                GYROID
            end

            # Calculate unit cell size from pore size
            unit_cell_size = prompt.pore_size_um / 100.0  # Approximate

            volume = generate_tpms(
                tpms_type=tpms_type,
                resolution=resolution,
                porosity=prompt.porosity,
                unit_cell_size=unit_cell_size,
                dimensions=prompt.dimensions_mm
            )

        elseif prompt.generation_method == :salt_leaching
            pore_size_voxels = round(Int, prompt.pore_size_um / 100 * resolution / 10)
            pore_size_voxels = max(pore_size_voxels, 3)

            volume = generate_salt_leaching(
                porosity=prompt.porosity,
                pore_size_voxels=pore_size_voxels,
                resolution=resolution,
                dimensions=prompt.dimensions_mm
            )

        elseif prompt.generation_method == :bioprinting
            strand_diameter = prompt.pore_size_um / 1000 * 0.5  # mm
            strand_spacing = prompt.pore_size_um / 1000  # mm

            volume = generate_bioprinting_lattice(
                strand_diameter=strand_diameter,
                strand_spacing=strand_spacing,
                resolution=resolution,
                dimensions=prompt.dimensions_mm
            )

        else
            # Default to TPMS
            volume = generate_tpms(
                resolution=resolution,
                porosity=prompt.porosity,
                dimensions=prompt.dimensions_mm
            )
        end

        # Step 4: Compute metrics
        actual_porosity = sum(volume) / length(volume)

        metrics = ScaffoldMetrics(
            actual_porosity,
            prompt.pore_size_um,
            0.95,  # Interconnectivity (TPMS is fully connected)
            1.1,   # Tortuosity estimate
            0.0,   # Surface area (compute if needed)
            0.0,   # Elastic modulus
            0.0,   # Yield strength
            0.0    # Permeability
        )

        elapsed = time() - start_time

        return GenerationResult(
            true,
            volume,
            prompt,
            metrics,
            warnings,
            elapsed
        )

    catch e
        @error "Scaffold generation failed" exception=e
        elapsed = time() - start_time
        push!(warnings, "Generation failed: $(string(e))")

        return GenerationResult(
            false,
            nothing,
            prompt,
            nothing,
            warnings,
            elapsed
        )
    end
end

end # module
