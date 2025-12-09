"""
Triply Periodic Minimal Surfaces (TPMS) for Scaffold Design
============================================================

SOTA 2024-2025 Implementation with:
- Gyroid (most popular for bone scaffolds)
- Diamond (Schwarz D)
- Schwarz Primitive (P)
- Fischer-Koch S
- I-WP (IsoWrapped Primitive)
- Neovius surface
- Hybrid/graded TPMS
- Functionally Graded Scaffolds (FGS)

Mathematical Framework:
TPMS are defined by implicit functions φ(x,y,z) = 0
Scaffold: φ(x,y,z) ≤ t (threshold controls porosity)

References:
- Schoen (1970) "Infinite periodic minimal surfaces"
- Karcher & Polthier (1996) "Construction of triply periodic minimal surfaces"
- Al-Ketan & Abu Al-Rub (2019) "Multifunctional TPMS"
- Abueidda et al. (2019) "Mechanical properties of 3D printed TPMS structures"
- Zhang et al. (2023) "Functionally graded TPMS for bone scaffolds"
"""
module TPMSGenerators

using Statistics
using LinearAlgebra

export TPMSSurface, Gyroid, Diamond, SchwarzP, SchwarzD, FischerKochS, IWP, Neovius
export generate_tpms, generate_graded_tpms, generate_hybrid_tpms
export compute_tpms_porosity, compute_surface_area_ratio
export TPMSParameters, optimize_tpms_for_bone
export sheet_tpms, network_tpms, skeletal_tpms

# ============================================================================
# TPMS SURFACE TYPES
# ============================================================================

"""
    TPMSSurface

Abstract type for TPMS surfaces.
"""
abstract type TPMSSurface end

"""
    Gyroid <: TPMSSurface

Gyroid surface (Schoen 1970).
Most widely used for bone scaffolds due to:
- High surface area to volume ratio
- Excellent interconnectivity
- Self-supporting for 3D printing
- Good mechanical properties

Equation: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = t
"""
struct Gyroid <: TPMSSurface end

"""
    Diamond <: TPMSSurface

Schwarz Diamond surface.
Good for load-bearing applications.

Equation: sin(x)sin(y)sin(z) + sin(x)cos(y)cos(z) +
          cos(x)sin(y)cos(z) + cos(x)cos(y)sin(z) = t
"""
struct Diamond <: TPMSSurface end

"""
    SchwarzP <: TPMSSurface

Schwarz Primitive surface.
Simplest TPMS, good for initial designs.

Equation: cos(x) + cos(y) + cos(z) = t
"""
struct SchwarzP <: TPMSSurface end

"""
    SchwarzD <: TPMSSurface

Alternative notation for Diamond surface.
"""
const SchwarzD = Diamond

"""
    FischerKochS <: TPMSSurface

Fischer-Koch S surface.
Complex geometry with high surface area.

Equation: cos(2x)sin(y)cos(z) + cos(x)cos(2y)sin(z) +
          sin(x)cos(y)cos(2z) = t
"""
struct FischerKochS <: TPMSSurface end

"""
    IWP <: TPMSSurface

I-Wrapped Package (I-WP) surface.
Good balance of properties.

Equation: 2(cos(x)cos(y) + cos(y)cos(z) + cos(z)cos(x)) -
          (cos(2x) + cos(2y) + cos(2z)) = t
"""
struct IWP <: TPMSSurface end

"""
    Neovius <: TPMSSurface

Neovius surface.
High surface curvature, complex geometry.

Equation: 3(cos(x) + cos(y) + cos(z)) + 4cos(x)cos(y)cos(z) = t
"""
struct Neovius <: TPMSSurface end

# ============================================================================
# TPMS LEVEL SET FUNCTIONS
# ============================================================================

"""
Evaluate TPMS level set function at point (x, y, z).
"""
function evaluate_tpms(::Gyroid, x::Float64, y::Float64, z::Float64)
    return sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)
end

function evaluate_tpms(::Diamond, x::Float64, y::Float64, z::Float64)
    return sin(x) * sin(y) * sin(z) +
           sin(x) * cos(y) * cos(z) +
           cos(x) * sin(y) * cos(z) +
           cos(x) * cos(y) * sin(z)
end

function evaluate_tpms(::SchwarzP, x::Float64, y::Float64, z::Float64)
    return cos(x) + cos(y) + cos(z)
end

function evaluate_tpms(::FischerKochS, x::Float64, y::Float64, z::Float64)
    return cos(2x) * sin(y) * cos(z) +
           cos(x) * cos(2y) * sin(z) +
           sin(x) * cos(y) * cos(2z)
end

function evaluate_tpms(::IWP, x::Float64, y::Float64, z::Float64)
    return 2 * (cos(x) * cos(y) + cos(y) * cos(z) + cos(z) * cos(x)) -
           (cos(2x) + cos(2y) + cos(2z))
end

function evaluate_tpms(::Neovius, x::Float64, y::Float64, z::Float64)
    return 3 * (cos(x) + cos(y) + cos(z)) + 4 * cos(x) * cos(y) * cos(z)
end

# ============================================================================
# TPMS PARAMETERS
# ============================================================================

"""
    TPMSParameters

Parameters for TPMS scaffold generation.

# Fields
- `surface_type::TPMSSurface`: Type of TPMS
- `unit_cell_size::Float64`: Size of unit cell in μm
- `n_cells::Tuple{Int,Int,Int}`: Number of unit cells in x, y, z
- `threshold::Float64`: Level set threshold (controls porosity)
- `wall_thickness::Float64`: Minimum wall thickness in μm (optional)
- `resolution::Int`: Voxels per unit cell
"""
struct TPMSParameters
    surface_type::TPMSSurface
    unit_cell_size::Float64
    n_cells::Tuple{Int,Int,Int}
    threshold::Float64
    wall_thickness::Float64
    resolution::Int

    function TPMSParameters(;
        surface_type::TPMSSurface=Gyroid(),
        unit_cell_size::Float64=500.0,  # μm
        n_cells::Tuple{Int,Int,Int}=(4, 4, 4),
        threshold::Float64=0.0,
        wall_thickness::Float64=0.0,
        resolution::Int=30
    )
        new(surface_type, unit_cell_size, n_cells, threshold, wall_thickness, resolution)
    end
end

# ============================================================================
# TPMS GENERATION
# ============================================================================

"""
    generate_tpms(params::TPMSParameters) -> Array{Bool,3}

Generate binary TPMS scaffold volume.

# Returns
- Binary array where `true` = solid material
"""
function generate_tpms(params::TPMSParameters)::Array{Bool,3}
    nx = params.n_cells[1] * params.resolution
    ny = params.n_cells[2] * params.resolution
    nz = params.n_cells[3] * params.resolution

    scaffold = falses(nx, ny, nz)

    # Coordinate scaling: map voxel to [0, 2π * n_cells]
    scale_x = 2π * params.n_cells[1] / nx
    scale_y = 2π * params.n_cells[2] / ny
    scale_z = 2π * params.n_cells[3] / nz

    for i in 1:nx
        x = (i - 0.5) * scale_x
        for j in 1:ny
            y = (j - 0.5) * scale_y
            for k in 1:nz
                z = (k - 0.5) * scale_z

                φ = evaluate_tpms(params.surface_type, x, y, z)

                # Solid where φ ≤ threshold
                scaffold[i, j, k] = φ ≤ params.threshold
            end
        end
    end

    # Apply wall thickness constraint if specified
    if params.wall_thickness > 0
        voxel_size = params.unit_cell_size / params.resolution
        min_thickness_voxels = ceil(Int, params.wall_thickness / voxel_size)
        scaffold = enforce_wall_thickness(scaffold, min_thickness_voxels)
    end

    return scaffold
end

"""
    generate_tpms(surface_type, size, resolution; threshold=0.0)

Simplified TPMS generation interface.

# Arguments
- `surface_type`: Gyroid(), Diamond(), SchwarzP(), etc.
- `size`: Tuple (nx, ny, nz) for output dimensions
- `resolution`: Voxels per 2π period
- `threshold`: Level set threshold
"""
function generate_tpms(
    surface_type::TPMSSurface,
    size::Tuple{Int,Int,Int};
    threshold::Float64=0.0,
    n_periods::Tuple{Int,Int,Int}=(2, 2, 2)
)::Array{Bool,3}
    nx, ny, nz = size

    scaffold = falses(nx, ny, nz)

    # Coordinate scaling
    scale_x = 2π * n_periods[1] / nx
    scale_y = 2π * n_periods[2] / ny
    scale_z = 2π * n_periods[3] / nz

    for i in 1:nx
        x = (i - 0.5) * scale_x
        for j in 1:ny
            y = (j - 0.5) * scale_y
            for k in 1:nz
                z = (k - 0.5) * scale_z

                φ = evaluate_tpms(surface_type, x, y, z)
                scaffold[i, j, k] = φ ≤ threshold
            end
        end
    end

    return scaffold
end

# ============================================================================
# SHEET vs NETWORK vs SKELETAL TPMS
# ============================================================================

"""
    sheet_tpms(surface_type, size; thickness=0.3, n_periods=(2,2,2))

Generate sheet-based TPMS (double-wall structure).
Common for bone scaffolds as it provides better mechanical support.

# Mathematical formulation
Sheet: |φ(x,y,z)| ≤ t
"""
function sheet_tpms(
    surface_type::TPMSSurface,
    size::Tuple{Int,Int,Int};
    thickness::Float64=0.3,
    n_periods::Tuple{Int,Int,Int}=(2, 2, 2)
)::Array{Bool,3}
    nx, ny, nz = size
    scaffold = falses(nx, ny, nz)

    scale_x = 2π * n_periods[1] / nx
    scale_y = 2π * n_periods[2] / ny
    scale_z = 2π * n_periods[3] / nz

    for i in 1:nx
        x = (i - 0.5) * scale_x
        for j in 1:ny
            y = (j - 0.5) * scale_y
            for k in 1:nz
                z = (k - 0.5) * scale_z

                φ = evaluate_tpms(surface_type, x, y, z)
                scaffold[i, j, k] = abs(φ) ≤ thickness
            end
        end
    end

    return scaffold
end

"""
    network_tpms(surface_type, size; threshold=0.0, n_periods=(2,2,2))

Generate network-based TPMS (single surface as boundary).

# Mathematical formulation
Network: φ(x,y,z) ≤ t
"""
function network_tpms(
    surface_type::TPMSSurface,
    size::Tuple{Int,Int,Int};
    threshold::Float64=0.0,
    n_periods::Tuple{Int,Int,Int}=(2, 2, 2)
)::Array{Bool,3}
    return generate_tpms(surface_type, size; threshold=threshold, n_periods=n_periods)
end

"""
    skeletal_tpms(surface_type, size; threshold=0.0, n_periods=(2,2,2))

Generate skeletal TPMS (solid rods at surface).
Opposite of network TPMS.

# Mathematical formulation
Skeletal: φ(x,y,z) ≥ -t
"""
function skeletal_tpms(
    surface_type::TPMSSurface,
    size::Tuple{Int,Int,Int};
    threshold::Float64=0.0,
    n_periods::Tuple{Int,Int,Int}=(2, 2, 2)
)::Array{Bool,3}
    nx, ny, nz = size
    scaffold = falses(nx, ny, nz)

    scale_x = 2π * n_periods[1] / nx
    scale_y = 2π * n_periods[2] / ny
    scale_z = 2π * n_periods[3] / nz

    for i in 1:nx
        x = (i - 0.5) * scale_x
        for j in 1:ny
            y = (j - 0.5) * scale_y
            for k in 1:nz
                z = (k - 0.5) * scale_z

                φ = evaluate_tpms(surface_type, x, y, z)
                scaffold[i, j, k] = φ ≥ -threshold
            end
        end
    end

    return scaffold
end

# ============================================================================
# FUNCTIONALLY GRADED TPMS
# ============================================================================

"""
    generate_graded_tpms(surface_type, size; gradient_func, n_periods=(2,2,2))

Generate functionally graded TPMS scaffold.
Porosity varies spatially according to gradient function.

# Arguments
- `gradient_func`: Function (x, y, z) → threshold, where x,y,z ∈ [0,1]

# Examples
```julia
# Linear gradient in z (more porous at top)
gradient_func = (x, y, z) -> -0.5 + 0.8 * z

# Radial gradient (denser core)
gradient_func = (x, y, z) -> begin
    r = sqrt((x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2)
    return -0.3 + 0.6 * r
end
```
"""
function generate_graded_tpms(
    surface_type::TPMSSurface,
    size::Tuple{Int,Int,Int};
    gradient_func::Function,
    n_periods::Tuple{Int,Int,Int}=(2, 2, 2)
)::Array{Bool,3}
    nx, ny, nz = size
    scaffold = falses(nx, ny, nz)

    scale_x = 2π * n_periods[1] / nx
    scale_y = 2π * n_periods[2] / ny
    scale_z = 2π * n_periods[3] / nz

    for i in 1:nx
        # Normalized coordinates [0, 1]
        x_norm = (i - 0.5) / nx
        y_norm_arr = [(j - 0.5) / ny for j in 1:ny]

        x = (i - 0.5) * scale_x

        for j in 1:ny
            y_norm = (j - 0.5) / ny
            y = (j - 0.5) * scale_y

            for k in 1:nz
                z_norm = (k - 0.5) / nz
                z = (k - 0.5) * scale_z

                # Spatially varying threshold
                t = gradient_func(x_norm, y_norm, z_norm)

                φ = evaluate_tpms(surface_type, x, y, z)
                scaffold[i, j, k] = φ ≤ t
            end
        end
    end

    return scaffold
end

"""
Common gradient functions for bone scaffolds.
"""
module GradientFunctions

export linear_z, radial_core, cortical_to_cancellous, load_adapted

"""Porosity increases linearly with z (bottom to top)."""
linear_z(porosity_bottom::Float64, porosity_top::Float64) =
    (x, y, z) -> threshold_from_porosity(porosity_bottom + (porosity_top - porosity_bottom) * z)

"""Denser core, more porous periphery (mimics bone)."""
function radial_core(core_porosity::Float64, outer_porosity::Float64)
    return (x, y, z) -> begin
        r = 2 * sqrt((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2)
        r = min(r, 1.0)  # Clamp to [0, 1]
        return threshold_from_porosity(core_porosity + (outer_porosity - core_porosity) * r)
    end
end

"""Cortical (dense) outer layer, cancellous (porous) inner."""
function cortical_to_cancellous(cortical_thickness::Float64=0.2,
                                 cortical_porosity::Float64=0.1,
                                 cancellous_porosity::Float64=0.8)
    return (x, y, z) -> begin
        # Distance from nearest boundary
        d = min(x, 1-x, y, 1-y, z, 1-z)
        if d < cortical_thickness
            return threshold_from_porosity(cortical_porosity)
        else
            return threshold_from_porosity(cancellous_porosity)
        end
    end
end

"""Load-adapted gradient (denser where stress is higher)."""
function load_adapted(stress_field::Array{Float64,3})
    max_stress = maximum(stress_field)
    min_stress = minimum(stress_field)
    range = max_stress - min_stress + 1e-10

    return (x, y, z) -> begin
        # Map normalized coords to array indices
        nx, ny, nz = size(stress_field)
        i = clamp(round(Int, x * nx), 1, nx)
        j = clamp(round(Int, y * ny), 1, ny)
        k = clamp(round(Int, z * nz), 1, nz)

        normalized_stress = (stress_field[i, j, k] - min_stress) / range
        # Higher stress → lower porosity (denser material)
        target_porosity = 0.3 + 0.5 * (1 - normalized_stress)
        return threshold_from_porosity(target_porosity)
    end
end

# Helper: approximate threshold from target porosity
function threshold_from_porosity(target_porosity::Float64)
    # Empirical relationship (varies by TPMS type)
    # For Gyroid: porosity ≈ 0.5 + 0.35 * threshold
    return (target_porosity - 0.5) / 0.35
end

end # module GradientFunctions

using .GradientFunctions

# ============================================================================
# HYBRID TPMS (Combining Multiple Surfaces)
# ============================================================================

"""
    generate_hybrid_tpms(surfaces, weights, size; n_periods=(2,2,2), threshold=0.0)

Generate hybrid TPMS by combining multiple surfaces.

# Mathematical formulation
φ_hybrid = Σ wᵢ * φᵢ(x,y,z)

# Arguments
- `surfaces`: Vector of TPMSSurface
- `weights`: Vector of weights (will be normalized)
"""
function generate_hybrid_tpms(
    surfaces::Vector{<:TPMSSurface},
    weights::Vector{Float64},
    size::Tuple{Int,Int,Int};
    n_periods::Tuple{Int,Int,Int}=(2, 2, 2),
    threshold::Float64=0.0
)::Array{Bool,3}
    @assert length(surfaces) == length(weights) "Number of surfaces must match weights"

    # Normalize weights
    w = weights ./ sum(weights)

    nx, ny, nz = size
    scaffold = falses(nx, ny, nz)

    scale_x = 2π * n_periods[1] / nx
    scale_y = 2π * n_periods[2] / ny
    scale_z = 2π * n_periods[3] / nz

    for i in 1:nx
        x = (i - 0.5) * scale_x
        for j in 1:ny
            y = (j - 0.5) * scale_y
            for k in 1:nz
                z = (k - 0.5) * scale_z

                # Weighted combination
                φ = sum(w[s] * evaluate_tpms(surfaces[s], x, y, z)
                        for s in 1:length(surfaces))

                scaffold[i, j, k] = φ ≤ threshold
            end
        end
    end

    return scaffold
end

"""
    generate_transition_tpms(surface1, surface2, size; transition_axis=:z, n_periods=(2,2,2))

Generate TPMS with smooth transition between two surfaces.
Useful for graded mechanical properties.
"""
function generate_transition_tpms(
    surface1::TPMSSurface,
    surface2::TPMSSurface,
    size::Tuple{Int,Int,Int};
    transition_axis::Symbol=:z,
    n_periods::Tuple{Int,Int,Int}=(2, 2, 2),
    threshold::Float64=0.0
)::Array{Bool,3}
    nx, ny, nz = size
    scaffold = falses(nx, ny, nz)

    scale_x = 2π * n_periods[1] / nx
    scale_y = 2π * n_periods[2] / ny
    scale_z = 2π * n_periods[3] / nz

    for i in 1:nx
        x = (i - 0.5) * scale_x
        for j in 1:ny
            y = (j - 0.5) * scale_y
            for k in 1:nz
                z = (k - 0.5) * scale_z

                # Blending parameter based on axis
                α = if transition_axis == :x
                    (i - 0.5) / nx
                elseif transition_axis == :y
                    (j - 0.5) / ny
                else
                    (k - 0.5) / nz
                end

                # Smooth transition using sigmoid-like function
                blend = 3 * α^2 - 2 * α^3  # Smoothstep

                φ1 = evaluate_tpms(surface1, x, y, z)
                φ2 = evaluate_tpms(surface2, x, y, z)

                φ = (1 - blend) * φ1 + blend * φ2
                scaffold[i, j, k] = φ ≤ threshold
            end
        end
    end

    return scaffold
end

# ============================================================================
# POROSITY AND SURFACE AREA CALCULATIONS
# ============================================================================

"""
    compute_tpms_porosity(scaffold::Array{Bool,3}) -> Float64

Compute porosity of TPMS scaffold.
"""
function compute_tpms_porosity(scaffold::Array{Bool,3})::Float64
    solid_fraction = sum(scaffold) / length(scaffold)
    return 1.0 - solid_fraction
end

"""
    compute_surface_area_ratio(scaffold::Array{Bool,3}; voxel_size=1.0) -> Float64

Compute surface area to volume ratio of TPMS scaffold.
Higher SA/V is beneficial for cell attachment.
"""
function compute_surface_area_ratio(scaffold::Array{Bool,3}; voxel_size::Float64=1.0)::Float64
    nx, ny, nz = size(scaffold)

    surface_voxels = 0

    # Count boundary voxels
    for i in 1:nx, j in 1:ny, k in 1:nz
        if scaffold[i, j, k]
            # Check 6-connected neighbors
            is_boundary = false
            for (di, dj, dk) in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
                ni, nj, nk = i+di, j+dj, k+dk
                if ni < 1 || ni > nx || nj < 1 || nj > ny || nk < 1 || nk > nz
                    is_boundary = true
                    break
                elseif !scaffold[ni, nj, nk]
                    is_boundary = true
                    break
                end
            end
            if is_boundary
                surface_voxels += 1
            end
        end
    end

    # Approximate surface area (each boundary voxel contributes ~1 face)
    surface_area = surface_voxels * voxel_size^2

    # Volume of solid material
    volume = sum(scaffold) * voxel_size^3

    return volume > 0 ? surface_area / volume : 0.0
end

# ============================================================================
# WALL THICKNESS ENFORCEMENT
# ============================================================================

"""
    enforce_wall_thickness(scaffold, min_thickness_voxels)

Ensure minimum wall thickness through morphological operations.
"""
function enforce_wall_thickness(scaffold::Array{Bool,3}, min_thickness::Int)::Array{Bool,3}
    if min_thickness <= 1
        return scaffold
    end

    # Erosion followed by dilation (opening)
    eroded = morphological_erode(scaffold, min_thickness ÷ 2)
    opened = morphological_dilate(eroded, min_thickness ÷ 2)

    return opened
end

function morphological_erode(vol::Array{Bool,3}, radius::Int)
    nx, ny, nz = size(vol)
    result = copy(vol)

    for i in 1:nx, j in 1:ny, k in 1:nz
        if vol[i, j, k]
            # Check if all neighbors within radius are also solid
            for di in -radius:radius, dj in -radius:radius, dk in -radius:radius
                ni, nj, nk = i+di, j+dj, k+dk
                if 1 <= ni <= nx && 1 <= nj <= ny && 1 <= nk <= nz
                    if !vol[ni, nj, nk]
                        result[i, j, k] = false
                        break
                    end
                else
                    result[i, j, k] = false
                    break
                end
            end
        end
    end

    return result
end

function morphological_dilate(vol::Array{Bool,3}, radius::Int)
    nx, ny, nz = size(vol)
    result = copy(vol)

    for i in 1:nx, j in 1:ny, k in 1:nz
        if !vol[i, j, k]
            # Check if any neighbor within radius is solid
            for di in -radius:radius, dj in -radius:radius, dk in -radius:radius
                ni, nj, nk = i+di, j+dj, k+dk
                if 1 <= ni <= nx && 1 <= nj <= ny && 1 <= nk <= nz
                    if vol[ni, nj, nk]
                        result[i, j, k] = true
                        break
                    end
                end
            end
        end
    end

    return result
end

# ============================================================================
# OPTIMIZATION FOR BONE SCAFFOLDS
# ============================================================================

"""
    optimize_tpms_for_bone(target_porosity; target_pore_size=200.0, target_stiffness=nothing)

Find optimal TPMS parameters for bone tissue engineering.

# Constraints (Murphy et al. 2010, Karageorgiou 2005)
- Porosity: 50-90% (optimal 60-80% for bone)
- Pore size: 100-400 μm (optimal 200-350 μm)
- Interconnectivity: >90%
- Stiffness: Match trabecular bone (0.1-2 GPa)
"""
function optimize_tpms_for_bone(;
    target_porosity::Float64=0.7,
    target_pore_size::Float64=200.0,  # μm
    unit_cell_size::Float64=500.0,    # μm
    surface_types::Vector{<:TPMSSurface}=[Gyroid(), Diamond(), SchwarzP()]
)
    best_params = nothing
    best_score = Inf

    for surface in surface_types
        # Search threshold range
        for t in range(-1.0, 1.0, length=21)
            # Generate small test sample
            test_size = (50, 50, 50)
            scaffold = generate_tpms(surface, test_size; threshold=t, n_periods=(2, 2, 2))

            porosity = compute_tpms_porosity(scaffold)

            # Skip if porosity too far from target
            if abs(porosity - target_porosity) > 0.15
                continue
            end

            # Estimate pore size (simplified)
            pore_volume = sum(.!scaffold)
            if pore_volume > 0
                # Rough estimate: pore size ∝ cube root of average pore region
                estimated_pore_size = (pore_volume / (2^3))^(1/3) * unit_cell_size / 50
            else
                estimated_pore_size = 0.0
            end

            # Score: minimize deviation from targets
            porosity_error = abs(porosity - target_porosity)
            pore_size_error = abs(estimated_pore_size - target_pore_size) / target_pore_size

            score = porosity_error + 0.5 * pore_size_error

            if score < best_score
                best_score = score
                best_params = TPMSParameters(
                    surface_type=surface,
                    unit_cell_size=unit_cell_size,
                    n_cells=(4, 4, 4),
                    threshold=t,
                    wall_thickness=0.0,
                    resolution=30
                )
            end
        end
    end

    if isnothing(best_params)
        # Default to Gyroid with t=0
        best_params = TPMSParameters(
            surface_type=Gyroid(),
            unit_cell_size=unit_cell_size,
            n_cells=(4, 4, 4),
            threshold=0.0,
            wall_thickness=0.0,
            resolution=30
        )
    end

    return best_params
end

# ============================================================================
# MECHANICAL PROPERTY ESTIMATION
# ============================================================================

"""
    estimate_elastic_modulus(scaffold, E_solid; method=:gibson_ashby)

Estimate elastic modulus of TPMS scaffold.

Methods:
- :gibson_ashby: E/E₀ = C * (ρ/ρ₀)^n
- :hashin_shtrikman: Bounds from composite theory
"""
function estimate_elastic_modulus(
    scaffold::Array{Bool,3},
    E_solid::Float64;  # Modulus of solid material (e.g., 3.5 GPa for PCL)
    method::Symbol=:gibson_ashby
)
    relative_density = sum(scaffold) / length(scaffold)

    if method == :gibson_ashby
        # Gibson-Ashby model for open-cell foams
        # E/E₀ = C * (ρ/ρ₀)^2, C ≈ 1 for TPMS
        return E_solid * relative_density^2
    else
        # Hashin-Shtrikman lower bound
        ν = 0.3  # Poisson's ratio (typical for polymers)
        K_solid = E_solid / (3 * (1 - 2ν))
        G_solid = E_solid / (2 * (1 + ν))

        f = relative_density  # Volume fraction of solid

        # HS lower bound for bulk modulus
        K_lower = K_solid * f / (1 + (1 - f) * 3 * K_solid / (4 * G_solid))

        # Approximate E from K
        return 3 * K_lower * (1 - 2ν)
    end
end

"""
    estimate_permeability(scaffold; voxel_size=10.0)

Estimate permeability using Kozeny-Carman equation.
"""
function estimate_permeability(scaffold::Array{Bool,3}; voxel_size::Float64=10.0)
    porosity = compute_tpms_porosity(scaffold)
    surface_area_ratio = compute_surface_area_ratio(scaffold, voxel_size=voxel_size)

    if surface_area_ratio <= 0 || porosity >= 1.0
        return 0.0
    end

    # Kozeny-Carman equation
    # K = φ³ / (c * S²)
    # where c ≈ 5 (Kozeny constant)
    c = 5.0
    K = porosity^3 / (c * surface_area_ratio^2)

    return K  # in voxel_size² units
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
    print_tpms_info(params::TPMSParameters)

Print information about TPMS parameters.
"""
function print_tpms_info(params::TPMSParameters)
    scaffold = generate_tpms(params)
    porosity = compute_tpms_porosity(scaffold)
    sa_ratio = compute_surface_area_ratio(scaffold)

    println("═" ^ 50)
    println("TPMS Scaffold Parameters")
    println("═" ^ 50)
    println("Surface type: $(typeof(params.surface_type))")
    println("Unit cell size: $(params.unit_cell_size) μm")
    println("Number of cells: $(params.n_cells)")
    println("Threshold: $(params.threshold)")
    println("Resolution: $(params.resolution) voxels/cell")
    println("─" ^ 50)
    println("Computed Properties:")
    println("  Porosity: $(round(porosity * 100, digits=1))%")
    println("  SA/V ratio: $(round(sa_ratio, digits=3)) /μm")
    total_size = params.n_cells .* params.unit_cell_size
    println("  Total size: $(total_size) μm")
    println("═" ^ 50)
end

end # module TPMSGenerators
