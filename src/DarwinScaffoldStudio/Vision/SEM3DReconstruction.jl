"""
SEM 3D Reconstruction Module

Reconstruct 3D surface/volume from 2D SEM images using multiple techniques:
1. Shape-from-Shading (SfS) - single image depth estimation
2. Stereo SEM - multi-view reconstruction
3. Deep Learning - learned depth priors
4. Focus Stacking - depth from focus variation

References:
- Horn (1970) "Shape from Shading: A Method for Obtaining the Shape of a Smooth Opaque Object"
- Marinello et al. (2008) "Critical factors in SEM 3D stereo microscopy"
- Tafti et al. (2015) "Recent advances in 3D SEM surface reconstruction"
"""

module SEM3DReconstruction

using Images
using ImageFiltering
using Statistics
using LinearAlgebra

# Try to use FFTW if available, otherwise use fallback
const FFTW_AVAILABLE = Ref(false)

function __init__()
    try
        @eval Main using FFTW
        FFTW_AVAILABLE[] = true
    catch
        FFTW_AVAILABLE[] = false
    end
end

export reconstruct_depth_sfs, reconstruct_stereo_sem, reconstruct_focus_stack
export SEM3DResult, estimate_surface_normals, depth_to_mesh

"""
    SEM3DResult

Result of SEM 3D reconstruction.

# Fields
- `depth_map::Matrix{Float64}`: Reconstructed depth (μm)
- `normal_map::Array{Float64,3}`: Surface normals (Nx, Ny, Nz)
- `confidence::Matrix{Float64}`: Reconstruction confidence (0-1)
- `method::String`: Method used
- `parameters::Dict`: Reconstruction parameters
"""
struct SEM3DResult
    depth_map::Matrix{Float64}
    normal_map::Array{Float64,3}
    confidence::Matrix{Float64}
    method::String
    parameters::Dict{String,Any}
end

# ============================================================================
# SHAPE FROM SHADING (SINGLE IMAGE)
# ============================================================================

"""
    reconstruct_depth_sfs(sem_image::AbstractMatrix;
                          light_direction::Vector{Float64}=[0.0, 0.0, 1.0],
                          albedo::Float64=0.5,
                          max_iterations::Int=1000,
                          tolerance::Float64=1e-6) -> SEM3DResult

Reconstruct 3D depth from single SEM image using Shape-from-Shading.

# Arguments
- `sem_image`: Grayscale SEM image (normalized 0-1)
- `light_direction`: Direction of electron beam (default: top-down)
- `albedo`: Surface reflectance (default: 0.5)
- `max_iterations`: Maximum iterations for solver
- `tolerance`: Convergence tolerance

# Returns
- `SEM3DResult` with depth map and normals

# Notes
SEM imaging approximates Lambertian reflectance model:
I = ρ * (L · N) where ρ = albedo, L = light, N = surface normal

For SEM with secondary electrons:
- Bright = surfaces facing detector
- Dark = surfaces away from detector
- Edge effect = enhanced edges
"""
function reconstruct_depth_sfs(
    sem_image::AbstractMatrix;
    light_direction::Vector{Float64}=[0.0, 0.0, 1.0],
    albedo::Float64=0.5,
    max_iterations::Int=1000,
    tolerance::Float64=1e-6
)::SEM3DResult
    # Normalize image
    img = Float64.(sem_image)
    if maximum(img) > 1.0
        img = img ./ maximum(img)
    end

    h, w = size(img)

    # Normalize light direction
    L = light_direction ./ norm(light_direction)

    # Initialize depth and gradients
    Z = zeros(Float64, h, w)
    p = zeros(Float64, h, w)  # dZ/dx
    q = zeros(Float64, h, w)  # dZ/dy

    # Horn's iterative method for SfS
    # Reflectance map R(p,q) = L · N = (Lx*p + Ly*q + Lz) / sqrt(1 + p² + q²)

    λ = 1.0  # Smoothness weight

    for iter in 1:max_iterations
        p_old = copy(p)
        q_old = copy(q)

        # Update p and q using Jacobi iteration
        for i in 2:(h-1)
            for j in 2:(w-1)
                # Average of neighbors
                p_avg = (p[i-1,j] + p[i+1,j] + p[i,j-1] + p[i,j+1]) / 4
                q_avg = (q[i-1,j] + q[i+1,j] + q[i,j-1] + q[i,j+1]) / 4

                # Reflectance at current (p,q)
                denom = sqrt(1.0 + p_avg^2 + q_avg^2)
                R = (L[1]*p_avg + L[2]*q_avg + L[3]) / denom

                # Image irradiance equation: I = ρ * R
                E = img[i,j] / albedo

                # Error
                error = E - R

                # Gradient of R with respect to p and q
                dR_dp = (L[1] * denom - (L[1]*p_avg + L[2]*q_avg + L[3]) * p_avg / denom) / denom^2
                dR_dq = (L[2] * denom - (L[1]*p_avg + L[2]*q_avg + L[3]) * q_avg / denom) / denom^2

                # Update
                factor = error / (λ + dR_dp^2 + dR_dq^2 + 1e-10)
                p[i,j] = p_avg + factor * dR_dp
                q[i,j] = q_avg + factor * dR_dq
            end
        end

        # Check convergence
        diff_p = maximum(abs.(p .- p_old))
        diff_q = maximum(abs.(q .- q_old))

        if max(diff_p, diff_q) < tolerance
            @info "SfS converged at iteration $iter"
            break
        end
    end

    # Integrate gradients to get depth (Frankot-Chellappa)
    Z = integrate_gradients_fc(p, q)

    # Compute normals from gradients
    normals = gradients_to_normals(p, q)

    # Confidence based on reflectance fit
    confidence = compute_sfs_confidence(img, p, q, L, albedo)

    return SEM3DResult(
        Z,
        normals,
        confidence,
        "shape_from_shading",
        Dict{String,Any}(
            "light_direction" => light_direction,
            "albedo" => albedo,
            "iterations" => max_iterations
        )
    )
end

"""
Frankot-Chellappa gradient integration.
Uses Fourier domain if FFTW available, otherwise falls back to iterative Poisson solver.
"""
function integrate_gradients_fc(p::Matrix{Float64}, q::Matrix{Float64})::Matrix{Float64}
    if FFTW_AVAILABLE[]
        return integrate_gradients_fft(p, q)
    else
        return integrate_gradients_poisson(p, q)
    end
end

"""
FFT-based Frankot-Chellappa integration (requires FFTW).
"""
function integrate_gradients_fft(p::Matrix{Float64}, q::Matrix{Float64})::Matrix{Float64}
    h, w = size(p)

    # Fourier transforms (use Main.FFTW to avoid import issues)
    P = Main.FFTW.fft(p)
    Q = Main.FFTW.fft(q)

    # Frequency grids
    u = fftfreq(w)
    v = fftfreq(h)

    # Integration in frequency domain
    # Z_hat = (-i*u*P - i*v*Q) / (u² + v²)
    Z_hat = zeros(ComplexF64, h, w)

    for j in 1:w
        for i in 1:h
            denom = u[j]^2 + v[i]^2
            if denom > 1e-10
                Z_hat[i,j] = (-im * u[j] * P[i,j] - im * v[i] * Q[i,j]) / denom
            end
        end
    end

    # Inverse FFT
    Z = real.(Main.FFTW.ifft(Z_hat))

    # Remove mean and scale
    Z = Z .- mean(Z)

    return Z
end

"""
Poisson solver for gradient integration (no FFTW required).
Uses Gauss-Seidel iteration to solve ∇²Z = ∂p/∂x + ∂q/∂y
"""
function integrate_gradients_poisson(p::Matrix{Float64}, q::Matrix{Float64};
                                      max_iter::Int=500, tol::Float64=1e-4)::Matrix{Float64}
    h, w = size(p)

    # Compute divergence of gradient field: div = ∂p/∂x + ∂q/∂y
    div = zeros(Float64, h, w)
    for i in 2:(h-1)
        for j in 2:(w-1)
            dp_dx = (p[i, j+1] - p[i, j-1]) / 2.0
            dq_dy = (q[i+1, j] - q[i-1, j]) / 2.0
            div[i, j] = dp_dx + dq_dy
        end
    end

    # Initialize depth
    Z = zeros(Float64, h, w)

    # Gauss-Seidel iteration for Poisson equation: ∇²Z = div
    for iter in 1:max_iter
        max_change = 0.0

        for i in 2:(h-1)
            for j in 2:(w-1)
                # Laplacian stencil: Z_new = (Z_left + Z_right + Z_up + Z_down - h²*div) / 4
                Z_new = (Z[i-1,j] + Z[i+1,j] + Z[i,j-1] + Z[i,j+1] - div[i,j]) / 4.0
                change = abs(Z_new - Z[i,j])
                if change > max_change
                    max_change = change
                end
                Z[i,j] = Z_new
            end
        end

        if max_change < tol
            break
        end
    end

    # Remove mean
    Z = Z .- mean(Z)

    return Z
end

"""
Create frequency grid for FFT.
"""
function fftfreq(n::Int)::Vector{Float64}
    freq = zeros(Float64, n)
    for i in 1:n
        if i <= div(n, 2) + 1
            freq[i] = (i - 1) / n
        else
            freq[i] = (i - 1 - n) / n
        end
    end
    return freq
end

"""
Convert gradient fields to surface normals.
"""
function gradients_to_normals(p::Matrix{Float64}, q::Matrix{Float64})::Array{Float64,3}
    h, w = size(p)
    normals = zeros(Float64, h, w, 3)

    for i in 1:h
        for j in 1:w
            # Normal = (-p, -q, 1) normalized
            nx = -p[i,j]
            ny = -q[i,j]
            nz = 1.0
            len = sqrt(nx^2 + ny^2 + nz^2)

            normals[i,j,1] = nx / len
            normals[i,j,2] = ny / len
            normals[i,j,3] = nz / len
        end
    end

    return normals
end

"""
Compute SfS reconstruction confidence.
"""
function compute_sfs_confidence(
    img::Matrix{Float64},
    p::Matrix{Float64},
    q::Matrix{Float64},
    L::Vector{Float64},
    albedo::Float64
)::Matrix{Float64}
    h, w = size(img)
    confidence = zeros(Float64, h, w)

    for i in 1:h
        for j in 1:w
            # Predicted reflectance
            denom = sqrt(1.0 + p[i,j]^2 + q[i,j]^2)
            R_pred = (L[1]*p[i,j] + L[2]*q[i,j] + L[3]) / denom
            I_pred = albedo * R_pred

            # Error
            error = abs(img[i,j] - I_pred)

            # Confidence = 1 - normalized error
            confidence[i,j] = max(0.0, 1.0 - error)
        end
    end

    return confidence
end

# ============================================================================
# STEREO SEM (MULTI-VIEW)
# ============================================================================

"""
    reconstruct_stereo_sem(left_image::AbstractMatrix,
                           right_image::AbstractMatrix;
                           tilt_angle::Float64=5.0,
                           pixel_size_um::Float64=1.0,
                           working_distance_mm::Float64=10.0) -> SEM3DResult

Reconstruct 3D from stereo SEM pair (tilted specimen).

# Arguments
- `left_image`: First SEM image (0° tilt)
- `right_image`: Second SEM image (tilted)
- `tilt_angle`: Tilt angle in degrees
- `pixel_size_um`: Pixel size in micrometers
- `working_distance_mm`: Working distance in mm

# Returns
- `SEM3DResult` with depth map

# Notes
Stereo SEM uses parallax from tilting the specimen.
Depth Z = Δx / (2 * sin(θ/2) * M)
where Δx = disparity, θ = tilt angle, M = magnification
"""
function reconstruct_stereo_sem(
    left_image::AbstractMatrix,
    right_image::AbstractMatrix;
    tilt_angle::Float64=5.0,
    pixel_size_um::Float64=1.0,
    working_distance_mm::Float64=10.0
)::SEM3DResult
    # Normalize images
    left = Float64.(left_image)
    right = Float64.(right_image)

    if maximum(left) > 1.0
        left = left ./ maximum(left)
    end
    if maximum(right) > 1.0
        right = right ./ maximum(right)
    end

    h, w = size(left)

    # Compute disparity map using block matching
    disparity = compute_disparity_map(left, right, block_size=11, max_disparity=50)

    # Convert disparity to depth
    # For eucentric tilting: Z = d * pixel_size / (2 * sin(θ/2))
    θ_rad = deg2rad(tilt_angle)
    depth_scale = pixel_size_um / (2 * sin(θ_rad / 2))

    depth_map = disparity .* depth_scale

    # Compute normals from depth gradient
    p, q = compute_depth_gradients(depth_map)
    normals = gradients_to_normals(p, q)

    # Confidence based on matching quality
    confidence = compute_stereo_confidence(left, right, disparity)

    return SEM3DResult(
        depth_map,
        normals,
        confidence,
        "stereo_sem",
        Dict{String,Any}(
            "tilt_angle" => tilt_angle,
            "pixel_size_um" => pixel_size_um,
            "working_distance_mm" => working_distance_mm
        )
    )
end

"""
Block matching for disparity estimation.
"""
function compute_disparity_map(
    left::Matrix{Float64},
    right::Matrix{Float64};
    block_size::Int=11,
    max_disparity::Int=50
)::Matrix{Float64}
    h, w = size(left)
    half_block = div(block_size, 2)
    disparity = zeros(Float64, h, w)

    for i in (half_block+1):(h-half_block)
        for j in (half_block+1+max_disparity):(w-half_block)
            # Extract reference block from left image
            block_left = left[(i-half_block):(i+half_block),
                             (j-half_block):(j+half_block)]

            best_ssd = Inf
            best_d = 0

            # Search for best match in right image
            for d in 0:max_disparity
                j_right = j - d
                if j_right - half_block < 1
                    continue
                end

                block_right = right[(i-half_block):(i+half_block),
                                   (j_right-half_block):(j_right+half_block)]

                # Sum of squared differences
                ssd = sum((block_left .- block_right).^2)

                if ssd < best_ssd
                    best_ssd = ssd
                    best_d = d
                end
            end

            disparity[i,j] = Float64(best_d)
        end
    end

    return disparity
end

"""
Compute depth gradients using Sobel operator.
"""
function compute_depth_gradients(depth::Matrix{Float64})::Tuple{Matrix{Float64}, Matrix{Float64}}
    # Sobel kernels
    sobel_x = [-1 0 1; -2 0 2; -1 0 1] ./ 8.0
    sobel_y = [-1 -2 -1; 0 0 0; 1 2 1] ./ 8.0

    p = imfilter(depth, sobel_x)
    q = imfilter(depth, sobel_y)

    return (p, q)
end

"""
Compute stereo matching confidence.
"""
function compute_stereo_confidence(
    left::Matrix{Float64},
    right::Matrix{Float64},
    disparity::Matrix{Float64}
)::Matrix{Float64}
    h, w = size(left)
    confidence = zeros(Float64, h, w)

    # Confidence based on texture (low texture = low confidence)
    texture = imfilter(left, Kernel.LoG(1.0))
    texture_norm = abs.(texture) ./ (maximum(abs.(texture)) + 1e-10)

    # Also penalize very high disparities
    disp_confidence = 1.0 .- (disparity ./ (maximum(disparity) + 1e-10))

    confidence = texture_norm .* disp_confidence

    return confidence
end

# ============================================================================
# FOCUS STACKING (DEPTH FROM FOCUS)
# ============================================================================

"""
    reconstruct_focus_stack(image_stack::Vector{<:AbstractMatrix},
                            focus_positions_um::Vector{Float64}) -> SEM3DResult

Reconstruct depth from focus stack (images at different focus distances).

# Arguments
- `image_stack`: Vector of images at different focus positions
- `focus_positions_um`: Z position of each image (micrometers)

# Returns
- `SEM3DResult` with depth map

# Notes
Each pixel's depth is estimated from the focus position where it appears sharpest.
Uses variance of Laplacian as focus measure.
"""
function reconstruct_focus_stack(
    image_stack::Vector{<:AbstractMatrix},
    focus_positions_um::Vector{Float64}
)::SEM3DResult
    n_images = length(image_stack)

    if n_images != length(focus_positions_um)
        error("Number of images must match number of focus positions")
    end

    # Normalize images
    stack = [Float64.(img) ./ maximum(Float64.(img)) for img in image_stack]

    h, w = size(stack[1])

    # Compute focus measure for each image (variance of Laplacian)
    focus_measures = [compute_focus_measure(img) for img in stack]

    # For each pixel, find the focus position with maximum sharpness
    depth_map = zeros(Float64, h, w)
    confidence = zeros(Float64, h, w)

    for i in 1:h
        for j in 1:w
            # Get focus measures at this pixel
            measures = [fm[i,j] for fm in focus_measures]

            # Find maximum
            max_idx = argmax(measures)
            max_val = measures[max_idx]

            # Subpixel refinement using parabolic fit
            if max_idx > 1 && max_idx < n_images && max_val > 0
                # Fit parabola to 3 points
                y1 = measures[max_idx - 1]
                y2 = measures[max_idx]
                y3 = measures[max_idx + 1]

                z1 = focus_positions_um[max_idx - 1]
                z2 = focus_positions_um[max_idx]
                z3 = focus_positions_um[max_idx + 1]

                # Parabolic interpolation
                denom = (z1 - z2) * (z1 - z3) * (z2 - z3)
                if abs(denom) > 1e-10
                    a = (z3 * (y2 - y1) + z2 * (y1 - y3) + z1 * (y3 - y2)) / denom
                    b = (z3^2 * (y1 - y2) + z2^2 * (y3 - y1) + z1^2 * (y2 - y3)) / denom

                    if abs(a) > 1e-10
                        depth_map[i,j] = -b / (2 * a)
                    else
                        depth_map[i,j] = z2
                    end
                else
                    depth_map[i,j] = z2
                end
            else
                depth_map[i,j] = focus_positions_um[max_idx]
            end

            # Confidence based on peak sharpness
            confidence[i,j] = max_val / (sum(measures) + 1e-10)
        end
    end

    # Compute normals
    p, q = compute_depth_gradients(depth_map)
    normals = gradients_to_normals(p, q)

    return SEM3DResult(
        depth_map,
        normals,
        confidence,
        "focus_stacking",
        Dict{String,Any}(
            "n_images" => n_images,
            "focus_range_um" => (minimum(focus_positions_um), maximum(focus_positions_um))
        )
    )
end

"""
Compute focus measure using variance of Laplacian.
"""
function compute_focus_measure(img::Matrix{Float64}; window_size::Int=9)::Matrix{Float64}
    h, w = size(img)
    half_win = div(window_size, 2)

    # Laplacian
    laplacian = imfilter(img, Kernel.Laplacian())

    # Local variance of Laplacian
    focus = zeros(Float64, h, w)

    for i in (half_win+1):(h-half_win)
        for j in (half_win+1):(w-half_win)
            window = laplacian[(i-half_win):(i+half_win), (j-half_win):(j+half_win)]
            focus[i,j] = var(window)
        end
    end

    return focus
end

# ============================================================================
# DEPTH TO MESH CONVERSION
# ============================================================================

"""
    depth_to_mesh(depth_map::Matrix{Float64}, pixel_size_um::Float64;
                  simplify::Bool=true, target_faces::Int=50000) -> Tuple

Convert depth map to triangle mesh.

# Arguments
- `depth_map`: 2D depth array (μm)
- `pixel_size_um`: XY pixel size
- `simplify`: Apply mesh simplification
- `target_faces`: Target number of faces after simplification

# Returns
- `(vertices, faces)` tuple
"""
function depth_to_mesh(
    depth_map::Matrix{Float64},
    pixel_size_um::Float64;
    simplify::Bool=true,
    target_faces::Int=50000
)::Tuple{Matrix{Float64}, Matrix{Int}}
    h, w = size(depth_map)

    # Create vertices (one per pixel)
    n_vertices = h * w
    vertices = Matrix{Float64}(undef, n_vertices, 3)

    idx = 1
    for i in 1:h
        for j in 1:w
            vertices[idx, 1] = (j - 1) * pixel_size_um  # X
            vertices[idx, 2] = (i - 1) * pixel_size_um  # Y
            vertices[idx, 3] = depth_map[i, j]          # Z
            idx += 1
        end
    end

    # Create faces (2 triangles per quad)
    n_quads = (h - 1) * (w - 1)
    faces = Matrix{Int}(undef, n_quads * 2, 3)

    fidx = 1
    for i in 1:(h-1)
        for j in 1:(w-1)
            # Vertex indices (1-based, row-major)
            v00 = (i - 1) * w + j
            v01 = (i - 1) * w + j + 1
            v10 = i * w + j
            v11 = i * w + j + 1

            # Triangle 1: v00, v10, v01
            faces[fidx, :] = [v00, v10, v01]
            fidx += 1

            # Triangle 2: v01, v10, v11
            faces[fidx, :] = [v01, v10, v11]
            fidx += 1
        end
    end

    # Simplification (quadric error decimation placeholder)
    if simplify && size(faces, 1) > target_faces
        # Simple uniform subsampling for now
        step = ceil(Int, sqrt(size(faces, 1) / target_faces))

        # Subsample depth map
        h_new = div(h, step)
        w_new = div(w, step)
        depth_sub = depth_map[1:step:end, 1:step:end][1:h_new, 1:w_new]

        # Recursively create mesh from subsampled depth
        return depth_to_mesh(depth_sub, pixel_size_um * step, simplify=false, target_faces=target_faces)
    end

    return (vertices, faces)
end

"""
    estimate_surface_normals(sem_image::AbstractMatrix) -> Array{Float64,3}

Quick normal estimation from SEM image gradients (no full reconstruction).
Useful for visualization and roughness analysis.
"""
function estimate_surface_normals(sem_image::AbstractMatrix)::Array{Float64,3}
    img = Float64.(sem_image)
    if maximum(img) > 1.0
        img = img ./ maximum(img)
    end

    # Gradient estimation
    sobel_x = [-1 0 1; -2 0 2; -1 0 1] ./ 8.0
    sobel_y = [-1 -2 -1; 0 0 0; 1 2 1] ./ 8.0

    gx = imfilter(img, sobel_x)
    gy = imfilter(img, sobel_y)

    # Assume intensity correlates with slope
    # Normals from gradients
    return gradients_to_normals(gx, gy)
end

# ============================================================================
# DEEP LEARNING DEPTH ESTIMATION (PLACEHOLDER)
# ============================================================================

"""
    reconstruct_depth_dl(sem_image::AbstractMatrix;
                         model::String="midas") -> SEM3DResult

Deep learning-based depth estimation (requires trained model).

# Notes
This is a placeholder for integration with:
- MiDaS (Intel)
- DPT (Vision Transformer for depth)
- Custom SEM-trained models

The actual implementation would call Python models via PyCall or ONNX.jl
"""
function reconstruct_depth_dl(
    sem_image::AbstractMatrix;
    model::String="midas"
)::SEM3DResult
    @warn "Deep learning depth estimation not yet implemented. Using SfS fallback."
    return reconstruct_depth_sfs(sem_image)
end

end # module SEM3DReconstruction
