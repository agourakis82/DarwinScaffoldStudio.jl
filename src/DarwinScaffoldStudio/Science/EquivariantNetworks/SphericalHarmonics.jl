"""
SphericalHarmonics.jl - Spherical Harmonic Computations for Equivariant Networks

Provides spherical harmonic (Y_l^m) computations essential for SE(3)-equivariant
neural networks and geometric deep learning on 3D structures.

SOTA 2024-2025 Features:
- Real spherical harmonics (Y_l^m)
- Wigner-D matrices for SO(3) rotations
- Clebsch-Gordan coefficients for tensor products
- Efficient recursion formulas
- GPU-compatible implementations

Applications:
- SE(3)-Transformers (Fuchs et al. 2020)
- Tensor Field Networks (Thomas et al. 2018)
- NequIP (Batzner et al. 2022)
- MACE (Batatia et al. 2022)

References:
- Weiler et al. 2018: "3D Steerable CNNs"
- Thomas et al. 2018: "Tensor Field Networks"
- Geiger & Smidt 2022: "e3nn: Euclidean Neural Networks"
"""
module SphericalHarmonics

using LinearAlgebra
using SpecialFunctions: gamma

export spherical_harmonic, spherical_harmonics_batch
export associated_legendre, wigner_d_small, wigner_D
export clebsch_gordan, tensor_product_irreps
export cartesian_to_spherical, spherical_to_cartesian
export IrrepType, Fiber, FiberElement

# ============================================================================
# IRREDUCIBLE REPRESENTATIONS
# ============================================================================

"""
    IrrepType

Represents an irreducible representation of SO(3).

# Fields
- `l::Int`: Angular momentum (degree) - dimension is 2l+1
- `parity::Int`: Parity (+1 for even/gerade, -1 for odd/ungerade)
"""
struct IrrepType
    l::Int
    parity::Int

    function IrrepType(l::Int, parity::Int=1)
        @assert l >= 0 "Angular momentum l must be non-negative"
        @assert parity in (-1, 1) "Parity must be +1 or -1"
        new(l, parity)
    end
end

# Convenience constructors
IrrepType(l::Int) = IrrepType(l, (-1)^l)  # Natural parity
dim(irrep::IrrepType) = 2 * irrep.l + 1

Base.show(io::IO, irrep::IrrepType) = print(io, "$(irrep.l)", irrep.parity == 1 ? "e" : "o")

# Standard irreps
const SCALAR = IrrepType(0, 1)      # 0e: scalar (1D)
const PSEUDOSCALAR = IrrepType(0, -1)  # 0o: pseudoscalar
const VECTOR = IrrepType(1, -1)     # 1o: vector (3D, odd parity)
const PSEUDOVECTOR = IrrepType(1, 1)   # 1e: pseudovector/axial vector

"""
    Fiber

A collection of irreducible representations with multiplicities.
Represents the type of geometric quantity at each point.
"""
struct Fiber
    irreps::Vector{Tuple{Int, IrrepType}}  # (multiplicity, irrep)
end

function Fiber(spec::String)
    # Parse specification like "16x0e + 8x1o + 4x2e"
    irreps = Tuple{Int, IrrepType}[]

    for part in split(spec, '+')
        part = strip(part)
        if isempty(part)
            continue
        end

        if occursin('x', part)
            mult_str, irrep_str = split(part, 'x')
            mult = parse(Int, mult_str)
        else
            mult = 1
            irrep_str = part
        end

        l = parse(Int, irrep_str[1:end-1])
        parity = irrep_str[end] == 'e' ? 1 : -1
        push!(irreps, (mult, IrrepType(l, parity)))
    end

    Fiber(irreps)
end

total_dim(fiber::Fiber) = sum(mult * dim(irrep) for (mult, irrep) in fiber.irreps)

"""
    FiberElement

An element of a fiber - actual feature values.
"""
struct FiberElement
    fiber::Fiber
    data::Vector{Float64}

    function FiberElement(fiber::Fiber, data::Vector{Float64})
        @assert length(data) == total_dim(fiber) "Data dimension mismatch"
        new(fiber, data)
    end
end

# ============================================================================
# COORDINATE CONVERSIONS
# ============================================================================

"""
    cartesian_to_spherical(x, y, z) -> (r, θ, φ)

Convert Cartesian to spherical coordinates.
- r: radial distance
- θ: polar angle (0 to π)
- φ: azimuthal angle (0 to 2π)
"""
function cartesian_to_spherical(x::Real, y::Real, z::Real)
    r = sqrt(x^2 + y^2 + z^2)
    if r < 1e-10
        return (0.0, 0.0, 0.0)
    end

    θ = acos(clamp(z / r, -1.0, 1.0))
    φ = atan(y, x)
    if φ < 0
        φ += 2π
    end

    return (r, θ, φ)
end

"""
    spherical_to_cartesian(r, θ, φ) -> (x, y, z)

Convert spherical to Cartesian coordinates.
"""
function spherical_to_cartesian(r::Real, θ::Real, φ::Real)
    x = r * sin(θ) * cos(φ)
    y = r * sin(θ) * sin(φ)
    z = r * cos(θ)
    return (x, y, z)
end

# ============================================================================
# ASSOCIATED LEGENDRE POLYNOMIALS
# ============================================================================

"""
    associated_legendre(l, m, x)

Compute associated Legendre polynomial P_l^m(x).

Uses stable recursion formulas. Returns 0 for |m| > l.
"""
function associated_legendre(l::Int, m::Int, x::Real)
    if abs(m) > l
        return 0.0
    end

    # Handle negative m
    if m < 0
        sign = iseven(abs(m)) ? 1.0 : -1.0
        factor = factorial(l + m) / factorial(l - m)
        return sign * factor * associated_legendre(l, abs(m), x)
    end

    # P_m^m using the formula
    pmm = 1.0
    if m > 0
        somx2 = sqrt(max(0.0, (1.0 - x) * (1.0 + x)))
        fact = 1.0
        for i in 1:m
            pmm *= -fact * somx2
            fact += 2.0
        end
    end

    if l == m
        return pmm
    end

    # P_{m+1}^m
    pmmp1 = x * (2 * m + 1) * pmm

    if l == m + 1
        return pmmp1
    end

    # Use recursion for higher l
    pll = 0.0
    for ll in (m + 2):l
        pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    end

    return pll
end

# ============================================================================
# SPHERICAL HARMONICS
# ============================================================================

"""
    spherical_harmonic(l, m, θ, φ)

Compute real spherical harmonic Y_l^m(θ, φ).

Real spherical harmonics are orthonormal and form a basis for
functions on the sphere.

# Arguments
- `l::Int`: Degree (l ≥ 0)
- `m::Int`: Order (-l ≤ m ≤ l)
- `θ::Real`: Polar angle (0 to π)
- `φ::Real`: Azimuthal angle (0 to 2π)
"""
function spherical_harmonic(l::Int, m::Int, θ::Real, φ::Real)
    @assert l >= 0 "l must be non-negative"
    @assert abs(m) <= l "m must satisfy |m| <= l"

    # Normalization factor
    if m == 0
        norm = sqrt((2l + 1) / (4π))
    else
        norm = sqrt((2l + 1) / (2π) * factorial(l - abs(m)) / factorial(l + abs(m)))
    end

    # Associated Legendre polynomial
    plm = associated_legendre(l, abs(m), cos(θ))

    # Real spherical harmonic
    if m > 0
        return norm * plm * cos(m * φ)
    elseif m < 0
        return norm * plm * sin(abs(m) * φ)
    else
        return norm * plm
    end
end

"""
    spherical_harmonic_xyz(l, m, x, y, z)

Compute real spherical harmonic from Cartesian coordinates.
Handles the singularity at the origin.
"""
function spherical_harmonic_xyz(l::Int, m::Int, x::Real, y::Real, z::Real)
    r, θ, φ = cartesian_to_spherical(x, y, z)
    if r < 1e-10
        return l == 0 ? sqrt(1 / (4π)) : 0.0
    end
    return spherical_harmonic(l, m, θ, φ)
end

"""
    spherical_harmonics_batch(lmax, θ, φ)

Compute all spherical harmonics up to degree lmax.

Returns a vector of length (lmax+1)² containing Y_l^m for
l = 0, ..., lmax and m = -l, ..., l.
"""
function spherical_harmonics_batch(lmax::Int, θ::Real, φ::Real)
    n_harmonics = (lmax + 1)^2
    Y = zeros(n_harmonics)

    idx = 1
    for l in 0:lmax
        for m in -l:l
            Y[idx] = spherical_harmonic(l, m, θ, φ)
            idx += 1
        end
    end

    return Y
end

"""
    spherical_harmonics_xyz_batch(lmax, x, y, z)

Compute all spherical harmonics from Cartesian coordinates.
"""
function spherical_harmonics_xyz_batch(lmax::Int, x::Real, y::Real, z::Real)
    r, θ, φ = cartesian_to_spherical(x, y, z)
    if r < 1e-10
        Y = zeros((lmax + 1)^2)
        Y[1] = sqrt(1 / (4π))  # Only l=0 is non-zero at origin
        return Y
    end
    return spherical_harmonics_batch(lmax, θ, φ)
end

# ============================================================================
# WIGNER D-MATRICES
# ============================================================================

"""
    wigner_d_small(l, m, mp, β)

Compute the small Wigner d-matrix element d^l_{m,m'}(β).

This represents the rotation by angle β about the y-axis.
"""
function wigner_d_small(l::Int, m::Int, mp::Int, β::Real)
    # Wigner formula
    jmin = max(0, m - mp)
    jmax = min(l + m, l - mp)

    sum_term = 0.0
    for j in jmin:jmax
        denom = factorial(j) * factorial(l + m - j) * factorial(l - mp - j) * factorial(mp - m + j)
        sign = iseven(j) ? 1.0 : -1.0
        sum_term += sign / denom * (cos(β/2))^(2l + m - mp - 2j) * (sin(β/2))^(2j + mp - m)
    end

    prefactor = sqrt(factorial(l + m) * factorial(l - m) * factorial(l + mp) * factorial(l - mp))

    return prefactor * sum_term
end

"""
    wigner_D(l, m, mp, α, β, γ)

Compute the Wigner D-matrix element D^l_{m,m'}(α, β, γ).

Represents a general SO(3) rotation with Euler angles (ZYZ convention).
"""
function wigner_D(l::Int, m::Int, mp::Int, α::Real, β::Real, γ::Real)
    d = wigner_d_small(l, m, mp, β)
    return exp(-im * m * α) * d * exp(-im * mp * γ)
end

"""
    wigner_D_matrix(l, α, β, γ)

Compute the full (2l+1) × (2l+1) Wigner D-matrix.
"""
function wigner_D_matrix(l::Int, α::Real, β::Real, γ::Real)
    dim = 2l + 1
    D = zeros(ComplexF64, dim, dim)

    for (i, m) in enumerate(-l:l)
        for (j, mp) in enumerate(-l:l)
            D[i, j] = wigner_D(l, m, mp, α, β, γ)
        end
    end

    return D
end

"""
    real_wigner_D_matrix(l, R)

Compute the real Wigner D-matrix for a rotation matrix R.

This transforms real spherical harmonics under rotation.
"""
function real_wigner_D_matrix(l::Int, R::Matrix{<:Real})
    # Convert rotation matrix to Euler angles
    α, β, γ = rotation_matrix_to_euler(R)

    # Get complex D-matrix
    D_complex = wigner_D_matrix(l, α, β, γ)

    # Convert to real basis
    dim = 2l + 1
    D_real = zeros(dim, dim)

    # The conversion involves the unitary matrix between
    # complex and real spherical harmonics
    for (i, m) in enumerate(-l:l)
        for (j, mp) in enumerate(-l:l)
            # Simplified: compute real parts correctly
            if m == 0 && mp == 0
                D_real[i, j] = real(D_complex[i, j])
            elseif m > 0 && mp > 0
                D_real[i, j] = real(D_complex[l+1+m, l+1+mp] + D_complex[l+1-m, l+1+mp]) / sqrt(2)
            elseif m > 0 && mp < 0
                D_real[i, j] = imag(D_complex[l+1+m, l+1-abs(mp)] - D_complex[l+1-m, l+1-abs(mp)]) / sqrt(2)
            elseif m < 0 && mp > 0
                D_real[i, j] = imag(D_complex[l+1+abs(m), l+1+mp] + D_complex[l+1-abs(m), l+1+mp]) / sqrt(2)
            elseif m < 0 && mp < 0
                D_real[i, j] = real(D_complex[l+1+abs(m), l+1-abs(mp)] - D_complex[l+1-abs(m), l+1-abs(mp)]) / sqrt(2)
            elseif m == 0 && mp > 0
                D_real[i, j] = real(D_complex[l+1, l+1+mp]) * sqrt(2)
            elseif m == 0 && mp < 0
                D_real[i, j] = imag(D_complex[l+1, l+1-abs(mp)]) * sqrt(2)
            elseif m > 0 && mp == 0
                D_real[i, j] = real(D_complex[l+1+m, l+1]) * sqrt(2)
            elseif m < 0 && mp == 0
                D_real[i, j] = imag(D_complex[l+1+abs(m), l+1]) * sqrt(2)
            end
        end
    end

    return D_real
end

"""
    rotation_matrix_to_euler(R)

Extract Euler angles (ZYZ convention) from rotation matrix.
"""
function rotation_matrix_to_euler(R::Matrix{<:Real})
    # ZYZ Euler angles
    β = acos(clamp(R[3,3], -1.0, 1.0))

    if abs(sin(β)) > 1e-10
        α = atan(R[2,3], R[1,3])
        γ = atan(R[3,2], -R[3,1])
    else
        # Gimbal lock
        α = atan(R[2,1], R[1,1])
        γ = 0.0
    end

    return (α, β, γ)
end

# ============================================================================
# CLEBSCH-GORDAN COEFFICIENTS
# ============================================================================

"""
    clebsch_gordan(l1, m1, l2, m2, l, m)

Compute Clebsch-Gordan coefficient ⟨l1 m1; l2 m2 | l m⟩.

Used for coupling angular momenta in tensor products.
"""
function clebsch_gordan(l1::Int, m1::Int, l2::Int, m2::Int, l::Int, m::Int)
    # Selection rules
    if m1 + m2 != m
        return 0.0
    end
    if l < abs(l1 - l2) || l > l1 + l2
        return 0.0
    end
    if abs(m1) > l1 || abs(m2) > l2 || abs(m) > l
        return 0.0
    end

    # Use Racah formula
    # This is a simplified version - full implementation would use
    # more numerically stable methods for large l

    prefactor = sqrt((2l + 1) * factorial(l1 + l2 - l) *
                     factorial(l1 - l2 + l) * factorial(-l1 + l2 + l) /
                     factorial(l1 + l2 + l + 1))

    prefactor *= sqrt(factorial(l + m) * factorial(l - m) *
                      factorial(l1 + m1) * factorial(l1 - m1) *
                      factorial(l2 + m2) * factorial(l2 - m2))

    sum_term = 0.0
    for k in 0:min(l1 + l2 - l, l1 - m1, l2 + m2)
        k1 = l1 + l2 - l - k
        k2 = l1 - m1 - k
        k3 = l2 + m2 - k
        k4 = l - l2 + m1 + k
        k5 = l - l1 - m2 + k

        if k1 >= 0 && k2 >= 0 && k3 >= 0 && k4 >= 0 && k5 >= 0
            sign = iseven(k) ? 1.0 : -1.0
            sum_term += sign / (factorial(k) * factorial(k1) * factorial(k2) *
                                factorial(k3) * factorial(k4) * factorial(k5))
        end
    end

    return prefactor * sum_term
end

"""
    clebsch_gordan_tensor(l1, l2, l)

Compute the full Clebsch-Gordan tensor for coupling l1 ⊗ l2 → l.

Returns a (2l1+1) × (2l2+1) × (2l+1) tensor.
"""
function clebsch_gordan_tensor(l1::Int, l2::Int, l::Int)
    d1 = 2l1 + 1
    d2 = 2l2 + 1
    d3 = 2l + 1

    C = zeros(d1, d2, d3)

    for (i1, m1) in enumerate(-l1:l1)
        for (i2, m2) in enumerate(-l2:l2)
            for (i3, m) in enumerate(-l:l)
                C[i1, i2, i3] = clebsch_gordan(l1, m1, l2, m2, l, m)
            end
        end
    end

    return C
end

"""
    tensor_product_irreps(l1, l2)

Compute the irreps appearing in the tensor product l1 ⊗ l2.

Returns vector of l values from |l1-l2| to l1+l2.
"""
function tensor_product_irreps(l1::Int, l2::Int)
    return collect(abs(l1 - l2):(l1 + l2))
end

# ============================================================================
# EQUIVARIANT TENSOR PRODUCT
# ============================================================================

"""
    equivariant_tensor_product(x1, l1, x2, l2; output_irreps=nothing)

Compute equivariant tensor product of two spherical tensors.

# Arguments
- `x1`: Tensor of type l1 (length 2l1+1)
- `l1`: Angular momentum of x1
- `x2`: Tensor of type l2 (length 2l2+1)
- `l2`: Angular momentum of x2
- `output_irreps`: Which irreps to output (default: all)

# Returns
- Dict mapping l → tensor of type l
"""
function equivariant_tensor_product(x1::Vector{<:Real}, l1::Int,
                                    x2::Vector{<:Real}, l2::Int;
                                    output_irreps::Union{Nothing, Vector{Int}}=nothing)
    if isnothing(output_irreps)
        output_irreps = tensor_product_irreps(l1, l2)
    end

    result = Dict{Int, Vector{Float64}}()

    for l in output_irreps
        if l < abs(l1 - l2) || l > l1 + l2
            continue
        end

        dim_l = 2l + 1
        y = zeros(dim_l)

        for (i1, m1) in enumerate(-l1:l1)
            for (i2, m2) in enumerate(-l2:l2)
                m = m1 + m2
                if abs(m) <= l
                    i = m + l + 1
                    cg = clebsch_gordan(l1, m1, l2, m2, l, m)
                    y[i] += cg * x1[i1] * x2[i2]
                end
            end
        end

        result[l] = y
    end

    return result
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
    l_index_to_lm(idx)

Convert flat index to (l, m) pair.
Indices run as (0,0), (1,-1), (1,0), (1,1), (2,-2), ...
"""
function l_index_to_lm(idx::Int)
    # idx = l² + l + m + 1
    l = floor(Int, sqrt(idx - 1))
    m = idx - 1 - l^2 - l
    return (l, m)
end

"""
    lm_to_index(l, m)

Convert (l, m) to flat index.
"""
function lm_to_index(l::Int, m::Int)
    return l^2 + l + m + 1
end

end # module
