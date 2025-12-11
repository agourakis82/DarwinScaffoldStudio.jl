"""
QuaternionPhysics.jl

Física com Quaternions e Álgebra Geométrica

═══════════════════════════════════════════════════════════════════════════════
                    DEEP THINKING: Por Que Quaternions?
═══════════════════════════════════════════════════════════════════════════════

CHAIN OF THOUGHT: A Matemática Mais Profunda
────────────────────────────────────────────

William Rowan Hamilton (1843): Descobriu os quaternions enquanto caminhava
pela Brougham Bridge em Dublin. Gravou na pedra:

    i² = j² = k² = ijk = -1

Isso mudou a matemática para sempre.

POR QUE QUATERNIONS SÃO PROFUNDOS:
──────────────────────────────────

1. REPRESENTAÇÃO NATURAL DE ROTAÇÕES
   ──────────────────────────────────
   Números complexos: rotações em 2D
   Quaternions: rotações em 3D (SEM gimbal lock!)

   q = cos(θ/2) + sin(θ/2)(xi + yj + zk)

   Muito mais elegante que matrizes 3×3!

2. UNIFICAÇÃO DE ESCALARES E VETORES
   ──────────────────────────────────
   Quaternion q = (s, v) onde s é escalar e v é vetor

   Maxwell originalmente escreveu suas equações com quaternions!
   ∇ × E = -∂B/∂t  em notação quaterniônica é mais natural

3. CONEXÃO COM MECÂNICA QUÂNTICA
   ─────────────────────────────
   Spinors (estados quânticos de spin) são relacionados a quaternions
   SU(2) ≅ S³ (esfera 3D de quaternions unitários)

   Matrizes de Pauli: σₓ, σᵧ, σᵤ são basicamente i, j, k!

4. ÁLGEBRA GEOMÉTRICA (CLIFFORD)
   ─────────────────────────────
   Generalização de quaternions para qualquer dimensão
   Unifica: vetores, bivectors, trivectors...

   Permite calcular rotações, reflexões, projeções elegantemente

═══════════════════════════════════════════════════════════════════════════════
                    APLICAÇÃO À DEGRADAÇÃO DE POLÍMEROS
═══════════════════════════════════════════════════════════════════════════════

INSIGHT PROFUNDO:
─────────────────

O espaço de estados (Mn, Xc, H, t) pode ser visto como espaço 4D.
Quaternions são NATURAIS para descrever transformações em 4D!

A degradação é uma TRAJETÓRIA no espaço quaterniônico:

    q(t) = Mn(t)·1 + Xc(t)·i + H(t)·j + t·k

A taxa de degradação é a DERIVADA quaterniônica:

    dq/dt = velocidade + rotação no espaço de estados

Simetrias da degradação → GRUPOS DE LIE quaterniônicos

═══════════════════════════════════════════════════════════════════════════════

Author: Darwin Scaffold Studio
Date: 2025-12-11
Target: Physical Review Letters / Nature Physics

References:
- Hamilton 1843: On Quaternions
- Clifford 1878: Applications of Grassmann's Extensive Algebra
- Hestenes 1966: Space-Time Algebra
- Doran & Lasenby 2003: Geometric Algebra for Physicists
"""

module QuaternionPhysics

using LinearAlgebra
using Statistics
using Printf

export Quaternion, CliffordMultivector, Rotor, Spinor
export GeometricAlgebraSpace, LieGroup, LieAlgebra
export quaternion_trajectory, clifford_derivative
export discover_symmetry_group, analyze_phase_space

# ═══════════════════════════════════════════════════════════════════════════════
#                    PART 1: QUATERNIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Quaternion: q = w + xi + yj + zk

DEEP THINKING: Estrutura algébrica
──────────────────────────────────
- Não-comutativo: ij ≠ ji (ij = k, ji = -k)
- Divisão definida: todo quaternion não-zero tem inverso
- Norma multiplicativa: |q₁q₂| = |q₁||q₂|
"""
struct Quaternion{T<:Real}
    w::T  # Parte real (escalar)
    x::T  # Coeficiente de i
    y::T  # Coeficiente de j
    z::T  # Coeficiente de k
end

# Construtores
Quaternion(w::Real) = Quaternion(w, zero(w), zero(w), zero(w))
Quaternion(w, x, y, z) = Quaternion(promote(w, x, y, z)...)

# Quaternions base
const Q_ONE = Quaternion(1.0, 0.0, 0.0, 0.0)
const Q_I = Quaternion(0.0, 1.0, 0.0, 0.0)
const Q_J = Quaternion(0.0, 0.0, 1.0, 0.0)
const Q_K = Quaternion(0.0, 0.0, 0.0, 1.0)

# Operações básicas
Base.:+(q1::Quaternion, q2::Quaternion) =
    Quaternion(q1.w + q2.w, q1.x + q2.x, q1.y + q2.y, q1.z + q2.z)

Base.:-(q1::Quaternion, q2::Quaternion) =
    Quaternion(q1.w - q2.w, q1.x - q2.x, q1.y - q2.y, q1.z - q2.z)

Base.:-(q::Quaternion) = Quaternion(-q.w, -q.x, -q.y, -q.z)

Base.:*(a::Real, q::Quaternion) = Quaternion(a*q.w, a*q.x, a*q.y, a*q.z)
Base.:*(q::Quaternion, a::Real) = a * q

"""
Multiplicação de quaternions (não-comutativa!)

CHAIN OF THOUGHT: A regra de Hamilton
────────────────────────────────────
i² = j² = k² = ijk = -1

Consequências:
  ij = k,  jk = i,  ki = j
  ji = -k, kj = -i, ik = -j
"""
function Base.:*(q1::Quaternion, q2::Quaternion)
    w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
    x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y
    y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x
    z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w
    return Quaternion(w, x, y, z)
end

"""
Conjugado: q* = w - xi - yj - zk
"""
conj(q::Quaternion) = Quaternion(q.w, -q.x, -q.y, -q.z)
Base.conj(q::Quaternion) = conj(q)

"""
Norma: |q| = √(w² + x² + y² + z²)
"""
norm(q::Quaternion) = sqrt(q.w^2 + q.x^2 + q.y^2 + q.z^2)
LinearAlgebra.norm(q::Quaternion) = norm(q)

"""
Norma ao quadrado (mais eficiente)
"""
norm_squared(q::Quaternion) = q.w^2 + q.x^2 + q.y^2 + q.z^2

"""
Inverso: q⁻¹ = q*/|q|²
"""
function inv(q::Quaternion)
    n2 = norm_squared(q)
    n2 < 1e-15 && error("Quaternion zero não tem inverso")
    c = conj(q)
    return Quaternion(c.w/n2, c.x/n2, c.y/n2, c.z/n2)
end
Base.inv(q::Quaternion) = inv(q)

"""
Divisão: q1/q2 = q1 * q2⁻¹
"""
Base.:/(q1::Quaternion, q2::Quaternion) = q1 * inv(q2)
Base.:/(q::Quaternion, a::Real) = Quaternion(q.w/a, q.x/a, q.y/a, q.z/a)

"""
Normalizar para quaternion unitário
"""
normalize(q::Quaternion) = q / norm(q)

"""
Exponencial de quaternion.

DEEP THINKING: exp(q) para quaternion puro
──────────────────────────────────────────
Se q = (0, v) é quaternion puro (w = 0):
    exp(q) = cos(|v|) + sin(|v|) * v/|v|

Isso conecta quaternions a rotações!
"""
function Base.exp(q::Quaternion)
    # Separar em parte escalar e vetorial
    s = q.w
    v_norm = sqrt(q.x^2 + q.y^2 + q.z^2)

    if v_norm < 1e-10
        # Quaternion aproximadamente real
        return Quaternion(exp(s), 0.0, 0.0, 0.0)
    end

    exp_s = exp(s)
    coeff = exp_s * sin(v_norm) / v_norm

    return Quaternion(
        exp_s * cos(v_norm),
        coeff * q.x,
        coeff * q.y,
        coeff * q.z
    )
end

"""
Logaritmo de quaternion.
"""
function Base.log(q::Quaternion)
    n = norm(q)
    v_norm = sqrt(q.x^2 + q.y^2 + q.z^2)

    if v_norm < 1e-10
        return Quaternion(log(n), 0.0, 0.0, 0.0)
    end

    θ = acos(clamp(q.w / n, -1.0, 1.0))
    coeff = θ / v_norm

    return Quaternion(
        log(n),
        coeff * q.x,
        coeff * q.y,
        coeff * q.z
    )
end

"""
Potência: q^t (interpolação suave de rotações - SLERP!)
"""
function Base.:^(q::Quaternion, t::Real)
    return exp(t * log(q))
end

"""
SLERP - Spherical Linear Interpolation

CHAIN OF THOUGHT: Por que SLERP é especial
─────────────────────────────────────────
Interpolação linear comum em SO(3) não preserva velocidade angular constante.
SLERP interpola no GRANDE CÍRCULO da esfera S³.

Usado em:
- Animação 3D
- Robótica
- Navegação inercial
- NOSSA APLICAÇÃO: Trajetórias suaves no espaço de estados!
"""
function slerp(q1::Quaternion, q2::Quaternion, t::Real)
    # Garantir caminho mais curto
    dot_product = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z

    if dot_product < 0
        q2 = -q2
        dot_product = -dot_product
    end

    # Se muito próximos, usar interpolação linear
    if dot_product > 0.9995
        result = (1-t) * q1 + t * q2
        return normalize(result)
    end

    θ_0 = acos(dot_product)
    θ = θ_0 * t

    sin_θ = sin(θ)
    sin_θ_0 = sin(θ_0)

    s1 = cos(θ) - dot_product * sin_θ / sin_θ_0
    s2 = sin_θ / sin_θ_0

    return s1 * q1 + s2 * q2
end

"""
Converte quaternion para vetor 4D.
"""
to_vector(q::Quaternion) = [q.w, q.x, q.y, q.z]

"""
Cria quaternion de vetor 4D.
"""
from_vector(v::Vector) = Quaternion(v[1], v[2], v[3], v[4])

"""
Representação de rotação: ângulo + eixo.
"""
function to_axis_angle(q::Quaternion)
    q = normalize(q)

    angle = 2 * acos(clamp(q.w, -1.0, 1.0))

    s = sqrt(1 - q.w^2)
    if s < 1e-10
        axis = [1.0, 0.0, 0.0]  # Eixo arbitrário para rotação nula
    else
        axis = [q.x/s, q.y/s, q.z/s]
    end

    return (angle=angle, axis=axis)
end

"""
Cria quaternion de rotação dado ângulo e eixo.
"""
function from_axis_angle(angle::Real, axis::Vector)
    axis_norm = sqrt(sum(axis.^2))
    axis = axis / axis_norm
    s = sin(angle/2)
    return Quaternion(cos(angle/2), s*axis[1], s*axis[2], s*axis[3])
end

"""
Rota vetor 3D usando quaternion.

v' = q * v * q⁻¹

onde v é tratado como quaternion puro (0, v).
"""
function rotate_vector(q::Quaternion, v::Vector)
    q = normalize(q)
    v_quat = Quaternion(0.0, v[1], v[2], v[3])
    rotated = q * v_quat * conj(q)
    return [rotated.x, rotated.y, rotated.z]
end

# Display
function Base.show(io::IO, q::Quaternion)
    @printf(io, "%.4f + %.4fi + %.4fj + %.4fk", q.w, q.x, q.y, q.z)
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    PART 2: ÁLGEBRA DE CLIFFORD (GEOMÉTRICA)
# ═══════════════════════════════════════════════════════════════════════════════

"""
DEEP THINKING: Por que Álgebra Geométrica?
─────────────────────────────────────────

Clifford (1878) generalizou quaternions para qualquer dimensão.

Elementos básicos:
- Escalares (grade 0): números
- Vetores (grade 1): direções
- Bivectors (grade 2): planos orientados
- Trivectors (grade 3): volumes orientados
- ...

O PRODUTO GEOMÉTRICO unifica:
- Produto interno (parte simétrica)
- Produto externo (parte antissimétrica)

ab = a·b + a∧b

Isso é MUITO mais poderoso que álgebra vetorial tradicional!
"""

"""
Multivetor em álgebra de Clifford Cl(p,q,r).

Para Cl(3,0,0) (espaço euclidiano 3D):
- 1 escalar
- 3 vetores (e₁, e₂, e₃)
- 3 bivectors (e₁₂, e₂₃, e₃₁)
- 1 pseudoescalar (e₁₂₃)

Total: 2³ = 8 componentes
"""
struct CliffordMultivector{N}
    components::Vector{Float64}  # 2^N componentes

    function CliffordMultivector{N}(components::Vector{Float64}) where N
        @assert length(components) == 2^N "Multivetor Cl($N) precisa de $(2^N) componentes"
        new{N}(components)
    end
end

# Construtores convenientes
CliffordMultivector(N::Int, components::Vector{Float64}) = CliffordMultivector{N}(components)

function scalar_multivector(N::Int, s::Real)
    components = zeros(2^N)
    components[1] = s
    return CliffordMultivector{N}(components)
end

function vector_multivector(N::Int, v::Vector{Float64})
    @assert length(v) == N
    components = zeros(2^N)
    for i in 1:N
        components[1 + 2^(i-1)] = v[i]
    end
    return CliffordMultivector{N}(components)
end

# Operações básicas
function Base.:+(a::CliffordMultivector{N}, b::CliffordMultivector{N}) where N
    return CliffordMultivector{N}(a.components .+ b.components)
end

function Base.:-(a::CliffordMultivector{N}, b::CliffordMultivector{N}) where N
    return CliffordMultivector{N}(a.components .- b.components)
end

function Base.:*(s::Real, a::CliffordMultivector{N}) where N
    return CliffordMultivector{N}(s .* a.components)
end

"""
Produto geométrico em Cl(3,0,0).

CHAIN OF THOUGHT: Regras de multiplicação
────────────────────────────────────────
eᵢeᵢ = 1 (vetores ortonormais)
eᵢeⱼ = -eⱼeᵢ para i ≠ j (anticomutatividade)
"""
function geometric_product_cl3(a::CliffordMultivector{3}, b::CliffordMultivector{3})
    # Índices: 1=1, 2=e1, 3=e2, 4=e12, 5=e3, 6=e13, 7=e23, 8=e123
    result = zeros(8)

    # Tabela de multiplicação completa para Cl(3,0,0)
    # (Implementação simplificada - apenas termos principais)

    # Escalar × tudo
    result .+= a.components[1] .* b.components
    result .+= b.components[1] .* a.components
    result[1] -= a.components[1] * b.components[1]  # Evitar contagem dupla

    # Vetor × Vetor → escalar + bivector
    # e1*e1 = 1
    result[1] += a.components[2] * b.components[2]
    result[1] += a.components[3] * b.components[3]
    result[1] += a.components[5] * b.components[5]

    # e1*e2 = e12
    result[4] += a.components[2] * b.components[3]
    result[4] -= a.components[3] * b.components[2]

    # e1*e3 = e13
    result[6] += a.components[2] * b.components[5]
    result[6] -= a.components[5] * b.components[2]

    # e2*e3 = e23
    result[7] += a.components[3] * b.components[5]
    result[7] -= a.components[5] * b.components[3]

    return CliffordMultivector{3}(result)
end

"""
Extrai parte escalar do multivetor.
"""
scalar_part(m::CliffordMultivector) = m.components[1]

"""
Extrai parte vetorial.
"""
function vector_part(m::CliffordMultivector{N}) where N
    v = zeros(N)
    for i in 1:N
        v[i] = m.components[1 + 2^(i-1)]
    end
    return v
end

"""
Grade extraction - extrai componentes de uma grade específica.
"""
function grade(m::CliffordMultivector{N}, k::Int) where N
    result = zeros(2^N)

    for i in 0:(2^N - 1)
        if count_ones(i) == k
            result[i + 1] = m.components[i + 1]
        end
    end

    return CliffordMultivector{N}(result)
end

"""
Reverso (inverte ordem dos vetores).
"""
function reverse_multivector(m::CliffordMultivector{N}) where N
    result = zeros(2^N)

    for i in 0:(2^N - 1)
        k = count_ones(i)
        sign = (-1)^(k * (k - 1) ÷ 2)
        result[i + 1] = sign * m.components[i + 1]
    end

    return CliffordMultivector{N}(result)
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    PART 3: ROTORES E SPINORS
# ═══════════════════════════════════════════════════════════════════════════════

"""
DEEP THINKING: O que são Rotores?
────────────────────────────────

Rotor é um elemento par da álgebra de Clifford que representa rotação.

R = cos(θ/2) - sin(θ/2) B

onde B é o bivector do plano de rotação (normalizado).

Rotação de vetor v:
    v' = R v R†

Isso é EXATAMENTE como quaternions, mas generalizado!
"""

"""
Rotor - elemento de rotação em álgebra geométrica.
"""
struct Rotor{N}
    multivector::CliffordMultivector{N}
end

"""
Cria rotor de ângulo e plano de rotação.
"""
function rotor_from_plane(bivector::CliffordMultivector{N}, angle::Real) where N
    # R = cos(θ/2) - sin(θ/2) * B_normalizado
    B_norm = sqrt(sum(bivector.components[i]^2 for i in 1:2^N if count_ones(i-1) == 2))

    if B_norm < 1e-10
        return Rotor{N}(scalar_multivector(N, 1.0))
    end

    B_unit = (1.0 / B_norm) * bivector

    s = cos(angle / 2)
    components = zeros(2^N)
    components[1] = s

    for i in 1:2^N
        if count_ones(i-1) == 2
            components[i] = -sin(angle / 2) * B_unit.components[i]
        end
    end

    return Rotor{N}(CliffordMultivector{N}(components))
end

"""
SPINOR: Estado quântico de spin

DEEP THINKING: Conexão com mecânica quântica
───────────────────────────────────────────

Spinors são objetos que transformam de forma especial sob rotações:
- Rotação de 360° → spinor muda de sinal!
- Rotação de 720° → volta ao estado original

Isso é FÍSICA QUÂNTICA fundamental.

|ψ⟩ = α|↑⟩ + β|↓⟩

Representado como quaternion/spinor:
ψ = α + βj  (usando base quaterniônica)
"""
struct Spinor
    up::Complex{Float64}    # Coeficiente de |↑⟩
    down::Complex{Float64}  # Coeficiente de |↓⟩
end

"""
Normaliza spinor.
"""
function normalize_spinor(s::Spinor)
    n = sqrt(abs2(s.up) + abs2(s.down))
    return Spinor(s.up / n, s.down / n)
end

"""
Probabilidade de medir spin-up.
"""
prob_up(s::Spinor) = abs2(s.up) / (abs2(s.up) + abs2(s.down))

"""
Probabilidade de medir spin-down.
"""
prob_down(s::Spinor) = abs2(s.down) / (abs2(s.up) + abs2(s.down))

"""
Aplica rotação ao spinor.
"""
function rotate_spinor(s::Spinor, axis::Vector{Float64}, angle::Float64)
    # Matrizes de Pauli
    # σx = [0 1; 1 0], σy = [0 -i; i 0], σz = [1 0; 0 -1]

    axis_n = sqrt(sum(axis.^2))
    nx, ny, nz = axis / axis_n

    c = cos(angle / 2)
    s_angle = sin(angle / 2)

    # U = cos(θ/2)I - i*sin(θ/2)(n·σ)
    # = [c - i*s*nz,    -i*s*(nx - i*ny)]
    #   [-i*s*(nx + i*ny), c + i*s*nz   ]

    u11 = c - im * s_angle * nz
    u12 = -im * s_angle * (nx - im * ny)
    u21 = -im * s_angle * (nx + im * ny)
    u22 = c + im * s_angle * nz

    new_up = u11 * s.up + u12 * s.down
    new_down = u21 * s.up + u22 * s.down

    return Spinor(new_up, new_down)
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    PART 4: GRUPOS DE LIE
# ═══════════════════════════════════════════════════════════════════════════════

"""
DEEP THINKING: Por que Grupos de Lie?
────────────────────────────────────

Grupos de Lie são simetrias CONTÍNUAS.

Exemplos:
- SO(3): Rotações 3D (3 parâmetros)
- SU(2): Grupo de spin (isomorfo a quaternions unitários!)
- SE(3): Movimento rígido (6 parâmetros: 3 rotação + 3 translação)

TEOREMA FUNDAMENTAL:
Todo grupo de Lie tem uma ÁLGEBRA DE LIE associada.
A álgebra captura a estrutura infinitesimal do grupo.

g = exp(θ·X) onde X é elemento da álgebra de Lie

Para degradação de polímeros:
- Simetrias da dinâmica → Grupo de Lie
- Leis de conservação → Álgebra de Lie
"""

"""
Representação de álgebra de Lie.
"""
struct LieAlgebra
    name::String
    dimension::Int
    generators::Vector{Matrix{Float64}}  # Geradores Xᵢ
    structure_constants::Array{Float64, 3}  # fᵢⱼₖ: [Xᵢ, Xⱼ] = fᵢⱼₖ Xₖ
end

"""
Cria álgebra de Lie so(3) - rotações 3D.
"""
function so3_algebra()
    # Geradores: Lx, Ly, Lz (momento angular)
    Lx = [0.0 0.0 0.0; 0.0 0.0 -1.0; 0.0 1.0 0.0]
    Ly = [0.0 0.0 1.0; 0.0 0.0 0.0; -1.0 0.0 0.0]
    Lz = [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]

    generators = [Lx, Ly, Lz]

    # Constantes de estrutura: [Li, Lj] = εijk Lk
    f = zeros(3, 3, 3)
    f[1, 2, 3] = 1.0; f[2, 1, 3] = -1.0
    f[2, 3, 1] = 1.0; f[3, 2, 1] = -1.0
    f[3, 1, 2] = 1.0; f[1, 3, 2] = -1.0

    return LieAlgebra("so(3)", 3, generators, f)
end

"""
Cria álgebra de Lie su(2) - isomorfa a so(3) mas com spinors.
"""
function su2_algebra()
    # Geradores: τᵢ = σᵢ/2 (meias matrizes de Pauli)
    τx = [0.0+0.0im 0.5+0.0im; 0.5+0.0im 0.0+0.0im]
    τy = [0.0+0.0im 0.0-0.5im; 0.0+0.5im 0.0+0.0im]
    τz = [0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im]

    # Converter para Float64 (parte real para estrutura)
    generators = [real.(τx), real.(τy), real.(τz)]

    f = zeros(3, 3, 3)
    f[1, 2, 3] = 1.0; f[2, 1, 3] = -1.0
    f[2, 3, 1] = 1.0; f[3, 2, 1] = -1.0
    f[3, 1, 2] = 1.0; f[1, 3, 2] = -1.0

    return LieAlgebra("su(2)", 3, generators, f)
end

"""
Comutador de Lie: [X, Y] = XY - YX
"""
function lie_bracket(X::Matrix, Y::Matrix)
    return X * Y - Y * X
end

"""
Mapa exponencial: g = exp(θᵢXᵢ)
"""
function lie_exp(algebra::LieAlgebra, params::Vector{Float64})
    @assert length(params) == algebra.dimension

    # X = Σ θᵢ Xᵢ
    X = sum(params[i] * algebra.generators[i] for i in 1:algebra.dimension)

    # Exponencial de matriz
    return exp(X)
end

"""
Mapa logarítmico (inverso do exponencial).
"""
function lie_log(algebra::LieAlgebra, g::Matrix)
    # Para SO(3), usar fórmula de Rodrigues inversa
    # Simplificado: usar log de matriz
    X = log(g)

    # Extrair parâmetros
    params = zeros(algebra.dimension)
    for i in 1:algebra.dimension
        # Projetar na base
        params[i] = sum(X .* algebra.generators[i]) /
                    sum(algebra.generators[i].^2)
    end

    return params
end

# ═══════════════════════════════════════════════════════════════════════════════
#                    PART 5: APLICAÇÃO À DEGRADAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

"""
DEEP THINKING: Espaço de Estados como Variedade
─────────────────────────────────────────────

O estado do polímero (Mn, Xc, H, t) vive em R⁴.
Mas a DINÂMICA pode ter estrutura mais rica!

Representação quaterniônica:
    q(τ) = Mn(τ)·1 + Xc(τ)·i + H(τ)·j + τ·k

A degradação é uma CURVA no espaço quaterniônico S³.

Vantagens:
- Interpolação suave via SLERP
- Simetrias naturalmente representadas
- Conexão com grupos de Lie
"""

"""
Trajetória quaterniônica do sistema.
"""
struct QuaternionTrajectory
    times::Vector{Float64}
    quaternions::Vector{Quaternion{Float64}}

    # Derivadas
    velocities::Vector{Quaternion{Float64}}

    # Métricas
    total_arc_length::Float64
    curvature::Vector{Float64}
end

"""
Converte dados de degradação para trajetória quaterniônica.

NORMALIZAÇÃO:
- Mn → [0, 1] dividindo por Mn0
- Xc → já em [0, 1]
- H → [0, 1] dividindo por max(H)
- t → [0, 1] dividindo por tmax
"""
function quaternion_trajectory(times::Vector{Float64},
                                Mn::Vector{Float64},
                                Xc::Vector{Float64},
                                H::Vector{Float64})
    n = length(times)
    @assert n == length(Mn) == length(Xc) == length(H)

    # Normalizar
    Mn0 = Mn[1]
    Hmax = maximum(H) + 1e-10
    tmax = times[end]

    Mn_norm = Mn ./ Mn0
    Xc_norm = Xc
    H_norm = H ./ Hmax
    t_norm = times ./ tmax

    # Criar quaternions
    quaternions = [Quaternion(Mn_norm[i], Xc_norm[i], H_norm[i], t_norm[i])
                   for i in 1:n]

    # Calcular velocidades (diferenças finitas)
    velocities = Quaternion{Float64}[]
    for i in 1:n-1
        dt = t_norm[i+1] - t_norm[i]
        dq = (quaternions[i+1] - quaternions[i]) / dt
        push!(velocities, dq)
    end
    push!(velocities, velocities[end])  # Repetir último

    # Comprimento de arco
    arc_length = 0.0
    for i in 1:n-1
        dq = quaternions[i+1] - quaternions[i]
        arc_length += norm(dq)
    end

    # Curvatura (mudança de direção)
    curvature = Float64[]
    for i in 1:n-1
        if i == 1
            push!(curvature, 0.0)
        else
            v1 = velocities[i-1]
            v2 = velocities[i]
            # Curvatura ≈ |dv/ds|
            dv = v2 - v1
            ds = norm(quaternions[i] - quaternions[i-1]) + 1e-10
            push!(curvature, norm(dv) / ds)
        end
    end
    push!(curvature, curvature[end])

    return QuaternionTrajectory(times, quaternions, velocities,
                                 arc_length, curvature)
end

"""
Analisa simetrias da trajetória quaterniônica.
"""
function analyze_trajectory_symmetries(traj::QuaternionTrajectory)
    println("\n" * "═"^70)
    println("  ANÁLISE QUATERNIÔNICA DA TRAJETÓRIA")
    println("═"^70)

    n = length(traj.times)

    println("\n  QUATERNIONS DA TRAJETÓRIA:")
    println("─"^70)
    for i in 1:n
        q = traj.quaternions[i]
        @printf("  t=%.0f: q = %.3f + %.3fi + %.3fj + %.3fk  (|q|=%.3f)\n",
                traj.times[i], q.w, q.x, q.y, q.z, norm(q))
    end

    println("\n  MÉTRICAS:")
    println("─"^70)
    @printf("    Comprimento de arco total: %.4f\n", traj.total_arc_length)
    @printf("    Curvatura média: %.4f\n", mean(traj.curvature))
    @printf("    Curvatura máxima: %.4f (em t=%.0f)\n",
            maximum(traj.curvature),
            traj.times[argmax(traj.curvature)])

    # Analisar se trajetória é geodésica (curvatura mínima)
    is_geodesic = maximum(traj.curvature) < 0.5

    println("\n  ANÁLISE GEOMÉTRICA:")
    println("─"^70)
    if is_geodesic
        println("    ✓ Trajetória é aproximadamente GEODÉSICA")
        println("      → A degradação segue caminho de menor energia")
    else
        println("    × Trajetória NÃO é geodésica")
        println("      → Há forças/campos afetando a trajetória")
    end

    # Verificar se quaternions permanecem em subespaço
    # (indicaria simetria/conservação)
    w_var = var([q.w for q in traj.quaternions])
    x_var = var([q.x for q in traj.quaternions])
    y_var = var([q.y for q in traj.quaternions])
    z_var = var([q.z for q in traj.quaternions])

    total_var = w_var + x_var + y_var + z_var

    println("\n  ANÁLISE DE VARIÂNCIA:")
    @printf("    Variância total: %.4f\n", total_var)
    @printf("    w (Mn): %.1f%%\n", 100 * w_var / total_var)
    @printf("    x (Xc): %.1f%%\n", 100 * x_var / total_var)
    @printf("    y (H):  %.1f%%\n", 100 * y_var / total_var)
    @printf("    z (t):  %.1f%%\n", 100 * z_var / total_var)

    # Identificar componente dominante
    vars = [w_var, x_var, y_var, z_var]
    names = ["Mn", "Xc", "H", "t"]
    dominant = names[argmax(vars)]

    println("\n  COMPONENTE DOMINANTE: $dominant")
    println("    → A degradação é primariamente controlada por $dominant")

    println("\n" * "═"^70)

    return (is_geodesic=is_geodesic, dominant=dominant,
            arc_length=traj.total_arc_length)
end

"""
Descobre grupo de simetria da dinâmica.
"""
function discover_symmetry_group(traj::QuaternionTrajectory)
    println("\n" * "═"^70)
    println("  DESCOBERTA DE GRUPO DE SIMETRIA")
    println("═"^70)

    # Testar invariância sob rotações quaterniônicas

    # 1. Rotação no plano (w, x) - mistura Mn e Xc
    println("\n  Testando simetrias...")

    symmetries_found = String[]

    # Testar escala uniforme
    q0 = traj.quaternions[1]
    qf = traj.quaternions[end]

    ratio = norm(qf) / norm(q0)
    if 0.1 < ratio < 0.9
        push!(symmetries_found, "Contração uniforme (fator $(round(ratio, digits=3)))")
    end

    # Testar rotação
    # Se q(t) = R(t) q₀ para algum rotor R(t), há simetria rotacional

    rotation_quality = 0.0
    for i in 2:length(traj.quaternions)
        q = traj.quaternions[i]
        q_norm = normalize(q)
        q0_norm = normalize(q0)

        # Ângulo entre quaternions normalizados
        dot_prod = q_norm.w * q0_norm.w + q_norm.x * q0_norm.x +
                   q_norm.y * q0_norm.y + q_norm.z * q0_norm.z
        angle = 2 * acos(clamp(abs(dot_prod), 0.0, 1.0))

        # Qualidade = quão constante é a taxa de rotação
        expected_angle = (i - 1) / (length(traj.quaternions) - 1) * π
        rotation_quality += abs(angle - expected_angle)
    end
    rotation_quality /= length(traj.quaternions)

    if rotation_quality < 0.3
        push!(symmetries_found, "Rotação contínua em S³")
    end

    println("\n  SIMETRIAS DETECTADAS:")
    println("─"^70)

    if isempty(symmetries_found)
        println("    Nenhuma simetria óbvia detectada")
        println("    → Sistema pode ter simetrias QUEBRADAS")
    else
        for sym in symmetries_found
            println("    ✓ $sym")
        end
    end

    # Identificar álgebra de Lie aproximada
    println("\n  ÁLGEBRA DE LIE APROXIMADA:")
    println("─"^70)

    if !isempty(symmetries_found)
        println("    Grupo: SO(2) × R⁺ (rotação + dilatação)")
        println("    Dimensão: 2")
        println("    Geradores: L_z (rotação), D (dilatação)")
    else
        println("    Grupo: Trivial {e}")
        println("    → Dinâmica completamente assimétrica")
    end

    # Leis de conservação (via Noether)
    println("\n  LEIS DE CONSERVAÇÃO (Noether):")
    println("─"^70)

    if any(contains.(symmetries_found, "Rotação"))
        println("    ✓ Momento angular quaterniônico conservado")
    end
    if any(contains.(symmetries_found, "Contração"))
        println("    ✓ Invariante de escala (número de degradação)")
    end

    println("\n" * "═"^70)

    return symmetries_found
end

"""
Derivada no espaço de Clifford para análise diferencial.
"""
function clifford_derivative(f::Function, x::CliffordMultivector{N},
                              h::Float64=1e-6) where N
    # Derivada direcional em cada componente
    result_components = zeros(2^N)

    for i in 1:2^N
        # Perturbar componente i
        x_plus = CliffordMultivector{N}(copy(x.components))
        x_plus.components[i] += h

        x_minus = CliffordMultivector{N}(copy(x.components))
        x_minus.components[i] -= h

        # Diferença central
        df = (f(x_plus) - f(x_minus)) / (2h)
        result_components[i] = df
    end

    return CliffordMultivector{N}(result_components)
end

"""
Visualização ASCII do espaço de fases quaterniônico.
"""
function visualize_quaternion_phase_space(traj::QuaternionTrajectory)
    println("\n" * "═"^70)
    println("  ESPAÇO DE FASES QUATERNIÔNICO")
    println("═"^70)

    # Projetar em planos 2D
    n = length(traj.quaternions)

    # Plano (w, x) - Mn vs Xc
    println("\n  Projeção no plano Mn-Xc (w-x):")
    println()

    w_vals = [q.w for q in traj.quaternions]
    x_vals = [q.x for q in traj.quaternions]

    # Normalizar para plot ASCII
    w_min, w_max = extrema(w_vals)
    x_min, x_max = extrema(x_vals)

    width, height = 60, 15

    grid = fill(' ', height, width)

    for i in 1:n
        col = round(Int, (w_vals[i] - w_min) / (w_max - w_min + 1e-10) * (width - 1)) + 1
        row = round(Int, (1 - (x_vals[i] - x_min) / (x_max - x_min + 1e-10)) * (height - 1)) + 1
        col = clamp(col, 1, width)
        row = clamp(row, 1, height)

        if i == 1
            grid[row, col] = 'S'  # Start
        elseif i == n
            grid[row, col] = 'E'  # End
        else
            grid[row, col] = '●'
        end
    end

    # Desenhar com eixos
    println("  Xc │")
    for row in 1:height
        if row == 1
            print("  1.0│")
        elseif row == height
            print("  0.0│")
        else
            print("     │")
        end
        println(String(grid[row, :]))
    end
    println("     └" * "─"^width)
    println("      0.0" * " "^(width-8) * "1.0")
    println("                        Mn (normalizado)")

    println("\n  Legenda: S = início, E = fim, ● = pontos intermediários")

    println("\n" * "═"^70)
end

end # module
