"""
Units.jl - Demetrios-Style Units of Measure for Julia

Provides compile-time dimensional analysis for scientific computing,
ensuring physical quantities are used correctly.

Inspired by Demetrios language units system, but implemented natively in Julia
for seamless integration with Darwin Scaffold Studio.

Features:
- Type-safe units of measure
- Automatic dimensional analysis
- Unit conversion
- SI and scaffold-specific units
- Uncertainty propagation (±)

Units for Tissue Engineering:
- Length: μm (micrometers), mm, cm, m
- Percentage: % (dimensionless ratio)
- Pressure: Pa, kPa, MPa (scaffold modulus)
- Concentration: mol/L, mg/mL
- Time: s, min, h, day
- Temperature: K, °C
- Mass: g, kg, mg

References:
- Demetrios Language: github.com/chiuratto-AI/demetrios
- Unitful.jl patterns
"""
module Units

export Quantity, Unit, Dimension
export μm, mm, cm, m, nm  # Length
export percent, %  # Percentage
export Pa, kPa, MPa, GPa  # Pressure
export mol, mmol, μmol  # Amount
export L, mL, μL  # Volume
export s, min, h, day, week  # Time
export K, °C  # Temperature
export g, kg, mg, μg  # Mass
export mol_per_L, mg_per_mL  # Concentration
export per_day, per_s  # Rate

export ustrip, unit, dimension
export uconvert, in_units
export @u_str

# Uncertainty exports
export ±, UncertainQuantity, uncertainty

# ============================================================================
# DIMENSION SYSTEM
# ============================================================================

"""
    Dimension

Represents physical dimensions using the SI base dimensions.

Stored as a tuple of exponents: (length, mass, time, temperature, amount)
"""
struct Dimension
    length::Int      # L (meters)
    mass::Int        # M (kilograms)
    time::Int        # T (seconds)
    temperature::Int # Θ (kelvin)
    amount::Int      # N (moles)
end

# Common dimensions
const DIM_NONE = Dimension(0, 0, 0, 0, 0)
const DIM_LENGTH = Dimension(1, 0, 0, 0, 0)
const DIM_MASS = Dimension(0, 1, 0, 0, 0)
const DIM_TIME = Dimension(0, 0, 1, 0, 0)
const DIM_TEMPERATURE = Dimension(0, 0, 0, 1, 0)
const DIM_AMOUNT = Dimension(0, 0, 0, 0, 1)
const DIM_VOLUME = Dimension(3, 0, 0, 0, 0)
const DIM_PRESSURE = Dimension(-1, 1, -2, 0, 0)
const DIM_CONCENTRATION = Dimension(-3, 0, 0, 0, 1)
const DIM_RATE = Dimension(0, 0, -1, 0, 0)

# Dimension arithmetic
function Base.:*(d1::Dimension, d2::Dimension)
    Dimension(
        d1.length + d2.length,
        d1.mass + d2.mass,
        d1.time + d2.time,
        d1.temperature + d2.temperature,
        d1.amount + d2.amount
    )
end

function Base.:/(d1::Dimension, d2::Dimension)
    Dimension(
        d1.length - d2.length,
        d1.mass - d2.mass,
        d1.time - d2.time,
        d1.temperature - d2.temperature,
        d1.amount - d2.amount
    )
end

function Base.:^(d::Dimension, n::Integer)
    Dimension(d.length * n, d.mass * n, d.time * n, d.temperature * n, d.amount * n)
end

Base.:(==)(d1::Dimension, d2::Dimension) =
    d1.length == d2.length && d1.mass == d2.mass && d1.time == d2.time &&
    d1.temperature == d2.temperature && d1.amount == d2.amount

Base.hash(d::Dimension, h::UInt) = hash((d.length, d.mass, d.time, d.temperature, d.amount), h)

function Base.show(io::IO, d::Dimension)
    parts = String[]
    d.length != 0 && push!(parts, "L^$(d.length)")
    d.mass != 0 && push!(parts, "M^$(d.mass)")
    d.time != 0 && push!(parts, "T^$(d.time)")
    d.temperature != 0 && push!(parts, "Θ^$(d.temperature)")
    d.amount != 0 && push!(parts, "N^$(d.amount)")

    if isempty(parts)
        print(io, "dimensionless")
    else
        print(io, join(parts, "·"))
    end
end

# ============================================================================
# UNIT SYSTEM
# ============================================================================

"""
    Unit{D}

A unit with dimension D and a scale factor relative to SI base units.
"""
struct Unit{D}
    name::Symbol
    scale::Float64  # Relative to SI base unit
    offset::Float64  # For affine units (e.g., °C)

    function Unit{D}(name::Symbol, scale::Float64, offset::Float64=0.0) where D
        new{D}(name, scale, offset)
    end
end

dimension(::Type{Unit{D}}) where D = D
dimension(u::Unit{D}) where D = D

Base.show(io::IO, u::Unit) = print(io, u.name)

# Unit arithmetic
function Base.:*(u1::Unit{D1}, u2::Unit{D2}) where {D1, D2}
    new_dim = D1 * D2
    Unit{new_dim}(Symbol(u1.name, "·", u2.name), u1.scale * u2.scale)
end

function Base.:/(u1::Unit{D1}, u2::Unit{D2}) where {D1, D2}
    new_dim = D1 / D2
    Unit{new_dim}(Symbol(u1.name, "/", u2.name), u1.scale / u2.scale)
end

function Base.:^(u::Unit{D}, n::Integer) where D
    Unit{D^n}(Symbol(u.name, "^", n), u.scale^n)
end

# ============================================================================
# QUANTITY TYPE
# ============================================================================

"""
    Quantity{T,U}

A physical quantity with value of type T and unit U.

# Examples
```julia
length = 150.0μm
porosity = 92.0percent
modulus = 2.5MPa
```
"""
struct Quantity{T<:Number, U<:Unit}
    value::T
    unit::U
end

# Convenience constructors
Quantity(v::Number, u::Unit) = Quantity{typeof(v), typeof(u)}(v, u)

# Allow unit * number and number * unit
Base.:*(v::Number, u::Unit) = Quantity(v, u)
Base.:*(u::Unit, v::Number) = Quantity(v, u)

# Access value and unit
ustrip(q::Quantity) = q.value
unit(q::Quantity) = q.unit
dimension(q::Quantity) = dimension(q.unit)

# Pretty printing
function Base.show(io::IO, q::Quantity)
    print(io, q.value, " ", q.unit.name)
end

# ============================================================================
# QUANTITY ARITHMETIC
# ============================================================================

# Addition/subtraction - same units only
function Base.:+(q1::Quantity{T1,U}, q2::Quantity{T2,U}) where {T1,T2,U}
    Quantity(q1.value + q2.value, q1.unit)
end

function Base.:-(q1::Quantity{T1,U}, q2::Quantity{T2,U}) where {T1,T2,U}
    Quantity(q1.value - q2.value, q1.unit)
end

# Error for mismatched units
function Base.:+(q1::Quantity, q2::Quantity)
    error("Cannot add quantities with different units: $(q1.unit) and $(q2.unit)")
end

function Base.:-(q1::Quantity, q2::Quantity)
    error("Cannot subtract quantities with different units: $(q1.unit) and $(q2.unit)")
end

# Multiplication
function Base.:*(q1::Quantity, q2::Quantity)
    new_unit = q1.unit * q2.unit
    Quantity(q1.value * q2.value, new_unit)
end

Base.:*(q::Quantity, v::Number) = Quantity(q.value * v, q.unit)
Base.:*(v::Number, q::Quantity) = Quantity(v * q.value, q.unit)

# Division
function Base.:/(q1::Quantity, q2::Quantity)
    new_unit = q1.unit / q2.unit
    Quantity(q1.value / q2.value, new_unit)
end

Base.:/(q::Quantity, v::Number) = Quantity(q.value / v, q.unit)
Base.:/(v::Number, q::Quantity) = Quantity(v / q.value, q.unit^(-1))

# Power
function Base.:^(q::Quantity, n::Integer)
    Quantity(q.value^n, q.unit^n)
end

# Comparison (same units only)
function Base.:<(q1::Quantity{T1,U}, q2::Quantity{T2,U}) where {T1,T2,U}
    q1.value < q2.value
end

function Base.:<=(q1::Quantity{T1,U}, q2::Quantity{T2,U}) where {T1,T2,U}
    q1.value <= q2.value
end

function Base.:(==)(q1::Quantity{T1,U}, q2::Quantity{T2,U}) where {T1,T2,U}
    q1.value == q2.value
end

# Math functions
Base.sqrt(q::Quantity) = error("sqrt of quantity not supported without explicit unit handling")
Base.abs(q::Quantity) = Quantity(abs(q.value), q.unit)
Base.sign(q::Quantity) = sign(q.value)
Base.zero(q::Quantity) = Quantity(zero(q.value), q.unit)
Base.one(::Type{Quantity{T,U}}) where {T,U} = error("one() not defined for Quantity")

# Promotion for mixed arithmetic
Base.promote_rule(::Type{Quantity{T1,U}}, ::Type{T2}) where {T1,T2<:Number,U} = Quantity{promote_type(T1,T2),U}

# ============================================================================
# UNIT CONVERSION
# ============================================================================

"""
    uconvert(target_unit::Unit, q::Quantity) -> Quantity

Convert quantity to different unit of same dimension.
"""
function uconvert(target::Unit{D}, q::Quantity) where D
    if dimension(q.unit) != D
        error("Cannot convert $(q.unit) to $(target): incompatible dimensions")
    end

    # Handle affine units (temperature)
    if q.unit.offset != 0 || target.offset != 0
        # Convert to SI first, then to target
        si_value = (q.value + q.unit.offset) * q.unit.scale
        target_value = si_value / target.scale - target.offset
    else
        # Simple scaling
        target_value = q.value * q.unit.scale / target.scale
    end

    return Quantity(target_value, target)
end

"""
    in_units(q::Quantity, target::Unit) -> Float64

Get the numeric value of a quantity in the specified units.
"""
function in_units(q::Quantity, target::Unit)
    converted = uconvert(target, q)
    return converted.value
end

# ============================================================================
# STANDARD UNITS
# ============================================================================

# Length (SI base: meter)
const m = Unit{DIM_LENGTH}(:m, 1.0)
const cm = Unit{DIM_LENGTH}(:cm, 0.01)
const mm = Unit{DIM_LENGTH}(:mm, 0.001)
const μm = Unit{DIM_LENGTH}(:μm, 1e-6)
const nm = Unit{DIM_LENGTH}(:nm, 1e-9)

# Mass (SI base: kilogram)
const kg = Unit{DIM_MASS}(:kg, 1.0)
const g = Unit{DIM_MASS}(:g, 0.001)
const mg = Unit{DIM_MASS}(:mg, 1e-6)
const μg = Unit{DIM_MASS}(:μg, 1e-9)

# Time (SI base: second)
const s = Unit{DIM_TIME}(:s, 1.0)
const min = Unit{DIM_TIME}(:min, 60.0)
const h = Unit{DIM_TIME}(:h, 3600.0)
const day = Unit{DIM_TIME}(:day, 86400.0)
const week = Unit{DIM_TIME}(:week, 604800.0)

# Temperature (SI base: kelvin)
const K = Unit{DIM_TEMPERATURE}(:K, 1.0)
const °C = Unit{DIM_TEMPERATURE}(:°C, 1.0, 273.15)  # Affine unit

# Amount (SI base: mole)
const mol = Unit{DIM_AMOUNT}(:mol, 1.0)
const mmol = Unit{DIM_AMOUNT}(:mmol, 0.001)
const μmol = Unit{DIM_AMOUNT}(:μmol, 1e-6)

# Volume (derived: L = dm³)
const L = Unit{DIM_VOLUME}(:L, 0.001)  # 1 L = 0.001 m³
const mL = Unit{DIM_VOLUME}(:mL, 1e-6)
const μL = Unit{DIM_VOLUME}(:μL, 1e-9)

# Pressure (derived: Pa = kg/(m·s²))
const Pa = Unit{DIM_PRESSURE}(:Pa, 1.0)
const kPa = Unit{DIM_PRESSURE}(:kPa, 1000.0)
const MPa = Unit{DIM_PRESSURE}(:MPa, 1e6)
const GPa = Unit{DIM_PRESSURE}(:GPa, 1e9)

# Rate (per time)
const per_s = Unit{DIM_RATE}(:per_s, 1.0)
const per_day = Unit{DIM_RATE}(:per_day, 1.0/86400.0)

# Concentration (derived)
const mol_per_L = mol / L
const mg_per_mL = mg / mL

# Percentage (dimensionless)
const percent = Unit{DIM_NONE}(:%, 0.01)
const pct = percent

# ============================================================================
# UNCERTAINTY SUPPORT (±)
# ============================================================================

"""
    UncertainQuantity{T,U}

A quantity with uncertainty (value ± error).
"""
struct UncertainQuantity{T<:Number, U<:Unit}
    value::T
    uncertainty::T
    unit::U
end

# Define ± operator (not in Base)
"""
    ±(value, uncertainty)

Create an uncertain value. Works with Quantities and Numbers.
"""
±(a::Number, b::Number) = (a, b)  # Default tuple fallback

# Create uncertain quantity with ±
function ±(q::Quantity{T,U}, u::Quantity{T,U}) where {T,U}
    UncertainQuantity(q.value, u.value, q.unit)
end

function ±(q::Quantity{T,U}, u::Number) where {T,U}
    UncertainQuantity(q.value, convert(T, u), q.unit)
end

# Pretty print uncertain quantity
function Base.show(io::IO, uq::UncertainQuantity)
    print(io, uq.value, " ± ", uq.uncertainty, " ", uq.unit.name)
end

# Access components
ustrip(uq::UncertainQuantity) = uq.value
unit(uq::UncertainQuantity) = uq.unit
uncertainty(uq::UncertainQuantity) = uq.uncertainty

# Uncertain arithmetic (error propagation)
function Base.:+(uq1::UncertainQuantity{T,U}, uq2::UncertainQuantity{T,U}) where {T,U}
    # σ = √(σ₁² + σ₂²) for addition
    new_unc = sqrt(uq1.uncertainty^2 + uq2.uncertainty^2)
    UncertainQuantity(uq1.value + uq2.value, new_unc, uq1.unit)
end

function Base.:*(uq1::UncertainQuantity, uq2::UncertainQuantity)
    # Relative error propagation: σ_rel = √(σ₁_rel² + σ₂_rel²)
    v1, v2 = uq1.value, uq2.value
    u1, u2 = uq1.uncertainty, uq2.uncertainty
    new_value = v1 * v2
    new_unit = uq1.unit * uq2.unit

    if v1 != 0 && v2 != 0
        rel_unc = sqrt((u1/v1)^2 + (u2/v2)^2)
        new_unc = abs(new_value) * rel_unc
    else
        new_unc = zero(new_value)
    end

    UncertainQuantity(new_value, new_unc, new_unit)
end

# ============================================================================
# STRING MACRO FOR UNITS
# ============================================================================

"""
    @u_str

String macro for unit parsing.

# Example
```julia
length = 150.0u"μm"
pressure = 2.5u"MPa"
```
"""
macro u_str(s)
    unit_map = Dict(
        "m" => :m, "cm" => :cm, "mm" => :mm, "μm" => :μm, "um" => :μm, "nm" => :nm,
        "kg" => :kg, "g" => :g, "mg" => :mg, "μg" => :μg, "ug" => :μg,
        "s" => :s, "min" => :min, "h" => :h, "day" => :day,
        "K" => :K, "°C" => :°C,
        "mol" => :mol, "mmol" => :mmol,
        "L" => :L, "mL" => :mL,
        "Pa" => :Pa, "kPa" => :kPa, "MPa" => :MPa, "GPa" => :GPa,
        "%" => :percent, "percent" => :percent
    )

    if haskey(unit_map, s)
        return esc(unit_map[s])
    else
        error("Unknown unit: $s")
    end
end

# ============================================================================
# SCAFFOLD-SPECIFIC HELPERS
# ============================================================================

"""
    scaffold_length(v::Number) -> Quantity

Create a length quantity in micrometers (scaffold scale).
"""
scaffold_length(v::Number) = v * μm

"""
    scaffold_porosity(v::Number) -> Quantity

Create a porosity quantity as percentage.
"""
scaffold_porosity(v::Number) = v * percent

"""
    scaffold_modulus(v::Number) -> Quantity

Create a mechanical modulus in MPa.
"""
scaffold_modulus(v::Number) = v * MPa

"""
    degradation_rate(v::Number) -> Quantity

Create a degradation rate constant in per_day.
"""
degradation_rate(v::Number) = v * per_day

end # module
