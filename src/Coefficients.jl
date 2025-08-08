module ChebyCoeffs

using FFTW

export ChebyTransform, ChebyCoeff
export makeSpectral!, makePhysical!, makeState!, setToZero!, setState!
export chebyfft!, ichebyfft!
export L2Norm2, L2Norm, L2InnerProduct, LinfNorm, L1Norm, mean_value
export evaluate, eval_a, eval_b, slope_a, slope_b
export bounds, domain_length, num_modes, state
export chebyNorm2, chebyInnerProduct, chebyNorm
export chebypoints
export legendre_polynomial, chebyshev_polynomial
export integrate, derivative, derivative2
export FieldState, Physical, Spectral, BC, Diri, Neumann, Parity, Even, Odd, NormType, Uniform, Cheby

@enum FieldState begin
    Physical
    Spectral
end

# Boundary conditions  
@enum BC Diri = 0 Neumann = 1

# Parity for reflections
@enum Parity Even = 0 Odd = 1

# Norm types
@enum NormType Uniform = 0 Cheby = 1


"""
Chebyshev Transformation using FFTW cosine transforms.
Manages FFTW plans for efficient repeated transformations.
"""
struct ChebyTransform
    N::Int
    cos_plan::FFTW.r2rFFTWPlan
end

"Default constructor for ChebyshevTransform"
function ChebyTransform(N::Int; flags=FFTW.ESTIMATE)
    @assert N > 0 "N must be positive"
    tmp = zeros(Float64, N)
    cos_plan = FFTW.plan_r2r!(tmp, FFTW.REDFT00; flags=flags)
    ChebyTransform(N, cos_plan)
end

Base.length(t::ChebyTransform) = t.N

"""
Chebyshev polynomial expansion on interval [a, b].
Can be in Physical (values at Chebyshev points) or Spectral (coefficients) state.
Supports both real and complex coefficients.
"""
mutable struct ChebyCoeff{T<:Number}
    data::Vector{T}
    a::Real
    b::Real
    state::FieldState

    # Constructors
    function ChebyCoeff{T}() where {T<:Number}
        new{T}(T[], 0.0, 0.0, Spectral)
    end

    function ChebyCoeff{T}(N::Int, a::Real=-1, b::Real=1, state::FieldState=Spectral) where {T<:Number}
        @assert b > a "Upper bound must be greater than lower bound"
        new{T}(zeros(T, N), Float64(a), Float64(b), state)
    end

    function ChebyCoeff{T}(data::Vector{<:Number}, a::Real=-1, b::Real=1, state::FieldState=Spectral) where {T<:Number}
        @assert b > a "Upper bound must be greater than lower bound"
        new{T}(T.(data), Float64(a), Float64(b), state)
    end

    # Copy constructor with different size
    function ChebyCoeff{T}(N::Int, u::ChebyCoeff{T}) where {T<:Number}
        result = new{T}(zeros(T, N), u.a, u.b, u.state)
        N_common = min(N, length(u.data))
        result.data[1:N_common] .= u.data[1:N_common]
        result
    end
end

function ChebyCoeff{T}(u::ChebyCoeff{T}) where {T<:Number}
    ChebyCoeff(u.data, u.a, u.b, u.state)
end

# ============================================================================
# Convenience aliases and type aliases
# ============================================================================

# Type aliases for common cases
const RealChebyCoeff = ChebyCoeff{Float64}
const ComplexChebyCoeff = ChebyCoeff{ComplexF64}


# Convenience constructors
ChebyCoeff(args...; kwargs...) = ChebyCoeff{Float64}(args...; kwargs...)
ChebyCoeff(data::Vector{T}, args...; kwargs...) where {T<:Number} = ChebyCoeff{T}(data, args...; kwargs...)

# Basic properties
Base.length(u::ChebyCoeff) = length(u.data)
Base.size(u::ChebyCoeff) = size(u.data)
Base.getindex(u::ChebyCoeff, i) = u.data[i]
Base.setindex!(u::ChebyCoeff, val, i) = (u.data[i] = val)
Base.eltype(::ChebyCoeff{T}) where {T} = T

# Domain properties
bounds(u::ChebyCoeff) = (u.a, u.b)
domain_length(u::ChebyCoeff) = u.b - u.a
numModes(u::ChebyCoeff) = length(u.data)
state(u::ChebyCoeff) = u.state

function setBounds!(u::ChebyCoeff, a::Real, b::Real)
    @assert b > a "Upper bound must be greater than lower bound"
    u.a = Float64(a)
    u.b = Float64(b)
end

function setState!(u::ChebyCoeff, s::FieldState)
    u.state = s
end

function Base.resize!(u::ChebyCoeff, N::Int)
    resize!(u.data, N)
end

function setToZero!(u::ChebyCoeff)
    fill!(u.data, 0.0)
end


"""Transform from Physical to Spectral state using provided transform."""
function chebyfft!(u::ChebyCoeff{T}, t::ChebyTransform) where {T<:Number}
    @assert u.state == Physical "Must be in Physical state"
    @assert length(t) == length(u.data) "Transform size must match data size"

    if length(u.data) < 2
        u.state = Spectral
        return
    end

    if T <: Real
        # Real case: direct cosine transform
        u.data .= t.cos_plan * u.data
    else
        # Complex case: transform real and imaginary parts separately
        real_part = real.(u.data)
        imag_part = imag.(u.data)

        real_part .= t.cos_plan * real_part
        imag_part .= t.cos_plan * imag_part

        u.data .= complex.(real_part, imag_part)
    end

    # Apply normalization factors
    N = length(u.data)
    nrm = 1.0 / (N - 1)
    u.data[1] *= 0.5 * nrm
    u.data[2:end-1] .*= nrm
    u.data[end] *= 0.5 * nrm

    u.state = Spectral
end

"""Transform from Spectral to Physical state using provided transform."""
function ichebyfft!(u::ChebyCoeff{T}, t::ChebyTransform) where {T<:Number}
    @assert u.state == Spectral "Must be in Spectral state"
    @assert length(t) == length(u.data) "Transform size must match data size"

    if length(u.data) < 2
        u.state = Physical
        return
    end

    # Undo normalization factors
    u.data[1] *= 2.0
    u.data[end] *= 2.0

    if T <: Real
        # Real case: direct inverse cosine transform
        tmp = t.cos_plan * u.data
        u.data .= tmp
        u.data .*= 0.5
    else
        # Complex case: transform real and imaginary parts separately
        real_part = real.(u.data)
        imag_part = imag.(u.data)

        real_part .= t.cos_plan * real_part
        imag_part .= t.cos_plan * imag_part

        u.data .= complex.(real_part .* 0.5, imag_part .* 0.5)
    end

    u.state = Physical
end

# Convenience methods that create temporary transforms
function makeSpectral!(u::ChebyCoeff{T}) where {T<:Number}
    if u.state == Physical
        t = ChebyTransform(length(u.data))
        chebyfft!(u, t)
    end
end

function makePhysical!(u::ChebyCoeff{T}) where {T<:Number}
    if u.state == Spectral
        t = ChebyTransform(length(u.data))
        ichebyfft!(u, t)
    end
end

function makeState!(u::ChebyCoeff{T}, s::FieldState) where {T<:Number}
    if u.state != s
        if u.state == Physical
            makeSpectral!(u)
        else
            makePhysical!(u)
        end
    end
end

# ============================================================================
# Evaluation and boundary values - work for both real and complex
# ============================================================================

"""Evaluate at right boundary (x = b)"""
function eval_b(u::ChebyCoeff{T}) where {T<:Number}
    isempty(u.data) && return zero(T)

    if u.state == Spectral
        return sum(u.data)
    else
        return u.data[1]
    end
end

"""Evaluate at left boundary (x = a)"""
function eval_a(u::ChebyCoeff{T}) where {T<:Number}
    isempty(u.data) && return zero(T)

    if u.state == Spectral
        sum_val = zero(T)
        for (n, coeff) in enumerate(u.data)
            sign = isodd(n - 1) ? -1 : 1
            sum_val += coeff * sign
        end
        return sum_val
    else
        return u.data[end]
    end
end

"""
Evaluate derivative at left boundary.

For mathematical details, see docs/Chebyshev.md#Slopes.
"""
function slope_a(u::ChebyCoeff{T}) where {T<:Number}
    @assert u.state == Spectral "Must be in Spectral state"
    N = length(u.data)
    sum_val = zero(T)

    for n in 1:2:N-1
        sum_val += -(n - 1)^2 * u.data[n] + n^2 * u.data[n+1]
    end

    if isodd(N)
        sum_val -= (N - 1)^2 * u.data[N]
    end

    return 2 * sum_val / (u.b - u.a) # chain rule for Chebyshev mapping from [a, b] -> [1, 1]
end

"""
Evaluate derivative at right boundary

For mathematical details, see docs/Chebyshev.md#Slopes.
"""
function slope_b(u::ChebyCoeff{T}) where {T<:Number}
    @assert u.state == Spectral "Must be in Spectral state"
    isempty(u.data) && return zero(T)

    sum_val = zero(T)
    for (n, coeff) in enumerate(u.data)
        sum_val += (n - 1)^2 * coeff # T'_N(1) = n^2
    end

    return 2 * sum_val / (u.b - u.a) # chain rule for Chebyshev mapping from [a, b] -> [1, 1]
end

"""
Evaluate at arbitrary point x using Clenshaw algorithm

See docs/Chebyshev.md#Evaluation.
"""
function evaluate(u::ChebyCoeff{T}, x::Real) where {T<:Number}
    @assert u.state == Spectral "Must be in Spectral state"
    N = length(u.data)
    if N == 0
        return zero(T)
    end

    # Map x to [-1,1]
    y = (2 * x - u.a - u.b) / (u.b - u.a)

    # Clenshaw recurrence
    b_kplus1 = zero(T)
    b_kplus2 = zero(T)

    for k in N:-1:2
        temp = b_kplus1
        b_kplus1 = 2 * y * b_kplus1 - b_kplus2 + u.data[k]
        b_kplus2 = temp
    end

    return y * b_kplus1 - b_kplus2 + u.data[1]
end

"""Mean value of the function"""
function mean_value(u::ChebyCoeff{T}) where {T<:Number}
    @assert u.state == Spectral "Must be in Spectral state"
    sum_val = u.data[1]
    N = length(u.data)

    for n in 3:2:N
        sum_val -= u.data[n] / ((n - 1)^2 - 1)
    end

    return sum_val
end

# ============================================================================
# Arithmetic operations - generic for real and complex
# ============================================================================

function Base.:+(u::ChebyCoeff{T}, v::ChebyCoeff{S}) where {T<:Number,S<:Number}
    @assert congruent_structure(u, v) "ChebyCoeff objects must have compatible structure"
    R = promote_type(T, S)
    result = ChebyCoeff{R}(u.data + v.data, u.a, u.b, u.state)
    return result
end

function Base.:-(u::ChebyCoeff{T}, v::ChebyCoeff{S}) where {T<:Number,S<:Number}
    @assert congruent_structure(u, v) "ChebyCoeff objects must have compatible structure"
    R = promote_type(T, S)
    result = ChebyCoeff{R}(u.data - v.data, u.a, u.b, u.state)
    return result
end

function Base.:*(c::Number, u::ChebyCoeff{T}) where {T<:Number}
    R = promote_type(typeof(c), T)
    result = ChebyCoeff{R}(c * u.data, u.a, u.b, u.state)
    return result
end

Base.:*(u::ChebyCoeff, c::Number) = c * u

"""Pointwise multiplication (must be in Physical state)"""
function Base.:*(u::ChebyCoeff{T}, v::ChebyCoeff{S}) where {T<:Number,S<:Number}
    @assert congruent_structure(u, v) "ChebyCoeff objects must have compatible structure"
    @assert u.state == Physical "Must be in Physical state for multiplication"
    R = promote_type(T, S)
    result = ChebyCoeff{R}(u.data .* v.data, u.a, u.b, u.state)
    return result
end

function congruent_structure(u::ChebyCoeff, v::ChebyCoeff)
    return (length(u) == length(v) &&
            u.a == v.a &&
            u.b == v.b &&
            u.state == v.state)
end

function congruent(u::ChebyCoeff{T}, v::ChebyCoeff{S}) where {T<:Number,S<:Number}
    return T == S && congruent_structure(u, v)
end

Base.:(==)(u::ChebyCoeff, v::ChebyCoeff) = congruent_structure(u, v) && u.data == v.data

# Complex conjugation
Base.conj(u::ChebyCoeff{T}) where {T<:Real} = u  # Real case: no change
Base.conj(u::ChebyCoeff{T}) where {T<:Complex} = ChebyCoeff{T}(conj.(u.data), u.a, u.b, u.state)

# Real and imaginary parts
Base.real(u::ChebyCoeff{T}) where {T<:Real} = u
Base.real(u::ChebyCoeff{T}) where {T<:Complex} = ChebyCoeff{real(T)}(real.(u.data), u.a, u.b, u.state)

Base.imag(u::ChebyCoeff{T}) where {T<:Real} = ChebyCoeff{T}(zeros(T, length(u.data)), u.a, u.b, u.state)
Base.imag(u::ChebyCoeff{T}) where {T<:Complex} = ChebyCoeff{real(T)}(imag.(u.data), u.a, u.b, u.state)


# ============================================================================
# Differentiation and Integration - work for both real and complex
# ============================================================================

"""
Compute derivative of Chebyshev expansion.

See "docs/Derivatives.md".
"""
function derivative(u::ChebyCoeff{T}) where {T<:Number}
    @assert u.state == Spectral "Must be in Spectral (Chebyshev coefficient) state."

    N = length(u.data)
    result = ChebyCoeff{T}(N, u.a, u.b, Spectral)

    # Nothing to do for constants
    if N <= 1
        return result
    end

    # Scale factor for converting derivative coefficients to the domain [a, b]
    scale = 4.0 / domain_length(u)

    # Initialize top of recurrence
    result.data[N] = zero(T) # input is nth degree, output is n-1; highest mode always zero
    if N >= 2
        result.data[N-1] = scale * (N - 1) * u.data[N]
    end

    # backward recurrence: b_n = b_{n+2} + scale * n * a_{n + 1}
    for n in N-2:-1:1
        result.data[n] = result.data[n+2] + scale * n * u.data[n+1]
    end

    # first coefficient correction
    result.data[1] *= 0.5

    return result
end

"""Compute second derivative"""
function derivative2(u::ChebyCoeff{T}) where {T<:Number}
    return derivative(derivative(u))
end

"""Compute n-th derivative"""
function derivative(u::ChebyCoeff{T}, n::Int) where {T<:Number}
    @assert n >= 0 "Derivative order must be non-negative"
    result = u
    for _ in 1:n
        result = derivative(result)
    end
    return result
end

"""Integrate Chebyshev expansion"""
function integrate(dudy::ChebyCoeff{T}) where {T<:Number}
    @assert dudy.state == Spectral "Must be in Spectral state"
    N = length(dudy.data)

    result = ChebyCoeff{T}(N, dudy.a, dudy.b, Spectral)

    if N == 0
        return result
    elseif N == 1
        result.data[1] = zero(T)
        return result
    elseif N == 2
        h2 = domain_length(dudy) / 2
        result.data[1] = zero(T)
        result.data[2] = h2 * dudy.data[1]
        return result
    end

    h2 = domain_length(dudy) / 2
    result.data[2] = h2 * (dudy.data[1] - dudy.data[3] / 2)

    for n in 3:N-1
        result.data[n] = h2 * (dudy.data[n-1] - dudy.data[n+1]) / (2 * (n - 1))
    end

    result.data[N] = h2 * dudy.data[N-1] / (2 * (N - 1))
    result.data[1] -= mean_value(result)  # Set constant term to make mean zero

    return result
end

# ============================================================================
# Utility functions
# ============================================================================

"""Generate Chebyshev collocation points"""
function chebypoints(N::Int, a::Real=-1, b::Real=1)
    @assert N > 0 "N must be positive"
    points = zeros(Float64, N)
    center = (b + a) / 2
    radius = (b - a) / 2

    for j in 1:N
        points[j] = center + radius * cos((j - 1) * π / (N - 1))
    end

    return points
end

"""Create pure Chebyshev polynomial T_n"""
function chebyshev(::Type{T}, N::Int, n::Int, normalize::Bool=false) where {T<:Number}
    @assert 0 <= n < N "Polynomial degree must be in valid range"
    result = ChebyCoeff{T}(N, -1, 1, Spectral)

    if normalize
        c_n = n == 0 ? 1.0 : 0.5  # Chebyshev normalization constant
        result.data[n+1] = T(sqrt(2 / (π * c_n)))
    else
        result.data[n+1] = one(T)
    end

    return result
end

# Convenience method for real case
chebyshev(N::Int, n::Int, normalize::Bool=false) = chebyshev(Float64, N, n, normalize)

"""Evaluate Chebyshev polynomial T_n(x)"""
function chebyshev_polynomial(n::Int, x::Real)
    return cos(n * acos(x))
end

"""Evaluate Legendre polynomial P_n(x)"""
function legendre_polynomial(n::Int, x::Real)
    if n == 0
        return 1.0
    elseif n == 1
        return x
    end

    p_prev2 = 1.0
    p_prev1 = x

    for m in 2:n
        p_curr = ((2 * m - 1) * x * p_prev1 - (m - 1) * p_prev2) / m
        p_prev2 = p_prev1
        p_prev1 = p_curr
    end

    return p_prev1
end


# ============================================================================
# Complex utility functions
# ============================================================================

"""Generate random complex Chebyshev expansion with specified decay"""
function random_complex_chebycoeff(N::Int, magnitude::Real=1.0, decay::Real=0.8,
    a::Real=-1, b::Real=1, bc_a::BC=Diri, bc_b::BC=Diri)
    u = ChebyCoeff{ComplexF64}(N, a, b, Spectral)

    mag = magnitude
    for n in 1:N
        u.data[n] = mag * (randn() + randn() * im)
        mag *= decay
    end

    # Apply boundary conditions (simplified version)
    if bc_a == Diri && bc_b == Diri && N >= 2
        # Homogeneous Dirichlet BCs: u(a) = u(b) = 0
        eval_a_val = eval_a(u)
        eval_b_val = eval_b(u)

        # Adjust coefficients to satisfy BCs
        u.data[2] -= 0.5 * (eval_b_val - eval_a_val)
        u.data[1] -= 0.5 * (eval_b_val + eval_a_val)
    elseif bc_a == Diri && N >= 1
        u.data[1] -= eval_a(u)
    elseif bc_b == Diri && N >= 1
        u.data[1] -= eval_b(u)
    end

    return u
end

"""Convert between real and complex Chebyshev expansions"""
function complexify(u::ChebyCoeff{Float64})
    return ChebyCoeff{ComplexF64}(complex.(u.data), u.a, u.b, u.state)
end

function realify(u::ChebyCoeff{ComplexF64})
    if all(imag.(u.data) .≈ 0)
        return ChebyCoeff{Float64}(real.(u.data), u.a, u.b, u.state)
    else
        error("Cannot convert complex ChebyCoeff with non-zero imaginary parts to real")
    end
end

"""Check if a complex expansion is effectively real"""
function is_effectively_real(u::ChebyCoeff{ComplexF64}, tol::Real=1e-14)
    return all(abs.(imag.(u.data)) .< tol)
end

# ============================================================================
# Norm and Inner Product Functions - unified for real and complex
# ============================================================================

"""L2 norm squared"""
function L2Norm2(u::ChebyCoeff{T}, normalize::Bool=true) where {T<:Number}
    @assert u.state == Spectral "Must be in Spectral state"
    N = length(u.data)
    sum_val = zero(real(T))

    for m in 1:N
        psum = zero(T)
        for n in (m % 2 == 1 ? 1 : 2):2:N
            factor = (1 - (m - 1)^2 - (n - 1)^2) /
                     ((1 + (m - 1) - (n - 1)) * (1 - (m - 1) + (n - 1)) *
                      (1 + (m - 1) + (n - 1)) * (1 - (m - 1) - (n - 1)))
            psum += u.data[n] * factor
        end
        if T <: Real
            sum_val += u.data[m] * psum
        else
            # For complex: use conjugate for proper inner product
            sum_val += real(conj(u.data[m]) * psum)
        end
    end

    return normalize ? sum_val : sum_val * domain_length(u)
end

L2Norm(u::ChebyCoeff{T}, normalize::Bool=true) where {T<:Number} = sqrt(L2Norm2(u, normalize))

function L2InnerProduct(u::ChebyCoeff{T}, v::ChebyCoeff{S}, normalize::Bool=true) where {T<:Number,S<:Number}
    @assert congruent_structure(u, v) "Arguments must have compatible structure"
    @assert u.state == Spectral "Must be in Spectral state"

    N = length(u.data)
    R = promote_type(T, S)
    sum_val = zero(R)

    for m in 1:N
        psum = zero(R)
        for n in (m % 2 == 1 ? 1 : 2):2:N
            factor = (1 - (m - 1)^2 - (n - 1)^2) /
                     ((1 + (m - 1) - (n - 1)) * (1 - (m - 1) + (n - 1)) *
                      (1 + (m - 1) + (n - 1)) * (1 - (m - 1) - (n - 1)))
            psum += v.data[n] * factor
        end
        # Complex inner product: <u,v> = u* · v
        if T <: Real && S <: Real
            sum_val += u.data[m] * psum
        else
            sum_val += conj(u.data[m]) * psum
        end
    end

    return normalize ? sum_val : sum_val * domain_length(u)
end

"""Chebyshev norm squared (coefficient-based)"""
function chebyNorm2(u::ChebyCoeff{T}, normalize::Bool=true) where {T<:Number}
    @assert u.state == Spectral "Must be in Spectral state"
    N = length(u.data)
    sum_val = zero(real(T))

    for m in 2:N
        if T <: Real
            sum_val += u.data[m]^2
        else
            sum_val += abs2(u.data[m])
        end
    end

    if N > 0 # T[0] term has weight π
        if T <: Real
            sum_val += 2 * u.data[1]^2  # T_0 coefficient has weight 2
        else
            sum_val += 2 * abs2(u.data[1])
        end
    end

    if !normalize
        sum_val *= domain_length(u)
    end

    return sum_val * π / 2
end

chebyNorm(u::ChebyCoeff{T}, normalize::Bool=true) where {T<:Number} = sqrt(chebyNorm2(u, normalize))

function chebyInnerProduct(u::ChebyCoeff{T}, v::ChebyCoeff{S}, normalize::Bool=true) where {T<:Number,S<:Number}
    @assert congruent_structure(u, v) "Arguments must have compatible structure"
    @assert u.state == Spectral "Must be in Spectral state"

    N = length(u.data)
    R = promote_type(T, S)
    sum_val = zero(R)

    for m in 2:N
        if T <: Real && S <: Real
            sum_val += u.data[m] * v.data[m]
        else
            sum_val += conj(u.data[m]) * v.data[m]
        end
    end

    if N > 0
        if T <: Real && S <: Real
            sum_val += 2 * u.data[1] * v.data[1]  # T_0 coefficient has weight 2
        else
            sum_val += 2 * conj(u.data[1]) * v.data[1]
        end
    end

    if !normalize
        sum_val *= domain_length(u)
    end

    return sum_val * π / 2
end

"""L∞ norm (maximum absolute value)"""
function LinfNorm(u::ChebyCoeff{T}) where {T<:Number}
    original_state = u.state
    makePhysical!(u)

    if T <: Real
        result = maximum(abs, u.data)
    else
        result = maximum(abs, u.data)
    end

    makeState!(u, original_state)
    return result
end

"""L1 norm (integral of absolute value)"""
function L1Norm(u::ChebyCoeff{T}, normalize::Bool=true) where {T<:Number}
    # For complex case, this computes ∫|Re(u)| + |Im(u)| dx
    if T <: Real
        u_copy = deepcopy(u)
        makePhysical!(u_copy)
        u_copy.data .= abs.(u_copy.data)
        makeSpectral!(u_copy)

        integrated = integrate(u_copy)
        result = eval_b(integrated) - eval_a(integrated)
    else
        # For complex: |u| = sqrt(Re(u)^2 + Im(u)^2)
        u_real = real(u)
        u_imag = imag(u)

        makePhysical!(u_real)
        makePhysical!(u_imag)

        magnitude = ChebyCoeff{Float64}(length(u), u.a, u.b, Physical)
        magnitude.data .= sqrt.(u_real.data .^ 2 .+ u_imag.data .^ 2)

        makeSpectral!(magnitude)
        integrated = integrate(magnitude)
        result = eval_b(integrated) - eval_a(integrated)
    end

    return normalize ? result : result * domain_length(u)
end

end
