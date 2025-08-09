#=
Defines the types and constructors for Chebyshev coefficients and transforms.
Should be the first include.
=#

using FFTW

export ChebyTransform, ChebyCoeff, FieldState, Physical,
    Spectral, BC, Diri, Neumann, Parity, Even, Odd,
    NormType, Uniform, Cheby

@enum FieldState Physical = 0 Spectral = 1

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
    cos_plan = FFTW.plan_r2r(tmp, FFTW.REDFT00; flags=flags)
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