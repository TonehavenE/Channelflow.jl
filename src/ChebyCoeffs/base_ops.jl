#= 
Basic arithmetic and manipulation operations for ChebyCoeff type.
Must be included after types_and_constructors.jl.
=#

export bounds, domain_length, num_modes, state,
    setBounds!, setState!, setToZero!

function congruent_structure(u::ChebyCoeff, v::ChebyCoeff)
    return (length(u) == length(v) &&
            u.a == v.a &&
            u.b == v.b &&
            u.state == v.state)
end

function congruent(u::ChebyCoeff{T}, v::ChebyCoeff{S}) where {T<:Number,S<:Number}
    return T == S && congruent_structure(u, v)
end
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


Base.:(==)(u::ChebyCoeff, v::ChebyCoeff) = congruent_structure(u, v) && u.data == v.data

# Complex conjugation
Base.conj(u::ChebyCoeff{T}) where {T<:Real} = u  # Real case: no change
Base.conj(u::ChebyCoeff{T}) where {T<:Complex} = ChebyCoeff{T}(conj.(u.data), u.a, u.b, u.state)

# Real and imaginary parts
Base.real(u::ChebyCoeff{T}) where {T<:Real} = u
Base.real(u::ChebyCoeff{T}) where {T<:Complex} = ChebyCoeff{real(T)}(real.(u.data), u.a, u.b, u.state)

Base.imag(u::ChebyCoeff{T}) where {T<:Real} = ChebyCoeff{T}(zeros(T, length(u.data)), u.a, u.b, u.state)
Base.imag(u::ChebyCoeff{T}) where {T<:Complex} = ChebyCoeff{real(T)}(imag.(u.data), u.a, u.b, u.state)
