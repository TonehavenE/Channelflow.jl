#= 
Defines calculus operations d/dx and ∫ for Chebyshev expansions.
=#

export derivative, derivative!, derivative2, derivative2!, swap!, integrate, integrate!

# ============================================================================
# Differentiation and Integration - work for both real and complex
# ============================================================================

"""
Compute derivative of Chebyshev expansion in place.

See "docs/Derivatives.md".
"""
function derivative!(u::ChebyCoeff{<:Number}, dudx_result::ChebyCoeff{<:Number})
    @assert u.state == Spectral "Must be in Spectral (Chebyshev coefficient) state."
    @assert dudx_result.state == Spectral "Must be in Spectral (Chebyshev coefficient) state."
    N = length(u.data)
    @assert length(dudx_result.data) == N "Input and output must have same length"

    # Nothing to do for constants
    if N <= 1
        fill!(dudx_result.data, zero(eltype(u.data)))
        return dudx_result
    end

    # Scale factor for converting derivative coefficients to the domain [a, b]
    scale = 4.0 / domain_length(u)

    # Initialize top of recurrence
    dudx_result.data[N] = zero(eltype(u.data)) # input is nth degree, output is n-1; highest mode always zero
    if N >= 2
        dudx_result.data[N-1] = scale * (N - 1) * u.data[N]
    end

    # backward recurrence: b_n = b_{n+2} + scale * n * a_{n + 1}
    for n = N-2:-1:1
        dudx_result.data[n] = dudx_result.data[n+2] + scale * n * u.data[n+1]
    end

    # first coefficient correction
    dudx_result.data[1] *= 0.5

    return dudx_result
end

function swap!(u::ChebyCoeff{<:Number}, v::ChebyCoeff{<:Number})
    @assert length(u.data) == length(v.data) "Input and output must have same length"
    @assert u.state == v.state "Both must be in the same state"
    @assert u.a == v.a && u.b == v.b "Both must have the same domain"
    tmp = copy(u.data)
    u.data .= v.data
    v.data .= tmp
    return nothing
end

"""
Compute derivative of Chebyshev expansion.

See "docs/Derivatives.md".
"""
function derivative(u::ChebyCoeff{T}) where {T<:Number}
    N = length(u.data)
    result = ChebyCoeff{T}(N, u.a, u.b, Spectral)
    derivative!(u, result)
    return result
end

"""Compute second derivative in place"""
function derivative2!(u::ChebyCoeff{<:Number}, dudx2_result::ChebyCoeff{<:Number})
    dudx = ChebyCoeff{eltype(u.data)}(length(u.data), u.a, u.b, Spectral)
    derivative!(u, dudx) 
    derivative!(dudx, dudx2_result)
    return dudx2_result
end

"""Compute second derivative"""
function derivative2(u::ChebyCoeff{T}) where {T<:Number}
    return derivative(derivative(u))
end


function derivative!(u::ChebyCoeff{<:Number}, du::ChebyCoeff{<:Number}, n::Int)
    @assert n >= 0 "Derivative order must be non-negative"
    du = u
    temp = ChebyCoeff{T}(length(u.data), u.a, u.b, Spectral)
    for _ = 1:n
        derivative!(du, temp)
        swap!(du, temp)
    end
    return nothing
end

"""Compute n-th derivative"""
function derivative(u::ChebyCoeff{T}, n::Int) where {T<:Number}
    du = ChebyCoeff{eltype(u.data)}(length(u.data), u.a, u.b, Spectral)
    derivative!(u, du, n)
    return du
end

"""Integrate Chebyshev expansion, with the result being modified in place."""
function integrate!(dudy::ChebyCoeff{<:Number}, result::ChebyCoeff{<:Number})
    @assert dudy.state == Spectral "Must be in Spectral state"
    N = length(dudy.data)

    if N == 0
        fill!(result.data, zero(eltype(u.data))),
        return nothing
    elseif N == 1
        result.data[1] = zero(T)
        return nothing
    elseif N == 2
        h2 = domain_length(dudy) / 2
        result.data[1] = zero(T)
        result.data[2] = h2 * dudy.data[1]
        return nothing
    end

    h2 = domain_length(dudy) / 2
    result.data[2] = h2 * (dudy.data[1] - dudy.data[3] / 2)

    for n = 3:N-1
        result.data[n] = h2 * (dudy.data[n-1] - dudy.data[n+1]) / (2 * (n - 1))
    end

    result.data[N] = h2 * dudy.data[N-1] / (2 * (N - 1))
    result.data[1] -= mean_value(result)  # Set constant term to make mean zero

    return nothing
end

"""Integrate Chebyshev expansion"""
function integrate(dudy::ChebyCoeff{T}) where {T<:Number}
    N = length(dudy.data)
    result = ChebyCoeff{T}(N, dudy.a, dudy.b, Spectral)
    integrate!(dudy, result)
    return result
end