#= 
Defines calculus operations d/dx and ∫ for Chebyshev expansions.
=#

export derivative, derivative2, integrate

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