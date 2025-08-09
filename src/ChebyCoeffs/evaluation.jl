#=
Defines ways to evaluate a Chebyshev coefficient expansion at a particular value.
=#

export evaluate, eval_a, eval_b, slope_a, slope_b, mean_value

# ==============================
# Evaluation and boundary values 
# ==============================

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