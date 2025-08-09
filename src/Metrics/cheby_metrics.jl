#=
Defines norms and inner product functions for Chebyshev coefficient expansions.
=#

using ..ChebyCoeffs

# ============================================================================
# Norm and Inner Product Functions - unified for real and complex
# ============================================================================

"""L2 norm squared"""
function L2Norm2(u::ChebyCoeff{T}, normalize::Bool=true) where {T<:Number}
    @assert u.state == Spectral "Must be in Spectral state"
    N = length(u.data)
    sum_val = zero(real(T))

    for m = 1:N
        psum = zero(T)
        for n = (m % 2 == 1 ? 1 : 2):2:N
            factor =
                (1 - (m - 1)^2 - (n - 1)^2) / (
                    (1 + (m - 1) - (n - 1)) *
                    (1 - (m - 1) + (n - 1)) *
                    (1 + (m - 1) + (n - 1)) *
                    (1 - (m - 1) - (n - 1))
                )
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

L2Norm(u::ChebyCoeff{T}, normalize::Bool=true) where {T<:Number} =
    sqrt(L2Norm2(u, normalize))

function L2InnerProduct(
    u::ChebyCoeff{T},
    v::ChebyCoeff{S},
    normalize::Bool=true,
) where {T<:Number,S<:Number}
    @assert congruent_structure(u, v) "Arguments must have compatible structure"
    @assert u.state == Spectral "Must be in Spectral state"

    N = length(u.data)
    R = promote_type(T, S)
    sum_val = zero(R)

    for m = 1:N
        psum = zero(R)
        for n = (m % 2 == 1 ? 1 : 2):2:N
            factor =
                (1 - (m - 1)^2 - (n - 1)^2) / (
                    (1 + (m - 1) - (n - 1)) *
                    (1 - (m - 1) + (n - 1)) *
                    (1 + (m - 1) + (n - 1)) *
                    (1 - (m - 1) - (n - 1))
                )
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

    for m = 2:N
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

chebyNorm(u::ChebyCoeff{T}, normalize::Bool=true) where {T<:Number} =
    sqrt(chebyNorm2(u, normalize))

function chebyInnerProduct(
    u::ChebyCoeff{T},
    v::ChebyCoeff{S},
    normalize::Bool=true,
) where {T<:Number,S<:Number}
    @assert congruent_structure(u, v) "Arguments must have compatible structure"
    @assert u.state == Spectral "Must be in Spectral state"

    N = length(u.data)
    R = promote_type(T, S)
    sum_val = zero(R)

    for m = 2:N
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