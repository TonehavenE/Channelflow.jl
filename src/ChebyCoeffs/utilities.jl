#=
Defines utility functions for Chebyshev expansions.
=#

export chebypoints,
    chebyshev,
    legendre_polynomial,
    chebyshev_polynomial,
    random_complex_chebycoeff,
    complexify,
    realify,
    is_effectively_real

# ============================================================================
# Utility functions
# ============================================================================

"""Generate Chebyshev collocation points"""
function chebypoints(N::Int, a::Real = -1, b::Real = 1)
    @assert N > 0 "N must be positive"
    points = zeros(Float64, N)
    center = (b + a) / 2
    radius = (b - a) / 2

    for j = 1:N
        points[j] = center + radius * cos((j - 1) * π / (N - 1))
    end

    return points
end

"""Create pure Chebyshev polynomial T_n"""
function chebyshev(::Type{T}, N::Int, n::Int, normalize::Bool = false) where {T<:Number}
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
chebyshev(N::Int, n::Int, normalize::Bool = false) = chebyshev(Float64, N, n, normalize)

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

    for m = 2:n
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
function random_complex_chebycoeff(
    N::Int,
    magnitude::Real = 1.0,
    decay::Real = 0.8,
    a::Real = -1,
    b::Real = 1,
    bc_a::BC = Diri,
    bc_b::BC = Diri,
)
    u = ChebyCoeff{ComplexF64}(N, a, b, Spectral)

    mag = magnitude
    for n = 1:N
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
function is_effectively_real(u::ChebyCoeff{ComplexF64}, tol::Real = 1e-14)
    return all(abs.(imag.(u.data)) .< tol)
end
