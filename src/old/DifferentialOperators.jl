module DifferentialOperators

using ..Collocation
using LinearAlgebra

export differentiation_operators, differentiation_operator

"""
    differentiation_operator(basis::Symbol, N::Int, L::Real, order::Int)

Return the `order`-th differentiation matrix for the given basis type (`:Fourier` or `:Chebyshev`),
grid size `N`, and domain length `L` (interpreted as `(-L/2, L/2)` for Fourier, or `(0, L)` for Chebyshev).
"""
function differentiation_operator(basis::Symbol, N::Int, L::Real, order::Int)
    @assert order ≥ 1 "Derivative order must be at least 1"

    if basis == :Fourier
        return fourier_derivative_matrix(N, L, order)
    elseif basis == :Chebyshev
        return chebyshev_derivative_matrix(N, L, order)
    else
        error("Unsupported basis: $basis. Use :Fourier or :Chebyshev.")
    end
end

"""
    differentiation_operators(basis::Symbol, N::Tuple{Int, Int}, L::Tuple{Real, Real}, order::Tuple{Int, Int}= (1, 1))

Returns a tuple of matrices depending on the `basis`, grid size `N`, domain size `L`, and differentiation order `order`.
"""
function differentiation_operators(basis::Symbol, N::Tuple{Int,Int}, L::Tuple{Real,Real}, order::Tuple{Int,Int}=(1, 2))
    D1 = differentiation_operator(basis, N[1], L[1], order[1])
    D2 = differentiation_operator(basis, N[2], L[2], order[2])
    return (D1, D2)
end

end

