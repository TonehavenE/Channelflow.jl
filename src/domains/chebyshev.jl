module Chebyshev

using ..Grid
using LinearAlgebra

export ChebyshevGrid, chebyshev_derivative_matrix

struct ChebyshevGrid <: SpectralGrid
    N::Int
    L::Tuple{<:Real,<:Real}
    x::CollocationPoints
    weights::Weights
    D1::DerivativeMatrix{1}
    D2::DerivativeMatrix{2}
end


"""
	chebyshev_points(N, (L1, L2))

Computes `N` Chebyshev-Gauss-Lobatto points on the domain `[L1, L2]`
"""
function chebyshev_points(N::Int, L::Tuple{<:Real,<:Real})
    L1, L2 = L
    x_cheb = cos.(pi .* (0:N-1) ./ (N - 1))
    return (L2 - L1) / 2 .* x_cheb .+ (L1 + L2) / 2
end

"""
	chebyshev_points(N, L)

Computes `N` Chebyshev-Gauss-Lobatto points on the domain `[-L, L]`.
"""
function chebyshev_points(N::Int, L::Real)
    chebyshev_points(N, (-L, L))
end


"""
	chebyshev_points(points, degree)

Generates the Chebyshev Derivative Matrix of `degree` at the (Chebyshev-Gauss-Lobatto) `points`.
"""
function chebyshev_derivative_matrix(points::CollocationPoints, degree::Int=1)::DerivativeMatrix
    if degree < 1
        throw(ArgumentError("Degree must be ≥ 1"))
    end

    # Compute D1
    N = length(points)
    X = repeat(points, 1, N)
    deltaX = X .- X'

    c = [2; ones(N - 2); 2] .* (-1) .^ (0:N-1)
    C = c * (1 ./ c)'

    # Assign off-diagonal elements
    D = C ./ (deltaX .+ I(N))
    # Compute diagonal elements as the values which make the row-sums 0
    D = D .- diagm(0 => sum(D, dims=2)[:])

    # Compute to degree K
    Dk = D
    for _ in 2:degree
        Dk = Dk * D
    end
    return DerivativeMatrix{degree}(Dk)
end

function chebyshev_derivative_matrix(N::Int, L::Union{<:Real,Tuple{<:Real,<:Real}}, degree::Int=1)::DerivativeMatrix
    x_cheb = chebyshev_points(N, L)
    return chebyshev_derivative_matrix(x_cheb, degree)
end

function ChebyshevGrid(N::Int, L::Tuple{<:Real,<:Real}=(-1.0, 1.0))::ChebyshevGrid
    w = Weights(ones(N))  # Placeholder; can be improved later
    x = chebyshev_points(N, L)
    D1 = chebyshev_derivative_matrix(x, 1)
    D2 = DerivativeMatrix{2}(D1.data * D1.data)
    return ChebyshevGrid(N, L, x, w, D1, D2)
end

end
