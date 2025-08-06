module HelmholtzSolver

using ..Collocation
using FFTW, LinearAlgebra

export solve_helmholtz
"""
    solve_helmholtz(grid::AbstractGrid, α::Real, β::Real, f::Vector{Float64})

Solve the Helmholtz equation `(α * I - β * D2) u = f` for a given 1D grid.

Returns the solution vector `u`.
"""
function solve_helmholtz(grid::AbstractGrid, α::Real, β::Real, f::Vector{Float64}; bc::Union{Nothing,Dict})
    error("solve_helmholtz not implemented for grid type $(typeof(grid))")
end


function apply_boundary_conditions!(grid::AbstractGrid, A::Matrix, f::Vector, bc::Dict)
    # Apply BC modifications depending on bc dict
    if haskey(bc, :left)
        bctype, bcval = bc[:left]
        if bctype == :Dirichlet
            A[1, :] .= 0
            A[1, 1] = 1
            f[1] = bcval
        elseif bctype == :Neumann
            A[1, :] .= grid.D[1].data[1, :]
            f[1] = bcval
        else
            error("Unsupported left BC type $bctype")
        end
    end
    if haskey(bc, :right)
        bctype, bcval = bc[:right]
        if bctype == :Dirichlet
            A[end, :] .= 0
            A[end, end] = 1
            f[end] = bcval
        elseif bctype == :Neumann
            A[end, :] .= grid.D[1].data[end, :]
            f[end] = bcval
        else
            error("Unsupported right BC type $bctype")
        end
    end # Apply BC modifications depending on bc dict
    if haskey(bc, :left)
        bctype, bcval = bc[:left]
        if bctype == :Dirichlet
            A[1, :] .= 0
            A[1, 1] = 1
            f[1] = bcval
        elseif bctype == :Neumann
            A[1, :] .= grid.D[1].data[1, :]
            f[1] = bcval
        else
            error("Unsupported left BC type $bctype")
        end
    end
    if haskey(bc, :right)
        bctype, bcval = bc[:right]
        if bctype == :Dirichlet
            A[end, :] .= 0
            A[end, end] = 1
            f[end] = bcval
        elseif bctype == :Neumann
            A[end, :] .= grid.D[1].data[end, :]
            f[end] = bcval
        else
            error("Unsupported right BC type $bctype")
        end
    end
end

function solve_helmholtz(grid::ChebyshevGrid, α::Real, β::Real, f::Vector{Float64}; bc::Union{Nothing,Dict}=nothing)
    N = length(f)
    @assert size(get_derivative_matrix(grid, 2).data) == (N, N) "Dimension mismatch between D2 and f"

    # Build Helmholtz matrix
    A = α * I(N) - β * get_derivative_matrix(grid, 2).data

    if bc !== nothing
        apply_boundary_conditions!(grid, A, f, bc)
    end

    # Solve linear system
    u = A \ f
    return u
end

function solve_helmholtz(grid::FourierGrid, α::Real, β::Real, f::Vector{Float64})
    N = length(f)
    k = grid.wave_numbers
    @assert length(k) == N "Dimension mismatch between k and f"

    f_hat = fft(f)
    denom = α .+ β .* (k .^ 2)

    # Avoid division by zero (if any denom entries are zero, handle accordingly)
    @assert all(abs.(denom) .> 1e-14) "Denominator near zero, check α and β"

    u_hat = f_hat ./ denom
    u = real(ifft(u_hat))
    return u
end

end
