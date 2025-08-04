module Grid

using LinearAlgebra

export AbstractGrid, SpectralGrid, CollocationPoints, Weights, WaveNumbers, DerivativeMatrix, collocation_points

abstract type AbstractGrid end

abstract type SpectralGrid <: AbstractGrid end

const CollocationPoints = Vector{Float64}
const Weights = Vector{Float64}
const WaveNumbers = Vector{Float64}

struct DerivativeMatrix{Order}
    data::Matrix{Float64}
end
Base.getindex(D::DerivativeMatrix, i::Int, j::Int) = D.data[i, j]
Base.size(D::DerivativeMatrix) = size(D.data)
Base.:*(D::DerivativeMatrix, v::AbstractArray) = D.data * v

function collocation_points(grid::AbstractGrid)
    grid.x
end

end

