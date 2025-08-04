module Fourier

using ..Grid
using LinearAlgebra, FFTW

export FourierGrid, fourier_derivative_matrix

struct FourierGrid <: SpectralGrid
    N::Int
    L::Real
    x::CollocationPoints
    wave_numbers::WaveNumbers
end

function fourier_points(N::Int, L::Real)
    dx = L / N
    return dx .* collect(0:N-1)
end

function wavenumbers(N::Int, L::Real)
    k = [0:N÷2; -N÷2+1:-1] .* (2π / L)
    return Float64.(k)
end

function derivative(f::Vector{Float64}, k::WaveNumbers, order::Int=1)
    f̂ = fft(f)
    df̂ = (im .* k) .^ order .* f̂
    return real(ifft(df̂))
end

function fourier_derivative_matrix(N::Int, L::Real, order::Int=1)::DerivativeMatrix
    # x = fourier_points(N, L)
    k = wavenumbers(N, L)
    Id = Matrix{Float64}(I(N))

    # Construct the full derivative matrix by applying FFT method to each basis vector
    D = hcat([derivative(Id[:, j], k, order) for j in 1:N]...)
    return DerivativeMatrix{order}(D)
end

function FourierGrid(N::Int, L::Real)::FourierGrid
    x = fourier_points(N, L)
    k = wavenumbers(N, L)
    return FourierGrid(N, L, x, k)
end


end
