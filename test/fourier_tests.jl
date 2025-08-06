using Test
using Channelflow

@testset "Fourier Grid Tests" begin
    N = 32
    L = 2π
    grid = FourierGrid(N, L)

    @test length(grid.x) == N
    @test grid.x[1] ≈ 0 atol = 1e-12
    @test isapprox(grid.x[end], L, atol=1e-12) == false  # periodic, excludes endpoint

    # Test derivative on u(x) = sin(x), where u' = cos(x)
    u = sin.(grid.x)
    derivative = get_derivative_matrix(grid, 1)

    u_exact = cos.(grid.x)
    u_numeric = real(derivative * u)
    @test maximum(abs.(u_exact - u_numeric)) < 1e-10
end
