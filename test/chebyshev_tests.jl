using Test
using Channelflow


@testset "Chebyshev Grid Tests" begin
    N = 32
    L1, L2 = -2.0, 3.0
    grid = ChebyshevGrid(N, (L1, L2))

    @test length(grid.x) == N
    @test minimum(grid.x) ≈ L1 atol = 1e-12
    @test maximum(grid.x) ≈ L2 atol = 1e-12

    # Test D on u(x) = x^2, where u'' = 2
    u = grid.x .^ 2
    uxx_exact = fill(2.0, N)
    uxx_numeric = grid.D2 * u
    @test maximum(abs.(uxx_exact - uxx_numeric)) < 1e-3
end
