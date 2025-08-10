using Test
using Channelflow

@testset "Norm Functions" begin
    # Test with known function
    N = 16
    u = ChebyCoeff(N, -1, 1, Spectral)
    u[2] = 1.0  # u(x) = 1.0 T_1(x) = 1.0 arccos(cos(x)) = x

    # L2 norm of x on [-1,1] should be sqrt(2/3)
    @test abs(L2Norm(u, false) - sqrt(2 / 3)) < 1e-10

    # Mean value should be 0
    @test abs(mean_value(u)) < 1e-14
end