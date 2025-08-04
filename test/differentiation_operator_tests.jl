using Test
using Channelflow

@testset "differentiation_operator" begin
    N = 8
    L = 2π

    # Fourier, order 1
    D1 = differentiation_operator(:Fourier, N, L, 1)
    @test size(D1) == (N, N)

    # Fourier, order 2
    D2 = differentiation_operator(:Fourier, N, L, 2)
    @test size(D2) == (N, N)

    @test D1 != D2

    # Chebyshev, (L1, L2)
    D1 = differentiation_operator(:Chebyshev, N, L, 2)
    @test size(D1) == (N, N)

    # Invalid basis
    @test_throws ErrorException differentiation_operator(:Legendre, N, L, 1)

    # Invalid order
    @test_throws AssertionError differentiation_operator(:Fourier, N, L, 0)

    # Invalid domain tuple
    @test_throws MethodError differentiation_operator(:Chebyshev, N, (1.0,), 1)


    # Double creation

    N_tuple = (N, N)
    L_tuple = (π, π)
    # Fourier, order 1
    Ds = differentiation_operators(:Fourier, N_tuple, L_tuple, (1, 1))
    @test length(Ds) == 2
    @test all(size(D) == (N, N) for D in Ds)

    # Fourier, order 2
    Ds = differentiation_operators(:Fourier, N_tuple, L_tuple, (1, 2))
    @test length(Ds) == 2
    @test all(size(D) == (N, N) for D in Ds)
    @test Ds[1] != Ds[2]

    # Chebyshev, order 2
    Ds = differentiation_operators(:Chebyshev, N_tuple, L_tuple, (1, 2))
    @test length(Ds) == 2
    @test all(size(D) == (N, N) for D in Ds)

    # Invalid basis
    @test_throws ErrorException differentiation_operators(:Legendre, N_tuple, L_tuple, (1, 2))

    # Invalid order
    @test_throws AssertionError differentiation_operators(:Fourier, N_tuple, L_tuple, (0, 0))

    # Invalid domain
    @test_throws MethodError differentiation_operators(:Chebyshev, N, (1.0,), 1)
end

