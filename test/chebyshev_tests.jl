using Test
using Channelflow
using LinearAlgebra

@testset "ChebyCoeff Tests" begin

    @testset "Construction and Basic Properties" begin
        # Test various constructors
        u1 = ChebyCoeff(10)
        @test length(u1) == 10
        @test u1.a == -1.0 && u1.b == 1.0
        @test u1.state == Spectral

        u2 = ChebyCoeff{ComplexF64}(5, 0, π, Physical)
        @test eltype(u2) == ComplexF64
        @test u2.a == 0.0
        @test u2.b ≈ pi
        @test u2.state == Physical

        # Test data constructor
        data = randn(8)
        u3 = ChebyCoeff(data, -2, 2, Spectral)
        @test u3.data == data
        @test bounds(u3) == (-2.0, 2.0)
    end

    @testset "Transform Verification" begin
        N = 16

        # Test 1: Transform of Chebyshev polynomial should be exact
        u = ChebyCoeff(N, -1, 1, Spectral)
        u[3] = 1.0  # T_2(x) = 2x^2 - 1

        makePhysical!(u)
        @test u.state == Physical

        # Check values at Chebyshev points
        points = chebypoints(N)
        for (i, x) in enumerate(points)
            expected = 2 * x^2 - 1  # T_2(x)
            @test abs(u.data[i] - expected) < 1e-12
        end

        # Transform back
        makeSpectral!(u)
        @test u.state == Spectral
        @test abs(u[3] - 1.0) < 1e-12
        @test maximum(abs, u.data[[1:2; 4:end]]) < 1e-12
    end

    @testset "Evaluation Functions" begin
        # Test with known polynomial: u(x) = x^2
        N = 10
        u = ChebyCoeff(N, -1, 1, Spectral)
        u[1] = 0.5   # T_0 coefficient for x^2 on [-1,1]
        u[3] = 0.5   # T_2 coefficient

        # Test boundary evaluation
        @test abs(eval_a(u) - 1.0) < 1e-12  # (-1)^2 = 1
        @test abs(eval_b(u) - 1.0) < 1e-12  # (1)^2 = 1

        # Test evaluation at arbitrary point
        @test abs(evaluate(u, 0.0) - 0.0) < 1e-12  # 0^2 = 0
        @test abs(evaluate(u, 0.5) - 0.25) < 1e-12  # 0.5^2 = 0.25

        # Test derivative evaluation
        @test abs(slope_a(u) - (-2.0)) < 1e-12  # d/dx(x^2)|_{x=-1} = -2
        @test abs(slope_b(u) - 2.0) < 1e-12     # d/dx(x^2)|_{x=1} = 2
    end

    @testset "Differentiation" begin
        # Test derivative of x^2 should be 2x
        N = 10
        u = ChebyCoeff(N, -1, 1, Spectral)
        u[1] = 0.5
        u[3] = 0.5  # x^2 representation

        du = derivative(u)

        # 2x in Chebyshev basis is just T_1
        expected = ChebyCoeff(N, -1, 1, Spectral)
        expected[2] = 2.0  # T_1 coefficient

        @test maximum(abs, du.data - expected.data) < 1e-12
    end

    @testset "Integration" begin
        # Test integration of 2x should be x^2 (plus constant)
        N = 10
        u = ChebyCoeff(N, -1, 1, Spectral)
        u[2] = 2.0  # 2x

        int_u = integrate(u)

        # Should get x^2 representation (with zero mean)
        @test abs(int_u[3] - 0.5) < 1e-12  # T_2 coefficient
        @test abs(mean_value(int_u)) < 1e-12  # Mean should be zero
    end

    @testset "Arithmetic Operations" begin
        N = 8
        u = ChebyCoeff(randn(N))
        v = ChebyCoeff(randn(N))
        c = 2.5

        # Test scalar multiplication
        cu = c * u
        @test cu.data ≈ c * u.data

        # Test addition/subtraction
        w = u + v
        @test w.data ≈ u.data + v.data

        w = u - v
        @test w.data ≈ u.data - v.data

        # Test pointwise multiplication (in Physical state)
        u_phys = ChebyCoeff(u)
        v_phys = ChebyCoeff(v)
        makePhysical!(u_phys)
        makePhysical!(v_phys)

        w_phys = u_phys * v_phys
        @test w_phys.data ≈ u_phys.data .* v_phys.data
    end

    @testset "Norm Functions" begin
        # Test with known function
        N = 16
        u = ChebyCoeff(N, -1, 1, Spectral)
        u[2] = 1.0  # u(x) = x

        # L2 norm of x on [-1,1] should be sqrt(2/3)
        @test abs(L2Norm(u, false) - sqrt(2 / 3)) < 1e-10

        # Mean value should be 0
        @test abs(mean_value(u)) < 1e-14
    end
end
