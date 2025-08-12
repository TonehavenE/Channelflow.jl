using Test
using Printf
using Channelflow
import Base.@kwdef

@kwdef struct HelmholtzTestCase
    a::Real
    b::Real
    ua::Real
    ub::Real
    u::Function
    f::Function
end


@testset "Helmholtz Tests" begin
    # global config
    allowable_error = 1e-12
    N = 63

    """
    Approximates the true `u` on the domain [a, b] 
    using boundary conditions u(a) = `ua`, u(b) = `ub`.
    Ranges λ and ν in the equation
    ```math
    νu'' - λu = f
    ```
 `f` should be a callable that takes in x, lambda, and nu.
 `u` should be a callable that takes in x. 
    """
    function evaluate_case(
        h::HelmholtzTestCase;
        N=63,
        allowable_error::Real=1e-12,
        lambdas=[0.5, 1.0, 2.0, 5.0],
        nus=[1.0, 5.0, 10.0, 100.0],
    )
        for lambda in lambdas
            for nu in nus
                x = chebypoints(N, h.a, h.b)
                f_data = h.f.(x, lambda, nu)
                u_exact = h.u.(x)

                rhs = ChebyCoeff(f_data, h.a, h.b, Physical)
                make_spectral!(rhs)

                prob = HelmholtzProblem(N, h.a, h.b, lambda, nu)
                u = solve(prob, rhs, h.ua, h.ub)
                make_physical!(u)

                max_error = maximum(abs.(u.data - u_exact))
                @test max_error < allowable_error
            end
        end
    end

    @testset "Known Solutions" begin
        # ----- u(x) = sin(kx) -----
        # vary k
        for k = -10:10
            test_case_sin = HelmholtzTestCase(
                a=0.0,
                b=2 * pi,
                ua=0.0,
                ub=0.0,
                u=x -> sin(k * x),
                f=(x, lambda, nu) -> -nu * k^2 * sin(k * x) - lambda * sin(k * x),
            )
            evaluate_case(test_case_sin)
        end


        # ----- u(x) = 1 - x^2 -----
        test_case_quadratic = HelmholtzTestCase(
            a=-1.0,
            b=1.0,
            ua=0.0,
            ub=0.0,
            u=x -> 1 - x^2,
            f=(x, lambda, nu) -> lambda * x^2 - lambda - 2 * nu,
        )
        evaluate_case(test_case_quadratic)

        # ----- u(x) = e^{kx} -----
        # vary k
        for k = -10:10
            test_case_e = HelmholtzTestCase(
                a=0.0,
                b=1.0,
                ua=1.0,
                ub=exp(k),
                u=x -> exp(k * x),
                f=(x, lambda, nu) -> k^2 * nu * exp(k * x) - lambda * exp(k * x),
            )
            evaluate_case(test_case_e; allowable_error=1.0e-10)
        end
    end

    @testset "Mean Constraint Test" begin
        N = 21
        h = HelmholtzProblem(N, 0, π, 1.0, 1.0)

        # Test with mean constraint
        rhs = ChebyCoeff(N, 0, π, Spectral)
        rhs[1] = 1.0  # Constant forcing

        target_mean = 2.0
        u = ChebyCoeff(N, 0, π, Spectral)
        solve!(h, u, rhs, target_mean, 0.0, 0.0)

        computed_mean = mean_value(u)
        @test abs(computed_mean - target_mean) < 1e-10
    end
end
