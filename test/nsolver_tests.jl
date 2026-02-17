using Channelflow
using LinearAlgebra
using Test

@testset "NSolver GMRES" begin
    A = [4.0 1 0; 1 3 1; 0 1 2]
    b = [1.0, 2.0, 0.0]
    g = GMRES(b, 8)
    for _ in 1:8
        q = test_vector(g)
        iterate!(g, A * q)
        residual(g) < 1e-10 && break
    end
    @test norm(A * solution(g) - b) < 1e-7
end

@testset "NSolver FGMRES" begin
    A = [3.0 1.0; 1.0 2.0]
    b = [1.0, -1.0]
    g = FGMRES(b, 6)
    for _ in 1:6
        q = test_vector(g)
        iterate!(g, q, A * q)
        residual(g) < 1e-10 && break
    end
    @test norm(A * solution(g) - b) < 1e-7
end

@testset "NSolver BiCGStab" begin
    A = [4.0 1 0; 1 3 1; 0 1 2]
    b = [1.0, 2.0, 0.0]
    bi = BiCGStab(b)
    for _ in 1:20
        p = step1!(bi)
        s = step2!(bi, A * p)
        step3!(bi, A * s)
        residual(bi) < 1e-10 && break
    end
    @test norm(A * solution(bi) - b) < 1e-6
end

@testset "NSolver NewtonAlgorithm" begin
    dsi = FunctionDSI(x -> [x[1]^2 - 2.0])
    alg = NewtonAlgorithm(NewtonSearchFlags(n_newton = 20, n_solver = 10, eps_search = 1e-12, optimization = :hookstep, delta = 0.5))
    x, gx = Channelflow.NSolver.solve(alg, dsi, [1.0])
    @test gx < 1e-10
    @test isapprox(x[1], sqrt(2), atol = 1e-8)
end


@testset "NSolver continuation in parameter" begin
    dsi = ParamDSI((x, μ) -> [x[1]^2 - μ], 1.0)
    nflags = NewtonSearchFlags(n_newton = 25, n_solver = 15, eps_search = 1e-12, optimization = :hookstep, delta = 0.5)
    cflags = ContinuationFlags(ds = 0.2, n_steps = 6, newton = nflags)

    branch = continue_branch(dsi, [1.0], 1.0; flags = cflags, direction = 1.0)

    @test length(branch) == 6
    @test issorted([p.μ for p in branch])
    for p in branch
        @test p.residual < 1e-8
        @test isapprox(p.x[1], sqrt(p.μ), atol = 1e-5)
    end
end
