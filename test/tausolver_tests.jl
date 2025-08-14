using Test
using ..TauSolvers
using ..ChebyCoeffs
using ..HelmholtzSolver

@testset "TauSolver Tests" begin
N = 15
a, b = -1.0, 1.0
length_x, length_z = 2π, 2π
kx, kz = 1, 1
lambda, nu = 1.0, 0.1
@testset "TauSolver Construction" begin
    tau = TauSolver(kx, kz, length_x, length_z, a, b, lambda, nu, N, true)
    @test tau.num_modes == N
    @test tau.kx == kx
    @test tau.kz == kz
    @test tau.lambda == lambda
    @test tau.nu == nu
    @test tau.tau_correction == true
    @test isa(tau.P_0, ChebyCoeff)
    @test isa(tau.v_0, ChebyCoeff)
end

@testset "TauSolver: Solve zero RHS" begin
    tau = TauSolver(kx, kz, length_x, length_z, a, b, lambda, nu, N, true)
    u = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    v = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    w = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    P = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    Rx = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    Ry = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    Rz = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    set_to_zero!(Rx)
    set_to_zero!(Ry)
    set_to_zero!(Rz)
    solve!(tau, u, v, w, P, Rx, Ry, Rz)
    @test all(isapprox.(u.data, 0.0, atol=1e-10))
    @test all(isapprox.(v.data, 0.0, atol=1e-10))
    @test all(isapprox.(w.data, 0.0, atol=1e-10))
    @test all(isapprox.(P.data, 0.0, atol=1e-10))
end

@testset "TauSolver: Real/Imag decoupling" begin
    tau = TauSolver(kx, kz, length_x, length_z, a, b, lambda, nu, N, true)
    u = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    v = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    w = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    P = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    Rx = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    Ry = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    Rz = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    # Set only real part of Ry
    Ry.data[1] = 1.0 + 0im
    solve!(tau, u, v, w, P, Rx, Ry, Rz)
    # Only real part of v and P should be nonzero
    @test all(isapprox.(imag.(v.data), 0.0, atol=1e-10))
    @test all(isapprox.(imag.(P.data), 0.0, atol=1e-10))
end

@testset "TauSolver: Degenerate case kx=kz=0" begin
    tau = TauSolver(0, 0, length_x, length_z, a, b, lambda, nu, N, true)
    u = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    v = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    w = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    P = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    Rx = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    Ry = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    Rz = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    Ry.data[3] = 2.0 + 0im
    solve!(tau, u, v, w, P, Rx, Ry, Rz)
    # v should be zero for degenerate case
    @test all(isapprox.(v.data, 0.0, atol=1e-10))
end

@testset "TauSolver: Consistency with known solution" begin
    tau = TauSolver(kx, kz, length_x, length_z, a, b, lambda, nu, N, true)
    u = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    v = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    w = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    P = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    Rx = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    Ry = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    Rz = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
    # Set up a manufactured solution (e.g., all ones)
    Rx.data .= 1.0 + 0im
    Ry.data .= 1.0 + 0im
    Rz.data .= 1.0 + 0im
    solve!(tau, u, v, w, P, Rx, Ry, Rz)
    # Check that solution is finite and not NaN
    @test all(isfinite, real.(u.data))
    @test all(isfinite, real.(v.data))
    @test all(isfinite, real.(w.data))
    @test all(isfinite, real.(P.data))
end

@testset "TauSolver: Influence correction does not error" begin
    tau = TauSolver(kx, kz, length_x, length_z, a, b, lambda, nu, N, true)
    P = ChebyCoeff{Float64}(N, a, b, Spectral)
    v = ChebyCoeff{Float64}(N, a, b, Spectral)
    @test isnothing(influence_correction!(tau, P, v))
end
end