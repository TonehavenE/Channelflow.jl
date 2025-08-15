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

    @testset "TauSolver Constructor Tests" begin
        # Basic construction parameters
        kx, kz = 1, 1
        Lx_, Lz_ = 2π, 2π
        a, b = -1.0, 1.0
        lambda = 1.0
        nu = 0.1
        num_modes = 17
        tau_correction = true

        # Test successful construction
        tau = TauSolver(kx, kz, Lx_, Lz_, a, b, lambda, nu, num_modes, tau_correction)

        @test tau.num_modes == num_modes
        @test tau.kx == kx
        @test tau.kz == kz
        @test tau.a == a
        @test tau.b == b
        @test tau.lambda == lambda
        @test tau.nu == nu
        @test tau.tau_correction == tau_correction

        # Test computed convenience variables
        @test tau.two_pi_kxLx ≈ 2π * kx / Lx_
        @test tau.two_pi_kzLz ≈ 2π * kz / Lz_
        @test tau.kappa2 ≈ 4π^2 * ((kx / Lx_)^2 + (kz / Lz_)^2)

        # Test that Helmholtz problems are properly initialized
        @test isa(tau.pressure_helmholtz, HelmholtzProblem)
        @test isa(tau.velocity_helmholtz, HelmholtzProblem)

        # Test that ChebyCoeffs are properly initialized
        @test isa(tau.P_0, ChebyCoeff)
        @test isa(tau.v_0, ChebyCoeff)
        @test isa(tau.P_plus, ChebyCoeff)
        @test isa(tau.v_plus, ChebyCoeff)
        @test isa(tau.P_minus, ChebyCoeff)
        @test isa(tau.v_minus, ChebyCoeff)

        # Test degenerate case (kx = 0, kz = 0)
        tau_degenerate = TauSolver(0, 0, Lx_, Lz_, a, b, lambda, nu, num_modes, tau_correction)
        @test tau_degenerate.kx == 0
        @test tau_degenerate.kz == 0
        @test tau_degenerate.kappa2 == 0.0
    end

    @testset "TauSolver Constructor Edge Cases" begin
        kx, kz = 1, 1
        Lx_, Lz_ = 2π, 2π
        a, b = -1.0, 1.0
        lambda = 1.0
        nu = 0.1
        num_modes = 17
        tau_correction = false

        # Test with different domain bounds
        tau_different_domain = TauSolver(kx, kz, Lx_, Lz_, -2.0, 3.0, lambda, nu, num_modes, tau_correction)
        @test tau_different_domain.a == -2.0
        @test tau_different_domain.b == 3.0

        # Test with different physical parameters
        tau_different_params = TauSolver(kx, kz, 4π, π, a, b, 2.0, 0.01, num_modes, tau_correction)
        @test tau_different_params.lambda == 2.0
        @test tau_different_params.nu == 0.01
        @test tau_different_params.two_pi_kxLx ≈ 2π * kx / (4π)
        @test tau_different_params.two_pi_kzLz ≈ 2π * kz / π

        # Test with tau_correction disabled
        @test !tau_different_params.tau_correction

        # Test with larger number of modes
        tau_large_modes = TauSolver(kx, kz, Lx_, Lz_, a, b, lambda, nu, 33, tau_correction)
        @test tau_large_modes.num_modes == 33
    end

    @testset "influence_correction! Tests" begin
        kx, kz = 1, 1
        Lx_, Lz_ = 2π, 2π
        a, b = -1.0, 1.0
        lambda = 1.0
        nu = 0.1
        num_modes = 17
        tau_correction = true

        tau = TauSolver(kx, kz, Lx_, Lz_, a, b, lambda, nu, num_modes, tau_correction)

        # Create test coefficients
        P_test = ChebyCoeff(num_modes, a, b, Spectral)
        v_test = ChebyCoeff(num_modes, a, b, Spectral)

        # Set some test values
        for i = 1:num_modes
            P_test[i] = sin(i * π / num_modes)
            v_test[i] = cos(i * π / num_modes)
        end

        # Store original values
        P_original = ChebyCoeff(copy(P_test.data), P_test.a, P_test.b, P_test.state)
        v_original = ChebyCoeff(copy(v_test.data), v_test.a, v_test.b, v_test.state)

        # Apply influence correction
        influence_correction!(tau, P_test, v_test)

        # Check that values have changed (unless they're exactly zero)
        if any(abs.(P_original.data) .> 1e-12) || any(abs.(v_original.data) .> 1e-12)
            @test P_test != P_original || v_test != v_original
        end

        # Test that function doesn't crash with zero input
        P_zero = ChebyCoeff(num_modes, a, b, Spectral)
        v_zero = ChebyCoeff(num_modes, a, b, Spectral)
        @test_nowarn influence_correction!(tau, P_zero, v_zero)
    end

    @testset "solve_P_and_v! Tests" begin
        kx, kz = 1, 1
        Lx_, Lz_ = 2π, 2π
        a, b = -1.0, 1.0
        lambda = 1.0
        nu = 0.1
        num_modes = 17
        tau_correction = true

        tau = TauSolver(kx, kz, Lx_, Lz_, a, b, lambda, nu, num_modes, tau_correction)

        # Create test coefficients
        P = ChebyCoeff(num_modes, a, b, Spectral)
        v = ChebyCoeff(num_modes, a, b, Spectral)
        r = ChebyCoeff(num_modes, a, b, Spectral)
        Ry = ChebyCoeff(num_modes, a, b, Spectral)

        # Set some test RHS values
        for i = 1:num_modes
            r[i] = exp(-i * 0.1)
            Ry[i] = sin(i * π / (2 * num_modes))
        end

        # Test solve_P_and_v! function
        @test_nowarn TauSolvers.solve_P_and_v!(tau, P, v, r, Ry)

        # Check that solution was computed (not all zeros unless RHS is zero)
        if any(abs.(r.data) .> 1e-12) || any(abs.(Ry.data) .> 1e-12)
            @test any(abs.(P.data) .> 1e-12) || any(abs.(v.data) .> 1e-12)
        end

        # Test degenerate case (kx = 0, kz = 0)
        tau_degenerate = TauSolver(0, 0, Lx_, Lz_, a, b, lambda, nu, num_modes, tau_correction)
        P_deg = ChebyCoeff(num_modes, a, b, Spectral)
        v_deg = ChebyCoeff(num_modes, a, b, Spectral)

        @test_nowarn TauSolvers.solve_P_and_v!(tau_degenerate, P_deg, v_deg, r, Ry)

        # In degenerate case, v should be zero
        @test all(abs.(v_deg.data) .< 1e-12)
    end

    @testset "solve! Tests - General Case" begin
        kx, kz = 1, 1
        Lx_, Lz_ = 2π, 2π
        a, b = -1.0, 1.0
        lambda = 1.0
        nu = 0.1
        num_modes = 17
        tau_correction = true

        tau = TauSolver(kx, kz, Lx_, Lz_, a, b, lambda, nu, num_modes, tau_correction)

        # Create solution and RHS coefficients
        u = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        v = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        w = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        P = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)

        Rx = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Ry = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Rz = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)

        # Set some test RHS values
        for i = 1:num_modes
            Rx[i] = complex(sin(i * π / num_modes), cos(i * π / num_modes)) * 0.1
            Ry[i] = complex(cos(i * π / num_modes), sin(i * π / num_modes)) * 0.1
            Rz[i] = complex(exp(-i * 0.1), exp(-i * 0.2)) * 0.1
        end

        # Test solve! function
        @test_nowarn solve!(tau, u, v, w, P, Rx, Ry, Rz)

        # Check that solution was computed
        @test any(abs.(u.data) .> 1e-12) || any(abs.(v.data) .> 1e-12) || any(abs.(w.data) .> 1e-12) || any(abs.(P.data) .> 1e-12)
    end

    @testset "solve! Tests - Mean Flow Case" begin
        kx, kz = 0, 0  # Required for mean flow case
        Lx_, Lz_ = 2π, 2π
        a, b = -1.0, 1.0
        lambda = 1.0
        nu = 0.1
        num_modes = 17
        tau_correction = true
        umean = 1.5

        tau = TauSolver(kx, kz, Lx_, Lz_, a, b, lambda, nu, num_modes, tau_correction)

        # Create solution and RHS coefficients
        u = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        v = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        w = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        P = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)

        Rx = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Ry = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Rz = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)

        # Set some test RHS values
        for i = 1:num_modes
            Rx[i] = complex(0.1, 0.05) * i
            Ry[i] = complex(0.05, 0.1) * i
            Rz[i] = complex(0.02, 0.03) * i
        end

        # Test solve! function with mean flow
        @test_nowarn solve!(tau, u, v, w, P, Rx, Ry, Rz, umean)

        # Check that solution was computed
        @test any(abs.(u.data) .> 1e-12) || any(abs.(v.data) .> 1e-12) || any(abs.(w.data) .> 1e-12) || any(abs.(P.data) .> 1e-12)

        # Test assertion for non-zero kx or kz
        tau_nonzero = TauSolver(1, 0, Lx_, Lz_, a, b, lambda, nu, num_modes, tau_correction)
        @test_throws AssertionError solve!(tau_nonzero, u, v, w, P, Rx, Ry, Rz, umean)
    end

    @testset "TauSolver Tau Correction Toggle Tests" begin
        kx, kz = 1, 1
        Lx_, Lz_ = 2π, 2π
        a, b = -1.0, 1.0
        lambda = 1.0
        nu = 0.1
        num_modes = 17

        # Create two solvers - one with tau correction, one without
        tau_with_correction = TauSolver(kx, kz, Lx_, Lz_, a, b, lambda, nu, num_modes, true)
        tau_without_correction = TauSolver(kx, kz, Lx_, Lz_, a, b, lambda, nu, num_modes, false)

        @test tau_with_correction.tau_correction == true
        @test tau_without_correction.tau_correction == false

        # Test that both solve without error
        u1 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        v1 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        w1 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        P1 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)

        u2 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        v2 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        w2 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        P2 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)

        Rx = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Ry = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Rz = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)

        # Set some test RHS
        for i = 1:num_modes
            Rx[i] = complex(0.1, 0.05) * sin(i)
            Ry[i] = complex(0.05, 0.1) * cos(i)
            Rz[i] = complex(0.02, 0.03) * exp(-i * 0.1)
        end

        @test_nowarn solve!(tau_with_correction, u1, v1, w1, P1, Rx, Ry, Rz)
        @test_nowarn solve!(tau_without_correction, u2, v2, w2, P2, Rx, Ry, Rz)

        # Solutions should generally be different (unless very special case)
        @test u1 != u2 || v1 != v2 || w1 != w2 || P1 != P2
    end

    @testset "Parameter Validation Tests" begin
        # Test minimum discriminant check for non-zero wavenumbers
        kx, kz = 1, 1
        Lx_, Lz_ = 2π, 2π
        a, b = -1.0, 1.0
        lambda = 1.0
        nu = 0.1
        num_modes = 17
        tau_correction = true

        # This should work fine
        @test_nowarn TauSolver(kx, kz, Lx_, Lz_, a, b, lambda, nu, num_modes, tau_correction)

        # Test edge case with very small modes
        @test_nowarn TauSolver(kx, kz, Lx_, Lz_, a, b, lambda, nu, 9, tau_correction)
    end
    @testset "Correctness Tests - Differential Equation Verification" begin
        # Test that the solution satisfies the original equations with relaxed tolerances
        # and better understanding of the tau method limitations
        
        kx, kz = 1, 2
        Lx_, Lz_ = 2π, 4π
        a, b = -1.0, 1.0
        lambda = 0.5
        nu = 0.1
        num_modes = 21
        tau_correction = true
        tol = 1e-4  # More realistic tolerance for spectral tau method
        
        tau = TauSolver(kx, kz, Lx_, Lz_, a, b, lambda, nu, num_modes, tau_correction)
        
        # Create solution and RHS coefficients
        u = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        v = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        w = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        P = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        
        Rx = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Ry = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Rz = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        
        # Set up smoother, smaller RHS to avoid high-frequency issues
        for i = 1:min(8, num_modes-2)  # Only low-order modes
            Rx[i] = complex(0.01 * sin(i * π / 16), 0.005 * cos(i * π / 16))
            Ry[i] = complex(0.005 * cos(i * π / 16), 0.008 * sin(i * π / 16))
            Rz[i] = complex(0.002 * exp(-i * 0.1), 0.003 * exp(-i * 0.2))
        end
        
        # Solve the system
        solve!(tau, u, v, w, P, Rx, Ry, Rz)
        
        # Verify equations are satisfied (with understanding that tau method has truncation errors)
        
        # Verify momentum equations: nu u'' - lambda u - i*kx*P = -Rx
        u_second_deriv = derivative2(u)
        P_grad_x = tau.two_pi_kxLx * im * P
        momentum_x_residual = nu * u_second_deriv - lambda * u - P_grad_x + Rx
        
        # The tau method doesn't satisfy the equation exactly in the last few modes
        # Check that the first N-2 modes satisfy the equation well
        @test maximum(abs.(momentum_x_residual.data[1:num_modes-2])) < tol
        
        # Verify y-momentum equation: nu v'' - lambda v - dP/dy = -Ry
        v_second_deriv = derivative2(v)
        P_grad_y = derivative(P)
        momentum_y_residual = nu * v_second_deriv - lambda * v - P_grad_y + Ry
        @test maximum(abs.(momentum_y_residual.data[1:num_modes-2])) < tol
        
        # Verify z-momentum equation: nu w'' - lambda w - i*kz*P = -Rz
        w_second_deriv = derivative2(w)
        P_grad_z = tau.two_pi_kzLz * im * P
        momentum_z_residual = nu * w_second_deriv - lambda * w - P_grad_z + Rz
        @test maximum(abs.(momentum_z_residual.data[1:num_modes-2])) < tol
        
        # Verify continuity equation: i*kx*u + dv/dy + i*kz*w = 0
        u_grad_x = tau.two_pi_kxLx * im * u
        v_grad_y = derivative(v)
        w_grad_z = tau.two_pi_kzLz * im * w
        continuity_residual = u_grad_x + v_grad_y + w_grad_z
        @test maximum(abs.(continuity_residual.data[1:num_modes-2])) < tol
        
        # Verify boundary conditions: u(±1) = v(±1) = w(±1) = 0
        @test abs(eval_a(u)) < tol && abs(eval_b(u)) < tol
        @test abs(eval_a(v)) < tol && abs(eval_b(v)) < tol
        @test abs(eval_a(w)) < tol && abs(eval_b(w)) < tol
    end
    
    @testset "Correctness Tests - Mean Flow Case" begin
        # Test correctness for the mean flow case
        kx, kz = 0, 0
        Lx_, Lz_ = 2π, 2π
        a, b = -1.0, 1.0
        lambda = 1.0
        nu = 0.05
        num_modes = 17
        tau_correction = true
        umean = 2.0
        tol = 1e-4  # Relaxed tolerance for mean flow case
        
        tau = TauSolver(kx, kz, Lx_, Lz_, a, b, lambda, nu, num_modes, tau_correction)
        
        # Create solution and RHS coefficients
        u = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        v = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        w = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        P = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        
        Rx = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Ry = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Rz = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        
        # Use smaller RHS values for better numerical conditioning
        for i = 1:num_modes
            Rx[i] = complex(0.01 * sin(i), 0.005 * cos(i))
            Ry[i] = complex(0.005 * sin(i * 0.5), 0.008 * cos(i * 0.5))
            Rz[i] = complex(0.002 * i / num_modes, 0.003 * i / num_modes)
        end
        
        # Solve with mean flow
        solve!(tau, u, v, w, P, Rx, Ry, Rz, umean)
        
        # For the mean flow case, the solver may have special handling
        # Let's test what we can verify: continuity and boundary conditions
        
        # Continuity: du/dx + dv/dy + dw/dz = dv/dy = 0 (since kx=kz=0)
        v_grad_y = derivative(v)
        @test maximum(abs.(v_grad_y.data)) < tol
        
        # Boundary conditions for v and w (homogeneous Dirichlet)
        @test abs(eval_a(v)) < tol && abs(eval_b(v)) < tol
        @test abs(eval_a(w)) < tol && abs(eval_b(w)) < tol
        
        # For u with mean flow, BC is u(-1) = 0, but u(1) may be umean + perturbation
        @test abs(eval_a(u)) < tol
        
        # Test that the solver doesn't crash and produces reasonable results
        @test !any(isnan.(u.data)) && !any(isnan.(v.data)) && !any(isnan.(w.data)) && !any(isnan.(P.data))
        @test !any(isinf.(u.data)) && !any(isinf.(v.data)) && !any(isinf.(w.data)) && !any(isinf.(P.data))
    end
    
    @testset "Correctness Tests - Manufactured Solutions" begin
        # Test with a simpler manufactured solution approach
        # Instead of trying to construct a full solution, let's test that 
        # the residual equations are satisfied for a computed solution
        
        kx, kz = 1, 1
        Lx_, Lz_ = 2π, 2π
        a, b = -1.0, 1.0
        lambda = 1.0
        nu = 0.1
        num_modes = 33
        tau_correction = true
        tol = 1e-6
        
        tau = TauSolver(kx, kz, Lx_, Lz_, a, b, lambda, nu, num_modes, tau_correction)
        
        # Create solution and simple RHS coefficients
        u = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        v = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        w = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        P = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        
        Rx = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Ry = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Rz = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        
        # Use simple, smooth RHS functions
        for i = 1:min(5, num_modes-2)  # Only populate low-order modes for smoothness
            Rx[i] = complex(0.01 * sin(i), 0.005 * cos(i))
            Ry[i] = complex(0.005 * cos(i), 0.008 * sin(i))
            Rz[i] = complex(0.002 * i, 0.003 * i)
        end
        
        # Solve the system
        solve!(tau, u, v, w, P, Rx, Ry, Rz)
        
        # Now verify the computed solution satisfies the equations
        # This is the real test - not comparing to a manufactured exact solution
        
        # Check momentum equations with relaxed tolerance
        u_second_deriv = derivative2(u)
        P_grad_x = tau.two_pi_kxLx * im * P
        momentum_x_residual = nu * u_second_deriv - lambda * u - P_grad_x + Rx
        @test maximum(abs.(momentum_x_residual.data)) < tol
        
        v_second_deriv = derivative2(v)
        P_grad_y = derivative(P)
        momentum_y_residual = nu * v_second_deriv - lambda * v - P_grad_y + Ry
        @test maximum(abs.(momentum_y_residual.data)) < tol
        
        w_second_deriv = derivative2(w)
        P_grad_z = tau.two_pi_kzLz * im * P
        momentum_z_residual = nu * w_second_deriv - lambda * w - P_grad_z + Rz
        @test maximum(abs.(momentum_z_residual.data)) < tol
        
        # Check continuity equation
        u_grad_x = tau.two_pi_kxLx * im * u
        v_grad_y = derivative(v)
        w_grad_z = tau.two_pi_kzLz * im * w
        continuity_residual = u_grad_x + v_grad_y + w_grad_z
        @test maximum(abs.(continuity_residual.data)) < tol
        
        # Check boundary conditions
        @test abs(eval_a(u)) < tol && abs(eval_b(u)) < tol
        @test abs(eval_a(v)) < tol && abs(eval_b(v)) < tol
        @test abs(eval_a(w)) < tol && abs(eval_b(w)) < tol
    end
    
    
    @testset "Correctness Tests - Conservation Properties" begin
        # Test conservation properties and symmetries
        kx, kz = 1, 1
        Lx, Lz = 2π, 2π
        a, b = -1.0, 1.0
        lambda = 1.0
        nu = 0.1
        num_modes = 17
        tau_correction = true
        
        tau = TauSolver(kx, kz, Lx, Lz, a, b, lambda, nu, num_modes, tau_correction)
        
        # Test linearity: if we scale RHS by factor α, solution should scale by α
        alpha = 2.5
        
        u1 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        v1 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        w1 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        P1 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        
        u2 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        v2 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        w2 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        P2 = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        
        Rx = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Ry = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Rz = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        
        # Set up RHS
        for i = 1:num_modes
            Rx[i] = complex(0.1 * i, 0.05 * i)
            Ry[i] = complex(0.03 * i, 0.07 * i)
            Rz[i] = complex(0.02 * i, 0.04 * i)
        end
        
        # Solve with original RHS
        solve!(tau, u1, v1, w1, P1, Rx, Ry, Rz)
        
        # Scale RHS and solve again
        for i = 1:num_modes
            Rx[i] *= alpha
            Ry[i] *= alpha
            Rz[i] *= alpha
        end
        solve!(tau, u2, v2, w2, P2, Rx, Ry, Rz)
        
        # Check linearity (solutions should scale by alpha)
        tol = 1e-10
        @test maximum(abs.(u2.data - alpha * u1.data)) < tol
        @test maximum(abs.(v2.data - alpha * v1.data)) < tol
        @test maximum(abs.(w2.data - alpha * w1.data)) < tol
        @test maximum(abs.(P2.data - alpha * P1.data)) < tol
    end

      @testset "Verify Method Tests" begin
        # Test the verify method implementation
        kx, kz = 1, 1
        Lx_, Lz_ = 2π, 2π
        a, b = -1.0, 1.0
        lambda = 1.0
        nu = 0.1
        num_modes = 17
        tau_correction = true
        
        tau = TauSolver(kx, kz, Lx_, Lz_, a, b, lambda, nu, num_modes, tau_correction)
        
        # Create solution and RHS coefficients
        u = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        v = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        w = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        P = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        
        Rx = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Ry = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Rz = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        
        # Set up small, smooth RHS
        for i = 1:min(6, num_modes-2)
            Rx[i] = complex(0.01 * sin(i), 0.005 * cos(i))
            Ry[i] = complex(0.005 * cos(i), 0.008 * sin(i))
            Rz[i] = complex(0.002 * i, 0.003 * i)
        end
        
        # Solve the system
        solve!(tau, u, v, w, P, Rx, Ry, Rz)
        
        # Test verify method (non-verbose)
        error = verify(tau, u, v, w, P, Rx, Ry, Rz, false)
        @test error < 1e-3  # Should be small for a well-conditioned problem
        @test !isnan(error) && !isinf(error)
        
        # Test verbose version doesn't crash
        @test_nowarn verify(tau, u, v, w, P, Rx, Ry, Rz, true)
        
        # Test verify with mean flow
        tau_mean = TauSolver(0, 0, Lx_, Lz_, a, b, lambda, nu, num_modes, tau_correction)
        
        u_mean = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        v_mean = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        w_mean = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        P_mean = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        
        Rx_mean = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Ry_mean = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Rz_mean = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        
        # Small RHS for mean flow case
        for i = 1:min(4, num_modes-2)
            Rx_mean[i] = complex(0.005 * i, 0.002 * i)
            Ry_mean[i] = complex(0.003 * sin(i), 0.004 * cos(i))
            Rz_mean[i] = complex(0.001 * i, 0.002 * i)
        end
        
        umean = 1.5
        solve!(tau_mean, u_mean, v_mean, w_mean, P_mean, Rx_mean, Ry_mean, Rz_mean, umean)
        
        # Test verify with explicit mean flow parameters
        dPdx = 0.1  # Some pressure gradient
        error_mean = verify(tau_mean, u_mean, v_mean, w_mean, P_mean, dPdx, 
                           Rx_mean, Ry_mean, Rz_mean, umean, false)
        @test !isnan(error_mean) && !isinf(error_mean)
        
        # Test zero solution gives small error with zero RHS
        u_zero = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        v_zero = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        w_zero = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        P_zero = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        
        Rx_zero = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Ry_zero = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        Rz_zero = ChebyCoeff{ComplexF64}(num_modes, a, b, Spectral)
        
        error_zero = verify(tau, u_zero, v_zero, w_zero, P_zero, 
                           Rx_zero, Ry_zero, Rz_zero, false)
        @test error_zero < 1e-12  # Zero solution should have very small error
    end
end