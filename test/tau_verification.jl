using Test
using Random
using ..TauSolvers
using ..HelmholtzSolver
using ..ChebyCoeffs
using ..Metrics

"""
    random_u_profile!(u, decay)

Create a random u(y) profile that satisfies u(±1) = 0 boundary conditions.
"""
function random_u_profile!(u::ChebyCoeff{ComplexF64}, decay::Real)
    N = length(u)
    
    # Set random coefficients with exponential decay
    mag = 1.0
    for n = 1:N
        u[n] = mag * complex(randn(), randn())
        if n > 2
            mag *= decay
        end
    end
    
    # Adjust u(y) so that u(±1) = 0
    u0 = (eval_b(u) + eval_a(u)) / 2.0
    u1 = (eval_b(u) - eval_a(u)) / 2.0
    u[1] -= u0  # Subtract constant term
    u[2] -= u1  # Subtract linear term
end

"""
    random_v_profile!(v, decay)

Create a random v(y) profile that satisfies v(±1) = v'(±1) = 0 boundary conditions.
"""
function random_v_profile!(v::ChebyCoeff{ComplexF64}, decay::Real)
    N = length(v)
    a_orig = v.a
    b_orig = v.b
    
    # Temporarily set bounds to [-1,1] for easier BC handling
    v_temp = ChebyCoeff{ComplexF64}(N, -1.0, 1.0, Spectral)
    
    # Set random coefficients with exponential decay
    mag = 1.0
    for n = 1:N
        v_temp[n] = mag * complex(randn(), randn())
        if n > 2
            mag *= decay
        end
    end
    
    # Compute derivative for BC enforcement
    vy_temp = derivative(v_temp)
    
    # Evaluate BCs
    A = eval_a(v_temp)      # v(-1)
    B = eval_b(v_temp)      # v(1)
    C = eval_a(vy_temp)     # v'(-1)
    D = eval_b(vy_temp)     # v'(1)
    
    # Solve for coefficients to subtract: s0*T0 + s1*T1 + s2*T2 + s3*T3
    # Based on the matrix inversion from the C++ code
    s0 = 0.5 * (A + B) + 0.125 * (C - D)
    s1 = 0.5625 * (B - A) - 0.0625 * (C + D)
    s2 = 0.125 * (D - C)
    s3 = 0.0625 * (A - B + C + D)
    
    # Subtract off the correction coefficients
    v_temp[1] -= s0
    v_temp[2] -= s1
    v_temp[3] -= s2
    v_temp[4] -= s3
    
    # Transform back to original bounds
    # This is a simplification - in practice you'd need proper bound transformation
    for n = 1:N
        v[n] = v_temp[n]
    end
    
    # Reset bounds
    v.a = a_orig
    v.b = b_orig
end

"""
    random_profile!(u, v, w, P, kx, kz, Lx, Lz, decay)

Generate a divergence-free random velocity field and random pressure field.
"""
function random_profile!(u::ChebyCoeff{ComplexF64}, v::ChebyCoeff{ComplexF64}, 
                        w::ChebyCoeff{ComplexF64}, P::ChebyCoeff{ComplexF64},
                        kx::Int, kz::Int, Lx::Real, Lz::Real, decay::Real)
    N = length(u)
    
    # Generate random pressure field
    mag = 1.0
    for n = 1:N
        P[n] = mag * complex(randn(), randn())
        if n > 2
            mag *= decay
        end
    end
    
    if kx == 0 && kz == 0
        # Special case: kx = kz = 0
        # Set w = 0, u is odd (even modes zero), v = 0
        fill!(w.data, 0.0)
        fill!(v.data, 0.0)
        
        random_u_profile!(u, decay)
        # Make u odd by zeroing even modes
        for n = 1:2:N
            u[n] = 0.0
        end
        
        return
    end
    
    # General case: start with random v profile
    random_v_profile!(v, decay)
    vy = derivative(v)
    
    if kx == 0
        # kx = 0, kz ≠ 0
        fill!(u.data, 0.0)
        # w = -Lz/(2πi*kz) * vy
        for n = 1:N
            w[n] = -Lz / (2π * im * kz) * vy[n]
        end
        
    elseif kz == 0
        # kz = 0, kx ≠ 0  
        fill!(w.data, 0.0)
        # u = -Lx/(2πi*kx) * vy
        for n = 1:N
            u[n] = -Lx / (2π * im * kx) * vy[n]
        end
        
    else
        # General case: kx ≠ 0, kz ≠ 0
        random_u_profile!(u, decay)
        
        # Calculate w from divergence-free condition: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
        # ux = (2πi*kx/Lx) * u
        # wz = (2πi*kz/Lz) * w
        # So: w = -Lz/(2πi*kz) * (ux + vy)
        
        for n = 1:N
            ux = 2π * im * (kx / Lx) * u[n]
            w[n] = -Lz / (2π * im * kz) * (ux + vy[n])
        end
    end
end

"""
    comprehensive_tau_solver_test(n_tests=100, verbose=false)

Run comprehensive randomized tests of the TauSolver, similar to the C++ version.
"""
function comprehensive_tau_solver_test(n_tests::Int=100, verbose::Bool=false)
    failures = 0
    epsilon = 1e-9
    tau_correct = true
    N = 49  # Chebyshev expansion length
    
    println("Comprehensive TauSolver Test: $n_tests random tests")
    if verbose
        println("=" ^ 60)
        println("comprehensive_tau_solver_test\n")
    end
    
    Random.seed!(12345)  # For reproducibility
    
    for test = 1:n_tests
        failure = false
        
        # Random parameters
        Lx = 2pi * (1 + rand())
        Lz = 2pi * (1 + rand()) 
        a = 1.0 + 0.1 * rand()
        b = a + (2 + rand())
        kx = rand(0:31)
        kz = rand(0:31)
        dt = 0.02
        nu = 1.0 / 1000.0
        lambda = 2.0 / dt + 4pi^2 * nu * ((kx / Lx)^2 + (kz / Lz)^2)
        decay = 0.5
        
        if verbose
            println("TauSolver test #$test")
            println("a b    == $a $b")
            println("Lx Lz  == $Lx $Lz") 
            println("kx kz  == $kx $kz")
            println("lambda == $lambda")
            println("nu     == $nu")
            println("decay  == $decay")
        end
        
        try
            # Create random divergence-free field
            P = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
            u = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
            v = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
            w = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
            
            random_profile!(u, v, w, P, kx, kz, Lx, Lz, decay)
            
            # Calculate R = nu*∇²u - λu - ∇P (the RHS we'll solve for)
            Rx = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
            Ry = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
            Rz = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
            
            # R = λu - nu*∇²u + ∇P
            copy!(Rx, u); Rx *= lambda
            copy!(Ry, v); Ry *= lambda  
            copy!(Rz, w); Rz *= lambda
            
            # Subtract nu * second derivatives
            u_second_deriv = derivative2(u)
            v_second_deriv = derivative2(v)
            w_second_deriv = derivative2(w)
            
            u_second_deriv *= -nu
            v_second_deriv *= -nu
            w_second_deriv *= -nu
            
            Rx += u_second_deriv
            Ry += v_second_deriv
            Rz += w_second_deriv
            
            # Add pressure gradients
            two_pi_kxLx = 2π * kx / Lx
            two_pi_kzLz = 2π * kz / Lz
            
            Px = ChebyCoeff(P.data, P.a, P.b, P.state)
            Px *= complex(0.0, two_pi_kxLx)
            Pz = ChebyCoeff(P.data, P.a, P.b, P.state)
            Pz *= complex(0.0, two_pi_kzLz)
            Py = derivative(P)
            
            Rx += Px
            Ry += Py
            Rz += Pz
            
            # Create TauSolver and solve
            if verbose
                println("Constructing TauSolver...")
            end
            
            tau_solver = TauSolver(kx, kz, Lx, Lz, a, b, lambda, nu, N, tau_correct)
            
            if verbose
                println("Verifying analytic solution...")
            end
            
            # Verify the original solution satisfies the equations
            if hasmethod(verify, (TauSolver, typeof(u), typeof(v), typeof(w), typeof(P), typeof(Rx), typeof(Ry), typeof(Rz), Bool))
                analytic_error = verify(tau_solver, u, v, w, P, Rx, Ry, Rz, verbose && test == 1)
                if verbose && test == 1
                    println("Analytic verification error: $analytic_error")
                end
            end
            
            # Solve numerically
            u_solve = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
            v_solve = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
            w_solve = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
            P_solve = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
            
            if verbose
                println("Solving numerically...")
            end
            
            solve!(tau_solver, u_solve, v_solve, w_solve, P_solve, Rx, Ry, Rz)
            
            if verbose
                println("Verifying numerical solution...")
            end
            
            # Verify numerical solution
            if hasmethod(verify, (TauSolver, typeof(u_solve), typeof(v_solve), typeof(w_solve), typeof(P_solve), typeof(Rx), typeof(Ry), typeof(Rz), Bool))
                numerical_error = verify(tau_solver, u_solve, v_solve, w_solve, P_solve, Rx, Ry, Rz, verbose && test == 1)
                
                if numerical_error > epsilon
                    failure = true
                    if verbose
                        println("FAILURE: Numerical verification error too large: $numerical_error")
                    end
                end
            end
            
            # Check solution accuracy
            if verbose || failure
                u_error = L2Dist(u, u_solve) / max(L2Norm(u), 1e-12)
                v_error = L2Dist(v, v_solve) / max(L2Norm(v), 1e-12)  
                w_error = L2Dist(w, w_solve) / max(L2Norm(w), 1e-12)
                P_error = L2Dist(P, P_solve) / max(L2Norm(P), 1e-12)
                
                println("Relative errors:")
                println("  u: $u_error")
                println("  v: $v_error") 
                println("  w: $w_error")
                println("  P: $P_error")
            end
            
            # Check divergence of both solutions
            vy = derivative(v)
            vy_solve = derivative(v_solve)
            
            # Original divergence
            div_orig = copy(vy)
            for n = 1:N
                div_orig[n] += im * (two_pi_kxLx * u[n] + two_pi_kzLz * w[n])
            end
            
            # Numerical solution divergence  
            div_solve = copy(vy_solve)
            for n = 1:N
                div_solve[n] += im * (two_pi_kxLx * u_solve[n] + two_pi_kzLz * w_solve[n])
            end
            
            div_orig_norm = L2Norm(div_orig)
            div_solve_norm = L2Norm(div_solve)
            
            if div_orig_norm > epsilon
                failure = true
                if verbose
                    println("FAILURE: Original field not divergence-free: $div_orig_norm")
                end
            end
            
            if verbose
                println("Divergence norms:")
                println("  Original: $div_orig_norm")
                println("  Numerical: $div_solve_norm")
                println("  Difference: $(L2Dist(div_orig, div_solve))")
            end
            
        catch e
            failure = true
            if verbose
                println("FAILURE: Exception occurred: $e")
            end
        end
        
        if failure
            failures += 1
            if !verbose
                print("F")
            end
        else
            if !verbose
                print(".")
            end
        end
        
        if verbose
            println("-" ^ 60)
        end
    end
    
    if !verbose
        println()
    end
    
    success_rate = (n_tests - failures) / n_tests
    println("Test Results:")
    println("  Tests run: $n_tests")
    println("  Failures: $failures") 
    println("  Success rate: $(round(success_rate * 100, digits=1))%")
    
    if failures == 0
        println("  *** ALL TESTS PASSED ***")
        return true
    else
        println("  *** SOME TESTS FAILED ***")
        return false
    end
end

# Run the comprehensive test
@testset "Comprehensive TauSolver Tests" begin
    @test comprehensive_tau_solver_test(20, false)  # Run 20 tests, non-verbose
end