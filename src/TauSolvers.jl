module TauSolvers

import ..HelmholtzSolver: solve!
using ..HelmholtzSolver
using ..ChebyCoeffs
using ..Metrics

export TauSolver, solve!, influence_correction!, verify

const MINIMUM_DISCRIMINANT = 1e-4

function n_func(n::Int, N::Int)
    k = n - 1
    Nb = N - 1
    # map back
    if k == 0
        return Nb - 1
    elseif k == Nb
        return 0
    elseif k % 2 == 0
        return 2 * (Nb - 1)
    else
        return 2 * Nb
    end
end

"""
Class for solving 7.3.18-7.3.20 of Canuto & Hussaini
- nu u''_jk(y) - lambda u_jk(y) - grad P_jk = -R_jk, 
- div u_jk = 0
- u_jk(+-1) = 0

where:
- u_jk(y) is the vector-valued jkth xz-Fourier coeff of u(x,y,z)
- P(y) is in R
- and the vector operators are interpreted according to Fourier transform conventions.
"""
mutable struct TauSolver
    num_modes::Int
    kx::Int # x wave number 
    kz::Int # z wave number
    a::Real # a domain limit
    b::Real # b domain limit
    lambda::Real # lambda in the equation
    nu::Real # nu in the equation
    tau_correction::Bool # whether or not to eliminate tau errors
    pressure_helmholtz::HelmholtzProblem
    velocity_helmholtz::HelmholtzProblem

    P_0::ChebyCoeff
    v_0::ChebyCoeff
    P_plus::ChebyCoeff
    v_plus::ChebyCoeff
    P_minus::ChebyCoeff
    v_minus::ChebyCoeff

    # Convenience variables
    two_pi_kxLx::Real # 2 pi kx/Lx
    two_pi_kzLz::Real # 2 pi kz/Lz
    kappa2::Real # 4 pi^2 [(kx/Lx)^2 + (kz/Lz)^2]
    i00::Real
    i01::Real
    i10::Real
    i11::Real

    sigma0_N1::Real
    sigma0_N::Real

    function TauSolver(kx::Int, kz::Int, Lx::Real, Lz::Real, a::Real, b::Real, lambda::Real, nu::Real, num_modes::Int, tau_correction::Bool)
        N = num_modes
        two_pi_kxLx = 2 * pi * kx / Lx
        two_pi_kzLz = 2 * pi * kz / Lz
        kappa2 = 4 * pi^2 * ((kx / Lx)^2 + (kz / Lz)^2)

        pressure_helmholtz = HelmholtzProblem(N, a, b, kappa2)
        velocity_helmholtz = HelmholtzProblem(N, a, b, lambda, nu)

        P_0 = ChebyCoeff(N, a, b, Spectral)
        v_0 = ChebyCoeff(N, a, b, Spectral)
        P_plus = ChebyCoeff(N, a, b, Spectral)
        v_plus = ChebyCoeff(N, a, b, Spectral)
        P_minus = ChebyCoeff(N, a, b, Spectral)
        v_minus = ChebyCoeff(N, a, b, Spectral)

        zero = ChebyCoeff(N, a, b, Spectral)

        solve!(pressure_helmholtz, P_plus, zero, 0.0, 1.0)
        dP_dy = derivative(P_plus)
        solve!(velocity_helmholtz, v_plus, dP_dy, 0.0, 0.0)

        solve!(pressure_helmholtz, P_minus, zero, 1.0, 0.0)
        dP_dy = derivative(P_minus)
        solve!(velocity_helmholtz, v_minus, dP_dy, 0.0, 0.0)

        dvplus_dy = derivative(v_plus)
        dvminus_dy = derivative(v_minus)

        A = eval_b(dvplus_dy)
        B = eval_b(dvminus_dy)
        C = eval_a(dvplus_dy)
        D = eval_a(dvminus_dy)
        discriminant = A * D - B * C

        if kx != 0 || kz != 0
            @assert (abs(discriminant) / max(abs(A * D), abs(B * C))) > MINIMUM_DISCRIMINANT
        end

        i00 = D / discriminant
        i01 = -B / discriminant
        i10 = -C / discriminant
        i11 = A / discriminant

        # solve the B0 problem for tau corrections in solve (P, v)
        p0_rhs = ChebyCoeff(N, a, b, Spectral)
        c = 2 / (b - a)
        for i = 1:N
            p0_rhs[i] = c * n_func(i, N)
        end

        solve!(pressure_helmholtz, P_0, p0_rhs, 0.0, 0.0)

        dP0_dy = derivative(P_0)

        solve!(velocity_helmholtz, v_0, dP0_dy, 0.0, 0.0)


        this = new(
            num_modes,
            kx,
            kz,
            a,
            b,
            lambda,
            nu,
            tau_correction,
            pressure_helmholtz,
            velocity_helmholtz,
            P_0,
            v_0,
            P_plus,
            v_plus,
            P_minus,
            v_minus,
            two_pi_kxLx,
            two_pi_kzLz,
            kappa2,
            i00,
            i01,
            i10,
            i11,
            0,
            0
        )

        influence_correction!(this, this.P_0, this.v_0)
        dv_dyy = derivative2(this.v_0)

        this.sigma0_N  = lambda * this.v_0[N]   + dP0_dy[N]   - nu * dv_dyy[N]
        this.sigma0_N1 = lambda * this.v_0[N-1] + dP0_dy[N-1] - nu * dv_dyy[N-1]
        this
    end
end

function influence_correction!(tau::TauSolver, P::ChebyCoeff, v::ChebyCoeff)
    tmp = derivative(v)
    dvp_dy_plus = eval_b(tmp)
    dvp_dy_minus = eval_a(tmp)
    delta_plus = -tau.i00 * dvp_dy_plus - tau.i01 * dvp_dy_minus
    delta_minus = -tau.i10 * dvp_dy_plus - tau.i11 * dvp_dy_minus

    for i = 1:tau.num_modes
        P[i] += delta_plus * tau.P_plus[i] + delta_minus * tau.P_minus[i]
        v[i] += delta_plus * tau.v_plus[i] + delta_minus * tau.v_minus[i]
    end
end
function solve_P_and_v!(
    tau::TauSolver,
    P::ChebyCoeff,
    v::ChebyCoeff,
    r::ChebyCoeff,
    Ry::ChebyCoeff,
)
    # Solve pressure Helmholtz: P'' - kappa^2 P = r, with Dirichlet BCs
    solve!(tau.pressure_helmholtz, P, r, 0.0, 0.0)

    # Degenerate case: kx == 0 && kz == 0
    if tau.kx == 0 && tau.kz == 0
        for i = 1:tau.num_modes
            v[i] = 0.0
        end
        return
    end

    # General case
    tmp = derivative(P)
    tmp -= Ry

    # Solve velocity Helmholtz: nu*v'' - lambda*v = tmp, with Dirichlet BCs
    solve!(tau.velocity_helmholtz, v, tmp, 0.0, 0.0)
    
    influence_correction!(tau, P, v)

    if !tau.tau_correction
        return
    end

    # Tau correction code follows
    vyy = derivative2(v)

    # sigma1_Nb and sigma1_Nb1 (Canuto & Hussaini notation)
    N = tau.num_modes
    λ = tau.lambda
    ν = tau.nu

    sigma1_N  = λ * v[N]   - ν * vyy[N]   - Ry[N]
    sigma1_N1 = λ * v[N-1] - ν * vyy[N-1] - Ry[N-1]

    tmp2 = derivative(P)
    sigma1_N  += tmp2[N]
    sigma1_N1 += tmp2[N-1]

    # sigma0_Nb and sigma0_Nb1 are precomputed in tau
    sigma_N  = sigma1_N  / (1.0 - tau.sigma0_N)
    sigma_N1 = sigma1_N1 / (1.0 - tau.sigma0_N1)

    # Apply tau correction to P and v
    for i = 1:tau.num_modes
        if iseven(i-1)
            P[i] += sigma_N1 * tau.P_0[i]
            v[i] += sigma_N  * tau.v_0[i]
        else
            P[i] += sigma_N  * tau.P_0[i]
            v[i] += sigma_N1 * tau.v_0[i]
        end
    end

    return sigma_N, sigma_N1
end

"""
    solve!(tau, u, v, w, P, Rx, Ry, Rz)

Solve the Tau equations for the given fields and return the solution.
"""
function solve!(tau::TauSolver, u::ChebyCoeff, v::ChebyCoeff, w::ChebyCoeff, P::ChebyCoeff, Rx::ChebyCoeff, Ry::ChebyCoeff, Rz::ChebyCoeff)
    N = tau.num_modes

    # Decouple: solve real
    rr = derivative(realview(Ry))
    for n = 1:N
        rr[n] -= tau.two_pi_kxLx*imag(Rx[n]) + tau.two_pi_kzLz*imag(Rz[n])
    end
    solve_P_and_v!(tau, realview(P), realview(v), rr, realview(Ry))

    # Solve imaginary
    rr = derivative(imagview(Ry))
    for n = 1:N
        rr[n] += tau.two_pi_kxLx*real(Rx[n]) + tau.two_pi_kzLz*real(Rz[n])
    end
    solve_P_and_v!(tau, imagview(P), imagview(v), rr, imagview(Ry))

    # Again, solve real and imaginary parts of u and w eqns separately
    r = ChebyCoeff{ComplexF64}(N, tau.a, tau.b, Spectral)
    for n = 1:N
        r[n] = tau.two_pi_kxLx * im * P[n] - Rx[n]
    end

    solve!(tau.velocity_helmholtz, realview(u), realview(r), 0.0, 0.0)
    solve!(tau.velocity_helmholtz, imagview(u), imagview(r), 0.0, 0.0)

    for n = 1:N
        r[n] = tau.two_pi_kzLz * im * P[n] - Rz[n]
    end
    solve!(tau.velocity_helmholtz, realview(w), realview(r), 0.0, 0.0)
    solve!(tau.velocity_helmholtz, imagview(w), imagview(r), 0.0, 0.0)

    return
end

"""
    solve!(tau, u, v, w, P, Rx, Ry, Rz, umean)

Solves the Tau equations for the given fields with a mean flow.
tau.kx and tau.kz must be zero for this method.
"""
function solve!(tau::TauSolver, u::ChebyCoeff, v::ChebyCoeff, w::ChebyCoeff, P::ChebyCoeff, Rx::ChebyCoeff, Ry::ChebyCoeff, Rz::ChebyCoeff, umean::Real)
    @assert tau.kx == 0 && tau.kz == 0 "This method is only for kx = 0 and kz = 0"

    N = tau.num_modes

    # Decouple: solve real
    rr = derivative(realview(Ry))
    for n = 1:N
        rr[n] -= tau.two_pi_kxLx*imag(Rx[n]) + tau.two_pi_kzLz*imag(Rz[n])
    end
    solve_P_and_v!(tau, realview(P), realview(v), rr, realview(Ry))

    # Solve imaginary
    rr = derivative(imagview(Ry))
    for n = 1:N
        rr[n] += tau.two_pi_kxLx*real(Rx[n]) + tau.two_pi_kzLz*real(Rz[n])
    end
    solve_P_and_v!(tau, imagview(P), imagview(v), rr, imagview(Ry))

    # Again, solve real and imaginary parts of u and w eqns separately
    r = ChebyCoeff{ComplexF64}(N, tau.a, tau.b, Spectral)
    for n = 1:N
        r[n] = tau.two_pi_kxLx * im * P[n] - Rx[n]
    end

    solve!(tau.velocity_helmholtz, realview(u), realview(r), umean, 0.0, 0.0)
    solve!(tau.velocity_helmholtz, imagview(u), imagview(r), 0.0, 0.0)

    for n = 1:N
        r[n] = tau.two_pi_kzLz * P[n] - Rz[n]
    end
    solve!(tau.velocity_helmholtz, realview(w), realview(r), 0.0, 0.0)
    solve!(tau.velocity_helmholtz, imagview(w), imagview(r), 0.0, 0.0)

    return
end

function tauNorm(u::ChebyCoeff{T}) where {T <: Number}
    tmp = ChebyCoeff{T}(num_modes(u) - 2, u)
    return L2Norm(u)
end

function tauDist(u::ChebyCoeff{T}, v::ChebyCoeff{T}) where {T <: Number}
    utmp = ChebyCoeff{T}(num_modes(u) - 2, u)
    vtmp = ChebyCoeff{T}(num_modes(v) - 2, v)
    return L2Dist(utmp, vtmp)
end

"""
    verify(tau, u, v, w, P, Rx, Ry, Rz, verbose=false)

Verify that the computed solution satisfies the Tau equations.
Returns the total verification error.
"""
function verify(tau::TauSolver, u::ChebyCoeff{ComplexF64}, v::ChebyCoeff{ComplexF64}, 
                w::ChebyCoeff{ComplexF64}, P::ChebyCoeff{ComplexF64}, 
                Rx::ChebyCoeff{ComplexF64}, Ry::ChebyCoeff{ComplexF64}, 
                Rz::ChebyCoeff{ComplexF64}, verbose::Bool=false)
    
    umean = real(mean_value(u))
    dPdx = 0.0
    return verify(tau, u, v, w, P, dPdx, Rx, Ry, Rz, umean, verbose)
end

"""
    verify(tau, u, v, w, P, dPdx, Rx, Ry, Rz, umean, verbose=false)

Verify that the computed solution satisfies the Tau equations with mean flow.
Returns the total verification error.
"""
function verify(tau::TauSolver, u::ChebyCoeff{ComplexF64}, v::ChebyCoeff{ComplexF64}, 
                w::ChebyCoeff{ComplexF64}, P::ChebyCoeff{ComplexF64}, dPdx::Real,
                Rx::ChebyCoeff{ComplexF64}, Ry::ChebyCoeff{ComplexF64}, 
                Rz::ChebyCoeff{ComplexF64}, umean::Real, verbose::Bool=false)
    
    # Verify nu u''(y) - lambda u(y) - grad P = -R
    #        div u = 0
    #        u(±1) = 0
    
    if verbose
        println("TauSolver.verify(u,v,w,P,dPdx,Rx,Ry,Rz,umean,verbose)")
        println(" kx kz == ", tau.kx, " ", tau.kz)
    end
    
    N = tau.num_modes
    lhs = ChebyCoeff{ComplexF64}(N, tau.a, tau.b, Spectral)
    tmp = ChebyCoeff{ComplexF64}(N, tau.a, tau.b, Spectral)
    error = 0.0
    terr = 0.0
    lerr = 0.0
    
    # Verify u equation: -nu u'' + lambda u + dP/dx == Rx
    lhs = ChebyCoeff(u.data, u.a, u.b, u.state)
    lhs *= tau.lambda
    
    u_second_deriv = derivative2(u)
    u_second_deriv *= tau.nu
    lhs -= u_second_deriv
    
    tmp = ChebyCoeff(P.data, P.a, P.b, P.state)
    tmp *= complex(0.0, tau.two_pi_kxLx)
    lhs += tmp
    
    # Add mean pressure gradient
    lhs[1] += dPdx
    
    terr = tauDist(lhs, Rx)
    lerr = L2Dist(lhs, Rx)
    error += lerr
    
    if verbose
        println("L2Norm(Rx) == ", L2Norm(Rx))
        println("tauDist(nu u'' - lambda u - dP/dx, -Rx) == ", terr)
        println(" L2Dist(nu u'' - lambda u - dP/dx, -Rx) == ", lerr)
    end
    
    # Verify v equation: nu v'' - lambda v - dP/dy == -Ry
    lhs = ChebyCoeff(v.data, v.a, v.b, v.state)
    lhs *= tau.lambda
    
    v_second_deriv = derivative2(v)
    v_second_deriv *= tau.nu
    lhs -= v_second_deriv
    
    P_grad_y = derivative(P)
    lhs += P_grad_y
    
    terr = tauDist(lhs, Ry)
    lerr = L2Dist(lhs, Ry)
    error += lerr
    
    if verbose
        println("L2Norm(Ry) == ", L2Norm(Ry))
        println("tauDist(nu v'' - lambda v - dP/dy, -Ry) == ", terr)
        println(" L2Dist(nu v'' - lambda v - dP/dy, -Ry) == ", lerr)
    end
    
    # Verify w equation: nu w'' - lambda w - dP/dz == -Rz
    lhs = ChebyCoeff(w.data, w.a, w.b, w.state)
    lhs *= tau.lambda
    
    w_second_deriv = derivative2(w)
    w_second_deriv *= tau.nu
    lhs -= w_second_deriv
    
    tmp = ChebyCoeff(P.data, P.a, P.b, P.state)
    tmp *= complex(0.0, tau.two_pi_kzLz)
    lhs += tmp
    
    terr = tauDist(lhs, Rz)
    lerr = L2Dist(lhs, Rz)
    error += lerr
    
    if verbose
        println("L2Norm(Rz) == ", L2Norm(Rz))
        println("tauDist(nu w'' - lambda w - dP/dz, -Rz) == ", terr)
        println(" L2Dist(nu w'' - lambda w - dP/dz, -Rz) == ", lerr)
    end
    
    # Verify pressure equation: P'' - kappa^2 P = div R
    P_second_deriv = derivative2(P)
    lhs = ChebyCoeff(P_second_deriv.data, P_second_deriv.a, P_second_deriv.b, P_second_deriv.state)
    
    tmp = ChebyCoeff(P.data, P.a, P.b, P.state)
    tmp *= -tau.kappa2
    lhs += tmp
    
    # Compute div R
    r = ChebyCoeff{ComplexF64}(N, tau.a, tau.b, Spectral)
    
    # Re and Im parts decouple
    Ry_grad = derivative(realview(Ry))
    r_re = ChebyCoeff(Ry_grad.data, Ry_grad.a, Ry_grad.b, Ry_grad.state)
    for n = 1:N
        r_re[n] -= tau.two_pi_kxLx * imag(Rx[n]) + tau.two_pi_kzLz * imag(Rz[n])
    end
    
    Ry_grad_im = derivative(imagview(Ry))
    r_im = ChebyCoeff(Ry_grad_im.data, Ry_grad_im.a, Ry_grad_im.b, Ry_grad_im.state)
    for n = 1:N
        r_im[n] += tau.two_pi_kxLx * real(Rx[n]) + tau.two_pi_kzLz * real(Rz[n])
    end
    
    # Combine real and imaginary parts
    for n = 1:N
        r[n] = complex(r_re[n], r_im[n])
    end
    
    terr = tauDist(lhs, r)
    lerr = L2Dist(lhs, r)
    error += lerr
    
    if verbose
        println("L2Norm(div R) == ", L2Norm(r))
        println("tauDist(P'' - k^2 P, div R) == ", terr)
        println(" L2Dist(P'' - k^2 P, div R) == ", lerr)
    end
    
    # Verify divergence: div u = i*kx*u + dv/dy + i*kz*w = 0
    v_grad_y = derivative(v)
    tmp = ChebyCoeff(v_grad_y.data, v_grad_y.a, v_grad_y.b, v_grad_y.state)
    for n = 1:N
        tmp[n] += im * (tau.two_pi_kxLx * u[n] + tau.two_pi_kzLz * w[n])
    end
    
    terr = tauNorm(tmp)
    lerr = L2Norm(tmp)
    error += lerr
    
    if verbose
        println("tauNorm(div) == ", terr)
        println(" L2Norm(div) == ", lerr)
    end
    
    # Boundary conditions
    ua = eval_a(u)
    ub = eval_b(u)
    error += abs(ua) + abs(ub)
    if verbose
        println("u(a),u(b) == ", ua, " ", ub)
    end
    
    va = eval_a(v)
    vb = eval_b(v)
    error += abs(va) + abs(vb)
    if verbose
        println("v(a),v(b) == ", va, " ", vb)
    end
    
    vy = derivative(v)
    vya = eval_a(vy)
    vyb = eval_b(vy)
    error += abs(vya) + abs(vyb)
    if verbose
        println("v' at a,b == ", vya, " ", vyb)
    end
    
    wa = eval_a(w)
    wb = eval_b(w)
    error += abs(wa) + abs(wb)
    if verbose
        println("w(a),w(b) == ", wa, " ", wb)
    end
    
    mean_error = abs2(real(mean_value(u)) - umean)
    error += mean_error
    if verbose
        println("abs2(u.mean() - umean) == ", mean_error)
    else
        @assert mean_error < 1e-12 "Mean flow error too large: $mean_error"
    end
    
    if verbose
        println("total verification error == ", error)
        println("} TauSolver.verify(...)")
    end
    
    return error
end

end

