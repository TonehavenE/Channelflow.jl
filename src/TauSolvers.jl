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
    a::Float64 # a domain limit
    b::Float64 # b domain limit
    lambda::Float64 # lambda in the equation
    nu::Float64 # nu in the equation
    tau_correction::Bool # whether or not to eliminate tau errors
    pressure_helmholtz::HelmholtzProblem
    velocity_helmholtz::HelmholtzProblem

    P_0::ChebyCoeff{Float64,Vector{Float64}}
    v_0::ChebyCoeff{Float64,Vector{Float64}}
    P_plus::ChebyCoeff{Float64,Vector{Float64}}
    v_plus::ChebyCoeff{Float64,Vector{Float64}}
    P_minus::ChebyCoeff{Float64,Vector{Float64}}
    v_minus::ChebyCoeff{Float64,Vector{Float64}}

    # Convenience variables
    two_pi_kxLx::Float64 # 2 pi kx/Lx
    two_pi_kzLz::Float64 # 2 pi kz/Lz
    kappa2::Float64 # 4 pi^2 [(kx/Lx)^2 + (kz/Lz)^2]
    i00::Float64
    i01::Float64
    i10::Float64
    i11::Float64

    sigma0_N1::Float64
    sigma0_N::Float64
    work_r::ChebyCoeff{Float64,Vector{Float64}}
    work_r2::ChebyCoeff{Float64,Vector{Float64}}
    work_c::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
    work_c_re::ChebyCoeff{Float64}
    work_c_im::ChebyCoeff{Float64}

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
        derivative!(P_minus, dP_dy)
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


        work_c = ChebyCoeff{ComplexF64}(N, a, b, Spectral)
        work_c_re = realview(work_c)
        work_c_im = imagview(work_c)

        this = new(
            num_modes,
            kx,
            kz,
            Float64(a),
            Float64(b),
            Float64(lambda),
            Float64(nu),
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
            0,
            ChebyCoeff(N, a, b, Spectral),
            ChebyCoeff(N, a, b, Spectral),
            work_c,
            work_c_re,
            work_c_im,
        )

        influence_correction!(this, this.P_0, this.v_0)
        dv_dyy = derivative2(this.v_0)

        this.sigma0_N = lambda * this.v_0[N] + dP0_dy[N] - nu * dv_dyy[N]
        this.sigma0_N1 = lambda * this.v_0[N-1] + dP0_dy[N-1] - nu * dv_dyy[N-1]
        this
    end
end

"""
    TauSolver()

Empty constructor for TauSolver.
"""
function TauSolver()
    a = -1.0
    b = 1.0
    TauSolver(
        0,
        0,
        2pi,
        2pi,
        a,
        b,
        1.0,
        1.0,
        5,
        false
    )
end

function influence_correction!(tau::TauSolver, P::ChebyCoeff{Float64,AP}, v::ChebyCoeff{Float64,AV}) where {AP<:AbstractArray{Float64},AV<:AbstractArray{Float64}}
    return influence_correction!(tau, P, v, tau.work_r2)
end

function influence_correction!(tau::TauSolver, P::ChebyCoeff{Float64,AP}, v::ChebyCoeff{Float64,AV}, tmp::ChebyCoeff{Float64,AT}) where {AP<:AbstractArray{Float64},AV<:AbstractArray{Float64},AT<:AbstractArray{Float64}}
    derivative!(v, tmp)
    dvp_dy_plus = eval_b(tmp)
    dvp_dy_minus = eval_a(tmp)
    delta_plus = -tau.i00 * dvp_dy_plus - tau.i01 * dvp_dy_minus
    delta_minus = -tau.i10 * dvp_dy_plus - tau.i11 * dvp_dy_minus

    Pd = P.data
    vd = v.data
    Ppd = tau.P_plus.data
    Pmd = tau.P_minus.data
    vpd = tau.v_plus.data
    vmd = tau.v_minus.data
    @inbounds for i = 1:tau.num_modes
        Pd[i] += delta_plus * Ppd[i] + delta_minus * Pmd[i]
        vd[i] += delta_plus * vpd[i] + delta_minus * vmd[i]
    end
end
function solve_P_and_v!(
    tau::TauSolver,
    P::ChebyCoeff{Float64,AP},
    v::ChebyCoeff{Float64,AV},
    r::ChebyCoeff{Float64,AR},
    Ry::ChebyCoeff{Float64,AY},
) where {AP<:AbstractArray{Float64},AV<:AbstractArray{Float64},AR<:AbstractArray{Float64},AY<:AbstractArray{Float64}}

    # Solve pressure Helmholtz: P'' - kappa^2 P = r, with Dirichlet BCs
    solve!(tau.pressure_helmholtz, P, r, 0.0, 0.0)

    # Degenerate case: kx == 0 && kz == 0
    if tau.kx == 0 && tau.kz == 0
        fill!(v.data, 0.0)
        return
    end

    # General case
    tmp = tau.work_r
    tmp2 = tau.work_r2
    derivative!(P, tmp)
    tmpd = tmp.data
    Ryd = Ry.data
    @inbounds for i = 1:tau.num_modes
        tmpd[i] -= Ryd[i]
    end

    # Solve velocity Helmholtz: nu*v'' - lambda*v = tmp, with Dirichlet BCs
    solve!(tau.velocity_helmholtz, v, tmp, 0.0, 0.0)

    influence_correction!(tau, P, v, tmp2)

    if !tau.tau_correction
        return
    end

    # Tau correction code follows
    derivative2!(v, tmp2, tmp)

    # sigma1_Nb and sigma1_Nb1 (Canuto & Hussaini notation)
    N = tau.num_modes
    λ = tau.lambda
    ν = tau.nu

    vd = v.data
    sigma1_N = λ * vd[N] - ν * tmpd[N] - Ryd[N]
    sigma1_N1 = λ * vd[N-1] - ν * tmpd[N-1] - Ryd[N-1]

    derivative!(P, tmp)

    sigma1_N += tmpd[N]
    sigma1_N1 += tmpd[N-1]

    # sigma0_Nb and sigma0_Nb1 are precomputed in tau
    sigma_N = sigma1_N / (1.0 - tau.sigma0_N)
    sigma_N1 = sigma1_N1 / (1.0 - tau.sigma0_N1)

    # Apply tau correction to P and v
    Pd = P.data
    P0d = tau.P_0.data
    v0d = tau.v_0.data
    @inbounds for i = 1:tau.num_modes
        if iseven(i - 1)
            Pd[i] += sigma_N1 * P0d[i]
            vd[i] += sigma_N * v0d[i]
        else
            Pd[i] += sigma_N * P0d[i]
            vd[i] += sigma_N1 * v0d[i]
        end
    end

    return sigma_N, sigma_N1
end

"""
    solve!(tau, u, v, w, P, Rx, Ry, Rz)

Solve the Tau equations for the given fields and return the solution.
"""
function solve!(
    tau::TauSolver,
    u::ChebyCoeff{ComplexF64,AU},
    v::ChebyCoeff{ComplexF64,AV},
    w::ChebyCoeff{ComplexF64,AW},
    P::ChebyCoeff{ComplexF64,AP},
    Rx::ChebyCoeff{ComplexF64,ARX},
    Ry::ChebyCoeff{ComplexF64,ARY},
    Rz::ChebyCoeff{ComplexF64,ARZ},
) where {AU<:AbstractArray{ComplexF64},AV<:AbstractArray{ComplexF64},AW<:AbstractArray{ComplexF64},AP<:AbstractArray{ComplexF64},ARX<:AbstractArray{ComplexF64},ARY<:AbstractArray{ComplexF64},ARZ<:AbstractArray{ComplexF64}}
    return solve!(
        tau,
        u,
        v,
        w,
        P,
        Rx,
        Ry,
        Rz,
        realview(u),
        imagview(u),
        realview(v),
        imagview(v),
        realview(w),
        imagview(w),
        realview(P),
        imagview(P),
        realview(Ry),
        imagview(Ry),
    )
end

function solve!(
    tau::TauSolver,
    u::ChebyCoeff{ComplexF64,AU},
    v::ChebyCoeff{ComplexF64,AV},
    w::ChebyCoeff{ComplexF64,AW},
    P::ChebyCoeff{ComplexF64,AP},
    Rx::ChebyCoeff{ComplexF64,ARX},
    Ry::ChebyCoeff{ComplexF64,ARY},
    Rz::ChebyCoeff{ComplexF64,ARZ},
    u_re::ChebyCoeff{Float64,AUR},
    u_im::ChebyCoeff{Float64,AUI},
    v_re::ChebyCoeff{Float64,AVR},
    v_im::ChebyCoeff{Float64,AVI},
    w_re::ChebyCoeff{Float64,AWR},
    w_im::ChebyCoeff{Float64,AWI},
    P_re::ChebyCoeff{Float64,APR},
    P_im::ChebyCoeff{Float64,API},
    Ry_re::ChebyCoeff{Float64,ARYR},
    Ry_im::ChebyCoeff{Float64,ARYI},
) where {AU<:AbstractArray{ComplexF64},AV<:AbstractArray{ComplexF64},AW<:AbstractArray{ComplexF64},AP<:AbstractArray{ComplexF64},ARX<:AbstractArray{ComplexF64},ARY<:AbstractArray{ComplexF64},ARZ<:AbstractArray{ComplexF64},AUR<:AbstractArray{Float64},AUI<:AbstractArray{Float64},AVR<:AbstractArray{Float64},AVI<:AbstractArray{Float64},AWR<:AbstractArray{Float64},AWI<:AbstractArray{Float64},APR<:AbstractArray{Float64},API<:AbstractArray{Float64},ARYR<:AbstractArray{Float64},ARYI<:AbstractArray{Float64}}
    N = tau.num_modes
    r_re = tau.work_c_re
    r_im = tau.work_c_im

    # Decouple: solve real
    rr = tau.work_r
    rrd = rr.data
    Rxd = Rx.data
    Rzd = Rz.data
    derivative!(Ry_re, rr)
    @inbounds for n = 1:N
        rrd[n] -= tau.two_pi_kxLx * imag(Rxd[n]) + tau.two_pi_kzLz * imag(Rzd[n])
    end
    solve_P_and_v!(tau, P_re, v_re, rr, Ry_re)

    # Solve imaginary
    derivative!(Ry_im, rr)
    @inbounds for n = 1:N
        rrd[n] += tau.two_pi_kxLx * real(Rxd[n]) + tau.two_pi_kzLz * real(Rzd[n])
    end
    solve_P_and_v!(tau, P_im, v_im, rr, Ry_im)

    # Again, solve real and imaginary parts of u and w eqns separately
    r = tau.work_c
    rd = r.data
    Pd = P.data
    @inbounds for n = 1:N
        rd[n] = tau.two_pi_kxLx * im * Pd[n] - Rxd[n]
    end

    solve!(tau.velocity_helmholtz, u_re, r_re, 0.0, 0.0)
    solve!(tau.velocity_helmholtz, u_im, r_im, 0.0, 0.0)

    @inbounds for n = 1:N
        rd[n] = tau.two_pi_kzLz * im * Pd[n] - Rzd[n]
    end
    solve!(tau.velocity_helmholtz, w_re, r_re, 0.0, 0.0)
    solve!(tau.velocity_helmholtz, w_im, r_im, 0.0, 0.0)

    return
end

"""
    solve!(tau, u, v, w, P, Rx, Ry, Rz, umean)

Solves the Tau equations for the given fields with a mean flow.
tau.kx and tau.kz must be zero for this method.
"""
function solve!(
    tau::TauSolver,
    u::ChebyCoeff{ComplexF64,AU},
    v::ChebyCoeff{ComplexF64,AV},
    w::ChebyCoeff{ComplexF64,AW},
    P::ChebyCoeff{ComplexF64,AP},
    Rx::ChebyCoeff{ComplexF64,ARX},
    Ry::ChebyCoeff{ComplexF64,ARY},
    Rz::ChebyCoeff{ComplexF64,ARZ},
    umean::Real,
) where {AU<:AbstractArray{ComplexF64},AV<:AbstractArray{ComplexF64},AW<:AbstractArray{ComplexF64},AP<:AbstractArray{ComplexF64},ARX<:AbstractArray{ComplexF64},ARY<:AbstractArray{ComplexF64},ARZ<:AbstractArray{ComplexF64}}
    @assert tau.kx == 0 && tau.kz == 0 "This method is only for kx = 0 and kz = 0"

    N = tau.num_modes
    Ry_re = realview(Ry)
    Ry_im = imagview(Ry)
    P_re = realview(P)
    P_im = imagview(P)
    v_re = realview(v)
    v_im = imagview(v)
    u_re = realview(u)
    u_im = imagview(u)
    w_re = realview(w)
    w_im = imagview(w)
    r_re = tau.work_c_re
    r_im = tau.work_c_im

    # Decouple: solve real
    rr = tau.work_r
    rrd = rr.data
    Rxd = Rx.data
    Rzd = Rz.data
    derivative!(Ry_re, rr)
    @inbounds for n = 1:N
        rrd[n] -= tau.two_pi_kxLx * imag(Rxd[n]) + tau.two_pi_kzLz * imag(Rzd[n])
    end
    solve_P_and_v!(tau, P_re, v_re, rr, Ry_re)

    # Solve imaginary
    derivative!(Ry_im, rr)
    @inbounds for n = 1:N
        rrd[n] += tau.two_pi_kxLx * real(Rxd[n]) + tau.two_pi_kzLz * real(Rzd[n])
    end
    solve_P_and_v!(tau, P_im, v_im, rr, Ry_im)

    # Again, solve real and imaginary parts of u and w eqns separately
    r = tau.work_c
    rd = r.data
    Pd = P.data
    @inbounds for n = 1:N
        rd[n] = tau.two_pi_kxLx * im * Pd[n] - Rxd[n]
    end

    solve!(tau.velocity_helmholtz, u_re, r_re, umean, 0.0, 0.0)
    solve!(tau.velocity_helmholtz, u_im, r_im, 0.0, 0.0)

    @inbounds for n = 1:N
        rd[n] = tau.two_pi_kzLz * Pd[n] - Rzd[n]
    end
    solve!(tau.velocity_helmholtz, w_re, r_re, 0.0, 0.0)
    solve!(tau.velocity_helmholtz, w_im, r_im, 0.0, 0.0)

    return
end

function tauNorm(u::ChebyCoeff{T}) where {T<:Number}
    return L2Norm(u)
end

function tauDist(u::ChebyCoeff{T}, v::ChebyCoeff{T}) where {T<:Number}
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
