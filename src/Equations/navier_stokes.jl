using ..TauSolvers
using ..ChebyCoeffs
using ..FlowFields
using ..BasisFuncs
import Base.@kwdef

import ..TauSolvers: solve!

export NSE, nonlinear!

function profile(ff::FlowField, mx::Int, mz::Int, i::Int)
    ret = ChebyCoeff{ComplexF64}(ff.domain.Ny, ff.domain.a, ff.domain.b, y_state(ff))
    if xz_state(ff) == Spectral
        for ny = 1:ff.domain.Ny
            ret[ny] = cmplx(ff, mx, ny, mz, i)
        end
    else
        for ny = 1:ff.domain.Ny
            ret[ny] = ff[mx, ny, mz, i]
        end
    end

    return ret
end

function profile(ff::FlowField, mx::Int, mz::Int)
    ret = BasisFunc(num_dimensions(ff), ff.domain.Ny, mx_to_kx(ff, mx), mz_to_kz(ff, mz), ff.domain.Lx, ff.domain.Lz, ff.domain.a, ff.domain.b, y_state(ff))
    for i = 1:num_dimensions(ff), ny = 1:ff.domain.Ny
        ret[i, ny] = cmplx(ff, mx, ny, mz, i)
    end
    return ret
end

function get_Ubulk(ff::FlowField)
    ubulk = mean_value(profile(ff, 1, 1, 1))
    if abs(ubulk) < 1e-15
        ubulk = 0.0
    end
    return ubulk
end

function get_Wbulk(ff::FlowField)
    wbulk = mean_value(profile(ff, 1, 1, 3))
    if abs(wbulk) < 1e-15
        wbulk = 0.0
    end
    return wbulk
end

function dudy_a(ff::FlowField)
    @assert y_state(ff) == Spectral
    prof = profile(ff, 1, 1)
    dudy = derivative(realview(get_u(prof)))
    return eval_a(dudy)
end

function dudy_b(ff::FlowField)
    @assert y_state(ff) == Spectral
    prof = profile(ff, 1, 1)
    dudy = derivative(realview(get_u(prof)))
    return eval_b(dudy)
end

function dwdy_a(ff::FlowField)
    @assert y_state(ff) == Spectral
    prof = profile(ff, 1, 1)
    dwdy = derivative(realview(get_w(prof)))
    return eval_a(dwdy)
end
function dwdy_b(ff::FlowField)
    @assert y_state(ff) == Spectral
    prof = profile(ff, 1, 1)
    dwdy = derivative(realview(get_w(prof)))
    return eval_b(dwdy)
end

function get_dPdx(ff::FlowField, nu::Real)
    nu * (dudy_b(ff) - dudy_a(ff)) / Ly(ff)
end

function get_dPdz(ff::FlowField, nu::Real)
    nu * (dwdy_b(ff) - dwdy_a(ff)) / Ly(ff)
end

@kwdef mutable struct SpatialParameters
    # Grid dimensions
    Nx::Int # num x gridpoints
    Ny::Int # num y gridpoints
    Nz::Int # num z gridpoints
    Mx::Int # num x modes
    Mz::Int # num z modes
    Nyd::Int # number of dealiased Chebyshev T(y) modes
    kxd_max::Int
    kzd_max::Int

    # Domain dimensions
    Lx::Real # x domain length
    Lz::Real # z domain length
    a::Real # y lower bound 
    b::Real # y upper bound
end

@kwdef mutable struct BaseFlowMembers
    # Pressure gradient constraints
    dPdx_Ref::Union{Real,Nothing} # enforced mean pressure gradient in x
    dPdx_Act::Union{Real,Nothing} # actual mean pressure gradient at previous timestep
    dPdz_Ref::Union{Real,Nothing} # enforced mean pressure gradient in z 
    dPdz_Act::Union{Real,Nothing} # actual mean pressure gradient at previous timestep

    # Bulk velocity constraints
    Ubulk_Ref::Union{Real,Nothing} # enforced total flow bulk velocity in x
    Ubulk_Act::Union{Real,Nothing} # actual total flow bulk velocity at previous timestep
    Ubulk_Base::Union{Real,Nothing} # Bulk velocity of Ubase
    Wbulk_Ref::Union{Real,Nothing} # enforced total flow bulk velocity in z
    Wbulk_Act::Union{Real,Nothing} # actual total flow bulk velocity at previous timestep
    Wbulk_Base::Union{Real,Nothing} # Bulk velocity of Wbase

    # Base flow functions
    Ubase::Union{ChebyCoeff} # baseflow physical
    Ubase_yy::Union{ChebyCoeff,Nothing} = nothing # baseflow second derivative in y
    Wbase::Union{ChebyCoeff} # baseflow physical
    Wbase_yy::Union{ChebyCoeff,Nothing} = nothing # baseflow second derivative in y
end

@kwdef mutable struct TransientFields
    ff::FlowField
    uk::ChebyCoeff{ComplexF64}
    vk::ChebyCoeff{ComplexF64}
    wk::ChebyCoeff{ComplexF64}
    Pk::ChebyCoeff{ComplexF64}
    Pyk::ChebyCoeff{ComplexF64}
    Ruk::ChebyCoeff{ComplexF64}
    Rvk::ChebyCoeff{ComplexF64}
    Rwk::ChebyCoeff{ComplexF64}
end

function TransientFields(temp::FlowField, Nyd::Int, a::Real, b::Real)
    return TransientFields(
        ff=temp,
        uk=ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral),
        vk=ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral),
        wk=ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral),
        Pk=ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral),
        Pyk=ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral),
        Ruk=ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral),
        Rvk=ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral),
        Rwk=ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral)
    )
end

@kwdef mutable struct NSE <: Equation
    lambda_t::Vector{Real}
    tausolvers::Union{AbstractArray,Nothing}

    # Refactored components
    spatial::SpatialParameters
    baseflow::BaseFlowMembers
    tmp::TransientFields
end

#TODO finish implementing laminar_profile
function laminar_profile(nu::Real, constraint::MeanConstraint, dPdx::Real, Ubulk::Real, Vsuck::Real, a::Real, b::Real, ua::Real, ub::Real, Ny::Int)
    u = ChebyCoeff(Ny, a, b, Spectral)
    H = b - a
    mu = Vsuck * H / nu
    if abs(mu) == 0.0
        if constraint == BulkVelocity
            u[1] = 0.125 * (ub + ua) + 0.75 * Ubulk
            u[2] = 0.5 * (ub - ua)
            u[3] = 0.375 * (ub + ua) - 0.75 * Ubulk
        else
            dPdx *= ((b - a) / 2)^2
            u[1] = 0.5 * (ub + ua) - 0.25 * dPdx / nu
            u[2] = 0.5 * (ub - ua)
            u[3] = 0.25 * dPdx / nu
        end
    else
        set_state!(u, Physical)
        dPdx_HH_nu = dPdx * H^2 / nu
        y = chebypoints(Ny, a, b)
        if abs(mu) > 1e-01
            if constraint == BulkVelocity
                k = -1.0 / expm1(-mu) - 1 / mu
                dPdx_HH_nu = mu * (Ubulk - ua - (ub - ua) * k) / (0.5 - k)
            end
            # TODO implement this!
        end
    end
    return u
end


"""
    create_base_flow(flags, My, a, b)

Creates a BaseFlow (Ubase, Wbase) as determined by flags.baseflow.
Used in the constructor of NSE.
"""
function create_base_flow(flags::DNSFlags, My::Int, a::Real, b::Real)
    @assert My > 0 "My must be positive, got $My"
    if flags.baseflow == ZeroBase
        Ubase = ChebyCoeff(My, a, b, Spectral)
        Wbase = ChebyCoeff(My, a, b, Spectral)
    elseif flags.baseflow == LinearBase
        Ubase = ChebyCoeff(My, a, b, Spectral)
        Wbase = ChebyCoeff(My, a, b, Spectral)
        Ubase[1] = 1
    elseif flags.baseflow == ParabolicBase
        @assert My > 2 "My must be greater than 2 for parabolic base flow"
        Ubase = ChebyCoeff(My, a, b, Spectral)
        Wbase = ChebyCoeff(My, a, b, Spectral)
        Ubase[1] = 0.5
        Ubase[3] = -0.5
    elseif flags.baseflow == SuctionBase
        error("Error in create_base_flow: SuctionBase not implemented")
        Ubase = laminar_profile(flags.nu, PressureGradient, 0, flags.Ubulk, flags.Vsuck, a, b, -0.5, 0.5, My)
        Wbase = ChebyCoeff(My, a, b, Spectral)
    elseif flags.baseflow == LaminarBase
        error("Error in create_base_flow: LaminarBase not implemented")
        Ubase = laminar_profile(flags.nu, flags.constraint, flags.dPdx, flags.Ubulk, flags.Vsuck, a, b, flags.ulowerwall, flags.uupperwall, My)
        Wbase = laminar_profile(flags.nu, flags.constraint, flags.dPdz, flags.Ubulk, flags.Vsuck, a, b, flags.wlowerwall, flags.wupperwall, My)
    elseif flags.baseflow == ArbitraryBase
        error("Error in create_base_flow: Arbitrary base flow not implemented. Please provide (Ubase, Wbase) when constructing DNS.")
    else
        error("Error in create_base_flow: Unknown base flow type: $(flags.baseflow)")
    end
    return Ubase, Wbase
end

"""
    init_cf_constraint(Ubase, Wbase, u, flags)

Initializes channel flow constraints based on Ubase, Wbase, the velocity field u, and DNSFlags.
"""
function init_cf_constraint(Ubase::ChebyCoeff, Wbase::ChebyCoeff, u::FlowField, flags::DNSFlags)
    Ubase_y = derivative(Ubase)
    Ubase_yy = derivative(Ubase_y)
    Wbase_y = derivative(Wbase)
    Wbase_yy = derivative(Wbase_y)

    Ubulk_Base = mean_value(Ubase)
    Wbulk_Base = mean_value(Wbase)

    Ubulk_Act = Ubulk_Base + get_Ubulk(u)
    Wbulk_Act = Wbulk_Base + get_Wbulk(u)
    dPdx_Act = get_dPdx(u, flags.nu)
    dPdz_Act = get_dPdz(u, flags.nu)

    if length(Ubase.data) != 0
        utmp = FlowField(u)
        utmp += Ubase
        dPdx_Act = get_dPdx(utmp, flags.nu)
    end
    if length(Wbase.data) != 0
        wtmp = FlowField(u)
        wtmp += Wbase
        dPdz_Act = get_dPdz(wtmp, flags.nu)
    end

    if flags.constraint == BulkVelocity
        Ubulk_Ref = flags.Ubulk
        Wbulk_Ref = flags.Wbulk
        dPdx_Ref = nothing
        dPdz_Ref = nothing
    else
        Ubulk_Ref = nothing
        Wbulk_Ref = nothing
        dPdx_Act = flags.dPdx
        dPdx_Ref = flags.dPdx
        dPdz_Act = flags.dPdz
        dPdz_Ref = flags.dPdz
    end

    return BaseFlowMembers(
        dPdx_Ref=dPdx_Ref,
        dPdx_Act=dPdx_Act,
        dPdz_Ref=dPdz_Ref,
        dPdz_Act=dPdz_Act,
        Ubulk_Ref=Ubulk_Ref,
        Ubulk_Act=Ubulk_Act,
        Ubulk_Base=Ubulk_Base,
        Wbulk_Ref=Wbulk_Ref,
        Wbulk_Act=Wbulk_Act,
        Wbulk_Base=Wbulk_Base,
        Ubase=Ubase,
        Ubase_yy=Ubase_yy,
        Wbase=Wbase,
        Wbase_yy=Wbase_yy
    )
end

function NSE(fields::Vector{FlowField{T}}, flags::DNSFlags) where {T<:Number}
    u = fields[1]
    Nyd = dealias_y(flags) ? 2 * (num_y_modes(u) - 1) / 3 + 1 : num_y_modes(u)
    kxd_max = dealias_xz(flags) ? div(u.domain.Nx, 3) - 1 : kx_max(u)
    kzd_max = dealias_xz(flags) ? div(u.domain.Nz, 3) - 1 : kz_max(u)

    Ubase, Wbase = create_base_flow(flags, u.domain.Ny, u.domain.a, u.domain.b)

    # Create spatial parameters struct
    spatial = SpatialParameters(
        Nx=u.domain.Nx,
        Ny=u.domain.Ny,
        Nz=u.domain.Nz,
        Mx=u.domain.Mx,
        Mz=u.domain.Mz,
        Nyd=Nyd,
        kxd_max=kxd_max,
        kzd_max=kzd_max,
        Lx=u.domain.Lx,
        Lz=u.domain.Lz,
        a=u.domain.a,
        b=u.domain.b
    )

    # Create base flow members struct
    baseflow = init_cf_constraint(Ubase, Wbase, u, flags)

    if flags.nonlinearity in [Alternating, Alternating_, Convection, LinearAboutProfile, Divergence, SkewSymmetric]
        tmp = FlowField(u.domain.Nx, u.domain.Ny, u.domain.Nz, 9, u.domain.Lx, u.domain.Lz, u.domain.a, u.domain.b)
    else
        tmp = FlowField(u.domain.Nx, u.domain.Ny, u.domain.Nz, 3, u.domain.Lx, u.domain.Lz, u.domain.a, u.domain.b)
    end

    transients = TransientFields(tmp, Nyd, u.domain.a, u.domain.b)

    return NSE(
        lambda_t=[0.0],
        tausolvers=nothing,
        spatial=spatial,
        baseflow=baseflow,
        tmp=transients
    )
end

function reset_lambda!(eqn::NSE, lambda_t::Vector{T}, flags::DNSFlags) where {T<:Real}
    eqn.lambda_t = lambda_t

    c = 4.0 * pi^2 * flags.nu

    # Create the fully configured 3D array of TauSolver objects in one step.
    eqn.tausolvers = [
        begin
            # These calculations are done for each element of the new array
            kx = mx_to_kx(eqn.tmp.ff, mx)
            kz = mz_to_kz(eqn.tmp.ff, mz)

            # Check the condition for this element
            if (kx != kx_max(eqn.tmp.ff) || kz != kz_max(eqn.tmp.ff)) && (!dealias_xz(flags) || !is_aliased(eqn.tmp.ff, kx, kz))
                lambda = lambda_t[j] + c * ((kx / eqn.spatial.Lx)^2 + (kz / eqn.spatial.Lz)^2)

                # Construct the configured TauSolver for this grid point
                TauSolver(kx, kz, eqn.spatial.Lx, eqn.spatial.Lz, eqn.spatial.a, eqn.spatial.b, lambda, flags.nu, eqn.spatial.Nyd, flags.taucorrection)
            else
                # Provide a default-constructed TauSolver if the condition is not met
                TauSolver()
            end
        end
        # Define the iteration ranges for the 3D array
        for j in 1:length(lambda_t), mx in 1:eqn.spatial.Mx, mz in 1:eqn.spatial.Mz
    ]
end

function create_RHS(eqn::NSE, fields::Vector{FlowField{T}}) where {T<:Number}
    return [FlowField(fields[1])]
end

function navierstokes_nonlinear!(u::FlowField, Ubase::ChebyCoeff, Wbase::ChebyCoeff, f::FlowField, tmp::FlowField, flags::DNSFlags)
    finalstate = Spectral
    @assert xz_state(u) == Spectral "xz_state(u) should be Spectral in navierstokes_nonlinear"
    @assert y_state(u) == Spectral "y_state(u) should be Spectral in navierstokes_nonlinear"
    @assert Ubase.state == Spectral "Ubase state should be Spectral in navierstokes_nonlinear"
    @assert Wbase.state == Spectral "Wbase state should be Spectral in navierstokes_nonlinear"

    if flags.rotation != 0.0
        finalstate = Physical
    end

    if flags.nonlinearity == LinearAboutProfile
        linearized_nonlinear!(u, Ubase, Wbase, f, finalstate)
        make_spectral!(u)
        return
    end

    # all other flags start like this
    u += Ubase

    if flags.nonlinearity == Rotational
        rotational_nonlinear!(u, f, tmp, finalstate)
    elseif flags.nonlinearity == Convection
        convection_nonlinear!(u, f, tmp, finalstate)
    elseif flags.nonlinearity == SkewSymmetric
        skew_symmetric_nonlinear!(u, f, tmp, finalstate)
    elseif flags.nonlinearity == Divergence
        divergence_nonlinear!(u, f, tmp, finalstate)
    elseif flags.nonlinearity == Alternating
        divergence_nonlinear(u, f, tmp, finalstate)
        flags.nonlinearity = Alternating_ # switch to Alternating_ for next step
    elseif flags.nonlinearity == Alternating_
        convection_nonlinear!(u, f, tmp, finalstate)
        flags.nonlinearity = Alternating # switch to Alternating for next step
    else
        error("Unknown nonlinearity method: $(flags.nonlinearity)")
    end
    # add rotation term
    if flags.rotation != 0.0
        make_physical!(u)
        for nx = 1:u.domain.Nx, ny = 1:u.domain.Ny, nz = 1:u.domain.Nz
            f[nx, ny, nz, 1] -= (flags.rotation) * u[nx, ny, nz, 2]
            f[nx, ny, nz, 2] += (flags.rotation) * u[nx, ny, nz, 1]
        end
        make_spectral!(u)
        make_spectral!(f)
    end

    u -= Ubase
    make_spectral!(u)
end

"""
    nonlinear!(eqn, infields, outfields)

Calculates the nonlinear terms of the Navier Stokes equations.
"""
function nonlinear!(eqn::NSE, infields::Vector{<:FlowField}, outfields::Vector{<:FlowField}, flags::DNSFlags)
    navierstokes_nonlinear!(infields[1], eqn.baseflow.Ubase, eqn.baseflow.Wbase, outfields[1], eqn.tmp.ff, flags)
    if dealias_xz(flags)
        zero_padded_modes!(outfields[1])
    end
end

function rotational_nonlinear!(u::FlowField, f::FlowField, tmp::FlowField, finalstate::FieldState)
    @assert num_dimensions(u) == 3 "FlowField must have 3 dimensions for rotational nonlinearity"
    u_xz_state = xz_state(u)
    u_y_state = y_state(u)
    vort = tmp

    if !geom_congruent(u, f) || num_dimensions(f) != 3
        resize!(f, u.domain.Nx, u.domain.Ny, u.domain.Nz, 3, u.domain.Lx, u.domain.Lz, u.domain.a, u.domain.b)
    end
    make_state!(f, Physical, Physical)

    if !geom_congruent(u, vort) || num_dimensions(vort) != 3
        resize!(vort, u.domain.Nx, u.domain.Ny, u.domain.Nz, 3, u.domain.Lx, u.domain.Lz, u.domain.a, u.domain.b)
    end
    make_spectral!(u)
    curl!(u, vort)

    make_physical!(u)
    make_physical!(vort)
    cross!(vort, u, f, finalstate)

    if finalstate == Spectral
        make_spectral!(f)
    end

    make_state!(u, u_xz_state, u_y_state)

    return
end

function linear!(eqn::NSE, infields::Vector{<:FlowField}, outfields::Vector{<:FlowField}, flags::DNSFlags)
    @assert length(infields) == length(outfields) + 1 "Dimension mismatch. There should be no pressure field in outfields."

    # Use correct field indices (assuming 1-based indexing for Julia vectors)
    u_field = infields[1]  # velocity field
    p_field = infields[2]  # pressure field

    kxmax = kx_max(u_field)
    kzmax = kz_max(u_field)
    Lx_ = u_field.domain.Lx
    Lz_ = u_field.domain.Lz
    Mx = u_field.domain.Mx
    Mz = u_field.domain.Mz

    for mx = 1:Mx
        kx = mx_to_kx(u_field, mx)  # FIXED: correct function call

        for mz = 1:Mz
            kz = mz_to_kz(u_field, mz)  # FIXED: correct function call

            # FIXED: Skip aliased modes, but continue to next iteration
            if (kx == kxmax || kz == kzmax) || (dealias_xz(flags) && is_aliased(u_field, kx, kz))
                continue  # Skip this mode, continue with next mz
            end

            # Extract Fourier modes
            for ny = 1:eqn.spatial.Nyd
                eqn.tmp.uk[ny] = flags.nu * cmplx(u_field, mx, ny, mz, 1)  # u component
                eqn.tmp.vk[ny] = flags.nu * cmplx(u_field, mx, ny, mz, 2)  # v component  
                eqn.tmp.wk[ny] = flags.nu * cmplx(u_field, mx, ny, mz, 3)  # w component
                eqn.tmp.Pk[ny] = cmplx(p_field, mx, ny, mz, 1)             # pressure
            end

            # Compute second derivatives
            eqn.tmp.Ruk = derivative2(eqn.tmp.uk)
            eqn.tmp.Rvk = derivative2(eqn.tmp.vk)
            eqn.tmp.Rwk = derivative2(eqn.tmp.wk)

            # Compute pressure gradient in y
            eqn.tmp.Pyk = derivative(eqn.tmp.Pk)

            # Compute linear terms
            kappa2 = 4 * pi^2 * ((kx / Lx_)^2 + (kz / Lz_)^2)
            Dx_ = Dx(u_field, mx)
            Dz_ = Dz(u_field, mz)

            for ny = 1:eqn.spatial.Nyd
                # FIXED: Use consistent accessor functions
                set_cmplx!(outfields[1], eqn.tmp.Ruk[ny] - kappa2 * eqn.tmp.uk[ny] - Dx_ * eqn.tmp.Pk[ny], mx, ny, mz, 1)
                set_cmplx!(outfields[1], eqn.tmp.Rvk[ny] - kappa2 * eqn.tmp.vk[ny] - eqn.tmp.Pyk[ny], mx, ny, mz, 2)
                set_cmplx!(outfields[1], eqn.tmp.Rwk[ny] - kappa2 * eqn.tmp.wk[ny] - Dz_ * eqn.tmp.Pk[ny], mx, ny, mz, 3)
            end

            # Add constant terms for kx=0, kz=0 mode
            if kx == 0 && kz == 0
                if length(eqn.baseflow.Ubase_yy.data) > 0
                    for ny = 1:eqn.spatial.Ny
                        current_val = cmplx(outfields[1], mx, ny, mz, 1)
                        set_cmplx!(outfields[1], current_val + Complex(flags.nu * eqn.baseflow.Ubase_yy[ny], 0.0), mx, ny, mz, 1)
                    end
                end
                if length(eqn.baseflow.Wbase_yy.data) > 0
                    for ny = 1:eqn.spatial.Ny
                        current_val = cmplx(outfields[1], mx, ny, mz, 3)
                        set_cmplx!(outfields[1], current_val + Complex(flags.nu * eqn.baseflow.Wbase_yy[ny], 0.0), mx, ny, mz, 3)
                    end
                end

                if flags.constraint == PressureGradient
                    # Apply reference pressure gradient
                    current_u = cmplx(outfields[1], mx, 1, mz, 1)
                    current_w = cmplx(outfields[1], mx, 1, mz, 3)
                    set_cmplx!(outfields[1], current_u - Complex(eqn.baseflow.dPdx_Ref, 0.0), mx, 1, mz, 1)
                    set_cmplx!(outfields[1], current_w - Complex(eqn.baseflow.dPdz_Ref, 0.0), mx, 1, mz, 3)
                else
                    # Bulk velocity constraint - compute actual pressure gradient
                    Ly = eqn.spatial.b - eqn.spatial.a
                    eqn.tmp.Ruk = derivative(eqn.tmp.uk)
                    eqn.tmp.Rwk = derivative(eqn.tmp.wk)
                    dPdxAct = real(eval_b(eqn.tmp.Ruk) - eval_a(eqn.tmp.Ruk)) / Ly
                    dPdzAct = real(eval_b(eqn.tmp.Rwk) - eval_a(eqn.tmp.Rwk)) / Ly

                    if length(eqn.baseflow.Ubase.data) != 0
                        Ubasey = derivative(eqn.baseflow.Ubase)
                        dPdxAct += flags.nu * (eval_b(Ubasey) - eval_a(Ubasey)) / Ly
                    end
                    if length(eqn.baseflow.Wbase.data) != 0
                        Wbasey = derivative(eqn.baseflow.Wbase)
                        dPdzAct += flags.nu * (eval_b(Wbasey) - eval_a(Wbasey)) / Ly
                    end

                    current_u = cmplx(outfields[1], mx, 1, mz, 1)
                    current_w = cmplx(outfields[1], mx, 1, mz, 3)
                    set_cmplx!(outfields[1], current_u - Complex(dPdxAct, 0.0), mx, 1, mz, 1)
                    set_cmplx!(outfields[1], current_w - Complex(dPdzAct, 0.0), mx, 1, mz, 3)
                end
            end
        end  # mz loop
    end  # mx loop
end

#=
function linear!(eqn::NSE, infields::Vector{<:FlowField}, outfields::Vector{<:FlowField}, flags::DNSFlags)
    @assert length(infields) == length(outfields) + 1 "Dimension mismatch. There should be no pressure field in outfields. Outfields should be create with create_RHS."

    kxmax = kx_max(infields[1])
    kzmax = kz_max(infields[1])
    Lx_ = infields[1].domain.Lx
    Lz_ = infields[1].domain.Lz
    Mx = infields[1].domain.Mx
    Mz = infields[1].domain.Mz

    for mx = 1:Mx, mz = 1:Mz
        kx = kx_to_mx(infields[1], mx)
        kz = kz_to_mz(infields[1], mz)

        if (kx == kxmax || kz == kzmax) || (dealias_xz(flags) && is_aliased(infields[1], kx, kz))
            break
        end

        for ny = 1:eqn.spatial.Nyd
            eqn.tmp.uk[ny] = flags.nu * cmplx(infields[1], mx, ny, mz, 1)
            eqn.tmp.vk[ny] = flags.nu * cmplx(infields[1], mx, ny, mz, 2)
            eqn.tmp.wk[ny] = flags.nu * cmplx(infields[1], mx, ny, mz, 3)
            eqn.tmp.Pk[ny] = cmplx(infields[2], mx, ny, mz, 1)
        end

        eqn.tmp.Ruk = derivative2(eqn.tmp.uk)
        eqn.tmp.Rvk = derivative2(eqn.tmp.vk)
        eqn.tmp.Rwk = derivative2(eqn.tmp.wk)
        eqn.tmp.Pyk = derivative(eqn.tmp.Pk)

        kappa2 = 4 * pi^2 * ((kx / Lx_)^2 + (kz / Lz_)^2)
        Dx_ = Dx(infields[1], mx)
        Dz_ = Dz(infields[1], mz)

        for ny = 1:eqn.spatial.Nyd
            set_cmplx!(outfields[1], eqn.tmp.Ruk[ny] - kappa2 * eqn.tmp.uk[ny] - Dx_ * eqn.tmp.Pk[ny], mx, ny, mz, 1)
            set_cmplx!(outfields[1], eqn.tmp.Rvk[ny] - kappa2 * eqn.tmp.vk[ny] - eqn.tmp.Pyk[ny], mx, ny, mz, 2)
            set_cmplx!(outfields[1], eqn.tmp.Rwk[ny] - kappa2 * eqn.tmp.wk[ny] - Dz_ * eqn.tmp.Pk[ny], mx, ny, mz, 3)
        end

        # add const terms

        if kx == 0 && kz == 0
            if length(eqn.baseflow.Ubase_yy.data) > 0
                for ny = 1:eqn.spatial.Ny
                    outfields[1].spectral_data[mx, ny, mz, 1] += Complex(flags.nu * eqn.baseflow.Ubase_yy[ny], 0.0)
                end
            end
            if length(eqn.baseflow.Wbase_yy.data) > 0
                for ny = 1:eqn.spatial.Ny
                    outfields[1].spectral_data[mx, ny, mz, 3] += Complex(flags.nu * eqn.baseflow.Wbase_yy[ny], 0.0)
                end
            end

            if flags.constraint == PressureGradient
                outfields[1].spectral_data[mx, 1, mz, 1] -= Complex(eqn.baseflow.dPdx_Ref, 0.0)
                outfields[1].spectral_data[mx, 1, mz, 3] -= Complex(eqn.baseflow.dPdz_Ref, 0.0)
            else
                Ly = eqn.spatial.b - eqn.spatial.a
                eqn.tmp.Ruk = derivative(eqn.tmp.uk)
                eqn.tmp.Rwk = derivative(eqn.tmp.wk)
                dPdxAct = real(eval_b(eqn.tmp.Ruk) - eval_a(eqn.tmp.Ruk)) / Ly
                dPdzAct = real(eval_b(eqn.tmp.Rwk) - eval_a(eqn.tmp.Rwk)) / Ly
                Ubasey = derivative(eqn.baseflow.Ubase)
                Wbasey = derivative(eqn.baseflow.Wbase)
                if length(eqn.baseflow.Ubase.data) != 0
                    dPdxAct += flags.nu * (eval_b(Ubasey) - eval_a(Ubasey)) / Ly
                end
                if length(eqn.baseflow.Wbase.data) != 0
                    dPdzAct += flags.nu * (eval_b(Wbasey) - eval_a(Wbasey)) / Ly
                end
                outfields[1].spectral_data[mx, 1, mz, 1] -= Complex(dPdxAct, 0.0)
                outfields[1].spectral_data[mx, 1, mz, 3] -= Complex(dPdzAct, 0.0)
            end
        end
    end
end
=#

function solve!(eqn::NSE, outfields::Union{AbstractArray{FlowField},AbstractArray{FlowField{T}}}, rhs::AbstractArray{FlowField{T}}, s::Int, flags::DNSFlags) where {T<:Number}
    # Method takes a right hand side {u} and solves for output fields {u,press}
    @assert length(outfields) == length(rhs) + 1 "Make sure user provides correct RHS which can be created outside NSE with create_RHS()"

    kxmax = kx_max(outfields[1])
    kzmax = kz_max(outfields[1])

    # println("outfields[1] is:")
    # display(outfields[1])
    # println("rhs[1] is:")
    # display(rhs[1])

    # Update each Fourier mode with solution of the implicit problem
    # Since we're not using MPI, we loop over all modes directly
    for mx = 1:eqn.spatial.Mx
        kx = mx_to_kx(outfields[1], mx)

        for mz = 1:eqn.spatial.Mz
            kz = mz_to_kz(outfields[1], mz)

            # Skip last and aliased modes
            if (kx == kxmax || kz == kzmax) || (dealias_xz(flags) && is_aliased(outfields[1], kx, kz))
                break
            end

            # Construct ComplexChebyCoeff from RHS
            for ny = 1:eqn.spatial.Nyd
                eqn.tmp.Ruk[ny] = cmplx(rhs[1], mx, ny, mz, 1)
                eqn.tmp.Rvk[ny] = cmplx(rhs[1], mx, ny, mz, 2)
                eqn.tmp.Rwk[ny] = cmplx(rhs[1], mx, ny, mz, 3)
            end

            # Solve the tau equations
            if kx != 0 || kz != 0
                solve!(eqn.tausolvers[s, mx, mz], eqn.tmp.uk, eqn.tmp.vk, eqn.tmp.wk, eqn.tmp.Pk,
                    eqn.tmp.Ruk, eqn.tmp.Rvk, eqn.tmp.Rwk)
            else  # kx,kz == 0,0
                # LHS includes also the constant terms C which can be added to RHS
                if length(eqn.baseflow.Ubase_yy.data) > 0
                    for ny = 1:eqn.spatial.Ny
                        eqn.tmp.Ruk[ny] += flags.nu * eqn.baseflow.Ubase_yy[ny]  # Rx has addl'l term from Ubase
                    end
                end
                if length(eqn.baseflow.Wbase_yy.data) > 0
                    for ny = 1:eqn.spatial.Ny
                        eqn.tmp.Rwk[ny] += flags.nu * eqn.baseflow.Wbase_yy[ny]  # Rz has addl'l term from Wbase
                    end
                end

                if flags.constraint == PressureGradient
                    # pressure is supplied, put on RHS of tau eqn
                    eqn.tmp.Ruk[1] -= Complex(eqn.baseflow.dPdx_Ref, 0)
                    eqn.tmp.Rwk[1] -= Complex(eqn.baseflow.dPdz_Ref, 0)
                    solve!(eqn.tausolvers[s, mx, mz], eqn.tmp.uk, eqn.tmp.vk, eqn.tmp.wk, eqn.tmp.Pk,
                        eqn.tmp.Ruk, eqn.tmp.Rvk, eqn.tmp.Rwk)
                    # Bulk vel is free variable determined from soln of tau eqn 
                    # TODO: write method that computes UbulkAct everytime it is needed

                else  # const bulk velocity
                    # bulk velocity is supplied, use alternative tau solver

                    # Use tausolver with additional variable and constraint:
                    # free variable: dPdxAct at next time-step,
                    # constraint:    UbulkBase + mean(u) = UbulkRef.
                    solve!(eqn.tausolvers[s][mx][mz], eqn.tmp.uk, eqn.tmp.vk, eqn.tmp.wk, eqn.tmp.Pk,
                        eqn.baseflow.dPdx_Act, eqn.baseflow.dPdz_Act, eqn.tmp.Ruk, eqn.tmp.Rvk, eqn.tmp.Rwk,
                        eqn.baseflow.Ubulk_Ref - eqn.baseflow.Ubulk_Base,
                        eqn.baseflow.Wbulk_Ref - eqn.baseflow.Wbulk_Base)

                    @assert abs(eqn.baseflow.Ubulk_Ref - eqn.baseflow.Ubulk_Base - mean_value(eqn.tmp.uk.re)) < 1e-15 "UbulkRef != UbulkAct = UbulkBase + uk.re.mean()"
                    @assert abs(eqn.baseflow.Wbulk_Ref - eqn.baseflow.Wbulk_Base - mean_value(eqn.tmp.wk.re)) < 1e-15 "WbulkRef != WbulkAct = WbulkBase + wk.re.mean()"
                end
            end

            # Load solutions into u and p.
            # Because of FFTW complex symmetries
            # The 0,0 mode must be real.
            # For Nx even, the kxmax,0 mode must be real
            # For Nz even, the 0,kzmax mode must be real
            # For Nx,Nz even, the kxmax,kzmax mode must be real
            if ((kx == 0 && kz == 0) ||
                (outfields[1].domain.Nx % 2 == 0 && kx == kxmax && kz == 0) ||
                (outfields[1].domain.Nz % 2 == 0 && kz == kzmax && kx == 0) ||
                (outfields[1].domain.Nx % 2 == 0 && outfields[1].domain.Nz % 2 == 0 && kx == kxmax && kz == kzmax))

                for ny = 1:eqn.spatial.Nyd
                    set_cmplx!(outfields[1], Complex(real(eqn.tmp.uk[ny]), 0.0), mx, ny, mz, 1)
                    set_cmplx!(outfields[1], Complex(real(eqn.tmp.vk[ny]), 0.0), mx, ny, mz, 2)
                    set_cmplx!(outfields[1], Complex(real(eqn.tmp.wk[ny]), 0.0), mx, ny, mz, 3)
                    set_cmplx!(outfields[2], Complex(real(eqn.tmp.Pk[ny]), 0.0), mx, ny, mz, 1)
                end
            else
                # The normal case, for general kx,kz
                for ny = 1:eqn.spatial.Nyd
                    set_cmplx!(outfields[1], eqn.tmp.uk[ny], mx, ny, mz, 1)
                    set_cmplx!(outfields[1], eqn.tmp.vk[ny], mx, ny, mz, 2)
                    set_cmplx!(outfields[1], eqn.tmp.wk[ny], mx, ny, mz, 3)
                    set_cmplx!(outfields[2], eqn.tmp.Pk[ny], mx, ny, mz, 1)
                end
            end
        end
    end
end
