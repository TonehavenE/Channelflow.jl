using ..TauSolvers
using ..DNSSettings
using ..ChebyCoeff
import Base.@kwdef

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
        temp,
        ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral),
        ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral),
        ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral),
        ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral),
        ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral),
        ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral),
        ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral)
    )
end

@kwdef mutable struct NSE <: Equation
    lambda_t::Vector{Real}
    tausolvers::Vector{Vector{Vector{TauSolver}}}

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
function init_cf_constraint(Ubase::ChebyCoeff, WBase::ChebyCoeff, u::FlowField, flags::DNSFlags)
    Ubase_y = derivative(Ubase)
    Ubase_yy = derivative(Ubase_y)
    Wbase_y = derivative(WBase)
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

function NSE(fields::Vector{FlowField}, flags::DNSFlags)
    u = fields[1]
    Nyd = dealias_y(flags) ? 2 * (num_y_modes(u) - 1) / 3 + 1 : num_y_modes(u)
    kxd_max = dealias_xz(flags) ? u.domain.Nx / 3 - 1 : kx_max(u)
    kzd_max = dealias_xz(flags) ? u.domain.Nz / 3 - 1 : kz_max(u)

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

    transients = TransientFields(tmp, Nyd, a, b)

    return NSE(
        lambda_t=[],
        tausolvers=[[[]]],
        spatial=spatial,
        baseflow=baseflow,
        tmp=transients
    )
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
function nonlinear!(eqn::NSE, infields::Vector{FlowField}, outfields::Vector{FlowField}, flags::DNSFlags)
    navierstokes_nonlinear!(infields[1], eqn.baseflow.Ubase, eqn.baseflow.Wbase, outfields[1], eqn.tmp.ff, flags)
    if dealias_xz(flags)
        zero_padding!(outfields[1])
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
    set_state!(f, Physical, Physical)

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

function linear!(eqn::NSE, infields::Vector{FlowField}, outfields::Vector{FlowField}, flags::DNSFlags)
    @assert num_dimensions(infields) == num_dimensions(outfields) + 1 "Dimension mismatch. There should be no pressure field in outfields. Outfields should be create with create_RHS."

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

        for ny = 1:eqn.Nyd
            eqn.tmp.uk[ny] = flags.nu * infields[1][mx, ny, mz, 1]
            eqn.tmp.vk[ny] = flags.nu * infields[1][mx, ny, mz, 2]
            eqn.tmp.wk[ny] = flags.nu * infields[1][mx, ny, mz, 3]
            eqn.tmp.Pk[ny] = infields[2][mx, ny, mz, 1]
        end

        eqn.tmp.Ruk = derivative2(eqn.tmp.uk)
        eqn.tmp.Rvk = derivative2(eqn.tmp.vk)
        eqn.tmp.Rwk = derivative2(eqn.tmp.wk)
        eqn.tmp.Pyk = derivative(eqn.tmp.Pk)

        kappa2 = 4 * pi^2 * ((kx / Lx_)^2 + (kz / Lz_)^2)
        Dx_ = Dx(infields[1], mx)
        Dz_ = Dz(infields[1], mz)

        for ny = 1:eqn.spatial.Nyd
            outfields[1][mx, ny, mz, 1] = eqn.tmp.Ruk[ny] - kappa2 * eqn.tmp.uk[ny] - Dx_ * eqn.tmp.Pk[ny]
            outfields[1][mx, ny, mz, 2] = eqn.tmp.Rvk[ny] - kappa2 * eqn.tmp.vk[ny] - eqn.tmp.Pyk[ny]
            outfields[1][mx, ny, mz, 3] = eqn.tmp.Rwk[ny] - kappa2 * eqn.tmp.wk[ny] - Dz_ * eqn.tmp.Pk[ny]
        end

        # add const terms

        if kx == 0 && kz == 0
            if length(eqn.baseflow.Ubase_yy.data) > 0
                for ny = 1:eqn.spatial.Ny
                    outfields[1][mx, ny, mz, 1] += Complex(flags.nu * eqn.baseflow.Ubase_yy[ny], 0.0)
                end
            end
            if length(eqn.baseflow.Wbase_yy.data) > 0
                for ny = 1:eqn.spatial.Ny
                    outfields[1][mx, ny, mz, 3] += Complex(flags.nu * eqn.baseflow.Wbase_yy[ny], 0.0)
                end
            end

            if flags.constraint == PressureGradient
                outfields[1][mx, 1, mz, 1] -= Complex(eqn.baseflow.dPdx_Ref, 0.0)
                outfields[1][mx, 1, mz, 3] -= Complex(eqn.baseflow.dPdz_Ref, 0.0)
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
                outfields[1][mx, 1, mz, 1] -= Complex(dPdxAct, 0.0)
                outfields[1][mx, 1, mz, 3] -= Complex(dPdzAct, 0.0)
            end
        end
    end
end

function solve!(eqn::NSE, outfields::Vector{FlowField}, rhs::Vector{FlowField}, s::Int, flags::DNSFlags)
    @assert num_dimensions(outfields) == num_dimensions(rhs) + 1 "Dimension mismatch. There should be no pressure field in outfields. Outfields should be created with create_RHS."
    kxmax = kx_max(outfields[1])
    kzmax = kz_max(outfields[1])
    # TODO: finish this
end