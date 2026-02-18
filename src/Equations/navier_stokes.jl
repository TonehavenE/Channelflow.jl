using ..TauSolvers
using ..ChebyCoeffs
using ..FlowFields
using ..BasisFuncs
import Base.@kwdef
using Base.Threads

import ..TauSolvers: solve!

export NSE, nonlinear!

function _configured_mode_threads()
    if haskey(ENV, "CHANNELFLOW_MODE_THREADS")
        cfg = strip(ENV["CHANNELFLOW_MODE_THREADS"])
        if lowercase(cfg) == "auto"
            return Threads.nthreads()
        end
        nthreads = tryparse(Int, cfg)
        if nthreads !== nothing && nthreads > 0
            return nthreads
        end
    end
    return 1
end

function profile(ff::FlowField{T}, mx::Int, mz::Int, i::Int) where {T<:Number}
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

function profile(ff::FlowField{T}, mx::Int, mz::Int) where {T<:Number}
    ret = BasisFunc(num_dimensions(ff), ff.domain.Ny, mx_to_kx(ff, mx), mz_to_kz(ff, mz), ff.domain.Lx, ff.domain.Lz, ff.domain.a, ff.domain.b, y_state(ff))
    for i = 1:num_dimensions(ff), ny = 1:ff.domain.Ny
        ret[i, ny] = cmplx(ff, mx, ny, mz, i)
    end
    return ret
end

function get_Ubulk(ff::FlowField{T}) where {T<:Number}
    ubulk = mean_value(profile(ff, 1, 1, 1))
    if abs(ubulk) < 1e-15
        ubulk = 0.0
    end
    return ubulk
end

function get_Wbulk(ff::FlowField{T}) where {T<:Number}
    wbulk = mean_value(profile(ff, 1, 1, 3))
    if abs(wbulk) < 1e-15
        wbulk = 0.0
    end
    return wbulk
end

function dudy_a(ff::FlowField{T}) where {T<:Number}
    @assert y_state(ff) == Spectral
    prof = profile(ff, 1, 1)
    dudy = derivative(realview(get_u(prof)))
    return eval_a(dudy)
end

function dudy_b(ff::FlowField{T}) where {T<:Number}
    @assert y_state(ff) == Spectral
    prof = profile(ff, 1, 1)
    dudy = derivative(realview(get_u(prof)))
    return eval_b(dudy)
end

function dwdy_a(ff::FlowField{T}) where {T<:Number}
    @assert y_state(ff) == Spectral
    prof = profile(ff, 1, 1)
    dwdy = derivative(realview(get_w(prof)))
    return eval_a(dwdy)
end
function dwdy_b(ff::FlowField{T}) where {T<:Number}
    @assert y_state(ff) == Spectral
    prof = profile(ff, 1, 1)
    dwdy = derivative(realview(get_w(prof)))
    return eval_b(dwdy)
end

function get_dPdx(ff::FlowField{T}, nu::Real) where {T<:Number}
    nu * (dudy_b(ff) - dudy_a(ff)) / Ly(ff)
end

function get_dPdz(ff::FlowField{T}, nu::Real) where {T<:Number}
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
    kx_vals::Vector{Int}
    kz_vals::Vector{Int}

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
    ff::FlowField{<:Number}
    ff2::FlowField{<:Number}
    uk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
    vk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
    wk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
    Pk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
    Pyk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
    Ruk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
    Rvk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
    Rwk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
end

function TransientFields(temp::FlowField, Nyd::Int, a::Real, b::Real)
    return TransientFields(
        ff=temp,
        ff2=FlowField(temp),
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

@kwdef mutable struct ModeScratch
    uk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
    vk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
    wk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
    Pk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
    Pyk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
    Ruk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
    Rvk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
    Rwk::ChebyCoeff{ComplexF64,Vector{ComplexF64}}
end

function ModeScratch(Nyd::Int, a::Real, b::Real)
    uk = ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral)
    vk = ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral)
    wk = ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral)
    Pk = ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral)
    Pyk = ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral)
    Ruk = ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral)
    Rvk = ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral)
    Rwk = ChebyCoeff{ComplexF64}(Nyd, a, b, Spectral)
    return ModeScratch(
        uk=uk,
        vk=vk,
        wk=wk,
        Pk=Pk,
        Pyk=Pyk,
        Ruk=Ruk,
        Rvk=Rvk,
        Rwk=Rwk,
    )
end

@kwdef mutable struct NSE <: Equation
    lambda_t::Vector{Float64}
    tausolvers::Union{Array{TauSolver,3},Nothing}

    # Refactored components
    spatial::SpatialParameters
    baseflow::BaseFlowMembers
    tmp::TransientFields
    mode_threads::Int
    mode_scratch::Vector{ModeScratch}
end

#TODO finish implementing laminar_profile
function laminar_profile(nu::Real, constraint::MeanConstraint, dPdx::Real, Ubulk::Real, Vsuck::Real, a::Real, b::Real, ua::Real, ub::Real, Ny::Int)
    u = ChebyCoeff(Ny, a, b, Spectral)
    H = b - a
    mu = Vsuck * H / nu

    # Non-zero suction is not implemented yet in Julia.
    if abs(mu) > 1e-12
        error("Error in laminar_profile: non-zero Vsuck is not implemented")
    end

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
        Ubase[2] = 1
    elseif flags.baseflow == ParabolicBase
        @assert My > 2 "My must be greater than 2 for parabolic base flow"
        Ubase = ChebyCoeff(My, a, b, Spectral)
        Wbase = ChebyCoeff(My, a, b, Spectral)
        Ubase[1] = 0.5
        Ubase[3] = -0.5
    elseif flags.baseflow == SuctionBase
        Ubase = laminar_profile(flags.nu, PressureGradient, 0, flags.Ubulk, flags.Vsuck, a, b, -0.5, 0.5, My)
        Wbase = ChebyCoeff(My, a, b, Spectral)
    elseif flags.baseflow == LaminarBase
        Ubase = laminar_profile(flags.nu, flags.constraint, flags.dPdx, flags.Ubulk, flags.Vsuck, a, b, flags.ulowerwall, flags.uupperwall, My)
        Wbase = laminar_profile(flags.nu, flags.constraint, flags.dPdz, flags.Wbulk, flags.Vsuck, a, b, flags.wlowerwall, flags.wupperwall, My)
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
        kx_vals=[mx_to_kx(u, mx) for mx in 1:u.domain.Mx],
        kz_vals=[mz_to_kz(u, mz) for mz in 1:u.domain.Mz],
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
    mode_threads = max(1, _configured_mode_threads())
    mode_scratch_count = mode_threads > 1 ? spatial.Mx : 1
    mode_scratch = [ModeScratch(Nyd, u.domain.a, u.domain.b) for _ = 1:mode_scratch_count]

    return NSE(
        lambda_t=[0.0],
        tausolvers=nothing,
        spatial=spatial,
        baseflow=baseflow,
        tmp=transients,
        mode_threads=mode_threads,
        mode_scratch=mode_scratch,
    )
end

function reset_lambda!(eqn::NSE, lambda_t::Vector{T}, flags::DNSFlags) where {T<:Real}
    eqn.lambda_t = Float64.(lambda_t)

    c = 4.0 * pi^2 * flags.nu

    # Create the fully configured 3D array of TauSolver objects in one step.
    eqn.tausolvers = [
        begin
            # These calculations are done for each element of the new array
            kx = eqn.spatial.kx_vals[mx]
            kz = eqn.spatial.kz_vals[mz]

            # Check the condition for this element
            if (kx != kx_max(eqn.tmp.ff) || kz != kz_max(eqn.tmp.ff)) && (!dealias_xz(flags) || !is_aliased(eqn.tmp.ff, kx, kz))
                lambda = eqn.lambda_t[j] + c * ((kx / eqn.spatial.Lx)^2 + (kz / eqn.spatial.Lz)^2)

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

function navierstokes_nonlinear!(u::FlowField{Q}, Ubase::ChebyCoeff{R}, Wbase::ChebyCoeff{S}, f::FlowField{T}, tmp::FlowField{U}, tmp2::FlowField{V}, flags::DNSFlags) where {Q,R,S,T,U,V<:Number}
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
        rotational_nonlinear!(u, f, tmp, tmp2, finalstate)
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
        if Threads.nthreads() > 1 && u.domain.Nz > 1
            Threads.@threads for nz = 1:u.domain.Nz
                @inbounds for ny = 1:u.domain.Ny, nx = 1:u.domain.Nx
                    f[nx, ny, nz, 1] -= (flags.rotation) * u[nx, ny, nz, 2]
                    f[nx, ny, nz, 2] += (flags.rotation) * u[nx, ny, nz, 1]
                end
            end
        else
            for nx = 1:u.domain.Nx, ny = 1:u.domain.Ny, nz = 1:u.domain.Nz
                f[nx, ny, nz, 1] -= (flags.rotation) * u[nx, ny, nz, 2]
                f[nx, ny, nz, 2] += (flags.rotation) * u[nx, ny, nz, 1]
            end
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
function nonlinear!(eqn::NSE, infields::Vector{<:FlowField{<:Number}}, outfields::Vector{<:FlowField{<:Number}}, flags::DNSFlags)
    navierstokes_nonlinear!(infields[1], eqn.baseflow.Ubase, eqn.baseflow.Wbase, outfields[1], eqn.tmp.ff, eqn.tmp.ff2, flags)
    if dealias_xz(flags)
        zero_padded_modes!(outfields[1])
    end
    return
end

function rotational_nonlinear!(u::FlowField{Q}, f::FlowField{R}, tmp::FlowField{S}, tmp2::FlowField{U}, finalstate::FieldState) where {Q,R,S,U<:Number}
    @assert num_dimensions(u) == 3 "FlowField must have 3 dimensions for rotational nonlinearity"
    vort = tmp
    uwork = tmp2

    if !geom_congruent(u, f) || num_dimensions(f) != 3
        resize!(f, u.domain.Nx, u.domain.Ny, u.domain.Nz, 3, u.domain.Lx, u.domain.Lz, u.domain.a, u.domain.b)
    end
    # f is fully overwritten in physical space below, so skip an unnecessary
    # spectral->physical transform when f currently holds spectral data.
    f.xz_state = Physical
    f.y_state = Physical

    if !geom_congruent(u, vort) || num_dimensions(vort) != 3
        resize!(vort, u.domain.Nx, u.domain.Ny, u.domain.Nz, 3, u.domain.Lx, u.domain.Lz, u.domain.a, u.domain.b)
    end
    if !geom_congruent(u, uwork) || num_dimensions(uwork) != 3
        resize!(uwork, u.domain.Nx, u.domain.Ny, u.domain.Nz, 3, u.domain.Lx, u.domain.Lz, u.domain.a, u.domain.b)
    end
    make_spectral!(u)
    uwork.xz_state = Spectral
    uwork.y_state = Spectral
    @inbounds uwork.spectral_data .= u.spectral_data
    curl!(u, vort)

    make_physical!(uwork)
    make_physical!(vort)

    # Compute f = vort x u directly in physical space to avoid extra state
    # transitions and duplicate transforms inside cross!.
    u_phys = uwork.physical_data
    vort_phys = vort.physical_data
    f_phys = f.physical_data
    if Threads.nthreads() > 1 && u.domain.Nz > 1
        Threads.@threads for nz = 1:u.domain.Nz
            @inbounds for ny = 1:u.domain.Ny, nx = 1:u.domain.Nx
                # (vort x u)_x
                f_phys[nx, ny, nz, 1] = vort_phys[nx, ny, nz, 2] * u_phys[nx, ny, nz, 3] - vort_phys[nx, ny, nz, 3] * u_phys[nx, ny, nz, 2]
                # (vort x u)_y
                f_phys[nx, ny, nz, 2] = vort_phys[nx, ny, nz, 3] * u_phys[nx, ny, nz, 1] - vort_phys[nx, ny, nz, 1] * u_phys[nx, ny, nz, 3]
                # (vort x u)_z
                f_phys[nx, ny, nz, 3] = vort_phys[nx, ny, nz, 1] * u_phys[nx, ny, nz, 2] - vort_phys[nx, ny, nz, 2] * u_phys[nx, ny, nz, 1]
            end
        end
    else
        @inbounds for nz = 1:u.domain.Nz, ny = 1:u.domain.Ny, nx = 1:u.domain.Nx
            # (vort x u)_x
            f_phys[nx, ny, nz, 1] = vort_phys[nx, ny, nz, 2] * u_phys[nx, ny, nz, 3] - vort_phys[nx, ny, nz, 3] * u_phys[nx, ny, nz, 2]
            # (vort x u)_y
            f_phys[nx, ny, nz, 2] = vort_phys[nx, ny, nz, 3] * u_phys[nx, ny, nz, 1] - vort_phys[nx, ny, nz, 1] * u_phys[nx, ny, nz, 3]
            # (vort x u)_z
            f_phys[nx, ny, nz, 3] = vort_phys[nx, ny, nz, 1] * u_phys[nx, ny, nz, 2] - vort_phys[nx, ny, nz, 2] * u_phys[nx, ny, nz, 1]
        end
    end

    if finalstate == Spectral
        make_spectral!(f)
    end

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
        kx = eqn.spatial.kx_vals[mx]

        for mz = 1:Mz
            kz = eqn.spatial.kz_vals[mz]

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
            derivative2!(eqn.tmp.uk, eqn.tmp.Pyk, eqn.tmp.Ruk)
            derivative2!(eqn.tmp.vk, eqn.tmp.Pyk, eqn.tmp.Rvk)
            derivative2!(eqn.tmp.wk, eqn.tmp.Pyk, eqn.tmp.Rwk)

            # eqn.tmp.Ruk = derivative2(eqn.tmp.uk)
            # eqn.tmp.Rvk = derivative2(eqn.tmp.vk)
            # eqn.tmp.Rwk = derivative2(eqn.tmp.wk)

            # Compute pressure gradient in y
            # eqn.tmp.Pyk = derivative(eqn.tmp.Pk)
            derivative!(eqn.tmp.Pk, eqn.tmp.Pyk)

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
                    # eqn.tmp.Ruk = derivative(eqn.tmp.uk)
                    # eqn.tmp.Rwk = derivative(eqn.tmp.wk)
                    derivative!(eqn.tmp.uk, eqn.tmp.Ruk)
                    derivative!(eqn.tmp.wk, eqn.tmp.Rwk)
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

function solve!(eqn::NSE, outfields::Vector{FlowField{T}}, rhs::Vector{FlowField{T}}, s::Int, flags::DNSFlags) where {T<:Number}
    @assert length(outfields) == length(rhs) + 1 "Make sure user provides correct RHS which can be created outside NSE with create_RHS()"
    _solve_nse!(eqn, outfields[1], outfields[2], rhs[1], s, flags)
end

function solve!(eqn::NSE, outfields::AbstractVector{<:FlowField}, rhs::AbstractVector{<:FlowField}, s::Int, flags::DNSFlags)
    @assert length(outfields) == length(rhs) + 1 "Make sure user provides correct RHS which can be created outside NSE with create_RHS()"
    _solve_nse!(eqn, outfields[1], outfields[2], rhs[1], s, flags)
end

function _solve_nse_mode!(
    eqn::NSE,
    u_spec::AbstractArray{Complex{T},4},
    p_spec::AbstractArray{Complex{T},4},
    rhs_spec::AbstractArray{Complex{T},4},
    mx::Int,
    mz::Int,
    kx::Int,
    kz::Int,
    kxmax::Int,
    kzmax::Int,
    nx_even::Bool,
    nz_even::Bool,
    flags::DNSFlags,
    scratch::ModeScratch,
    uk_re,
    uk_im,
    vk_re,
    vk_im,
    wk_re,
    wk_im,
    Pk_re,
    Pk_im,
    Rvk_re,
    Rvk_im,
    tausolvers::Array{TauSolver,3},
    s::Int,
) where {T<:Real}
    ukd = scratch.uk.data
    vkd = scratch.vk.data
    wkd = scratch.wk.data
    Pkd = scratch.Pk.data
    Rukd = scratch.Ruk.data
    Rvkd = scratch.Rvk.data
    Rwkd = scratch.Rwk.data

    # Construct ComplexChebyCoeff from RHS
    @inbounds for ny = 1:eqn.spatial.Nyd
        Rukd[ny] = rhs_spec[mx, ny, mz, 1]
        Rvkd[ny] = rhs_spec[mx, ny, mz, 2]
        Rwkd[ny] = rhs_spec[mx, ny, mz, 3]
    end

    # Solve the tau equations
    if kx != 0 || kz != 0
        solve!(tausolvers[s, mx, mz], scratch.uk, scratch.vk, scratch.wk, scratch.Pk,
            scratch.Ruk, scratch.Rvk, scratch.Rwk,
            uk_re, uk_im, vk_re, vk_im,
            wk_re, wk_im, Pk_re, Pk_im,
            Rvk_re, Rvk_im)
    else # kx,kz == 0,0
        if length(eqn.baseflow.Ubase_yy.data) > 0
            @inbounds for ny = 1:eqn.spatial.Ny
                Rukd[ny] += flags.nu * eqn.baseflow.Ubase_yy[ny]
            end
        end
        if length(eqn.baseflow.Wbase_yy.data) > 0
            @inbounds for ny = 1:eqn.spatial.Ny
                Rwkd[ny] += flags.nu * eqn.baseflow.Wbase_yy[ny]
            end
        end

        if flags.constraint == PressureGradient
            Rukd[1] -= Complex(eqn.baseflow.dPdx_Ref, 0)
            Rwkd[1] -= Complex(eqn.baseflow.dPdz_Ref, 0)
            solve!(tausolvers[s, mx, mz], scratch.uk, scratch.vk, scratch.wk, scratch.Pk,
                scratch.Ruk, scratch.Rvk, scratch.Rwk,
                uk_re, uk_im, vk_re, vk_im,
                wk_re, wk_im, Pk_re, Pk_im,
                Rvk_re, Rvk_im)
        else
            solve!(tausolvers[s, mx, mz], scratch.uk, scratch.vk, scratch.wk, scratch.Pk,
                eqn.baseflow.dPdx_Act, eqn.baseflow.dPdz_Act, scratch.Ruk, scratch.Rvk, scratch.Rwk,
                eqn.baseflow.Ubulk_Ref - eqn.baseflow.Ubulk_Base,
                eqn.baseflow.Wbulk_Ref - eqn.baseflow.Wbulk_Base)

            @assert abs(eqn.baseflow.Ubulk_Ref - eqn.baseflow.Ubulk_Base - mean_value(scratch.uk.re)) < 1e-15 "UbulkRef != UbulkAct = UbulkBase + uk.re.mean()"
            @assert abs(eqn.baseflow.Wbulk_Ref - eqn.baseflow.Wbulk_Base - mean_value(scratch.wk.re)) < 1e-15 "WbulkRef != WbulkAct = WbulkBase + wk.re.mean()"
        end
    end

    # Load solutions into u and p.
    force_real = ((kx == 0 && kz == 0) ||
                  (nx_even && kx == kxmax && kz == 0) ||
                  (nz_even && kz == kzmax && kx == 0) ||
                  (nx_even && nz_even && kx == kxmax && kz == kzmax))

    if force_real
        @inbounds for ny = 1:eqn.spatial.Nyd
            u_spec[mx, ny, mz, 1] = Complex(real(ukd[ny]), 0.0)
            u_spec[mx, ny, mz, 2] = Complex(real(vkd[ny]), 0.0)
            u_spec[mx, ny, mz, 3] = Complex(real(wkd[ny]), 0.0)
            p_spec[mx, ny, mz, 1] = Complex(real(Pkd[ny]), 0.0)
        end
    else
        @inbounds for ny = 1:eqn.spatial.Nyd
            u_spec[mx, ny, mz, 1] = ukd[ny]
            u_spec[mx, ny, mz, 2] = vkd[ny]
            u_spec[mx, ny, mz, 3] = wkd[ny]
            p_spec[mx, ny, mz, 1] = Pkd[ny]
        end
    end
    return
end

function _solve_nse_threaded!(
    eqn::NSE,
    uout::FlowField{T},
    u_spec::Array{Complex{T},4},
    p_spec::Array{Complex{T},4},
    rhs_spec::Array{Complex{T},4},
    tausolvers::Array{TauSolver,3},
    s::Int,
    flags::DNSFlags,
    kxmax::Int,
    kzmax::Int,
    nx_even::Bool,
    nz_even::Bool,
    dealias::Bool,
) where {T<:Real}
    Threads.@threads for mx = 1:eqn.spatial.Mx
        kx = eqn.spatial.kx_vals[mx]
        scratch = eqn.mode_scratch[mx]
        uk_re = realview(scratch.uk)
        uk_im = imagview(scratch.uk)
        vk_re = realview(scratch.vk)
        vk_im = imagview(scratch.vk)
        wk_re = realview(scratch.wk)
        wk_im = imagview(scratch.wk)
        Pk_re = realview(scratch.Pk)
        Pk_im = imagview(scratch.Pk)
        Rvk_re = realview(scratch.Rvk)
        Rvk_im = imagview(scratch.Rvk)
        for mz = 1:eqn.spatial.Mz
            kz = eqn.spatial.kz_vals[mz]
            if (kx == kxmax || kz == kzmax) || (dealias && is_aliased(uout, kx, kz))
                continue
            end
            _solve_nse_mode!(
                eqn, u_spec, p_spec, rhs_spec, mx, mz, kx, kz, kxmax, kzmax,
                nx_even, nz_even, flags, scratch,
                uk_re, uk_im, vk_re, vk_im, wk_re, wk_im, Pk_re, Pk_im, Rvk_re, Rvk_im,
                tausolvers, s,
            )
        end
    end
    return
end

function _solve_nse_serial!(
    eqn::NSE,
    uout::FlowField{T},
    u_spec::Array{Complex{T},4},
    p_spec::Array{Complex{T},4},
    rhs_spec::Array{Complex{T},4},
    tausolvers::Array{TauSolver,3},
    s::Int,
    flags::DNSFlags,
    kxmax::Int,
    kzmax::Int,
    nx_even::Bool,
    nz_even::Bool,
    dealias::Bool,
) where {T<:Real}
    scratch = eqn.mode_scratch[1]
    uk_re = realview(scratch.uk)
    uk_im = imagview(scratch.uk)
    vk_re = realview(scratch.vk)
    vk_im = imagview(scratch.vk)
    wk_re = realview(scratch.wk)
    wk_im = imagview(scratch.wk)
    Pk_re = realview(scratch.Pk)
    Pk_im = imagview(scratch.Pk)
    Rvk_re = realview(scratch.Rvk)
    Rvk_im = imagview(scratch.Rvk)
    @inbounds for mx = 1:eqn.spatial.Mx
        kx = eqn.spatial.kx_vals[mx]
        for mz = 1:eqn.spatial.Mz
            kz = eqn.spatial.kz_vals[mz]
            if (kx == kxmax || kz == kzmax) || (dealias && is_aliased(uout, kx, kz))
                continue
            end
            _solve_nse_mode!(
                eqn, u_spec, p_spec, rhs_spec, mx, mz, kx, kz, kxmax, kzmax,
                nx_even, nz_even, flags, scratch,
                uk_re, uk_im, vk_re, vk_im, wk_re, wk_im, Pk_re, Pk_im, Rvk_re, Rvk_im,
                tausolvers, s,
            )
        end
    end
    return
end

function _solve_nse!(eqn::NSE, uout::FlowField{T}, pout::FlowField{T}, rhsu::FlowField{T}, s::Int, flags::DNSFlags) where {T<:Number}
    @assert xz_state(uout) == Spectral && y_state(uout) == Spectral
    @assert xz_state(pout) == Spectral && y_state(pout) == Spectral
    @assert xz_state(rhsu) == Spectral && y_state(rhsu) == Spectral

    tausolvers = eqn.tausolvers::Array{TauSolver,3}

    kxmax = kx_max(uout)
    kzmax = kz_max(uout)
    nx_even = iseven(uout.domain.Nx)
    nz_even = iseven(uout.domain.Nz)
    u_spec = uout.spectral_data
    p_spec = pout.spectral_data
    rhs_spec = rhsu.spectral_data
    dealias = dealias_xz(flags)

    nsolve_threads = min(max(1, eqn.mode_threads), Threads.nthreads())
    use_threads = nsolve_threads > 1 && eqn.spatial.Mx > 1

    if use_threads
        _solve_nse_threaded!(
            eqn, uout, u_spec, p_spec, rhs_spec, tausolvers, s, flags,
            kxmax, kzmax, nx_even, nz_even, dealias,
        )
    else
        _solve_nse_serial!(
            eqn, uout, u_spec, p_spec, rhs_spec, tausolvers, s, flags,
            kxmax, kzmax, nx_even, nz_even, dealias,
        )
    end
    return
end
