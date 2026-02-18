using Channelflow
using LinearAlgebra
using Printf

const GEOM = "test/data/u_guess_converted.geom"
const ASC = "test/data/u_guess_converted.asc"
const SYMM = "test/data/symm_E.asc"
const OUT = "test/data/julia_convergence.asc"

struct SymOp
    s::Int
    sx::Int
    sy::Int
    sz::Int
    ax::Float64
    az::Float64
end

struct ReducedCoeff
    mx::Int
    ny::Int
    mz::Int
    d::Int
    has_im::Bool
    real_plane::Bool
end

struct ReducedLayout
    coeffs::Vector{ReducedCoeff}
    kxd::Int
    kzd::Int
    Ny::Int
    include_nyquist_plane::Bool
    scheme::Symbol
end

mutable struct EvalCounter
    total::Int
    newt::Int
    opt::Int
    mode::Symbol
end

Base.@kwdef mutable struct CppNewtonFlags
    eps_search::Float64 = 1e-13
    eps_krylov::Float64 = 1e-14
    eps_dx::Float64 = 1e-7
    eps_solver::Float64 = 1e-3
    eps_solver_final::Float64 = 5e-2
    centered::Bool = false
    n_newton::Int = 20
    n_solver_max::Int = 500
    n_hook::Int = 20
    delta::Float64 = 1e-2
    delta_min::Float64 = 1e-12
    delta_max::Float64 = 1e-1
    delta_fuzz::Float64 = 1e-6
    lambda_min::Float64 = 0.2
    lambda_max::Float64 = 1.5
    improve_req::Float64 = 1e-3
    improve_ok::Float64 = 0.1
    improve_good::Float64 = 0.75
    g_ratio::Float64 = 10.0
end

function read_symm(path::String)
    ops = SymOp[]
    for ln in eachline(path)
        t = strip(ln)
        isempty(t) && continue
        startswith(t, "%") && continue
        p = split(t)
        length(p) < 6 && continue
        push!(ops, SymOp(parse(Int, p[1]), parse(Int, p[2]), parse(Int, p[3]), parse(Int, p[4]), parse(Float64, p[5]), parse(Float64, p[6])))
    end
    return ops
end

function apply_symmetry!(dst::FlowField, src::FlowField, op::SymOp)
    Nx, Ny, Nz = src.domain.Nx, src.domain.Ny, src.domain.Nz

    sxshift = mod(round(Int, op.ax * Nx), Nx)
    szshift = mod(round(Int, op.az * Nz), Nz)

    c1 = op.s * op.sx
    c2 = op.s * op.sy
    c3 = op.s * op.sz

    @inbounds for i in 0:Nx-1, j in 0:Ny-1, k in 0:Nz-1
        isrc = mod(op.sx == 1 ? (i + sxshift) : (-i + sxshift), Nx)
        jsrc = op.sy == 1 ? j : (Ny - 1 - j)
        ksrc = mod(op.sz == 1 ? (k + szshift) : (-k + szshift), Nz)

        dst.physical_data[i + 1, j + 1, k + 1, 1] = c1 * src.physical_data[isrc + 1, jsrc + 1, ksrc + 1, 1]
        dst.physical_data[i + 1, j + 1, k + 1, 2] = c2 * src.physical_data[isrc + 1, jsrc + 1, ksrc + 1, 2]
        dst.physical_data[i + 1, j + 1, k + 1, 3] = c3 * src.physical_data[isrc + 1, jsrc + 1, ksrc + 1, 3]
    end
    return dst
end

function project_symmetry!(u::FlowField, ops::Vector{SymOp}, work::FlowField; passes::Int = 1)
    make_state!(u, Physical, Physical)
    make_state!(work, Physical, Physical)
    for _ in 1:passes
        for op in ops
            apply_symmetry!(work, u, op)
            @inbounds u.physical_data .= 0.5 .* (u.physical_data .+ work.physical_data)
        end
    end
    return u
end

function reset_multistep!(dns::MultistepDNS, fields::Vector{<:FlowField})
    dns.fields_history[1] = fields
    for l in 1:dns.common.num_fields
        set_to_zero!(dns.nonlf_history[1][l])
    end
    for j in 2:dns.common.order
        for l in 1:dns.common.num_fields
            set_to_zero!(dns.fields_history[j][l])
            set_to_zero!(dns.nonlf_history[j][l])
        end
    end
    dns.countdown = dns.common.num_initsteps
    dns.common.t = dns.common.flags.t0
    return dns
end

function build_layout(u::FlowField, dealiased_state::Bool, scheme::Symbol)
    kxd = dealiased_state ? (div(u.domain.Nx, 3) - 1) : kx_max(u)
    kzd = dealiased_state ? (div(u.domain.Nz, 3) - 1) : kz_max(u)
    include_nyquist = (!dealiased_state) && (u.domain.Nz % 2 == 0)
    Ny = u.domain.My

    if scheme == :cppcount
        return ReducedLayout(ReducedCoeff[], kxd, kzd, Ny, false, scheme)
    end

    coeffs = ReducedCoeff[]
    for kz in 0:kzd
        mz = kz_to_mz(u, kz)
        real_plane = (kz == 0) || (include_nyquist && kz == kz_max(u))
        if real_plane
            for kx in 0:kxd
                mx = kx_to_mx(u, kx)
                has_im = kx != 0
                for ny in 1:u.domain.My, d in 1:3
                    push!(coeffs, ReducedCoeff(mx, ny, mz, d, has_im, true))
                end
            end
        else
            for kx in -kxd:kxd
                mx = kx_to_mx(u, kx)
                for ny in 1:u.domain.My, d in 1:3
                    push!(coeffs, ReducedCoeff(mx, ny, mz, d, true, false))
                end
            end
        end
    end
    return ReducedLayout(coeffs, kxd, kzd, Ny, include_nyquist, scheme)
end

function state_dim(layout::ReducedLayout)
    if layout.scheme == :cppcount
        # Exact C++ field2vector_size count:
        # 2*(Ny-2) + (Kx+Kz+2*Kx*Kz)*(2*(Ny-2)+2*(Ny-4))
        Ny = layout.Ny
        return 2 * (Ny - 2) + (layout.kxd + layout.kzd + 2 * layout.kxd * layout.kzd) * (2 * (Ny - 2) + 2 * (Ny - 4))
    end
    s = 0
    for c in layout.coeffs
        s += c.has_im ? 2 : 1
    end
    return s
end

function fix_diri!(f::ChebyCoeff{ComplexF64})
    fa = eval_a(f)
    fb = eval_b(f)
    mean = 0.5 * (fb + fa)
    slop = 0.5 * (fb - fa)
    f[1] -= mean
    f[2] -= slop
    return f
end

function fix_diri_mean!(f::ChebyCoeff{ComplexF64})
    fa = eval_a(f)
    fb = eval_b(f)
    fm = mean_value(f)
    f[1] -= 0.125 * (fa + fb) + 0.75 * fm
    f[2] -= 0.5 * (fb - fa)
    f[3] -= 0.375 * (fa + fb) - 0.75 * fm
    return f
end

function pack_cppcount!(x::Vector{Float64}, u::FlowField, layout::ReducedLayout)
    make_state!(u, Spectral, Spectral)
    Ny = layout.Ny
    Kx = layout.kxd
    Kz = layout.kzd
    k = 1

    @inbounds begin
        for ny in 3:Ny
            x[k] = real(cmplx(u, 1, ny, 1, 1))
            k += 1
        end
        for ny in 3:Ny
            x[k] = real(cmplx(u, 1, ny, 1, 3))
            k += 1
        end

        for kx in 1:Kx
            mx = kx_to_mx(u, kx)
            for ny in 3:Ny
                v = cmplx(u, mx, ny, 1, 3)
                x[k] = real(v)
                x[k + 1] = imag(v)
                k += 2
            end
            for ny in 4:Ny-1
                v = cmplx(u, mx, ny, 1, 1)
                x[k] = real(v)
                x[k + 1] = imag(v)
                k += 2
            end
        end

        for kz in 1:Kz
            mz = kz_to_mz(u, kz)
            for ny in 3:Ny
                v = cmplx(u, 1, ny, mz, 1)
                x[k] = real(v)
                x[k + 1] = imag(v)
                k += 2
            end
            for ny in 4:Ny-1
                v = cmplx(u, 1, ny, mz, 3)
                x[k] = real(v)
                x[k + 1] = imag(v)
                k += 2
            end
        end

        for kx in -Kx:Kx
            kx == 0 && continue
            mx = kx_to_mx(u, kx)
            for kz in 1:Kz
                mz = kz_to_mz(u, kz)
                for ny in 3:Ny
                    v = cmplx(u, mx, ny, mz, 1)
                    x[k] = real(v)
                    x[k + 1] = imag(v)
                    k += 2
                end
                for ny in 4:Ny-1
                    v = cmplx(u, mx, ny, mz, 3)
                    x[k] = real(v)
                    x[k + 1] = imag(v)
                    k += 2
                end
            end
        end
    end
    return x
end

function unpack_cppcount!(u::FlowField, x::AbstractVector{<:Real}, layout::ReducedLayout)
    make_state!(u, Spectral, Spectral)
    set_to_zero!(u)

    Ny = layout.Ny
    Kx = layout.kxd
    Kz = layout.kzd
    Lx = u.domain.Lx
    Lz = u.domain.Lz

    f0 = ChebyCoeff{ComplexF64}(Ny, u.domain.a, u.domain.b, Spectral)
    f1 = ChebyCoeff{ComplexF64}(Ny, u.domain.a, u.domain.b, Spectral)
    f2 = ChebyCoeff{ComplexF64}(Ny, u.domain.a, u.domain.b, Spectral)
    rhs = ChebyCoeff{ComplexF64}(Ny, u.domain.a, u.domain.b, Spectral)

    k = 1
    @inbounds begin
        # (0,0)
        fill!(f0.data, 0)
        fill!(f2.data, 0)
        for ny in 3:Ny
            f0[ny] = ComplexF64(x[k], 0)
            k += 1
        end
        fix_diri!(f0)
        for ny in 1:Ny
            set_cmplx!(u, f0[ny], 1, ny, 1, 1)
        end

        for ny in 3:Ny
            f2[ny] = ComplexF64(x[k], 0)
            k += 1
        end
        fix_diri!(f2)
        for ny in 1:Ny
            set_cmplx!(u, f2[ny], 1, ny, 1, 3)
        end

        # (kx,0), kx > 0
        for kx in 1:Kx
            mx = kx_to_mx(u, kx)
            mxm = kx_to_mx(u, -kx)
            fill!(f0.data, 0)
            fill!(f1.data, 0)
            fill!(f2.data, 0)

            for ny in 3:Ny
                f2[ny] = ComplexF64(x[k], x[k + 1])
                k += 2
            end
            fix_diri!(f2)

            for ny in 4:Ny-1
                f0[ny] = ComplexF64(x[k], x[k + 1])
                k += 2
            end
            f0[Ny] = 0
            fix_diri_mean!(f0)

            integrate!(f0, f1)
            f1[1] -= 0.5 * (eval_a(f1) + eval_b(f1))
            f1.data .*= ComplexF64(0, -(2 * pi * kx) / Lx)

            for ny in 1:Ny
                set_cmplx!(u, f0[ny], mx, ny, 1, 1)
                set_cmplx!(u, f1[ny], mx, ny, 1, 2)
                set_cmplx!(u, f2[ny], mx, ny, 1, 3)
                set_cmplx!(u, conj(f0[ny]), mxm, ny, 1, 1)
                set_cmplx!(u, conj(f1[ny]), mxm, ny, 1, 2)
                set_cmplx!(u, conj(f2[ny]), mxm, ny, 1, 3)
            end
        end

        # (0,kz), kz > 0
        for kz in 1:Kz
            mz = kz_to_mz(u, kz)
            fill!(f0.data, 0)
            fill!(f1.data, 0)
            fill!(f2.data, 0)

            for ny in 3:Ny
                f0[ny] = ComplexF64(x[k], x[k + 1])
                k += 2
            end
            fix_diri!(f0)

            for ny in 4:Ny-1
                f2[ny] = ComplexF64(x[k], x[k + 1])
                k += 2
            end
            f2[Ny] = 0
            fix_diri_mean!(f2)

            integrate!(f2, f1)
            f1[1] -= 0.5 * (eval_a(f1) + eval_b(f1))
            f1.data .*= ComplexF64(0, -(2 * pi * kz) / Lz)

            for ny in 1:Ny
                set_cmplx!(u, f0[ny], 1, ny, mz, 1)
                set_cmplx!(u, f1[ny], 1, ny, mz, 2)
                set_cmplx!(u, f2[ny], 1, ny, mz, 3)
            end
        end

        # (kx,kz), kx != 0, kz > 0
        for kx in -Kx:Kx
            kx == 0 && continue
            mx = kx_to_mx(u, kx)
            for kz in 1:Kz
                mz = kz_to_mz(u, kz)
                fill!(f0.data, 0)
                fill!(f1.data, 0)
                fill!(f2.data, 0)
                fill!(rhs.data, 0)

                for ny in 3:Ny
                    f0[ny] = ComplexF64(x[k], x[k + 1])
                    k += 2
                end
                fix_diri!(f0)
                for ny in 1:Ny
                    set_cmplx!(u, f0[ny], mx, ny, mz, 1)
                end

                for ny in 4:Ny-1
                    f2[ny] = ComplexF64(x[k], x[k + 1])
                    k += 2
                end
                f2[Ny] = -f0[Ny] * ((kx * Lz) / (kz * Lx))

                f2a = eval_a(f2)
                f2b = eval_b(f2)
                f0m = mean_value(f0)
                f2m = mean_value(f2) + ((kx * Lz) / (kz * Lx)) * f0m
                f2[1] -= 0.125 * (f2a + f2b) + 0.75 * f2m
                f2[2] -= 0.5 * (f2b - f2a)
                f2[3] -= 0.375 * (f2a + f2b) - 0.75 * f2m
                for ny in 1:Ny
                    set_cmplx!(u, f2[ny], mx, ny, mz, 3)
                end

                for ny in 1:Ny
                    rhs[ny] = ComplexF64(0, -2 * pi * kx / Lx) * f0[ny] + ComplexF64(0, -2 * pi * kz / Lz) * f2[ny]
                end
                integrate!(rhs, f1)
                f1[1] -= 0.5 * (eval_a(f1) + eval_b(f1))
                for ny in 1:Ny
                    set_cmplx!(u, f1[ny], mx, ny, mz, 2)
                end
            end
        end
    end
    u.padded = true
    return u
end

function pack_reduced!(x::Vector{Float64}, u::FlowField, layout::ReducedLayout)
    if layout.scheme == :cppcount
        return pack_cppcount!(x, u, layout)
    end
    make_state!(u, Spectral, Spectral)
    k = 0
    @inbounds for c in layout.coeffs
        val = cmplx(u, c.mx, c.ny, c.mz, c.d)
        x[k += 1] = real(val)
        if c.has_im
            x[k += 1] = imag(val)
        end
    end
    return x
end

function unpack_reduced!(u::FlowField, x::AbstractVector{<:Real}, layout::ReducedLayout)
    if layout.scheme == :cppcount
        return unpack_cppcount!(u, x, layout)
    end
    make_state!(u, Spectral, Spectral)
    set_to_zero!(u)

    k = 0
    @inbounds for c in layout.coeffs
        re = x[k += 1]
        im = c.has_im ? x[k += 1] : 0.0
        set_cmplx!(u, Complex(re, im), c.mx, c.ny, c.mz, c.d)
    end

    # Restore conjugate completion on real z-planes.
    for kz in (0, layout.include_nyquist_plane ? kz_max(u) : -1)
        kz < 0 && continue
        mz = kz_to_mz(u, kz)
        for kx in 1:layout.kxd
            mxp = kx_to_mx(u, kx)
            mxn = kx_to_mx(u, -kx)
            for ny in 1:u.domain.My, d in 1:3
                set_cmplx!(u, conj(cmplx(u, mxp, ny, mz, d)), mxn, ny, mz, d)
            end
        end
    end

    return u
end

function l2_from_state!(tmp::FlowField, x::AbstractVector{<:Real}, layout::ReducedLayout)
    unpack_reduced!(tmp, x, layout)
    return L2Norm(tmp)
end

function linear_step_logged!(dsi::FunctionDSI, x::Vector{Float64}, gx::Vector{Float64}, flags::CppNewtonFlags, nsolver::Int, ctr::EvalCounter)
    gmr = Channelflow.NSolver.GMRES(-gx, nsolver, flags.eps_krylov)
    solver_res = 1.0
    ctr.mode = :newt
    for k in 1:nsolver
        q = test_vector(gmr)
        aq = jacobian_action(dsi, x, q, gx; eps_dx = flags.eps_dx, centered = flags.centered)
        iterate!(gmr, aq)
        solver_res = residual(gmr)
        if solver_res < flags.eps_solver || (k == nsolver && solver_res < flags.eps_solver_final)
            return solution(gmr), solver_res, k
        end
    end
    return solution(gmr), solver_res, nsolver
end

function hookstep_accept!(
    dsi::FunctionDSI,
    x::Vector{Float64},
    gx::Vector{Float64},
    dxN::Vector{Float64},
    flags::CppNewtonFlags,
    delta::Float64,
    ctr::EvalCounter,
)
    gnorm = norm(gx)
    dnorm = norm(dxN)
    dnorm < 1e-30 && return false, x, gx, delta, 0.0

    # In C++, this hookstep Jacobian probe is accounted under optimization calls.
    ctr.mode = :opt
    jdxN = jacobian_action(dsi, x, dxN, gx; eps_dx = flags.eps_dx, centered = flags.centered)

    for _ in 1:flags.n_hook
        s = min(1.0, delta / dnorm)
        dx = s .* dxN

        ctr.mode = :opt
        xtrial = x .+ dx
        gtrial = eval_dsi(dsi, xtrial)

        gtrial_norm = norm(gtrial)
        if gtrial_norm >= gnorm
            delta *= flags.lambda_min
            delta < flags.delta_min && break
            continue
        end

        pred = gnorm - norm(gx .+ s .* jdxN)
        actual = gnorm - gtrial_norm
        ratio = pred > 0 ? actual / pred : 0.0

        if ratio >= flags.improve_req
            hookstep_equals_newtonstep = s >= (1 - flags.delta_fuzz)
            if ratio < flags.improve_ok
                delta = max(flags.delta_min, flags.lambda_min * delta)
            elseif ratio > flags.improve_good && !hookstep_equals_newtonstep
                delta = min(flags.delta_max, flags.lambda_max * delta)
            end
            return true, xtrial, gtrial, delta, norm(dx)
        end

        delta *= flags.lambda_min
        delta < flags.delta_min && break
    end

    return false, x, gx, delta, 0.0
end

function main()
    n_newton = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
    outpath = length(ARGS) >= 2 ? ARGS[2] : OUT
    dealiased_state = length(ARGS) >= 3 ? (parse(Int, ARGS[3]) != 0) : true
    nsolver_init = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 80
    project_passes = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 0
    geom_path = length(ARGS) >= 6 ? ARGS[6] : GEOM
    data_path = length(ARGS) >= 7 ? ARGS[7] : ASC
    symm_path = length(ARGS) >= 8 ? ARGS[8] : SYMM
    layout_mode = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : 0
    layout_scheme = layout_mode == 1 ? :cppcount : :full

    domain = read_geom(geom_path)
    ops = read_symm(symm_path)

    u_init = read_data(data_path, domain)
    u_proj = FlowField(u_init)
    if project_passes > 0
        project_symmetry!(u_proj, ops, FlowField(u_proj); passes = project_passes)
    end
    make_state!(u_proj, Spectral, Spectral)

    flags = DNSFlags(
        nu = 1 / 300,
        dPdx = 0.0,
        dPdz = 0.0,
        Ubulk = 0.0,
        Wbulk = 0.0,
        Uwall = 1.0,
        uupperwall = 1.0,
        ulowerwall = -1.0,
        wupperwall = 0.0,
        wlowerwall = 0.0,
        t0 = 0.0,
        T = 10.0,
        dT = 1.0,
        dt = 0.03125,
        variabledt = true,
        dtmin = 0.001,
        dtmax = 0.2,
        CFLmin = 0.4,
        CFLmax = 0.6,
        baseflow = LaminarBase,
        constraint = PressureGradient,
        timestepping = SBDF3,
        initstepping = SMRK2,
        nonlinearity = Rotational,
        dealiasing = DealiasXZ,
        taucorrection = true,
        verbosity = Silent,
    )

    layout = build_layout(u_proj, dealiased_state, layout_scheme)
    N = state_dim(layout)
    @printf("Reduced state dimension: %d (dealiased_state=%s, scheme=%s)\n", N, string(dealiased_state), String(layout_scheme))

    x0 = zeros(Float64, N)
    pack_reduced!(x0, u_proj, layout)

    u_work = FlowField(u_proj)
    p_work = FlowField(domain.Nx, domain.Ny, domain.Nz, 1, domain.Lx, domain.Lz, domain.a, domain.b)
    sym_work = FlowField(u_work)
    x_proj = similar(x0)
    rbuf = similar(x0)
    tmp_u = FlowField(u_work)
    tmp_g = FlowField(u_work)
    tmp_dx = FlowField(u_work)
    fields = [u_work, p_work]
    nse = NSE(fields, flags)
    dns = MultistepDNS(fields, nse, flags)

    nsteps = Int(round(flags.T / flags.dt))
    tnormalize = true
    ctr = EvalCounter(0, 0, 0, :newt)
    profile_map = get(ENV, "CHANNELFLOW_PROFILE_MAP", "0") == "1"

    function map_residual(x::Vector{Float64})
        ctr.total += 1
        if ctr.mode == :newt
            ctr.newt += 1
        else
            ctr.opt += 1
        end

        t_unpack = time()
        unpack_reduced!(u_work, x, layout)
        if project_passes > 0
            project_symmetry!(u_work, ops, sym_work; passes = project_passes)
            pack_reduced!(x_proj, u_work, layout)
        else
            x_proj .= x
        end
        t_unpack = time() - t_unpack

        t_adv = time()
        adv_alloc = 0
        make_state!(u_work, Spectral, Spectral)
        set_to_zero!(p_work)
        make_state!(p_work, Spectral, Spectral)

        reset_multistep!(dns, fields)
        if profile_map
            stats = @timed advance!(dns, fields, nsteps)
            adv_alloc = stats.bytes
        else
            advance!(dns, fields, nsteps)
        end
        t_adv = time() - t_adv

        t_pack = time()
        pack_reduced!(rbuf, u_work, layout)
        @inbounds @. rbuf = rbuf - x_proj
        if tnormalize
            @inbounds @. rbuf = rbuf / flags.T
        end
        t_pack = time() - t_pack

        if profile_map
            @printf("eval %-4d |Rvec| %.6e mode=%s unpack=%.3fs advance=%.3fs pack=%.3fs alloc=%.3f GB\n",
                    ctr.total, norm(rbuf), String(ctr.mode), t_unpack, t_adv, t_pack, adv_alloc / 1024^3)
        else
            @printf("eval %-4d |Rvec| %.6e mode=%s\n", ctr.total, norm(rbuf), String(ctr.mode))
        end
        return copy(rbuf)
    end

    dsi = FunctionDSI(map_residual)
    nflags = CppNewtonFlags(n_newton = n_newton)

    open(outpath, "w") do io
        @printf(io, "%-14s %-14s %-14s %-14s %-14s %-14s %-14s %-14s %-9s %-9s %-9s\n",
                "%-L2Norm(Gx)", "rx", "delta", "L2Norm(x)", "L2Norm(u)", "L2Norm(dxN)", "L2Norm(dxOpt)", "SolverRes", "ftotal", "fnewt", "fopt")

        x = copy(x0)
        ctr.mode = :newt
        gx = eval_dsi(dsi, x)
        delta = nflags.delta
        nsolver = min(max(1, nsolver_init), nflags.n_solver_max)
        gx_prev = norm(gx)

        function write_row(dxN_l2::Float64, dxOpt_l2::Float64, solver_res::Float64)
            gxnorm = norm(gx)
            rx = 0.5 * gxnorm^2
            lx = norm(x)
            lu = l2_from_state!(tmp_u, x, layout)
            @printf(io, "%-14.6e %-14.6e %-14.6g %-14.6e %-14.6e %-14.6e %-14.6e %-14.6e %-9d %-9d %-9d\n",
                    gxnorm, rx, delta, lx, lu, dxN_l2, dxOpt_l2, solver_res, ctr.total, ctr.newt, ctr.opt)
            flush(io)
        end

        write_row(0.0, 0.0, 1.0)

        for it in 1:nflags.n_newton
            gnorm = norm(gx)
            if gnorm < nflags.eps_search
                @printf("Converged at Newton step %d\n", it - 1)
                break
            end

            dxN, solver_res, _ = linear_step_logged!(dsi, x, gx, nflags, nsolver, ctr)
            dxN_l2 = norm(dxN)

            accepted, xnew, gnew, delta_new, dxOpt_norm = hookstep_accept!(dsi, x, gx, dxN, nflags, delta, ctr)

            if !accepted
                delta = delta_new
                write_row(dxN_l2, 0.0, solver_res)
                if delta < nflags.delta_min
                    println("delta below delta_min, stopping")
                    break
                end
                continue
            end

            x = xnew
            gx = gnew
            delta = delta_new
            dxOpt_l2 = dxOpt_norm
            write_row(dxN_l2, dxOpt_l2, solver_res)

            gcurr = norm(gx)
            gx_prev = gcurr
        end
    end

    println("Wrote $(outpath)")
    println("Counts: total=$(ctr.total) newt=$(ctr.newt) opt=$(ctr.opt)")
end

main()
