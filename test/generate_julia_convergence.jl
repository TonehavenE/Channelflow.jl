using Channelflow
using LinearAlgebra
using Printf

const GEOM = "test/data/E_guess_converted.geom"
const ASC = "test/data/E_guess_converted.asc"
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

mutable struct EvalCounter
    total::Int
    newt::Int
    opt::Int
    mode::Symbol
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

function pack_velocity!(x::Vector{Float64}, u::FlowField)
    make_state!(u, Physical, Physical)
    Nx, Ny, Nz = u.domain.Nx, u.domain.Ny, u.domain.Nz
    k = 0
    @inbounds for nx in 1:Nx, ny in 1:Ny, nz in 1:Nz
        x[k += 1] = u.physical_data[nx, ny, nz, 1]
        x[k += 1] = u.physical_data[nx, ny, nz, 2]
        x[k += 1] = u.physical_data[nx, ny, nz, 3]
    end
    return x
end

function unpack_velocity!(u::FlowField, x::AbstractVector{<:Real})
    make_state!(u, Physical, Physical)
    Nx, Ny, Nz = u.domain.Nx, u.domain.Ny, u.domain.Nz
    k = 0
    @inbounds for nx in 1:Nx, ny in 1:Ny, nz in 1:Nz
        u.physical_data[nx, ny, nz, 1] = x[k += 1]
        u.physical_data[nx, ny, nz, 2] = x[k += 1]
        u.physical_data[nx, ny, nz, 3] = x[k += 1]
    end
    return u
end

function apply_symmetry!(dst::FlowField, src::FlowField, op::SymOp)
    make_state!(dst, Physical, Physical)
    make_state!(src, Physical, Physical)
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

function project_symmetry!(u::FlowField, ops::Vector{SymOp}, work::FlowField)
    make_state!(u, Physical, Physical)
    for _ in 1:2
        for op in ops
            apply_symmetry!(work, u, op)
            @inbounds u.physical_data .= 0.5 .* (u.physical_data .+ work.physical_data)
        end
    end
    return u
end

function l2_from_vec!(tmp::FlowField, v::AbstractVector{<:Real})
    unpack_velocity!(tmp, v)
    make_state!(tmp, Spectral, Spectral)
    return L2Norm(tmp)
end

function linear_step_logged!(dsi::FunctionDSI, x::Vector{Float64}, gx::Vector{Float64}, flags::NewtonSearchFlags, ctr::EvalCounter)
    if flags.solver == :direct
        n = length(x)
        j = zeros(Float64, n, n)
        ctr.mode = :newt
        for i in 1:n
            e = zeros(Float64, n)
            e[i] = 1.0
            j[:, i] .= jacobian_action(dsi, x, e, gx; eps_dx = flags.eps_dx, centered = flags.centered)
        end
        return -(j \ gx), 0.0
    end

    gmr = Channelflow.NSolver.GMRES(-gx, flags.n_solver, flags.eps_krylov)
    solver_res = 1.0
    ctr.mode = :newt
    for k in 1:flags.n_solver
        q = test_vector(gmr)
        aq = jacobian_action(dsi, x, q, gx; eps_dx = flags.eps_dx, centered = flags.centered)
        iterate!(gmr, aq)
        solver_res = residual(gmr)
        if solver_res < flags.eps_solver || (k == flags.n_solver && solver_res < flags.eps_solver_final)
            return solution(gmr), solver_res
        end
    end
    return solution(gmr), solver_res
end

function main()
    domain = read_geom(GEOM)
    ops = read_symm(SYMM)

    u_init = read_data(ASC, domain)
    u_proj = FlowField(u_init)
    project_symmetry!(u_proj, ops, FlowField(u_proj))
    make_state!(u_proj, Spectral, Spectral)

    N = domain.Nx * domain.Ny * domain.Nz * 3
    x0 = zeros(Float64, N)
    pack_velocity!(x0, u_proj)

    # Work buffers
    u_work = FlowField(u_proj)
    p_work = FlowField(domain.Nx, domain.Ny, domain.Nz, 1, domain.Lx, domain.Lz, domain.a, domain.b)
    sym_work = FlowField(u_work)
    x_proj = similar(x0)
    rbuf = similar(x0)
    tmp_u = FlowField(u_work)
    tmp_g = FlowField(u_work)
    tmp_dx = FlowField(u_work)

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
        variabledt = false,
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
    nsteps = Int(round(flags.T / flags.dt))

    ctr = EvalCounter(0, 0, 0, :newt)

    function map_residual(x::Vector{Float64})
        ctr.total += 1
        if ctr.mode == :newt
            ctr.newt += 1
        elseif ctr.mode == :opt
            ctr.opt += 1
        end

        unpack_velocity!(u_work, x)
        project_symmetry!(u_work, ops, sym_work)
        pack_velocity!(x_proj, u_work)
        make_state!(u_work, Spectral, Spectral)
        set_to_zero!(p_work)
        make_state!(p_work, Spectral, Spectral)

        fields = [u_work, p_work]
        nse = NSE(fields, flags)
        dns = MultistepDNS(fields, nse, flags)
        advance!(dns, fields, nsteps)

        pack_velocity!(rbuf, u_work)
        @inbounds @. rbuf = rbuf - x_proj

        @printf("eval %-4d |Rvec| %.6e mode=%s\n", ctr.total, norm(rbuf), String(ctr.mode))
        return copy(rbuf)
    end

    dsi = FunctionDSI(map_residual)
    nflags = NewtonSearchFlags(
        solver = :gmres,
        optimization = :hookstep,
        eps_search = 1e-13,
        eps_krylov = 1e-14,
        eps_dx = 1e-7,
        eps_solver = 1e-3,
        eps_solver_final = 5e-2,
        centered = false,
        n_newton = 8,
        n_solver = 4,
        delta = 0.01,
        delta_min = 1e-12,
        delta_max = 0.1,
        improv_req = 1e-3,
        lambda_min = 0.2,
    )

    open(OUT, "w") do io
        @printf(io, "%-14s %-14s %-14s %-14s %-14s %-14s %-14s %-14s %-9s %-9s %-9s\n",
                "%-L2Norm(Gx)", "rx", "delta", "L2Norm(x)", "L2Norm(u)", "L2Norm(dxN)", "L2Norm(dxOpt)", "SolverRes", "ftotal", "fnewt", "fopt")

        ctr.mode = :newt
        x = copy(x0)
        gx = eval_dsi(dsi, x)
        delta = nflags.delta

        function write_row(dxN_l2::Float64, dxOpt_l2::Float64, solver_res::Float64)
            lx = norm(x)
            lu = l2_from_vec!(tmp_u, x)
            lg = l2_from_vec!(tmp_g, gx)
            grel = lg / max(lu, 1e-30)
            rx = 0.5 * grel^2
            @printf(io, "%-14.6e %-14.6e %-14.6g %-14.6e %-14.6e %-14.6e %-14.6e %-14.6e %-9d %-9d %-9d\n",
                    grel, rx, delta, lx, lu, dxN_l2, dxOpt_l2, solver_res, ctr.total, ctr.newt, ctr.opt)
            flush(io)
        end

        write_row(0.0, 0.0, 1.0)

        for it in 1:nflags.n_newton
            gnorm = norm(gx)
            if gnorm < nflags.eps_search
                @printf("Converged at Newton step %d\n", it - 1)
                break
            end

            dx, solver_res = linear_step_logged!(dsi, x, gx, nflags, ctr)
            dxN_l2 = l2_from_vec!(tmp_dx, dx)
            dxc = copy(dx)

            if nflags.optimization == :hookstep && norm(dxc) > delta
                dxc .*= delta / norm(dxc)
            end

            λ = 1.0
            accepted = false
            dxOpt_l2 = 0.0

            while λ >= nflags.lambda_min
                xtrial = x .+ λ .* dxc
                ctr.mode = :opt
                gtrial = eval_dsi(dsi, xtrial)
                if norm(gtrial) <= (1 - nflags.improv_req * λ) * gnorm
                    x = xtrial
                    gx = gtrial
                    accepted = true
                    dxOpt_l2 = l2_from_vec!(tmp_dx, λ .* dxc)
                    delta = min(nflags.delta_max, max(delta, 2 * λ * norm(dxc)))
                    break
                end
                λ *= 0.5
            end

            if !accepted
                delta *= 0.5
                if delta < nflags.delta_min
                    write_row(dxN_l2, 0.0, solver_res)
                    println("Trust-region radius below delta_min; stopping")
                    break
                end
            end

            write_row(dxN_l2, dxOpt_l2, solver_res)
        end
    end

    println("Wrote $(OUT)")
    println("Counts: total=$(ctr.total) newt=$(ctr.newt) opt=$(ctr.opt)")
end

main()
