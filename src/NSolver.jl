module NSolver

using LinearAlgebra

export AbstractDSI, FunctionDSI, eval_dsi, jacobian_action,
       ParamDSI, update_parameter!, parameter,
       GMRES, FGMRES, BiCGStab,
       iterate!, test_vector, solution, residual, step1!, step2!, step3!,
       NewtonSearchFlags, NewtonAlgorithm, solve,
       ContinuationFlags, ContinuationPoint, continue_branch,
       gpu_backend_available

abstract type AbstractDSI end

struct FunctionDSI{F,J} <: AbstractDSI
    f::F
    jacobian::J
end

FunctionDSI(f::F) where {F} = FunctionDSI{F,Nothing}(f, nothing)

mutable struct ParamDSI{F,J,T} <: AbstractDSI
    f::F
    jacobian::J
    μ::T
end

ParamDSI(f::F, μ; jacobian = nothing) where {F} = ParamDSI{F,typeof(jacobian),typeof(μ)}(f, jacobian, μ)

function eval_dsi(dsi::FunctionDSI, x::AbstractVector)
    return dsi.f(x)
end

function eval_dsi(dsi::ParamDSI, x::AbstractVector)
    return dsi.f(x, dsi.μ)
end

parameter(dsi::ParamDSI) = dsi.μ
update_parameter!(dsi::ParamDSI, μ) = (dsi.μ = μ)

const _gpu_backend_available = Ref(false)
const _to_gpu_vector = Ref{Function}(x -> throw(ArgumentError("CUDA backend unavailable. Load CUDA.jl and the Channelflow CUDA extension.")))
const _sync_gpu = Ref{Function}(() -> nothing)

gpu_backend_available() = _gpu_backend_available[]

function register_gpu_backend!(; to_gpu::Function, synchronize::Function = () -> nothing)
    _to_gpu_vector[] = to_gpu
    _sync_gpu[] = synchronize
    _gpu_backend_available[] = true
    return nothing
end

function _zeros_like(template::AbstractVector{T}, n::Int) where {T}
    out = similar(template, T, n)
    fill!(out, zero(T))
    return out
end

function _zeros_like(template::AbstractVector{T}, m::Int, n::Int) where {T}
    out = similar(template, T, m, n)
    fill!(out, zero(T))
    return out
end

function _zeros_like(template::AbstractMatrix{T}, m::Int, n::Int) where {T}
    out = similar(template, T, m, n)
    fill!(out, zero(T))
    return out
end

function _vector_like(v::AbstractVector, template::AbstractVector{T}) where {T}
    if typeof(v) === typeof(template)
        return v
    end
    out = similar(template, T, length(v))
    out .= v
    return out
end

function _copy_to_backend(x::AbstractVector, backend::Symbol)
    if backend === :cpu
        return collect(x)
    elseif backend === :gpu
        gpu_backend_available() || throw(ArgumentError("NewtonSearchFlags.backend=:gpu requires CUDA.jl and ChannelflowCUDAExt."))
        return _to_gpu_vector[](x)
    end
    throw(ArgumentError("Unsupported backend=$(backend). Use :cpu or :gpu."))
end

function jacobian_action(dsi::FunctionDSI, x::AbstractVector, dx::AbstractVector, gx::AbstractVector;
                         eps_dx::Real = 1e-7, centered::Bool = false)
    if dsi.jacobian !== nothing
        return dsi.jacobian(x, dx)
    end
    step_norm = norm(dx)
    eps = step_norm < eps_dx ? one(eltype(x)) : eps_dx / step_norm
    if centered
        gp = _vector_like(eval_dsi(dsi, x .+ 0.5 * eps .* dx), gx)
        gm = _vector_like(eval_dsi(dsi, x .- 0.5 * eps .* dx), gx)
        return (gp .- gm) ./ eps
    end
    gp = _vector_like(eval_dsi(dsi, x .+ eps .* dx), gx)
    return (gp .- gx) ./ eps
end

function jacobian_action(dsi::ParamDSI, x::AbstractVector, dx::AbstractVector, gx::AbstractVector;
                         eps_dx::Real = 1e-7, centered::Bool = false)
    if dsi.jacobian !== nothing
        return dsi.jacobian(x, dx, dsi.μ)
    end
    step_norm = norm(dx)
    eps = step_norm < eps_dx ? one(eltype(x)) : eps_dx / step_norm
    if centered
        gp = _vector_like(eval_dsi(dsi, x .+ 0.5 * eps .* dx), gx)
        gm = _vector_like(eval_dsi(dsi, x .- 0.5 * eps .* dx), gx)
        return (gp .- gm) ./ eps
    end
    gp = _vector_like(eval_dsi(dsi, x .+ eps .* dx), gx)
    return (gp .- gx) ./ eps
end

mutable struct GMRES{T,QT<:AbstractMatrix{T},VT<:AbstractVector{T}}
    m::Int
    niter::Int
    n::Int
    condition::T
    h::Matrix{T}
    q::QT
    qn::VT
    xn::VT
    bnorm::T
    residual_value::T
end

function GMRES(b::AbstractVector{T}, niterations::Integer, min_condition::Real = 1e-13) where {T<:Real}
    m = length(b)
    bnorm = norm(b)
    qn = _zeros_like(b, m)
    if bnorm != 0
        qn .= b ./ bnorm
    end
    q = _zeros_like(b, m, 1)
    q[:, 1] .= qn
    xn = _zeros_like(b, m)
    return GMRES{T,typeof(q),typeof(qn)}(m, Int(niterations), 0, T(min_condition), zeros(T, niterations + 1, niterations), q, qn,
                                         xn, bnorm, bnorm == 0 ? zero(T) : one(T))
end

function test_vector(g::GMRES)
    return g.qn
end

function iterate!(g::GMRES{T}, aq::AbstractVector{T}) where {T}
    g.n == g.niter && return g
    v = copy(aq)
    for j in 1:(g.n + 1)
        qj = view(g.q, :, j)
        g.h[j, g.n + 1] = dot(qj, v)
        v .-= g.h[j, g.n + 1] .* qj
    end
    vnorm = norm(v)
    breakdown = abs(vnorm) < g.condition
    if breakdown
        g.h[g.n + 2, g.n + 1] = zero(T)
    else
        g.h[g.n + 2, g.n + 1] = vnorm
        v ./= vnorm
        if size(g.q, 2) <= g.n + 1
            newsize = min(size(g.q, 2) + 100, g.niter + 2)
            resize_q = _zeros_like(g.q, g.m, newsize)
            resize_q[:, 1:size(g.q, 2)] .= g.q
            g.q = resize_q
        end
        g.q[:, g.n + 2] .= v
        g.qn .= v
    end

    hn_rows = breakdown ? (g.n + 1) : (g.n + 2)
    hn = @view g.h[1:hn_rows, 1:(g.n + 1)]
    bk = zeros(T, hn_rows)
    bk[1] = g.bnorm
    y = hn \ bk
    g.residual_value = norm(hn * y - bk) / norm(bk)
    y_backend = _vector_like(y, g.qn)
    g.xn .= g.q[:, 1:(g.n + 1)] * y_backend
    g.n += 1
    return g
end

solution(g::GMRES) = g.xn
residual(g::GMRES) = g.residual_value

mutable struct FGMRES{T,QT<:AbstractMatrix{T},VT<:AbstractVector{T}}
    gmres::GMRES{T,QT,VT}
    z::QT
    az::QT
end

function FGMRES(b::AbstractVector{T}, niterations::Integer, min_condition::Real = 1e-13) where {T<:Real}
    gm = GMRES(b, niterations, min_condition)
    z = _zeros_like(b, length(b), 1)
    az = _zeros_like(b, length(b), 1)
    return FGMRES{T,typeof(z),typeof(gm.qn)}(gm, z, az)
end

test_vector(f::FGMRES) = test_vector(f.gmres)
solution(f::FGMRES) = solution(f.gmres)
residual(f::FGMRES) = residual(f.gmres)

function iterate!(f::FGMRES{T}, q::AbstractVector{T}, aq::AbstractVector{T}) where {T}
    n = f.gmres.n + 1
    if size(f.z, 2) < n
        newsize = min(size(f.z, 2) + 100, f.gmres.niter + 2)
        z = _zeros_like(f.z, size(f.z, 1), newsize)
        az = _zeros_like(f.az, size(f.az, 1), newsize)
        z[:, 1:size(f.z, 2)] .= f.z
        az[:, 1:size(f.az, 2)] .= f.az
        f.z = z
        f.az = az
    end
    f.z[:, n] .= q
    f.az[:, n] .= aq
    iterate!(f.gmres, aq)
    y = (@view f.gmres.h[1:(f.gmres.n + 1), 1:f.gmres.n]) \ vcat(f.gmres.bnorm, zeros(T, f.gmres.n))
    y_backend = _vector_like(y, f.gmres.qn)
    f.gmres.xn .= f.z[:, 1:f.gmres.n] * y_backend
    return f
end

mutable struct BiCGStab{T,VT<:AbstractVector{T}}
    r::VT
    r0::VT
    r0_sqnorm::T
    rhs_sqnorm::T
    rho::T
    alpha::T
    omega::T
    rho_old::T
    beta::T
    v::VT
    p::VT
    s::VT
    t::VT
    x::VT
    best_solution::VT
    residual_value::T
end

function BiCGStab(rhs::AbstractVector{T}) where {T<:Real}
    n = length(rhs)
    r = copy(rhs)
    rhs_sqnorm = dot(rhs, rhs)
    residual0 = rhs_sqnorm > 0 ? sqrt(dot(r, r) / rhs_sqnorm) : zero(T)
    return BiCGStab{T,typeof(r)}(r, copy(r), dot(r, r), rhs_sqnorm, one(T), one(T), one(T), zero(T), zero(T),
                                 _zeros_like(rhs, n), _zeros_like(rhs, n), _zeros_like(rhs, n), _zeros_like(rhs, n), _zeros_like(rhs, n), _zeros_like(rhs, n),
                                 residual0)
end

function step1!(b::BiCGStab)
    b.rho_old = b.rho
    b.rho = dot(b.r0, b.r)
    if abs(b.rho) < 1e-16 * b.r0_sqnorm
        b.r0 .= b.r
        b.rho = b.r0_sqnorm = dot(b.r, b.r)
    end
    b.beta = (b.rho / b.rho_old) * (b.alpha / b.omega)
    b.p .= b.r .+ b.beta .* (b.p .- b.omega .* b.v)
    return b.p
end

function step2!(b::BiCGStab, ap::AbstractVector)
    b.v .= ap
    b.alpha = b.rho / dot(b.r0, b.v)
    b.s .= b.r .- b.alpha .* b.v
    return b.s
end

function step3!(b::BiCGStab, as::AbstractVector)
    b.t .= as
    tmp = dot(b.t, b.t)
    b.omega = tmp > 0 ? dot(b.t, b.s) / tmp : zero(eltype(b.t))
    b.r .= b.s .- b.omega .* b.t
    b.x .+= b.alpha .* b.p .+ b.omega .* b.s
    current = sqrt(dot(b.r, b.r) / b.rhs_sqnorm)
    if current < b.residual_value
        b.residual_value = current
        b.best_solution .= b.x
    end
    return b.x
end

solution(b::BiCGStab) = b.best_solution
residual(b::BiCGStab) = b.residual_value

Base.@kwdef struct NewtonSearchFlags
    solver::Symbol = :gmres
    optimization::Symbol = :hookstep
    eps_search::Float64 = 1e-13
    eps_krylov::Float64 = 1e-14
    eps_dx::Float64 = 1e-7
    eps_solver::Float64 = 1e-6
    eps_solver_final::Float64 = 5e-2
    centered::Bool = false
    n_newton::Int = 20
    n_solver::Int = 200
    n_solver_max::Int = 500
    n_hook::Int = 20
    delta::Float64 = 1e-2
    delta_min::Float64 = 1e-12
    delta_max::Float64 = 1e-1
    delta_fuzz::Float64 = 1e-6
    improv_req::Float64 = 1e-3
    improve_ok::Float64 = 1e-1
    improve_good::Float64 = 7.5e-1
    lambda_min::Float64 = 0.2
    lambda_max::Float64 = 1.5
    g_ratio::Float64 = 10.0
    backend::Symbol = :cpu
end

struct NewtonAlgorithm
    flags::NewtonSearchFlags
end

Base.@kwdef struct ContinuationFlags
    ds::Float64 = 0.05
    n_steps::Int = 10
    ds_min::Float64 = 1e-4
    ds_max::Float64 = 0.2
    adapt::Bool = true
    max_corrections::Int = 8
    newton::NewtonSearchFlags = NewtonSearchFlags()
end

struct ContinuationPoint{T,V}
    μ::T
    x::V
    residual::Float64
end

function _linear_step(dsi::AbstractDSI, x::AbstractVector, gx::AbstractVector, flags::NewtonSearchFlags; nsolver::Int = flags.n_solver)
    n = length(x)
    if norm(gx) == 0
        return _zeros_like(x, n), 0.0
    end
    if flags.solver == :direct
        flags.backend === :gpu && throw(ArgumentError("solver=:direct is not supported with backend=:gpu; use solver=:gmres."))
        j = zeros(eltype(x), n, n)
        for i in 1:n
            e = zeros(eltype(x), n)
            e[i] = 1
            j[:, i] .= jacobian_action(dsi, x, e, gx; eps_dx = flags.eps_dx, centered = flags.centered)
        end
        return -(j \ gx), 0.0
    end
    nsolver = max(1, nsolver)
    gmr = GMRES(-gx, nsolver, flags.eps_krylov)
    solver_res = 1.0
    for k in 1:nsolver
        q = test_vector(gmr)
        aq = jacobian_action(dsi, x, q, gx; eps_dx = flags.eps_dx, centered = flags.centered)
        iterate!(gmr, aq)
        solver_res = residual(gmr)
        if solver_res < flags.eps_solver || (k == nsolver && solver_res < flags.eps_solver_final)
            return solution(gmr), solver_res
        end
    end
    return solution(gmr), solver_res
end

function _hookstep_accept(
    dsi::AbstractDSI,
    x::AbstractVector,
    gx::AbstractVector,
    dxN::AbstractVector,
    flags::NewtonSearchFlags,
    delta::Float64,
)
    gnorm = norm(gx)
    dnorm = norm(dxN)
    dnorm < 1e-30 && return false, x, gx, delta

    jdxN = jacobian_action(dsi, x, dxN, gx; eps_dx = flags.eps_dx, centered = flags.centered)

    for _ in 1:flags.n_hook
        s = min(1.0, delta / dnorm)
        dx = s .* dxN

        xtrial = x .+ dx
        gtrial = eval_dsi(dsi, xtrial)
        gtrial_norm = norm(gtrial)

        if gtrial_norm >= (1 - flags.delta_fuzz) * gnorm
            delta *= flags.lambda_min
            delta < flags.delta_min && break
            continue
        end

        pred = gnorm - norm(gx .+ s .* jdxN)
        actual = gnorm - gtrial_norm
        ratio = pred > 0 ? actual / pred : 0.0

        if ratio >= flags.improv_req
            hookstep_equals_newtonstep = s >= (1 - flags.delta_fuzz)
            if ratio < flags.improve_ok
                delta = max(flags.delta_min, flags.lambda_min * delta)
            elseif ratio > flags.improve_good && !hookstep_equals_newtonstep
                delta = min(flags.delta_max, flags.lambda_max * delta)
            end
            return true, xtrial, gtrial, delta
        end

        delta *= flags.lambda_min
        delta < flags.delta_min && break
    end

    return false, x, gx, delta
end

function solve(alg::NewtonAlgorithm, dsi::AbstractDSI, x0::AbstractVector)
    x = _copy_to_backend(x0, alg.flags.backend)
    gx = _vector_like(eval_dsi(dsi, x), x)
    delta = alg.flags.delta
    nsolver = min(max(1, alg.flags.n_solver), max(1, alg.flags.n_solver_max))
    gx_prev = norm(gx)
    for _ in 1:alg.flags.n_newton
        gnorm = norm(gx)
        if gnorm < alg.flags.eps_search
            return x, gnorm
        end
        dxN, _ = _linear_step(dsi, x, gx, alg.flags; nsolver = nsolver)

        if alg.flags.optimization == :hookstep
            accepted, xnew, gnew, delta_new = _hookstep_accept(dsi, x, gx, dxN, alg.flags, delta)
            if !accepted
                delta = delta_new
                if delta < alg.flags.delta_min
                    break
                end
                continue
            end
            x = xnew
            gx = gnew
            delta = delta_new
        else
            dx = copy(dxN)
            if norm(dx) > delta
                dx .*= delta / norm(dx)
            end

            λ = 1.0
            accepted = false
            while λ >= alg.flags.lambda_min
                xtrial = x .+ λ .* dx
                gtrial = eval_dsi(dsi, xtrial)
                if norm(gtrial) <= (1 - alg.flags.improv_req * λ) * gnorm
                    x = xtrial
                    gx = gtrial
                    accepted = true
                    delta = min(alg.flags.delta_max, max(delta, 2 * λ * norm(dx)))
                    break
                end
                λ *= 0.5
            end
            if !accepted
                delta *= 0.5
                if delta < alg.flags.delta_min
                    break
                end
                continue
            end
        end

        gcurr = norm(gx)
        rate = gx_prev / max(gcurr, 1e-30)
        if rate < alg.flags.g_ratio
            nsolver = min(max(1, alg.flags.n_solver_max), max(nsolver + 20, Int(ceil(1.25 * nsolver))))
        elseif rate > 2 * alg.flags.g_ratio
            nsolver = max(1, Int(floor(0.8 * nsolver)))
        end
        gx_prev = gcurr
    end
    return x, norm(gx)
end

function _augment_newton_flags(flags::NewtonSearchFlags, n_unknown::Int)
    return NewtonSearchFlags(; solver = :direct,
                             optimization = :linear,
                             eps_search = flags.eps_search,
                             eps_krylov = flags.eps_krylov,
                             eps_dx = flags.eps_dx,
                             eps_solver = flags.eps_solver,
                             eps_solver_final = flags.eps_solver_final,
                             centered = flags.centered,
                             n_newton = flags.n_newton,
                             n_solver = max(flags.n_solver, n_unknown),
                             n_solver_max = max(flags.n_solver_max, n_unknown),
                             n_hook = flags.n_hook,
                             delta = flags.delta,
                             delta_min = flags.delta_min,
                             delta_max = flags.delta_max,
                             delta_fuzz = flags.delta_fuzz,
                             improv_req = flags.improv_req,
                             improve_ok = flags.improve_ok,
                             improve_good = flags.improve_good,
                             lambda_min = flags.lambda_min,
                             lambda_max = flags.lambda_max,
                             g_ratio = flags.g_ratio,
                             backend = flags.backend)
end

function continue_branch(dsi::ParamDSI, x_init::AbstractVector, μ_start::Real;
                         flags::ContinuationFlags = ContinuationFlags(), direction::Real = 1.0)
    direction == 0 && throw(ArgumentError("direction cannot be zero"))

    alg = NewtonAlgorithm(flags.newton)
    points = ContinuationPoint[]

    update_parameter!(dsi, μ_start)
    x0, r0 = solve(alg, dsi, x_init)
    push!(points, ContinuationPoint(μ_start, copy(x0), r0))
    flags.n_steps == 1 && return points

    ds = abs(flags.ds) * sign(direction)
    μ1 = μ_start + ds
    update_parameter!(dsi, μ1)
    x1, r1 = solve(alg, dsi, x0)
    push!(points, ContinuationPoint(μ1, copy(x1), r1))
    flags.n_steps == 2 && return points

    ac_flags = _augment_newton_flags(flags.newton, length(x1) + 1)
    ac_alg = NewtonAlgorithm(ac_flags)

    for _ in 3:flags.n_steps
        prev = points[end - 1]
        curr = points[end]

        secx = curr.x .- prev.x
        secμ = curr.μ - prev.μ
        secnorm = sqrt(norm(secx)^2 + secμ^2)
        secx ./= secnorm
        secμ /= secnorm

        accepted = false
        x_next = copy(curr.x)
        μ_next = curr.μ
        r_next = curr.residual
        ds_trial = ds

        for _ in 1:flags.max_corrections
            x_pred = curr.x .+ ds_trial .* secx
            μ_pred = curr.μ + ds_trial * secμ

            augf = y -> begin
                x = y[1:end-1]
                μ = y[end]
                update_parameter!(dsi, μ)
                fx = eval_dsi(dsi, x)
                c = dot(x .- x_pred, secx) + (μ - μ_pred) * secμ
                return vcat(fx, c)
            end

            y0 = vcat(x_pred, μ_pred)
            y, ry = solve(ac_alg, FunctionDSI(augf), y0)

            if ry <= 10 * flags.newton.eps_search
                x_next = y[1:end-1]
                μ_next = y[end]
                update_parameter!(dsi, μ_next)
                r_next = norm(eval_dsi(dsi, x_next))
                accepted = true
                if flags.adapt
                    ds = if ds > 0
                        clamp(1.2 * ds_trial, flags.ds_min, flags.ds_max)
                    else
                        clamp(1.2 * ds_trial, -flags.ds_max, -flags.ds_min)
                    end
                end
                break
            end

            ds_trial *= 0.5
            if abs(ds_trial) < flags.ds_min
                break
            end
        end

        accepted || break
        push!(points, ContinuationPoint(μ_next, copy(x_next), r_next))
    end

    return points
end

end
