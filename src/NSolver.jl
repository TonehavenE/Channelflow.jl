module NSolver

using LinearAlgebra
using Random

export AbstractDSI, FunctionDSI, eval_dsi, jacobian_action,
       ParamDSI, update_parameter!, parameter,
       GMRES, FGMRES, BiCGStab,
       iterate!, test_vector, solution, residual, step1!, step2!, step3!,
       NewtonSearchFlags, NewtonAlgorithm, solve,
       ContinuationFlags, ContinuationPoint, continue_branch

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

function jacobian_action(dsi::FunctionDSI, x::AbstractVector, dx::AbstractVector, gx::AbstractVector;
                         eps_dx::Real = 1e-7, centered::Bool = false)
    if dsi.jacobian !== nothing
        return dsi.jacobian(x, dx)
    end
    step_norm = norm(dx)
    eps = step_norm < eps_dx ? one(eltype(x)) : eps_dx / step_norm
    if centered
        return (eval_dsi(dsi, x .+ 0.5 * eps .* dx) .- eval_dsi(dsi, x .- 0.5 * eps .* dx)) ./ eps
    end
    return (eval_dsi(dsi, x .+ eps .* dx) .- gx) ./ eps
end

function jacobian_action(dsi::ParamDSI, x::AbstractVector, dx::AbstractVector, gx::AbstractVector;
                         eps_dx::Real = 1e-7, centered::Bool = false)
    if dsi.jacobian !== nothing
        return dsi.jacobian(x, dx, dsi.μ)
    end
    step_norm = norm(dx)
    eps = step_norm < eps_dx ? one(eltype(x)) : eps_dx / step_norm
    if centered
        return (eval_dsi(dsi, x .+ 0.5 * eps .* dx) .- eval_dsi(dsi, x .- 0.5 * eps .* dx)) ./ eps
    end
    return (eval_dsi(dsi, x .+ eps .* dx) .- gx) ./ eps
end

mutable struct GMRES{T}
    m::Int
    niter::Int
    n::Int
    condition::T
    h::Matrix{T}
    q::Matrix{T}
    qn::Vector{T}
    xn::Vector{T}
    bnorm::T
    residual_value::T
end

function GMRES(b::AbstractVector{T}, niterations::Integer, min_condition::Real = 1e-13) where {T<:Real}
    m = length(b)
    qn = collect(b ./ norm(b))
    q = zeros(T, m, 1)
    q[:, 1] .= qn
    return GMRES{T}(m, Int(niterations), 0, T(min_condition), zeros(T, niterations + 1, niterations), q, qn,
                    zeros(T, m), norm(b), one(T))
end

function test_vector(g::GMRES)
    return g.qn
end

function iterate!(g::GMRES{T}, aq::AbstractVector{T}) where {T}
    g.n == g.niter && return g
    v = collect(aq)
    for j in 1:(g.n + 1)
        qj = view(g.q, :, j)
        g.h[j, g.n + 1] = dot(qj, v)
        v .-= g.h[j, g.n + 1] .* qj
    end
    vnorm = norm(v)
    retries = 0
    while abs(vnorm) < g.condition && retries < 10
        retries += 1
        randn!(v)
        for j in 1:(g.n + 1)
            qj = view(g.q, :, j)
            g.h[j, g.n + 1] = dot(qj, v)
            v .-= g.h[j, g.n + 1] .* qj
        end
        vnorm = norm(v)
    end
    g.h[g.n + 2, g.n + 1] = vnorm
    v ./= vnorm
    if size(g.q, 2) <= g.n + 1
        newsize = min(size(g.q, 2) + 100, g.niter + 2)
        resize_q = zeros(T, g.m, newsize)
        resize_q[:, 1:size(g.q, 2)] .= g.q
        g.q = resize_q
    end
    g.q[:, g.n + 2] .= v
    g.qn .= v

    hn = @view g.h[1:(g.n + 2), 1:(g.n + 1)]
    bk = zeros(T, g.n + 2)
    bk[1] = g.bnorm
    y = hn \ bk
    g.residual_value = norm(hn * y - bk) / norm(bk)
    g.xn .= g.q[:, 1:(g.n + 1)] * y
    g.n += 1
    return g
end

solution(g::GMRES) = g.xn
residual(g::GMRES) = g.residual_value

mutable struct FGMRES{T}
    gmres::GMRES{T}
    z::Matrix{T}
    az::Matrix{T}
end

function FGMRES(b::AbstractVector{T}, niterations::Integer, min_condition::Real = 1e-13) where {T<:Real}
    gm = GMRES(b, niterations, min_condition)
    z = zeros(T, length(b), 1)
    az = zeros(T, length(b), 1)
    return FGMRES{T}(gm, z, az)
end

test_vector(f::FGMRES) = test_vector(f.gmres)
solution(f::FGMRES) = solution(f.gmres)
residual(f::FGMRES) = residual(f.gmres)

function iterate!(f::FGMRES{T}, q::AbstractVector{T}, aq::AbstractVector{T}) where {T}
    n = f.gmres.n + 1
    if size(f.z, 2) < n
        newsize = min(size(f.z, 2) + 100, f.gmres.niter + 2)
        z = zeros(T, size(f.z, 1), newsize)
        az = zeros(T, size(f.az, 1), newsize)
        z[:, 1:size(f.z, 2)] .= f.z
        az[:, 1:size(f.az, 2)] .= f.az
        f.z = z
        f.az = az
    end
    f.z[:, n] .= q
    f.az[:, n] .= aq
    iterate!(f.gmres, aq)
    y = (@view f.gmres.h[1:(f.gmres.n + 1), 1:f.gmres.n]) \ vcat(f.gmres.bnorm, zeros(T, f.gmres.n))
    f.gmres.xn .= f.z[:, 1:f.gmres.n] * y
    return f
end

mutable struct BiCGStab{T}
    r::Vector{T}
    r0::Vector{T}
    r0_sqnorm::T
    rhs_sqnorm::T
    rho::T
    alpha::T
    omega::T
    rho_old::T
    beta::T
    v::Vector{T}
    p::Vector{T}
    s::Vector{T}
    t::Vector{T}
    x::Vector{T}
    best_solution::Vector{T}
    residual_value::T
end

function BiCGStab(rhs::AbstractVector{T}) where {T<:Real}
    n = length(rhs)
    r = collect(rhs)
    rhs_sqnorm = dot(rhs, rhs)
    return BiCGStab{T}(r, copy(r), dot(r, r), rhs_sqnorm, one(T), one(T), one(T), zero(T), zero(T),
                       zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n),
                       sqrt(dot(r, r) / rhs_sqnorm))
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
    delta::Float64 = 1e-2
    delta_min::Float64 = 1e-12
    delta_max::Float64 = 1e-1
    improv_req::Float64 = 1e-3
    lambda_min::Float64 = 0.2
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

function _linear_step(dsi::AbstractDSI, x::AbstractVector, gx::AbstractVector, flags::NewtonSearchFlags)
    n = length(x)
    if flags.solver == :direct
        j = zeros(eltype(x), n, n)
        for i in 1:n
            e = zeros(eltype(x), n)
            e[i] = 1
            j[:, i] .= jacobian_action(dsi, x, e, gx; eps_dx = flags.eps_dx, centered = flags.centered)
        end
        return -(j \ gx), 0.0
    end
    gmr = GMRES(-gx, flags.n_solver, flags.eps_krylov)
    solver_res = 1.0
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

function solve(alg::NewtonAlgorithm, dsi::AbstractDSI, x0::AbstractVector)
    x = collect(x0)
    gx = eval_dsi(dsi, x)
    delta = alg.flags.delta
    for _ in 1:alg.flags.n_newton
        gnorm = norm(gx)
        if gnorm < alg.flags.eps_search
            return x, gnorm
        end
        dx, _ = _linear_step(dsi, x, gx, alg.flags)
        if alg.flags.optimization == :hookstep && norm(dx) > delta
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
        end
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
                             delta = flags.delta,
                             delta_min = flags.delta_min,
                             delta_max = flags.delta_max,
                             improv_req = flags.improv_req,
                             lambda_min = flags.lambda_min)
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
