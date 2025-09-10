export RungeKuttaDNS

"""
    RungeKuttaDNS

Implements a Runge-Kutta time-stepping algorithm for DNS simulations,
based on the three-stage scheme (RK3/CN) from Peyret (2002).
"""
mutable struct RungeKuttaDNS <: DNSAlgorithm
    common::DNSAlgorithmCommon  # Composition with common DNS data

    # Fields specific to RungeKuttaDNS
    Nsubsteps::Int
    A::Vector{Float64}
    B::Vector{Float64}
    C::Vector{Float64}
    Qj::Vector{FlowField}
    Qj1::Vector{FlowField}
end

function RungeKuttaDNS(fields::Vector{FlowField{T}}, equations::Equation, flags::DNSFlags) where {T}
    algorithm = flags.timestepping

    # Coefficients from Peyret (2002), section 4.5.2.c.2
    if algorithm == CNRK2
        order = 2
        Nsubsteps = 3
        A::Vector{Float64} = [0.0, -5.0 / 9.0, -153.0 / 128.0]
        B::Vector{Float64} = [1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0]
        C::Vector{Float64} = [1.0 / 6.0, 5.0 / 24.0, 1.0 / 8.0]
    else
        error("Unsupported Runge-Kutta algorithm: $algorithm")
    end

    # Initialize fields for nonlinear terms
    Qj = [FlowField(f) for f in fields]
    Qj1 = [FlowField(f) for f in fields]
    for i in eachindex(fields)
        set_to_zero!(Qj[i])
        set_to_zero!(Qj1[i])
    end

    # Define time-stepping constants for the implicit solver
    lambda_t = [1.0 / (C[j] * flags.dt) for j in 1:Nsubsteps]
    reset_lambda!(equations, lambda_t, flags)

    common = DNSAlgorithmCommon(
        flags,
        order,
        length(fields),
        0, # Ninitsteps is 0 for RK methods
        flags.t0,
        lambda_t,
        equations,
        [], # Symmetries not implemented
    )

    return RungeKuttaDNS(
        common,
        Nsubsteps,
        A,
        B,
        C,
        Qj,
        Qj1,
    )
end

function advance!(alg::RungeKuttaDNS, fields::Vector{FlowField{T}}, Nsteps::Int) where {T<:Number}
    rhs = create_RHS(alg.common.equations, fields)
    linear_terms = [FlowField(f) for f in rhs] # Linear terms
    advance!(alg, fields, Nsteps, rhs, linear_terms)
end

function advance!(alg::RungeKuttaDNS, fields::Vector{FlowField{T}}, Nsteps::Int, rhs, lt) where {T<:Number}
    len = length(rhs)

    # Main time-stepping loop
    for n in 1:Nsteps
        # Loop over Runge-Kutta substeps
        for j in 1:alg.Nsubsteps
            uj = fields # Alias for clarity


            # Update nonlinear history: Q_{j+1} = A_j * Q_j + N(u_j)
            scale!(alg.Qj[1], alg.A[j])

            # The function `nonlinear!` calculates N(u_j) and stores it in Qj1
            nonlinear!(alg.common.equations, uj, alg.Qj1, alg.common.flags)

            subtract!(alg.Qj[1], alg.Qj1[1])

            # Get linear term
            linear!(alg.common.equations, uj, lt, alg.common.flags)

            # Combine terms to form the right-hand side for the implicit solve
            B_C = alg.B[j] / alg.C[j]
            for l in 1:len
                # Start with the linear term
                copy!(rhs[l], lt[l])

                add!(rhs[l], alg.common.lambda_t[j], uj[l], B_C, alg.Qj[l])
            end

            # Solve the implicit system: L(u_j) - lambda_t * u_j = RHS

            solve!(alg.common.equations, fields, rhs, j, alg.common.flags)
        end

        alg.common.t += alg.common.flags.dt
    end

    return nothing
end

function reset_dt!(alg::RungeKuttaDNS, dt::Real)
    alg.common.flags = DNSFlags(alg.common.flags; dt=dt) # Update dt in flags

    # Recompute lambda_t for the implicit solver
    for j in 1:alg.Nsubsteps
        alg.common.lambda_t[j] = 1.0 / (alg.C[j] * alg.common.flags.dt)
    end

    # Reconfigure the TauSolvers with the new lambda values
    reset_lambda!(alg.common.equations, alg.common.lambda_t, alg.common.flags)
end
