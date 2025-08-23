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
        A = [0.0, -5.0 / 9.0, -153.0 / 128.0]
        B = [1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0]
        C = [1.0 / 6.0, 5.0 / 24.0, 1.0 / 8.0]
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
    lt = [FlowField(f) for f in rhs] # Linear terms
    len = length(rhs)

    # Main time-stepping loop
    for n in 1:Nsteps
        # Loop over Runge-Kutta substeps
        for j in 1:alg.Nsubsteps
            uj = fields # Alias for clarity

            # Update nonlinear history: Q_{j+1} = A_j * Q_j + N(u_j)
            # Note: The C++ code uses Qj_[0] *= A_[j]. This implies Qj_ is a single FlowField,
            # but the logic seems to apply to a vector of fields. Here we assume a vector.
            for i in eachindex(alg.Qj)
                scale!(alg.Qj[i], alg.A[j])
            end

            # The function `nonlinear!` calculates N(u_j) and stores it in Qj1
            nonlinear!(alg.common.equations, uj, alg.Qj1, alg.common.flags)

            # Note: C++ does Qj -= Qj1 because their nonlinear term is defined as u*grad(u), 
            # while the formula uses N(u) = -u*grad(u). Your `nonlinear!` should match this.
            for i in eachindex(alg.Qj)
                subtract!(alg.Qj[i], alg.Qj1[i])
            end

            # Get linear term
            linear!(alg.common.equations, uj, lt, alg.common.flags)

            # Combine terms to form the right-hand side for the implicit solve
            B_C = alg.B[j] / alg.C[j]
            for l in 1:len
                # Start with the linear term
                copy!(rhs[l], lt[l])

                # Add the multistep terms: add!(ff, a, ff1, b, ff2) => ff = a*ff1 + b*ff2
                # Here we do: rhs = 1.0*lt + lambda_t*uj + B_C*Qj
                # We use an intermediate add! to achieve this
                add!(rhs[l], alg.common.lambda_t[j], uj[l]) # rhs += lambda_t*uj
                add!(rhs[l], B_C, alg.Qj[l]) # rhs += B_C*Qj
            end

            # Solve the implicit system: L(u_j) - lambda_t * u_j = RHS
            solve!(alg.common.equations, fields, rhs, j, alg.common.flags)
        end # End of substep loop

        alg.common.t += alg.common.flags.dt

        # Optional: Add verbosity/logging here as in the C++ code

    end # End of main step loop

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