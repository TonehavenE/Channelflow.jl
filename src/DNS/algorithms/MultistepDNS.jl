export MultistepDNS

mutable struct MultistepDNS <: DNSAlgorithm
    common::DNSAlgorithmCommon  # Composition: HAS the common data

    # Fields specific to MultistepDNS
    eta::Float64
    alpha::Vector{Float64}
    beta::Vector{Float64}
    fields_history::Vector{Vector{FlowField}}
    nonlf_history::Vector{Vector{FlowField}}
    countdown::Int
end

function order(DNS::MultistepDNS)
    return DNS.common.order
end

function equations(DNS::MultistepDNS)
    return DNS.common.equations
end

function flags(DNS::MultistepDNS)
    return DNS.common.flags
end

function _sbdf_coeffs(order::Int)
    if order == 1
        return (1.0, [-1.0], [1.0])
    elseif order == 2
        return (1.5, [-2.0, 0.5], [2.0, -1.0])
    elseif order == 3
        return (11.0 / 6.0, [-3.0, 1.5, -1.0 / 3.0], [3.0, -3.0, 1.0])
    elseif order == 4
        return (25.0 / 12.0, [-4.0, 3.0, -4.0 / 3.0, 0.25], [4.0, -6.0, 4.0, -1.0])
    else
        error("Unsupported SBDF order: $order")
    end
end

function MultistepDNS(fields::Vector{FlowField{T}}, equations::Equation, flags::DNSFlags) where {T}
    algorithm = flags.timestepping
    if algorithm == SBDF1
        order = 1
    elseif algorithm == SBDF2
        order = 2
    elseif algorithm == SBDF3
        order = 3
    elseif algorithm == SBDF4
        order = 4
    else
        error("Unsupported timestepping algorithm: $algorithm")
    end
    eta, alpha, beta = _sbdf_coeffs(order)

    lambda_t = [eta / flags.dt]
    reset_lambda!(equations, lambda_t, flags)

    fields_history = Vector{Vector{FlowField}}(undef, order)
    nonlf_history = Vector{Vector{FlowField}}(undef, order)
    for j = 1:order
        fields_history[j] = [FlowField(f) for f in fields]
        nonlf_history[j] = [FlowField(f) for f in fields]
        for l = 1:length(fields)
            set_to_zero!(fields_history[j][l])
            set_to_zero!(nonlf_history[j][l])
        end
    end

    num_initsteps = order - 1
    countdown = num_initsteps
    common = DNSAlgorithmCommon(
        flags,
        order,
        length(fields),
        num_initsteps,
        flags.t0,
        lambda_t,
        equations,
        [],
    )
    return MultistepDNS(
        common,
        eta,
        alpha,
        beta,
        fields_history,
        nonlf_history,
        countdown,
    )
end

function MultistepDNS(other::MultistepDNS)
    return MultistepDNS(
        other.common,
        other.eta,
        copy(other.alpha),
        copy(other.beta),
        copy(other.fields_history),
        copy(other.nonlf_history),
        other.countdown,
    )
end

function advance!(alg::MultistepDNS, fields::Vector{FlowField{T}}, num_steps::Int) where {T<:Number}
    J = order(alg)
    rhs = create_RHS(equations(alg), fields)
    len = length(rhs)
    alg.fields_history[1] = fields
    prev_eff_order = 0
    profile_advance = get(ENV, "CHANNELFLOW_PROFILE_ADVANCE", "0") == "1"
    t_nonlin = 0.0
    t_rhs = 0.0
    t_solve = 0.0
    b_nonlin = 0
    b_solve = 0

    # time stepping loop
    for step = 1:num_steps
        eff_order = order(alg) - alg.countdown
        eta, alpha, beta = _sbdf_coeffs(eff_order)
        if step == 1 || eff_order != prev_eff_order
            alg.common.lambda_t[1] = eta / flags(alg).dt
            reset_lambda!(equations(alg), [Float64(alg.common.lambda_t[1])], flags(alg))
            prev_eff_order = eff_order
        end

        if order(alg) > 0
            # evaluate nonlinear terms
            if profile_advance
                stats = @timed nonlinear!(equations(alg), alg.fields_history[1], alg.nonlf_history[1], flags(alg))
                t_nonlin += stats.time
                b_nonlin += stats.bytes
            else
                nonlinear!(equations(alg), alg.fields_history[1], alg.nonlf_history[1], flags(alg))
            end
        end

        t_rhs0 = profile_advance ? time() : 0.0
        for l = 1:len
            set_to_zero!(rhs[l])
            # sum over multistep loop
            for j = 1:eff_order
                a = -alpha[j] / flags(alg).dt
                b = -beta[j]
                add!(rhs[l], a, alg.fields_history[j][l], b, alg.nonlf_history[j][l])
            end
        end
        if profile_advance
            t_rhs += time() - t_rhs0
        end

        # solve the implicit problem 
        # Multistep has one implicit stage; use stage index 1.
        if profile_advance
            stats = @timed solve!(equations(alg), alg.fields_history[J], rhs, 1, flags(alg))
            t_solve += stats.time
            b_solve += stats.bytes
        else
            solve!(equations(alg), alg.fields_history[J], rhs, 1, flags(alg))
        end
        # now we need to shift all of the fields over...

        for j = J:-1:2
            for l = 1:alg.common.num_fields
                swap!(alg.fields_history[j][l], alg.fields_history[j-1][l])
                swap!(alg.nonlf_history[j][l], alg.nonlf_history[j-1][l])
            end
        end
        if alg.countdown > 0
            alg.countdown -= 1
        end
        alg.common.t += flags(alg).dt
    end
    if profile_advance
        println("advance profile: steps=$(num_steps) nonlin=$(round(t_nonlin, digits=3))s rhs=$(round(t_rhs, digits=3))s solve=$(round(t_solve, digits=3))s alloc_nonlin=$(round(b_nonlin / 1024^3, digits=3))GB alloc_solve=$(round(b_solve / 1024^3, digits=3))GB")
    end
    return
end
