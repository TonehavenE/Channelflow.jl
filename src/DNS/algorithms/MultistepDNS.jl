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

function MultistepDNS(fields::Vector{FlowField{T}}, equations::Equation, flags::DNSFlags) where {T}
    algorithm = flags.timestepping
    if algorithm == SBDF1
        order = 1
        eta = 1.0
        alpha = [-1.0]
        beta = [1.0]
    elseif algorithm == SBDF2
        order = 2
        eta = 1.5
        alpha = [-2.0, 0.5]
        beta = [2.0, -1.0]
    elseif algorithm == SBDF3
        order = 3
        eta = 11.0 / 6.0
        alpha = [-3.0, 1.5, -1.0 / 3.0]
        beta = [3.0, -3.0, 1.0]
    elseif algorithm == SBDF4
        order = 4
        eta = 25.0 / 12.0
        alpha = [-4.0, 3.0, -4.0 / 3.0, 0.25]
        beta = [4.0, -6.0, 4.0, -1.0]
    else
        error("Unsupported timestepping algorithm: $algorithm")
    end

    lambda_t = [eta / flags.dt]
    reset_lambda!(equations, lambda_t, flags)

    temp_array = []
    for i = 1:length(fields)
        push!(temp_array, FlowField(fields[i]))
        set_to_zero!(temp_array[i])
    end

    fields_history = []
    nonlf_history = []
    for j = 1:order
        push!(fields_history, temp_array)
        push!(nonlf_history, temp_array)
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
    J = order(alg) - 1
    rhs = create_RHS(equations(alg), fields)
    len = length(rhs)
    alg.fields_history[1] = fields

    # time stepping loop
    for step = 1:num_steps
        if order(alg) > 0
            # evaluate nonlinear terms
            nonlinear!(equations(alg), alg.fields_history[1], alg.nonlf_history[1], flags(alg))
        end

        for l = 1:len
            set_to_zero!(rhs[l])
            # sum over multistep loop
            for j = 1:order(alg)
                a = -alg.alpha[j] / flags(alg).dt
                b = -alg.beta[j]
                rhs[l] += a * alg.fields_history[j][l] + b * alg.nonlf_history[j][l]
            end
        end

        # solve the implicit problem 
        solve!(equations(alg), alg.fields_history[J], rhs, num_steps, flags(alg))
        # now we need to shift all of the fields over...

        for j = J:-1:1
            for l = 1:alg.num_fields
                swap!(alg.fields_history[j][l], alg.fields_history[j-1][l])
                swap!(alg.nonlf_history[j][l], alg.nonlf_history[j-1][l])
            end
        end
        alg.t += flags(alg).dt
    end
    fields = alg.fields[1]

    return
end
