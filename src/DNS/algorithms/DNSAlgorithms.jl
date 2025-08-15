module DNSAlgorithms

using ..DNSSettings
using ..FlowFields

"""
    DNSAlgorithm

The virtual type for DNS algorithms.
Subtypes represent algorithms for time stepping the Navier-Stokes equations.
"""
abstract type DNSAlgorithm end

# =================
# Generic fallbacks
# =================

"""
    advance!(algorithm, fields, num_steps)

Advances the simulation by a given number of time steps.
"""
function advance!(algorithm::DNSAlgorithm, fields::Vector{FlowField}, num_steps::Int = 1)
    error("The function `advance!` is not implemented for $(typeof(algorithm))!")
end

"""
    project(algorithm)

Project onto a symmetric subspace (which is a member of flags).
"""
function project(algorithm::DNSAlgorithm)
    error("The function `project` is not implemented for $(typeof(algorithm))!")
end

"""
    reset_dt(algorithm, dt)

Reset the time step size.
"""
function reset_dt(algorithm::DNSAlgorithm, dt::Real)
    error("The function `reset_dt` is not implemented for $(typeof(algorithm))!")
end

function Base.push!(algorithm::DNSAlgorithm, fields::Vector{FlowField})
    error("The function `push!` is not implemented for $(typeof(algorithm))!")
end

function is_full(algorithm::DNSAlgorithm)
    error("The function `is_full` is not implemented for $(typeof(algorithm))!")
end


function reset_time!(alg::DNSAlgorithm, t::Float64)
    alg.common.t = t
end

"""
    DNSAlgorithmCommon

This struct contains certain common fields for DNS algorithms.
It is intended to be used as a "base type" for specific DNS algorithm implementations.
Each of those implementations should contain a field of type `DNSAlgorithmCommon` to hold these common settings.
Typically, that field should be named `common`.
"""
mutable struct DNSAlgorithmCommon
    flags::DNSFlags
    order::Int
    num_fields::Int
    num_initsteps::Int
    t::Real
    lambda_t::Vector{Real} # time stepping factors for implicit solver
    equations::Vector{Any} # TODO implement equations
    symmetries::Vector{Vector{Any}} # TODO implement symmetries
end

function DNSAlgorithmCommon(fields::Vector{FlowField}, equations, flags::DNSFlags)
    return DNSAlgorithmCommon(
        flags,
        0,
        length(fields),
        0,
        flags.t0,
        [], # placeholder
        equations,
        [], # placeholder
    )
end

end
